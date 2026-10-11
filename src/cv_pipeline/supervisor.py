"""Parent-only owner of process lifecycle, signals and final resource cleanup.

Start consumers before producers and monitor process exit before receiving
status. Graceful stopping drains FIFO EOS; failure recovery relies only on
parent-owned process handles and never on the dead worker returning credits.
Signal handlers record intent without IPC or I/O. Diagnostic shutdown, memory
disposal and signal restoration have independent cleanup scopes.
"""

import json
import logging
import os
import signal
import time
from collections.abc import Callable
from contextlib import suppress
from multiprocessing.process import BaseProcess
from queue import Empty
from typing import Any

from .capture import run_capture
from .config import PipelineConfig, validate_config
from .ipc import DatagramChannel
from .logging_setup import BoundedQueueHandler, ListenerState, LogListenerOwner
from .messages import (
    CameraMetrics,
    FrameAccounting,
    HealthEvent,
    Stage,
    StageSummary,
    empty_camera_metrics,
)
from .output import ResultOutput
from .processing import run_processing
from .reporter import run_reporter
from .resources import MemoryMonitor, MemoryObservation
from .transport import TransportOwner, WorkerResources


class HealthMonitor:
    """Interpret worker state without reading data queues or acquiring worker-owned locks."""

    def __init__(self) -> None:
        """Track last event per stage and stages with received final summaries."""
        self.events: dict[Stage, HealthEvent] = {}
        self.finished: set[Stage] = set()

    def observe(self, event: HealthEvent) -> None:
        """Store a fresh health event, raising immediately when it contains a worker error."""
        if event.error is not None:
            raise RuntimeError(f"{event.stage.value} failed: {event.error}")
        self.events[event.stage] = event

    def check(self, now_ns: int, timeout_s: float) -> list[Stage]:
        """Return stages past heartbeat or active-work budget, excluding finished stages.

        Legitimate idle skips progress expiry but never heartbeat expiry. Work-start
        refresh is supplied by StageRuntime, not guessed from heartbeat timestamps.
        """
        return [
            stage
            for stage, event in self.events.items()
            if stage not in self.finished
            and (
                now_ns - event.monotonic_ns > timeout_s * 1e9
                or (not event.idle and now_ns - event.last_progress_monotonic_ns > timeout_s * 1e9)
            )
        ]


class PipelineSupervisor:
    """One-shot main-thread lifecycle owner; all children and IPC end with run()."""

    def __init__(
        self,
        config: PipelineConfig,
        duration_s: float | None = None,
        summary_path: str | None = None,
    ) -> None:
        """Validate config and initialize parent-local state without starting workers.

        Signal installation requires run() on the main thread. duration_s counts from
        full readiness; the CLI validates its positivity before constructing this owner.
        """
        validate_config(config)
        self.config = config
        if duration_s is not None and (
            type(duration_s) not in (int, float) or not 0 < duration_s <= 86400
        ):
            raise ValueError("duration must be finite and between 0 and 86400 seconds")
        self.duration_s = duration_s
        self.summary_path = summary_path
        self._ran = False
        self._owner: TransportOwner | None = None
        self._resources: WorkerResources | None = None
        self._children: dict[Stage, BaseProcess] = {}
        self._health = HealthMonitor()
        self._summaries: dict[Stage, dict[str, CameraMetrics]] = {}
        self._diagnostics: dict[str, dict[str, int]] = {}
        self._memory_monitor = MemoryMonitor(config.memory_budget_mib * 2**20)
        self._memory_observation: MemoryObservation | None = None
        self._last_memory_sample = 0.0
        self._shutdown_at: float | None = None
        self._shutdown_requested = False
        self._logger = logging.getLogger("cv_pipeline.parent")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())

    def request_shutdown(self) -> None:
        """Record ordinary stop intent and its deadline without logging or IPC calls.

        Safe for Python signal callbacks: the lifecycle loop subsequently publishes the
        stop byte. Duplicate requests do not restart the drain deadline.
        """
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._shutdown_at = time.monotonic()
            # Signal-safe at the Python level: record parent-local intent only.
            # The regular lifecycle loop publishes stop and diagnostics later.

    def _receive_health(self, timeout: float = 0.1) -> None:
        """Receive at most one atomic status message within timeout seconds.

        Unlike a multiprocessing Queue, this channel cannot expose a partially written
        message or a user-space writer lock left by a killed worker.
        """
        assert self._resources is not None
        try:
            message = self._resources.health.get(timeout=timeout)
        except Empty:
            return
        if isinstance(message, HealthEvent):
            self._health.observe(message)
        elif isinstance(message, StageSummary):
            self._summaries[message.stage] = message.cameras
            self._diagnostics[message.stage.value] = message.diagnostics
            self._health.finished.add(message.stage)
        else:
            raise ValueError("unexpected health channel message")

    def _check_exits(self) -> None:
        """Detect child death before doing further control reads; reject premature exit.

        The cooperative abort indicator is a lock-free byte. This method never queries
        a data Queue that a dead producer might have corrupted.
        """
        assert self._resources is not None
        for stage, child in self._children.items():
            if child.exitcode is not None:
                if child.exitcode != 0:
                    raise RuntimeError(f"{stage.value} exited with code {child.exitcode}")
                if not self._shutdown_requested:
                    raise RuntimeError(f"{stage.value} exited before shutdown was requested")
        if self._resources.abort.is_set():
            raise RuntimeError("worker requested emergency abort")

    def _start(self) -> None:
        # Wait for each downstream stage before enabling upstream producers.
        """Start Reporter, Processing and Capture in dependency order and await readiness.

        Each stage has a startup deadline. A signal during startup triggers bounded
        cleanup rather than pretending a partially started pipeline drained normally.
        """
        assert self._resources is not None
        assert self._owner is not None
        for stage, target in (
            (Stage.REPORTER, run_reporter),
            (Stage.PROCESSING, run_processing),
            (Stage.CAPTURE, run_capture),
        ):
            child = self._owner.context.Process(
                target=target,
                args=(self.config, self._owner.resources_for(stage)),
                name=stage.value,
            )
            child.start()
            self._children[stage] = child
            deadline = time.monotonic() + self.config.startup_timeout_s
            while stage not in self._health.events or not self._health.events[stage].ready:
                self._check_exits()
                self._receive_health()
                if self._shutdown_requested:
                    raise RuntimeError("shutdown requested during startup")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{stage.value} readiness timeout")
            self._logger.info("stage_ready stage=%s pid=%d", stage.value, child.pid)

    def _verify_accounting(self) -> dict[str, FrameAccounting]:
        """After every child exits normally, reconcile results/drops and all slot credits.

        Never call this on a crash path: credit queues may be corrupt. Clean drain must
        have captured = reported + explicit drops and processed = reported per camera.
        """
        assert self._resources is not None
        if set(self._summaries) != set(Stage):
            raise RuntimeError("missing final stage summary")
        accounting: dict[str, FrameAccounting] = {}
        for camera in self.config.cameras:
            capture = self._summaries[Stage.CAPTURE].get(
                camera.camera_id, empty_camera_metrics(camera.camera_id)
            )
            processing = self._summaries[Stage.PROCESSING].get(
                camera.camera_id, empty_camera_metrics(camera.camera_id)
            )
            reporter = self._summaries[Stage.REPORTER].get(
                camera.camera_id, empty_camera_metrics(camera.camera_id)
            )
            captured = capture.get("captured", 0)
            processed = processing.get("processed", 0)
            reported = reporter.get("reported", 0)
            drops = (
                capture.get("dropped_no_slot", 0)
                + capture.get("dropped_shutdown", 0)
                + processing.get("dropped_stale", 0)
                + capture.get("dropped_publication_timeout", 0)
                + processing.get("dropped_superseded", 0)
            )
            if processed != reported or captured != reported + drops:
                raise RuntimeError(f"frame accounting mismatch for {camera.camera_id}")
            accounting[camera.camera_id] = {
                "captured": captured,
                "processed": processed,
                "reported": reported,
                "dropped_no_slot": capture.get("dropped_no_slot", 0),
                "dropped_stale": processing.get("dropped_stale", 0),
                "dropped_shutdown": capture.get("dropped_shutdown", 0),
                "dropped_publication_timeout": capture.get("dropped_publication_timeout", 0),
                "dropped_superseded": processing.get("dropped_superseded", 0),
                "in_flight": 0,
            }
        # No child is alive now; all credits must have returned exactly once.
        for i, queue in enumerate(self._resources.free_slots):
            slots = [queue.get(timeout=1) for _ in range(self._resources.memory.slots_per_camera)]
            expected = set(
                range(
                    i * self._resources.memory.slots_per_camera,
                    (i + 1) * self._resources.memory.slots_per_camera,
                )
            )
            if set(slots) != expected:
                raise RuntimeError("shared-memory slot credit mismatch")
        return accounting

    @staticmethod
    def _kill_worker(child: BaseProcess) -> None:
        """Ignore the normal exit race between the liveness check and SIGKILL."""
        try:
            if child.is_alive():
                child.kill()
        except ProcessLookupError:
            pass

    def _reap_workers(self, code: int) -> None:
        """Attempt every worker independently; report failures after all reap attempts."""
        errors: list[Exception] = []
        if code != 0:
            for child in self._children.values():
                try:
                    self._kill_worker(child)
                except Exception as exc:
                    errors.append(exc)
        deadline = time.monotonic() + 3.0
        for child in self._children.values():
            reaped = False
            try:
                child.join(timeout=max(0, deadline - time.monotonic()))
                if child.is_alive():
                    self._kill_worker(child)
                    child.join(timeout=1)
                if child.is_alive():
                    raise RuntimeError("worker could not be reaped after SIGKILL")
                reaped = True
                child.close()
            except Exception as exc:
                errors.append(exc)
                if reaped:
                    # close() can fail after releasing handles; do not query or
                    # terminate an already reaped, possibly closed process.
                    continue
                # A failed join must not bypass the final termination attempt.
                try:
                    self._kill_worker(child)
                    child.join(timeout=1)
                    if child.is_alive():
                        raise RuntimeError("worker could not be reaped after retry")
                    child.close()
                except Exception as retry_error:
                    errors.append(retry_error)
        if errors:
            raise RuntimeError(f"{len(errors)} worker cleanup failure(s)") from errors[0]

    def _monitor_workers(self) -> None:
        """Monitor readiness-complete workers until normal drain or a bounded failure."""
        assert self._resources is not None

        started = time.monotonic()
        stop_published = False
        while True:
            self._check_exits()
            self._receive_health()
            if self.duration_s is not None and time.monotonic() - started >= self.duration_s:
                self.request_shutdown()
            if self._shutdown_requested and not stop_published:
                self._resources.capture_stop.set()
                stop_published = True
                self._logger.info("shutdown_requested: stop capture, drain processing and reporter")
            if time.monotonic() - self._last_memory_sample >= 1:
                self._memory_observation = self._memory_monitor.sample(
                    [
                        os.getpid(),
                        *[
                            p.pid
                            for p in self._children.values()
                            if p.is_alive() and p.pid is not None
                        ],
                    ]
                )
                self._last_memory_sample = time.monotonic()
            stalled = self._health.check(time.monotonic_ns(), self.config.health_timeout_s)
            if stalled:
                raise RuntimeError(f"stalled stages: {[s.value for s in stalled]}")
            if (
                self._shutdown_at is not None
                and time.monotonic() - self._shutdown_at > self.config.shutdown_timeout_s
            ):
                raise RuntimeError("graceful shutdown deadline exceeded")
            if self._children and all(p.exitcode is not None for p in self._children.values()):
                break

    def _reserve_summary(self) -> ResultOutput | None:
        """Reserve the secure output descriptor before allocating pools or workers."""
        if self.summary_path is None:
            return None
        if self.config.output_path is not None and os.path.abspath(
            self.summary_path
        ) == os.path.abspath(self.config.output_path):
            raise ValueError("summary and result output must use different paths")
        return ResultOutput(self.summary_path, 65536)

    @staticmethod
    def _failure_listener(
        listener: LogListenerOwner | None, channel: DatagramChannel | None
    ) -> LogListenerOwner | None:
        """Provide a bounded error recipient even after the terminal diagnostic drain."""
        if (
            listener is not None
            and not listener.is_alive
            and listener.state in (ListenerState.CLOSED, ListenerState.FAILED)
        ):
            assert channel is not None
            listener = LogListenerOwner(channel)
            listener.start()
        return listener

    def run(self) -> int:
        """Own allocation, readiness, monitoring, drain and emergency cleanup; return status.

        Signals only record intent. On failure, use parent process handles to kill/reap
        children without asking dead participants to notify or return data. Nested finally
        scopes isolate logging, mapping disposal and signal restoration. Diagnostic
        output is best-effort and has a bounded close budget.
        """
        if self._ran:
            raise RuntimeError("PipelineSupervisor is one-shot; create a new instance")
        self._ran = True
        listener = None
        parent_logs = None
        summary_output: ResultOutput | None = None
        old_handlers = {}
        final = None
        code = 1
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, lambda *_: self.request_shutdown())
            parent_logs = DatagramChannel(self.config.log_queue_capacity * 2048)
            listener = LogListenerOwner(parent_logs)
            listener.start()
            self._logger.handlers.clear()
            self._logger.addHandler(BoundedQueueHandler(parent_logs))
            summary_output = self._reserve_summary()
            self._owner = TransportOwner(self.config, logs=parent_logs)
            self._resources = self._owner.worker_resources()
            self._logger.info(
                "pool_created name=%s slots=%d bytes=%d",
                self._resources.memory.name,
                self._resources.memory.slot_count,
                self._owner.size_bytes,
            )
            if self._shutdown_requested:
                raise RuntimeError("shutdown requested before startup")
            self._start()
            self._owner.seal()
            self._monitor_workers()
            for child in self._children.values():
                child.join()
            # Final atomic summaries were sent before process exit; drain them now.
            while True:
                try:
                    message = self._resources.health.get_nowait()
                except Empty:
                    break
                if isinstance(message, StageSummary):
                    self._summaries[message.stage] = message.cameras
                    self._diagnostics[message.stage.value] = message.diagnostics
                elif isinstance(message, HealthEvent):
                    self._health.observe(message)
            accounting = self._verify_accounting()
            self._logger.info(
                "pipeline_summary %s",
                json.dumps(
                    {
                        "cameras": accounting,
                        "status": "frames_drained",
                        "memory": self._memory_observation,
                        "diagnostics": {"log_counts_final": False},
                    }
                ),
            )
            # All producers have exited. Finish the bounded diagnostic drain before
            # taking the final loss counters; no additional success logs follow.
            logs_drained = listener.close(graceful=True)
            diagnostics = {
                "workers": self._diagnostics,
                "parent_dropped_logs": sum(
                    h.dropped_records
                    for h in self._logger.handlers
                    if isinstance(h, BoundedQueueHandler)
                ),
                "listener_dropped_logs": listener.dropped_records,
                "listener_drained": logs_drained and listener.state is ListenerState.CLOSED,
                "listener_state": listener.state.value,
                "log_counts_final": logs_drained,
            }
            final = {
                "schema_version": 1,
                "cameras": accounting,
                "exit_code": 0,
                "diagnostics": diagnostics,
                "memory": self._memory_observation,
            }
            code = 0
        except Exception:
            # Mandatory summary output can fail after the success-path log drain.
            # Reopen a bounded diagnostic consumer so the failure has a recipient.
            listener = self._failure_listener(listener, parent_logs)
            self._logger.exception("pipeline_failed")
            # A dead worker may have corrupted any data Queue it was using.
            # Emergency cleanup never needs those queues, returned credits or
            # Condition notifications. Only parent-owned process handles matter.
        finally:
            cleanup_errors = self._cleanup(code, listener, parent_logs, old_handlers)
            if cleanup_errors:
                code = 1
                self._report_finalization_errors(cleanup_errors)
            if summary_output is not None:
                try:
                    if final is not None:
                        final["exit_code"] = code
                        final["cleanup_completed"] = not cleanup_errors
                        final["cleanup_errors"] = [str(error)[:1024] for error in cleanup_errors]
                        summary_output.finish(json.dumps(final) + "\n")
                    else:
                        summary_output.close()
                except Exception as exc:
                    code = 1
                    self._report_finalization_errors([exc])
        return code

    @staticmethod
    def _report_finalization_errors(errors: list[Exception]) -> None:
        """Use a fresh bounded diagnostic owner after original IPC has been disposed.

        Final summary failures must stay observable without synchronous stderr writes
        or resurrecting disposed transport handles. Diagnostic failures remain best effort.
        """
        with suppress(Exception):
            channel = DatagramChannel()
            listener = None
            try:
                listener = LogListenerOwner(channel)
                listener.start()
                handler = BoundedQueueHandler(channel)
                for error in errors:
                    handler.handle(
                        logging.LogRecord(
                            "cv_pipeline.parent",
                            logging.ERROR,
                            "",
                            0,
                            "pipeline_failed during finalization: %s",
                            (str(error)[:2048],),
                            None,
                        )
                    )
            finally:
                try:
                    if listener is not None:
                        listener.close()
                finally:
                    channel.close()

    def _cleanup(
        self,
        code: int,
        listener: LogListenerOwner | None,
        parent_logs: DatagramChannel | None,
        old_handlers: dict[signal.Signals, Any],
    ) -> list[Exception]:
        """Attempt every independent cleanup, preserving failures for final status.

        Essential teardown precedes summary publication. Diagnostic shutdown remains
        bounded; a blocked custom sink is best effort, not an essential resource owner.
        """
        actions: list[Callable[[], object]] = [lambda: self._reap_workers(code)]
        if listener is not None:
            actions.append(lambda: listener.close(graceful=True))
        if self._owner is not None:
            actions.append(self._owner.close)
        if parent_logs is not None:
            actions.append(parent_logs.close)
        errors = []
        for action in actions:
            try:
                action()
            except Exception as exc:
                errors.append(exc)
        self._logger.handlers.clear()
        self._logger.addHandler(logging.NullHandler())
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception as exc:
                errors.append(exc)
        return errors
