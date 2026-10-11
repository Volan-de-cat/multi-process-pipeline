"""Worker boundary for signals, health transitions and queue cleanup.

Both ordinary SIGINT and SIGTERM belong to the parent. Work-start transitions
refresh active budgets after valid idle time, but recurring heartbeats do not
refresh progress. Stop bytes and atomic status datagrams remain usable after a
peer crash. Data Queue cleanup is cooperative normally and nonessential in the
parent emergency path; the parent may force-kill a blocked child.
"""

import logging
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from queue import Full
from typing import TypeVar

from .contracts import MessageQueue, StopSignal
from .lifecycle import bind_parent_death
from .logging_setup import BoundedQueueHandler, configure_worker_logging
from .messages import HealthEvent, Stage, StageSummary
from .observability import MetricsRecorder
from .transport import ControlResources, WorkerResources, close_worker_queues

T = TypeVar("T")


class PipelineAborted(RuntimeError):
    """Another stage failed or the parent exhausted its shutdown deadline."""


def put_until_stopped(
    queue: MessageQueue[T], item: T, abort: StopSignal, tick: Callable[..., None] | None = None
) -> None:
    """Retry a bounded Queue put with 100 ms checks of the monotonic abort byte.

    Call tick during backpressure so the parent can distinguish a live but stuck
    producer. Abort raises PipelineAborted rather than silently losing required output.
    """
    while not abort.is_set():
        try:
            queue.put(item, timeout=0.1)
            return
        except Full:
            if tick is not None:
                tick()
    raise PipelineAborted("pipeline aborted while publishing")


class StageRuntime:
    """Worker-local readiness/progress state; its control messages never publish pixels."""

    def __init__(self, stage: Stage, resources: ControlResources | WorkerResources) -> None:
        """Begin unready and idle with a monotonic progress baseline."""
        self.stage = stage
        self.resources = resources
        self.last_progress = time.monotonic_ns()
        self._last_health = 0
        self._ready = False
        self._idle = True

    def ready(self) -> None:
        """Publish readiness immediately after worker resources and dependencies initialize."""
        self._ready = True
        self.tick(progress=True, force=True, idle=True)

    def tick(self, progress: bool = False, idle: bool = False, force: bool = False) -> None:
        """Publish health periodically or on demand, refreshing only real work budgets.

        A completed operation or idle-to-active transition refreshes last_progress.
        Recurring active heartbeats do not, so they cannot hide a stuck operation.
        Diagnostic status overflow may drop a heartbeat, never block the stage indefinitely.
        """
        if self.resources.abort.is_set():
            raise PipelineAborted("pipeline emergency stop")
        now = time.monotonic_ns()
        if progress:
            self.last_progress = now
        elif self._idle and not idle:
            # New work gets a new budget after a legitimate idle period. This
            # only runs on the transition, so repeated heartbeats cannot hide a stall.
            self.last_progress = now
        self._idle = idle
        if force or now - self._last_health >= 500_000_000:
            event = HealthEvent(self.stage, now, self._ready, self.last_progress, idle=idle)
            with suppress(Full):
                self.resources.health.put(event, timeout=0.1)
            self._last_health = now

    def summary(self, metrics: MetricsRecorder) -> None:
        """Publish required final counters over bounded crash-safe status IPC and log a copy."""
        cameras = metrics.summary()
        logging.info("stage_summary stage=%s cameras=%s", self.stage.value, cameras)
        dropped = sum(
            h.dropped_records
            for h in logging.getLogger().handlers
            if isinstance(h, BoundedQueueHandler)
        )
        message = StageSummary(self.stage, cameras, {"dropped_logs": dropped})
        put_until_stopped(self.resources.health, message, self.resources.abort)


@contextmanager
def worker_scope(
    stage: Stage, resources: ControlResources | WorkerResources
) -> Iterator[StageRuntime]:
    # Both ordinary signals belong to the parent, including process-group TERM.
    # Emergency escalation therefore uses kill(), not terminate() alone.
    """Install parent-owned signal policy and communicate worker failures.

    Both ordinary signals are ignored here; parent emergency cleanup uses SIGKILL.
    Errors publish status, request cooperative abort through a lock-free byte, and
    propagate a nonzero worker exit. Finally close local socket handles. Parent death always sends SIGKILL.
    """
    bind_parent_death(resources.parent_pid)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    configure_worker_logging(resources.logs)
    runtime = StageRuntime(stage, resources)
    try:
        yield runtime
    except BaseException as exc:
        logging.exception("stage_failed stage=%s", stage.value)
        event = HealthEvent(
            stage, time.monotonic_ns(), False, runtime.last_progress, error=repr(exc)[:2048]
        )
        with suppress(Full):
            resources.health.put(event, timeout=0.1)
        resources.abort.set()
        raise
    finally:
        close_worker_queues(resources)
