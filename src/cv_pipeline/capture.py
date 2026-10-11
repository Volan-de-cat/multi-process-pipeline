"""Capture stage: camera-local ownership inside one spawned process.

Each CameraWorker owns exactly one camera at a time, a continuous successful-read
sequence and retry policy. Shared-memory slots are camera-local; no free credit
means an accounted drop. The process target owns thread lifecycle and emits EOS
only after every producer has stopped, preserving FIFO drain ordering.
"""

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue

from .adapters import CameraDisconnectedError, SystemClock, create_camera
from .config import CameraConfig, PipelineConfig
from .contracts import Camera, Clock, StopSignal
from .messages import DropReason, EndOfStream, FrameDescriptor, FrameMetadata, Stage
from .observability import MetricsRecorder
from .runtime import StageRuntime, put_until_stopped, worker_scope
from .transport import CaptureResources, FrameTransport, WorkerResources


class CameraState(Enum):
    """Local camera operation used by the owner watchdog."""

    STARTING = "starting"
    CONNECTING = "connecting"
    READING = "reading"
    PUBLISHING = "publishing"
    RECONNECTING = "reconnecting"
    RELEASING = "releasing"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class CameraObservation:
    """Consistent operation-start and successful-read clocks under a local lock."""

    state: CameraState
    last_activity_ns: int
    last_success_ns: int


class ReconnectBackoff:
    """Camera-local bounded exponential retry timing with jitter; not shared across cameras."""

    def __init__(self, initial_s: float, maximum_s: float) -> None:
        """Store validated seconds and begin at the initial retry bound."""
        self._initial = initial_s
        self._maximum = maximum_s
        self._next = initial_s

    def next_delay_s(self) -> float:
        """Return jittered seconds between 80% and 100% of the current bound.

        Double the next bound up to maximum_s; the randomization avoids synchronized
        retries. Waiting is owned by CameraWorker so ordinary stop can interrupt it.
        """
        delay = random.uniform(self._next * 0.8, self._next)
        self._next = min(self._maximum, self._next * 2)
        return delay

    def reset(self) -> None:
        """Restore the initial bound after the configured stable-read policy succeeds."""
        self._next = self._initial


class CameraWorker:
    """Own camera identity, connection epoch and sequence in exactly one Capture thread."""

    def __init__(
        self,
        camera: CameraConfig,
        camera_index: int,
        config: PipelineConfig,
        transport: FrameTransport,
        stop: StopSignal,
        abort: StopSignal,
        clock: Clock,
        metrics: MetricsRecorder | None = None,
        factory: Callable[[CameraConfig], Camera] = create_camera,
    ) -> None:
        """Bind camera-local resources and injectable clock/factory without opening a camera.

        metrics may be shared by Capture threads and uses its own local lock. Transport
        is shared within Capture but reservations remain partitioned by camera_index.
        """
        self.camera_config = camera
        self.index = camera_index
        self.config = config
        self.transport = transport
        self.stop = stop
        self.abort = abort
        self.clock = clock
        self.metrics = metrics if metrics is not None else MetricsRecorder()
        self.factory = factory
        self._observation_lock = threading.Lock()
        now = clock.monotonic_ns()
        self._observation = CameraObservation(CameraState.STARTING, now, now)

    def observation(self) -> CameraObservation:
        """Read state and both clocks atomically without holding a lock during camera I/O."""
        with self._observation_lock:
            return self._observation

    def _observe(self, state: CameraState, success_ns: int | None = None) -> None:
        """Start an operation with a fresh budget or publish a completed read."""
        with self._observation_lock:
            self._observation = CameraObservation(
                state,
                self.clock.monotonic_ns(),
                self._observation.last_success_ns if success_ns is None else success_ns,
            )

    def run(self) -> None:
        """Read until ordinary stop or emergency abort, releasing the camera on exit.

        Only successful reads advance frame_id; reconnect advances connection_epoch.
        Timestamp after read, reserve a raw slot, then publish or count an explicit drop.
        A read completed during shutdown is counted and dropped, preserving accounting.
        Only expected disconnects retry; unexpected errors propagate to run_capture.
        """
        camera = None
        frame_id = 0
        epoch = 0
        stable_reads = 0
        backoff = ReconnectBackoff(self.config.reconnect_initial_s, self.config.reconnect_max_s)
        try:
            while not self.stop.is_set() and not self.abort.is_set():
                try:
                    if camera is None:
                        self._observe(CameraState.CONNECTING)
                        camera = self.factory(self.camera_config)
                        epoch += 1
                        stable_reads = 0
                        logging.info(
                            "camera_connected camera=%s epoch=%d",
                            self.camera_config.camera_id,
                            epoch,
                        )
                    self._observe(CameraState.READING)
                    frame = camera.read()
                except CameraDisconnectedError as exc:
                    logging.warning(
                        "camera_disconnected camera=%s error=%s", self.camera_config.camera_id, exc
                    )
                    if camera is not None:
                        self._observe(CameraState.RELEASING)
                        camera.release()
                        camera = None
                    self._observe(CameraState.RECONNECTING)
                    delay = backoff.next_delay_s()
                    logging.info(
                        "camera_retry camera=%s delay_s=%.3f", self.camera_config.camera_id, delay
                    )
                    self.stop.wait(delay)
                    continue
                frame_id += 1
                stable_reads += 1
                metadata = FrameMetadata(
                    self.camera_config.camera_id,
                    frame_id,
                    epoch,
                    self.clock.utc_ns(),
                    self.clock.monotonic_ns(),
                )
                self.metrics.record_frame(metadata.camera_id, Stage.CAPTURE)
                self._observe(CameraState.READING, metadata.received_monotonic_ns)
                if stable_reads >= 10:
                    # Brief connectivity must not erase an accumulated retry budget.
                    backoff.reset()
                if self.stop.is_set() or self.abort.is_set():
                    self.metrics.record_drop(metadata.camera_id, DropReason.SHUTDOWN)
                    del frame
                    break
                self._observe(CameraState.PUBLISHING)
                slot = self.transport.try_reserve(self.index)
                if slot is None:
                    self.metrics.record_drop(metadata.camera_id, DropReason.NO_FREE_SLOT)
                elif not self.transport.publish(FrameDescriptor(metadata, slot), frame):
                    self.metrics.record_drop(
                        metadata.camera_id,
                        DropReason.SHUTDOWN
                        if self.abort.is_set() or self.stop.is_set()
                        else DropReason.PUBLICATION_TIMEOUT,
                    )
                del frame
        finally:
            try:
                if camera is not None:
                    self._observe(CameraState.RELEASING)
                    camera.release()
            finally:
                self._observe(CameraState.STOPPED)


def _monitor_camera_threads(
    threads: list[threading.Thread],
    workers: list[CameraWorker],
    failures: Queue[tuple[str, BaseException]],
    metrics: MetricsRecorder,
    config: PipelineConfig,
    runtime: StageRuntime,
) -> None:
    """Apply per-camera stall/all-camera availability policy while publishing health."""
    last_count = 0
    last_log = time.monotonic()
    while any(t.is_alive() for t in threads):
        try:
            camera_id, error = failures.get_nowait()
        except Empty:
            pass
        else:
            raise RuntimeError(f"camera thread {camera_id} failed") from error
        now = time.monotonic_ns()
        observations = [worker.observation() for worker in workers]
        for worker, observation in zip(workers, observations, strict=True):
            # read() legitimately includes the camera's inter-frame wait.
            budget_s = config.health_timeout_s
            if observation.state is CameraState.READING:
                budget_s = max(budget_s, 1 / worker.camera_config.fps + 0.5)
            if (
                observation.state
                in (
                    CameraState.READING,
                    CameraState.PUBLISHING,
                    CameraState.CONNECTING,
                    CameraState.RELEASING,
                )
                and now - observation.last_activity_ns > budget_s * 1e9
            ):
                raise RuntimeError(f"camera thread stalled: {worker.camera_config.camera_id}")
        count = sum(metrics.snapshot(c.camera_id).captured for c in config.cameras)
        if all(now - w.last_success_ns > config.availability_timeout_s * 1e9 for w in observations):
            raise RuntimeError("all cameras unavailable beyond availability timeout")
        runtime.tick(progress=count != last_count, idle=count == last_count)
        last_count = count
        if time.monotonic() - last_log >= 2:
            logging.info("capture_metrics cameras=%s", metrics.summary())
            last_log = time.monotonic()
        time.sleep(0.05)


def run_capture(config: PipelineConfig, resources: CaptureResources | WorkerResources) -> None:
    """Spawn-safe process entry: attach pool, run camera threads, then publish EOS.

    Thread exceptions travel to the owner loop, which checks per-camera activity
    for stalls. Join producers before EOS so no frame can arrive behind it. During
    a stalled-thread failure, do not unmap memory still referenced by that thread;
    the parent force-kills/reaps the process before disposing its pool.
    """
    with worker_scope(Stage.CAPTURE, resources) as runtime:
        transport = FrameTransport(resources)
        metrics = MetricsRecorder()
        failures: Queue[tuple[str, BaseException]] = Queue(maxsize=len(config.cameras))
        workers = [
            CameraWorker(
                c,
                i,
                config,
                transport,
                resources.capture_stop,
                resources.abort,
                SystemClock(),
                metrics,
                factory=create_camera,
            )
            for i, c in enumerate(config.cameras)
        ]

        def thread_target(worker: CameraWorker) -> None:
            """Forward camera-thread errors and stop producers before owner cleanup."""
            try:
                worker.run()
            except BaseException as exc:
                failures.put((worker.camera_config.camera_id, exc))
                resources.capture_stop.set()

        threads = [
            threading.Thread(
                target=thread_target, args=(w,), name=f"camera-{w.camera_config.camera_id}"
            )
            for w in workers
        ]
        started = []
        try:
            for thread in threads:
                thread.start()
                started.append(thread)
            runtime.ready()
            _monitor_camera_threads(started, workers, failures, metrics, config, runtime)
            for thread in started:
                thread.join()
            if not failures.empty():
                camera_id, error = failures.get_nowait()
                raise RuntimeError(f"camera thread {camera_id} failed") from error
            for queue in resources.ready_frames:
                put_until_stopped(queue, EndOfStream(Stage.CAPTURE), resources.abort, runtime.tick)
            runtime.summary(metrics)
        finally:
            resources.capture_stop.set()
            for thread in started:
                thread.join(timeout=0.5)
            # A stalled thread can still hold a mapping view. The parent will
            # terminate this failed process; never unmap underneath that thread.
            if not any(thread.is_alive() for thread in started):
                transport.close()
