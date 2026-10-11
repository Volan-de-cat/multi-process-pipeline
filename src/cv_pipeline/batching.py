"""Fair, deadline-bounded scheduling of raw frame descriptors.

The collector never touches pixels or waits for a particular camera. It tracks
per-camera EOS and flushes partial batches. Its first-frame deadline bounds batch
assembly, while the Supervisor independently bounds stage inactivity.
"""

from collections.abc import Callable

from .contracts import Clock, StopSignal
from .messages import DropReason, EndOfStream, FrameDescriptor, Stage
from .observability import MetricsRecorder
from .runtime import PipelineAborted
from .transport import FrameTransport


class BatchCollector:
    """Keep round-robin cursor and EOS state across calls; descriptors remain FIFO per camera."""

    def __init__(
        self,
        transport: FrameTransport,
        camera_count: int,
        batch_size: int,
        wait_ms: float,
        clock: Clock,
        tick: Callable[..., None] | None = None,
        frame_policy: str = "fifo",
        metrics: MetricsRecorder | None = None,
    ) -> None:
        """Bind transport, camera count, batch limit, first-frame wait and injectable clock.

        The optional tick callback publishes health without owning scheduling. Pending
        inputs are bounded by batch_size; input queues have independent slot bounds.
        """
        self._transport = transport
        self._camera_count = camera_count
        self._size = batch_size
        self._wait_ns = int(wait_ms * 1e6)
        self._clock = clock
        self._tick = tick
        self._ended: set[int] = set()
        self._cursor = 0
        self._policy = frame_policy
        self._metrics = metrics
        self._stop: StopSignal | None = None

    def set_control(self, tick: Callable[..., None], stop: StopSignal) -> None:
        """Bind preparer-local observation/cancellation before its thread starts.

        The callback observes polling operations rather than publishing stage health
        from a second thread. Cancellation leaves accepted leases for parent disposal.
        """
        self._tick = tick
        self._stop = stop

    def _check_stop(self) -> None:
        if self._transport.resources.abort.is_set() or (
            self._stop is not None and self._stop.is_set()
        ):
            raise PipelineAborted("batch collection aborted")

    def _observe(self, *, idle: bool) -> None:
        if self._tick is not None:
            self._tick(idle=idle)

    def _accept_eos(self, index: int, message: EndOfStream) -> None:
        """Apply the same producer validation in FIFO and latest scans."""
        if message.stage != Stage.CAPTURE:
            raise ValueError("unexpected EOS producer")
        self._ended.add(index)

    def collect(self) -> list[FrameDescriptor]:
        """Return up to batch_size descriptors or an EOS-flushed partial batch.

        The first descriptor starts the deadline. Poll every live camera fairly without
        blocking on a disconnected one; an empty return means all camera EOS arrived.
        Emergency abort raises PipelineAborted and parent recovery disposes the pool.
        """
        batch: list[FrameDescriptor] = []
        deadline = None
        while len(self._ended) < self._camera_count:
            self._check_stop()
            self._observe(idle=not batch)
            received = False
            for _ in range(self._camera_count):
                if deadline is not None and self._clock.monotonic_ns() >= deadline:
                    return batch
                self._check_stop()
                index = self._cursor
                self._cursor = (self._cursor + 1) % self._camera_count
                if index in self._ended:
                    continue
                # A stuck receive is active work, even when no frame has arrived yet.
                self._observe(idle=False)
                message = self._transport.receive(index, 0)
                if isinstance(message, EndOfStream):
                    self._accept_eos(index, message)
                elif message is not None:
                    if deadline is None:
                        deadline = self._clock.monotonic_ns() + self._wait_ns
                    if self._policy == "latest":
                        message = self._latest(index, message, deadline)
                    batch.append(message)
                    received = True
                    if len(batch) == self._size:
                        return batch
            if deadline is not None and self._clock.monotonic_ns() >= deadline:
                return batch
            if not received and len(self._ended) < self._camera_count:
                self._observe(idle=not batch)
                self._transport.resources.abort.wait(0.002)
        return batch

    def _latest(self, index: int, current: FrameDescriptor, deadline: int) -> FrameDescriptor:
        """Replace queued older descriptors only from the consuming owner.

        Bound each scan by the camera slot count so a live producer cannot starve
        other cameras. Release superseded leases and account every dropped frame.
        """
        count = self._transport.resources.memory.slots_per_camera
        for _ in range(count):
            if self._clock.monotonic_ns() >= deadline:
                break
            self._check_stop()
            self._observe(idle=False)
            newer = self._transport.receive(index, 0)
            if newer is None:
                break
            if isinstance(newer, EndOfStream):
                self._accept_eos(index, newer)
                break
            self._transport.release(current)
            if self._metrics is not None:
                self._metrics.record_drop(current.metadata.camera_id, DropReason.SUPERSEDED)
            current = newer
        return current
