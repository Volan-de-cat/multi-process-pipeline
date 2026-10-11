"""One Processing process composes scheduling, preprocessing and inference.

Raw-slot leases end before inference, so capture can reuse its bounded storage.
Per-frame processing latency includes the full batch inference interval and time
spent preprocessing later inputs. Results preserve the retained input order.
A failure releases collected unvisited slots before propagating to the worker
boundary; a crashed process is recovered by parent-owned pool disposal.
"""

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Full, Queue

import cv2
import numpy as np
from numpy.typing import NDArray

from .adapters import FakeInferenceAdapter, SystemClock
from .batching import BatchCollector
from .config import PipelineConfig
from .contracts import Clock, InferenceEngine, StopSignal
from .messages import (
    DropReason,
    EndOfStream,
    FrameDescriptor,
    FrameMetadata,
    ProcessedResult,
    ProcessingTimings,
    Stage,
)
from .observability import MetricsRecorder
from .preprocessing import FramePreprocessor
from .runtime import PipelineAborted, StageRuntime, put_until_stopped, worker_scope
from .transport import FrameTransport, ProcessingResources, WorkerResources


class GrayscalePool:
    """Fixed reusable destinations shared by one preparer and one inference owner."""

    def __init__(self, count: int, size: tuple[int, int], abort: StopSignal) -> None:
        self._abort = abort
        self._free: Queue[NDArray[np.uint8]] = Queue(maxsize=count)
        for _ in range(count):
            self._free.put(np.empty((size[1], size[0]), dtype=np.uint8))

    def acquire(self) -> NDArray[np.uint8]:
        """Wait cooperatively for exclusive ownership of a reusable destination."""
        while not self._abort.is_set():
            try:
                return self._free.get(timeout=0.1)
            except Empty:
                continue
        raise PipelineAborted("grayscale buffer acquisition aborted")

    def release(self, frame: NDArray[np.uint8]) -> None:
        """Return ownership after inference completes; overflow exposes double returns."""
        self._free.put_nowait(frame)


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """Named timing/identity fields for one detached input, without positional tuples."""

    metadata: FrameMetadata
    started_monotonic_ns: int
    preprocessing_finished_monotonic_ns: int


@dataclass(slots=True)
class PreparedBatch:
    """Independent grayscale pixels with their identities and preprocessing boundaries."""

    inputs: list[NDArray[np.uint8]]
    retained: list[PreparedFrame]


class BatchProcessor:
    """Single inference owner; detach all retained pixels before simulated GPU work."""

    def __init__(
        self,
        transport: FrameTransport,
        preprocessor: FramePreprocessor,
        engine: InferenceEngine,
        clock: Clock,
        max_frame_age_ms: float,
        metrics: MetricsRecorder | None = None,
        buffers: GrayscalePool | None = None,
    ) -> None:
        """Compose transport, preprocessor, inference backend, clock and input-age limit.

        max_frame_age_ms is checked at preprocessing start, not an end-to-end deadline.
        Optional metrics are stage-local and remain bounded.
        """
        self._transport = transport
        self._preprocessor = preprocessor
        self._engine = engine
        self._clock = clock
        self._max_age_ns = int(max_frame_age_ms * 1e6)
        self.metrics = metrics if metrics is not None else MetricsRecorder()
        self._buffers = buffers

    def process(self, descriptors: list[FrameDescriptor]) -> list[ProcessedResult]:
        """Serial compatibility path; preparation and inference obey identical ownership."""
        return self.infer(self.prepare(descriptors))

    def prepare(
        self, descriptors: list[FrameDescriptor], progress: Callable[[], None] | None = None
    ) -> PreparedBatch:
        """Detach raw leases into owned grayscale buffers; roll back every acquired buffer."""
        inputs: list[NDArray[np.uint8]] = []
        retained: list[PreparedFrame] = []
        pending = iter(descriptors)
        try:
            for descriptor in pending:
                started = self._clock.monotonic_ns()
                destination = None
                try:
                    with self._transport.lease(descriptor) as frame:
                        if started - descriptor.metadata.received_monotonic_ns > self._max_age_ns:
                            self.metrics.record_drop(
                                descriptor.metadata.camera_id, DropReason.STALE
                            )
                            if progress is not None:
                                progress()
                            continue
                        if self._buffers is not None:
                            destination = self._buffers.acquire()
                            processed = self._preprocessor.apply(frame, destination)
                        else:
                            processed = self._preprocessor.apply(frame)
                    del frame
                    inputs.append(processed)
                    destination = None  # PreparedBatch owns this pooled destination now.
                    retained.append(
                        PreparedFrame(descriptor.metadata, started, self._clock.monotonic_ns())
                    )
                    if progress is not None:
                        progress()
                finally:
                    if destination is not None:
                        assert self._buffers is not None
                        self._buffers.release(destination)
        except BaseException:
            for descriptor in pending:
                self._transport.release(descriptor)
            self.recycle(PreparedBatch(inputs, retained))
            raise
        return PreparedBatch(inputs, retained)

    def recycle(self, prepared: PreparedBatch) -> None:
        """Return detached pixels after inference or cancellation; never before engine return."""
        if self._buffers is not None:
            for frame in prepared.inputs:
                self._buffers.release(frame)
        prepared.inputs.clear()

    def infer(self, prepared: PreparedBatch) -> list[ProcessedResult]:
        """Run inference on the calling owner; release pooled pixels even on engine failure."""
        try:
            if not prepared.inputs:
                return []
            inference_started = self._clock.monotonic_ns()
            inferred = self._engine.infer(prepared.inputs)
            finished = self._clock.monotonic_ns()
            if len(inferred) != len(prepared.retained):
                raise RuntimeError("inference result count differs from inputs")
            results: list[ProcessedResult] = []
            for retained, inference in zip(prepared.retained, inferred, strict=True):
                metadata = retained.metadata
                started = retained.started_monotonic_ns
                timings = ProcessingTimings(
                    started,
                    retained.preprocessing_finished_monotonic_ns,
                    inference_started,
                    finished,
                )
                results.append(ProcessedResult(metadata, inference, timings, len(prepared.inputs)))
                self.metrics.record_frame(metadata.camera_id, Stage.PROCESSING)
                self.metrics.record_latency(metadata.camera_id, (finished - started) / 1e6)
            return results
        finally:
            self.recycle(prepared)


class PreparationPhase(Enum):
    """Local preparer state; only empty-input polling is legitimately idle."""

    IDLE = "idle"
    COLLECTING = "collecting"
    PREPARING = "preparing"
    WAITING = "waiting"


class OverlappedBatches:
    """One preparer thread, one waiting batch, and inference only on the stage main thread.

    The total grayscale pool covers the current, waiting and preparing batches.
    Errors travel through the same bounded queue. Active preparation remains subject
    to the main thread's health budget; waiting for source input is legitimate idle.
    """

    def __init__(
        self,
        collector: BatchCollector,
        processor: BatchProcessor,
        abort: StopSignal,
        runtime: StageRuntime,
        health_timeout_s: float = 5.0,
    ) -> None:
        self.collector = collector
        self.processor = processor
        self.abort = abort
        self.runtime = runtime
        self.queue: Queue[PreparedBatch | BaseException | None] = Queue(maxsize=1)
        self.stop = threading.Event()
        self._state_lock = threading.Lock()
        self._phase = PreparationPhase.COLLECTING
        self._last_observation_ns = time.monotonic_ns()
        self._health_timeout_ns = int(health_timeout_s * 1e9)
        self.collector.set_control(self.observe_collection, self.stop)
        self.thread = threading.Thread(target=self._produce, name="batch-preparer")

    def _observe(self, phase: PreparationPhase) -> None:
        """Publish a consistent local observation without touching StageRuntime."""
        with self._state_lock:
            self._phase = phase
            self._last_observation_ns = time.monotonic_ns()

    def observe_collection(self, *, idle: bool) -> None:
        """A live idle poll refreshes liveness; a blocked poll stops observations."""
        self._observe(PreparationPhase.IDLE if idle else PreparationPhase.COLLECTING)

    def _check_preparer(self) -> None:
        """Bound preparer inactivity independently of main-thread heartbeats.

        Completed frames refresh preparation observations; the health budget
        bounds time between observations. Live empty polling refreshes liveness.
        """
        with self._state_lock:
            phase, observed = self._phase, self._last_observation_ns
        if time.monotonic_ns() - observed > self._health_timeout_ns:
            raise RuntimeError(f"batch preparer stalled: {phase.value}")
        self.runtime.tick(idle=phase is PreparationPhase.IDLE)

    def _put(self, value: PreparedBatch | BaseException | None) -> bool:
        while not self.stop.is_set() and not self.abort.is_set():
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            while not self.stop.is_set() and not self.abort.is_set():
                self._observe(PreparationPhase.COLLECTING)
                descriptors = self.collector.collect()
                if not descriptors:
                    self._put(None)
                    return
                self._observe(PreparationPhase.PREPARING)
                prepared = self.processor.prepare(
                    descriptors, lambda: self._observe(PreparationPhase.PREPARING)
                )
                self._observe(PreparationPhase.WAITING)
                if not self._put(prepared):
                    self.processor.recycle(prepared)
                    return
        except BaseException as exc:
            self._put(exc)

    def batches(self) -> Iterator[PreparedBatch]:
        self.thread.start()
        while True:
            try:
                value = self.queue.get(timeout=0.1)
            except Empty:
                self._check_preparer()
                continue
            if isinstance(value, BaseException):
                raise value
            if value is None:
                return
            yield value

    def close(self) -> bool:
        """Stop/reap the preparer; a stalled thread leaves mapping disposal to parent kill."""
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.5)
        while True:
            try:
                value = self.queue.get_nowait()
            except Empty:
                break
            if isinstance(value, PreparedBatch):
                self.processor.recycle(value)
        return not self.thread.is_alive()


def serial_results(
    collector: BatchCollector, processor: BatchProcessor, runtime: StageRuntime
) -> Iterator[list[ProcessedResult]]:
    """Collect descriptors and synchronously prepare/infer on one owner thread."""
    while descriptors := collector.collect():
        runtime.tick(progress=True, force=True)
        prepared = processor.prepare(descriptors, lambda: runtime.tick(progress=True, force=True))
        # Give inference its own budget after preparation, including partial batches.
        runtime.tick(progress=True, force=True)
        yield processor.infer(prepared)


def overlapped_results(
    overlap: OverlappedBatches, processor: BatchProcessor, runtime: StageRuntime
) -> Iterator[list[ProcessedResult]]:
    """Consume exclusively PreparedBatch values; inference retains main-thread ownership."""
    for prepared in overlap.batches():
        runtime.tick(progress=True, force=True)
        yield processor.infer(prepared)


def publish_results(
    results: list[ProcessedResult],
    resources: ProcessingResources | WorkerResources,
    runtime: StageRuntime,
) -> None:
    """Publish one completed batch with bounded backpressure and real progress."""
    for result in results:
        put_until_stopped(resources.results, result, resources.abort, runtime.tick)
        # Publish the refreshed budget before the next potentially long operation.
        runtime.tick(progress=True, force=True)


def run_processing(
    config: PipelineConfig, resources: ProcessingResources | WorkerResources
) -> None:
    """Spawn-safe process entry for fair batching and bounded result publication.

    Configure OpenCV threads once. Input EOS flushes the last partial batch, followed
    by result EOS and final metrics. Child mapping close never unlinks parent memory.
    """
    with worker_scope(Stage.PROCESSING, resources) as runtime:
        cv2.setNumThreads(config.preprocessing_threads)
        transport = FrameTransport(resources)
        overlap: OverlappedBatches | None = None
        try:
            clock = SystemClock()
            processor = BatchProcessor(
                transport,
                FramePreprocessor(config.output_size, config.color_order),
                FakeInferenceAdapter(),
                clock,
                config.max_frame_age_ms,
                buffers=GrayscalePool(
                    config.batch_size * (3 if config.processing_overlap else 1),
                    config.output_size,
                    resources.abort,
                ),
            )
            collector = BatchCollector(
                transport,
                len(config.cameras),
                config.batch_size,
                config.batch_wait_ms,
                clock,
                None if config.processing_overlap else runtime.tick,
                config.frame_policy,
                processor.metrics,
            )
            overlap = (
                OverlappedBatches(
                    collector, processor, resources.abort, runtime, config.health_timeout_s
                )
                if config.processing_overlap
                else None
            )
            runtime.ready()
            results_batches = (
                overlapped_results(overlap, processor, runtime)
                if overlap is not None
                else serial_results(collector, processor, runtime)
            )
            for results in results_batches:
                publish_results(results, resources, runtime)
            put_until_stopped(
                resources.results, EndOfStream(Stage.PROCESSING), resources.abort, runtime.tick
            )
            runtime.summary(processor.metrics)
        finally:
            if overlap is None or overlap.close():
                transport.close()
