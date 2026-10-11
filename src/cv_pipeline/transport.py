"""Raw-frame ownership through fixed shared memory and bounded FIFO queues.

Supervisor creates and commits the pool; each child attaches by name and only
closes its local mapping. Capture owns a reserved slot until descriptor publication
commits. Processing reads through a lease and returns the credit after copying
into independent grayscale storage. Health/log datagrams and monotonic stop
bytes are separate from data queues so a killed producer cannot block the parent
control path. Supervisor unlinks after readers attach; mappings close after workers are reaped.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import uuid
from dataclasses import dataclass
from logging import LogRecord
from multiprocessing import shared_memory
from queue import Empty, Full
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import PipelineConfig, validate_config
from .contracts import MessageQueue, StopSignal
from .ipc import DatagramChannel, SharedStopSignal, SlotCredits
from .messages import (
    EndOfStream,
    FrameDescriptor,
    HealthEvent,
    ProcessedResult,
    Stage,
    StageSummary,
)
from .resources import reserve_shared_memory


@dataclass(frozen=True, slots=True)
class SharedMemorySpec:
    """Spawn-safe mapping geometry and camera slot partitions; contains no attached view."""

    name: str
    frame_shape: tuple[int, int, int]
    slot_count: int
    slots_per_camera: int
    camera_ids: tuple[str, ...]
    dtype: str = "uint8"


@dataclass(frozen=True, slots=True)
class WorkerResources:
    """Serializable IPC handles and plain memory spec; never carry live camera/NumPy views."""

    memory: SharedMemorySpec
    parent_pid: int
    free_slots: tuple[MessageQueue[int], ...]
    ready_frames: tuple[MessageQueue[FrameDescriptor | EndOfStream], ...]
    results: MessageQueue[ProcessedResult | EndOfStream]
    health: MessageQueue[HealthEvent | StageSummary]
    logs: MessageQueue[LogRecord]
    capture_stop: StopSignal
    abort: StopSignal


@dataclass(frozen=True, slots=True)
class ControlResources:
    """Common worker control handles, excluding unrelated data channels."""

    parent_pid: int
    health: MessageQueue[HealthEvent | StageSummary]
    logs: MessageQueue[LogRecord]
    abort: StopSignal


@dataclass(frozen=True, slots=True)
class FrameResources(ControlResources):
    """Shared raw-frame mapping and ownership channels for Capture/Processing."""

    memory: SharedMemorySpec
    free_slots: tuple[MessageQueue[int], ...]
    ready_frames: tuple[MessageQueue[FrameDescriptor | EndOfStream], ...]


@dataclass(frozen=True, slots=True)
class CaptureResources(FrameResources):
    """Capture receives its stop signal but no inference-result channel."""

    capture_stop: StopSignal


@dataclass(frozen=True, slots=True)
class ProcessingResources(FrameResources):
    """Processing receives the result channel but cannot request ordinary capture stop."""

    results: MessageQueue[ProcessedResult | EndOfStream]


@dataclass(frozen=True, slots=True)
class ReporterResources(ControlResources):
    """Reporter sees only small results and common control; no pixel mapping or credits."""

    results: MessageQueue[ProcessedResult | EndOfStream]


class FrameLease:
    """Exclusive read scope that returns one raw-slot credit on normal or exceptional exit."""

    def __init__(self, descriptor: FrameDescriptor, transport: FrameTransport) -> None:
        """Bind a descriptor and process-local transport without reading the slot yet."""
        self._descriptor = descriptor
        self._transport = transport
        self._entered = False
        self._released = False

    def __enter__(self) -> NDArray[np.uint8]:
        """Validate the descriptor and expose a non-owning view; forbid re-entering this lease."""
        if self._entered:
            raise RuntimeError("lease cannot be entered twice")
        self._transport._validate(self._descriptor)
        self._entered = True
        return self._transport._view(self._descriptor.slot_id)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Return the credit exactly once; the caller must stop using the view after exit."""
        if self._entered and not self._released:
            self._transport.release(self._descriptor)
            self._released = True


class FrameTransport:
    """Child-local attachment and ownership transfer; it never unlinks the shared pool."""

    def __init__(self, resources: FrameResources | WorkerResources) -> None:
        """Attach by name, validate backing size/dtype and create a local reservation lock."""
        self.resources = resources
        self._spec = resources.memory
        self._memory = shared_memory.SharedMemory(name=self._spec.name)
        self._frame_bytes = int(np.prod(self._spec.frame_shape))
        if (
            self._memory.size != self._frame_bytes * self._spec.slot_count
            or self._spec.dtype != "uint8"
        ):
            self._memory.close()
            raise ValueError("shared-memory size or dtype mismatch")
        self._reserved: set[int] = set()
        self._lock = threading.Lock()
        self._closed = False

    def _validate(self, descriptor: FrameDescriptor) -> int:
        """Validate protocol, slot bounds, camera partition and successful-read identity.

        Return the owning camera index. This validates structure, not remote ownership:
        correct publication and FIFO credit transfer provide the latter.
        """
        if descriptor.protocol_version != 1:
            raise ValueError("unsupported frame protocol")
        slot = descriptor.slot_id
        if type(slot) is not int or not 0 <= slot < self._spec.slot_count:
            raise ValueError("invalid slot id")
        index = slot // self._spec.slots_per_camera
        if self._spec.camera_ids[index] != descriptor.metadata.camera_id:
            raise ValueError("slot belongs to a different camera")
        if (
            type(descriptor.metadata.frame_id) is not int
            or descriptor.metadata.frame_id <= 0
            or type(descriptor.metadata.connection_epoch) is not int
            or descriptor.metadata.connection_epoch <= 0
            or type(descriptor.metadata.received_monotonic_ns) is not int
            or descriptor.metadata.received_monotonic_ns <= 0
            or type(descriptor.metadata.received_utc_ns) is not int
            or descriptor.metadata.received_utc_ns <= 0
        ):
            raise ValueError("invalid frame identity")
        return index

    def _view(self, slot: int) -> NDArray[np.uint8]:
        """Create a temporary uint8 view at a slot offset without copying or claiming ownership."""
        if self._closed:
            raise RuntimeError("transport is closed")
        return np.ndarray(
            self._spec.frame_shape,
            dtype=np.uint8,
            buffer=self._memory.buf,
            offset=slot * self._frame_bytes,
        )

    def publish(self, metadata_frame: FrameDescriptor, frame: NDArray[np.uint8]) -> bool:
        """Copy into a reserved slot and commit ownership by publishing its descriptor.

        Return False on abort/queue timeout, and restore the credit on every failed
        publication including invalid metadata. Once committed, Capture must not modify
        this memory until Processing returns its credit.
        """
        with self._lock:
            if metadata_frame.slot_id not in self._reserved:
                raise RuntimeError("slot must be reserved before publication")
        published = False
        try:
            index = self._validate(metadata_frame)
            if frame.shape != self._spec.frame_shape or frame.dtype != np.uint8:
                raise ValueError("camera frame does not match shared-memory geometry")
            if self.resources.abort.is_set():
                return False
            np.copyto(self._view(metadata_frame.slot_id), frame)
            self.resources.ready_frames[index].put(metadata_frame, timeout=0.1)
            published = True
            return True
        except Full:
            return False
        finally:
            with self._lock:
                self._reserved.discard(metadata_frame.slot_id)
            if not published:
                # The reservation is trustworthy even if metadata validation
                # failed. Returning by slot avoids revalidating the bad metadata.
                self._return_credit(metadata_frame.slot_id)

    def try_reserve(self, camera_index: int) -> int | None:
        """Consume a camera-local credit without waiting, or return None for an explicit drop."""
        try:
            slot = self.resources.free_slots[camera_index].get(block=False)
        except Empty:
            return None
        if (
            not camera_index * self._spec.slots_per_camera
            <= slot
            < (camera_index + 1) * self._spec.slots_per_camera
        ):
            raise RuntimeError("free slot queue contains an invalid credit")
        with self._lock:
            if slot in self._reserved:
                raise RuntimeError("duplicate slot credit")
            self._reserved.add(slot)
        return slot

    def release(self, descriptor: FrameDescriptor) -> None:
        """Validate a consumed descriptor and return its raw-slot credit to that camera."""
        self._validate(descriptor)
        with self._lock:
            self._reserved.discard(descriptor.slot_id)
        self._return_credit(descriptor.slot_id)

    def _return_credit(self, slot_id: int) -> None:
        """Return a known valid slot after read completion or publication rollback."""
        index = slot_id // self._spec.slots_per_camera
        try:
            self.resources.free_slots[index].put(slot_id, timeout=0.1)
        except Full as exc:
            raise RuntimeError("slot released twice or credit queue corrupted") from exc

    def receive(self, camera_index: int, timeout_s: float) -> FrameDescriptor | EndOfStream | None:
        """Receive a validated descriptor/EOS, or None on timeout; preserve per-camera FIFO."""
        try:
            message = self.resources.ready_frames[camera_index].get(
                block=timeout_s > 0, timeout=timeout_s if timeout_s > 0 else None
            )
        except Empty:
            return None
        if isinstance(message, FrameDescriptor):
            if self._validate(message) != camera_index:
                raise ValueError("descriptor arrived on the wrong camera queue")
        elif not isinstance(message, EndOfStream):
            raise ValueError("unexpected capture message")
        return message

    def lease(self, descriptor: FrameDescriptor) -> FrameLease:
        """Create a read context manager; no raw view is exposed until context entry."""
        return FrameLease(descriptor, self)

    def close(self) -> None:
        """Close only this process mapping; all local raw views must already be out of use."""
        if not self._closed:
            self._memory.close()
            self._closed = True


class TransportOwner:
    """Supervisor-only pool creator/disposer; children receive handles but not this owner."""

    def __init__(self, config: PipelineConfig, logs: MessageQueue[LogRecord] | None = None) -> None:
        """Validate and allocate fixed storage, partition credits and construct spawn-safe IPC.

        Reserve backing pages before children start. posix_fallocate turns tmpfs
        ENOSPC into a recoverable exception before touching the mapping. All
        records use JSON datagrams with exact data-channel capacity credits.
        """
        validate_config(config)
        self.context = mp.get_context("spawn")
        self._queues: list[MessageQueue[Any]] = []
        self._memory = None
        self._unlinked = False
        self._closed = False
        slots_per_camera = config.resolved_slots_per_camera
        total_slots = len(config.cameras) * slots_per_camera
        size = int(np.prod(config.frame_shape)) * total_slots
        self.size_bytes = size
        # Linux shm_open may succeed despite insufficient tmpfs space; touching it
        # later would SIGBUS. Check and commit pages before starting any children.
        # Fixed tmpfs mount, not a predictable temporary filename.
        if os.path.isdir("/dev/shm"):  # nosec B108
            usage = os.statvfs("/dev/shm")  # nosec B108
            available = usage.f_bavail * usage.f_frsize
            if size + 1024 * 1024 > available:
                raise ValueError(
                    f"shared-memory pool needs {size / 2**20:.1f} MiB; "
                    f"only {available / 2**20:.1f} MiB available in /dev/shm"
                )
        try:
            self._memory = shared_memory.SharedMemory(
                create=True, size=size, name=f"cv_pipeline_{os.getpid()}_{uuid.uuid4().hex}"
            )
            # Reserve tmpfs backing atomically before touching the mapping: an
            # ENOSPC exception is recoverable, unlike a later SIGBUS page fault.
            # CPython's Linux SharedMemory owns this already-open shm descriptor.
            # Reserve that exact inode instead of reopening its raceable pathname.
            reserve_shared_memory(self._memory, size)
            np.ndarray((size,), dtype=np.uint8, buffer=self._memory.buf).fill(0)
            spec = SharedMemorySpec(
                self._memory.name,
                config.frame_shape,
                total_slots,
                slots_per_camera,
                tuple(c.camera_id for c in config.cameras),
            )
            free = tuple(
                self._slot_queue(i * slots_per_camera, slots_per_camera)
                for i in range(len(config.cameras))
            )
            ready = tuple(self._queue(slots_per_camera) for _ in config.cameras)
            health = self._queue()
            if logs is None:
                logs = self._queue(buffer_bytes=config.log_queue_capacity * 2048)
            else:
                self._queues.append(logs)
            self._resources = WorkerResources(
                spec,
                os.getpid(),
                free,
                ready,
                self._queue(config.result_queue_capacity),
                health,
                logs,
                SharedStopSignal(self.context),
                SharedStopSignal(self.context),
            )
        except BaseException:
            self.close()
            raise

    def _slot_queue(self, start: int, count: int) -> SlotCredits:
        """Register each counter immediately so partial initialization cannot leak handles."""
        slots = SlotCredits(self.context, start, count)
        self._queues.append(slots)
        return slots

    def _queue(
        self, capacity: int | None = None, *, buffer_bytes: int = 256_000
    ) -> DatagramChannel:
        """Create one bounded data Queue in the explicit spawn context and register cleanup."""
        queue = DatagramChannel(buffer_bytes, capacity=capacity, context=self.context)
        self._queues.append(queue)
        return queue

    def worker_resources(self) -> WorkerResources:
        """Export the handle bundle to workers; never serialize TransportOwner itself."""
        return self._resources

    def resources_for(
        self, stage: Stage
    ) -> CaptureResources | ProcessingResources | ReporterResources:
        """Export only the handles used by a stage; full resources stay in the parent."""
        r = self._resources
        common = (r.parent_pid, r.health, r.logs, r.abort)
        if stage is Stage.REPORTER:
            return ReporterResources(*common, r.results)
        frames = (*common, r.memory, r.free_slots, r.ready_frames)
        if stage is Stage.CAPTURE:
            return CaptureResources(*frames, r.capture_stop)
        if stage is Stage.PROCESSING:
            return ProcessingResources(*frames, r.results)
        raise ValueError("unknown worker stage")

    def seal(self) -> None:
        """Unlink the name after readers attach, keeping existing mappings usable.

        Stage restart is unsupported. Early unlink makes kernel mapping lifetime
        reclaim steady-state storage even after whole-tree SIGKILL/tracker death.
        Startup retains the name so Capture and Processing can attach by name.
        """
        if not self._unlinked:
            assert self._memory is not None
            self._memory.unlink()
            self._unlinked = True

    def close(self) -> None:
        """After workers are reaped, close IPC/mapping and unlink unsealed startup pools.

        Close local IPC copies, including unnamed eventfd counters.
        Mapping unlink has an independent finally even when IPC close fails.
        """
        if self._closed:
            return
        self._closed = True
        try:
            close_channels(self._queues)
        finally:
            if self._memory is not None:
                try:
                    self._memory.close()
                finally:
                    if not self._unlinked:
                        self._memory.unlink()
                        self._unlinked = True


def close_channels(channels: list[MessageQueue[Any]] | tuple[MessageQueue[Any], ...]) -> None:
    """Attempt every local close, then raise the first cleanup error with its cause.

    Mapping disposal has a separate owner finally scope. Preserve the original
    exception chain; the error message includes the number of failed closes.
    """
    errors: list[Exception] = []
    for channel in channels:
        try:
            channel.close()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"{len(errors)} IPC resource close(s) failed") from errors[0]


def close_worker_queues(resources: ControlResources | WorkerResources) -> None:
    """Close process-local sockets, counters and handles; no feeder threads exist."""
    channels: list[MessageQueue[Any]] = [resources.health, resources.logs]
    if isinstance(resources, (FrameResources, WorkerResources)):
        channels.extend((*resources.free_slots, *resources.ready_frames))
    if isinstance(resources, (ProcessingResources, ReporterResources, WorkerResources)):
        channels.append(resources.results)
    close_channels(channels)
