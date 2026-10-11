"""Serializable pipeline contracts, with no pixel arrays or live views.

Metadata survives camera reconnects; successful-read IDs may have gaps due to
drops but never reset. UTC timestamps label receipt, while monotonic nanoseconds
measure intervals across same-host processes. Health messages and stage summaries
travel independently of bounded result delivery and best-effort diagnostic logs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypedDict


class Stage(str, Enum):
    """Names of the three required independent worker processes."""

    CAPTURE = "capture"
    PROCESSING = "processing"
    REPORTER = "reporter"


class DropReason(str, Enum):
    """Explicit mutually classified frame outcomes used in final accounting."""

    NO_FREE_SLOT = "no_free_slot"
    STALE = "stale"
    SHUTDOWN = "shutdown"
    PUBLICATION_TIMEOUT = "publication_timeout"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Camera identity and receipt clocks. frame_id increases after successful reads
    and does not reset on reconnect; connection_epoch identifies the new instance.
    """

    camera_id: str
    frame_id: int
    connection_epoch: int
    received_utc_ns: int
    received_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class FrameDescriptor:
    """Small raw-slot reference with a protocol version. Publication transfers
    exclusive ownership; it contains no frame bytes or process-local pointer.
    """

    metadata: FrameMetadata
    slot_id: int
    protocol_version: int = 1


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Normalized ordered output from the original simulated model."""

    mean_brightness: float
    detection: bool


@dataclass(frozen=True, slots=True)
class ProcessingTimings:
    """Monotonic boundaries in nanoseconds, including one shared batch inference
    interval. UTC is deliberately excluded from duration calculations.
    """

    started_monotonic_ns: int
    preprocessing_finished_monotonic_ns: int
    inference_started_monotonic_ns: int
    finished_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class ProcessedResult:
    """Small output record linking retained input identity, detection and timing."""

    metadata: FrameMetadata
    inference: InferenceResult
    timings: ProcessingTimings
    batch_size: int


class ReportedResult(TypedDict):
    """Complete external JSONL result contract; timings are milliseconds."""

    type: Literal["result"]
    camera_id: str
    frame_id: int
    connection_epoch: int
    timestamp: str
    received_utc_ns: int
    mean_brightness: float
    detection: bool
    batch_size: int
    processing_latency_ms: float
    queue_and_batch_wait_ms: float
    preprocessing_ms: float
    inference_ms: float
    capture_to_report_ms: float


@dataclass(frozen=True, slots=True)
class EndOfStream:
    """FIFO drain marker. Capture emits one per camera; Processing emits one
    only after every accepted result has been published.
    """

    stage: Stage


@dataclass(frozen=True, slots=True)
class HealthEvent:
    """Atomic worker state snapshot. Idle suppresses progress expiry, not heartbeat
    expiry; errors request whole-pipeline failure rather than silent restart.
    """

    stage: Stage
    monotonic_ns: int
    ready: bool
    last_progress_monotonic_ns: int
    error: str | None = None
    idle: bool = False


class CameraCounters(TypedDict):
    """Required identity and nonnegative stage-local totals, including explicit zeroes."""

    camera_id: str
    captured: int
    processed: int
    reported: int
    dropped_no_slot: int
    dropped_stale: int
    dropped_shutdown: int
    dropped_publication_timeout: int
    dropped_superseded: int


class CameraMetrics(CameraCounters, total=False):
    """Complete counters with optional recent latency percentiles."""

    latency_ms_recent: dict[str, float]


def empty_camera_metrics(camera_id: str) -> CameraMetrics:
    """Represent a configured camera with no outcomes at a particular stage."""
    return CameraMetrics(
        camera_id=camera_id,
        captured=0,
        processed=0,
        reported=0,
        dropped_no_slot=0,
        dropped_stale=0,
        dropped_shutdown=0,
        dropped_publication_timeout=0,
        dropped_superseded=0,
    )


class FrameAccounting(TypedDict):
    """Complete per-camera terminal counters, with no unaccounted in-flight frames."""

    captured: int
    processed: int
    reported: int
    dropped_no_slot: int
    dropped_stale: int
    dropped_shutdown: int
    dropped_publication_timeout: int
    dropped_superseded: int
    in_flight: int


@dataclass(frozen=True, slots=True)
class StageSummary:
    """Required final stage-local counters, combined by Supervisor after clean exit."""

    stage: Stage
    cameras: dict[str, CameraMetrics]
    diagnostics: dict[str, int] = field(default_factory=dict)
