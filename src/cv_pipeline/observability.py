"""Bounded stage-local counters, latency samples and Reporter FPS windows.

Capture shares a recorder between camera threads, so snapshots use a local lock.
Each stage records only its own outcomes; Supervisor combines final summaries.
Latency percentiles describe the latest 2048 samples per camera, not all history.
Reporter windows count emitted results and must span at least ten seconds.
"""

import threading
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np

from .contracts import Clock
from .messages import CameraMetrics, DropReason, Stage


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Immutable camera-local totals; zero fields belonging to other stages are intentional."""

    camera_id: str
    captured: int = 0
    processed: int = 0
    reported: int = 0
    dropped_no_slot: int = 0
    dropped_stale: int = 0
    dropped_shutdown: int = 0
    dropped_publication_timeout: int = 0
    dropped_superseded: int = 0


class MetricsRecorder:
    """Local recorder with bounded sample rings and a lock safe for Capture threads."""

    def __init__(self) -> None:
        """Allocate counters lazily by camera and latency rings capped at 2048 entries."""
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2048))
        self._lock = threading.Lock()

    def record_frame(self, camera_id: str, stage: Stage) -> None:
        """Increment the captured, processed or reported count for one successful stage outcome."""
        field = {
            Stage.CAPTURE: "captured",
            Stage.PROCESSING: "processed",
            Stage.REPORTER: "reported",
        }[stage]
        with self._lock:
            self._counts[camera_id][field] += 1

    def record_drop(self, camera_id: str, reason: DropReason) -> None:
        """Count one explicit drop reason; callers must avoid recording the same outcome twice."""
        field = {
            DropReason.NO_FREE_SLOT: "dropped_no_slot",
            DropReason.STALE: "dropped_stale",
            DropReason.SHUTDOWN: "dropped_shutdown",
            DropReason.PUBLICATION_TIMEOUT: "dropped_publication_timeout",
            DropReason.SUPERSEDED: "dropped_superseded",
        }[reason]
        with self._lock:
            self._counts[camera_id][field] += 1

    def record_latency(self, camera_id: str, milliseconds: float) -> None:
        """Append milliseconds to the bounded recent-history ring for this camera."""
        with self._lock:
            self._latencies[camera_id].append(milliseconds)

    def snapshot(self, camera_id: str) -> MetricsSnapshot:
        """Copy current counters under the local lock; never imply a cross-process snapshot."""
        with self._lock:
            return MetricsSnapshot(camera_id, **dict(self._counts[camera_id]))

    def summary(self) -> dict[str, CameraMetrics]:
        """Return serializable per-camera totals and recent p50/p95/p99 latency values.

        Final summaries travel over the status channel. Percentiles do not retain
        unbounded frame history and are calculated only at snapshot/report boundaries.
        """
        with self._lock:
            result: dict[str, CameraMetrics] = {}
            for camera_id, counts in self._counts.items():
                values = cast(CameraMetrics, asdict(MetricsSnapshot(camera_id, **dict(counts))))
                samples = list(self._latencies[camera_id])
                if samples:
                    values["latency_ms_recent"] = dict(
                        zip(
                            ("p50", "p95", "p99"),
                            [float(x) for x in np.percentile(samples, [50, 95, 99])],
                            strict=True,
                        )
                    )
                result[camera_id] = values
            return result


class ThroughputWindow:
    """Reporter-local non-overlapping windows based on actual successful emission counts."""

    def __init__(self, clock: Clock, duration_s: float = 10.0) -> None:
        """Start a monotonic window of at least ten seconds; reject shorter assessment windows."""
        if duration_s < 10:
            raise ValueError("Reporter throughput windows must be at least 10 seconds")
        self._clock = clock
        self._duration_ns = int(duration_s * 1e9)
        self._start = clock.monotonic_ns()
        self._counts: Counter[str] = Counter()
        self.elapsed_s = 0.0
        self.counts: dict[str, int] = {}

    def record(self, camera_id: str) -> None:
        """Count one result only after write and flush succeed."""
        self._counts[camera_id] += 1

    @property
    def due(self) -> bool:
        """Allow Reporter to complete pending writes before closing this window."""
        return self._clock.monotonic_ns() - self._start >= self._duration_ns

    def fps(self) -> dict[str, float] | None:
        """Return per-camera and total FPS after a full window, otherwise None.

        Divide by actual elapsed seconds, not nominal duration; then reset counters and
        start the next window. Startup/idle periods inside a window contribute elapsed time.
        """
        now = self._clock.monotonic_ns()
        elapsed = now - self._start
        if elapsed < self._duration_ns:
            return None
        seconds = elapsed / 1e9
        self.elapsed_s = seconds
        self.counts = dict(self._counts)
        rates = {camera: count / seconds for camera, count in self._counts.items()}
        rates["total"] = sum(self._counts.values()) / seconds
        self._counts.clear()
        self._start = now
        return rates
