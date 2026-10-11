"""Final consumer of small results; pixels never reach this module.

Writes JSON Lines to stdout or a fresh private buffered file, measures complete throughput
windows and flushes on EOS. Receipt-to-report is sampled before serialization,
and separate delivery metrics include write/flush completion. Failed output is a pipeline failure, while
service-log output is explicitly best-effort and independent of this stream.
"""

import json
import math
from collections import deque
from datetime import datetime, timezone
from queue import Empty

import numpy as np

from .adapters import SystemClock
from .config import PipelineConfig
from .messages import EndOfStream, ProcessedResult, ReportedResult, Stage
from .observability import MetricsRecorder, ThroughputWindow
from .output import ResultOutput
from .runtime import worker_scope
from .transport import ReporterResources, WorkerResources


def serialize_result(result: ProcessedResult, reported_monotonic_ns: int) -> str:
    """Build one JSON result with identity, UTC receipt label and millisecond intervals.

    reported_monotonic_ns is taken before serialization/write, so capture_to_report_ms
    is not physical-output completion latency. Reject nonfinite JSON numbers and
    retain received_utc_ns separately from the human-readable timestamp.
    """
    meta = result.metadata
    timing = result.timings
    row: ReportedResult = {
        "type": "result",
        "camera_id": meta.camera_id,
        "frame_id": meta.frame_id,
        "connection_epoch": meta.connection_epoch,
        "timestamp": datetime.fromtimestamp(meta.received_utc_ns / 1e9, timezone.utc).isoformat(),
        "received_utc_ns": meta.received_utc_ns,
        "mean_brightness": result.inference.mean_brightness,
        "detection": result.inference.detection,
        "batch_size": result.batch_size,
        "processing_latency_ms": (timing.finished_monotonic_ns - timing.started_monotonic_ns) / 1e6,
        "queue_and_batch_wait_ms": (timing.started_monotonic_ns - meta.received_monotonic_ns) / 1e6,
        "preprocessing_ms": (
            timing.preprocessing_finished_monotonic_ns - timing.started_monotonic_ns
        )
        / 1e6,
        "inference_ms": (timing.finished_monotonic_ns - timing.inference_started_monotonic_ns)
        / 1e6,
        "capture_to_report_ms": (reported_monotonic_ns - meta.received_monotonic_ns) / 1e6,
    }
    return json.dumps(row, allow_nan=False)


class EmissionBatch:
    """Account deliveries only after successful flush, including the actual output time.

    Buffered records are bounded by configuration. Flush occurs at the count limit,
    interval deadline, throughput boundary or EOS; durability remains ResultOutput's
    responsibility. Receipt-to-serialization stays in each original result record.
    """

    def __init__(
        self,
        output: ResultOutput,
        config: PipelineConfig,
        clock: SystemClock,
        metrics: MetricsRecorder,
        window: ThroughputWindow,
    ) -> None:
        self.output = output
        self.config = config
        self.clock = clock
        self.metrics = metrics
        self.window = window
        self.pending: list[ProcessedResult] = []
        self.since_ns = clock.monotonic_ns()
        self.delivery_ms: deque[float] = deque(maxlen=2048)
        self.batch_counts: dict[int, int] = {}
        # Millisecond upper bounds, with >=2000 ms in the overflow bucket.
        # This fixed histogram covers every delivered result in each window.
        self._window_latency = [0] * 2001
        self.worst_window_p95_ms: int | None = None

    def write(self, result: ProcessedResult) -> None:
        self.output.write(serialize_result(result, self.clock.monotonic_ns()) + "\n")
        if not self.pending:
            self.since_ns = self.clock.monotonic_ns()
        self.pending.append(result)

    def receive_timeout(self, poll_s: float = 0.1) -> float:
        """Never sleep past a pending flush deadline; empty input uses normal polling."""
        if not self.pending:
            return poll_s
        remaining_ns = (
            self.since_ns
            + int(self.config.output_flush_interval_ms * 1e6)
            - self.clock.monotonic_ns()
        )
        return min(poll_s, max(0.0, remaining_ns / 1e9))

    def flush(self, force: bool = False) -> bool:
        expired = (
            self.clock.monotonic_ns() - self.since_ns >= self.config.output_flush_interval_ms * 1e6
        )
        if not self.pending or not (
            force or expired or len(self.pending) >= self.config.output_flush_records
        ):
            return False
        self.output.flush()
        completed = self.clock.monotonic_ns()
        for result in self.pending:
            camera_id = result.metadata.camera_id
            latency = (completed - result.metadata.received_monotonic_ns) / 1e6
            self.metrics.record_frame(camera_id, Stage.REPORTER)
            self.metrics.record_latency(camera_id, latency)
            self.delivery_ms.append(latency)
            self._window_latency[min(2000, max(0, math.ceil(latency)))] += 1
            self.batch_counts[result.batch_size] = self.batch_counts.get(result.batch_size, 0) + 1
            self.window.record(camera_id)
        self.pending.clear()
        return True

    def complete_latency_window(self) -> int | None:
        """Close a bounded histogram window; include the final partial window at EOS."""
        count = sum(self._window_latency)
        if not count:
            return None
        threshold = math.ceil(count * 0.95)
        cumulative = 0
        percentile = 2000
        for bucket, frequency in enumerate(self._window_latency):
            cumulative += frequency
            if cumulative >= threshold:
                percentile = bucket
                break
        self.worst_window_p95_ms = max(self.worst_window_p95_ms or 0, percentile)
        self._window_latency = [0] * 2001
        return percentile

    def delivery_summary(self) -> dict[str, float]:
        return (
            {f"p{p}": float(np.percentile(self.delivery_ms, p)) for p in (50, 95, 99)}
            if self.delivery_ms
            else {}
        )


def run_reporter(config: PipelineConfig, resources: ReporterResources | WorkerResources) -> None:
    """Spawn-safe final consumer: emit ordered JSON Lines until Processing EOS.

    Only completed writes/flushes increment throughput. Waiting for input is valid
    idle; active output has a fresh health budget and blocked output is detectable.
    Close only owned files, never sys.stdout; emit final bounded latency summaries.
    """
    with worker_scope(Stage.REPORTER, resources) as runtime:
        output = ResultOutput(
            config.output_path,
            config.output_max_bytes,
            config.output_durable,
            config.output_max_files,
        )
        clock = SystemClock()
        metrics = MetricsRecorder()
        # Lifetime includes every throughput window, even if setup is preempted.
        started = clock.monotonic_ns()
        window = ThroughputWindow(clock)
        emission = EmissionBatch(output, config, clock, metrics, window)
        try:
            runtime.ready()
            while True:
                runtime.tick(idle=True)
                try:
                    message = resources.results.get(timeout=emission.receive_timeout())
                except Empty:
                    message = None
                if isinstance(message, EndOfStream):
                    if message.stage != Stage.PROCESSING:
                        raise ValueError("unexpected result EOS producer")
                    break
                if message is not None:
                    if not isinstance(message, ProcessedResult):
                        raise ValueError("unexpected result message")
                    runtime.tick(force=True)
                    emission.write(message)
                if emission.pending:
                    runtime.tick()
                    if emission.flush():
                        runtime.tick(progress=True)
                if window.due:
                    # Include every preceding JSONL result only after successful flush.
                    emission.flush(force=True)
                rates = window.fps()
                if rates is not None:
                    emission.flush(force=True)
                    output.write(
                        json.dumps(
                            {
                                "type": "throughput",
                                "window_min_s": 10,
                                "delivery_p95_upper_ms": emission.complete_latency_window(),
                                "fps": rates,
                                "elapsed_s": window.elapsed_s,
                                "counts": window.counts,
                            }
                        )
                        + "\n"
                    )
                    output.flush()
            emission.flush(force=True)
            emission.complete_latency_window()
            summary = metrics.summary()
            output.write(
                json.dumps(
                    {
                        "type": "reporter_summary",
                        "elapsed_s": (clock.monotonic_ns() - started) / 1e9,
                        "cameras": summary,
                        "result_counts_by_batch_size": emission.batch_counts,
                        "delivery_latency_ms_recent": emission.delivery_summary(),
                        "worst_delivery_window_p95_upper_ms": emission.worst_window_p95_ms,
                    }
                )
                + "\n"
            )
            output.flush()
            runtime.summary(metrics)
        finally:
            output.close()
