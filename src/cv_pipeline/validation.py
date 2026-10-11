"""Semantic wire contracts after structural type validation.

Only record values are checked here; cross-record ownership, configured camera
membership and accounting reconciliation belong to their consuming components.
Validators also run on encoding, so programming errors fail before publication.
"""

import math
import re
from datetime import datetime, timezone
from typing import Any

from .messages import (
    FrameDescriptor,
    FrameMetadata,
    HealthEvent,
    InferenceResult,
    ProcessedResult,
    ProcessingTimings,
    ReportedResult,
    StageSummary,
)


def validate_report_result(row: dict[str, Any]) -> None:
    """Validate external results before accepting benchmark evidence.

    Require every declared field, allowing additional future fields. UTC labels
    match receipt nanoseconds; timing comparisons allow one nanosecond of rounding.
    Processing includes preprocessing/inference, while total latency includes waiting.
    """
    if not ReportedResult.__required_keys__ <= row.keys() or row["type"] != "result":
        raise ValueError("missing Reporter result fields")
    if (
        not isinstance(row["camera_id"], str)
        or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", row["camera_id"]) is None
    ):
        raise ValueError("invalid Reporter camera identity")
    for key, maximum in (
        ("frame_id", 2**63 - 1),
        ("connection_epoch", 2**63 - 1),
        ("received_utc_ns", 253402300799000000000 - 1),
        ("batch_size", 16),
    ):
        if type(row[key]) is not int or not 1 <= row[key] <= maximum:
            raise ValueError(f"invalid Reporter {key}")
    expected_timestamp = datetime.fromtimestamp(
        row["received_utc_ns"] / 1e9, timezone.utc
    ).isoformat()
    if row["timestamp"] != expected_timestamp:
        raise ValueError("Reporter timestamp differs from receipt time")
    if type(row["detection"]) is not bool:
        raise ValueError("invalid Reporter detection")
    numeric_fields = (
        "mean_brightness",
        "processing_latency_ms",
        "queue_and_batch_wait_ms",
        "preprocessing_ms",
        "inference_ms",
        "capture_to_report_ms",
    )
    try:
        valid_numbers = all(
            type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
            for key in numeric_fields
        )
    except OverflowError as exc:
        raise ValueError("Reporter numeric field exceeds supported range") from exc
    if not valid_numbers or row["mean_brightness"] > 255:
        raise ValueError("invalid Reporter numeric fields")
    processing = row["processing_latency_ms"]
    if (
        processing + 1e-6 < row["preprocessing_ms"] + row["inference_ms"]
        or row["capture_to_report_ms"] + 1e-6 < processing + row["queue_and_batch_wait_ms"]
    ):
        raise ValueError("inconsistent Reporter latency intervals")


def validate_record(value: object) -> None:
    """Reject impossible ranges and temporal relationships in known records."""
    if isinstance(value, FrameMetadata):
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value.camera_id) is None
            or min(
                value.frame_id,
                value.connection_epoch,
                value.received_utc_ns,
                value.received_monotonic_ns,
            )
            <= 0
            # datetime ISO output supports years 1..9999; avoid float rounding at the limit.
            or value.received_utc_ns >= 253402300799000000000
        ):
            raise ValueError("invalid frame identity")
    elif isinstance(value, FrameDescriptor):
        if value.protocol_version != 1 or value.slot_id < 0:
            raise ValueError("invalid frame descriptor")
    elif isinstance(value, InferenceResult):
        if not 0 <= value.mean_brightness <= 255:
            raise ValueError("invalid inference brightness")
    elif isinstance(value, ProcessingTimings):
        if not (
            0
            < value.started_monotonic_ns
            <= value.preprocessing_finished_monotonic_ns
            <= value.inference_started_monotonic_ns
            <= value.finished_monotonic_ns
        ):
            raise ValueError("invalid processing boundaries")
    elif isinstance(value, ProcessedResult):
        if not (
            1 <= value.batch_size <= 16
            and value.metadata.received_monotonic_ns <= value.timings.started_monotonic_ns
        ):
            raise ValueError("invalid processed result")
    elif isinstance(value, HealthEvent):
        if not 0 < value.last_progress_monotonic_ns <= value.monotonic_ns:
            raise ValueError("invalid health boundaries")
    elif isinstance(value, StageSummary):
        validate_summary(value)


def validate_summary(value: StageSummary) -> None:
    """Check identities, nonnegative counters and ordered latency percentiles."""
    if len(value.cameras) > 32 or any(v < 0 for v in value.diagnostics.values()):
        raise ValueError("invalid stage diagnostics")
    for camera, metrics in value.cameras.items():
        if camera != metrics["camera_id"] or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", camera) is None:
            raise ValueError("summary camera identity mismatch")
        if (
            min(
                metrics["captured"],
                metrics["processed"],
                metrics["reported"],
                metrics["dropped_no_slot"],
                metrics["dropped_stale"],
                metrics["dropped_shutdown"],
                metrics["dropped_publication_timeout"],
                metrics["dropped_superseded"],
            )
            < 0
        ):
            raise ValueError("negative stage counter")
        latency = metrics.get("latency_ms_recent")
        if latency is not None and (
            set(latency) != {"p50", "p95", "p99"}
            or not 0 <= latency["p50"] <= latency["p95"] <= latency["p99"]
        ):
            raise ValueError("invalid stage percentiles")
