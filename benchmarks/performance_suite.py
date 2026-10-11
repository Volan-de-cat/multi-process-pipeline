"""Repeatable performance acceptance matrix across batches and adverse workloads.

Each trial runs sequentially to avoid self-interference. Thresholds are absolute
acceptance limits, not a statistical regression claim on variable hosted runners.
Low-input and single-core cases intentionally have different minimum FPS targets.
"""

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from run_benchmark import run_trial

from cv_pipeline.config import CameraConfig, PipelineConfig


def cases() -> Iterator[tuple[str, PipelineConfig, str, float, int | None]]:
    """Yield full-load batches, sparse input, CPU restriction, disconnect and slow stages."""
    base = PipelineConfig(shutdown_timeout_s=20)
    for batch in (4, 8, 16):
        yield f"batch-{batch}", replace(base, batch_size=batch), "normal", 10.0, None
    yield (
        "low-input",
        replace(base, cameras=tuple(CameraConfig(f"cam_{i:02d}", 2) for i in range(1, 5))),
        "normal",
        5.0,
        None,
    )
    yield (
        "sparse-camera",
        replace(base, cameras=(CameraConfig("slow", 1),), health_timeout_s=0.7),
        "normal",
        0.5,
        None,
    )
    yield (
        "large-output",
        replace(
            base,
            batch_size=16,
            batch_wait_ms=100,
            health_timeout_s=1.0,
            output_size=(3840, 3840),
            preprocessing_threads=1,
            memory_budget_mib=2048,
            max_frame_age_ms=3500,
        ),
        "normal",
        5.0,
        None,
    )
    yield "four-threads", replace(base, preprocessing_threads=4), "normal", 10.0, None
    yield "overlap", replace(base, processing_overlap=True), "normal", 10.0, None
    yield "latest", replace(base, frame_policy="latest"), "normal", 10.0, None
    yield (
        "buffered-output",
        replace(base, output_flush_records=16, output_flush_interval_ms=100),
        "normal",
        10.0,
        None,
    )
    yield "single-core", replace(base, preprocessing_threads=1), "normal", 5.0, 1
    yield "disconnect", base, "disconnect", 10.0, None
    yield "slow-inference", base, "slow-inference", 8.0, None
    yield (
        "slow-output-large-batch",
        replace(
            base,
            batch_size=16,
            batch_wait_ms=100,
            health_timeout_s=1.0,
            result_queue_capacity=1,
            max_frame_age_ms=3500,
        ),
        "slow-output",
        8.0,
        None,
    )
    yield "slow-output", replace(base, result_queue_capacity=8), "slow-output", 8.0, None


def main() -> int:
    """Run the matrix, print a concise summary, and fail if an acceptance check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/performance-matrix"))
    args = parser.parse_args()
    if (
        not math.isfinite(args.duration)
        or not 12 <= args.duration <= 60
        or not 1 <= args.repeats <= 3
    ):
        parser.error("duration must be 12..60 seconds and repeats 1..3")
    args.directory.mkdir(parents=True, exist_ok=False)
    results = []
    for name, config, scenario, minimum, cpu_count in cases():
        for repeat in range(args.repeats):
            report = run_trial(
                args.directory / f"{name}-{repeat + 1}",
                args.duration,
                config,
                scenario=scenario,
                min_fps=minimum,
                cpu_count=cpu_count,
            )
            cameras = report["accounting"]["cameras"] if report["accounting"] else {}
            captured = sum(c["captured"] for c in cameras.values())
            drops = sum(
                sum(
                    c.get(k, 0)
                    for k in (
                        "dropped_no_slot",
                        "dropped_stale",
                        "dropped_shutdown",
                        "dropped_publication_timeout",
                        "dropped_superseded",
                    )
                )
                for c in cameras.values()
            )
            # Slow output tests bounded backpressure, not the ordinary latency target.
            latency_limit = 3000 if scenario == "slow-output" else 2000
            if name == "large-output":
                latency_limit = 5000
            if name == "slow-output-large-batch":
                # 3.5 s allowed input age plus 16 serial 80 ms writes; this is a
                # progress/ownership stress case with an explicit latency tradeoff.
                latency_limit = 5000
            passed = (
                report["passed"]
                and report["delivery_latency_recent_ms"].get("p95", float("inf")) < latency_limit
            )
            if scenario == "disconnect":
                passed = passed and report["disconnect_events"] > 0
            result = {
                "case": name,
                "repeat": repeat + 1,
                "passed": bool(passed),
                "fps": [w["fps"]["total"] for w in report["throughput_windows"]],
                "min_fps": minimum,
                "p95_ms": report["delivery_latency_recent_ms"].get("p95"),
                "serialization_p95_ms": report["latency_recent_ms"].get("p95"),
                "p95_limit_ms": latency_limit,
                "reported": report["reported_results"],
                "drop_percent": 100 * drops / captured if captured else None,
                "disconnect_events": report["disconnect_events"],
                "rss_peak_mib": report["rss_peak_mib"],
            }
            results.append(result)
            (args.directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
