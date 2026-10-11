"""Sequential sustained-load checks with CPU, memory drift and cleanup evidence.

Runs serial and overlapped processing, each with stable cameras and deterministic
camera reconnects. CPU utilization is per process (100% = one core); RSS includes
shared mappings and must not be summed. Median drift excludes startup and compares
the first steady minute with the final minute. It is a leak indicator, not proof
of leak freedom or a replacement for deployment cgroup limits.
"""

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from run_benchmark import run_trial

from cv_pipeline.config import PipelineConfig


def sampling_adequate(times: list[float], cpu_count: int, duration: int) -> bool:
    """Reject sparse/discontinuous observations and require 90% CPU coverage."""
    # At least one observation every two seconds, including window boundaries.
    return bool(
        len(times) >= (duration - 30) / 2
        and times[0] <= 32
        and times[-1] >= duration - 2
        and all(0 < b - a <= 2 for a, b in zip(times, times[1:], strict=False))
        and cpu_count >= 0.9 * len(times)
    )


def stage_summary(
    samples: list[dict[str, float | int]], peak: float | None, duration: int
) -> dict[str, float | None]:
    """Compare steady-state RSS medians and CPU utilization for one process."""
    samples = [{**s, "elapsed_s": s.get("workload_elapsed_s", s["elapsed_s"])} for s in samples]
    early = [s["rss_bytes"] / 2**20 for s in samples if 30 <= s["elapsed_s"] < 90]
    late = [s["rss_bytes"] / 2**20 for s in samples if duration - 60 <= s["elapsed_s"] <= duration]
    cpu = [
        s["cpu_percent"] for s in samples if 30 <= s["elapsed_s"] <= duration and "cpu_percent" in s
    ]
    times = [float(s["elapsed_s"]) for s in samples if 30 <= s["elapsed_s"] <= duration]
    adequate = sampling_adequate(times, len(cpu), duration)
    return {
        "sampling_adequate": adequate,
        "steady_samples": len(times),
        "early_median_rss_mib": statistics.median(early) if early else None,
        "late_median_rss_mib": statistics.median(late) if late else None,
        "rss_drift_mib": statistics.median(late) - statistics.median(early)
        if early and late
        else None,
        "mean_cpu_percent": statistics.mean(cpu) if cpu else None,
        "peak_rss_mib": peak,
    }


def memory_adequate(stages: dict[str, dict[str, float | None]]) -> bool:
    """Require every process, representative sampling and bounded median growth."""
    return set(stages) == {"supervisor", "capture", "processing", "reporter"} and all(
        s["rss_drift_mib"] is not None
        and s["rss_drift_mib"] <= 64
        and s["mean_cpu_percent"] is not None
        and s["sampling_adequate"]
        for s in stages.values()
    )


def acceptance_failures(
    report: dict[str, Any],
    stages: dict[str, dict[str, float | None]],
    p95: float | None,
    worst_p95: float | None,
) -> list[str]:
    """Explain every violated acceptance condition without discarding base failures."""
    failures = []
    if not report["passed"]:
        failures.append(report["failure"] or "base benchmark acceptance failed")
    if not memory_adequate(stages):
        failures.append(
            "missing stages, insufficient sample coverage, CPU data or RSS growth >64 MiB"
        )
    if p95 is None or p95 >= 2000:
        failures.append("recent delivery p95 must be below 2000 ms")
    if worst_p95 is None or worst_p95 >= 2000:
        failures.append(
            "every delivery window p95 must be below 2000 ms; missing metrics fail closed"
        )
    if report["scenario"] == "disconnect" and report["disconnect_events"] <= 0:
        failures.append("disconnect workload did not exercise reconnection")
    return failures


def summarize(report: dict[str, Any], duration: int) -> dict[str, Any]:
    """Apply sustained acceptance limits and retain interpretable per-stage metrics."""
    stages = {
        stage: stage_summary(samples, report["rss_peak_mib"].get(stage), duration)
        for stage, samples in report["rss_samples"].items()
    }
    accounting = report["accounting"] or {}
    cameras = accounting.get("cameras", {})
    captured = sum(c["captured"] for c in cameras.values())
    reported = report["reported_results"]
    p95 = report["delivery_latency_recent_ms"].get("p95")
    windows = report["throughput_windows"]
    worst_p95 = (report.get("reporter_summary") or {}).get("worst_delivery_window_p95_upper_ms")
    failures = acceptance_failures(report, stages, p95, worst_p95)
    return {
        "passed": not failures,
        "failure": "; ".join(failures) or None,
        "failures": failures,
        "duration_s": duration,
        "scenario": report["scenario"],
        "overlap": report["config"]["processing_overlap"],
        "completed_windows": len(windows),
        "min_fps": min((w["fps"]["total"] for w in windows), default=None),
        "mean_window_fps": statistics.mean(w["fps"]["total"] for w in windows) if windows else None,
        "delivery_p95_recent_ms": p95,
        "worst_delivery_window_p95_upper_ms": worst_p95,
        "reported": reported,
        "drop_percent": 100 * (captured - reported) / captured if captured else None,
        "disconnect_events": report["disconnect_events"],
        "cleanup_completed": accounting.get("cleanup_completed"),
        "memory": accounting.get("memory"),
        "stages": stages,
        "limits": {
            "min_fps_exclusive": 10,
            "recent_p95_ms_exclusive": 2000,
            "max_median_rss_growth_mib": 64,
        },
    }


def main() -> int:
    """Run fresh sequential 3–5 minute trials; preserve each full benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/soak"))
    args = parser.parse_args()
    if not 180 <= args.duration <= 300:
        parser.error("duration must be 180..300 seconds per trial")
    args.directory.mkdir(parents=True, exist_ok=False)
    results = []
    base = PipelineConfig(shutdown_timeout_s=20)
    for name, config, scenario in (
        ("default", base, "normal"),
        ("overlap", replace(base, processing_overlap=True), "normal"),
        ("reconnect", base, "disconnect"),
        ("overlap-reconnect", replace(base, processing_overlap=True), "disconnect"),
    ):
        report = run_trial(args.directory / name, args.duration, config, scenario=scenario)
        result = {"case": name, **summarize(report, args.duration)}
        results.append(result)
        (args.directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
