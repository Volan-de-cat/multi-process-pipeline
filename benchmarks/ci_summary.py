"""Render independently validated benchmark sections as a resilient CI summary.

Missing, truncated or structurally invalid artifacts are diagnostic text, never a
reason to lose other sections. Rendering does not change benchmark acceptance.
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cv_pipeline.wire import finite_float, reject_constant, unique_fields


def load_report(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Reject malformed/nonfinite/duplicate-key JSON and unsupported root shapes."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=finite_float,
            parse_constant=reject_constant,
            object_pairs_hook=unique_fields,
        )
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and all(isinstance(row, dict) for row in value):
            return value
    except (OSError, UnicodeError, ValueError, RecursionError):
        pass
    return None


def number(value: Any) -> float:
    """Accept only finite nonnegative numeric measurements, excluding booleans."""
    if type(value) not in (int, float) or value < 0:
        raise ValueError("invalid measurement")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite measurement")
    return result


def label(value: Any) -> str:
    """Render a bounded string as plain Markdown table content."""
    if not isinstance(value, str):
        raise ValueError("invalid label")
    return value[:256].replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def rows(values: Any, render: Callable[[dict[str, Any]], str]) -> None:
    """Retain valid rows even when adjacent records are missing or damaged."""
    if not isinstance(values, list):
        print("Invalid report: expected a list of measurements.\n")
        return
    for index, row in enumerate(values, 1):
        try:
            if not isinstance(row, dict):
                raise ValueError("expected an object")
            line = render(row)
        except (KeyError, TypeError, ValueError, OverflowError):
            print(f"Invalid report row {index}; inspect raw artifacts.\n")
        else:
            print(line)


def matrix_row(row: dict[str, Any]) -> str:
    """Validate and format one acceptance case without relying on global totals."""
    if type(row["passed"]) is not bool or type(row["repeat"]) is not int:
        raise ValueError("invalid case status")
    if not isinstance(row["fps"], list):
        raise ValueError("invalid throughput windows")
    fps = "/".join(f"{number(v):.2f}" for v in row["fps"])
    p95 = "unavailable" if row["p95_ms"] is None else f"{number(row['p95_ms']):.2f}"
    drops = "unavailable" if row["drop_percent"] is None else f"{number(row['drop_percent']):.2f}"
    return f"| {label(row['case'])} | {row['repeat']} | {row['passed']} | {fps} | {p95} | {drops} |"


def pipeline(report: dict[str, Any]) -> None:
    """Render single-trial context and rows; malformed context stays local to this section."""
    exit_code = report["exit_code"]
    if exit_code is not None and type(exit_code) is not int:
        raise ValueError("invalid exit status")
    print(f"Exit code: {exit_code}; results: {number(report['reported_results']):.0f}.\n")
    print(f"CPU: {label(report['environment']['cpu'])}.\n")
    print("| Total FPS |\n| ---: |")
    rows(report["throughput_windows"], lambda row: f"| {number(row['fps']['total']):.2f} |")
    print("\n| Process | Peak RSS, MiB |\n| --- | ---: |")
    peaks = report["rss_peak_mib"]
    if not isinstance(peaks, dict):
        raise ValueError("invalid memory observations")
    rows(
        [{"stage": k, "peak": v} for k, v in peaks.items()],
        lambda row: f"| {label(row['stage'])} | {number(row['peak']):.1f} |",
    )
    print("\nRSS includes shared pages; do not sum these values.\n")
    print("| Camera | Recent p50 ms | Recent p95 ms | Recent p99 ms |")
    print("| --- | ---: | ---: | ---: |")
    summary = report.get("reporter_summary") or {}
    cameras = summary.get("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("invalid camera summary")
    rows([{"camera": k, "metrics": v} for k, v in cameras.items()], camera_row)
    print("\nLatency percentiles describe the bounded recent sample window.\n")


def camera_row(row: dict[str, Any]) -> str:
    """Validate a camera latency record independently of other cameras."""
    latency = row["metrics"]["latency_ms_recent"]
    values = [number(latency[key]) for key in ("p50", "p95", "p99")]
    return f"| {label(row['camera'])} | " + " | ".join(f"{v:.2f}" for v in values) + " |"


def preprocessing_row(row: dict[str, Any]) -> str:
    """Format a validated preprocessing quality/timing measurement."""
    return (
        f"| {label(row['source'])} | {label(row['method'])} | {number(row['threads']):.0f} | "
        f"{number(row['median_ms']):.3f} | {number(row['max_error']):.0f} |"
    )


def main() -> int:
    """Render every available section; explain malformed artifacts without traceback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    matrix = load_report(args.directory / "summary.json")
    if isinstance(matrix, list):
        print("## Performance acceptance matrix\n")
        print("| Case | Repeat | Passed | Window FPS | p95 flush ms | Drops % |")
        print("| --- | ---: | --- | --- | ---: | ---: |")
        rows(matrix, matrix_row)
    else:
        print("## Pipeline performance\n")
        if (args.directory / "summary.json").exists():
            print("Invalid matrix report; inspect raw artifacts.\n")
        report = load_report(args.directory / "report.json")
        if not isinstance(report, dict):
            print("No valid end-to-end report was produced. Inspect the job and service logs.\n")
        else:
            try:
                pipeline(report)
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                print("Invalid pipeline report; inspect raw artifacts.\n")
    preprocessing = load_report(args.directory / "preprocessing.json")
    print("## Preprocessing\n")
    if not isinstance(preprocessing, dict):
        print("No valid preprocessing report was produced.")
    else:
        print("| Input | Method | Threads | Median ms | Max pixel error |")
        print("| --- | --- | ---: | ---: | ---: |")
        rows(preprocessing.get("measurements"), preprocessing_row)
        if "component_measurements" in preprocessing:
            print("\n| Isolated component (one OpenCV thread) | Median ms |")
            print("| --- | ---: |")
            rows(
                preprocessing["component_measurements"],
                lambda row: f"| {label(row['component'])} | {number(row['median_ms']):.3f} |",
            )
            print(
                "\nHost copy excludes IPC; generation excludes FPS pacing. "
                "Independent timings do not predict overlapped throughput."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
