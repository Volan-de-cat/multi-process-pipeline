"""Isolated Linux benchmark with tree cleanup, failure reports and streaming metrics.

A fresh run directory prevents stale data reuse. PIDs and RSS are diagnostic;
final accounting comes from a separate create-only JSON file, never service logs.
Every path, workload and acceptance threshold is recorded in report.json.
"""

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cv_pipeline.config import PipelineConfig, config_json, load_config, validate_config
from cv_pipeline.lifecycle import reap_group
from cv_pipeline.validation import validate_report_result
from cv_pipeline.wire import finite_float, reject_constant, unique_fields


def rss_bytes(pid: int) -> int | None:
    """Return Linux resident bytes or None after process exit; do not sum shared RSS."""
    try:
        match = re.search(
            r"^VmRSS:\s+(\d+) kB", Path(f"/proc/{pid}/status").read_text(), re.MULTILINE
        )
        return int(match.group(1)) * 1024 if match else None
    except OSError:
        return None


def process_cpu_seconds(pid: int) -> float | None:
    """Read user+kernel CPU time without including already exited descendants.

    Linux stat's command name may contain spaces or parentheses; split after its
    final closing parenthesis. Fields 14/15 are indices 11/12 in the remaining
    fields starting at process state (field 3). Exit races are missing samples.
    """
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def read_json(text: str) -> Any:
    """Reject nonfinite numbers and duplicate keys in benchmark artifacts too."""
    return json.loads(
        text,
        parse_float=finite_float,
        parse_constant=reject_constant,
        object_pairs_hook=unique_fields,
    )


def nonnegative_number(value: Any) -> bool:
    """Exclude booleans and nonfinite values from recorded timings and rates."""
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def validate_throughput(row: dict[str, Any]) -> None:
    """Recompute rates from actual completed-window counts, never nominal duration."""
    elapsed, counts, rates = row.get("elapsed_s"), row.get("counts"), row.get("fps")
    if not isinstance(elapsed, (int, float)) or not nonnegative_number(elapsed) or elapsed < 10:
        raise ValueError("throughput window must span at least ten seconds")
    if not isinstance(counts, dict) or not isinstance(rates, dict):
        raise ValueError("missing throughput counts or rates")
    if "total" in counts or set(rates) != set(counts) | {"total"}:
        raise ValueError("invalid throughput camera keys")
    if not all(type(v) is int and 0 <= v <= 2**63 - 1 for v in counts.values()):
        raise ValueError("invalid throughput counts")
    expected = {camera: count / elapsed for camera, count in counts.items()}
    expected["total"] = sum(counts.values()) / elapsed
    if any(
        not nonnegative_number(rates[k])
        or not math.isclose(rates[k], v, rel_tol=1e-9, abs_tol=1e-12)
        for k, v in expected.items()
    ):
        raise ValueError("throughput rates disagree with counts")


def validate_window_latency(row: dict[str, Any], key: str) -> None:
    """Validate optional bounded histogram fields; old artifacts remain readable."""
    if key in row:
        value = row[key]
        if value is not None and (type(value) is not int or not 0 <= value <= 2000):
            raise ValueError("invalid delivery histogram percentile")


def validate_result_row(row: Any) -> None:
    """Check all fields consumed by aggregation before trusting an artifact."""
    if not isinstance(row, dict):
        raise ValueError("result row must be an object")
    kind = row.get("type")
    if kind == "result":
        validate_report_result(row)
    elif kind == "throughput":
        validate_throughput(row)
        validate_window_latency(row, "delivery_p95_upper_ms")
        if (
            not isinstance(row.get("fps"), dict)
            or not all(nonnegative_number(v) for v in row["fps"].values())
            or "total" not in row["fps"]
        ):
            raise ValueError("invalid throughput window")
    elif kind == "reporter_summary":
        if not isinstance(row.get("cameras"), dict):
            raise ValueError("invalid reporter camera summary")
        validate_percentiles(row.get("delivery_latency_ms_recent"))
        validate_window_latency(row, "worst_delivery_window_p95_upper_ms")
    else:
        raise ValueError("unexpected result row type")


def accounting_cameras(accounting: Any, config: PipelineConfig) -> dict[str, Any]:
    """Validate lifecycle status and configured camera membership before counters."""
    if (
        not isinstance(accounting, dict)
        or type(accounting.get("exit_code")) is not int
        or accounting["exit_code"] != 0
        or type(accounting.get("schema_version")) is not int
        or accounting["schema_version"] != 1
        or accounting.get("cleanup_completed") is not True
        or accounting.get("cleanup_errors") != []
    ):
        raise ValueError("invalid accounting summary")
    cameras = accounting.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {c.camera_id for c in config.cameras}:
        raise ValueError("missing accounting cameras")
    return cameras


def validate_accounting(
    accounting: Any,
    config: PipelineConfig,
    reported_count: int,
    per_camera: dict[str, int] | None = None,
) -> None:
    """Require complete per-camera counters and the same totals as the result stream."""
    cameras = accounting_cameras(accounting, config)
    drops = (
        "dropped_no_slot",
        "dropped_stale",
        "dropped_shutdown",
        "dropped_publication_timeout",
        "dropped_superseded",
    )
    for counters in cameras.values():
        required = ("captured", "processed", "reported", "in_flight", *drops)
        if not isinstance(counters, dict) or not all(
            type(counters.get(k)) is int and counters[k] >= 0 for k in required
        ):
            raise ValueError("invalid accounting counters")
        if counters["in_flight"] != 0 or counters["processed"] != counters["reported"]:
            raise ValueError("unfinished accounting")
        if counters["captured"] != counters["reported"] + sum(counters[k] for k in drops):
            raise ValueError("inconsistent accounting")
    if per_camera is not None and any(
        counters["reported"] != per_camera.get(camera, 0) for camera, counters in cameras.items()
    ):
        raise ValueError("accounting differs from per-camera emitted results")
    if sum(c["reported"] for c in cameras.values()) != reported_count:
        raise ValueError("accounting differs from emitted results")


class ResultIdentities:
    """Bounded per-camera sequence verification; gaps are allowed, repeats are rejected."""

    def __init__(self, config: PipelineConfig) -> None:
        self.last: dict[str, tuple[int, int]] = {}
        self.counts: Counter[str] = Counter()
        self.cameras = {camera.camera_id for camera in config.cameras}
        self.window_counts: Counter[str] = Counter()

    def accept_window(self, row: dict[str, Any]) -> None:
        """Require each window to match result rows since the previous window."""
        counts = row["counts"]
        if not set(counts) <= self.cameras or counts != dict(self.window_counts):
            raise ValueError("throughput counts differ from emitted results")
        self.window_counts.clear()

    def accept(self, row: dict[str, Any]) -> None:
        camera, frame, epoch = (
            row.get("camera_id"),
            row.get("frame_id"),
            row.get("connection_epoch"),
        )
        if not isinstance(camera, str) or camera not in self.cameras:
            raise ValueError("unknown result camera")
        if type(frame) is not int or frame < 1 or type(epoch) is not int or epoch < 1:
            raise ValueError("invalid frame identity")
        previous = self.last.get(camera)
        if previous is not None and (frame <= previous[0] or epoch < previous[1]):
            raise ValueError("duplicate, reordered or regressed result identity")
        self.last[camera] = (frame, epoch)
        self.counts[camera] += 1
        self.window_counts[camera] += 1


def monitor_trial(
    directory: Path, command: list[str], duration: float, config: PipelineConfig, started: float
) -> tuple[
    dict[str, list[dict[str, float | int]]],
    dict[str, int],
    subprocess.Popen[bytes] | None,
    str | None,
]:
    """Launch, sample bounded RSS history and always reap the isolated process group."""
    samples: dict[str, list[dict[str, float | int]]] = {}
    peaks: dict[str, int] = {}
    pids = {}
    cpu_previous: dict[str, tuple[float, float]] = {}
    workload_started: float | None = None
    failure = None
    process = None
    allowance_s = 3 * config.startup_timeout_s + config.shutdown_timeout_s + 10
    with (directory / "service.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=log,
                start_new_session=True,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
            pids["supervisor"] = process.pid
            with (directory / "service.log").open(encoding="utf-8") as reader:
                while process.poll() is None:
                    # Incremental parsing avoids re-reading an ever-growing file.
                    for stage, pid in re.findall(
                        r"stage_ready stage=(\w+) pid=(\d+)", reader.read()
                    ):
                        pids[stage] = int(pid)
                    now = time.monotonic()
                    if workload_started is None and set(pids) == {
                        "supervisor",
                        "capture",
                        "processing",
                        "reporter",
                    }:
                        workload_started = now
                    elapsed = now - started
                    for stage, pid in pids.items():
                        rss = rss_bytes(pid)
                        if rss is not None:
                            peaks[stage] = max(peaks.get(stage, 0), rss)
                            values = samples.setdefault(stage, [])
                            if len(values) >= 1200:
                                del values[0]
                            sample: dict[str, float | int] = {
                                "elapsed_s": elapsed,
                                "rss_bytes": rss,
                            }
                            if workload_started is not None:
                                sample["workload_elapsed_s"] = now - workload_started
                            cpu = process_cpu_seconds(pid)
                            if cpu is not None:
                                sample["cpu_seconds"] = cpu
                                previous = cpu_previous.get(stage)
                                if previous is not None and elapsed > previous[0]:
                                    # 100% means one full CPU core; multithreaded
                                    # stage utilization may legitimately exceed it.
                                    sample["cpu_percent"] = max(
                                        0.0, 100 * (cpu - previous[1]) / (elapsed - previous[0])
                                    )
                                cpu_previous[stage] = (elapsed, cpu)
                            values.append(sample)
                    if elapsed > duration + allowance_s:
                        raise TimeoutError("benchmark startup/drain allowance exceeded")
                    time.sleep(0.25)
        except (OSError, TimeoutError) as exc:
            failure = str(exc)
        finally:
            if process is not None:
                try:
                    reap_group(process)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    failure = failure or f"process cleanup failed: {exc}"
    return samples, peaks, process, failure


@dataclass(slots=True)
class ArtifactData:
    """Named validated aggregates; JSON dictionaries stay at the artifact boundary."""

    windows: list[dict[str, Any]]
    count: int
    histogram: dict[str, int]
    latency: deque[float]
    prep: deque[float]
    reporter_summary: dict[str, Any] | None
    accounting: dict[str, Any] | None
    failure: str | None


def validate_reporter_summary(
    row: dict[str, Any],
    counts: Counter[str],
    histogram: dict[str, int],
    latency: deque[float],
    completed_window_s: float = 0.0,
) -> None:
    """Reconcile the single terminal record against all emitted result identities."""
    cameras = row["cameras"]
    if set(cameras) != set(counts):
        raise ValueError("reporter camera membership differs from emitted results")
    batches = row.get("result_counts_by_batch_size")
    if (
        not isinstance(batches, dict)
        or not all(type(v) is int and v > 0 for v in batches.values())
        or batches != histogram
    ):
        raise ValueError("reporter batch histogram differs from emitted results")
    if not nonnegative_number(row.get("elapsed_s")):
        raise ValueError("invalid reporter duration")
    # Allow one millisecond for legacy artifacts whose first window started
    # before the lifetime clock. Current reporters start the lifetime first.
    if row["elapsed_s"] + 0.001 < completed_window_s:
        raise ValueError("reporter duration is shorter than completed throughput windows")
    if latency:
        recent = list(latency)[-2048:]
        delivery = row["delivery_latency_ms_recent"]
        if any(delivery[f"p{p}"] + 1e-6 < float(np.percentile(recent, p)) for p in (50, 95, 99)):
            raise ValueError("delivery percentiles precede serialization latencies")
    for camera, count in counts.items():
        metrics = cameras[camera]
        if not isinstance(metrics, dict) or metrics.get("camera_id") != camera:
            raise ValueError("invalid reporter camera identity")
        if type(metrics.get("reported")) is not int or metrics["reported"] != count:
            raise ValueError("reporter counters differ from emitted results")
        validate_percentiles(metrics.get("latency_ms_recent"))


def validate_percentiles(value: Any) -> None:
    """Require finite nonnegative ordered p50/p95/p99 values."""
    if not isinstance(value, dict) or set(value) != {"p50", "p95", "p99"}:
        raise ValueError("missing latency percentiles")
    if not all(nonnegative_number(v) for v in value.values()) or not (
        value["p50"] <= value["p95"] <= value["p99"]
    ):
        raise ValueError("invalid latency percentiles")


def validate_delivery_windows(summary: dict[str, Any], windows: list[dict[str, Any]]) -> None:
    """Final worst-window evidence must include every completed delivery window."""
    bounds = [
        w["delivery_p95_upper_ms"] for w in windows if w.get("delivery_p95_upper_ms") is not None
    ]
    if bounds:
        worst = summary.get("worst_delivery_window_p95_upper_ms")
        if worst is None or worst < max(bounds):
            raise ValueError("final delivery percentile omits a completed window")


def result_lines(directory: Path, max_files: int) -> Iterator[str]:
    """Stream secure-writer segments in order without loading the full output."""
    for segment in range(max_files):
        suffix = f".{segment:04d}" if segment else ""
        path = directory / f"results.jsonl{suffix}"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as source:
            yield from source


def read_artifacts(directory: Path, config: PipelineConfig, failure: str | None) -> ArtifactData:
    """Read bounded samples and reconcile stream identities, windows and final accounting."""
    windows: list[dict[str, Any]] = []
    count = 0
    histogram: dict[str, int] = {}
    latency: deque[float] = deque(maxlen=8192)
    prep: deque[float] = deque(maxlen=8192)
    reporter_summary = None
    identities = ResultIdentities(config)
    accounting = None
    try:
        for line in result_lines(directory, config.output_max_files):
            try:
                row = read_json(line)
                validate_result_row(row)
                if reporter_summary is not None:
                    raise ValueError("records after terminal reporter summary")
                if row["type"] == "result":
                    identities.accept(row)
                elif row["type"] == "throughput":
                    identities.accept_window(row)
            except (ValueError, KeyError, TypeError, RecursionError, OverflowError):
                failure = failure or "incomplete or invalid result JSONL"
                break
            if row["type"] == "result":
                count += 1
                latency.append(row["capture_to_report_ms"])
                prep.append(row["preprocessing_ms"])
                size = str(row["batch_size"])
                histogram[size] = histogram.get(size, 0) + 1
            elif row["type"] == "throughput":
                windows.append(row)
            elif row["type"] == "reporter_summary":
                validate_delivery_windows(row, windows)
                validate_reporter_summary(
                    row,
                    identities.counts,
                    histogram,
                    latency,
                    sum(window["elapsed_s"] for window in windows),
                )
                reporter_summary = row
        accounting_path = directory / "accounting.json"
        accounting = read_json(accounting_path.read_text()) if accounting_path.exists() else None
        if accounting is not None:
            validate_accounting(accounting, config, count, identities.counts)
        if accounting is None or reporter_summary is None:
            failure = failure or "missing final accounting or reporter summary"
    except (OSError, ValueError, KeyError, TypeError, RecursionError, OverflowError) as exc:
        failure = failure or f"invalid benchmark artifact: {exc}"
        accounting = None
    return ArtifactData(
        windows, count, histogram, latency, prep, reporter_summary, accounting, failure
    )


def run_trial(
    directory: Path,
    duration: float,
    config: PipelineConfig,
    *,
    scenario: str = "normal",
    min_fps: float = 10.0,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    """Run one fresh isolated trial and write a report even on pipeline failure.

    RSS samples are downsampled to a bounded ring. JSONL parsing retains only
    the most recent 8192 latency/preprocessing samples. No shell processes are used.
    """
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = replace(config, output_path=str(directory / "results.jsonl"))
    validate_config(config)
    (directory / "config.json").write_text(config_json(config) + "\n")
    command = [
        sys.executable,
        str(Path(__file__).with_name("workload.py")),
        "--config",
        str(directory / "config.json"),
        "--duration",
        str(duration),
        "--summary",
        str(directory / "accounting.json"),
        "--scenario",
        scenario,
    ]
    if cpu_count is not None:
        cpus = sorted(os.sched_getaffinity(0))[:cpu_count]
        command = ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command]
    else:
        cpus = sorted(os.sched_getaffinity(0))
    started = time.monotonic()
    allowance_s = 3 * config.startup_timeout_s + config.shutdown_timeout_s + 10
    samples, peaks, process, failure = monitor_trial(directory, command, duration, config, started)
    artifacts = read_artifacts(directory, config, failure)
    windows, count, histogram = artifacts.windows, artifacts.count, artifacts.histogram
    latency, prep = artifacts.latency, artifacts.prep
    reporter_summary, accounting, failure = (
        artifacts.reporter_summary,
        artifacts.accounting,
        artifacts.failure,
    )
    passed = (
        failure is None
        and process is not None
        and process.returncode == 0
        and accounting is not None
        and reporter_summary is not None
        and bool(windows)
        and all(w["fps"]["total"] > min_fps for w in windows)
    )
    cpu = re.search(r"model name\s+: (.+)", Path("/proc/cpuinfo").read_text())
    report = {
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
            "cpu": cpu.group(1) if cpu else "unknown",
            "cpu_count": os.cpu_count(),
            "affinity": cpus,
        },
        "config": json.loads(config_json(config)),
        "scenario": scenario,
        "source_policy": {
            "random_disconnects": False,
            "disconnect_attempt": 30 if scenario == "disconnect" else None,
            "disconnect_camera": "cam_01" if scenario == "disconnect" else None,
            "retry_jitter": scenario != "disconnect",
        },
        "duration_requested_s": duration,
        "startup_and_drain_allowance_s": allowance_s,
        "elapsed_wall_s": time.monotonic() - started,
        "exit_code": process.returncode if process else None,
        "failure": failure,
        "passed": passed,
        "min_fps": min_fps,
        "reported_results": count,
        "throughput_windows": windows,
        "reporter_summary": reporter_summary,
        "delivery_latency_recent_ms": (reporter_summary or {}).get(
            "delivery_latency_ms_recent", {}
        ),
        "accounting": accounting,
        "batch_result_histogram": histogram,
        "latency_recent_ms": {f"p{p}": float(np.percentile(latency, p)) for p in (50, 95, 99)}
        if latency
        else {},
        "preprocessing_recent_mean_ms": float(np.mean(prep)) if prep else None,
        "memory_note": "RSS includes shared mappings; do not sum process RSS as physical RAM.",
        "rss_samples": samples,
        "rss_peak_mib": {s: n / 2**20 for s, n in peaks.items()},
        "disconnect_events": (directory / "service.log").read_text().count("camera_disconnected"),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    """Validate bounded command-line inputs and execute a single normal benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--directory", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 12 <= args.duration <= 300:
        parser.error("duration must be between 12 and 300 seconds")
    report = run_trial(args.directory, args.duration, load_config(args.config))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "exit_code",
                    "passed",
                    "reported_results",
                    "latency_recent_ms",
                    "rss_peak_mib",
                )
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
