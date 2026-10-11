"""Failure-path regression checks for benchmark artifacts and the CI summary renderer."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cv_pipeline.config import PipelineConfig

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def valid_result(**changes):
    row = {
        "type": "result",
        "camera_id": "cam",
        "frame_id": 1,
        "connection_epoch": 1,
        "timestamp": "1970-01-01T00:00:01+00:00",
        "received_utc_ns": 1_000_000_000,
        "mean_brightness": 100,
        "detection": False,
        "batch_size": 1,
        "processing_latency_ms": 1,
        "queue_and_batch_wait_ms": 0,
        "preprocessing_ms": 0.5,
        "inference_ms": 0.5,
        "capture_to_report_ms": 1,
    }
    return {**row, **changes}


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRegressionTests(unittest.TestCase):
    def test_cpu_samples_handle_process_names_and_exit_races(self):
        benchmark = load_module("run_benchmark")
        stat = "123 (camera (owner)) " + " ".join(["R", *(["0"] * 10), "300", "200"])
        with (
            patch.object(benchmark.Path, "read_text", return_value=stat),
            patch.object(benchmark.os, "sysconf", return_value=100),
        ):
            self.assertEqual(benchmark.process_cpu_seconds(123), 5)
        with patch.object(benchmark.Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(benchmark.process_cpu_seconds(123))

    def test_soak_memory_gate_rejects_growth_and_missing_measurements(self):
        import sys

        with patch.object(sys, "path", [str(BENCHMARKS), *sys.path]):
            soak = load_module("soak_test")
        sample = [
            {"elapsed_s": t, "rss_bytes": (100 if t < 150 else 110) * 2**20, "cpu_percent": 100}
            for t in range(30, 301)
        ]
        sample.append({"elapsed_s": 310, "rss_bytes": 0, "cpu_percent": 0})
        report = {
            "rss_samples": {s: sample for s in ("supervisor", "capture", "processing", "reporter")},
            "rss_peak_mib": {},
            "accounting": {"cameras": {}},
            "reported_results": 1,
            "delivery_latency_recent_ms": {"p95": 20},
            "reporter_summary": {"worst_delivery_window_p95_upper_ms": 20},
            "throughput_windows": [{"fps": {"total": 20}}],
            "passed": True,
            "failure": None,
            "scenario": "normal",
            "disconnect_events": 0,
            "config": {"processing_overlap": False},
        }
        result = soak.summarize(report, 300)
        self.assertTrue(result["passed"])
        self.assertEqual(result["stages"]["capture"]["rss_drift_mib"], 10)
        self.assertEqual(result["stages"]["capture"]["mean_cpu_percent"], 100)
        report["delivery_latency_recent_ms"]["p95"] = 2500
        self.assertIn("recent delivery", soak.summarize(report, 300)["failure"])
        report["delivery_latency_recent_ms"]["p95"] = 20
        report["reporter_summary"]["worst_delivery_window_p95_upper_ms"] = 2000
        self.assertFalse(soak.summarize(report, 300)["passed"])
        report["reporter_summary"]["worst_delivery_window_p95_upper_ms"] = 20
        report["rss_samples"]["capture"] = [sample[10], sample[250]]
        self.assertFalse(soak.summarize(report, 300)["passed"])
        report["rss_samples"]["capture"] = [
            {**v, "rss_bytes": 200 * 2**20} if v["elapsed_s"] >= 240 else v for v in sample
        ]
        self.assertFalse(soak.summarize(report, 300)["passed"])
        report["rss_samples"].pop("capture")
        self.assertFalse(soak.summarize(report, 300)["passed"])
        report["rss_samples"]["capture"] = [
            {k: v for k, v in s.items() if k != "cpu_percent"} for s in sample
        ]
        self.assertFalse(soak.summarize(report, 300)["passed"])

    def test_window_histogram_evidence_is_typed_and_reconciled(self):
        benchmark = load_module("run_benchmark")
        for value in (True, -1, 2001, "0", 0.5, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.validate_window_latency({"p95": value}, "p95")
        with self.assertRaises(ValueError):
            benchmark.validate_delivery_windows(
                {"worst_delivery_window_p95_upper_ms": 10},
                [{"delivery_p95_upper_ms": 2000}],
            )
        benchmark.validate_delivery_windows(
            {"worst_delivery_window_p95_upper_ms": 2000},
            [{"delivery_p95_upper_ms": 10}],
        )

    def test_corrupt_artifacts_still_produce_failure_report(self):
        benchmark = load_module("run_benchmark")
        for output, accounting in [
            ("", '{"cameras":'),
            ('{"type":"result"}\n', "{}"),
            ("[]\n", "{}"),
            ('{"type":"throughput","fps":{"total":NaN}}\n', "{}"),
            (
                json.dumps(
                    {
                        "type": "throughput",
                        "elapsed_s": 10,
                        "counts": {"cam_01": 10**400},
                        "fps": {"cam_01": 0, "total": 0},
                    }
                ),
                "{}",
            ),
        ]:
            with self.subTest(output=output), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / "trial"

                def start(
                    *args, directory=directory, accounting=accounting, output=output, **kwargs
                ):
                    (directory / "accounting.json").write_text(accounting)
                    (directory / "results.jsonl").write_text(output)
                    return SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)

                with (
                    patch.object(benchmark.subprocess, "Popen", side_effect=start),
                    patch.object(benchmark, "reap_group"),
                    patch.object(benchmark.platform, "platform", return_value="test-linux"),
                ):
                    report = benchmark.run_trial(directory, 12, PipelineConfig())
                self.assertFalse(report["passed"])
                self.assertIsNotNone(report["failure"])
                self.assertEqual(json.loads((directory / "report.json").read_text()), report)
                self.assertEqual(report["startup_and_drain_allowance_s"], 60)

    def test_throughput_recomputed_and_reconciled_with_results(self):
        benchmark = load_module("run_benchmark")
        row = {
            "type": "throughput",
            "elapsed_s": 10,
            "counts": {"cam_01": 1},
            "fps": {"cam_01": 0.1, "total": 0.1},
        }
        benchmark.validate_result_row(row)
        for changes in (
            {"elapsed_s": 0.1},
            {"fps": {"cam_01": 0.1, "total": 99}},
            {"counts": {"cam_01": True}},
        ):
            with self.assertRaises(ValueError):
                benchmark.validate_result_row({**row, **changes})
        identities = benchmark.ResultIdentities(PipelineConfig())
        with self.assertRaises(ValueError):
            identities.accept_window(row)
        identities.accept(dict(camera_id="cam_01", frame_id=1, connection_epoch=1))
        identities.accept_window(row)
        with self.assertRaises(ValueError):
            identities.accept_window(row)
        with self.assertRaises(ValueError):
            benchmark.validate_result_row(
                {
                    "type": "reporter_summary",
                    "cameras": {},
                    "delivery_latency_ms_recent": {"p50": 3, "p95": 2, "p99": 1},
                }
            )

    def test_matrix_summary_keeps_preprocessing_section(self):
        renderer = load_module("ci_summary")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "summary.json").write_text("[]")
            (root / "preprocessing.json").write_text(
                json.dumps(
                    {
                        "measurements": [
                            {
                                "source": "noise",
                                "method": "pooled",
                                "threads": 2,
                                "median_ms": 1.5,
                                "max_error": 0,
                            }
                        ]
                    }
                )
            )
            output = io.StringIO()
            with patch("sys.argv", ["ci_summary", "--directory", folder]), redirect_stdout(output):
                self.assertEqual(renderer.main(), 0)
            self.assertIn("Performance acceptance matrix", output.getvalue())
            self.assertIn("Preprocessing", output.getvalue())
            self.assertIn("pooled", output.getvalue())

    def test_accounting_rejects_incomplete_and_mismatched_counters(self):
        benchmark = load_module("run_benchmark")
        with self.assertRaisesRegex(ValueError, "missing accounting cameras"):
            benchmark.validate_accounting(
                {
                    "schema_version": 1,
                    "cleanup_completed": True,
                    "cleanup_errors": [],
                    "exit_code": 0,
                    "cameras": {},
                },
                PipelineConfig(),
                0,
            )
        with self.assertRaisesRegex(ValueError, "invalid Reporter batch_size"):
            benchmark.validate_result_row(valid_result(batch_size=True))

    def test_result_identity_rejects_repeats_regressions_and_unknown_cameras(self):
        benchmark = load_module("run_benchmark")
        identities = benchmark.ResultIdentities(PipelineConfig())
        row = dict(camera_id="cam_01", frame_id=1, connection_epoch=1)
        identities.accept(row)
        for invalid in [row, {**row, "camera_id": "unknown"}, {**row, "frame_id": True}]:
            with self.assertRaises(ValueError):
                identities.accept(invalid)
        identities.accept({**row, "frame_id": 4, "connection_epoch": 2})
        with self.assertRaises(ValueError):
            identities.accept({**row, "frame_id": 5})
        self.assertEqual(identities.counts["cam_01"], 2)

    def test_per_camera_accounting_catches_equal_total_wrong_distribution(self):
        benchmark = load_module("run_benchmark")
        counters = dict(
            captured=1,
            processed=1,
            reported=1,
            in_flight=0,
            dropped_no_slot=0,
            dropped_stale=0,
            dropped_shutdown=0,
            dropped_publication_timeout=0,
            dropped_superseded=0,
        )
        config = PipelineConfig()
        accounting = {
            "schema_version": 1,
            "cleanup_completed": True,
            "cleanup_errors": [],
            "exit_code": 0,
            "cameras": {c.camera_id: counters for c in config.cameras},
        }
        with self.assertRaisesRegex(ValueError, "per-camera"):
            benchmark.validate_accounting(accounting, config, 4, {"cam_01": 4})
        benchmark.validate_accounting(
            accounting, config, 4, {c.camera_id: 1 for c in config.cameras}
        )

    def test_malformed_renderer_rows_retain_valid_sections(self):
        renderer = load_module("ci_summary")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "summary.json").write_text("[{}]")
            (root / "preprocessing.json").write_text(
                json.dumps(
                    {
                        "measurements": [
                            {},
                            {
                                "source": "noise",
                                "method": "retained",
                                "threads": 1,
                                "median_ms": 1,
                                "max_error": 0,
                            },
                        ],
                        "component_measurements": [None],
                    }
                )
            )
            output = io.StringIO()
            with patch("sys.argv", ["ci_summary", "--directory", folder]), redirect_stdout(output):
                self.assertEqual(renderer.main(), 0)
            self.assertIn("Invalid report row", output.getvalue())
            self.assertIn("retained", output.getvalue())

    def test_malformed_pipeline_and_nonfinite_json_do_not_crash_renderer(self):
        renderer = load_module("ci_summary")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for text in ("{}", '{"exit_code":NaN}', '{"a":1,"a":2}'):
                (root / "report.json").write_text(text)
                with (
                    patch("sys.argv", ["ci_summary", "--directory", folder]),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(renderer.main(), 0)

    def test_terminal_summary_matches_counts_histogram_and_latency(self):
        from collections import Counter, deque

        benchmark = load_module("run_benchmark")
        row = {
            "type": "reporter_summary",
            "elapsed_s": 12,
            "cameras": {
                "cam_01": {
                    "camera_id": "cam_01",
                    "reported": 1,
                    "latency_ms_recent": {"p50": 2, "p95": 2, "p99": 2},
                }
            },
            "result_counts_by_batch_size": {"1": 1},
            "delivery_latency_ms_recent": {"p50": 2, "p95": 2, "p99": 2},
        }
        counts, histogram, latency = Counter(cam_01=1), {"1": 1}, deque([1])
        benchmark.validate_reporter_summary(row, counts, histogram, latency)
        for change in (
            {"cameras": {}},
            {"result_counts_by_batch_size": {}},
            {"delivery_latency_ms_recent": {"p50": 0, "p95": 0, "p99": 0}},
        ):
            with self.assertRaises(ValueError):
                benchmark.validate_reporter_summary({**row, **change}, counts, histogram, latency)

    def test_summary_must_be_unique_and_terminal(self):
        from cv_pipeline.config import CameraConfig

        benchmark = load_module("run_benchmark")
        result = valid_result()
        summary = {
            "type": "reporter_summary",
            "elapsed_s": 12,
            "cameras": {
                "cam": {
                    "camera_id": "cam",
                    "reported": 1,
                    "latency_ms_recent": {"p50": 2, "p95": 2, "p99": 2},
                }
            },
            "result_counts_by_batch_size": {"1": 1},
            "delivery_latency_ms_recent": {"p50": 2, "p95": 2, "p99": 2},
        }
        counters = dict(
            captured=1,
            processed=1,
            reported=1,
            in_flight=0,
            dropped_no_slot=0,
            dropped_stale=0,
            dropped_shutdown=0,
            dropped_publication_timeout=0,
            dropped_superseded=0,
        )
        config = PipelineConfig(cameras=(CameraConfig("cam", 15),))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "accounting.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cleanup_completed": True,
                        "cleanup_errors": [],
                        "exit_code": 0,
                        "cameras": {"cam": counters},
                    }
                )
            )
            for extra in ([], [summary], [{**result, "frame_id": 2}]):
                (path / "results.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in [result, summary, *extra])
                )
                artifacts = benchmark.read_artifacts(path, config, None)
                self.assertEqual(artifacts.count, 1)
                if extra:
                    self.assertIsNotNone(artifacts.failure)
                else:
                    self.assertIsNone(artifacts.failure)

            window = {
                "type": "throughput",
                "elapsed_s": 10,
                "counts": {"cam": 1},
                "fps": {"cam": 0.1, "total": 0.1},
            }
            for elapsed, valid in ((0, False), (9.99, False), (9.9995, True), (12, True)):
                (path / "results.jsonl").write_text(
                    "\n".join(
                        json.dumps(r) for r in [result, window, {**summary, "elapsed_s": elapsed}]
                    )
                )
                artifacts = benchmark.read_artifacts(path, config, None)
                self.assertEqual(artifacts.failure is None, valid, artifacts.failure)

            # Exercise the complete reader with genuine accounting, not only the
            # validator helper: corrupted evidence must never count as success.
            invalid_rows = [{k: v for k, v in result.items() if k != field} for field in result]
            invalid_rows.extend(
                valid_result(**change)
                for change in (
                    {"processing_latency_ms": -100},
                    {"timestamp": "not-a-timestamp"},
                    {"timestamp": "1970-01-01T00:00:02+00:00"},
                    {"received_utc_ns": True},
                    {"detection": 1},
                    {"mean_brightness": 256},
                    {"inference_ms": 2},
                    {"queue_and_batch_wait_ms": 2},
                    {"processing_latency_ms": 10**400},
                )
            )
            for invalid in invalid_rows:
                with self.subTest(invalid=invalid):
                    (path / "results.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in [invalid, summary])
                    )
                    self.assertIsNotNone(benchmark.read_artifacts(path, config, None).failure)
            for changes in (
                {"counts": {"cam": 10**400}},
                {"elapsed_s": 10**400},
                {"fps": {"cam": 10**400, "total": 10**400}},
            ):
                (path / "results.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in [result, {**window, **changes}, summary])
                )
                self.assertIsNotNone(benchmark.read_artifacts(path, config, None).failure)

    def test_accounting_requires_version_and_consistent_cleanup_status(self):
        benchmark = load_module("run_benchmark")
        config = PipelineConfig()
        base = {
            "schema_version": 1,
            "cleanup_completed": True,
            "cleanup_errors": [],
            "exit_code": 0,
            "cameras": {camera.camera_id: {} for camera in config.cameras},
        }
        benchmark.accounting_cameras(base, config)
        for key in ("schema_version", "cleanup_completed", "cleanup_errors"):
            invalid = {k: v for k, v in base.items() if k != key}
            with self.assertRaises(ValueError):
                benchmark.accounting_cameras(invalid, config)
        for changes in (
            {"schema_version": True},
            {"cleanup_errors": ["error"]},
            {"cleanup_completed": False},
        ):
            with self.assertRaises(ValueError):
                benchmark.accounting_cameras({**base, **changes}, config)

    def test_benchmark_sources_have_explicit_deterministic_disconnect_policy(self):
        workload = load_module("workload")
        from cv_pipeline.config import CameraConfig

        for name in ("cam_01", "cam_02"):
            config = CameraConfig(name, 15)
            stable = workload.stable_factory(config)
            faulty = workload.disconnect_factory(config)
            try:
                self.assertEqual(stable._camera._disconnect_at, 2**63 - 1)
                self.assertEqual(
                    faulty._camera._disconnect_at, 30 if name == "cam_01" else 2**63 - 1
                )
            finally:
                stable.release()
                faulty.release()
