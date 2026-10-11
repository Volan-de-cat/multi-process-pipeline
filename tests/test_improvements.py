"""Ownership, rotation and full-run latency acceptance regressions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from cv_pipeline.config import CameraConfig, PipelineConfig, load_config, validate_config
from cv_pipeline.output import ResultOutput
from cv_pipeline.reporter import EmissionBatch
from cv_pipeline.resources import reserve_shared_memory


class ImprovementTests(unittest.TestCase):
    def test_configuration_snapshots_mutable_api_inputs(self):
        cameras = [CameraConfig("camera", 15)]
        size = [640, 480]
        config = PipelineConfig(cameras=cameras, output_size=size)
        validate_config(config)
        cameras.clear()
        size[0] = 1
        self.assertEqual(config.cameras, (CameraConfig("camera", 15),))
        self.assertEqual(config.output_size, (640, 480))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"output_max_files": 2}')
            with self.assertRaises(ValueError):
                load_config(path)
            self.assertEqual(load_config(path, output_path="results.jsonl").output_max_files, 2)

    def test_rotation_preserves_records_and_enforces_aggregate_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            output = ResultOutput(str(path), 8, max_files=2)
            try:
                with self.assertRaises(ValueError):
                    output.finish("summary")
                output.write("1234567\n")
                output.write("abcdefg\n")
                with self.assertRaises(OSError):
                    output.write("x\n")
            finally:
                output.close()
            self.assertEqual(path.read_text(), "1234567\n")
            self.assertEqual(Path(str(path) + ".0001").read_text(), "abcdefg\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_rotation_never_overwrites_existing_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results"
            successor = Path(str(path) + ".0001")
            successor.write_text("keep")
            output = ResultOutput(str(path), 4, max_files=2)
            try:
                output.write("1234")
                with self.assertRaises(FileExistsError):
                    output.write("x")
                with self.assertRaises(OSError):
                    output.write("oversized")
            finally:
                output.close()
            self.assertEqual(successor.read_text(), "keep")
            self.assertEqual(path.read_text(), "1234")

    def test_histogram_keeps_early_bad_window_and_final_partial_window(self):
        emission = EmissionBatch(Mock(), PipelineConfig(), Mock(), Mock(), Mock())
        emission._window_latency[2000] = 100
        self.assertEqual(emission.complete_latency_window(), 2000)
        emission._window_latency[10] = 100
        self.assertEqual(emission.complete_latency_window(), 10)
        self.assertEqual(emission.worst_window_p95_ms, 2000)
        self.assertIsNone(emission.complete_latency_window())
        self.assertEqual(len(emission._window_latency), 2001)

    def test_backing_adapter_fails_explicitly_for_unsupported_handle(self):
        with self.assertRaises(RuntimeError):
            reserve_shared_memory(Mock(_fd=None), 100)
