"""Regression tests for malicious input, output safety and parent-death cleanup."""

import json
import stat
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cv_pipeline.adapters import FakeInferenceAdapter
from cv_pipeline.config import CameraConfig, PipelineConfig, load_config, validate_config
from cv_pipeline.ipc import DatagramChannel
from cv_pipeline.output import ResultOutput
from cv_pipeline.wire import decode, encode


class SecurityTests(unittest.TestCase):
    def test_sealed_pool_remains_usable_without_a_named_segment(self):
        """Early unlink preserves attached pixels and needs no tracker during steady state."""
        from cv_pipeline.transport import FrameTransport, TransportOwner

        config = replace(
            PipelineConfig(), cameras=(CameraConfig("cam_01", 15),), slots_per_camera=1
        )
        owner = TransportOwner(config)
        transport = FrameTransport(owner.worker_resources())
        try:
            owner.seal()
            self.assertFalse(Path("/dev/shm", owner.worker_resources().memory.name).exists())
            frame = transport._view(0)
            frame[0, 0, 0] = 23
            self.assertEqual(int(frame[0, 0, 0]), 23)
            del frame
        finally:
            transport.close()
            owner.close()

    def test_resource_and_payload_limits_reject_pathological_configuration(self):
        """Bad configurations fail before any large memory allocation or process spawn."""
        for change in (
            {"output_size": (100000, 100000)},
            {"cameras": (CameraConfig("x" * 70000, 15),)},
            {"cameras": (CameraConfig("forged\nlog", 15),)},
            {"slots_per_camera": 65},
            {"result_queue_capacity": 100000},
            {"batch_wait_ms": 1e-12},
            {"health_timeout_s": 1e308},
            {"memory_budget_mib": 64},
            {"output_max_bytes": -1},
        ):
            with self.subTest(change=next(iter(change))), self.assertRaises(ValueError):
                validate_config(replace(PipelineConfig(), **change))

    def test_backing_reservation_failure_unlinks_pool(self):
        """ENOSPC from atomic backing reservation cannot leave the partially created pool."""
        from multiprocessing import shared_memory

        from cv_pipeline.transport import TransportOwner

        names = []
        original = shared_memory.SharedMemory

        def create(*args, **kwargs):
            memory = original(*args, **kwargs)
            names.append(memory.name)
            return memory

        config = replace(
            PipelineConfig(), cameras=(CameraConfig("cam_01", 15),), slots_per_camera=1
        )
        with (
            patch("cv_pipeline.transport.shared_memory.SharedMemory", side_effect=create),
            patch(
                "cv_pipeline.transport.os.posix_fallocate", side_effect=OSError("injected ENOSPC")
            ),
            self.assertRaisesRegex(OSError, "ENOSPC"),
        ):
            TransportOwner(config)
        self.assertEqual(len(names), 1)
        self.assertFalse(Path("/dev/shm", names[0]).exists())

    def test_duplicate_configuration_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"batch_size":4,"batch_size":8}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_config(path)

    def test_configuration_size_symlinks_and_special_files_rejected(self):
        """Config reads are bounded and cannot block on FIFO/device inputs."""
        with tempfile.TemporaryDirectory() as folder:
            oversized = Path(folder) / "oversized.json"
            oversized.write_bytes(b" " * 65537)
            with self.assertRaisesRegex(ValueError, "64 KiB"):
                load_config(oversized)
            link = Path(folder) / "link"
            link.symlink_to(oversized)
            with self.assertRaises(OSError):
                load_config(link)
            fifo = Path(folder) / "fifo"
            import os

            os.mkfifo(fifo)
            with self.assertRaisesRegex(ValueError, "regular file"):
                load_config(fifo)

    def test_output_rejects_existing_files_and_symlinks_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "original"
            target.write_text("keep")
            link = Path(folder) / "link"
            link.symlink_to(target)
            for path in (target, link):
                with self.assertRaises(OSError):
                    ResultOutput(str(path), 1024)
            self.assertEqual(target.read_text(), "keep")
            real = Path(folder) / "directory"
            real.mkdir()
            alias = Path(folder) / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(OSError):
                ResultOutput(str(alias / "results"), 1024)
            self.assertFalse((real / "results").exists())

    def test_output_is_private_and_quota_failure_does_not_partially_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results"
            output = ResultOutput(str(path), 5, durable=True)
            try:
                output.write("abc\n")
                output.flush()
                with self.assertRaises(OSError):
                    output.write("é\n")
            finally:
                output.close()
            self.assertEqual(path.read_text(), "abc\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_wire_rejects_pickle_unknown_tags_bad_types_and_nonfinite_values(self):
        """The decoder never dispatches arbitrary constructors or accepts coercion."""
        for payload in (
            b"\x80\x04malicious-pickle",
            b'{"type":"os.system","value":"touch /tmp/unwanted"}',
            b'{"type":"EndOfStream","value":{"stage":"capture"}}',
            b"NaN",
            b"[1,2]",
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode(payload)
        self.assertEqual(decode(encode({"value": [1, True, "text"]})), {"value": [1, True, "text"]})

    def test_exact_queue_capacity_and_schema_errors_do_not_leak_credits(self):
        import multiprocessing as mp
        from queue import Full

        channel = DatagramChannel(capacity=1, context=mp.get_context("spawn"))
        try:
            channel.put(1, block=False)
            with self.assertRaises(Full):
                channel.put(2, block=False)
            self.assertEqual(channel.get(timeout=0.1), 1)
            with self.assertRaises(ValueError):
                channel.put(object(), block=False)
            channel.put(3, block=False)
            self.assertEqual(channel.get(timeout=0.1), 3)
        finally:
            channel.close()

    def test_inference_schema_is_not_silently_coerced(self):
        with (
            patch(
                "cv_pipeline.adapters.inference_batch",
                return_value=[{"mean_brightness": float("nan"), "detection": True}],
            ),
            self.assertRaises(ValueError),
        ):
            FakeInferenceAdapter().infer([None])
        with (
            patch(
                "cv_pipeline.adapters.inference_batch",
                return_value=[{"mean_brightness": 100.0, "detection": "false"}],
            ),
            self.assertRaises(ValueError),
        ):
            FakeInferenceAdapter().infer([None])

    def test_parent_sigkill_stops_all_workers_and_tracker_reclaims_memory(self):
        """A separate subreaper isolates fault injection and cleans up grandchildren."""
        with tempfile.TemporaryDirectory() as folder:
            helper = Path(__file__).with_name("parent_death_scenario.py")
            process = subprocess.run(
                [sys.executable, str(helper), folder], capture_output=True, text=True, timeout=35
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertTrue(
                all(
                    state == "gone" or state.startswith("Z")
                    for state in report["worker_states"].values()
                ),
                report,
            )
            self.assertFalse(report["pool_still_exists"], report)

    def test_forced_group_kill_reclaims_pool_when_tracker_was_also_killed(self):
        """Launcher fallback handles tree-wide death without leaving named shared memory."""
        with tempfile.TemporaryDirectory() as folder:
            helper = Path(__file__).with_name("parent_death_scenario.py")
            process = subprocess.run(
                [sys.executable, str(helper), folder, "group"],
                capture_output=True,
                text=True,
                timeout=35,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            self.assertFalse(report["pool_still_exists"], report)
            self.assertTrue(
                all(
                    state == "gone" or state.startswith("Z")
                    for state in report["worker_states"].values()
                ),
                report,
            )
