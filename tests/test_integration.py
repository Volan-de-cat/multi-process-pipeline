"""Real spawned-stage lifecycle tests using the unmodified 4K simulator."""

import json
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "tests/pipeline_scenario.py"


class PipelineIntegrationTests(unittest.TestCase):
    def launch(self, scenario="normal", direct=False, pipe_logs=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.output = self.folder / "results.jsonl"
        self.logs = self.folder / "service.log"
        self.log_handle = self.logs.open("w")
        self.addCleanup(self.log_handle.close)
        command = (
            [sys.executable, "-m", "cv_pipeline", "--output", str(self.output)]
            if direct
            else [sys.executable, str(SCENARIO), scenario, str(self.output)]
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if pipe_logs else self.log_handle,
            start_new_session=True,
        )
        self.addCleanup(self.reap, process)
        return process

    @staticmethod
    def reap(process):
        """Clean the complete isolated group even when its direct parent has exited."""
        from cv_pipeline.lifecycle import reap_group

        reap_group(process)

    def wait_ready(self, process):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            text = self.logs.read_text()
            if "stage_ready stage=capture" in text:
                return text
            if process.poll() is not None:
                self.fail(text)
            time.sleep(0.05)
        self.fail("startup timed out: " + self.logs.read_text())

    def rows(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def assert_cleanup(self):
        text = self.logs.read_text()
        names = re.findall(r"pool_created name=(\S+)", text)
        self.assertEqual(len(names), 1, text)
        self.assertFalse((Path("/dev/shm") / names[0]).exists())
        self.assertNotIn("resource_tracker: There appear", text)
        # Pool removal alone does not prove worker exit: check the recorded PIDs.
        for pid in re.findall(r"stage_ready stage=\w+ pid=(\d+)", text):
            self.assertFalse(Path(f"/proc/{pid}").exists(), f"worker PID {pid} survived")
        return text

    def assert_normal(self, process):
        self.assertEqual(process.wait(timeout=20), 0, self.logs.read_text())
        text = self.assert_cleanup()
        self.assertIn('"in_flight": 0', text)
        rows = self.rows()
        self.assertEqual(rows[-1]["type"], "reporter_summary")
        results = [r for r in rows if r["type"] == "result"]
        self.assertEqual(
            {r["camera_id"] for r in results}, {"cam_01", "cam_02", "cam_03", "cam_04"}
        )
        for camera in ("cam_01", "cam_02", "cam_03", "cam_04"):
            ids = [r["frame_id"] for r in results if r["camera_id"] == camera]
            self.assertEqual(ids, sorted(set(ids)))
        return results, text

    def test_overlap_engine_failure_reaps_workers_and_pool(self):
        process = self.launch("overlap-engine-error")
        self.assertNotEqual(process.wait(timeout=15), 0)
        self.assert_cleanup()

    def test_overlap_preparation_failure_reaps_workers_and_pool(self):
        process = self.launch("overlap-preprocess-error")
        self.assertNotEqual(process.wait(timeout=15), 0)
        self.assert_cleanup()

    def test_overlap_preparation_stall_is_detected_while_main_thread_is_alive(self):
        process = self.launch("overlap-preprocess-stuck")
        self.wait_ready(process)
        self.assertNotEqual(process.wait(timeout=10), 0)
        text = self.assert_cleanup()
        self.assertIn("stalled stages", text)

    def test_overlap_sigterm_drains_with_slow_output_and_full_result_queue(self):
        process = self.launch("overlap-slow-reporter")
        self.wait_ready(process)
        time.sleep(1)
        process.send_signal(signal.SIGTERM)
        self.assert_normal(process)

    def test_overlap_drains_with_complete_slot_accounting(self):
        self.assert_normal(self.launch("overlap"))

    def test_latest_policy_drains_and_accounts_replaced_frames(self):
        self.assert_normal(self.launch("latest"))

    def test_buffered_output_flushes_all_results_before_eos(self):
        self.assert_normal(self.launch("buffered"))

    def test_overlap_stalled_collector_cannot_hide_behind_idle_heartbeats(self):
        process = self.launch("overlap-collector-stuck")
        self.wait_ready(process)
        self.assertNotEqual(process.wait(timeout=15), 0)
        # Either the preparer's local deadline or the parent's active-work budget
        # can expire first; both must identify Processing while main heartbeats live.
        self.assertRegex(
            self.assert_cleanup(), r"batch preparer stalled|stalled stages: \['processing'\]"
        )

    def test_low_fps_camera_is_not_mistaken_for_stalled_read(self):
        process = self.launch("low-fps")
        self.assertEqual(process.wait(timeout=20), 0, self.logs.read_text())
        self.assert_cleanup()
        results = [row for row in self.rows() if row["type"] == "result"]
        self.assertGreaterEqual(len(results), 2)
        self.assertEqual({row["camera_id"] for row in results}, {"slow"})

    def test_summary_error_remains_visible_after_diagnostic_drain(self):
        process = self.launch("summary-error")
        self.assertNotEqual(process.wait(timeout=20), 0)
        logs = self.assert_cleanup()
        self.assertIn("pipeline_failed", logs)
        self.assertIn("injected summary ENOSPC", logs)
        self.assertNotIn('"exit_code": 0', logs)

    def test_end_to_end_eos_and_slot_accounting(self):
        process = self.launch()
        self.assert_normal(process)

    def test_reconnect_isolated_and_sequence_preserved(self):
        process = self.launch("reconnect")
        results, text = self.assert_normal(process)
        self.assertIn("camera_disconnected camera=cam_01", text)
        self.assertIn(2, {r["connection_epoch"] for r in results if r["camera_id"] == "cam_01"})
        self.assertTrue(any(r["camera_id"] == "cam_02" and r["frame_id"] > 3 for r in results))

    def test_overload_has_explicit_drops_and_cleans_up(self):
        process = self.launch("overload")
        results, text = self.assert_normal(process)
        summary = json.loads(text.split("pipeline_summary ")[-1].splitlines()[0])
        self.assertGreater(sum(c["dropped_no_slot"] for c in summary["cameras"].values()), 0)
        self.assertGreater(sum(c["dropped_stale"] for c in summary["cameras"].values()), 0)

    def test_output_quota_failure_cleans_up_without_partial_json_record(self):
        """Output quota is a controlled pipeline failure, not an unbounded disk write."""
        process = self.launch("quota")
        self.assertNotEqual(process.wait(timeout=15), 0)
        text = self.assert_cleanup()
        self.assertIn("quota exceeded", text)
        self.assertLessEqual(self.output.stat().st_size, 1024)
        self.rows()  # Every emitted line remains complete JSON.

    def test_all_cameras_offline_fails_availability_despite_healthy_processes(self):
        """Reconnect heartbeats cannot disguise permanent loss of all useful input."""
        process = self.launch("offline")
        self.assertNotEqual(process.wait(timeout=15), 0)
        text = self.assert_cleanup()
        self.assertIn("all cameras unavailable", text)

    def test_stuck_processing_is_detected(self):
        process = self.launch("stuck")
        self.assertNotEqual(process.wait(timeout=15), 0)
        text = self.assert_cleanup()
        self.assertIn("stalled stages", text)
        self.assertIn("processing", text)

    def test_sigint_drains(self):
        process = self.launch(direct=True)
        self.wait_ready(process)
        time.sleep(0.8)
        process.send_signal(signal.SIGINT)
        self.assert_normal(process)

    def test_group_sigterm_drains(self):
        import os

        process = self.launch(direct=True)
        self.wait_ready(process)
        time.sleep(0.8)
        os.killpg(process.pid, signal.SIGTERM)
        self.assert_normal(process)

    def test_saturated_unread_stderr_does_not_prevent_exit(self):
        process = self.launch("noisy", pipe_logs=True)
        # Deliberately leave stderr unread until exit: blocking stream writes
        # would prevent the parent/logger from reaching cleanup or finalization.
        self.assertEqual(process.wait(timeout=15), 0)
        text = process.stderr.read().decode()
        process.stderr.close()
        self.logs.write_text(text)
        self.assert_cleanup()
        self.assertEqual(self.rows()[-1]["type"], "reporter_summary")

    def test_repeated_processing_crashes_leave_no_waiters_or_workers(self):
        import os

        for delay in (0, 0.007, 0.021, 0.05, 0.12, 0.3):
            with self.subTest(delay=delay):
                process = self.launch(direct=True)
                text = self.wait_ready(process)
                pid = int(re.search(r"stage_ready stage=processing pid=(\d+)", text).group(1))
                time.sleep(delay)
                started = time.monotonic()
                os.kill(pid, signal.SIGKILL)
                self.assertNotEqual(process.wait(timeout=5), 0)
                self.assertLess(time.monotonic() - started, 5)
                self.assert_cleanup()

    def test_sigterm_with_slow_reporter_and_full_result_queue_drains(self):
        process = self.launch("slow-reporter")
        self.wait_ready(process)
        time.sleep(1)
        process.send_signal(signal.SIGTERM)
        self.assert_normal(process)

    def test_each_worker_crash_is_detected_and_memory_removed(self):
        for stage in ("capture", "processing", "reporter"):
            with self.subTest(stage=stage):
                process = self.launch(direct=True)
                text = self.wait_ready(process)
                pid = int(re.search(rf"stage_ready stage={stage} pid=(\d+)", text).group(1))
                import os

                os.kill(pid, signal.SIGKILL)
                self.assertNotEqual(process.wait(timeout=15), 0)
                text = self.assert_cleanup()
                self.assertIn(stage + " exited with code", text)

    def test_reporter_startup_error_returns_nonzero_and_cleans_memory(self):
        process = self.launch()
        # This run verifies the CLI/config boundary separately below; reap normal run.
        self.assert_normal(process)
        config = self.folder / "config.json"
        config.write_text(
            json.dumps({"output_path": str(self.folder / "missing" / "results.jsonl")})
        )
        result = subprocess.run(
            [sys.executable, "-m", "cv_pipeline", "--config", str(config)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        name = re.search(r"pool_created name=(\S+)", result.stderr).group(1)
        self.assertFalse((Path("/dev/shm") / name).exists())
        self.assertIn("reporter", result.stderr)


if __name__ == "__main__":
    unittest.main()
