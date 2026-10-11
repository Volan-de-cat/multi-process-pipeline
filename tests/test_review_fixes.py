"""Regressions for publication progress, diagnostic ownership and metadata limits."""

import io
import logging
import queue
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cv_pipeline.config import CameraConfig, PipelineConfig, validate_config
from cv_pipeline.logging_setup import LogListenerOwner
from cv_pipeline.messages import FrameMetadata
from cv_pipeline.processing import publish_results
from cv_pipeline.supervisor import PipelineSupervisor
from cv_pipeline.validation import validate_record


class ReviewFixTests(unittest.TestCase):
    def test_each_published_result_refreshes_progress(self):
        runtime = Mock()
        resources = SimpleNamespace(results=object(), abort=object())
        with patch("cv_pipeline.processing.put_until_stopped") as publish:
            publish_results([object(), object(), object()], resources, runtime)
        self.assertEqual(publish.call_count, 3)
        self.assertEqual(runtime.tick.call_count, 3)
        for call in runtime.tick.call_args_list:
            self.assertEqual(call.kwargs, {"progress": True, "force": True})

    def test_live_degraded_listener_is_never_replaced(self):
        class CorruptQueue(queue.Queue):
            def get(self, *args, **kwargs):
                value = super().get(*args, **kwargs)
                if value is None:
                    raise ValueError("malformed consumed datagram")
                return value

        channel = CorruptQueue()
        stream = io.StringIO()
        listener = LogListenerOwner(channel, stream)
        channel.put(None)
        channel.put(logging.LogRecord("test", logging.INFO, "", 0, "survived", (), None))
        listener.start()
        try:
            deadline = time.monotonic() + 1
            while "survived" not in stream.getvalue() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIn("survived", stream.getvalue())
            self.assertTrue(listener.is_alive)
            with patch("cv_pipeline.supervisor.LogListenerOwner") as factory:
                self.assertIs(PipelineSupervisor._failure_listener(listener, channel), listener)
                factory.assert_not_called()
        finally:
            self.assertTrue(listener.close())

    def test_reserved_camera_and_unsupported_architecture_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_config(replace(PipelineConfig(), cameras=(CameraConfig("total", 15),)))
        with (
            patch("cv_pipeline.config.platform.machine", return_value="aarch64"),
            self.assertRaisesRegex(ValueError, "x86_64"),
        ):
            validate_config(PipelineConfig())

    def test_utc_metadata_must_be_serializable(self):
        validate_record(FrameMetadata("cam", 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            validate_record(FrameMetadata("cam", 1, 1, 10**30, 1))

    def test_invalid_log_time_rejected_and_formatter_failure_is_local(self):
        from cv_pipeline.wire import decode, encode

        bad = logging.LogRecord("bad", 20, "", 0, "bad time", (), None)
        bad.created = 1e300
        with self.assertRaises(ValueError):
            encode(bad)
        valid = logging.LogRecord("good", 20, "", 0, "valid later log", (), None)
        payload = encode(valid).replace(str(valid.created).encode(), b"1e300")
        with self.assertRaises(ValueError):
            decode(payload)
        channel = queue.Queue()
        channel.put(bad)
        channel.put(valid)
        stream = io.StringIO()
        listener = LogListenerOwner(channel, stream)
        listener.start()
        self.assertTrue(listener.close())
        self.assertEqual(listener.dropped_records, 1)
        self.assertIn("valid later log", stream.getvalue())

    def test_serial_preparation_refreshes_each_frame_then_inference_budget(self):
        from cv_pipeline.processing import PreparedBatch, serial_results

        collector, processor, runtime = Mock(), Mock(), Mock()
        collector.collect.side_effect = [[object(), object()], []]

        def prepare(descriptors, progress):
            for _ in descriptors:
                progress()
            return PreparedBatch([], [])

        processor.prepare.side_effect = prepare
        processor.infer.return_value = []
        self.assertEqual(list(serial_results(collector, processor, runtime)), [[]])
        self.assertEqual(runtime.tick.call_count, 4)
        processor.infer.assert_called_once()

    def test_summary_close_failure_invalidates_published_bytes(self):
        import tempfile
        from pathlib import Path

        from cv_pipeline.output import ResultOutput

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            output = ResultOutput(str(path), 65536)
            original = output.close

            def fail_close():
                original()
                raise OSError("injected close failure")

            with (
                patch.object(output, "close", side_effect=fail_close),
                self.assertRaises(OSError),
            ):
                output.finish('{"exit_code":0}\n')
            self.assertEqual(path.read_bytes(), b"")

    def test_cleanup_failure_is_published_as_nonzero_after_all_cleanup(self):
        import json
        from queue import Empty

        from cv_pipeline.logging_setup import ListenerState

        owner = Mock()
        owner.close.side_effect = OSError("injected owner cleanup failure")
        owner.worker_resources.return_value.health.get_nowait.side_effect = Empty
        listener = Mock(state=ListenerState.CLOSED, dropped_records=0)
        listener.close.return_value = True
        output = Mock()
        supervisor = PipelineSupervisor(PipelineConfig(), summary_path="unused.json")
        with (
            patch("cv_pipeline.supervisor.DatagramChannel") as channel,
            patch("cv_pipeline.supervisor.LogListenerOwner", return_value=listener),
            patch("cv_pipeline.supervisor.TransportOwner", return_value=owner),
            patch.object(supervisor, "_reserve_summary", return_value=output),
            patch.object(supervisor, "_start"),
            patch.object(supervisor, "_monitor_workers"),
            patch.object(supervisor, "_verify_accounting", return_value={}),
            patch.object(supervisor, "_reap_workers"),
            patch.object(supervisor, "_report_finalization_errors"),
        ):
            self.assertEqual(supervisor.run(), 1)
            channel.return_value.close.assert_called_once()
        summary = json.loads(output.finish.call_args.args[0])
        self.assertEqual(summary["exit_code"], 1)
        self.assertFalse(summary["cleanup_completed"])
        self.assertIn("injected owner cleanup failure", summary["cleanup_errors"])

    def test_worker_cleanup_continues_after_join_failure_and_exit_race(self):
        from cv_pipeline.messages import Stage

        supervisor = PipelineSupervisor(PipelineConfig())
        first, second = Mock(), Mock()
        first.is_alive.return_value = True
        first.kill.side_effect = ProcessLookupError("exit race")
        first.join.side_effect = OSError("join failure")
        second.is_alive.side_effect = [True, False, False]
        supervisor._children = {Stage.CAPTURE: first, Stage.PROCESSING: second}
        with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
            supervisor._reap_workers(1)
        second.kill.assert_called_once()
        second.join.assert_called_once()
        second.close.assert_called_once()

    def test_recovered_join_closes_handles_and_preserves_original_error(self):
        from cv_pipeline.messages import Stage

        supervisor = PipelineSupervisor(PipelineConfig())
        child = Mock()
        original = OSError("temporary join failure")
        child.join.side_effect = [original, None]
        child.is_alive.return_value = False
        supervisor._children = {Stage.CAPTURE: child}
        with self.assertRaises(RuntimeError) as raised:
            supervisor._reap_workers(0)
        self.assertIs(raised.exception.__cause__, original)
        self.assertEqual(child.join.call_count, 2)
        child.close.assert_called_once()

    def test_retry_does_not_close_live_worker(self):
        from cv_pipeline.messages import Stage

        supervisor = PipelineSupervisor(PipelineConfig())
        child = Mock()
        child.join.side_effect = [OSError("join failure"), None]
        child.is_alive.return_value = True
        supervisor._children = {Stage.CAPTURE: child}
        with self.assertRaisesRegex(RuntimeError, "2 worker cleanup failure"):
            supervisor._reap_workers(0)
        child.close.assert_not_called()

    def test_close_failure_does_not_retry_join_on_closed_handles(self):
        from cv_pipeline.messages import Stage

        supervisor = PipelineSupervisor(PipelineConfig())
        child = Mock()
        child.is_alive.return_value = False
        child.close.side_effect = OSError("close failure")
        supervisor._children = {Stage.CAPTURE: child}
        with self.assertRaises(RuntimeError):
            supervisor._reap_workers(0)
        child.join.assert_called_once()
        child.close.assert_called_once()
        child.kill.assert_not_called()

    def test_camera_release_is_active_and_watchdog_detects_stall(self):
        import threading

        from cv_pipeline.adapters import SystemClock
        from cv_pipeline.capture import (
            CameraObservation,
            CameraState,
            CameraWorker,
            _monitor_camera_threads,
        )
        from cv_pipeline.observability import MetricsRecorder

        camera = Mock()
        camera.read.side_effect = RuntimeError("read failure")
        worker = CameraWorker(
            PipelineConfig().cameras[0],
            0,
            PipelineConfig(),
            Mock(),
            threading.Event(),
            threading.Event(),
            SystemClock(),
            factory=lambda _: camera,
        )
        states = []
        camera.release.side_effect = lambda: states.append(worker.observation().state)
        with self.assertRaisesRegex(RuntimeError, "read failure"):
            worker.run()
        self.assertEqual(states, [CameraState.RELEASING])
        self.assertEqual(worker.observation().state, CameraState.STOPPED)
        stuck = Mock(camera_config=PipelineConfig().cameras[0])
        stuck.observation.return_value = CameraObservation(
            CameraState.RELEASING, 1, time.monotonic_ns()
        )
        thread = Mock()
        thread.is_alive.return_value = True
        with self.assertRaisesRegex(RuntimeError, "camera thread stalled"):
            _monitor_camera_threads(
                [thread], [stuck], queue.Queue(), MetricsRecorder(), PipelineConfig(), Mock()
            )
