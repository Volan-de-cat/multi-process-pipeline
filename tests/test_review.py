"""Regression coverage for the independent review's IPC and optimization findings."""

import json
import multiprocessing as mp
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from queue import Empty, Full
from unittest.mock import patch

import numpy as np
from test_core import ManualClock, descriptor

from cv_pipeline.batching import BatchCollector
from cv_pipeline.cli import main
from cv_pipeline.config import CameraConfig, PipelineConfig, config_json, load_config
from cv_pipeline.ipc import DatagramChannel, SlotCredits
from cv_pipeline.messages import EndOfStream, Stage
from cv_pipeline.observability import MetricsRecorder, ThroughputWindow
from cv_pipeline.output import ResultOutput
from cv_pipeline.preprocessing import FramePreprocessor
from cv_pipeline.processing import BatchProcessor, GrayscalePool
from cv_pipeline.reporter import EmissionBatch
from cv_pipeline.resources import MemoryMonitor, cgroup_memory
from cv_pipeline.wire import decode


def send_records(channel, producer, count):
    """Independent spawned producer; parent verifies every identity and FIFO order."""
    try:
        for sequence in range(count):
            channel.put({"producer": producer, "sequence": sequence}, timeout=10)
    finally:
        channel.close()


class ReviewRegressionTests(unittest.TestCase):
    def test_large_credit_counter_survives_many_roundtrips(self):
        for capacity in (256, 1024, 4096):
            with self.subTest(capacity=capacity):
                channel = DatagramChannel(capacity=capacity)
                try:
                    for value in range(3000):
                        channel.put_nowait(value)
                        self.assertEqual(channel.get_nowait(), value)
                finally:
                    channel.close()

    def test_kernel_buffer_full_rolls_back_record_credit(self):
        channel = DatagramChannel(capacity=4096)
        try:
            sent = []
            while True:
                try:
                    channel.put_nowait(len(sent))
                    sent.append(len(sent))
                except Full:
                    break
            self.assertLess(len(sent), 4096)
            self.assertEqual([channel.get_nowait() for _ in sent], sent)
            for value in range(1000):
                channel.put_nowait(value)
                self.assertEqual(channel.get_nowait(), value)
        finally:
            channel.close()

    def test_spawned_producers_preserve_all_records_and_per_producer_order(self):
        context = mp.get_context("spawn")
        channel = DatagramChannel(capacity=1024)
        children = [context.Process(target=send_records, args=(channel, i, 500)) for i in range(3)]
        try:
            for child in children:
                child.start()
            received = {i: [] for i in range(3)}
            for _ in range(1500):
                item = channel.get(timeout=10)
                received[item["producer"]].append(item["sequence"])
            for child in children:
                child.join(timeout=5)
                self.assertEqual(child.exitcode, 0)
            self.assertEqual(received, {i: list(range(500)) for i in range(3)})
        finally:
            for child in children:
                if child.pid is not None:
                    if child.is_alive():
                        child.kill()
                    child.join(timeout=3)
                    child.close()
            channel.close()

    def test_partial_slot_initialization_closes_already_created_counters(self):
        import os

        from cv_pipeline.transport import TransportOwner

        created = []

        def create_slots(context, start, count):
            if created:
                raise OSError("injected descriptor exhaustion")
            channel = SlotCredits(context, start, count)
            created.append(channel._available._fd)
            return channel

        config = replace(
            PipelineConfig(),
            slots_per_camera=1,
            cameras=(CameraConfig("a", 15), CameraConfig("b", 15)),
        )
        with (
            patch("cv_pipeline.transport.SlotCredits", side_effect=create_slots),
            self.assertRaises(OSError),
        ):
            TransportOwner(config)
        with self.assertRaises(OSError):
            os.fstat(created[0])

    def test_large_slot_partition_needs_no_prefilled_socket(self):
        slots = SlotCredits(mp.get_context("spawn"), 4096, 4096)
        try:
            values = [slots.get_nowait() for _ in range(4096)]
            self.assertEqual(set(values), set(range(4096, 8192)))
            with self.assertRaises(Empty):
                slots.get_nowait()
            for value in reversed(values):
                slots.put(value)
            with self.assertRaises(RuntimeError):
                slots.put(values[0])
            self.assertEqual({slots.get_nowait() for _ in values}, set(values))
        finally:
            slots.close()

    def test_json_rejects_numeric_overflow_duplicates_and_excess_bytes(self):
        for payload in (
            b"1e400",
            b'{"type":"dict","value":{"x":1e400}}',
            b'{"type":"dict","type":"list","value":[]}',
            b" " * 60001,
        ):
            with self.subTest(payload=payload[:80]), self.assertRaises(ValueError):
                decode(payload)

    def test_cli_reports_invalid_duration_and_deep_configuration(self):
        with self.assertRaises(SystemExit) as error:
            main(["--duration", "86401"])
        self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text("[" * 1500 + "0" + "]" * 1500)
            with self.assertRaises(ValueError):
                load_config(path)
            with self.assertRaises(SystemExit) as error:
                main(["--config", str(path), "--print-config"])
            self.assertEqual(error.exception.code, 2)
            path.write_text('{"batch_wait_ms":1e400}')
            with self.assertRaises(ValueError):
                load_config(path)

    def test_huge_integer_timing_is_rejected_without_overflow(self):
        from cv_pipeline.config import validate_config

        with self.assertRaises(ValueError):
            validate_config(replace(PipelineConfig(), batch_wait_ms=10**500))

    def test_new_options_roundtrip(self):
        config = replace(
            PipelineConfig(),
            processing_overlap=True,
            frame_policy="latest",
            output_flush_records=16,
            output_flush_interval_ms=50,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(config_json(config))
            self.assertEqual(load_config(path), config)

    def test_cgroup_limits_include_nested_ancestors_and_current_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proc, mount = root / "proc", root / "cgroup"
            (proc / "self").mkdir(parents=True)
            leaf = mount / "tenant/app"
            leaf.mkdir(parents=True)
            for directory, limit, current in (
                (mount, 10000, 1000),
                (leaf.parent, 8000, 2000),
                (leaf, 7000, 3000),
            ):
                (directory / "memory.max").write_text(str(limit))
                (directory / "memory.current").write_text(str(current))
            (proc / "self/cgroup").write_text("0::/tenant/app\n")
            (proc / "self/mountinfo").write_text(f"1 0 0:1 / {mount} rw - cgroup2 cgroup rw\n")
            observed = cgroup_memory(proc)
            self.assertEqual(
                [o["limit_bytes"] - o["current_bytes"] for o in observed], [4000, 6000, 9000]
            )

    def test_pss_monitor_detects_observed_budget_excess(self):
        monitor = MemoryMonitor(100)
        with patch("cv_pipeline.resources.process_pss", side_effect=[40, 30]):
            self.assertEqual(monitor.sample([1, 2])["pss_bytes"], 70)
        with (
            patch("cv_pipeline.resources.process_pss", return_value=101),
            self.assertRaises(RuntimeError),
        ):
            monitor.sample([1])

    def test_preallocated_grayscale_is_exact_and_does_not_alias_scratch(self):
        frame = np.random.default_rng(21).integers(0, 256, (72, 96, 3), dtype=np.uint8)
        preprocessor = FramePreprocessor((32, 24), "RGB")
        reference = preprocessor.apply(frame)
        destination = np.empty_like(reference)
        self.assertIs(preprocessor.apply(frame, destination), destination)
        np.testing.assert_array_equal(reference, destination)
        preprocessor.apply(np.zeros_like(frame))
        np.testing.assert_array_equal(reference, destination)

    def test_inference_failure_returns_all_pooled_destinations(self):
        from contextlib import contextmanager

        class Transport:
            @contextmanager
            def lease(self, item):
                yield np.zeros((2, 2, 3), np.uint8)

        class Engine:
            def infer(self, frames):
                raise ValueError("injected engine failure")

        pool = GrayscalePool(2, (2, 2), threading.Event())
        processor = BatchProcessor(
            Transport(),
            FramePreprocessor((2, 2), "RGB"),
            Engine(),
            ManualClock(),
            500,
            buffers=pool,
        )
        with self.assertRaises(ValueError):
            processor.process([descriptor(), descriptor(slot=1)])
        destinations = [pool.acquire(), pool.acquire()]
        self.assertIsNot(destinations[0], destinations[1])

    def test_latest_releases_and_accounts_superseded_frames_before_eos(self):
        from types import SimpleNamespace

        released = []

        class Transport:
            resources = SimpleNamespace(
                abort=threading.Event(), memory=SimpleNamespace(slots_per_camera=3)
            )
            items = [
                descriptor(frame_id=1),
                descriptor(frame_id=2),
                descriptor(frame_id=3),
                EndOfStream(Stage.CAPTURE),
            ]

            def receive(self, index, timeout):
                return self.items.pop(0) if self.items else None

            def release(self, item):
                released.append(item.metadata.frame_id)

        metrics = MetricsRecorder()
        collector = BatchCollector(
            Transport(), 1, 4, 50, ManualClock(), frame_policy="latest", metrics=metrics
        )
        self.assertEqual([d.metadata.frame_id for d in collector.collect()], [3])
        self.assertEqual(released, [1, 2])
        self.assertEqual(metrics.snapshot("cam_01").dropped_superseded, 2)
        self.assertEqual(collector.collect(), [])

    def test_flush_batches_account_only_after_flush_and_include_output_duration(self):
        from cv_pipeline.messages import InferenceResult, ProcessedResult, ProcessingTimings

        clock = ManualClock()

        class Output:
            flushes = 0

            def write(self, text):
                clock.now += 10_000_000

            def flush(self):
                self.flushes += 1
                clock.now += 20_000_000

        output, metrics = Output(), MetricsRecorder()
        emission = EmissionBatch(
            output,
            replace(PipelineConfig(), output_flush_records=2),
            clock,
            metrics,
            ThroughputWindow(clock),
        )
        metadata = descriptor().metadata
        result = ProcessedResult(
            metadata,
            InferenceResult(0.0, False),
            ProcessingTimings(clock.now, clock.now, clock.now, clock.now),
            1,
        )
        emission.write(result)
        self.assertFalse(emission.flush())
        self.assertEqual(metrics.snapshot("cam_01").reported, 0)
        emission.write(result)
        self.assertTrue(emission.flush())
        self.assertEqual(output.flushes, 1)
        self.assertEqual(metrics.snapshot("cam_01").reported, 2)
        self.assertEqual(emission.delivery_summary()["p95"], 40)

    def test_buffered_file_flushes_before_close(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.jsonl"
            output = ResultOutput(str(path), 1024)
            try:
                output.write(json.dumps({"x": 1}) + "\n")
                output.flush()
                self.assertEqual(json.loads(path.read_text()), {"x": 1})
            finally:
                output.close()

    def test_flush_receive_timeout_follows_pending_deadline(self):
        from types import SimpleNamespace

        from cv_pipeline.messages import InferenceResult, ProcessedResult, ProcessingTimings

        clock = ManualClock()
        output = SimpleNamespace(write=lambda text: None, flush=lambda: None)
        emission = EmissionBatch(
            output,
            replace(PipelineConfig(), output_flush_records=16, output_flush_interval_ms=1),
            clock,
            MetricsRecorder(),
            ThroughputWindow(clock),
        )
        result = ProcessedResult(
            descriptor().metadata,
            InferenceResult(0.0, False),
            ProcessingTimings(clock.now, clock.now, clock.now, clock.now),
            1,
        )
        self.assertEqual(emission.receive_timeout(), 0.1)
        emission.write(result)
        self.assertAlmostEqual(emission.receive_timeout(), 0.001)
        clock.now += 900_000
        self.assertAlmostEqual(emission.receive_timeout(), 0.0001)
        clock.now += 100_000
        self.assertEqual(emission.receive_timeout(), 0)
        self.assertTrue(emission.flush())
        self.assertEqual(emission.receive_timeout(), 0.1)

    def test_cgroup_mount_paths_decode_escapes_once(self):
        from cv_pipeline.resources import mountinfo_path

        self.assertEqual(str(mountinfo_path(r"/a\040b\134040")), "/a b\\040")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proc, mount = root / "proc", root / "memory group"
            (proc / "self").mkdir(parents=True)
            mount.mkdir()
            (mount / "memory.max").write_text("1000")
            (mount / "memory.current").write_text("100")
            escaped = str(mount).replace(" ", r"\040")
            (proc / "self/cgroup").write_text("0::/\n")
            (proc / "self/mountinfo").write_text(f"1 0 0:1 / {escaped} rw - cgroup2 cgroup rw\n")
            self.assertEqual(cgroup_memory(proc)[0]["limit_bytes"], 1000)

    def test_missing_pss_observations_are_explicit(self):
        with patch("cv_pipeline.resources.process_pss", side_effect=[10, None]):
            observation = MemoryMonitor(100).sample([1, 2])
        self.assertFalse(observation["pss_available"])
        self.assertEqual(observation["missing_pss_pids"], [2])

    def test_latest_rejects_wrong_eos_producer(self):
        from types import SimpleNamespace

        items = iter([descriptor(), EndOfStream(Stage.REPORTER)])
        transport = SimpleNamespace(
            resources=SimpleNamespace(
                abort=threading.Event(), memory=SimpleNamespace(slots_per_camera=2)
            ),
            receive=lambda index, timeout: next(items, None),
            release=lambda item: None,
        )
        collector = BatchCollector(transport, 1, 4, 50, ManualClock(), frame_policy="latest")
        with self.assertRaisesRegex(ValueError, "unexpected EOS producer"):
            collector.collect()

    def test_overlap_detects_stalled_receive_and_live_idle(self):
        from types import SimpleNamespace

        from cv_pipeline.processing import OverlappedBatches

        entered, release = threading.Event(), threading.Event()

        def receive(index, timeout):
            entered.set()
            release.wait(2)
            return EndOfStream(Stage.CAPTURE)

        transport = SimpleNamespace(
            resources=SimpleNamespace(abort=threading.Event()),
            receive=receive,
        )
        collector = BatchCollector(transport, 1, 4, 50, ManualClock())
        ticks = []
        overlap = OverlappedBatches(
            collector,
            None,
            transport.resources.abort,
            SimpleNamespace(tick=lambda **kwargs: ticks.append(kwargs)),
            0.1,
        )
        overlap.thread.start()
        try:
            self.assertTrue(entered.wait(1))
            with (
                patch(
                    "cv_pipeline.processing.time.monotonic_ns",
                    return_value=overlap._last_observation_ns + 200_000_000,
                ),
                self.assertRaisesRegex(RuntimeError, "collecting"),
            ):
                overlap._check_preparer()
        finally:
            release.set()
            self.assertTrue(overlap.close())
        overlap.observe_collection(idle=True)
        overlap._check_preparer()
        self.assertEqual(ticks[-1], {"idle": True})
        with (
            patch(
                "cv_pipeline.processing.time.monotonic_ns",
                return_value=overlap._last_observation_ns + 200_000_000,
            ),
            self.assertRaisesRegex(RuntimeError, "idle"),
        ):
            overlap._check_preparer()

    def test_overlap_close_cancels_empty_collection(self):
        from types import SimpleNamespace

        from cv_pipeline.processing import OverlappedBatches

        collector = BatchCollector(
            SimpleNamespace(
                resources=SimpleNamespace(abort=threading.Event()),
                receive=lambda index, timeout: None,
            ),
            1,
            4,
            50,
            ManualClock(),
        )
        overlap = OverlappedBatches(
            collector,
            None,
            threading.Event(),
            SimpleNamespace(tick=lambda **kw: None),
        )
        overlap.thread.start()
        self.assertTrue(overlap.close())

    def test_summary_failure_precedes_transport_allocation(self):
        from cv_pipeline.supervisor import PipelineSupervisor

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("original")
            cases = [
                (PipelineConfig(), path),
                (replace(PipelineConfig(), output_path=str(path)), path),
                (PipelineConfig(), Path(folder) / "missing/summary.json"),
            ]
            for config, summary in cases:
                with (
                    self.subTest(summary=summary, config=config.output_path),
                    patch("cv_pipeline.supervisor.DatagramChannel"),
                    patch("cv_pipeline.supervisor.LogListenerOwner"),
                    patch("cv_pipeline.supervisor.TransportOwner") as owner,
                ):
                    self.assertEqual(PipelineSupervisor(config, 1, str(summary)).run(), 1)
                    owner.assert_not_called()
            self.assertEqual(path.read_text(), "original")

    def test_listener_counts_pending_records(self):
        from queue import Queue

        from cv_pipeline.logging_setup import LogListenerOwner

        channel = Queue()
        channel.put("first")
        channel.put("second")
        listener = LogListenerOwner(channel)
        listener._discard_pending()
        self.assertEqual(listener.dropped_records, 2)
        self.assertTrue(listener.close())

    def test_worker_summary_includes_last_log_loss(self):
        import logging
        from types import SimpleNamespace

        from cv_pipeline.logging_setup import BoundedQueueHandler
        from cv_pipeline.runtime import StageRuntime

        def full(record):
            raise Full

        handler = BoundedQueueHandler(SimpleNamespace(put_nowait=full))
        root = logging.getLogger()
        old_handlers, old_level = root.handlers[:], root.level
        health = []
        try:
            root.handlers = [handler]
            root.setLevel(logging.INFO)
            runtime = StageRuntime(
                Stage.PROCESSING,
                SimpleNamespace(
                    health=SimpleNamespace(put=lambda item, **kw: health.append(item)),
                    abort=threading.Event(),
                ),
            )
            runtime.summary(MetricsRecorder())
            self.assertEqual(health[0].diagnostics["dropped_logs"], 1)
        finally:
            root.handlers, root.level = old_handlers, old_level

    def test_listener_counts_failed_record_and_abandoned_queue(self):
        import logging
        from queue import Queue

        from cv_pipeline.logging_setup import LogListenerOwner

        class BrokenStream:
            def write(self, text):
                raise OSError("injected sink failure")

            def flush(self):
                pass

        channel = Queue()
        for i in range(3):
            channel.put(logging.LogRecord("test", logging.INFO, "", 0, str(i), (), None))
        listener = LogListenerOwner(channel, stream=BrokenStream())
        listener._consume()
        self.assertEqual(listener.dropped_records, 3)
        self.assertTrue(channel.empty())

    def test_partial_log_initialization_closes_registered_health_channel(self):
        from unittest.mock import MagicMock

        from cv_pipeline.transport import TransportOwner

        ready, health = MagicMock(), MagicMock()
        config = replace(PipelineConfig(), cameras=(CameraConfig("a", 15),), slots_per_camera=1)
        with (
            patch(
                "cv_pipeline.transport.DatagramChannel",
                side_effect=[ready, health, OSError("injected log initialization error")],
            ),
            self.assertRaisesRegex(OSError, "log initialization"),
        ):
            TransportOwner(config)
        ready.close.assert_called_once()
        health.close.assert_called_once()

    def test_fdopen_failure_closes_untransferred_descriptor(self):
        import os

        opened = []

        def fail(fd, *args, **kwargs):
            opened.append(fd)
            raise OSError("injected fdopen failure")

        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("cv_pipeline.output.os.fdopen", side_effect=fail),
                self.assertRaises(OSError),
            ):
                ResultOutput(str(Path(folder) / "output"), 1024)
            with self.assertRaises(OSError):
                os.fstat(opened[0])

    def test_listener_counts_logs_arriving_after_sink_failure(self):
        import logging
        from queue import Queue

        from cv_pipeline.logging_setup import ListenerState, LogListenerOwner

        class FailedSink:
            def write(self, text):
                raise OSError("sink failure")

            def flush(self):
                pass

        channel = Queue()
        record = logging.LogRecord("test", 20, "", 0, "message", (), None)
        channel.put(record)
        listener = LogListenerOwner(channel, stream=FailedSink())
        listener.start()
        listener._thread.join(1)
        self.assertEqual(listener.state, ListenerState.FAILED)
        channel.put(record)
        self.assertTrue(listener.close())
        self.assertTrue(channel.empty())
        self.assertEqual(listener.dropped_records, 2)

    def test_latest_scan_observes_first_descriptor_deadline(self):
        from types import SimpleNamespace

        clock = ManualClock()
        received = []

        def receive(index, timeout):
            clock.now += 20_000_000
            item = descriptor(frame_id=len(received) + 1, slot=len(received))
            received.append(item)
            return item

        transport = SimpleNamespace(
            resources=SimpleNamespace(
                abort=threading.Event(), memory=SimpleNamespace(slots_per_camera=8)
            ),
            receive=receive,
            release=lambda item: None,
        )
        collector = BatchCollector(transport, 1, 1, 50, clock, frame_policy="latest")
        collector.collect()
        self.assertEqual(len(received), 4)  # first descriptor + at most one operation past deadline

    def test_typed_stage_summary_rejects_invalid_nested_counters(self):
        from cv_pipeline.messages import StageSummary, empty_camera_metrics
        from cv_pipeline.wire import encode

        counters = empty_camera_metrics("cam_01")
        counters["captured"] = 1
        valid = StageSummary(Stage.CAPTURE, {"cam_01": counters})
        self.assertEqual(decode(encode(valid)), valid)
        invalid = StageSummary(Stage.CAPTURE, {"cam_01": {**counters, "captured": True}})
        with self.assertRaises(ValueError):
            decode(encode(invalid))

    def test_logging_errors_never_fall_back_to_producer_stderr(self):
        import logging
        from types import SimpleNamespace

        from cv_pipeline.logging_setup import BoundedQueueHandler

        for error in (OSError("closed queue"), RuntimeError("unexpected transport error")):
            handler = BoundedQueueHandler(
                SimpleNamespace(put_nowait=lambda _, error=error: (_ for _ in ()).throw(error))
            )
            with patch("sys.stderr") as stderr:
                handler.emit(logging.LogRecord("review", logging.ERROR, "", 0, "failure", (), None))
            stderr.write.assert_not_called()
            self.assertEqual(handler.dropped_records, 1)

    def test_corrupt_log_is_counted_and_later_valid_record_consumed(self):
        import io
        import logging

        from cv_pipeline.logging_setup import ListenerState, LogListenerOwner

        channel = DatagramChannel()
        output = io.StringIO()
        listener = LogListenerOwner(channel, output)
        try:
            channel._sender.send(b"{bad-json")
            channel.put_nowait(
                logging.LogRecord("review", logging.INFO, "", 0, "surviving log", (), None)
            )
            listener.start()
            self.assertTrue(listener.close())
            self.assertEqual(listener.state, ListenerState.FAILED)
            self.assertEqual(listener.dropped_records, 1)
            self.assertIn("surviving log", output.getvalue())
        finally:
            listener.close()
            channel.close()

    def test_deadline_stops_outer_camera_scan_and_retains_fair_cursor(self):
        from types import SimpleNamespace

        clock = ManualClock()
        calls = []

        def receive(index, timeout):
            calls.append(index)
            clock.now += 20_000_000
            return descriptor(camera=f"cam_{index}")

        transport = SimpleNamespace(
            resources=SimpleNamespace(abort=threading.Event()), receive=receive
        )
        collector = BatchCollector(transport, 8, 16, 50, clock)
        self.assertEqual(len(collector.collect()), 4)
        self.assertEqual(calls, [0, 1, 2, 3])
        collector.collect()
        self.assertEqual(calls[4], 4)

    def test_wire_rejects_impossible_values_on_both_protocol_directions(self):
        from cv_pipeline.messages import (
            HealthEvent,
            InferenceResult,
            ProcessedResult,
            ProcessingTimings,
            StageSummary,
            empty_camera_metrics,
        )
        from cv_pipeline.wire import encode, pack

        metadata = descriptor().metadata
        counters = empty_camera_metrics("cam_01")
        valid = StageSummary(Stage.CAPTURE, {"cam_01": counters})
        payload = pack(valid)
        payload["value"]["cameras"]["value"]["cam_01"]["value"]["captured"] = -1
        with self.assertRaises(ValueError):
            decode(json.dumps(payload).encode())
        invalid_records = [
            StageSummary(Stage.CAPTURE, {"cam_01": {**counters, "captured": -1}}),
            StageSummary(Stage.CAPTURE, {"cam_01": {"captured": 1}}),
            StageSummary(Stage.CAPTURE, {"cam_02": counters}),
            StageSummary(
                Stage.CAPTURE,
                {"cam_01": {**counters, "latency_ms_recent": {"p50": 3.0, "p95": 2.0, "p99": 4.0}}},
            ),
            InferenceResult(256.0, False),
            ProcessingTimings(4, 3, 5, 6),
            HealthEvent(Stage.CAPTURE, 1, True, 2),
            ProcessedResult(
                metadata, InferenceResult(1.0, False), ProcessingTimings(1, 2, 3, 4), 17
            ),
        ]
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                encode(record)

    def test_stage_exports_omit_unrelated_handles(self):
        from cv_pipeline.transport import TransportOwner

        config = replace(PipelineConfig(), cameras=(CameraConfig("one", 15),), slots_per_camera=1)
        owner = TransportOwner(config)
        try:
            capture = owner.resources_for(Stage.CAPTURE)
            processing = owner.resources_for(Stage.PROCESSING)
            reporter = owner.resources_for(Stage.REPORTER)
            self.assertFalse(hasattr(capture, "results"))
            self.assertFalse(hasattr(processing, "capture_stop"))
            for name in ("memory", "free_slots", "ready_frames", "capture_stop"):
                self.assertFalse(hasattr(reporter, name))
        finally:
            owner.close()

    def test_cleanup_attempts_every_channel_after_first_error(self):
        from types import SimpleNamespace

        from cv_pipeline.transport import close_channels

        closed = []

        def fail():
            closed.append("bad")
            raise OSError("injected close failure")

        with self.assertRaisesRegex(RuntimeError, "1 IPC resource"):
            close_channels(
                [SimpleNamespace(close=fail), SimpleNamespace(close=lambda: closed.append("good"))]
            )
        self.assertEqual(closed, ["bad", "good"])

    def test_availability_validation_rejects_impossible_camera_budget(self):
        from cv_pipeline.config import validate_config

        with self.assertRaisesRegex(ValueError, "availability timeout"):
            validate_config(
                replace(
                    PipelineConfig(), cameras=(CameraConfig("slow", 1),), availability_timeout_s=1
                )
            )
        validate_config(
            replace(PipelineConfig(), cameras=(CameraConfig("slow", 1),), availability_timeout_s=2)
        )

        with self.assertRaisesRegex(ValueError, "availability timeout"):
            validate_config(
                replace(
                    PipelineConfig(),
                    cameras=(CameraConfig("fast", 15), CameraConfig("slow", 1)),
                    availability_timeout_s=0.7,
                )
            )

    def test_camera_operation_observation_is_immutable_and_refreshes_read_budget(self):
        from cv_pipeline.capture import CameraState, CameraWorker

        clock = ManualClock()
        worker = CameraWorker(
            CameraConfig("cam", 1),
            0,
            PipelineConfig(),
            None,
            threading.Event(),
            threading.Event(),
            clock,
        )
        previous = worker.observation()
        clock.now += 100
        worker._observe(CameraState.READING)
        current = worker.observation()
        self.assertEqual(current.state, CameraState.READING)
        self.assertEqual(current.last_activity_ns, clock.now)
        self.assertEqual(current.last_success_ns, previous.last_success_ns)
        self.assertEqual(previous.state, CameraState.STARTING)

    def test_owner_disposes_mapping_even_if_an_ipc_close_fails(self):
        from unittest.mock import Mock

        from cv_pipeline.transport import TransportOwner

        owner = TransportOwner.__new__(TransportOwner)
        owner._closed = owner._unlinked = False
        owner._memory = Mock()
        bad, good = Mock(), Mock()
        bad.close.side_effect = OSError("failed channel")
        owner._queues = [bad, good]
        with self.assertRaises(RuntimeError):
            owner.close()
        good.close.assert_called_once()
        owner._memory.close.assert_called_once()
        owner._memory.unlink.assert_called_once()
        owner.close()
        self.assertTrue(owner._closed)
