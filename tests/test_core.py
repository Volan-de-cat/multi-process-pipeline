"""Behavioral checks for contracts, clocks, ownership and scheduling."""

import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from cv_pipeline.adapters import CameraDisconnectedError, FakeCameraAdapter, FakeInferenceAdapter
from cv_pipeline.batching import BatchCollector
from cv_pipeline.capture import CameraWorker, ReconnectBackoff
from cv_pipeline.config import CameraConfig, PipelineConfig, load_config, validate_config
from cv_pipeline.messages import (
    EndOfStream,
    FrameDescriptor,
    FrameMetadata,
    HealthEvent,
    InferenceResult,
    Stage,
)
from cv_pipeline.observability import ThroughputWindow
from cv_pipeline.preprocessing import FramePreprocessor
from cv_pipeline.processing import BatchProcessor
from cv_pipeline.reporter import serialize_result
from cv_pipeline.supervisor import HealthMonitor
from cv_pipeline.transport import FrameTransport, TransportOwner


class ManualClock:
    def __init__(self):
        self.now = 1_000_000_000

    def monotonic_ns(self):
        return self.now

    def utc_ns(self):
        return self.now


def descriptor(camera="cam_01", frame_id=1, slot=0, received=1_000_000_000):
    return FrameDescriptor(FrameMetadata(camera, frame_id, 1, 1_000_000_000, received), slot)


class ConfigurationTests(unittest.TestCase):
    def test_automatic_pool_tracks_batch_and_explicit_override(self):
        from dataclasses import replace

        config = load_config()
        self.assertIsNone(config.slots_per_camera)
        self.assertEqual(config.resolved_slots_per_camera, 2)
        self.assertEqual(replace(config, batch_size=8).resolved_slots_per_camera, 3)
        self.assertEqual(replace(config, batch_size=16).resolved_slots_per_camera, 5)
        explicit = replace(config, slots_per_camera=7, memory_budget_mib=1536)
        validate_config(explicit)
        self.assertEqual(explicit.resolved_slots_per_camera, 7)

    def test_defaults_and_json_roundtrip(self):
        config = load_config()
        self.assertEqual([c.fps for c in config.cameras], [15, 15, 10, 10])
        from cv_pipeline.config import config_json

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(config_json(config))
            self.assertEqual(load_config(path), config)

    def test_reject_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text('{"queue_szie": 5}')
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_config(path)

    def test_reject_invalid_values(self):
        for changes in (
            {"batch_size": True},
            {"batch_wait_ms": float("nan")},
            {"slots_per_camera": 0},
            {"color_order": "XYZ"},
            {"cameras": (CameraConfig("same", 15), CameraConfig("same", 10))},
            {"health_timeout_s": 0.1},
            {"frame_shape": (2160.0, 3840, 3)},
            {"reconnect_initial_s": 20},
            {"output_size": (640, -1)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(replace(PipelineConfig(), **changes))


class ImageAndInferenceTests(unittest.TestCase):
    def test_rgb_preprocessing_shape_color_and_independence(self):
        frame = np.zeros((12, 16, 3), np.uint8)
        frame[:, :, 0] = 255
        output = FramePreprocessor((8, 6), "RGB").apply(frame)
        self.assertEqual(output.shape, (6, 8))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.all(output == 76))
        frame.fill(0)
        self.assertTrue(np.all(output == 76))
        self.assertTrue(
            np.all(
                FramePreprocessor((8, 6), "BGR").apply(np.full((12, 16, 3), 255, np.uint8)) == 255
            )
        )

    def test_reject_bad_input(self):
        with self.assertRaises(ValueError):
            FramePreprocessor((8, 6), "RGB").apply(np.zeros((12, 16), np.uint8))

    def test_original_inference_threshold_and_order(self):
        results = FakeInferenceAdapter().infer(
            [np.full((2, 2), 128, np.uint8), np.full((2, 2), 129, np.uint8)]
        )
        self.assertEqual(results, [InferenceResult(128.0, False), InferenceResult(129.0, True)])
        self.assertEqual(FakeInferenceAdapter().infer([]), [])

    def test_camera_disconnect_translation_and_release(self):
        camera = FakeCameraAdapter(CameraConfig("test", 1000))
        camera._camera._disconnect_at = 1  # deterministic injection; reference source unchanged
        with self.assertRaises(CameraDisconnectedError):
            camera.read()
        self.assertTrue(camera._camera.is_released)
        camera.release()


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        config = replace(
            PipelineConfig(), cameras=(CameraConfig("cam_01", 15),), slots_per_camera=1
        )
        self.owner = TransportOwner(config)
        self.resources = self.owner.worker_resources()
        self.transport = FrameTransport(self.resources)

    def tearDown(self):
        self.transport.close()
        self.owner.close()

    def reserve(self):
        # Probe/return a credit before reserving through the transport.
        slot = self.resources.free_slots[0].get(timeout=1)
        self.resources.free_slots[0].put(slot)
        import time

        deadline = time.monotonic() + 1
        while (slot := self.transport.try_reserve(0)) is None:
            if time.monotonic() > deadline:
                self.fail("credit not available")
            time.sleep(0.001)
        return slot

    def test_copy_immutability_credit_and_exception_release(self):
        slot = self.reserve()
        self.assertIsNone(self.transport.try_reserve(0))
        frame = np.full(self.resources.memory.frame_shape, 73, np.uint8)
        self.assertTrue(self.transport.publish(descriptor(slot=slot), frame))
        frame.fill(12)
        message = self.transport.receive(0, 1)
        with (
            self.assertRaisesRegex(RuntimeError, "injected"),
            self.transport.lease(message) as view,
        ):
            self.assertTrue(np.all(view == 73))
            raise RuntimeError("injected")
        del view
        self.assertEqual(self.resources.free_slots[0].get(timeout=1), slot)

    def test_invalid_frame_returns_reserved_slot(self):
        slot = self.reserve()
        with self.assertRaises(ValueError):
            self.transport.publish(descriptor(slot=slot), np.zeros((2, 2, 3), np.uint8))
        self.assertEqual(self.resources.free_slots[0].get(timeout=1), slot)

    def test_invalid_descriptor_is_rejected(self):
        with self.assertRaises(ValueError), self.transport.lease(descriptor(camera="wrong")):
            pass

    def test_invalid_publication_metadata_returns_reserved_slot(self):
        slot = self.reserve()
        with self.assertRaises(ValueError):
            self.transport.publish(
                descriptor(camera="wrong", slot=slot), np.zeros((2, 2, 3), np.uint8)
            )
        self.assertEqual(self.resources.free_slots[0].get(timeout=1), slot)
        with self.assertRaises(ValueError), self.transport.lease(descriptor(slot=99)):
            pass


class SchedulingTests(unittest.TestCase):
    def test_missing_camera_does_not_block_partial_batch(self):
        clock = ManualClock()

        class Abort:
            def is_set(self):
                return False

            def wait(self, seconds):
                clock.now += int(seconds * 1e9)

        class Transport:
            resources = type("Resources", (), {"abort": Abort()})()

            def __init__(self):
                self.items = [descriptor()]

            def receive(self, index, timeout):
                return self.items.pop(0) if index == 0 and self.items else None

        transport = Transport()
        collector = BatchCollector(transport, 2, 4, 50, clock)
        self.assertEqual(collector.collect(), [descriptor()])
        self.assertGreaterEqual(clock.now, 1_050_000_000)

    def test_round_robin_and_eos_flush(self):
        items = {
            0: [descriptor(), EndOfStream(Stage.CAPTURE)],
            1: [descriptor("cam_02", slot=3), EndOfStream(Stage.CAPTURE)],
        }

        class Transport:
            resources = type("Resources", (), {"abort": threading.Event()})()

            def receive(self, index, timeout):
                return items[index].pop(0) if items[index] else None

        collector = BatchCollector(Transport(), 2, 4, 50, ManualClock())
        self.assertEqual([d.metadata.camera_id for d in collector.collect()], ["cam_01", "cam_02"])
        self.assertEqual(collector.collect(), [])

    def test_preprocessing_failure_releases_unvisited_slots(self):
        released = []

        class Transport:
            @contextmanager
            def lease(self, item):
                try:
                    yield np.zeros((2, 2, 3), np.uint8)
                finally:
                    released.append(item.slot_id)

            def release(self, item):
                released.append(item.slot_id)

        class BadPreprocessor:
            def apply(self, frame):
                raise ValueError("bad frame")

        processor = BatchProcessor(
            Transport(), BadPreprocessor(), FakeInferenceAdapter(), ManualClock(), 500
        )
        with self.assertRaises(ValueError):
            processor.process([descriptor(slot=0), descriptor(slot=1)])
        self.assertEqual(released, [0, 1])

    def test_stale_drop_and_full_batch_latency(self):
        clock = ManualClock()
        released = []

        class Transport:
            @contextmanager
            def lease(self, item):
                try:
                    yield np.zeros((2, 2, 3), np.uint8)
                finally:
                    released.append(item.slot_id)

        class Engine:
            def infer(self, frames):
                clock.now += 100_000_000
                return [InferenceResult(0.0, False) for _ in frames]

        processor = BatchProcessor(
            Transport(), FramePreprocessor((2, 2), "RGB"), Engine(), clock, 500
        )
        results = processor.process(
            [descriptor(received=0), descriptor(slot=1), descriptor(slot=2)]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(released, [0, 1, 2])
        self.assertEqual(processor.metrics.snapshot("cam_01").dropped_stale, 1)
        for result in results:
            payload = json.loads(serialize_result(result, clock.now))
            self.assertEqual(payload["processing_latency_ms"], 100)
            self.assertEqual(payload["batch_size"], 2)


class ReliabilityTests(unittest.TestCase):
    def test_backoff_bounded_and_reset(self):
        policy = ReconnectBackoff(0.5, 2)
        delays = [policy.next_delay_s() for _ in range(20)]
        self.assertTrue(0.4 <= delays[0] <= 0.5)
        self.assertTrue(all(1.6 <= d <= 2 for d in delays[2:]))
        policy.reset()
        self.assertTrue(0.4 <= policy.next_delay_s() <= 0.5)

    def test_reconnect_preserves_frame_numbering(self):
        clock = ManualClock()
        stop = threading.Event()
        abort = threading.Event()
        delivered = []
        cameras = []

        class Camera:
            def __init__(self):
                self.reads = 0
                self.released = False

            def read(self):
                self.reads += 1
                if len(cameras) == 1 and self.reads == 2:
                    raise CameraDisconnectedError("injected")
                if len(delivered) == 2:
                    stop.set()
                return np.zeros((2, 2, 3), np.uint8)

            def release(self):
                self.released = True

        class Transport:
            def try_reserve(self, index):
                return 0

            def publish(self, item, frame):
                delivered.append(item.metadata)
                return True

        def factory(config):
            camera = Camera()
            cameras.append(camera)
            return camera

        config = replace(PipelineConfig(), reconnect_initial_s=0.001, reconnect_max_s=0.001)
        worker = CameraWorker(
            config.cameras[0], 0, config, Transport(), stop, abort, clock, factory=factory
        )
        worker.run()
        self.assertEqual([(m.frame_id, m.connection_epoch) for m in delivered], [(1, 1), (2, 2)])
        self.assertTrue(all(c.released for c in cameras))
        self.assertEqual(worker.metrics.snapshot("cam_01").dropped_shutdown, 1)

    def test_heartbeat_does_not_hide_stuck_work(self):
        monitor = HealthMonitor()
        monitor.observe(HealthEvent(Stage.PROCESSING, 10_000_000_000, True, 1_000_000_000))
        self.assertEqual(monitor.check(10_000_000_000, 5), [Stage.PROCESSING])
        monitor.observe(
            HealthEvent(Stage.PROCESSING, 10_000_000_000, True, 1_000_000_000, idle=True)
        )
        self.assertEqual(monitor.check(10_000_000_000, 5), [])
        self.assertEqual(monitor.check(16_000_000_000, 5), [Stage.PROCESSING])

    def test_throughput_requires_complete_ten_second_window(self):
        clock = ManualClock()
        window = ThroughputWindow(clock)
        for _ in range(120):
            window.record("cam_01")
        self.assertIsNone(window.fps())
        clock.now += 10_000_000_000
        self.assertEqual(window.fps(), {"cam_01": 12.0, "total": 12.0})
        self.assertIsNone(window.fps())
        with self.assertRaises(ValueError):
            ThroughputWindow(clock, 9)


if __name__ == "__main__":
    unittest.main()
