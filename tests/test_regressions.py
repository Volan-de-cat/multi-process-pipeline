"""Targeted regressions for crash-safe control, sink shutdown and idle budgets."""

import logging
import multiprocessing as mp
import threading
import time
import unittest
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from cv_pipeline.ipc import DatagramChannel, SharedStopSignal
from cv_pipeline.logging_setup import LogListenerOwner
from cv_pipeline.messages import Stage
from cv_pipeline.preprocessing import FramePreprocessor
from cv_pipeline.runtime import StageRuntime
from cv_pipeline.supervisor import HealthMonitor


def wait_forever(stop, ready):
    """Waiter killed mid-poll must not become an Event notification dependency."""
    ready.set()
    stop.wait()


def datagram_producer(channel):
    """Publish until killed; every delivered datagram must remain complete."""
    while True:
        channel.put({"payload": "x" * 1000}, timeout=0.1)


class RegressionTests(unittest.TestCase):
    def test_stop_after_killed_waiter_never_notifies_dead_process(self):
        context = mp.get_context("spawn")
        stop = SharedStopSignal(context)
        ready = SharedStopSignal(context)
        worker = context.Process(target=wait_forever, args=(stop, ready))
        worker.start()
        self.assertTrue(ready.wait(3))
        time.sleep(0.02)
        worker.kill()
        worker.join(timeout=2)
        started = time.monotonic()
        stop.set()
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertTrue(stop.wait(0.1))
        worker.close()

    def test_killed_status_producer_leaves_no_partial_message(self):
        channel = DatagramChannel()
        context = mp.get_context("spawn")
        worker = context.Process(target=datagram_producer, args=(channel,))
        worker.start()
        try:
            self.assertEqual(channel.get(timeout=3)["payload"], "x" * 1000)
            worker.kill()
            worker.join(timeout=2)
            while True:
                try:
                    self.assertEqual(channel.get(timeout=0.02)["payload"], "x" * 1000)
                except Empty:
                    break
            channel.put("still usable", timeout=0.1)
            self.assertEqual(channel.get(timeout=0.1), "still usable")
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=2)
            worker.close()
            channel.close()

    def test_idle_to_active_gets_new_budget_but_heartbeat_cannot_hide_stall(self):
        now = [1_000_000_000]
        resources = SimpleNamespace(health=Queue(), abort=threading.Event())
        with patch("cv_pipeline.runtime.time.monotonic_ns", side_effect=lambda: now[0]):
            runtime = StageRuntime(Stage.REPORTER, resources)
            runtime.ready()
            resources.health.get()
            now[0] += 20_000_000_000
            runtime.tick(idle=True, force=True)
            resources.health.get()
            runtime.tick(force=True)
            monitor = HealthMonitor()
            monitor.observe(resources.health.get())
            self.assertEqual(monitor.check(now[0], 5), [])
            now[0] += 6_000_000_000
            runtime.tick(force=True)
            monitor.observe(resources.health.get())
            self.assertEqual(monitor.check(now[0], 5), [Stage.REPORTER])

    def test_blocked_log_sink_does_not_block_close(self):
        entered = threading.Event()
        release = threading.Event()

        class Stream:
            def write(self, message):
                entered.set()
                release.wait(5)
                return len(message)

            def flush(self):
                pass

        queue = Queue()
        listener = LogListenerOwner(queue, stream=Stream())
        listener.start()
        queue.put(logging.LogRecord("test", logging.INFO, "", 0, "test", (), None))
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        try:
            self.assertFalse(listener.close(timeout_s=0.05))
            self.assertLess(time.monotonic() - started, 0.2)
        finally:
            release.set()
            listener._thread.join(timeout=1)
        self.assertFalse(listener._thread.is_alive())

    def test_area_parallelism_and_scratch_reuse_preserve_results(self):
        generator = np.random.default_rng(12)
        # Actual target geometry exercises the noninteger vertical area ratio.
        frame = generator.integers(0, 256, (2160, 3840, 3), dtype=np.uint8)
        original_threads = cv2.getNumThreads()
        try:
            cv2.setNumThreads(1)
            expected = cv2.cvtColor(
                cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
            )
            cv2.setNumThreads(2)
            preprocessor = FramePreprocessor((640, 480), "RGB")
            actual = preprocessor.apply(frame)
            np.testing.assert_array_equal(actual, expected)
            preserved = actual.copy()
            preprocessor.apply(np.zeros_like(frame))
            np.testing.assert_array_equal(actual, preserved)
        finally:
            cv2.setNumThreads(original_threads)


if __name__ == "__main__":
    unittest.main()
