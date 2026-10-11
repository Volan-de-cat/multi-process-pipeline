"""Spawn-safe fault injection helpers; production has no test-mode switches."""

import sys
import time
from dataclasses import replace

from cv_pipeline import capture, processing, supervisor
from cv_pipeline.adapters import FakeCameraAdapter, FakeInferenceAdapter
from cv_pipeline.config import CameraConfig, PipelineConfig
from cv_pipeline.messages import Stage
from cv_pipeline.runtime import worker_scope

_connections = {}


def reconnect_factory(config):
    camera = FakeCameraAdapter(config)
    attempt = _connections.get(config.camera_id, 0)
    _connections[config.camera_id] = attempt + 1
    if config.camera_id == "cam_01" and attempt == 0:
        camera._camera._disconnect_at = 3
    return camera


class SlowEngine(FakeInferenceAdapter):
    def infer(self, frames):
        time.sleep(0.3)
        return super().infer(frames)


def slow_processing(config, resources):
    processing.FakeInferenceAdapter = SlowEngine
    processing.run_processing(config, resources)


class FailedEngine(FakeInferenceAdapter):
    def infer(self, frames):
        raise ValueError("injected inference error")


class FailedPreprocessor(processing.FramePreprocessor):
    def apply(self, frame, destination=None):
        raise ValueError("injected preprocessing error")


class StalledPreprocessor(processing.FramePreprocessor):
    def apply(self, frame, destination=None):
        time.sleep(60)
        return super().apply(frame, destination)


def failed_overlap_engine(config, resources):
    processing.FakeInferenceAdapter = FailedEngine
    processing.run_processing(config, resources)


def failed_overlap_preprocessor(config, resources):
    processing.FramePreprocessor = FailedPreprocessor
    processing.run_processing(config, resources)


def stalled_overlap_preprocessor(config, resources):
    processing.FramePreprocessor = StalledPreprocessor
    processing.run_processing(config, resources)


class StalledTransport(processing.FrameTransport):
    """Keep Processing alive while its preparer blocks inside descriptor polling."""

    def receive(self, index, timeout):
        time.sleep(60)
        return super().receive(index, timeout)


def stalled_overlap_collector(config, resources):
    processing.FrameTransport = StalledTransport
    processing.run_processing(config, resources)


def reconnect_capture(config, resources):
    capture.create_camera = reconnect_factory
    capture.run_capture(config, resources)


def offline_factory(config):
    """All sources fail expected connection attempts without ever producing frames."""
    from cv_pipeline.adapters import CameraDisconnectedError

    raise CameraDisconnectedError("injected offline source")


def offline_capture(config, resources):
    """Exercise availability failure independently of healthy heartbeat publication."""
    capture.create_camera = offline_factory
    capture.run_capture(config, resources)


def stuck_processing(config, resources):
    with worker_scope(Stage.PROCESSING, resources) as runtime:
        runtime.ready()
        time.sleep(60)  # parent must detect missing cooperative heartbeat


def slow_reporter(config, resources):
    from cv_pipeline import reporter

    serialize = reporter.serialize_result

    def slow_serialize(result, now):
        time.sleep(0.15)
        return serialize(result, now)

    reporter.serialize_result = slow_serialize
    reporter.run_reporter(config, resources)


def noisy_capture(config, resources):
    """Saturate diagnostic output without changing camera/result behavior."""
    import logging
    import threading

    def noise():
        time.sleep(0.5)
        for _ in range(5000):
            logging.info("diagnostic-pressure %s", "x" * 1000)

    threading.Thread(target=noise, daemon=True).start()
    capture.run_capture(config, resources)


if __name__ == "__main__":
    scenario, output = sys.argv[1:]
    config = replace(
        PipelineConfig(),
        output_path=output,
        shutdown_timeout_s=8,
        slots_per_camera=1,
        max_frame_age_ms=250,
        reconnect_initial_s=0.01,
        reconnect_max_s=0.05,
    )
    duration = 2.5
    if scenario == "reconnect":
        supervisor.run_capture = reconnect_capture
    elif scenario == "offline":
        supervisor.run_capture = offline_capture
        config = replace(config, availability_timeout_s=0.7)
    elif scenario == "quota":
        config = replace(config, output_max_bytes=1024)
    elif scenario == "noisy":
        supervisor.run_capture = noisy_capture
    elif scenario == "overload":
        supervisor.run_processing = slow_processing
    elif scenario == "stuck":
        supervisor.run_processing = stuck_processing
        config = replace(config, health_timeout_s=1)
        duration = 5
    elif scenario == "slow-reporter":
        supervisor.run_reporter = slow_reporter
        config = replace(config, result_queue_capacity=1)
        duration = None
    elif scenario == "overlap-engine-error":
        supervisor.run_processing = failed_overlap_engine
        config = replace(config, processing_overlap=True)
    elif scenario == "overlap-preprocess-error":
        supervisor.run_processing = failed_overlap_preprocessor
        config = replace(config, processing_overlap=True)
    elif scenario == "overlap-preprocess-stuck":
        supervisor.run_processing = stalled_overlap_preprocessor
        config = replace(config, processing_overlap=True, health_timeout_s=1)
    elif scenario == "overlap-collector-stuck":
        supervisor.run_processing = stalled_overlap_collector
        config = replace(config, processing_overlap=True, health_timeout_s=1)
    elif scenario == "low-fps":
        config = replace(config, cameras=(CameraConfig("slow", 1),), health_timeout_s=0.7)
    elif scenario == "summary-error":

        class FailedSummary(supervisor.ResultOutput):
            def write(self, text):
                raise OSError("injected summary ENOSPC")

        supervisor.ResultOutput = FailedSummary
    elif scenario == "overlap":
        config = replace(config, processing_overlap=True)
    elif scenario == "latest":
        config = replace(config, slots_per_camera=4, frame_policy="latest")
    elif scenario == "buffered":
        config = replace(config, output_flush_records=16)
    elif scenario == "overlap-slow-reporter":
        supervisor.run_reporter = slow_reporter
        config = replace(config, processing_overlap=True, result_queue_capacity=1)
        duration = None
    elif scenario != "normal":
        raise ValueError(scenario)
    summary = output + ".accounting.json" if scenario == "summary-error" else None
    raise SystemExit(supervisor.PipelineSupervisor(config, duration, summary).run())
