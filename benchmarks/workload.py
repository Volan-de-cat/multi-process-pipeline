"""Spawn-safe performance workloads; fault/slowdown injection is benchmark-only.

The installed production CLI has no test-mode switches. This helper wraps the
same stages while making a camera disconnect or a slow engine/output repeatable.
"""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cv_pipeline import capture, processing, reporter, supervisor
from cv_pipeline.adapters import FakeCameraAdapter, FakeInferenceAdapter
from cv_pipeline.config import CameraConfig, PipelineConfig, load_config
from cv_pipeline.messages import InferenceResult, ProcessedResult
from cv_pipeline.transport import (
    CaptureResources,
    ProcessingResources,
    ReporterResources,
    WorkerResources,
)


class SlowEngine(FakeInferenceAdapter):
    """Add a controlled 200 ms inference penalty per batch."""

    def infer(self, frames: Sequence[NDArray[np.uint8]]) -> list[InferenceResult]:
        """Preserve simulator results while simulating a slower model/device."""
        time.sleep(0.2)
        return super().infer(frames)


def slow_processing(
    config: PipelineConfig, resources: ProcessingResources | WorkerResources
) -> None:
    """Install the slower adapter in the spawned inference owner only."""
    processing.FakeInferenceAdapter = SlowEngine  # type: ignore[misc]  # benchmark-only class injection
    processing.run_processing(config, resources)


def stable_factory(config: CameraConfig) -> FakeCameraAdapter:
    """Keep ordinary trials connected; only explicit fault cases may disconnect."""
    camera = FakeCameraAdapter(config)
    camera._camera._disconnect_at = 2**63 - 1
    return camera


def stable_capture(config: PipelineConfig, resources: CaptureResources | WorkerResources) -> None:
    """Install a deterministic source policy only in the spawned benchmark worker."""
    capture.create_camera = stable_factory
    capture.run_capture(config, resources)


def disconnect_factory(config: CameraConfig) -> FakeCameraAdapter:
    """Disconnect cam_01 on read attempt 30 of each connection without altering supplied code."""
    camera = stable_factory(config)
    if config.camera_id == "cam_01":
        camera._camera._disconnect_at = 30
    return camera


def disconnect_capture(
    config: PipelineConfig, resources: CaptureResources | WorkerResources
) -> None:
    """Use repeatable reconnect pressure inside the camera-owning process."""
    # Remove retry jitter as well as random source faults from this controlled case.
    capture.random.uniform = lambda a, b: b
    capture.create_camera = disconnect_factory
    capture.run_capture(config, resources)


def slow_reporter(config: PipelineConfig, resources: ReporterResources | WorkerResources) -> None:
    """Add 80 ms per result before serialization to model a constrained output."""
    original = reporter.serialize_result

    def serialize(result: ProcessedResult, reported_monotonic_ns: int) -> str:
        """Delay this output operation while preserving result schema and identity."""
        time.sleep(0.08)
        return original(result, reported_monotonic_ns)

    reporter.serialize_result = serialize
    reporter.run_reporter(config, resources)


def main() -> int:
    """Run a selected workload with the normal Supervisor lifecycle and accounting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--scenario",
        choices=("normal", "disconnect", "slow-inference", "slow-output"),
        default="normal",
    )
    args = parser.parse_args()
    supervisor.run_capture = stable_capture
    if args.scenario == "disconnect":
        supervisor.run_capture = disconnect_capture
    elif args.scenario == "slow-inference":
        supervisor.run_processing = slow_processing
    elif args.scenario == "slow-output":
        supervisor.run_reporter = slow_reporter
    return supervisor.PipelineSupervisor(
        load_config(Path(args.config)), args.duration, args.summary
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
