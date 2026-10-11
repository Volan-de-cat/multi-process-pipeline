"""External API boundaries for the unchanged assessment simulators.

Camera construction happens in its owning Capture thread. Only CameraError is
translated into the expected disconnect exception; programming errors propagate.
Inference preserves input order and clocks separate UTC labels from durations.
"""

import math
import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from provided.fake_camera import CameraError, FakeCamera
from provided.fake_inference import inference_batch

from .config import CameraConfig
from .contracts import Camera
from .messages import InferenceResult


class CameraDisconnectedError(Exception):
    """Expected disconnect: discard this camera before retrying."""


class FakeCameraAdapter:
    """Own one supplied camera instance; never share it between threads or processes."""

    def __init__(self, config: CameraConfig) -> None:
        """Create the camera in the calling owner thread.

        The validated FPS and camera ID are passed unchanged to FakeCamera. This
        constructor does not reconnect; CameraWorker creates a replacement when needed.
        """
        self._camera = FakeCamera(camera_id=config.camera_id, fps=config.fps)

    def read(self) -> NDArray[np.uint8]:
        """Return one original 4K uint8 frame.

        Translate only CameraError into CameraDisconnectedError. The underlying instance
        is unusable afterward; RuntimeError and other unexpected errors propagate.
        """
        try:
            return self._camera.read()
        except CameraError as exc:
            raise CameraDisconnectedError(str(exc)) from exc

    def release(self) -> None:
        """Release the underlying instance, including an already-disconnected camera.

        The supplied release operation is idempotent and creates no replacement.
        """
        self._camera.release()


def create_camera(config: CameraConfig) -> Camera:
    """Construct a fresh adapter in the owner thread for initial connection or retry."""
    return FakeCameraAdapter(config)


class FakeInferenceAdapter:
    """Normalize the supplied ordered dictionaries into immutable result messages."""

    def infer(self, frames: Sequence[NDArray[np.uint8]]) -> list[InferenceResult]:
        """Call inference_batch once for the entire independent grayscale batch.

        Validate output length before associating detections with input metadata. Empty
        batches retain the original simulator behavior and return no detections.
        """
        results = inference_batch(list(frames))
        if len(results) != len(frames):
            raise RuntimeError("inference returned an unexpected result count")
        normalized = []
        for result in results:
            if not isinstance(result, dict) or set(result) != {"mean_brightness", "detection"}:
                raise ValueError("invalid inference result schema")
            brightness = result["mean_brightness"]
            if (
                type(brightness) not in (int, float)
                or not math.isfinite(brightness)
                or not 0 <= brightness <= 255
            ):
                raise ValueError("invalid inference brightness")
            if type(result["detection"]) is not bool:
                raise ValueError("inference detection must be boolean")
            normalized.append(InferenceResult(float(brightness), result["detection"]))
        return normalized


class SystemClock:
    """Same-host clocks: UTC for human labels; monotonic nanoseconds for elapsed time."""

    def monotonic_ns(self) -> int:
        """Return a non-wall-clock time base for cross-process latency and deadlines."""
        return time.monotonic_ns()

    def utc_ns(self) -> int:
        """Return receipt label time in Unix UTC nanoseconds; never subtract it for latency."""
        return time.time_ns()
