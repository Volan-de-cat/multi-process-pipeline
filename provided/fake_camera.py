"""
FakeCamera — simulates a camera stream for the technical assessment.

Usage:
    from fake_camera import FakeCamera, CameraError

    camera = FakeCamera(camera_id="cam_01", fps=15)
    try:
        frame = camera.read()  # np.ndarray (2160, 3840, 3) uint8
    except CameraError:
        # Camera disconnected — create a new instance to reconnect
        camera = FakeCamera(camera_id="cam_01", fps=15)
"""

import time
import random
import numpy as np


class CameraError(Exception):
    """Raised when the camera disconnects."""
    pass


class FakeCamera:
    """
    Simulates a camera that produces 4K frames at a given FPS.

    Periodically raises CameraError to simulate disconnection.
    After CameraError, the instance is dead — create a new one to "reconnect".

    Attributes:
        camera_id: Unique identifier for this camera.
        fps: Target frames per second.
        width: Frame width in pixels
        height: Frame height in pixels
        frame_count: Number of frames successfully produced.
    """

    WIDTH = 3840
    HEIGHT = 2160
    CHANNELS = 3

    def __init__(self, camera_id: str, fps: int = 15) -> None:
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        self.camera_id = camera_id
        self.fps = fps
        self.width = self.WIDTH
        self.height = self.HEIGHT
        self.frame_count = 0

        self._frame_interval = 1.0 / fps
        self._released = False
        self._last_frame_time = time.monotonic()

        # Disconnect happens after a random number of frames (200-800)
        self._disconnect_at = random.randint(200, 800)

        # Deterministic color per camera for visual distinction
        self._color_seed = hash(camera_id) % 256

    def read(self) -> np.ndarray:
        """
        Read a single frame.

        Returns:
            np.ndarray of shape (2160, 3840, 3), dtype uint8.

        Raises:
            CameraError: Camera disconnected. Create a new instance.
            RuntimeError: Called after release().
        """
        if self._released:
            raise RuntimeError(
                f"Camera {self.camera_id} has been released. "
                f"Create a new instance to reconnect."
            )

        # Respect FPS timing
        now = time.monotonic()
        elapsed = now - self._last_frame_time
        sleep_time = self._frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_frame_time = time.monotonic()

        self.frame_count += 1

        # Simulate disconnect
        if self.frame_count >= self._disconnect_at:
            self._released = True
            raise CameraError(
                f"Camera {self.camera_id} disconnected after "
                f"{self.frame_count} frames."
            )

        return self._generate_frame()

    def release(self) -> None:
        self._released = True

    @property
    def is_released(self) -> bool:
        return self._released

    def _generate_frame(self) -> np.ndarray:
        frame = np.zeros(
            (self.HEIGHT, self.WIDTH, self.CHANNELS),
            dtype=np.uint8,
        )

        # Smooth hue rotation using sine waves
        phase = self.frame_count * 0.02 + self._color_seed
        x = np.linspace(0, np.pi * 2, self.WIDTH, dtype=np.float32)
        for c in range(self.CHANNELS):
            wave = np.sin(x + phase + c * (2 * np.pi / 3))
            row = ((wave + 1) * 0.5 * 255).astype(np.uint8)
            frame[:, :, c] = row[np.newaxis, :]

        return frame

    def __repr__(self) -> str:
        status = "released" if self._released else "active"
        return (
            f"FakeCamera(id={self.camera_id!r}, fps={self.fps}, "
            f"frames={self.frame_count}, status={status})"
        )

    def __del__(self) -> None:
        if getattr(self, "_released", True) is False:
            self.release()
