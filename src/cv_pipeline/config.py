"""Immutable, validated configuration for the supplied 4K workload.

Loading accepts partial JSON overrides and rejects unknown fields. Validation
runs before resource creation; direct dataclass construction remains lightweight.
Timing values use seconds unless the field explicitly ends in _ms. output_size
uses OpenCV width/height order, unlike NumPy frame_shape height/width/channels.
"""

import json
import math
import os
import platform
import re
import stat
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from .resources import cgroup_memory
from .wire import finite_float, reject_constant, unique_fields


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Immutable identity/FPS; validate through load_config or Supervisor before use."""

    camera_id: str
    fps: int


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable workload and runtime budgets; defaults match the original four cameras."""

    cameras: tuple[CameraConfig, ...] = field(
        default_factory=lambda: (
            CameraConfig("cam_01", 15),
            CameraConfig("cam_02", 15),
            CameraConfig("cam_03", 10),
            CameraConfig("cam_04", 10),
        )
    )
    frame_shape: tuple[int, int, int] = (2160, 3840, 3)
    color_order: str = "RGB"
    output_size: tuple[int, int] = (640, 480)
    slots_per_camera: int | None = None
    result_queue_capacity: int = 64
    log_queue_capacity: int = 256
    batch_size: int = 4
    batch_wait_ms: float = 50.0
    max_frame_age_ms: float = 500.0
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 10.0
    shutdown_timeout_s: float = 5.0
    health_timeout_s: float = 5.0
    startup_timeout_s: float = 15.0
    output_path: str | None = None
    preprocessing_threads: int = 2
    memory_budget_mib: int = 1024
    output_max_bytes: int = 256 * 1024 * 1024
    output_max_files: int = 1
    output_durable: bool = False
    availability_timeout_s: float = 30.0
    processing_overlap: bool = False
    frame_policy: str = "fifo"
    output_flush_records: int = 1
    output_flush_interval_ms: float = 100.0

    def __post_init__(self) -> None:
        """Snapshot caller-owned sequences so validated settings cannot change later."""
        for name in ("cameras", "frame_shape", "output_size"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            object.__setattr__(self, name, tuple(value))

    @property
    def resolved_slots_per_camera(self) -> int:
        """Resolve raw buffering independently of the three processing stages.

        Auto mode provides each camera its rounded-up share of a full batch plus
        one capture credit while Processing holds collected inputs. Explicit positive
        counts remain supported for deliberate latency/memory tuning.
        """
        if self.slots_per_camera is not None:
            return self.slots_per_camera
        return (self.batch_size + len(self.cameras) - 1) // len(self.cameras) + 1


def validate_config(config: PipelineConfig) -> None:
    """Apply independent camera, geometry, buffering, timing and resource policies."""
    _validate_cameras(config)
    _validate_geometry(config)
    _validate_buffers(config)
    _validate_timings(config)
    _validate_output(config)
    _validate_memory(config)


def _validate_cameras(config: PipelineConfig) -> None:
    """Check the Linux target, camera identities, FPS and OpenCV thread count."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("this implementation requires Linux/x86_64")
    if not config.cameras or any(not isinstance(c, CameraConfig) for c in config.cameras):
        raise ValueError("cameras must be a non-empty list of camera configurations")
    if len(config.cameras) > 32:
        raise ValueError("at most 32 cameras are supported by the bounded status protocol")
    if type(config.preprocessing_threads) is not int or not 1 <= config.preprocessing_threads <= 8:
        raise ValueError("preprocessing_threads must be an integer between 1 and 8")
    ids = [c.camera_id for c in config.cameras]
    if any(
        not isinstance(i, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", i) is None for i in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("camera_id values must be unique ASCII identifiers of 1 to 64 characters")
    if "total" in ids:
        raise ValueError("camera_id total is reserved for aggregate throughput")
    for camera in config.cameras:
        if type(camera.fps) is not int or not 1 <= camera.fps <= 1000:
            raise ValueError("camera fps must be an integer between 1 and 1000")


def _validate_geometry(config: PipelineConfig) -> None:
    """Check fixed simulator input and bounded output image dimensions."""
    # The supplied camera cannot emit other geometry without changing its implementation.
    if config.frame_shape != (2160, 3840, 3) or any(type(v) is not int for v in config.frame_shape):
        raise ValueError("the supplied FakeCamera requires frame_shape [2160, 3840, 3]")
    if config.color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    if len(config.output_size) != 2 or any(
        type(v) is not int or not 1 <= v <= 3840 for v in config.output_size
    ):
        raise ValueError("output_size must contain integer width and height between 1 and 3840")


def _validate_buffers(config: PipelineConfig) -> None:
    """Validate queue capacities and memory/output budgets before allocation."""
    if config.slots_per_camera is not None and (
        type(config.slots_per_camera) is not int or config.slots_per_camera <= 0
    ):
        raise ValueError("slots_per_camera must be a positive integer or null (automatic)")
    for name in ("result_queue_capacity", "log_queue_capacity", "batch_size"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, limit in (("result_queue_capacity", 1024), ("log_queue_capacity", 4096)):
        if getattr(config, name) > limit:
            raise ValueError(f"{name} exceeds {limit}")
    if type(config.memory_budget_mib) is not int or not 64 <= config.memory_budget_mib <= 65536:
        raise ValueError("memory_budget_mib must be an integer between 64 and 65536")
    if type(config.output_max_bytes) is not int or not 1024 <= config.output_max_bytes <= 2**40:
        raise ValueError("output_max_bytes must be between 1024 and 2**40")
    if type(config.output_durable) is not bool:
        raise ValueError("output_durable must be boolean")
    if type(config.processing_overlap) is not bool:
        raise ValueError("processing_overlap must be boolean")
    if config.frame_policy not in ("fifo", "latest"):
        raise ValueError("frame_policy must be fifo or latest")
    if type(config.output_flush_records) is not int or not 1 <= config.output_flush_records <= 1024:
        raise ValueError("output_flush_records must be an integer between 1 and 1024")
    if config.batch_size > 16:
        raise ValueError("batch_size must not exceed 16 for bounded latency")


def _validate_timings(config: PipelineConfig) -> None:
    """Bound all durations and allow the simulator inference budget."""
    for name in (
        "batch_wait_ms",
        "max_frame_age_ms",
        "reconnect_initial_s",
        "reconnect_max_s",
        "shutdown_timeout_s",
        "health_timeout_s",
        "startup_timeout_s",
        "availability_timeout_s",
        "output_flush_interval_ms",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.001 <= value <= 3600
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite and between 0.001 and 3600")
    if config.reconnect_initial_s > config.reconnect_max_s:
        raise ValueError("initial reconnect delay exceeds maximum delay")
    inference_s = 0.020 + 0.020 * config.batch_size
    if config.health_timeout_s <= inference_s + config.batch_wait_ms / 1000 + 0.5:
        raise ValueError("health timeout must allow a batch wait and inference with headroom")
    if config.availability_timeout_s <= max(1 / c.fps for c in config.cameras) + 0.5:
        raise ValueError(
            "availability timeout must allow the slowest camera interval with headroom"
        )


def _validate_output(config: PipelineConfig) -> None:
    """Validate optional output path syntax; secure opening is handled by ResultOutput."""
    if type(config.output_max_files) is not int or not 1 <= config.output_max_files <= 1024:
        raise ValueError("output_max_files must be an integer between 1 and 1024")
    if config.output_max_files > 1 and config.output_path is None:
        raise ValueError("output rotation requires an owned output_path")
    if config.output_path is not None and (
        not isinstance(config.output_path, str) or not config.output_path.strip()
    ):
        raise ValueError("output_path must be a non-empty path or null")


def _validate_memory(config: PipelineConfig) -> None:
    """Reject estimated raw, temporary, batch and process memory beyond deployment limits."""
    raw = 2160 * 3840 * 3
    pixels = config.output_size[0] * config.output_size[1]
    estimate = raw * (
        len(config.cameras) * config.resolved_slots_per_camera + 2 * len(config.cameras)
    )
    estimate += pixels * (3 + config.batch_size * (3 if config.processing_overlap else 1))
    estimate += config.result_queue_capacity * 4096 + 192 * 2**20
    estimate += config.log_queue_capacity * 2048
    estimate = int(estimate * 1.1)
    if estimate > config.memory_budget_mib * 2**20:
        raise ValueError("estimated pipeline memory exceeds memory_budget_mib")
    for observation in cgroup_memory():
        remaining = observation["limit_bytes"] - observation["current_bytes"]
        if estimate > remaining:
            raise ValueError("estimated pipeline memory exceeds available cgroup memory")


def load_config(path: Path | None = None, *, output_path: str | None = None) -> PipelineConfig:
    """Load defaults or a partial UTF-8 JSON object and validate the result.

    Apply an optional CLI output override before validating rotation constraints.
    Convert JSON lists into immutable tuples. Unknown fields and malformed camera
    objects raise ValueError; file errors retain OSError for CLI handling.
    """
    if path is None:
        config = PipelineConfig()
    else:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("configuration must be a regular file")
            payload = source.read(65537)
        if len(payload) > 65536:
            raise ValueError("configuration file exceeds 64 KiB")
        try:
            data = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=unique_fields,
                parse_float=finite_float,
                parse_constant=reject_constant,
            )
        except (RecursionError, UnicodeError) as exc:
            raise ValueError("configuration is too deeply nested or not UTF-8") from exc
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a JSON object")
        unknown = set(data) - {f.name for f in fields(PipelineConfig)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        if "cameras" in data:
            if not isinstance(data["cameras"], list):
                raise ValueError("cameras must be a JSON array")
            try:
                data["cameras"] = tuple(CameraConfig(**c) for c in data["cameras"])
            except TypeError as exc:
                raise ValueError("each camera needs exactly camera_id and fps") from exc
        for name in ("frame_shape", "output_size"):
            if name in data:
                if not isinstance(data[name], list):
                    raise ValueError(f"{name} must be a JSON array")
                data[name] = tuple(data[name])
        config = PipelineConfig(**data)
    if output_path is not None:
        config = replace(config, output_path=output_path)
    validate_config(config)
    return config


def config_json(config: PipelineConfig) -> str:
    """Serialize a validated configuration into readable JSON suitable for round-tripping."""
    return json.dumps(asdict(config), indent=2)
