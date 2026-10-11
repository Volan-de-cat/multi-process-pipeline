"""Measure exact area preprocessing and quality-changing faster alternatives.

Compare a real simulator frame with deterministic noise. Median timings exclude
warm-up and errors are measured against resize-then-grayscale INTER_AREA. Faster
linear/two-step resizing is experimental, not adopted pipeline behavior. Restore
OpenCV's global thread setting so importing/running this utility has no lasting
configuration side effect in a caller. Run from an installed Python environment.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from cv_pipeline.preprocessing import FramePreprocessor
from provided.fake_camera import FakeCamera
from provided.fake_inference import inference_batch


def component_profile(frame: np.ndarray, iterations: int) -> list[dict[str, object]]:
    """Time isolated costs, excluding camera pacing, IPC and pipeline contention.

    Host copy uses a preallocated NumPy destination, not a shared-memory lease.
    Generation calls the supplied simulator's generator only to exclude FPS sleep.
    These independent medians must not be summed to predict overlapped throughput.
    """
    camera = FakeCamera("component_benchmark", 1000)
    destination = np.empty_like(frame)
    rgb = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    gray = np.empty((480, 640), dtype=np.uint8)
    operations: dict[str, Callable[[], object]] = {
        "simulator_generation": camera._generate_frame,
        "host_copy": lambda: np.copyto(destination, frame),
        "resize_area": lambda: cv2.resize(frame, (640, 480), dst=rgb, interpolation=cv2.INTER_AREA),
        "grayscale": lambda: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY, dst=gray),
    }
    # Partial binds each batch independently of the loop variable.
    for size in (1, 4, 8, 16):
        operations[f"inference_batch_{size}"] = partial(inference_batch, [gray] * size)
    results: list[dict[str, object]] = []
    try:
        for name, operation in operations.items():
            for _ in range(3):
                operation()
            samples = []
            for _ in range(iterations):
                started = time.perf_counter_ns()
                operation()
                samples.append((time.perf_counter_ns() - started) / 1e6)
            results.append({"component": name, "median_ms": statistics.median(samples)})
    finally:
        camera.release()
    return results


def resized_gray(frame: np.ndarray, interpolation: int, two_step: bool = False) -> np.ndarray:
    """Benchmark allocating resize/conversion; two-step/linear remain experimental."""
    if two_step:
        frame = cv2.resize(frame, (640, 540), interpolation=interpolation)
    return cv2.cvtColor(
        cv2.resize(frame, (640, 480), interpolation=interpolation), cv2.COLOR_RGB2GRAY
    )


def main() -> int:
    """Measure 30 iterations by default for 1/2/4 threads; write timing/error JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("preprocessing-report.json"))
    args = parser.parse_args()
    if not 5 <= args.iterations <= 1000:
        parser.error("measured iterations must be between 5 and 1000")
    camera = FakeCamera("preprocessing_benchmark", 1000)
    try:
        simulator = camera.read()
    finally:
        camera.release()
    inputs = {
        "simulator": simulator,
        "noise": np.random.default_rng(12).integers(0, 256, simulator.shape, dtype=np.uint8),
    }
    results = []
    original_threads = cv2.getNumThreads()
    try:
        for source, frame in inputs.items():
            cv2.setNumThreads(1)
            reference = cv2.cvtColor(
                cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
            )
            for threads in (1, 2, 4):
                cv2.setNumThreads(threads)
                optimized = FramePreprocessor((640, 480), "RGB")
                destination = np.empty((480, 640), dtype=np.uint8)
                methods: dict[str, Callable[[], np.ndarray]] = {
                    "area_original": partial(resized_gray, frame, cv2.INTER_AREA),
                    "area_reused_scratch": partial(optimized.apply, frame),
                    "area_pooled_grayscale": partial(optimized.apply, frame, destination),
                    "two_step_area": partial(resized_gray, frame, cv2.INTER_AREA, True),
                    "linear": partial(resized_gray, frame, cv2.INTER_LINEAR),
                }
                for name, operation in methods.items():
                    for _ in range(3):
                        operation()
                    samples = []
                    for _ in range(args.iterations):
                        started = time.perf_counter_ns()
                        output = operation()
                        samples.append((time.perf_counter_ns() - started) / 1e6)
                    error = np.abs(output.astype(np.int16) - reference.astype(np.int16))
                    results.append(
                        {
                            "source": source,
                            "threads": threads,
                            "method": name,
                            "median_ms": statistics.median(samples),
                            "max_error": int(error.max()),
                            "mean_absolute_error": float(error.mean()),
                            "bitwise_equal": bool(np.array_equal(output, reference)),
                        }
                    )
        cv2.setNumThreads(1)
        components = component_profile(simulator, args.iterations)
    finally:
        cv2.setNumThreads(original_threads)
    report = {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "iterations": args.iterations,
        "output_size": [640, 480],
        "input_shape": list(simulator.shape),
        "measurements": results,
        "component_threads": 1,
        "component_measurements": components,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return (
        0
        if all(
            r["bitwise_equal"]
            for r in results
            if r["method"] in ("area_reused_scratch", "area_pooled_grayscale")
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
