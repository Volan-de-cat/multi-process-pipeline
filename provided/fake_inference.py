"""
fake_inference.py — simulates GPU batch inference for the technical assessment.

This represents a single GPU running a CV model. Inference latency depends on
the batch size:

    batch_time = BASE_SEC + PER_FRAME_SEC * len(frames)

Larger batches are more efficient PER FRAME (the fixed GPU overhead — memory
transfer, kernel launch — is amortized across the batch). But the speedup has
diminishing returns: per-frame GPU compute still scales with the amount of
data, so total throughput caps at roughly 2x no matter how large the batch.

    Batch  1:  40ms  ->  25.0 fps
    Batch  4: 100ms  ->  40.0 fps   (1.6x)
    Batch  8: 180ms  ->  44.4 fps   (1.78x)
    Batch 16: 340ms  ->  47.1 fps   (1.88x)

Usage:
    from fake_inference import inference_batch

    results = inference_batch(frames)   # frames: list of numpy arrays
    # Pass a single-element list to run inference on one frame: [frame]
"""

import time
from typing import List

import numpy as np

# Fixed GPU overhead per call (memory transfer + kernel launch), amortized
# across the batch.
BASE_SEC = 0.020

# Per-frame GPU compute — the floor that batching cannot avoid. Because this
# equals BASE_SEC, the maximum batch speedup approaches 2x.
PER_FRAME_SEC = 0.020


def inference_batch(frames: List[np.ndarray]) -> List[dict]:
    """
    Run simulated GPU inference on a batch of frames.

    Latency grows with batch size: ``BASE_SEC + PER_FRAME_SEC * len(frames)``.
    Batching is more efficient per frame, but with diminishing returns —
    throughput tops out around 2x, just like a real GPU.

    Args:
        frames: list of numpy arrays. Pass ``[frame]`` to run inference on a
                single frame (batch of 1).

    Returns:
        list of detection result dicts, one per input frame.
    """
    if not frames:
        return []

    # Simulated GPU time: fixed overhead + per-frame compute.
    time.sleep(BASE_SEC + PER_FRAME_SEC * len(frames))

    return [
        {
            "mean_brightness": float(frame.mean()),
            "detection": bool(frame.mean() > 128),
        }
        for frame in frames
    ]
