"""
Example: reading from 4 cameras + running inference.

This shows the basic FakeCamera and inference_batch APIs.
Your task is to build a multi-process pipeline around them.

Run: python example.py
"""

import time
from fake_camera import FakeCamera, CameraError
from fake_inference import inference_batch


def main():
    # Setup: 4 cameras at different FPS (could happen in real life)
    camera_configs = [
        {"camera_id": "cam_01", "fps": 15},
        {"camera_id": "cam_02", "fps": 15},
        {"camera_id": "cam_03", "fps": 10},
        {"camera_id": "cam_04", "fps": 10},
    ]

    cameras = [FakeCamera(**cfg) for cfg in camera_configs]

    print(f"Started {len(cameras)} cameras")
    for cam in cameras:
        print(f"  {cam}")

    # Read frames until all cameras disconnect
    active = set(range(len(cameras)))
    total_frames = 0
    start = time.monotonic()

    while active:
        for i in list(active):
            cam = cameras[i]
            try:
                frame = cam.read()
                total_frames += 1

                # Run inference on a single frame (batch of 1).
                # In a real pipeline, you'd collect several frames and pass
                # them together — inference_batch is more efficient per frame.
                results = inference_batch([frame])

                # Print every 100th frame
                if cam.frame_count % 100 == 0:
                    print(
                        f"  [{cam.camera_id}] frame {cam.frame_count}, "
                        f"shape={frame.shape}, "
                        f"size={frame.nbytes / 1024 / 1024:.1f} MB, "
                        f"detection={results[0]['detection']}"
                    )

            except CameraError as e:
                print(f"  [{cam.camera_id}] DISCONNECTED: {e}")
                active.discard(i)

    elapsed = time.monotonic() - start
    print(f"\nDone: {total_frames} frames in {elapsed:.1f}s")
    print(f"Average: {total_frames / elapsed:.1f} FPS across all cameras")


if __name__ == "__main__":
    main()
