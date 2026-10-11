"""Processing-owned resize and grayscale with explicit buffer lifetimes.

INTER_AREA resize precedes conversion as required by the assessment. The color
scratch buffer is reused by one owner, while grayscale outputs remain independent
until batch inference finishes. OpenCV parallelism is configured once per worker,
not by this transformer or by each frame call.
"""

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


class FramePreprocessor:
    """Single-owner transformer with reusable RGB scratch and independent outputs.

    Resize still precedes grayscale, preserving the assessment's exact operation
    order and INTER_AREA anti-aliasing. The intermediate color buffer is reused;
    caller-owned grayscale destinations may also be reused after inference
    completes. Do not share this mutable transformer between Python threads.
    """

    def __init__(self, output_size: tuple[int, int], color_order: str) -> None:
        """Choose RGB/BGR conversion and allocate one output-sized color scratch array.

        output_size is (width, height). The transformer has one caller at a time; batch
        grayscale arrays must not share this reused intermediate storage.
        """
        if color_order not in ("RGB", "BGR"):
            raise ValueError("unsupported color order")
        self._size = output_size
        self._conversion = cv2.COLOR_RGB2GRAY if color_order == "RGB" else cv2.COLOR_BGR2GRAY
        self._resized = np.empty((output_size[1], output_size[0], 3), dtype=np.uint8)

    def apply(
        self, frame: NDArray[np.uint8], destination: NDArray[np.uint8] | None = None
    ) -> NDArray[np.uint8]:
        """Validate HxWx3 uint8 input, area-resize, then return independent grayscale.

        The returned (height, width) array may outlive the raw lease and future apply
        calls. Scratch reuse removes a color allocation but preserves the exact
        resize-before-grayscale sequence; no interpolation quality is traded for speed.
        """
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected an HxWx3 uint8 array")
        # Reusing RGB scratch avoids per-frame color allocation without retaining
        # raw-slot views. Grayscale uses independent allocated or caller-pooled storage.
        resized = cv2.resize(frame, self._size, dst=self._resized, interpolation=cv2.INTER_AREA)
        if destination is not None and (
            destination.dtype != np.uint8
            or destination.shape != self._resized.shape[:2]
            or not destination.flags.c_contiguous
        ):
            raise ValueError("invalid grayscale destination")
        return cast(NDArray[np.uint8], cv2.cvtColor(resized, self._conversion, dst=destination))
