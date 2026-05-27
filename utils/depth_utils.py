from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def normalize_depth_image(depth_image: np.ndarray) -> np.ndarray:
    depth_image = np.asarray(depth_image)
    if depth_image.ndim == 3:
        depth_image = depth_image[..., 0]

    if depth_image.dtype.kind in {"u", "i"}:
        max_value = np.iinfo(depth_image.dtype).max
        if max_value > 0:
            return depth_image.astype(np.float32) / float(max_value)

    return depth_image.astype(np.float32)


def load_depth_image(path) -> Optional[np.ndarray]:
    depth_image = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        return None
    return normalize_depth_image(depth_image)
