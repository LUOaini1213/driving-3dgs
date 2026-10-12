"""OpenCV codecs with pathlib I/O, including Windows Unicode paths."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    path = Path(path)
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def write_image(path, image, params=None):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image, params or [])
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())
