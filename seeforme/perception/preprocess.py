"""Frame preprocessing and image-quality estimation.

Blind-user imagery is often blurry, dark, or badly framed (VizWiz, Chiu et al.).
We resize for speed and compute cheap quality indicators that the pipeline logs
and that can lower the effective confidence of predictions on bad frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class FrameQuality:
    blur_score: float      # variance of Laplacian; low = blurry
    brightness: float      # mean intensity 0..255
    is_blurry: bool
    is_dark: bool

    @property
    def confidence_scale(self) -> float:
        """Multiplier applied to detection confidence when the frame is degraded."""
        s = 1.0
        if self.is_blurry:
            s *= 0.85
        if self.is_dark:
            s *= 0.85
        return s


def resize_for_inference(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def assess_quality(frame: np.ndarray, blur_threshold: float = 60.0, dark_threshold: float = 50.0) -> FrameQuality:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    small = cv2.resize(gray, (320, int(320 * gray.shape[0] / gray.shape[1])))
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    bright = float(small.mean())
    return FrameQuality(blur_score=blur, brightness=bright, is_blurry=blur < blur_threshold, is_dark=bright < dark_threshold)


def enhance_dark(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel for low-light frames."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
