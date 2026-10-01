"""Pinhole camera model shared by the metric simulator and the live dynamics estimator.

Coordinates (metres, relative to the user):
    x  lateral, +right
    z  forward range along the walking direction
    y  up, with the ground at y = 0 and the camera at y = height_m

The camera looks forward and is pitched *down* by `pitch_deg` (a phone held at
chest height is rarely level). Image coordinates are normalised to [0, 1]:
u = 0.5 + fx * x' / z',  v = 0.5 + fy * y' / z' with y' pointing down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..types import BBox

# physical (height, width) in metres used to derive distance from a bbox when no depth is available
TYPICAL_SIZE_M = {
    "person": (1.7, 0.5), "chair": (0.9, 0.5), "door": (2.1, 0.9), "car": (1.5, 1.8), "bicycle": (1.0, 0.6),
    "motorcycle": (1.1, 0.8), "bus": (3.0, 2.5), "truck": (2.5, 2.4), "dog": (0.5, 0.4), "trash can": (0.9, 0.5),
    "bench": (0.9, 1.6), "dining table": (0.75, 1.4), "couch": (0.9, 1.8), "backpack": (0.45, 0.35),
    "suitcase": (0.7, 0.45), "potted plant": (0.6, 0.4), "stairs": (1.5, 1.5), "fire hydrant": (0.8, 0.3),
    "stop sign": (0.75, 0.75), "traffic light": (1.0, 0.35), "cone": (0.7, 0.4), "pole": (2.5, 0.2),
    "box": (0.5, 0.5), "cart": (1.0, 0.7), "barrier": (1.0, 1.5), "wet floor sign": (0.65, 0.3),
    "curb": (0.15, 2.0), "elevator": (2.1, 1.0), "scooter": (1.1, 0.4), "text": (0.3, 0.8), "umbrella": (1.0, 1.0),
}
DEFAULT_SIZE_M = (0.8, 0.5)


@dataclass
class CameraModel:
    hfov_deg: float = 70.0
    aspect: float = 4.0 / 3.0       # width / height
    height_m: float = 1.35          # phone held at chest height
    pitch_deg: float = 15.0         # tilted down

    @property
    def fx(self) -> float:
        return 0.5 / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        return self.fx * self.aspect

    # ---- forward projection -------------------------------------------------
    def _tilt(self, y_down: float, z: float, pitch_deg: Optional[float] = None) -> Tuple[float, float]:
        th = math.radians(self.pitch_deg if pitch_deg is None else pitch_deg)
        zc = z * math.cos(th) + y_down * math.sin(th)
        yc = y_down * math.cos(th) - z * math.sin(th)
        return yc, zc

    def project_box(self, x: float, z: float, height: float, width: float, base_height: float = 0.0,
                    pitch_deg: Optional[float] = None, yaw_deg: float = 0.0) -> Optional[BBox]:
        """Normalised bbox of an upright object of the given size whose base sits at `base_height` above ground."""
        if z <= 0.05:
            return None
        yaw = math.radians(yaw_deg)
        # yaw rotates the scene about the vertical axis (camera pointing slightly left/right)
        xr = x * math.cos(yaw) - z * math.sin(yaw)
        zr = x * math.sin(yaw) + z * math.cos(yaw)
        if zr <= 0.05:
            return None
        y_bottom = self.height_m - base_height            # down-positive camera coordinates
        y_top = self.height_m - base_height - height
        yb, zb = self._tilt(y_bottom, zr, pitch_deg)
        yt, zt = self._tilt(y_top, zr, pitch_deg)
        if zb <= 0.05 or zt <= 0.05:
            return None
        v1 = 0.5 + self.fy * yt / zt
        v2 = 0.5 + self.fy * yb / zb
        u1 = 0.5 + self.fx * (xr - width / 2) / zr
        u2 = 0.5 + self.fx * (xr + width / 2) / zr
        return (u1, v1, u2, v2)

    @staticmethod
    def clip(b: BBox) -> Optional[BBox]:
        x1, y1, x2, y2 = max(0.0, b[0]), max(0.0, b[1]), min(1.0, b[2]), min(1.0, b[3])
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            return None
        return (x1, y1, x2, y2)

    # ---- inverse ------------------------------------------------------------
    def lateral_from_u(self, u: float, z: float) -> float:
        return (u - 0.5) * z / self.fx

    def angle_deg_from_u(self, u: float) -> float:
        return math.degrees(math.atan2(u - 0.5, self.fx))

    def distance_from_bbox(self, bbox: BBox, label: str) -> float:
        """Range estimate from projected height and a typical physical height (coarse fallback)."""
        h_norm = max(1e-3, bbox[3] - bbox[1])
        H, _ = TYPICAL_SIZE_M.get(label, DEFAULT_SIZE_M)
        return max(0.3, self.fy * H / h_norm)
