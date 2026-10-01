"""Per-track metric dynamics: distance, lateral position, relative speed, time-to-contact,
and the probability that the object will be inside the walking corridor when the user reaches it.

Distance comes from a depth model (Detection.distance_m) when present, otherwise from the
projected height and a typical object size (low confidence). Velocities are least-squares
slopes over the last ~1.2 s of the track so single-frame jitter does not produce bogus TTCs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..types import Detection, Track, bbox_center
from .camera import CameraModel, TYPICAL_SIZE_M, DEFAULT_SIZE_M


@dataclass
class DynamicsConfig:
    corridor_half_m: float = 0.45       # half-width of the walking corridor the user occupies
    contact_m: float = 0.6              # range at which an object in the corridor is "reached"
    history_s: float = 2.0              # window for velocity estimation
    speed_ema: float = 0.35             # smoothing of the velocity estimates between frames
    min_samples: int = 4
    min_approach_mps: float = 0.12      # slower than this is "not approaching"
    default_user_speed: float = 1.0     # assumed walking speed when only a bbox guess exists (m/s)
    lateral_sigma_m: float = 0.25       # base uncertainty of the predicted lateral position at arrival
    lateral_sigma_per_s: float = 0.12   # grows with TTC
    depth_ema: float = 0.5


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _slope(points, idx: int) -> float:
    """Least-squares slope of points[i][idx] against points[i][0] (time)."""
    n = len(points)
    if n < 2:
        return 0.0
    tm = sum(p[0] for p in points) / n
    vm = sum(p[idx] for p in points) / n
    num = sum((p[0] - tm) * (p[idx] - vm) for p in points)
    den = sum((p[0] - tm) ** 2 for p in points)
    return num / den if den > 1e-9 else 0.0


class TrackDynamics:
    def __init__(self, camera: Optional[CameraModel] = None, cfg: Optional[DynamicsConfig] = None):
        self.cam = camera or CameraModel()
        self.cfg = cfg or DynamicsConfig()

    def update(self, trk: Track, det: Optional[Detection], t: float) -> None:
        """Refresh the metric state of `trk` after it was matched to `det` at time t."""
        if det is not None and det.distance_m is not None and det.depth_conf > 0:
            z_meas, conf = float(det.distance_m), float(det.depth_conf)
        else:
            z_meas, conf = self.cam.distance_from_bbox(trk.bbox, trk.label), 0.25
        if trk.distance_m is None or trk.depth_conf <= 0:
            z = z_meas
        else:
            a = self.cfg.depth_ema
            z = a * z_meas + (1 - a) * trk.distance_m
        cx, _ = bbox_center(trk.bbox)
        x = self.cam.lateral_from_u(cx, z)
        trk.distance_m, trk.depth_conf, trk.lateral_m = z, conf, x
        trk.dist_history.append((t, z, x))
        # drop samples older than the estimation window
        while len(trk.dist_history) > 2 and t - trk.dist_history[0][0] > self.cfg.history_s:
            trk.dist_history.popleft()
        pts = list(trk.dist_history)
        if len(pts) >= self.cfg.min_samples:
            a = self.cfg.speed_ema
            vz, vx = _slope(pts, 1), _slope(pts, 2)
            trk.rel_speed_mps = a * vz + (1 - a) * trk.rel_speed_mps if trk.rel_speed_mps != 0.0 else vz
            trk.lateral_speed_mps = a * vx + (1 - a) * trk.lateral_speed_mps if trk.lateral_speed_mps != 0.0 else vx
        self._ttc_and_path(trk)

    def _ttc_and_path(self, trk: Track) -> None:
        c = self.cfg
        z, x = trk.distance_m or 0.0, trk.lateral_m
        _, w = TYPICAL_SIZE_M.get(trk.label, DEFAULT_SIZE_M)
        approach = -trk.rel_speed_mps
        if len(trk.dist_history) < c.min_samples or trk.depth_conf < 0.4:
            # no reliable velocity yet: assume the user walks toward a static object
            approach = c.default_user_speed
        if approach >= c.min_approach_mps:
            ttc = max(0.0, (z - c.contact_m) / approach)
        else:
            ttc = float("inf")
        trk.ttc_s = ttc
        horizon = min(ttc, 6.0) if math.isfinite(ttc) else 6.0
        x_arr = x + trk.lateral_speed_mps * horizon
        sigma = c.lateral_sigma_m + c.lateral_sigma_per_s * horizon + (0.2 if trk.depth_conf < 0.4 else 0.0)
        half = c.corridor_half_m + w / 2
        # P(|x_arr| < half) under a Gaussian around the predicted lateral position
        p = _phi((half - x_arr) / sigma) - _phi((-half - x_arr) / sigma)
        if not math.isfinite(ttc):
            p *= 0.5    # not converging on the user: only matters if they walk into it
        # near-term crossing: the object sweeps through the corridor close in front of the user
        vx = trk.lateral_speed_mps
        if abs(vx) > 0.3 and x * vx < 0:          # moving toward the centre line
            tau = -x / vx                           # seconds until it is straight ahead
            z_then = z + trk.rel_speed_mps * tau
            if 0.0 <= tau <= 2.5 and z_then < 3.5:
                p = max(p, 0.95 - 0.1 * tau)
        trk.path_prob = max(0.0, min(1.0, p))

    @staticmethod
    def direction_deg(trk: Track) -> float:
        z = trk.distance_m or 1.0
        return math.degrees(math.atan2(trk.lateral_m, max(0.1, z)))
