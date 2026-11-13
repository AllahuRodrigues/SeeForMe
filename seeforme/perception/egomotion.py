"""Camera ego-motion between consecutive frames (CPSC 480 A: L6 Shi-Tomasi corners, L9 alignment, L11 RANSAC).

Per frame pair: Shi-Tomasi corners in the earlier frame, tracked into the later one with pyramidal Lucas-Kanade
(forward-backward checked), then a RANSAC homography. A homography is exact only for a rotating camera
(H = K R K^-1) or a planar scene (L9 p20), so:

* the rotation K^-1 H K gives yaw and pitch rates (valid when translation parallax is small);
* what is left after H on the inliers (residual parallax) is an image-based walking proxy;
* the inlier ratio and the spatial spread of inliers say whether the fit can be trusted (covered lens, a wall
  filling the view, a sudden scene change).

Sign conventions (camera x right, y down, z forward): yaw_rate > 0 when the camera turns right (the scene moves
left), pitch_rate > 0 when the camera tilts up (the scene moves down). Pixel quantities are at the working width
(640 px). Every output is finite; when the fit is degenerate the sample is all zeros with reliable = False.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geometry.camera import CameraModel

WORK_WIDTH = 640
MAX_CORNERS = 300
QUALITY = 0.01
MIN_DIST = 10
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FB_MAX_PX = 1.0             # forward-backward consistency of an LK track
RANSAC_PX = 3.0             # reprojection threshold at 640 px
MIN_FIT = 8                 # fewer matches than this: no homography at all
GRID = 4
STILL_PX_S = 8.0            # median raw flow below this (px/s at 640 wide, about 0.9 deg/s) = camera still
RELIABLE_INLIER, RELIABLE_COVERAGE, RELIABLE_MATCHES = 0.5, 0.4, 30


def ransac_iterations(p: float = 0.99, w: float = 0.5, s: int = 4) -> int:
    """Samples needed so that, with probability p, at least one s-point sample is all inliers (L11): N = log(1-p) / log(1-w^s)."""
    if w >= 1.0:
        return 1
    return int(math.ceil(math.log(1.0 - p) / math.log(1.0 - w ** s)))


# iteration budget: enough for the lowest inlier ratio worth fitting (0.4) at p = 0.999; OpenCV stops earlier
# once its own adaptive estimate of w allows.
RANSAC_ITERS = ransac_iterations(p=0.999, w=0.4, s=4)


def intrinsics(width: int, height: int, hfov_deg: Optional[float] = None) -> np.ndarray:
    """Pixel K from the shared CameraModel (normalised fx = 0.5 / tan(hfov / 2), square pixels, centred principal point)."""
    cam = CameraModel() if hfov_deg is None else CameraModel(hfov_deg=hfov_deg)
    f = cam.fx * width
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def prepare_frame(frame: np.ndarray, width: int = WORK_WIDTH) -> np.ndarray:
    """BGR or gray frame -> gray uint8 at `width` pixels wide."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = g.shape[:2]
    if w != width:
        g = cv2.resize(g, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return g


def fit_homography(p0: np.ndarray, p1: np.ndarray, thresh: float = RANSAC_PX) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC homography p0 -> p1 (4-point samples, then refit on inliers). Returns (H or None, inlier mask)."""
    n = len(p0)
    if n < MIN_FIT:
        return None, np.zeros(n, bool)
    Hm, mask = cv2.findHomography(np.asarray(p0, np.float32), np.asarray(p1, np.float32), cv2.RANSAC, thresh,
                                  maxIters=RANSAC_ITERS, confidence=0.995)
    if Hm is None or not np.all(np.isfinite(Hm)):
        return None, np.zeros(n, bool)
    return Hm, mask.ravel().astype(bool)


def rotation_from_homography(Hm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Nearest rotation to K^-1 H K (scale removed, projected onto SO(3) by SVD)."""
    M = np.linalg.inv(K) @ Hm @ K
    d = np.linalg.det(M)
    if not np.isfinite(d) or abs(d) < 1e-12:
        return np.eye(3)
    M = M / np.cbrt(d)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _transfer(Hm: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(p0, np.float32).reshape(-1, 1, 2), Hm).reshape(-1, 2)


@dataclass
class EgoSample:
    """One frame pair. FIELDS is the cached row; H and the matches are kept for per-box queries."""
    n_matches: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    yaw_rate: float = 0.0          # deg/s, + = camera turns right
    pitch_rate: float = 0.0        # deg/s, + = camera tilts up
    parallax: float = 0.0          # median inlier residual after H, px/s (walking proxy)
    resid_all: float = 0.0         # median residual after H over all matches, px/s
    flow: float = 0.0              # median raw flow, px/s
    camera_still: bool = False
    reliable: bool = False
    H: Optional[np.ndarray] = field(default=None, repr=False)
    p0: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32), repr=False)
    p1: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32), repr=False)
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool), repr=False)
    size: Tuple[int, int] = (WORK_WIDTH, 480)     # (width, height) of the frames

    FIELDS: ClassVar[Tuple[str, ...]] = ("n_matches", "inlier_ratio", "coverage", "yaw_rate", "pitch_rate", "parallax",
                                         "resid_all", "flow", "camera_still", "reliable")

    def as_row(self) -> list:
        return [float(getattr(self, k)) for k in self.FIELDS]


def estimate_pair(prev: np.ndarray, cur: np.ndarray, dt: float, K: np.ndarray) -> EgoSample:
    """Ego-motion from frame `prev` to frame `cur` (gray uint8, same size), `dt` seconds apart."""
    h, w = cur.shape[:2]
    out = EgoSample(size=(w, h))
    dt = max(float(dt), 1e-3)
    c = cv2.goodFeaturesToTrack(prev, MAX_CORNERS, QUALITY, MIN_DIST, blockSize=7)
    if c is None or len(c) < MIN_FIT:
        out.n_matches = 0 if c is None else len(c)
        return out
    c = c.astype(np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, c, None, **LK_PARAMS)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK_PARAMS)
    fb = np.linalg.norm((c - back).reshape(-1, 2), axis=1)
    p1r = nxt.reshape(-1, 2)
    inside = (p1r[:, 0] >= 0) & (p1r[:, 0] < w) & (p1r[:, 1] >= 0) & (p1r[:, 1] < h)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_MAX_PX) & inside
    p0, p1 = c.reshape(-1, 2)[ok], p1r[ok]
    out.n_matches, out.p0, out.p1 = int(len(p0)), p0, p1
    if len(p0) < MIN_FIT:
        out.inliers = np.zeros(len(p0), bool)
        return out
    Hm, inl = fit_homography(p0, p1)
    out.inliers = inl
    out.flow = float(np.median(np.linalg.norm(p1 - p0, axis=1))) / dt
    out.camera_still = bool(out.flow < STILL_PX_S)
    if Hm is None or inl.sum() < 4:
        return out
    out.H = Hm
    res = np.linalg.norm(p1 - _transfer(Hm, p0), axis=1)
    out.inlier_ratio = float(inl.mean())
    gx = np.clip((p1[inl, 0] * GRID / w).astype(int), 0, GRID - 1)
    gy = np.clip((p1[inl, 1] * GRID / h).astype(int), 0, GRID - 1)
    out.coverage = len(set(zip(gx.tolist(), gy.tolist()))) / float(GRID * GRID)
    rv = cv2.Rodrigues(rotation_from_homography(Hm, K))[0].ravel()
    out.yaw_rate = -math.degrees(float(rv[1])) / dt
    out.pitch_rate = -math.degrees(float(rv[0])) / dt
    out.parallax = float(np.median(res[inl])) / dt
    out.resid_all = float(np.median(res)) / dt
    out.reliable = bool(out.inlier_ratio >= RELIABLE_INLIER and out.coverage >= RELIABLE_COVERAGE
                        and out.n_matches >= RELIABLE_MATCHES)
    vals = [out.yaw_rate, out.pitch_rate, out.parallax, out.resid_all, out.flow]
    if not all(math.isfinite(v) for v in vals):          # never hand NaN downstream
        return EgoSample(n_matches=out.n_matches, size=(w, h))
    return out


def box_residual(sample: EgoSample, box: Sequence[float]) -> Tuple[float, int]:
    """Residual motion inside a normalised box (x1, y1, x2, y2) of the later frame: median |x' - pi(H x)| in px
    (at the working width, per frame pair) over all tracked corners whose later position lies in the box.
    Large for an object moving against the scene. Returns (residual, n_points); (0.0, 0) if nothing to measure."""
    if sample.H is None or len(sample.p1) == 0:
        return 0.0, 0
    w, h = sample.size
    x1, y1, x2, y2 = box
    p1 = sample.p1
    m = (p1[:, 0] >= x1 * w) & (p1[:, 0] <= x2 * w) & (p1[:, 1] >= y1 * h) & (p1[:, 1] <= y2 * h)
    if not m.any():
        return 0.0, 0
    res = np.linalg.norm(p1[m] - _transfer(sample.H, sample.p0[m]), axis=1)
    return float(np.median(res)), int(m.sum())


def egomotion_series(frames: Sequence[np.ndarray], times: Sequence[float], K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Samples for consecutive pairs; the sample at times[k] uses frames k-1 and k only. Returns (t, rows)."""
    ts, rows = [], []
    for k in range(1, len(frames)):
        s = estimate_pair(frames[k - 1], frames[k], times[k] - times[k - 1], K)
        ts.append(times[k])
        rows.append(s.as_row())
    return np.asarray(ts, float), np.asarray(rows, float).reshape(-1, len(EgoSample.FIELDS))


# ---- decision-level causal features ------------------------------------------------------------------------------

_SIG = ("yaw_abs", "still", "parallax", "inlier")
DECISION_FEATURES = ([f"ego_{k}" for k in _SIG] + ["ego_reliable"] + [f"ego_{k}_max3s" for k in _SIG]
                     + [f"ego_{k}_mean3s" for k in _SIG] + ["ego_reliable_frac3s"])


def decision_features(t: np.ndarray, rows: np.ndarray, at: float, window_s: float = 3.0) -> np.ndarray:
    """Features at decision time `at` from samples with t <= at only: the last sample and the max / mean over
    the previous `window_s` seconds. Yaw magnitude, still flag and parallax of unreliable samples are set to 0;
    the inlier ratio is kept (it is itself the reliability evidence) and the reliability indicator is added."""
    out = np.zeros(len(DECISION_FEATURES))
    i = int(np.searchsorted(t, at + 1e-6)) - 1
    if i < 0 or len(rows) == 0:
        return out
    j0 = int(np.searchsorted(t, at - window_s))
    F = rows[j0:i + 1]
    ix = {k: EgoSample.FIELDS.index(k) for k in EgoSample.FIELDS}
    rel = F[:, ix["reliable"]]
    sig = np.column_stack([np.abs(F[:, ix["yaw_rate"]]) * rel, F[:, ix["camera_still"]] * rel,
                           F[:, ix["parallax"]] * rel, F[:, ix["inlier_ratio"]]])
    return np.concatenate([sig[-1], [rel[-1]], sig.max(0), sig.mean(0), [rel.mean()]])
