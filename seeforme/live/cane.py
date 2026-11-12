"""A white cane and a guide dog made of geometry: class-agnostic obstacles from the metric depth map.

The COCO detector cannot name walls, doors, poles, posts, stairs, kerbs or barriers, so the scheduler never announces
them. This layer looks only at the Depth Anything metric depth map (metres) and asks what a cane and a guide dog ask.

    analyze(depth, cfg=None, prior=None) -> CaneFrame        one depth map, stateless (server/app.py uses this)
    Cane(cfg).update(depth, t, tracks) -> CaneFrame          every new depth map (LiveController.on_frame)
    Cane.propose(t, moving, busy, read_mode, units) -> Cue   at a decision step: the cane cue to say now, or None
    Cane.commit(cue, t)                                      the controller said it: remember it

Floor. A plane n.P = H in camera coordinates has an inverse depth that is *linear* in the normalised image
coordinates: 1/Z = (n_x x + n_y y + n_z) / H. So the floor is fitted by RANSAC on 1/Z over the lower half of the image,
which gives the camera height H, pitch and roll without knowing any of them (a chest-height phone is 1.2-1.6 m up and
never level). Guards: candidates must look like floor locally (inverse depth growing down the image, whatever the depth
scale; a wall facing the camera does not), hypotheses must point down (pitch within 80 degrees: a user looking down at
the phone sees only floor; roll within 30 degrees), the inliers must span 8% of the rows, and the floor of the last 3 s
is kept while it explains the floor in view nearly as well (a sliver of floor in front of a wall otherwise admits a
slanted plane through the corner). A camera more than 1.9 m up means the depth scale is off (the outdoor model reads
about 4x too far up close): the map is rescaled to the nominal 1.4 m. No fit: the recent floor, else a frontal wall
(walls are vertical, so it gives the pitch; a "wall" whose normal points down is a floor seen steeply and is rejected:
read as a wall, it put an obstacle 1.4 m ahead of every user looking down), else a prior (1.4 m, 20 degrees down; no
levels, lower confidence). A fitted floor under 0.9 m below the camera is a table, an exhibit panel or a seated user's
lap: not a walking view, and the cane says nothing.

Cane: surfaces that rise out of the floor in the walking corridor, from the ground to body height. A floor's inverse
depth grows steadily down the image (d(1/Z)/dy = n_y / H); a vertical surface's barely changes. A grid cell is an
obstacle when it is part of a vertical run of rising cells (gradient below 0.45 of the floor's: the pitch-tolerant ratio
test) that climbs at least 0.12 m more steeply than 1:2, or when it stands 0.5 m or more above the floor right below it
in its column and is not floor-like. Heights are measured from that local floor, not the fitted plane: the depth model
bends a floor by decimetres far away. Nearer than 0.3 m is the user's own hands or what they hold; above 1.95 m is
clearance (door lintels). Sectors are metric strips on the floor: ahead = |lateral| <= 0.35 m (the cane's sweep: body
width and a margin), left / right out to 1.25 m. Per sector: the nearest band (0.3 m deep) holding at least 0.025 m^2
of obstacle surface and reaching 0.15 m above its floor is the obstacle; its distance (10th percentile forward range),
level (ground: top at most 0.45 m; head: bottom at least 1.0 m with the floor below it in view; body otherwise; an edge
the frame cuts off is unknown, so never head or ground on that side), confidence (range and depth spread x surface area
x floor quality) and image box; clear_m is the obstacle's distance, the 4 m horizon when the sector was seen without
one, None when it was not seen.

Guide dog: what a cane cannot do. Head-height obstacles over a clear floor ("Something at head height ahead"); path
choice when the corridor is blocked within 1.5 m and one side was seen open at least 1 m further (never inferred from
silence): the cue reports only what blocks ("Obstacle ahead, about 1 meter, and on your right"), never that a side is
clear or clearer, since drop-offs are not seen; intelligent disobedience: a haptic "Stop." while walking when
something of at least 0.04 m^2 ahead will be within 0.6 m by the next decision (0.7 s at its closing speed; a walker
covers 0.5 m between decisions), once per obstacle, and only if the user walked up to it (0.3 m closer within 3 s,
still closing at 0.25 m/s or more: a seat back in a car, a wall the user stands at or a noisy floor patch never
triggers it); selective attention: quiet about an obstacle a detected track explains (same image region and
distance: the scheduler speaks for it), about static surroundings while standing still, and about something within
reach the user never walked up to (what they are handling). Nothing here ever says or implies that the way is clear.

Speaking. An obstacle must persist in at least 3 depth maps spanning at least 0.5 s (per sector, its range may change
by 0.35 m + 1.5 m/s between maps: the user walks) with confidence 0.3. Identity memory "obstacle:<sector>:<level>"
(ground and body share one: the level of one obstacle may flip between maps): said once, then held until it is closer
by max(0.35 m, 15%), or 10 s pass while walking (no repeat within 2.5 s unless it enters the 1 m band); forgotten when
not seen for 2 s or when it reappears clearly farther (a new obstacle); nothing for 3 s after a stop (the buzz said
it). While still, only an approaching obstacle speaks (its range fell by max(0.35 m, 15%) within 2 s). Spoken cues are
about the corridor ahead only: a wall alongside a corridor is always "on your left", and the sides serve path choice.
`clear_m` is for the web app's display and path choice; it is never to be read out as "clear". note() puts the
confirmed obstacle ahead into words for answers ("is the path clear?" never ignores a wall the depth map shows).

Known gaps: drop-offs (a kerb or stairs going down) are below the floor, not above it, and are not detected; the near
floor (about 1.5 m for a chest-height camera pitched 15 degrees) is outside the view, so a low obstacle is seen while
it is still farther away, not when the cane would touch it; a level camera that sees only a thin strip of far floor
can get the pitch wrong by 10 degrees (about 10% on range); the depth model's scale and the floor are monocular
estimates.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from ..cues.compose import distance_phrase, speech_seconds
from ..types import BBox, Cue

SECTORS = ("left", "ahead", "right")
LEVELS = ("ground", "body", "head")
CANE_TRACK_ID = -7            # Cue.track_id of cane cues: never a tracker id (those start at 1)
CANE_POLICY = "cane"


@dataclass
class CaneConfig:
    hfov_long_deg: float = 70.0      # field of view across the longer image side (square pixels; phone or webcam)
    grid_rows: int = 96              # the depth map is sampled (nearest, so no mixed depths at edges) to about this many
    grid_cols: int = 96              # ... rows and columns
    min_depth_m: float = 0.15
    max_depth_m: float = 20.0
    # floor fit (RANSAC on inverse depth over the lower part of the image)
    floor_from: float = 0.5          # candidates: rows below this fraction of the image height
    floor_max_m: float = 40.0        # ... nearer than this (raw map units: the outdoor model reads 4x far up close)
    floor_tol: float = 0.06          # inlier: |1/Z - plane| <= floor_tol / Z (6% of the depth)
    floor_iters: int = 64
    floor_min_inliers: int = 60
    floor_min_frac: float = 0.25     # ... and this share of the candidates
    floor_keep_frac: float = 0.7     # the recent floor is kept while it has this share of a new fit's inliers
    floor_min_rows: float = 0.08     # inliers must span this share of the grid rows
    floor_min_grad: float = 0.4      # candidates: relative vertical inverse-depth gradient above this (a floor: n_y Z / H,
                                     # >= 0.6 in the lower half of the image for pitch <= 50 deg; a wall facing it: <= 0)
    max_tilt_deg: float = 80.0       # the floor normal is at most this far from the camera's down axis (pitch and roll)
    max_roll_deg: float = 30.0
    wall_max_up_deg: float = 4.0     # a frontal "wall" (no floor fit) implies a camera pitched up at most this much: a
                                     # steeper normal pointing down is a floor seen from above, not a wall
    fit_height_m: Tuple[float, float] = (0.6, 8.0)      # a fit outside this is not the floor
    max_camera_height_m: float = 1.9  # a fit higher than this: the depth scale is off, rescale to nominal_height_m
    nominal_height_m: float = 1.4
    prior_pitch_deg: float = 20.0    # no fit yet: camera this far down at nominal_height_m
    reuse_floor_s: float = 3.0       # no fit in this map: reuse the last fit this long
    # obstacle cells
    corridor_half_m: float = 0.35    # ahead: |lateral| <= this (body half-width and a margin: the cane's sweep)
    side_m: float = 1.25             # left / right: out to this lateral offset
    horizon_m: float = 4.0
    min_range_m: float = 0.3         # nearer than this is the user's own hands, lap or what they hold, not an obstacle
    tall_m: float = 0.5              # higher than this above the floor below it (and not floor-like): an obstacle
    rise_ratio: float = 0.45         # vertical inverse-depth gradient below this x the floor's: a rising surface ...
    min_rise_m: float = 0.12         # ... in a vertical run of such cells rising at least this much
    clearance_m: float = 1.95        # above this: overhead clearance (door lintels), not an obstacle
    head_bottom_m: float = 1.0       # an obstacle whose bottom is at least this high (seen, not cut off) is head level
    ground_top_m: float = 0.45       # ... whose top is at most this high (seen) is ground level
    band_m: float = 0.3              # an obstacle is the nearest band this deep ...
    min_area_m2: float = 0.025       # ... holding at least this much obstacle surface (a 5 cm pole 0.5 m tall; not the
                                     # user's own white cane, a strap or tactile paving) ...
    min_top_m: float = 0.15          # ... and reaching at least this high above the floor below it (else the next band)
    min_seen_cells: int = 20         # a sector is "seen" with at least this many valid cells inside the horizon
    # guide dog (single frame)
    stop_m: float = 0.6              # something this near ahead: stop
    blocked_m: float = 1.5           # path choice when the corridor is blocked within this ...
    clearer_by_m: float = 1.0        # ... and a side is clear at least this much further (and at least 2 m)
    # speaking (Cane.propose)
    speak_m: float = 2.0             # obstacles ahead within this are said
    near_m: float = 1.0              # entering this band may repeat sooner than min_gap_s
    persist_maps: int = 3            # depth maps an obstacle must persist in ...
    persist_s: float = 0.5           # ... spanning at least this long
    max_walk_mps: float = 1.5        # its range may change by 0.35 m + this x dt between maps
    lost_s: float = 0.75             # a sector track without a hit this long is reset
    closer_m: float = 0.35           # identity memory: say again when closer by max(closer_m, closer_frac x d)
    closer_frac: float = 0.15
    remind_s: float = 10.0           # ... or after this long while walking
    min_gap_s: float = 2.5           # ... but not sooner than this (unless it entered the near_m band)
    forget_s: float = 2.0            # an identity not seen this long is forgotten (the next sighting is new)
    approach_s: float = 2.0          # while still: approaching = range fell by max(closer_m, closer_frac x d) in this window
    stop_gap_s: float = 3.0          # at most one stop cue in this window
    stop_area_m2: float = 0.04       # a stop needs this much surface (a pole, a door, a person; not a hand or a strap) ...
    stop_seen_m: float = 0.3         # ... that came at least this much closer within the last 3 s (the user walked up to
                                     # it: a seat back or a wall the user stands at does not trigger it) ...
    stop_lookahead_s: float = 0.7    # ... and will be within stop_m in this long at its closing speed (one decision + the
                                     # depth map's age) ...
    stop_rearm_s: float = 2.0        # ... once per obstacle: re-armed after nothing was within near_m for this long
    stop_min_mps: float = 0.25       # ... and only while it closes in at least this fast (least squares over 1.2 s): a
                                     # noisy floor patch whose range jumps about is not a walker closing on something
    speak_conf: float = 0.3          # obstacle confidence needed to speak
    walk_min_height_m: float = 0.9   # a fitted "floor" nearer below the camera than this is a table, a panel or a seated
                                     # user's lap: not a walking view, the cane says nothing
    explain_overlap: float = 0.3     # a track explains an obstacle: this share of the obstacle box inside the track box ...
    explain_m: float = 0.5           # ... and the track's distance within max(explain_m, explain_frac x d)
    explain_frac: float = 0.3
    stale_s: float = 1.0             # no depth map this long: the cane says nothing


@dataclass
class Floor:
    normal: Tuple[float, float, float]   # unit "down" normal in camera coordinates (x right, y down, z forward)
    height_m: float                      # camera height above the floor, after rescaling
    scale: float                         # factor applied to the depth map (1 = trusted as metric)
    source: str                          # fit | reused (a recent fit) | wall (a frontal wall, walls are vertical) | prior
    inliers: int = 0

    @property
    def pitch_deg(self) -> float:
        return math.degrees(math.atan2(self.normal[2], self.normal[1]))

    @property
    def roll_deg(self) -> float:
        return math.degrees(math.atan2(self.normal[0], self.normal[1]))


@dataclass
class Obstacle:
    sector: str
    distance_m: float
    lateral_m: float
    level: str
    confidence: float
    bbox: BBox
    area_m2: float
    bottom_m: Optional[float]            # None: cut off by the frame edge
    top_m: Optional[float]
    explained: bool = False              # a detected track covers it (set by Cane.update)


@dataclass
class CaneFrame:
    clear_m: Dict[str, Optional[float]]
    obstacles: Dict[str, Optional[Obstacle]]
    floor: Optional[Floor]
    ms: float = 0.0

    @property
    def obstacle(self) -> Optional[Obstacle]:
        """The nearest obstacle in the walking corridor, else the nearest on either side."""
        if self.obstacles.get("ahead") is not None:
            return self.obstacles["ahead"]
        side = [o for o in (self.obstacles.get("left"), self.obstacles.get("right")) if o is not None]
        return min(side, key=lambda o: o.distance_m) if side else None

    def suggest(self, cfg: Optional["CaneConfig"] = None) -> Optional[str]:
        """stop: something within stop_m ahead; left / right: the corridor is blocked within blocked_m and that side
        was seen clear at least clearer_by_m further (the clearer side); None otherwise (never "go ahead"), and in a view
        that is not a walking view (a fitted floor less than walk_min_height_m below the camera)."""
        c = cfg or CaneConfig()
        ahead = self.obstacles.get("ahead")
        fl = self.floor
        if ahead is None or (fl is not None and fl.source == "fit" and fl.height_m < c.walk_min_height_m):
            return None                     # nothing ahead, or not a walking view (a table, a panel, seated)
        if ahead.distance_m <= c.stop_m:
            return "stop"
        if ahead.distance_m > c.blocked_m or ahead.level == "head":
            return None
        return clearer_side(self.clear_m.get("left"), self.clear_m.get("right"), ahead.distance_m, c)

    def as_contract(self, cfg: Optional["CaneConfig"] = None) -> dict:
        """The "cane" field of server/app.py POST /api/depth (directions are in the camera image)."""
        o = self.obstacle
        r = lambda v: None if v is None else round(float(v), 2)   # noqa: E731
        return {"clear_m": {s: r(self.clear_m.get(s)) for s in SECTORS},
                "obstacle": None if o is None else {"sector": o.sector, "distance_m": r(o.distance_m), "level": o.level,
                                                    "confidence": round(float(o.confidence), 2),
                                                    "bbox": [round(float(v), 4) for v in o.bbox]},
                "suggest": self.suggest(cfg)}


def clearer_side(left: Optional[float], right: Optional[float], d_ahead: float, cfg: CaneConfig) -> Optional[str]:
    need = max(2.0, d_ahead + cfg.clearer_by_m)
    ok = {s: v for s, v in (("left", left), ("right", right)) if v is not None and v >= need}
    if not ok:
        return None
    if len(ok) == 2 and abs(ok["left"] - ok["right"]) < 0.25:
        return None                  # both equally open: no preference worth saying
    return max(ok, key=lambda s: ok[s])


# ---- single depth map ----------------------------------------------------------------------------------------------
_RNG_SEED = 1234


def _grid(depth: np.ndarray, cfg: CaneConfig):
    h, w = depth.shape[:2]
    sy, sx = max(1, math.ceil(h / cfg.grid_rows)), max(1, math.ceil(w / cfg.grid_cols))
    rows, cols = np.arange(sy // 2, h, sy), np.arange(sx // 2, w, sx)
    z = np.asarray(depth, dtype=np.float32)[np.ix_(rows, cols)]
    f = 0.5 * max(h, w) / math.tan(math.radians(cfg.hfov_long_deg) / 2)
    x = ((cols + 0.5 - w / 2) / f).astype(np.float32)
    y = ((rows + 0.5 - h / 2) / f).astype(np.float32)
    return z, x, y, rows, cols, (h, w), (sx / f) * (sy / f)


def _plane_ok(n: np.ndarray, hgt: np.ndarray, cfg: CaneConfig) -> np.ndarray:
    return ((n[..., 1] >= math.cos(math.radians(cfg.max_tilt_deg))) & (np.abs(n[..., 0]) <= math.sin(math.radians(cfg.max_roll_deg)))
            & (hgt >= cfg.fit_height_m[0]) & (hgt <= cfg.fit_height_m[1]))


def _ransac_plane(X: np.ndarray, wv: np.ndarray, tol: np.ndarray, ok_fn, iters: int,
                  extra: Optional[np.ndarray] = None) -> Optional[Tuple[np.ndarray, int]]:
    """RANSAC + least-squares refit of 1/Z = p . (x, y, 1) under a constraint on the plane: (p, inliers) or None."""
    rng = np.random.default_rng(_RNG_SEED)
    idx = rng.integers(0, len(wv), size=(iters, 3))
    A, bvec = X[idx], wv[idx]
    good = np.abs(np.linalg.det(A)) > 1e-9
    hyps = np.zeros((iters, 3))
    if good.any():
        hyps[good] = np.linalg.solve(A[good], bvec[good][..., None])[..., 0]
    if extra is not None:
        hyps = np.vstack([hyps, extra[None, :]])
        good = np.append(good, True)
    good &= ok_fn(hyps)
    if not good.any():
        return None
    counts = (np.abs(X @ hyps[good].T - wv[:, None]) <= tol[:, None]).sum(0)
    best = hyps[good][int(np.argmax(counts))]
    for _ in range(2):                                               # least-squares refit on the inliers
        inl = np.abs(X @ best - wv) <= tol
        if inl.sum() < 3:
            return None
        best = np.linalg.lstsq(X[inl], wv[inl], rcond=None)[0]
    if not bool(ok_fn(best[None, :])[0]):
        return None
    return best, int((np.abs(X @ best - wv) <= tol).sum())


def _samples(z: np.ndarray, x: np.ndarray, y: np.ndarray, rows: np.ndarray, cfg: CaneConfig, max_m: float,
             keep: Optional[np.ndarray] = None):
    zz = z[rows]
    xx = np.broadcast_to(x[None, :], zz.shape)
    yy = np.broadcast_to(y[rows][:, None], zz.shape)
    ok = np.isfinite(zz) & (zz > cfg.min_depth_m) & (zz < max_m)
    if keep is not None:
        ok &= keep[rows]
    X = np.stack([xx[ok], yy[ok], np.ones(int(ok.sum()))], 1).astype(np.float64)
    return X, 1.0 / zz[ok].astype(np.float64)


def fit_floor(z: np.ndarray, x: np.ndarray, y: np.ndarray, cfg: CaneConfig,
              prior: Optional[Tuple[np.ndarray, float]] = None) -> Optional[Tuple[np.ndarray, float, int]]:
    """RANSAC plane on inverse depth over the lower image rows, pointing down (pitch and roll limits) at a plausible
    height: (unit down normal, camera height, inliers) in the depth map's own scale, or None."""
    # candidates must look like floor locally: inverse depth growing down the image, (d(1/Z)/dy) / (1/Z) = n_y Z / H > 0
    # whatever the depth scale, while a wall facing the camera gives <= 0 and a side wall ~0. Without this a sliver of
    # floor in front of a wall is fitted by a slanted plane through the corner.
    with np.errstate(invalid="ignore", divide="ignore"):
        w = 1.0 / z
        floorish = np.zeros_like(w, bool)
        k = 3                                                         # +-3 rows of 5-row means: noise-robust
        if w.shape[0] > 2 * k + 4:
            sm = w.copy()
            sm[2:-2] = (w[:-4] + w[1:-3] + w[2:-2] + w[3:-1] + w[4:]) / 5.0
            g = (sm[2 * k:] - sm[:-2 * k]) / (y[2 * k:] - y[:-2 * k])[:, None]
            floorish[k:-k] = (g / w[k:-k]) > cfg.floor_min_grad
    X, wv = _samples(z, x, y, y > (y[0] + (y[-1] - y[0]) * cfg.floor_from), cfg, cfg.floor_max_m, floorish)
    if len(wv) < cfg.floor_min_inliers:
        return None
    cos_t, sin_r = math.cos(math.radians(cfg.max_tilt_deg)), math.sin(math.radians(cfg.max_roll_deg))

    def ok(p: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(p, axis=1)
        n = p / np.maximum(nrm, 1e-12)[:, None]
        h = 1.0 / np.maximum(nrm, 1e-12)
        return (nrm > 1e-9) & (n[:, 1] >= cos_t) & (np.abs(n[:, 0]) <= sin_r) & (h >= cfg.fit_height_m[0]) & (h <= cfg.fit_height_m[1])
    tol = cfg.floor_tol * wv
    extra = None if prior is None else np.asarray(prior[0], np.float64) / prior[1]
    res = _ransac_plane(X, wv, tol, ok, cfg.floor_iters, extra)
    if res is None or res[1] < max(cfg.floor_min_inliers, cfg.floor_min_frac * len(wv)):
        return None
    p, k = res
    if extra is not None and bool(ok(extra[None, :])[0]):
        # temporal consistency: the recent floor still explains the floor in view nearly as well. A sliver of floor in
        # front of a wall admits a slanted plane through the corner; the floor seen a moment ago does not drift.
        k_prior = int((np.abs(X @ extra - wv) <= tol).sum())
        if k_prior >= cfg.floor_keep_frac * k:
            p, k = extra, k_prior
    inl = np.abs(X @ p - wv) <= tol
    if len(np.unique(X[inl, 1])) < cfg.floor_min_rows * len(y):
        return None                                                   # too thin a strip of floor to trust its slope
    nb = float(np.linalg.norm(p))
    return p / nb, 1.0 / nb, k


def fit_wall(z: np.ndarray, x: np.ndarray, y: np.ndarray, cfg: CaneConfig) -> Optional[Tuple[np.ndarray, float]]:
    """No floor in view: is the lower image one surface facing the camera within 4 m (a wall, a door, a cupboard)?
    Walls are vertical, so its normal gives the pitch (roll taken as zero): (down normal, wall distance) or None."""
    X, wv = _samples(z, x, y, y > (y[0] + (y[-1] - y[0]) * 0.3), cfg, cfg.horizon_m)
    if len(wv) < cfg.floor_min_inliers:
        return None

    def ok(p: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(p, axis=1)
        m = p / np.maximum(nrm, 1e-12)[:, None]
        return ((nrm > 1e-9) & (m[:, 2] >= math.cos(math.radians(50))) & (np.abs(m[:, 0]) <= 0.5)
                & (m[:, 1] <= math.sin(math.radians(cfg.wall_max_up_deg))) & (1.0 / np.maximum(nrm, 1e-12) <= cfg.horizon_m))
    res = _ransac_plane(X, wv, cfg.floor_tol * wv, ok, cfg.floor_iters)
    if res is None or res[1] < 0.5 * len(wv):
        return None
    p, _ = res
    m = np.array([0.0, p[1], p[2]])
    m /= np.linalg.norm(m)
    return np.array([0.0, m[2], -m[1]]), 1.0 / float(np.linalg.norm(p))


def _prior_floor(cfg: CaneConfig) -> Floor:
    p = math.radians(cfg.prior_pitch_deg)
    return Floor((0.0, math.cos(p), math.sin(p)), cfg.nominal_height_m, 1.0, "prior")


def _rising_runs(rising: np.ndarray, height: np.ndarray, d_fwd: np.ndarray, min_rise: float) -> np.ndarray:
    """Cells of vertical runs (one column, consecutive rows, single-row gaps filled) of rising cells whose height
    extent is at least min_rise and that rise more steeply than about 1:2 (forward spread <= 2 x extent + 0.15 m:
    a far floor whose depth flattens out is not a wall)."""
    r = rising.copy()
    if r.shape[0] > 2:
        r[1:-1] |= rising[:-2] & rising[2:] & np.isfinite(height[1:-1])
    nr, nc = r.shape
    flat = r.T.ravel()
    if not flat.any():
        return np.zeros_like(r)
    prev = np.concatenate([[False], flat[:-1]])
    start = flat & (~prev | (np.tile(np.arange(nr), nc) == 0))
    ids = np.cumsum(start) * flat
    sel = ids > 0
    first = np.flatnonzero(start[sel])
    hs, ds = height.T.ravel()[sel], d_fwd.T.ravel()[sel]
    ext = np.maximum.reduceat(hs, first) - np.minimum.reduceat(hs, first)
    spread = np.maximum.reduceat(ds, first) - np.minimum.reduceat(ds, first)
    good = np.concatenate([[False], (ext >= min_rise) & (spread <= 2.0 * ext + 0.15)])
    return good[ids].reshape(nc, nr).T


def _floor_below(floorlike: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Per cell, the height of the nearest floor below it in its column (two floor-like rows in a row), else NaN.
    Obstacle heights are measured from there: the fitted plane can be off by decimetres far away (a curved floor in
    the depth map), the floor right below an object is not."""
    sure = floorlike.copy()
    sure[:-1] &= floorlike[1:]
    f = np.where(sure, height, np.nan)
    ref = np.full_like(height, np.nan)
    last = np.full(height.shape[1], np.nan)
    for v in range(height.shape[0] - 1, -1, -1):
        last = np.where(np.isfinite(f[v]), f[v], last)
        ref[v] = last
    return ref


def analyze(depth: np.ndarray, cfg: Optional[CaneConfig] = None, prior: Optional[Floor] = None) -> CaneFrame:
    """Obstacles, clear distance per sector and the floor from one metric depth map (metres, any resolution).
    `prior`: a recent floor to fall back on (and to try as a RANSAC hypothesis); None uses the fixed prior."""
    c = cfg or CaneConfig()
    t0 = time.perf_counter()
    z, x, y, _rows, _cols, (hh, ww), pix = _grid(depth, c)
    valid = np.isfinite(z) & (z > c.min_depth_m) & (z < c.max_depth_m)
    if valid.sum() < c.min_seen_cells:
        return CaneFrame({s: None for s in SECTORS}, {s: None for s in SECTORS}, None, (time.perf_counter() - t0) * 1000)
    zs = np.where(valid, z, np.nan)
    pr = None
    if prior is not None and prior.source == "fit":
        pr = (np.asarray(prior.normal), prior.height_m / prior.scale)      # back to this map's raw scale
    fit = fit_floor(zs, x, y, c, pr)
    if fit is not None:
        n, h_raw, k_in = fit
        # a camera more than max_camera_height_m up: the depth scale is off (the outdoor model can be 4x too far up
        # close); lower is believed (a seated user, a child, a wheelchair)
        scale = c.nominal_height_m / h_raw if h_raw > c.max_camera_height_m else 1.0
        floor = Floor(tuple(float(v) for v in n), float(h_raw * scale), float(scale), "fit", k_in)
    elif prior is not None and prior.source == "fit":
        floor = Floor(prior.normal, prior.height_m, prior.scale, "reused")
    else:
        wall = fit_wall(zs, x, y, c)
        floor = (Floor(tuple(float(v) for v in wall[0]), c.nominal_height_m, 1.0, "wall") if wall is not None
                 else _prior_floor(c))
    n = np.asarray(floor.normal, np.float64)
    H = floor.height_m
    zm = zs * floor.scale                                            # metres, rescaled
    X3, Y3 = x[None, :] * zm, y[:, None] * zm
    height = H - (n[0] * X3 + n[1] * Y3 + n[2] * zm)
    fwd = np.array([0.0, 0.0, 1.0]) - n[2] * n
    fwd /= max(1e-9, np.linalg.norm(fwd))
    lat_ax = np.cross(n, fwd)
    d_fwd = fwd[0] * X3 + fwd[1] * Y3 + fwd[2] * zm
    lat = lat_ax[0] * X3 + lat_ax[1] * Y3 + lat_ax[2] * zm
    # rising-surface ratio: vertical inverse-depth gradient against the floor's (n_y / H): ~1 on the floor, ~0 on a
    # vertical surface, > 1 on a raised horizontal one; huge either way across a depth jump (not a surface)
    winv = np.where(valid, 1.0 / np.maximum(zm, 1e-3), np.nan)
    sm = winv.copy()
    if winv.shape[0] > 2:
        sm[1:-1] = (winv[:-2] + winv[1:-1] + winv[2:]) / 3.0            # an invalid neighbour makes it NaN: not rising
    grad = np.gradient(sm, y, axis=0) if sm.shape[0] > 1 else np.zeros_like(sm)
    rho = grad / max(1e-6, n[1] / H)
    area = (zm ** 2) * pix
    with np.errstate(invalid="ignore"):
        floorlike = valid & (rho >= 0.6) & (rho <= 1.6)
        ref = _floor_below(floorlike, height)
        has_ref = np.isfinite(ref)
        rel = height - np.where(has_ref, ref, 0.0)                   # height above the floor right below
        surface = valid & (rho > -2.5)
        runs = _rising_runs(surface & (rho < c.rise_ratio), np.where(valid, height, np.nan), np.where(valid, d_fwd, np.nan),
                            c.min_rise_m)
        inside = valid & (d_fwd >= c.min_range_m) & (d_fwd <= c.horizon_m) & (np.abs(lat) <= c.side_m)
        ob = inside & (rel <= c.clearance_m) & (runs | ((rel >= c.tall_m) & ~floorlike & surface))
    floor_q = {"fit": 1.0, "reused": 0.8, "wall": 0.75, "prior": 0.55}[floor.source]
    bands = {"left": (-c.side_m, -c.corridor_half_m), "ahead": (-c.corridor_half_m, c.corridor_half_m),
             "right": (c.corridor_half_m, c.side_m)}
    clear: Dict[str, Optional[float]] = {}
    obstacles: Dict[str, Optional[Obstacle]] = {}
    nr, nc = z.shape
    for s, (lo_l, hi_l) in bands.items():
        in_s = (lat >= lo_l) & (lat <= hi_l) if s == "ahead" else ((lat >= lo_l) & (lat < hi_l) if s == "left" else (lat > lo_l) & (lat <= hi_l))
        seen = int((inside & in_s).sum())
        m = ob & in_s
        o = _nearest(m, d_fwd, lat, rel, has_ref, zm, area, nr, nc, s, c, floor_q, floor.source != "prior") if m.any() else None
        obstacles[s] = o
        clear[s] = o.distance_m if o is not None else (c.horizon_m if seen >= c.min_seen_cells else None)
    return CaneFrame(clear, obstacles, floor, (time.perf_counter() - t0) * 1000)


def _nearest(m: np.ndarray, d_fwd, lat, rel, has_ref, zm, area, nr: int, nc: int, sector: str, c: CaneConfig,
             floor_q: float, heights_known: bool = True) -> Optional[Obstacle]:
    d = d_fwd[m]
    order = np.argsort(d)
    ds, a = d[order], area[m][order]
    cum = np.concatenate([[0.0], np.cumsum(a)])
    j = np.searchsorted(ds, ds + c.band_m, side="right")
    tot = cum[j] - cum[np.arange(len(ds))]
    cl = None
    lo = -np.inf
    for _ in range(6):                  # the nearest band with enough surface that also stands up from the floor
        hit = np.nonzero((tot >= c.min_area_m2) & (ds > lo))[0]
        if not len(hit):
            return None
        d0 = float(ds[hit[0]])
        cand = m & (d_fwd >= d0 - 1e-6) & (d_fwd <= d0 + c.band_m)
        r, k = np.nonzero(cand)
        if r.min() <= 0 or float(np.percentile(rel[cand], 95)) >= c.min_top_m:
            cl = cand
            break
        lo = d0 + c.band_m
    if cl is None:
        return None
    dist = float(np.percentile(d_fwd[cl], 10))
    hs = rel[cl]
    # the bottom is known when the frame does not cut the obstacle off and the floor below it is in view
    low = r >= np.percentile(r, 80)
    cut_bottom = bool(r.max() >= nr - 1) or float(np.mean(has_ref[cl][low])) < 0.5
    cut_top = bool(r.min() <= 0)
    bottom = None if cut_bottom or not heights_known else float(np.percentile(hs, 5))
    top = None if cut_top or not heights_known else float(np.percentile(hs, 95))
    if bottom is not None and bottom >= c.head_bottom_m:
        level = "head"
    elif top is not None and top <= c.ground_top_m:
        level = "ground"
    else:
        level = "body"
    zc = zm[cl]
    z30, z70 = np.percentile(zc, [30, 70])
    spread = float(z70 - z30) / max(float(z30), 0.3)
    depth_q = min(0.98, max(0.05, 0.95 - 0.03 * dist - 0.6 * min(1.0, spread)))
    ar = float(area[cl].sum())
    conf = depth_q * (1.0 - math.exp(-ar / 0.03)) * floor_q
    bbox = (float(k.min()) / nc, float(r.min()) / nr, float(k.max() + 1) / nc, float(r.max() + 1) / nr)
    return Obstacle(sector, dist, float(np.median(lat[cl])), level, float(conf), bbox, ar, bottom, top)


# ---- over time: persistence, identity memory, what to say ----------------------------------------------------------
@dataclass
class _Hit:
    t: float
    d: float
    level: str
    conf: float
    bbox: BBox
    explained: bool
    area: float = 0.0


@dataclass
class _SectorTrack:
    hits: Deque[_Hit] = field(default_factory=lambda: deque(maxlen=40))
    last_t: float = -math.inf        # last depth map seen (hit or miss)

    def observe(self, t: float, o: Optional[Obstacle], cfg: CaneConfig) -> None:
        self.last_t = t
        if self.hits and t - self.hits[-1].t > cfg.lost_s:
            self.hits.clear()
        if o is None:
            return
        if self.hits:
            p = self.hits[-1]
            if abs(o.distance_m - p.d) > cfg.closer_m + cfg.max_walk_mps * (t - p.t) + 0.15 * o.distance_m:
                self.hits.clear()                          # a different obstacle (or a jump): start over
        self.hits.append(_Hit(t, o.distance_m, o.level, o.confidence, o.bbox, o.explained, o.area_m2))
        while self.hits and t - self.hits[0].t > max(cfg.approach_s + 1.0, 3.0):
            self.hits.popleft()

    def confirmed(self, t: float, cfg: CaneConfig) -> bool:
        h = self.hits
        return (len(h) >= cfg.persist_maps and h[-1].t - h[0].t >= cfg.persist_s - 1e-6 and t - h[-1].t <= cfg.lost_s)

    @property
    def distance(self) -> float:
        return float(np.median([h.d for h in list(self.hits)[-3:]]))

    @property
    def level(self) -> str:
        last = [h.level for h in list(self.hits)[-5:]]
        best = max(set(last), key=lambda lv: (last.count(lv), LEVELS.index(lv) == 1, -last[::-1].index(lv)))
        return best

    @property
    def confidence(self) -> float:
        last = list(self.hits)[-3:]
        return float(np.mean([h.conf for h in last])) * min(1.0, len(self.hits) / 3.0)

    @property
    def explained(self) -> bool:
        last = list(self.hits)[-3:]
        return sum(h.explained for h in last) * 2 > len(last)

    def closing_speed(self, t: float, window_s: float = 1.2) -> float:
        """m/s toward the camera over the last window_s (least squares over >= 3 maps spanning >= 0.4 s), >= 0."""
        hs = [h for h in self.hits if t - h.t <= window_s]
        if len(hs) < 3 or hs[-1].t - hs[0].t < 0.4:
            return 0.0
        tt = np.array([h.t for h in hs])
        dd = np.array([h.d for h in hs])
        return float(min(2.0, max(0.0, -np.polyfit(tt - tt[0], dd, 1)[0])))

    def approaching(self, t: float, cfg: CaneConfig) -> bool:
        """Range fell by max(closer_m, closer_frac x then) within approach_s (median of 3 hits at each end)."""
        hs = [h for h in self.hits if t - h.t <= cfg.approach_s]
        if len(hs) < 5 or hs[-1].t - hs[0].t < 0.5:
            return False
        then = float(np.median([h.d for h in hs[:3]]))
        now = float(np.median([h.d for h in hs[-3:]]))
        return then - now >= max(cfg.closer_m, cfg.closer_frac * then)


@dataclass
class _Said:
    t: float
    d: float
    seen: float


def explained_by(o: Obstacle, tracks: Optional[Sequence], cfg: CaneConfig) -> bool:
    """A visible detected track covers this obstacle (image region and, when it has one, distance)."""
    if not tracks:
        return False
    ox1, oy1, ox2, oy2 = o.bbox
    oa = max(1e-6, (ox2 - ox1) * (oy2 - oy1))
    for trk in tracks:
        if getattr(trk, "misses", 0) or getattr(trk, "remembered", False) or getattr(trk, "kind", "object") != "object":
            continue
        x1, y1, x2, y2 = trk.bbox
        inter = max(0.0, min(ox2, x2) - max(ox1, x1)) * max(0.0, min(oy2, y2) - max(oy1, y1))
        if inter / oa < cfg.explain_overlap:
            continue
        dist = getattr(trk, "distance_m", None)
        if dist is None or not math.isfinite(dist) or abs(dist - o.distance_m) <= max(cfg.explain_m, cfg.explain_frac * o.distance_m):
            return True
    return False


class Cane:
    """The temporal cane: feed every new depth map, ask for a cue at each decision step, commit what was said."""

    def __init__(self, cfg: Optional[CaneConfig] = None):
        self.cfg = cfg or CaneConfig()
        self.reset()

    def reset(self) -> None:
        self.frame: Optional[CaneFrame] = None
        self.floor: Optional[Floor] = None
        self._floor_t = -math.inf
        self._last_map_ref = None
        self._t = -math.inf
        self.sectors: Dict[str, _SectorTrack] = {s: _SectorTrack() for s in SECTORS}
        self._side_clear: Dict[str, Deque[Tuple[float, Optional[float]]]] = {s: deque(maxlen=6) for s in ("left", "right")}
        self.memory: Dict[str, _Said] = {}
        self._stop_t = -math.inf
        self._stop_armed = True
        self._near_t = -math.inf          # last map with a confirmed obstacle within near_m ahead
        self.n_maps = 0
        self.ms: Deque[float] = deque(maxlen=200)
        self.stats: Dict[str, int] = {"said": 0, "stop": 0, "path": 0, "head": 0, "held": 0, "explained": 0, "still": 0,
                                      "close_up": 0, "noted": 0}

    # ---- input --------------------------------------------------------------------------------------------------
    def update(self, depth: Optional[np.ndarray], t: float, tracks: Optional[Sequence] = None) -> Optional[CaneFrame]:
        """Analyse a *new* depth map (the same array object again, e.g. the pipeline's map between depth runs, is
        ignored) and update the per-sector persistence."""
        if depth is None:
            return self.frame
        if depth is self._last_map_ref:
            return self.frame
        self._last_map_ref = depth
        prior = self.floor if (self.floor is not None and t - self._floor_t <= self.cfg.reuse_floor_s) else None
        fr = analyze(depth, self.cfg, prior)
        if fr.floor is not None and fr.floor.source == "fit":
            self.floor, self._floor_t = fr.floor, t
        for s in SECTORS:
            o = fr.obstacles.get(s)
            if o is not None:
                o.explained = explained_by(o, tracks, self.cfg)
            self.sectors[s].observe(t, o, self.cfg)
        for s in ("left", "right"):
            self._side_clear[s].append((t, fr.clear_m.get(s)))
        self.frame, self._t = fr, t
        self.n_maps += 1
        self.ms.append(fr.ms)
        ahead = self.sectors["ahead"]
        if ahead.confirmed(t, self.cfg):
            for key, e in self.memory.items():
                if key.startswith("obstacle:ahead:"):
                    e.seen = t
            if ahead.distance <= self.cfg.near_m:
                self._near_t = t
        if not self._stop_armed and t - self._near_t >= self.cfg.stop_rearm_s:
            self._stop_armed = True
        return fr

    def _stop_due(self, t: float, d: float) -> bool:
        """Intelligent disobedience: something big within stop_m that the user walked up to, once per obstacle."""
        c, ahead = self.cfg, self.sectors["ahead"]
        if not self._stop_armed or t - self._stop_t < c.stop_gap_s or ahead.hits[-1].area < c.stop_area_m2:
            return False
        # where it will be at the next decision: a walker covers 0.5 m between decisions, so waiting for 0.6 m is late
        # (closing speed capped at a brisk walk; the latest map must be within near_m: noise is not an approach)
        v = min(c.max_walk_mps, ahead.closing_speed(t))
        if v < c.stop_min_mps or d - v * c.stop_lookahead_s > c.stop_m or ahead.hits[-1].d > c.near_m:
            return False
        return self._approached(t, d)

    def _approached(self, t: float, d: float) -> bool:
        """The obstacle ahead was seen at least stop_seen_m farther than d within the last 3 s: the user walked up to it
        (a seat back, a wall the user stands at or a panel they read never came closer)."""
        return any(h.d >= d + self.cfg.stop_seen_m for h in self.sectors["ahead"].hits if t - h.t <= 3.0)

    @property
    def walking_view(self) -> bool:
        """False when the floor fit says the camera is less than walk_min_height_m above it (a close-up or seated)."""
        fl = self.frame.floor if self.frame is not None else None
        return not (fl is not None and fl.source in ("fit", "reused") and fl.height_m < self.cfg.walk_min_height_m)

    # ---- output -------------------------------------------------------------------------------------------------
    def side_clear(self, side: str, t: float) -> Optional[float]:
        """The smallest clear distance of a side over its recent maps (None if any of them could not tell)."""
        vals = [v for tt, v in self._side_clear[side] if t - tt <= 1.0]
        if not vals or any(v is None for v in vals):
            return None
        return float(min(vals))

    def _prune(self, t: float) -> None:
        for k in [k for k, e in self.memory.items() if t - e.seen > self.cfg.forget_s]:
            del self.memory[k]

    def propose(self, t: float, moving: bool = True, busy: bool = False, read_mode: bool = False,
                units: str = "m") -> Optional[Cue]:
        """The cane cue to say at this decision step, or None. Not remembered until commit(). `moving`: the user walks
        (or the activity is not known to be still); `busy`: speech is playing (only a stop cue, which is haptic, may go)."""
        c = self.cfg
        self._prune(t)
        if self.frame is None or t - self._t > c.stale_s:
            return None
        ahead = self.sectors["ahead"]
        if not ahead.confirmed(t, c):
            return None
        d, level, conf = ahead.distance, ahead.level, ahead.confidence
        if not self.walking_view:
            self.stats["close_up"] += 1
            return None
        if moving and self._stop_due(t, d):
            return self._cue(t, "Stop.", "stop:ahead", d, conf, 1.0, action="haptic")
        if read_mode or busy or d > c.speak_m or conf < c.speak_conf or t - self._stop_t < c.stop_gap_s:
            return None                     # (right after a stop the buzz has said it)
        if d <= c.stop_m and not self._approached(t, d):
            self.stats["close_up"] += 1     # within reach and never walked up to: something the user is handling
            return None
        if ahead.explained:
            self.stats["explained"] += 1
            return None
        approaching = ahead.approaching(t, c)
        if not moving and not approaching:
            self.stats["still"] += 1
            return None
        key = f"obstacle:ahead:{level}"
        e = self._recall(key, level)
        if e is not None and d > e.d + max(0.5, 0.25 * e.d):
            self.memory.pop(key, None)                # clearly farther than when it was said: a new obstacle
            e = None
        if e is not None:
            closer = d <= e.d - max(c.closer_m, c.closer_frac * e.d)
            band = d <= c.near_m < e.d
            if t - e.t < c.min_gap_s and not band:
                self.stats["held"] += 1
                return None
            if not (closer or (moving and t - e.t >= c.remind_s)):
                self.stats["held"] += 1
                return None
        dist = distance_phrase(d, conf, units)
        side = blocked = None
        if level != "head" and d <= c.blocked_m:
            side = clearer_side(self.side_clear("left", t), self.side_clear("right", t), d, c)
            other = None if side is None else ("right" if side == "left" else "left")
            v = None if other is None else self.side_clear(other, t)
            blocked = other if v is not None and v < max(2.0, d + c.clearer_by_m) else None
        if blocked is not None:
            # path choice by what blocks, never by what looks open (drop-offs are not seen): "Obstacle ahead, about
            # 1 meter, and on your right" leaves the left to the user's cane
            msg, sev = "Obstacle ahead" + (f", {dist}," if dist else "") + f" and on your {blocked}.", 0.75
        elif level == "head":
            msg, sev = "Something at head height ahead" + (f", {dist}." if dist else "."), 0.8
        elif level == "ground":
            msg, sev = "Low obstacle ahead" + (f", {dist}." if dist else "."), 0.7
        else:
            msg, sev = "Obstacle ahead" + (f", {dist}." if dist else "."), 0.7
        if d <= c.near_m:
            sev = max(sev, 0.85)
        return self._cue(t, msg, key, d, conf, sev, path=side)

    def _recall(self, key: str, level: str) -> Optional[_Said]:
        """The memory of this identity; for ground / body also the other of the two in the same sector (the level of
        one obstacle may flip between maps: that is not news). Head level is the dog's own message and stays apart."""
        e = self.memory.get(key)
        if e is not None or level == "head":
            return e
        sib = self.memory.get(key.rsplit(":", 1)[0] + ":" + ("body" if level == "ground" else "ground"))
        return sib

    def _cue(self, t: float, msg: str, key: str, d: float, conf: float, severity: float, action: str = "speak",
             path: Optional[str] = None) -> Cue:
        cue = Cue(t=t, track_id=CANE_TRACK_ID, label="obstacle", kind="object", message=msg, score=severity,
                  policy=CANE_POLICY, action=action, duration_s=0.0 if action == "haptic" else speech_seconds(msg),
                  slots=("object", "distance", "direction"), distance_m=round(d, 2), severity=severity, utility=severity)
        cue.cane_key = key                     # type: ignore[attr-defined]
        cue.cane_path = path                   # type: ignore[attr-defined]
        return cue

    def commit(self, cue: Cue, t: float) -> None:
        """The controller let this cane cue through: remember it."""
        key = getattr(cue, "cane_key", "")
        d = cue.distance_m if cue.distance_m is not None else 0.0
        if key == "stop:ahead":
            self._stop_t, self._stop_armed = t, False
            self.stats["stop"] += 1
            # the stop is the warning about what is ahead: no "right in front of you" after it unless it closes in further
            lv = self.sectors["ahead"].level if self.sectors["ahead"].hits else "body"
            self.memory[f"obstacle:ahead:{lv}"] = _Said(t, float(d), t)
            return
        self.memory[key] = _Said(t, float(d), t)
        self.stats["said"] += 1
        if getattr(cue, "cane_path", None):
            self.stats["path"] += 1
        if key.endswith(":head"):
            self.stats["head"] += 1

    # ---- reporting ----------------------------------------------------------------------------------------------
    def note(self, t: float, units: str = "m", commit: bool = False) -> str:
        """The confirmed obstacle ahead in words, for answers ("The depth map shows an obstacle ahead, about 1 meter."),
        or "": none confirmed (3 maps over 0.5 s), no recent map, not a walking view, or a detected track explains it.
        Whether the user walks does not matter here: the user asked. commit=True: the answer says it, so the identity
        memory holds it like a spoken cue (no "Obstacle ahead" right after the reply that said so)."""
        c, ahead = self.cfg, self.sectors["ahead"]
        if (self.frame is None or t - self._t > c.stale_s or not ahead.confirmed(t, c) or not self.walking_view
                or ahead.explained or ahead.confidence < c.speak_conf):
            return ""
        d, level = ahead.distance, ahead.level
        what = {"head": "something at head height", "ground": "a low obstacle"}.get(level, "an obstacle")
        dist = distance_phrase(d, ahead.confidence, units)
        if commit:
            self.memory[f"obstacle:ahead:{level}"] = _Said(t, float(d), t)
            self.stats["noted"] += 1
        return f"The depth map shows {what} ahead" + (f", {dist}." if dist else ".")

    def summary(self) -> Optional[dict]:
        return None if self.frame is None else self.frame.as_contract(self.cfg)

    def describe(self) -> str:
        s, fl = self.stats, self.floor
        ms = f"{np.mean(self.ms):.1f} ms/map" if self.ms else "no maps"
        floor = (f"floor {fl.height_m:.2f} m, pitch {fl.pitch_deg:.0f} deg" + (f", depth x{fl.scale:.2f}" if abs(fl.scale - 1) > 1e-3 else "")
                 if fl is not None else "no floor fit yet")
        return (f"cane: {s['said']} said ({s['path']} path, {s['head']} head), {s['stop']} stop; quiet: {s['held']} held, "
                f"{s['explained']} explained by a track, {s['still']} still; {self.n_maps} maps, {ms}; {floor}")
