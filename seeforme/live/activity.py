"""What is the user doing? Stationary, walking, turning or reading, from camera motion and visible text, and the
live-demo mode that follows from it.

Used by the live demo only (scripts/run_live.py), never by the benchmarks.

ActivityDetector.update(frame_bgr, t, tracks=None, text_present=None) -> Activity
---------------------------------------------------------------------------------
Frame-based motion (the normal path, ~1 ms per measurement): each frame is shrunk to `width` (160) px grayscale and
compared with a reference frame at least `min_gap_s` old, so the numbers are the same at 5, 15 or 30 fps (they were
calibrated on 5 fps footage):
    shift     global image shift from cv2.phaseCorrelate, in frame widths per second (dx > 0: the scene moves right
              in the image, i.e. the camera turns left)
    energy    mean |difference| of the two mean-normalised frames / 255, per 0.2 s
    flow      energy / mean |image gradient|: roughly how many pixels (at 160 px) the content moved per 0.2 s, whatever
              the texture or lighting. It is the main still-vs-moving signal: walking moves *everything*, while a desk
              webcam or a phone held still sees only hands or passers-by change part of the frame
    coverage  share of the textured 3x3 tiles whose content moved (> 0.5 px per 0.2 s): whole-frame (ego) motion
    forward   expansion rate (1/s) from the phase-correlation shifts of the four quadrants (a zoom moves each quadrant
              outwards); walking forward makes the image expand
    bob       |vertical shift| rate: a hand-held phone bobs at every step
Track-based motion (frame_bgr is None, e.g. replays of cached detections): the median displacement of the box centres
of the object tracks detected in both updates (shift), their median size growth (forward) and the track churn
(1 - ids kept / ids seen: objects coming and going); no energy / flow.

Every quantity is smoothed by an exponential moving average whose factor `ema` is defined per 0.2 s and rescaled to
the real step, then classified (`ActivityDetector.classify`):
    TURNING     |EMA signed horizontal shift| >= turn_rate and >= turn_consistency x EMA |dx| (a sustained yaw one way;
                walking sway and a swinging phone cancel out)
    STATIONARY  shift < still_shift and (flow < still_flow or energy < still_energy)
                (tracks: shift < track_still_shift and churn < track_still_churn)
    READING     STATIONARY and text visible (text_present, or a track with kind == "text") for >= text_min_s
    WALKING     flow >= move_flow or shift >= move_shift, with a forward cue, bob or whole-frame coverage
                (tracks: shift >= track_move_shift or churn >= track_move_churn)
    UNKNOWN     no measurement yet, or moving without any walking cue; in the band between the still and the moving
                thresholds the current settled state is kept
A new activity is adopted only when it wins `switch_frac` of the per-update candidates over the last `dwell_s`
(`dwell_motion_s` for WALKING / TURNING: entering a moving state matters for safety, so it is quicker); a single noisy
frame never flips the state.

The thresholds were calibrated on EgoBlind development footage only (20 finetune / select clips decoded at 5 fps,
18 of them hand-labelled from contact sheets into standing / walking / turning segments) and on synthetic desk-webcam
sequences; the per-threshold evidence is in the ActivityConfig comments. With these defaults, frame-based: walking
segments are STATIONARY 1.8% of the time (navigate 98.2%), standing segments 66% (the rest are close-up hands that
move the whole frame), synthetic desk webcams (5 / 15 / 30 fps, person, hands, noise, flicker) STATIONARY after 1.8 s
and for good; track-based (cached detections): walking 2.8% / standing 60% STATIONARY.

TaskAdapter(gate=None, policy=None, goal_fn=None).step(activity, t, has_goal) -> navigate | scene | read | find
---------------------------------------------------------------------------------------------------------------
Applies the gate / verbosity settings of a mode when the mode changes (see MODE_SETTINGS and the class docstring).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # OpenCV is a hard dependency of the live demo; the track-based fallback works without it
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class Activity(str, Enum):
    STATIONARY = "stationary"
    WALKING = "walking"
    TURNING = "turning"
    READING = "reading"
    UNKNOWN = "unknown"


MOVING = (Activity.WALKING, Activity.TURNING)
STILL = (Activity.STATIONARY, Activity.READING)
STEP_S = 0.2   # the nominal measurement step every per-step quantity is expressed in (calibration footage: 5 fps)


@dataclass
class ActivityConfig:
    """Thresholds. Percentiles are of the EMA-smoothed features over the hand-labelled development segments:
    S = standing / phone held or resting still (1500 samples, 9 clips, incl. hands and passers-by in view),
    W = walking (2442 samples, 11 clips), DESK = synthetic desk webcam (person, hand, talking, sensor noise,
    exposure flicker; 1740 samples)."""
    # --- measurement ------------------------------------------------------------------------------------------
    width: int = 160                 # frames are shrunk to this width (grayscale)
    min_gap_s: float = 0.15          # measure against a reference at least this old (fps-independent numbers)
    max_gap_s: float = 1.0           # an older reference (pipeline stalled) is replaced, not measured against
    stale_s: float = 2.0             # no usable measurement for this long (camera stalled): forget the state (navigate)
    track_aspect: float = 0.75       # frame height / width, to put track-based dy in image widths like dx (4:3)
    min_response: float = 0.05       # phase-correlation peak below this: the shift is unreliable and counted as 0
    max_rate: float = 3.0            # widths/s; faster shifts are clipped (whip pan, dropped frames)
    min_gradient: float = 2.0        # gray levels / px; floor of the gradient that normalises `flow` (dark frames)
    # --- smoothing and hysteresis -------------------------------------------------------------------------------
    ema: float = 0.35                # EMA factor per 0.2 s step (rescaled: 1 - (1 - ema) ** (dt / 0.2))
    dwell_s: float = 1.5             # a new STATIONARY / READING / UNKNOWN state must win this long
    dwell_motion_s: float = 1.0      # a new WALKING / TURNING state must win this long
    switch_frac: float = 0.7         # share of the candidates over the dwell window the new state needs
    # --- frame-based thresholds ---------------------------------------------------------------------------------
    still_flow: float = 0.7          # px / 0.2 s. W p5 0.91, S p50 0.56, DESK max 0.28 (0.55 dark + noisy + JPEG)
    still_energy: float = 0.01       # also still when frames hardly differ at all (low-texture / dim desk scene)
    still_shift: float = 0.05        # widths/s. S p90 0.047, DESK max 0.018
    move_flow: float = 1.2           # W p25 1.25, S p80 1.21 (close-up hands move the whole frame)
    move_shift: float = 0.15         # widths/s. W p75 0.14, S p99 0.15
    turn_rate: float = 0.2           # |EMA signed dx| widths/s (~14 deg/s, 70 deg lens). S p99 0.10, W p90 0.21
    turn_consistency: float = 0.5    # ... and |EMA dx| / EMA |dx| at least this: one direction, not a swinging phone
    walk_forward: float = 0.1        # expansion 1/s. W p75 0.11, S p95 0.07
    walk_bob: float = 0.06           # EMA |dy| widths/s. W p80 0.05 / p85 0.07, S p99 0.053
    walk_coverage: float = 0.85      # share of textured tiles moving. W p5 0.86, DESK max 0.24
    # --- track-based thresholds (frame_bgr is None). Same labelled clips, their cached detections through the
    # IoU tracker: S 1475 / W 2385 samples. Box jitter and objects' own motion make this much weaker evidence.
    track_still_shift: float = 0.04  # median box-centre speed (widths/s). S p50 0.025, W p5 0.039
    track_still_churn: float = 0.3   # 1 - |ids kept| / |ids seen|. S p50 0.24, W p25 0.33
    track_move_shift: float = 0.08   # W p25 0.067, S p75 0.063
    track_move_churn: float = 0.5    # W p50 0.50, S p90 0.51
    track_turn_rate: float = 0.2     # |EMA signed dx| (consistency as turn_consistency). S p95 0.09, W p95 0.19
    track_min_tracks: int = 1        # object tracks seen in both updates needed for a shift measurement
    # --- text ---------------------------------------------------------------------------------------------------
    text_min_s: float = 1.0          # text visible this long while steady is READING
    text_grace_s: float = 1.0        # a gap shorter than this (OCR flicker) does not reset the text timer
    text_stale_s: float = 1.0        # a text track not seen for this long does not count as visible


@dataclass
class MotionSample:
    """One motion measurement between two frames (or two track sets) dt seconds apart."""
    dt: float
    dx: Optional[float]                # widths/s, + = the scene moves right in the image (camera turns left)
    dy: Optional[float]                # widths/s, + = the scene moves down
    forward: Optional[float]           # expansion rate 1/s (+ = the image grows: moving towards the scene)
    energy: Optional[float] = None     # mean |difference| / 255 per 0.2 s (None for track-based samples)
    flow: Optional[float] = None       # texture-normalised displacement, px (at `width`) per 0.2 s
    coverage: Optional[float] = None   # share of textured tiles that moved (None: fewer than 3 textured tiles)
    churn: Optional[float] = None      # tracks: 1 - |ids in both| / |ids in either| (objects coming and going)
    response: float = 1.0              # phase-correlation peak (reliability of dx, dy)
    source: str = "frame"              # frame | tracks


@dataclass
class ActivityState:
    activity: Activity = Activity.UNKNOWN
    since: float = 0.0                 # time the current activity was adopted
    candidate: Activity = Activity.UNKNOWN   # the instantaneous classification of the last update
    motion: float = 0.0                # EMA |shift| (widths/s)
    shift: Tuple[float, float] = (0.0, 0.0)  # EMA signed (dx, dy) (widths/s)
    energy: float = 0.0                # EMA energy
    flow: float = 0.0                  # EMA flow
    coverage: float = 0.0              # EMA coverage
    churn: float = 0.0                 # EMA track churn (track-based only)
    forward: float = 0.0               # EMA expansion rate (1/s)
    bob: float = 0.0                   # EMA |dy| (widths/s)
    confidence: float = 0.0            # share of the recent candidates that agree with `activity` (x0.6 track-based)
    source: str = "none"               # frame | tracks | none: where the last measurement came from
    text_s: float = 0.0                # how long text has been visible
    samples: int = 0


# ---- frame measurements ------------------------------------------------------------------------------------------
def gray_small(frame: np.ndarray, width: int = 160) -> np.ndarray:
    """BGR / BGRA / grayscale frame -> float32 grayscale `width` px wide."""
    a = np.asarray(frame)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    h, w = a.shape[:2]
    W = int(width)
    H = max(16, int(round(h * W / max(1, w))))
    if (w, h) != (W, H):
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY if a.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


_WINDOWS: Dict[Tuple[int, int], np.ndarray] = {}


def _window(h: int, w: int) -> np.ndarray:
    win = _WINDOWS.get((h, w))
    if win is None:
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        _WINDOWS[(h, w)] = win
    return win


def _phase(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    # cv2.phaseCorrelate windows its inputs in place: always hand it copies
    (sx, sy), r = cv2.phaseCorrelate(np.array(a, np.float32), np.array(b, np.float32), _window(*a.shape[:2]))
    if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(r)):
        return 0.0, 0.0, 0.0
    return float(sx), float(sy), float(r)


def measure_motion(prev: np.ndarray, cur: np.ndarray, dt: float, min_response: float = 0.05, max_rate: float = 3.0,
                   min_gradient: float = 2.0) -> MotionSample:
    """Motion between two small float32 grayscale frames of the same size taken dt seconds apart (see module doc)."""
    H, W = cur.shape[:2]
    dt = max(1e-3, float(dt))
    per_step = STEP_S / dt
    clip = lambda v: max(-max_rate, min(max_rate, v))
    # energy, flow, coverage
    diff = np.abs((cur - float(cur.mean())) - (prev - float(prev.mean())))
    grad = (np.abs(cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3))) / 8.0
    energy = float(diff.mean()) / 255.0 * per_step
    flow = float(diff.mean()) / max(min_gradient, float(grad.mean())) * per_step
    moved = textured = 0
    for i in range(3):
        for j in range(3):
            ys = slice(i * H // 3, (i + 1) * H // 3)
            xs = slice(j * W // 3, (j + 1) * W // 3)
            g = float(grad[ys, xs].mean())
            if g >= min_gradient:
                textured += 1
                moved += float(diff[ys, xs].mean()) / g * per_step > 0.5
    coverage = moved / textured if textured >= 3 else None
    # global shift
    sx, sy, resp = _phase(prev, cur)
    if resp < min_response:
        sx = sy = 0.0
    # expansion from the four quadrants: a zoom by (1 + k) moves the quadrant centred at r by k * r
    h2, w2 = H // 2, W // 2
    quads = []
    for (y0, x0) in ((0, 0), (0, w2), (h2, 0), (h2, w2)):
        qx, qy, qr = _phase(prev[y0:y0 + h2, x0:x0 + w2], cur[y0:y0 + h2, x0:x0 + w2])
        quads.append((qx, qy, qr >= min_response, x0 + w2 / 2.0 - W / 2.0, y0 + h2 / 2.0 - H / 2.0))
    forward = 0.0
    if all(q[2] for q in quads):
        mx = sum(q[0] for q in quads) / 4.0
        my = sum(q[1] for q in quads) / 4.0
        num = sum((qx - mx) * rx + (qy - my) * ry for qx, qy, _, rx, ry in quads)
        den = sum(rx * rx + ry * ry for _, _, _, rx, ry in quads)
        k = num / den if den > 0 else 0.0
        forward = math.log(max(1e-3, 1.0 + k)) / dt
    return MotionSample(dt=dt, dx=clip(sx / W / dt), dy=clip(sy / W / dt), forward=clip(forward), energy=energy,
                        flow=flow, coverage=coverage, response=resp, source="frame")


def _median(v: List[float]) -> float:
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


class ActivityDetector:
    """Classifies what the user is doing from camera motion and visible text (see the module docstring)."""

    def __init__(self, cfg: Optional[ActivityConfig] = None):
        self.cfg = cfg or ActivityConfig()
        self.reset()

    def reset(self) -> None:
        self.state = ActivityState()
        self._started = False
        self._start_t: Optional[float] = None     # time of the first vote (dwell windows count from it)
        self._ref: Optional[np.ndarray] = None
        self._ref_t: Optional[float] = None
        self._prev_tracks: Dict[int, Tuple[float, float, float]] = {}
        self._prev_tracks_t: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_frame_t: Optional[float] = None
        self._votes: Deque[Tuple[float, Activity]] = deque()
        self._text_since: Optional[float] = None
        self._text_last: Optional[float] = None
        self._ema: Dict[str, Optional[float]] = dict.fromkeys(
            ("dx", "dy", "adx", "motion", "forward", "bob", "energy", "flow", "coverage", "churn"))
        self.last_sample: Optional[MotionSample] = None

    # ---- read-only views ---------------------------------------------------------------------------------------
    @property
    def activity(self) -> Activity:
        return self.state.activity

    @property
    def since(self) -> float:
        return self.state.since

    @property
    def motion(self) -> float:
        return self.state.motion

    @property
    def shift(self) -> Tuple[float, float]:
        return self.state.shift

    @property
    def energy(self) -> float:
        return self.state.energy

    @property
    def confidence(self) -> float:
        return self.state.confidence

    # ---- measurements --------------------------------------------------------------------------------------
    def _frame_sample(self, frame: np.ndarray, t: float) -> Optional[MotionSample]:
        c = self.cfg
        if cv2 is None:  # pragma: no cover
            return None
        small = gray_small(frame, c.width)
        ref_ok = (self._ref is not None and self._ref.shape == small.shape and self._ref_t is not None
                  and self._ref_t <= t <= self._ref_t + c.max_gap_s)
        if not ref_ok:
            self._ref, self._ref_t = small, t
            return None
        if t - self._ref_t < c.min_gap_s:
            return None
        s = measure_motion(self._ref, small, t - self._ref_t, c.min_response, c.max_rate, c.min_gradient)
        self._ref, self._ref_t = small, t
        return s

    def _track_sample(self, tracks: Sequence, t: float) -> Optional[MotionSample]:
        c = self.cfg
        prev_t = self._prev_tracks_t
        if prev_t is not None and prev_t <= t < prev_t + c.min_gap_s:
            return None                     # keep the older reference set (fps-independent numbers)
        cur: Dict[int, Tuple[float, float, float]] = {}
        for tr in tracks:
            if getattr(tr, "remembered", False) or getattr(tr, "misses", 0) > 0 or getattr(tr, "kind", "object") != "object":
                continue                    # only boxes detected now are evidence of camera motion; OCR boxes jitter
            x1, y1, x2, y2 = tr.bbox
            cur[tr.track_id] = (0.5 * (x1 + x2), 0.5 * (y1 + y2), max(1e-6, (x2 - x1) * (y2 - y1)))
        prev = self._prev_tracks
        self._prev_tracks, self._prev_tracks_t = cur, t
        if prev_t is None or t <= prev_t or t - prev_t > c.max_gap_s:
            return None
        union = set(cur) | set(prev)
        if not union:
            return None                     # nothing visible: no evidence either way
        common = [k for k in cur if k in prev]
        dt = t - prev_t
        clip = lambda v: max(-c.max_rate, min(c.max_rate, v))
        s = MotionSample(dt=dt, dx=None, dy=None, forward=None, churn=1.0 - len(common) / len(union), source="tracks")
        if len(common) >= max(1, c.track_min_tracks):
            s.dx = clip(_median([cur[k][0] - prev[k][0] for k in common]) / dt)
            # box centres are normalised per axis: express dy in image widths like dx (frames are wider than tall)
            s.dy = clip(_median([cur[k][1] - prev[k][1] for k in common]) * c.track_aspect / dt)
            s.forward = clip(_median([0.5 * math.log(cur[k][2] / prev[k][2]) for k in common]) / dt)
        return s

    def _update_text(self, t: float, tracks: Optional[Sequence], text_present: Optional[bool]) -> float:
        c = self.cfg
        if text_present is None:
            text_present = any(getattr(tr, "kind", "object") == "text" and not getattr(tr, "remembered", False)
                               and (getattr(tr, "last_seen", None) is None or t - tr.last_seen <= c.text_stale_s)
                               for tr in (tracks or []))
        if text_present:
            if self._text_since is None:
                self._text_since = t
            self._text_last = t
        elif self._text_last is not None and t - self._text_last > c.text_grace_s:
            self._text_since = self._text_last = None
        return 0.0 if self._text_since is None else max(0.0, t - self._text_since)

    def _smooth(self, s: MotionSample) -> None:
        a = min(1.0, max(0.0, 1.0 - (1.0 - self.cfg.ema) ** (max(1e-3, s.dt) / STEP_S)))
        restart = self.state.source != s.source          # frame <-> tracks: the scales differ, start afresh
        has_xy = s.dx is not None and s.dy is not None
        vals = {"dx": s.dx, "dy": s.dy, "adx": abs(s.dx) if has_xy else None,
                "motion": math.hypot(s.dx, s.dy) if has_xy else None, "forward": s.forward,
                "bob": abs(s.dy) if has_xy else None, "energy": s.energy, "flow": s.flow, "coverage": s.coverage,
                "churn": s.churn}
        for k, v in vals.items():
            old = self._ema.get(k)
            if v is None:
                if restart:
                    self._ema[k] = None
                continue
            self._ema[k] = v if (old is None or restart) else old + a * (v - old)
        e, st = self._ema, self.state
        st.shift = (e["dx"] or 0.0, e["dy"] or 0.0)
        st.motion, st.forward, st.bob = e["motion"] or 0.0, e["forward"] or 0.0, e["bob"] or 0.0
        st.energy, st.flow, st.coverage = e["energy"] or 0.0, e["flow"] or 0.0, e["coverage"] or 0.0
        st.churn = e["churn"] or 0.0
        st.source = s.source
        st.samples += 1

    # ---- classification ------------------------------------------------------------------------------------
    def classify(self, text_s: float = 0.0) -> Activity:
        """Instantaneous classification of the current smoothed features (no dwell)."""
        c, st = self.cfg, self.state
        if st.samples == 0:
            return Activity.UNKNOWN
        frame = st.source == "frame"
        sdx = abs(st.shift[0])
        turn = c.turn_rate if frame else c.track_turn_rate
        if sdx >= turn and sdx >= c.turn_consistency * (self._ema["adx"] or 0.0):
            return Activity.TURNING
        if frame:
            still = st.motion < c.still_shift and (st.flow < c.still_flow or st.energy < c.still_energy)
            moving = st.motion >= c.move_shift or st.flow >= c.move_flow
            walk_cue = (st.forward >= c.walk_forward or st.bob >= c.walk_bob
                        or (self._ema["coverage"] is not None and st.coverage >= c.walk_coverage))
        else:   # tracks: no bob / coverage; sustained box motion or objects coming and going is the walking cue
            still = st.motion < c.track_still_shift and st.churn < c.track_still_churn
            moving = st.motion >= c.track_move_shift or st.churn >= c.track_move_churn
            walk_cue = True
        if still:
            return Activity.READING if text_s >= c.text_min_s else Activity.STATIONARY
        if moving:
            return Activity.WALKING if walk_cue else Activity.UNKNOWN
        # band between the still and the moving thresholds: keep a settled state, except that a forward or bob cue
        # (slow walking: flow can stay under move_flow) leaves a still state for UNKNOWN, which maps to navigate
        cur = st.activity
        if cur in STILL:
            if frame and (st.forward >= c.walk_forward or st.bob >= c.walk_bob):
                return Activity.UNKNOWN
            return Activity.READING if text_s >= c.text_min_s else Activity.STATIONARY
        return cur if cur == Activity.WALKING else Activity.UNKNOWN

    def _switch(self, t: float) -> None:
        c, st = self.cfg, self.state
        horizon = max(c.dwell_s, c.dwell_motion_s)
        while self._votes and self._votes[0][0] < t - horizon - 1e-9:
            self._votes.popleft()
        if not self._votes or self._start_t is None:
            return

        def window(d: float) -> List[Activity]:
            if t - self._start_t < d - 1e-9:
                return []                    # not observed long enough yet
            return [a for (tv, a) in self._votes if tv >= t - d - 1e-9]

        def wins(group: Sequence[Activity], w: List[Activity]) -> Optional[Activity]:
            inside = [a for a in w if a in group]
            if not w or len(inside) < c.switch_frac * len(w):
                return None
            return max(group, key=inside.count)      # ties: the earlier entry of `group`

        cur = st.activity
        wm, ws = window(c.dwell_motion_s), window(c.dwell_s)
        if cur in MOVING:
            target = wins([a for a in MOVING if a != cur], wm) or wins(STILL, ws)
        elif cur in STILL:
            # leaving a still state is the safe direction: walking, turning and unknown votes all count against it
            leave = wins((*MOVING, Activity.UNKNOWN), wm)
            target = leave or wins([a for a in STILL if a != cur], ws)
        else:
            target = wins(MOVING, wm) or wins(STILL, ws)
        if target is None and cur != Activity.UNKNOWN:
            target = wins([Activity.UNKNOWN], ws)
        if target is not None and target != cur:
            st.activity, st.since = target, t

    def _confidence(self, t: float) -> float:
        st = self.state
        w = [a for (tv, a) in self._votes if tv >= t - self.cfg.dwell_s - 1e-9]
        if st.activity == Activity.UNKNOWN or not w:
            return 0.0
        agree = sum(1 for a in w if a == st.activity) / len(w)
        return round(agree * (0.6 if st.source == "tracks" else 1.0), 3)

    # ---- main entry ----------------------------------------------------------------------------------------
    def update(self, frame_bgr: Optional[np.ndarray], t: float, tracks: Optional[Sequence] = None,
               text_present: Optional[bool] = None) -> Activity:
        """Feed one frame (or None: use the tracks' box motion) at time t (s); returns the current activity."""
        t = float(t)
        if self._last_t is not None and t < self._last_t - 1e-6:
            self.reset()                     # time went backwards: a new stream (e.g. a replayed video restarted)
        self._last_t = t
        if not self._started:
            self._started = True
            self.state.since = t
        sample = None
        if frame_bgr is not None:
            self._last_frame_t = t
            sample = self._frame_sample(frame_bgr, t)
        elif tracks is not None and (self._last_frame_t is None or t - self._last_frame_t > 2.0):
            sample = self._track_sample(tracks, t)   # a frame missing now and then does not switch to tracks
        if sample is not None:
            self.last_sample = sample
            self._last_sample_t = t
            self._smooth(sample)
        elif getattr(self, "_last_sample_t", None) is not None and t - self._last_sample_t > self.cfg.stale_s:
            # no frame and no usable tracks for a while (camera stalled): stop trusting the old state
            self.reset()
            self._started, self._last_t, self.state.since = True, t, t
            return self.state.activity
        self.state.text_s = self._update_text(t, tracks, text_present)
        cand = self.classify(self.state.text_s)
        self.state.candidate = cand
        if self.state.samples > 0:
            if self._start_t is None:
                self._start_t = t
            self._votes.append((t, cand))
            self._switch(t)
        self.state.confidence = self._confidence(t)
        return self.state.activity

    def describe(self) -> str:
        st = self.state
        dx, dy = st.shift
        name = st.activity.value
        if st.activity == Activity.TURNING:
            name += " left" if dx > 0 else " right"
        if st.source == "tracks":
            return (f"{name} {st.confidence:.0%} [tracks] shift {dx:+.2f},{dy:+.2f} w/s churn {st.churn:.2f} "
                    f"fwd {st.forward:+.2f} text {st.text_s:.1f}s")
        return (f"{name} {st.confidence:.0%} [{st.source}] shift {dx:+.2f},{dy:+.2f} w/s flow {st.flow:.2f} "
                f"energy {st.energy:.3f} cover {st.coverage:.2f} fwd {st.forward:+.2f} bob {st.bob:.2f} "
                f"text {st.text_s:.1f}s")

    def as_dict(self) -> Dict[str, Any]:
        st = self.state
        return {"activity": st.activity.value, "since": st.since, "candidate": st.candidate.value, "motion": st.motion,
                "shift": st.shift, "energy": st.energy, "flow": st.flow, "coverage": st.coverage, "churn": st.churn,
                "forward": st.forward,
                "bob": st.bob, "confidence": st.confidence, "source": st.source, "text_s": st.text_s}


# ---- task adapter ----------------------------------------------------------------------------------------------
MODES = ("navigate", "scene", "read", "find")


@dataclass
class ModeSettings:
    hold_static_s: float          # gate: how long a static, already-announced object stays silent
    background: bool              # gate: objects static through the warm-up are background (only approaching speaks)
    verbosity: str                # policy.cfg.verbosity: quiet | balanced | detailed
    objects_muted: bool = False   # only text / critical object cues (gate field if it has one, else filter_cues)


MODE_SETTINGS: Dict[str, ModeSettings] = {
    # walking / turning / unknown: the scene changes, keep the user informed
    "navigate": ModeSettings(hold_static_s=20.0, background=False, verbosity="balanced"),
    # standing still (a desk webcam): only arrivals, approaching objects and critical events
    "scene": ModeSettings(hold_static_s=60.0, background=True, verbosity="quiet"),
    # reading: text cues, object cues muted unless critical. No background: the sign the user holds the camera steady
    # on has been static since the mode began, and must still be read (objects are muted here anyway)
    "read": ModeSettings(hold_static_s=60.0, background=False, verbosity="quiet", objects_muted=True),
    # a FIND goal is set: the goal object must come through; static objects may come back sooner
    "find": ModeSettings(hold_static_s=20.0, background=False, verbosity="balanced"),
}

ACTIVITY_MODE: Dict[Activity, str] = {
    Activity.WALKING: "navigate", Activity.TURNING: "navigate", Activity.UNKNOWN: "navigate",
    Activity.STATIONARY: "scene", Activity.READING: "read",
}
STILL_MODES = ("scene", "read")


class TaskAdapter:
    """Turns the activity (and whether a FIND goal is set) into a mode and applies its settings when it changes.

        mode = adapter.step(detector.activity, t, has_goal=bool(goal_tags))

    Priority: manual override (force_mode / cycle_override, e.g. a keyboard key) > find (has_goal, or goal_fn())
    > the activity: WALKING / TURNING / UNKNOWN -> navigate (UNKNOWN is the conservative default: the camera moves
    but not clearly, or nothing is measured yet), STATIONARY -> scene, READING -> read.

    On a mode change it applies MODE_SETTINGS[mode]:
      gate    hold_static_s, background (and objects_muted when the gate's config has that field) through
              gate.set_mode_params(**params) restricted to the fields of gate.gate_cfg (AnnouncementGate), or by
              setting the attributes the gate config already has; entering scene / read from another mode also calls
              gate.rebaseline() so what is static *here* becomes the background. Nothing when gate is None.
      policy  policy.cfg.verbosity, when the policy has a cfg with that attribute (an AnnouncementGate forwards cfg
              to its inner scheduler, so the gate itself can be passed as the policy). Nothing otherwise.
    `applied` records what was actually set. Read mode mutes object cues that are not about safety (text cues,
    haptics, interrupts, severity >= critical_severity and anything approaching pass). The AnnouncementGate has an
    objects_muted field and applies that filter itself *before* it remembers a cue as said: a cue muted after the gate
    had committed it would make the scheduler's later warning about the same object a "repeat" of a cue nobody heard.
    Only for a gate without that field (or none) is read mode enforced here: `should_mute_objects` is True and
    `filter_cues(cues)` drops those cues. filter_cues is a no-op in every other case.
    """

    def __init__(self, gate=None, policy=None, goal_fn: Optional[Callable[[], bool]] = None,
                 modes: Optional[Dict[str, ModeSettings]] = None, initial: str = "navigate",
                 critical_severity: float = 0.9, approach_ttc_s: float = 6.0):
        self.gate = gate
        self.policy = policy
        self.goal_fn = goal_fn
        self.modes = {k: replace(v) for k, v in (modes or MODE_SETTINGS).items()}
        if initial not in self.modes:
            raise ValueError(f"unknown mode {initial!r}")
        self.critical_severity = critical_severity
        self.approach_ttc_s = approach_ttc_s
        self._clock_set = False          # the first step() sets `since` and the first history entry to the real clock
        self.mode = initial
        self.since = 0.0
        self.override: Optional[str] = None
        self.changed = False
        self.reason = "start"
        self.history: List[Tuple[float, str, str]] = []
        self.applied: Dict[str, Any] = {}
        self._gate_mutes = False
        self._t = 0.0
        self._apply(initial, 0.0, "start", rebaseline=False)

    # ---- settings ----------------------------------------------------------------------------------------------
    @property
    def settings(self) -> ModeSettings:
        return self.modes[self.mode]

    def _gate_config(self):
        """The gate's GateConfig-like object (AnnouncementGate.gate_cfg; its `cfg` is the scheduler's), or None."""
        g = self.gate
        gc = getattr(g, "gate_cfg", None)
        if gc is None:
            gc = getattr(g, "cfg", None)
        return gc if gc is not None and hasattr(gc, "hold_static_s") else None

    def _apply_gate(self, params: Dict[str, Any], rebaseline: bool) -> Dict[str, Any]:
        g = self.gate
        if g is None:
            return {}
        gc = self._gate_config()
        if gc is not None and is_dataclass(gc):
            valid = {f.name for f in fields(gc)}
            params = {k: v for k, v in params.items() if k in valid}
        elif gc is not None:
            params = {k: v for k, v in params.items() if hasattr(gc, k)}
        done: Dict[str, Any] = {}
        setter = getattr(g, "set_mode_params", None)
        if callable(setter):
            try:
                setter(**params)
                done = dict(params)
            except (TypeError, ValueError):       # unknown keyword: set what the gate accepts, one by one
                for k, v in params.items():
                    try:
                        setter(**{k: v})
                        done[k] = v
                    except (TypeError, ValueError):
                        pass
        elif gc is not None:
            for k, v in params.items():
                setattr(gc, k, v)
                done[k] = v
        if rebaseline and callable(getattr(g, "rebaseline", None)):
            g.rebaseline()
            done["rebaseline"] = True
        return done

    def _apply_policy(self, s: ModeSettings) -> Dict[str, Any]:
        cfg = getattr(self.policy, "cfg", None) if self.policy is not None else None
        if cfg is not None and hasattr(cfg, "verbosity"):
            cfg.verbosity = s.verbosity
            return {"verbosity": s.verbosity}
        return {}

    def _apply(self, mode: str, t: float, reason: str, rebaseline: Optional[bool] = None) -> None:
        s = self.modes[mode]
        if rebaseline is None:
            rebaseline = mode in STILL_MODES and self.mode not in STILL_MODES
        gate_done = self._apply_gate({"hold_static_s": s.hold_static_s, "background": s.background,
                                      "objects_muted": s.objects_muted}, rebaseline)
        self._gate_mutes = "objects_muted" in gate_done
        self.applied = {**{f"gate.{k}": v for k, v in gate_done.items()},
                        **{f"policy.{k}": v for k, v in self._apply_policy(s).items()}}
        if mode != self.mode or not self.history:
            self.history.append((t, mode, reason))
        self.mode, self.since, self.reason = mode, t, reason

    def set_policy(self, policy) -> None:
        """The demo swapped the scheduler (keys 1-5): re-apply the current mode's verbosity to the new one."""
        self.policy = policy
        self.applied.update({f"policy.{k}": v for k, v in self._apply_policy(self.settings).items()})

    def set_gate(self, gate) -> None:
        """A new gate (e.g. it wraps a newly selected scheduler): re-apply the current mode's gate settings."""
        self.gate = gate
        self._apply(self.mode, self._t, self.reason, rebaseline=False)

    # ---- mode selection ----------------------------------------------------------------------------------------
    def target_mode(self, activity, has_goal: bool) -> Tuple[str, str]:
        if self.override is not None:
            return self.override, "manual"
        if has_goal:
            return "find", "goal"
        a = activity if isinstance(activity, Activity) else Activity(activity)
        return ACTIVITY_MODE.get(a, "navigate"), a.value

    def step(self, activity, t: float, has_goal: Optional[bool] = None) -> str:
        """The mode for this activity at time t; applies the mode's settings when it changes (`changed` tells)."""
        self._t = float(t)
        if not self._clock_set:
            self._clock_set = True
            self.since = self._t
            if self.history:
                self.history[0] = (self._t, *self.history[0][1:])
        if has_goal is None:
            has_goal = bool(self.goal_fn()) if self.goal_fn is not None else False
        mode, reason = self.target_mode(activity, bool(has_goal))
        self.changed = mode != self.mode
        if self.changed:
            self._apply(mode, self._t, reason)
        return self.mode

    def force_mode(self, mode: Optional[str]) -> Optional[str]:
        """Manual override (keyboard): a mode name pins it, None / "auto" returns to automatic at the next step.
        A pinned mode is applied at once."""
        if mode in (None, "auto"):
            self.override = None
            return None
        if mode not in self.modes:
            raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(self.modes)} or auto")
        self.override = mode
        self.changed = mode != self.mode
        if self.changed:
            self._apply(mode, self._t, "manual")
        return mode

    def cycle_override(self) -> Optional[str]:
        """One key for the demo: auto -> navigate -> scene -> read -> find -> auto. Returns the new override."""
        order: List[Optional[str]] = [None, *[m for m in MODES if m in self.modes]]
        i = order.index(self.override) if self.override in order else 0
        return self.force_mode(order[(i + 1) % len(order)])

    # ---- read mode -----------------------------------------------------------------------------------------
    @property
    def should_mute_objects(self) -> bool:
        """True when the caller must drop non-critical object cues (read mode and the gate cannot mute them)."""
        return self.settings.objects_muted and not self._gate_mutes

    def filter_cues(self, cues: Optional[Iterable]) -> List:
        """Drop non-critical object cues while should_mute_objects; keeps text cues, haptics, interrupts, cues with
        severity >= critical_severity, and any object that is approaching (ttc under approach_ttc_s): severity is a
        fixed value per class (a curb is 0.8, a person 0.5), so it cannot be the only test for danger. Returns the
        cues unchanged otherwise."""
        cues = list(cues or [])
        if not self.should_mute_objects:
            return cues

        def approaching(c) -> bool:
            ttc = getattr(c, "ttc_s", None)
            return ttc is not None and math.isfinite(ttc) and ttc < self.approach_ttc_s
        return [c for c in cues if getattr(c, "kind", "object") == "text"
                or getattr(c, "action", "speak") in ("haptic", "interrupt")
                or (getattr(c, "severity", 0.0) or 0.0) >= self.critical_severity or approaching(c)]

    def describe(self) -> str:
        s = self.settings
        src = "manual" if self.override is not None else self.reason
        return (f"mode {self.mode} ({src}) hold {s.hold_static_s:.0f}s background {'on' if s.background else 'off'} "
                f"verbosity {s.verbosity}" + (" objects muted" if s.objects_muted else ""))
