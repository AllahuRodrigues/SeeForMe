"""AnnouncementGate: a "what did I already tell the user?" memory in front of any scheduler (live demo only).

The schedulers decide per track id and forget quickly (P-CACS blocks a track for `recent_block_s` = 2.8 s and
its novelty recovers with tau = 10 s). On a desk webcam that turns the user's own body into "Person." /
"Person, about 1.5 meters." every few seconds, and every tracker id switch looks like a brand-new person.
The gate remembers announcements by *object identity* instead of track id:

    identity   label (with a few aliases: truck/bus -> car, ...) for objects; normalised text for signs, where two
               OCR readings are one sign only if their numbers are identical and the rest nearly so ("ROOM 204" /
               "R0OM 2O4" are one sign; "room 204" / "room 206", "gate 12" / "gate 13", "exit" / "exit 2" are not)
    sector     left (< -sector_deg) | ahead | right (> +sector_deg) of the box centre, with hysteresis
    memory     last announcement time, distance, angle, track id, box and cue of that identity in that sector

Safety principle: the gate only removes cues that carry no new safety information. Silence about the desk user
themself and about static repeats is the goal; silence about something approaching the user, in the walking path,
a vehicle closing in, or a FIND goal coming into view is a bug.

A cue passes when one of these is true, checked in this order:

    critical      measured TTC < critical_ttc_s and path_prob >= critical_path (passes even unconfirmed). For a track
                  the gate still holds as the operator, background or warm-up object it also needs a measured closing
                  speed of at least approach_min_mps (the user leaning toward the laptop at 0.8 m is "TTC 0", not a
                  hazard). At most one critical cue per object every critical_gap_s, unless its TTC dropped by
                  critical_worse_ttc_s or its distance by critical_worse_m since the last critical cue (worsening)
    approach      a background object or the operator that was released because it measurably closes in (below): its
                  first cue after the release passes (within approach_keep_s), whatever was said about it before
    new           no memory of this identity in this sector (a second object of the same label that was in view
                  together with the remembered one, as a separate box at the same decision, counts as new: an id
                  switch never has both tracks visible at once; a duplicate box of the same object overlaps it)
    sector        the same track was last announced in another sector (at most every min_gap_s)
    closer        distance dropped by max(closer_m, closer_frac x distance at the last announcement) (min_gap_s)
    near          no measured depth (depth off, or depth_conf < 0.4), so nothing is ever critical: an object in the
                  walking path or corridor that is near (range < nodepth_near_m, or its box bottom in the lowest
                  nodepth_bottom_frac of the frame) and closing at >= nodepth_approach_mps, by the range rate over >= 1 s
                  or by the box growing over 0.5-1 s (only the sides the frame edge does not cut count: a floor obstacle
                  a step away is cut off at the bottom, so its box-height range *grows* while the user walks into it),
                  at most every min_gap_s. A static object (the user at a desk) does not loom, so it is not repeated
    closing       an object in the walking path or corridor nearer than bg_path_m that is max(release_m, release_frac x
                  range) nearer than when it was last said and still closing (measured depth, else the box), at most
                  every min_gap_s: finer than `closer` for what is a few steps away (a person or a car that starts
                  coming at a user who stands still is said again before its TTC turns critical)
    reminder      hold time elapsed: hold_goal_s for a FIND goal, hold_moving_s if the object moves, else hold_static_s
    interrupt     the scheduler wants to cut the current message for it, the object moves or approaches (measured),
                  and it was said at least min_gap_s ago
and it is held back when:

    unconfirmed   the track has fewer than min_hits matches (or, frame-rate independent, 2 matches spanning
                  confirm_age_s); the cue is deferred, not dropped: it is said at a later decision (within pending_s)
                  once the track is confirmed, so a flicker is never spoken but a real object waits one decision step
    warmup        during the warm-up (the first warmup_s after the camera first sees something): a track the gate has
                  seen static for warm_min_samples box samples spanning warm_hold_span_s (never on its first samples)
                  and that is not a vehicle, not closing in and not in the walking path nearby, or that looks like the
                  operator
    background    static through the warm-up (warm_min_samples box samples spanning bg_min_span_s inside it, no
                  measurable closing), until it is released. Vehicles (car, bus, truck, bicycle, motorcycle, ...) and
                  tracks closing in never become background. An object in the walking path (path_prob >= critical_path,
                  or its box inside the walking corridor) nearer than bg_path_m is said once before it is silenced
    operator      operator_filter (desk webcam): the user, i.e. the largest centred person present and still through
                  the warm-up and at most operator_max_m away (measured depth; without depth the nearer of the box-height
                  and box-width estimates, since a desk user's box is cut off at the bottom), until released. A person
                  standing 2 m in front of the camera is an ordinary object (background if static)
    critical_gap  a critical cue for the same object less than critical_gap_s ago and the hazard did not worsen
    haptic        a haptic for the same identity and sector less than haptic_gap_s ago
    repeat        none of the pass rules hold
A FIND goal (label or sign text, multi-word goals match as a phrase) is never held as warm-up or background, nor as
the operator unless it is the user themself (FIND "person" at a desk must not announce the user every 8 s).

Release. A background object or the operator becomes an ordinary object (so the critical rule applies) as soon as it
measurably closes in: its distance is below its reference by max(release_m, release_frac x reference) (measured
depth: the median over the last release_smooth_s; without depth: the box's unclipped width, else height, else area
over the last 3 frames against its reference size), or its latest sample is that much nearer while it closes at
>= release_fast_mps over the last 0.5 s (a median lags a fast approach), or, without measured depth, it looms in the
walking path (the near rule), or it closes at >= release_mps (measured) at every decision for release_span_s, or its
measured TTC is under release_ttc_s while it closes at approach_min_mps. That is an "approach" release (its next cue
passes). A box that moved by bg_release_disp is released without one. The reference starts as the warm-up median
and follows the object while it rests (recapture_s at rest: depth within rest_spread_m, box size within
rest_size_frac, not closing), so a lean during the warm-up or the user settling a little nearer is not an approach
later; an object released as approaching that then rests for recapture_s is held again (it came up and stopped:
nothing new until it moves), so a false release costs one cue, never a "Person." loop.

Rescue. A release must not depend on the scheduler offering the object again: P-CACS blocks a track for 2.8 s after
offering it, and its novelty decays, so its one offer may have been held. The gate keeps the last cue it held per
track (held_keep_s) and the last cue it let through per object. At a decision where no offered or pending cue
passes, it says one of them itself (distance and direction phrases refreshed) when that object now qualifies for
safety: a held cue for critical / approach / near / closing, or for new once its warm-up, background or operator hold
no longer applies; a cue said earlier for approach (and then critical) or near, the latter only when the user walks
straight into the object (box centre inside the corridor). When the channel is busy only critical, approach and near
cut in. A held cue's seconds go back to the scheduler's speech budget (and are charged again if the gate says it
later); its 'announced' marks stay on the track, so the scheduler's own recent-block and novelty decay stop it from
offering the same object at every step.

"Measured" TTC / speed means the track has depth (depth_conf >= 0.4) and >= 4 range samples spanning >= 1 s.
Without depth, geometry.TrackDynamics *assumes* the user walks at 1 m/s, so a static person 1.5 m in front of a
laptop gets TTC 0.9 s and would look critical forever; a slope through the first 4 samples (0.3 s) is noise.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..geometry.camera import DEFAULT_SIZE_M, TYPICAL_SIZE_M, CameraModel
from ..types import BBox, Cue, Track, bbox_area, bbox_center, bbox_iou

# labels that are the same physical object to the user when a tracker id switch also flips the class
LABEL_ALIAS = {"truck": "car", "bus": "car", "van": "car", "motorcycle": "bicycle", "motorbike": "bicycle",
               "scooter": "bicycle", "bike": "bicycle", "tram": "train", "pedestrian": "person", "people": "person"}
# identities (after LABEL_ALIAS) that are never background or held for the warm-up: they move under their own power
VEHICLES = frozenset({"car", "bicycle", "train"})
SPEECH_ACTIONS = ("speak", "interrupt")
# which stats counter a hold reason goes to (anything else: suppressed_background)
_HOLD_STAT = {"repeat": "suppressed_repeat", "critical_gap": "suppressed_repeat",
              "unconfirmed": "suppressed_unconfirmed", "haptic_gap": "suppressed_haptic"}
_SILENCE = ("warmup", "background", "operator")
# rescue: what the gate says itself, from a cue it held (the scheduler wanted it said) or one it let through earlier
_RESCUE_HELD = {"critical": 5, "approach": 4, "near": 3, "closing": 3, "new": 1}
_RESCUE_SAID = {"approach": 4, "near": 3}
_RESCUE_SAID_APPROACH = {"critical": 5, **_RESCUE_SAID}      # released as approaching: its critical cue too
_EDGE = 0.01                     # a box side within this of the frame edge is cut off (its size says nothing)


@dataclass
class GateConfig:
    hold_static_s: float = 30.0      # same object, same sector, not materially closer: do not repeat within this
    hold_moving_s: float = 8.0       # ... for an object that is moving
    closer_m: float = 1.0            # re-announce when the distance dropped by at least closer_m
    closer_frac: float = 0.25        # ... and by at least this fraction of the distance at the last announcement
    sector_deg: float = 20.0         # left < -sector_deg, ahead, right > +sector_deg (camera angle of the box centre)
    min_hits: int = 3                # tracker matches before a track's first spoken cue (held cues wait up to pending_s)
    confirm_age_s: float = 0.15      # ... or 2 matches at least this far apart: 3 hits at 10-15 fps, 2 at the 5 fps replays
    haptic_gap_s: float = 3.0        # at most one haptic per identity and sector in this window
    critical_ttc_s: float = 2.0      # measured TTC below this ...
    critical_path: float = 0.5       # ... with path_prob at or above this always passes
    warmup_s: float = 3.0            # objects present and static during the first warmup_s become background
    static_disp: float = 0.03        # max box-centre displacement (normalised) to count as static during the warm-up
    background: bool = True          # suppress background objects unless they start approaching
    operator_filter: bool = False    # desk webcam: the large, centred, static person present from the start is the user
    # refinements beyond the brief (module docstring; evidence from EgoBlind finetune+select replays and a desk simulation)
    hold_goal_s: float = 8.0         # a FIND goal may be repeated this often (the user asked for it)
    pending_s: float = 1.0           # a cue held for confirmation is said within this long once the track is confirmed
    min_gap_s: float = 2.5           # no non-critical repeat of the same object sooner than this (closer / sector / near /
                                     # closing / interrupt)
    switch_gap_s: float = 5.0        # a new track inherits a vanished track's memory only if it appeared within this long
                                     # after the old one was last seen (a different car 30 s later is new, not a repeat)
    sector_hyst_deg: float = 5.0     # an angle must cross a sector boundary by this much to count as a sector change
    moving_speed_mps: float = 0.3    # |range rate| or lateral speed (+0.05 m/s per metre of range) above this = moving
    approach_ttc_s: float = 6.0      # interrupt rule: an object with measured TTC below this ...
    approach_min_mps: float = 0.5    # ... closing at least this fast is approaching (leaning toward a laptop is ~0.2 m/s);
                                     # also the closing speed an operator / background track needs to be critical
    static_size_frac: float = 0.2    # max relative change of the box height to count as static during the warm-up
    bg_release_disp: float = 0.15    # a background object whose box centre moved this far is no longer background
    operator_min_height: float = 0.55   # operator: box height (normalised) above this ...
    operator_center_tol: float = 0.2    # ... box centre within 0.5 +- this horizontally ...
    operator_disp: float = 0.12         # ... and centre displacement during the warm-up at most this (people fidget)
    operator_dup_iou: float = 0.6       # while the user's track is visible, another person box overlapping it this much
                                        # is a duplicate detection of the user (held, but the user box does not move)
    critical_gap_s: float = 2.0      # at most one critical cue per object in this window ...
    critical_worse_ttc_s: float = 0.5   # ... unless its TTC dropped by this much since the last critical cue ...
    critical_worse_m: float = 0.3       # ... or its distance by this much (a worsening hazard always passes)
    nodepth_near_m: float = 2.0      # without measured depth: an object in the path nearer than this ...
    nodepth_approach_mps: float = 0.3   # ... that the range rate or the box growth shows closing this fast: after min_gap_s
    bg_path_m: float = 3.0           # an object in the walking path nearer than this is said once before it is background
    corridor_half_m: float = 0.45    # half-width of the walking corridor (geometry.DynamicsConfig.corridor_half_m)
    # safety: who may be held at all, when a held object is released, and the gate saying a held cue itself
    operator_max_m: float = 1.2      # operator: at most this far away (a laptop user sits 0.4-1.0 m from the webcam)
    release_m: float = 0.35          # background / operator released when its distance is below its reference (where it
                                     # rests) by max(release_m, release_frac x reference) (0.3 m + a margin for 5 cm depth
                                     # jitter: the desk user leans in by 0.25 m); also the step of the `closing` rule ...
    release_frac: float = 0.15
    release_smooth_s: float = 0.3    # ... (measured depth: the median over this long; no depth: the last 3 boxes) ...
    release_fast_mps: float = 0.6    # ... or by its latest sample while closing this fast over the last 0.5 s ...
    release_mps: float = 0.25        # ... or it closes at least this fast (measured) at every decision ...
    release_span_s: float = 1.0      # ... for this long ...
    release_ttc_s: float = 3.0       # ... or its measured TTC is under this while it closes at approach_min_mps
    recapture_s: float = 3.0         # background / operator at rest this long: its reference moves there, and one released
                                     # as approaching is held again (one false release costs one cue, not a "Person." loop)
    rest_spread_m: float = 0.15      # at rest: smoothed depth within this ...
    rest_size_frac: float = 0.08     # ... and the box width / height within this fraction
    warm_min_samples: int = 3        # box samples before the warm-up may hold a track (a first sample is never "static")
    warm_hold_span_s: float = 0.4    # ... spanning at least this long (10 fps: from the first decision after it appears)
    bg_min_span_s: float = 1.0       # background: samples spanning at least this long inside the warm-up, no closing
    held_keep_s: float = 3.0         # the gate says a cue it held itself when the object qualifies within this long
    approach_keep_s: float = 5.0     # after an approach release the object's next cue passes, within this long
    nodepth_grow: float = 0.10       # no measured depth: the box grew by at least this fraction (closing evidence) ...
    nodepth_bottom_frac: float = 0.15   # ... near: box bottom in the lowest nodepth_bottom_frac of the frame (in the corridor)


@dataclass
class _Said:
    """One remembered announcement."""
    identity: Tuple[str, str]
    sector: str
    angle: float
    t: float
    dist: Optional[float]
    track_id: int
    bbox: Optional[BBox]
    goal: bool = False
    seen: float = 0.0                # last decision time its track was in view (id-switch inheritance window)
    apart: Set[int] = field(default_factory=set)   # same-identity tracks seen at the same decision as a separate box
    crit_t: Optional[float] = None   # last critical cue about this object: time, TTC and distance then
    crit_ttc: float = math.inf
    crit_dist: Optional[float] = None
    cue: Optional[Cue] = None        # the cue said (the gate may say it again itself: rescue)
    rng: Optional[float] = None      # range then by measured depth, else by the box (no-depth closing rule)


@dataclass
class _Warm:
    """What the gate saw of one track during the warm-up (box samples from the track's history, one per frame)."""
    identity: Tuple[str, str]
    kind: str
    label: str
    c0: Tuple[float, float]
    h0: float
    first_t: float
    last_t: float
    bbox: BBox
    n: int = 0                       # box samples seen (the track's history, one per frame)
    hits: int = 0                    # tracker matches (a scripted track without box history still counts its frames)
    max_disp: float = 0.0
    max_size: float = 0.0
    path: float = 0.0                # path_prob at the last decision
    closing: bool = False            # a measured closing speed >= release_mps was seen
    dists: List[float] = field(default_factory=list)    # distance at each decision (any source)
    depths: List[float] = field(default_factory=list)   # measured depth (depth_conf >= 0.4) at each decision
    widths: List[float] = field(default_factory=list)   # box widths not cut off at the left / right frame edge
    heights: List[float] = field(default_factory=list)  # box heights not cut off at the top / bottom frame edge
    areas: List[float] = field(default_factory=list)


@dataclass
class _Background:
    identity: Tuple[str, str]
    kind: str
    label: str
    bbox0: BBox                      # box at the end of the warm-up (release-on-displacement reference)
    bbox: BBox                       # last matched box (re-association after an id switch)
    dist_ref: Optional[float]        # median measured depth in the warm-up
    w_ref: Optional[float] = None    # median unclipped box width / height / area in the warm-up (no-depth release)
    h_ref: Optional[float] = None
    a_ref: Optional[float] = None
    track_ids: Set[int] = field(default_factory=set)
    released: bool = False
    say_once: bool = False           # in the walking path and near: said once before it is silenced
    said: bool = False
    approach: bool = False           # released because it closed in (held again once at rest)
    rest: List[Tuple] = field(default_factory=list)   # (t, depth, width, height, area, centre) while not closing


def normalize_text(s: Optional[str]) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


# OCR confusions: letters read in place of digits inside a number, digits read in place of letters inside a word
_OCR_TO_DIGIT = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "z": "2", "b": "8"})
_OCR_TO_LETTER = str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b"})


def sign_key(txt: str) -> Tuple[str, str]:
    """(canonical text, its digit sequence) of a normalised OCR reading, for comparing two readings: a token that is
    at least half digits is a number and letters OCR confuses with digits are read as digits ("2o4" -> "204"); in a
    word such digits are read as letters ("r0om" -> "room")."""
    toks = []
    for tok in txt.split():
        nd = sum(ch.isdigit() for ch in tok)
        if nd and 2 * nd >= len(tok):
            tok = tok.translate(_OCR_TO_DIGIT)
        elif nd:
            tok = tok.translate(_OCR_TO_LETTER)
        toks.append(tok)
    canon = " ".join(toks)
    return canon, "".join(ch for ch in canon if ch.isdigit())


def same_sign(a: str, b: str, ratio: float = 0.8) -> bool:
    """Are two normalised OCR readings the same sign? Only if their numbers are identical (room 204 is not room 206,
    exit is not exit 2), their lengths are similar, and the rest differs by OCR noise (SequenceMatcher >= ratio)."""
    if a == b:
        return True
    ca, da = sign_key(a)
    cb, db = sign_key(b)
    if da != db or abs(len(ca) - len(cb)) > max(1, round(0.2 * max(len(ca), len(cb)))):
        return False
    return ca == cb or SequenceMatcher(None, ca, cb).ratio() >= ratio


def goal_words_match(goal: str, key: str) -> bool:
    """Does the normalised goal ("room 206", "chair") name this normalised label or sign text ("room 206 lab")? As a
    whole word / phrase, so "206" names "room 206" but "20" does not."""
    return bool(goal) and bool(key) and (goal == key or f" {goal} " in f" {key} ")


def sector_of(angle_deg: float, sector_deg: float) -> str:
    if angle_deg < -sector_deg:
        return "left"
    return "right" if angle_deg > sector_deg else "ahead"


def in_sector(angle_deg: float, sector: str, sector_deg: float, hyst_deg: float = 0.0) -> bool:
    """Is the angle inside `sector`, widened by `hyst_deg` on every boundary (hysteresis)?"""
    if sector == "left":
        return angle_deg < -sector_deg + hyst_deg
    if sector == "right":
        return angle_deg > sector_deg - hyst_deg
    return -sector_deg - hyst_deg <= angle_deg <= sector_deg + hyst_deg


def history_span_ok(trk: Track, min_span_s: float = 1.0) -> bool:
    """>= 4 range samples spanning >= min_span_s (whatever their source: depth model or box size)."""
    h = trk.dist_history
    return len(h) >= 4 and (h[-1][0] - h[0][0]) >= min_span_s


def measured(trk: Track, min_span_s: float = 1.0) -> bool:
    """Range rate / TTC come from real depth samples over at least `min_span_s`, not from the assumed 1 m/s walking
    speed (depth_conf < 0.4) or a slope through the first few noisy samples (4 samples over 0.3 s at 10 fps)."""
    return trk.depth_conf >= 0.4 and history_span_ok(trk, min_span_s)


def materially_closer(d_then: Optional[float], d_now: Optional[float], closer_m: float, closer_frac: float) -> bool:
    if d_then is None or d_now is None:
        return False
    return (d_then - d_now) >= max(closer_m, closer_frac * d_then)


def _cut_lr(b: BBox) -> bool:
    return b[0] <= _EDGE or b[2] >= 1.0 - _EDGE


def _cut_tb(b: BBox) -> bool:
    return b[1] <= _EDGE or b[3] >= 1.0 - _EDGE


def _med(xs: Sequence[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


class AnnouncementGate:
    """Wraps any policy (duck-typed: decide/select, reset, name, decision_interval, cfg, ...) and filters its cues.

    `gate.cfg` is the *inner* policy's config (verbosity, units); the gate's own settings are `gate.gate_cfg`
    and change at run time through `set_mode_params(**kw)`. Unknown attributes are read from (and, when the inner
    policy has them, written to) the inner policy, so the gate is a drop-in replacement in the live pipeline.
    """

    _OWN = frozenset({"gate_cfg", "cam", "stats", "reasons", "last_reason", "last_verdict", "releases"})

    def __init__(self, policy, cfg: Optional[GateConfig] = None, camera: Optional[CameraModel] = None,
                 name: Optional[str] = None):
        object.__setattr__(self, "_inner", policy)
        self.gate_cfg = cfg or GateConfig()
        self.cam = camera or getattr(policy, "cam", None) or CameraModel()
        self._name = name
        self._goal_tags: List[str] = [g.lower() for g in (getattr(policy, "goal_tags", None) or [])]
        self._clear()

    # ---- delegation -----------------------------------------------------------------------------------------
    def __getattr__(self, k: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None or k.startswith("__"):
            raise AttributeError(k)
        return getattr(inner, k)

    def __setattr__(self, k: str, v: Any) -> None:
        if k == "goal_tags":                     # keep the gate's own copy (used by _is_goal) in step
            object.__setattr__(self, "_goal_tags", [str(g).lower() for g in (v or [])])
        if k in self._OWN or k.startswith("_") or isinstance(getattr(type(self), k, None), property):
            object.__setattr__(self, k, v)
        elif hasattr(self._inner, k):
            setattr(self._inner, k, v)
        else:
            object.__setattr__(self, k, v)

    @property
    def inner(self):
        return self._inner

    @property
    def name(self) -> str:
        return self._name or getattr(self._inner, "name", "policy")

    @name.setter
    def name(self, v: str) -> None:
        self._name = v

    @property
    def decision_interval(self) -> float:
        return getattr(self._inner, "decision_interval", 0.5)

    @decision_interval.setter
    def decision_interval(self, v: float) -> None:
        self._inner.decision_interval = v

    @property
    def supports_actions(self) -> bool:
        return bool(getattr(self._inner, "supports_actions", False))

    @property
    def cfg(self):
        return getattr(self._inner, "cfg", None)

    @cfg.setter
    def cfg(self, v) -> None:
        self._inner.cfg = v

    def last_observations(self):
        f = getattr(self._inner, "last_observations", None)
        return f() if callable(f) else []

    def _current_goal_tags(self) -> List[str]:
        """The FIND goal in force: the inner policy's goal_tags when it has that attribute (so `gate.goal_tags = [...]`,
        written through to the inner policy, and a goal set on the inner policy directly both count), else the gate's
        own copy (set_goal / gate.goal_tags)."""
        tags = getattr(self._inner, "goal_tags", None)
        if tags is None:
            tags = self._goal_tags
        return [str(g).lower() for g in tags or []]

    def set_goal(self, tags: Optional[Sequence[str]]) -> None:
        """Forward a FIND request; memory of objects and signs the new goal names (as a word or phrase, normalised:
        "room 206" names the sign "ROOM 206 LAB") is dropped, so the next goal cue is said at once."""
        new = [g.lower() for g in (tags or [])]
        old = self._current_goal_tags()
        f = getattr(self._inner, "set_goal", None)
        if callable(f):
            f(tags)
        if new and new != old:
            goals = [k for k in (normalize_text(g) for g in new) if k]
            for ident in list(self._mem):
                if any(self._names(gk, ident[1]) for gk in goals):
                    del self._mem[ident]
        self._goal_tags = new

    @staticmethod
    def _names(goal: str, key: str) -> bool:
        """Does a normalised goal name a normalised label / text? ("seat" names a chair or a bench, as in P-CACS.)"""
        return goal_words_match(goal, key) or (goal == "seat" and key in ("chair", "bench"))

    # ---- state ----------------------------------------------------------------------------------------------
    def _clear(self) -> None:
        self._mem: Dict[Tuple[str, str], List[_Said]] = {}
        self._haptic: Dict[Tuple[Tuple[str, str], str], float] = {}
        self._crit_haptic: Dict[Tuple[Tuple[str, str], str], Tuple[float, float, Optional[float]]] = {}
        self.stats: Dict[str, int] = {"passed": 0, "suppressed_repeat": 0, "suppressed_background": 0,
                                      "suppressed_unconfirmed": 0, "suppressed_haptic": 0, "rescued": 0}
        self.reasons: Counter = Counter()
        self._pending: Dict[int, Tuple[Cue, float]] = {}
        self._held: Dict[int, Tuple[Cue, float, str, float]] = {}    # track id -> (cue, time, hold reason, angle)
        self._approach: Dict[int, float] = {}                        # track id -> time of its approach release
        self._closing_since: Dict[int, float] = {}                   # track id -> since when it closes >= release_mps
        self.releases: List[Tuple[float, int, str]] = []             # (time, track id, why) of every release / re-capture
        self._rescuing = False
        self.last_reason: str = ""
        self.last_verdict: Optional[Dict[str, Any]] = None
        self._rebaseline_state()

    def _rebaseline_state(self) -> None:
        self._t0: Optional[float] = None
        self._warm: Dict[int, _Warm] = {}
        self._warm_done = False
        self._bg: List[_Background] = []
        self._operator: Optional[_Background] = None

    def reset(self) -> None:
        self._clear()
        f = getattr(self._inner, "reset", None)
        if callable(f):
            f()

    def rebaseline(self) -> None:
        """Forget the background / operator and restart the warm-up at the next decision (e.g. the user sat down
        somewhere new). What was said, held cues and approach releases are kept."""
        self._rebaseline_state()

    def set_mode_params(self, **kw) -> Dict[str, Any]:
        """Change GateConfig fields at run time (in place); returns the previous values, so
        `gate.set_mode_params(**old)` restores them. Unknown field names raise ValueError."""
        valid = {f.name for f in fields(GateConfig)}
        bad = sorted(set(kw) - valid)
        if bad:
            raise ValueError(f"unknown GateConfig field(s) {bad}; valid: {sorted(valid)}")
        old = {k: getattr(self.gate_cfg, k) for k in kw}
        for k, v in kw.items():
            setattr(self.gate_cfg, k, v)
        return old

    # ---- scheduler interface --------------------------------------------------------------------------------
    def decide(self, tracks: List[Track], t: float, ch=None, *args, **kw) -> List[Cue]:
        cues = self._inner.decide(tracks, t, ch, *args, **kw)
        return self.filter(cues, tracks, t, ch=ch)

    def select(self, tracks: List[Track], t: float, ctx=None, *args, **kw) -> List[Cue]:
        cues = self._inner.select(tracks, t, ctx, *args, **kw)
        return self.filter(cues, tracks, t)

    def observe(self, tracks: List[Track], t: float) -> None:
        """Warm-up bookkeeping. `filter` calls it; a caller may also call it every frame. Box samples come from the
        track's own history (every frame), so a track is judged static on what was seen of it, not on one sample."""
        g = self.gate_cfg
        if self._warm_done:
            return
        if self._t0 is None:
            # the warm-up clock starts when the camera first sees something (the first frames of a webcam are often
            # dark while auto-exposure settles), so the user is not missed because YOLO found them 1 s late
            if not any(not trk.remembered and trk.misses == 0 for trk in tracks):
                return
            self._t0 = t
        if t >= self._t0 + g.warmup_s:
            self._finish_warmup()
            return
        for trk in tracks:
            if trk.remembered or trk.misses > 0:
                continue
            samples = [(tt, b) for tt, b in trk.bbox_history if tt <= t + 1e-9] or [(t, trk.bbox)]
            w = self._warm.get(trk.track_id)
            if w is None:
                t0, b0 = samples[0]
                w = _Warm(self._identity_of(trk), trk.kind, trk.label, bbox_center(b0), max(1e-3, b0[3] - b0[1]),
                          t0, t0, b0)
                self._warm[trk.track_id] = w
                new = samples
            else:
                new = [(tt, b) for tt, b in samples if tt > w.last_t + 1e-9]
            for tt, b in new:
                self._warm_sample(w, tt, b)
            w.path, w.hits = float(trk.path_prob), max(w.hits, int(trk.hits))
            if trk.distance_m is not None:
                w.dists.append(float(trk.distance_m))
                if trk.depth_conf >= 0.4:
                    w.depths.append(float(trk.distance_m))
            if measured(trk) and -trk.rel_speed_mps >= g.release_mps:
                w.closing = True

    @staticmethod
    def _warm_sample(w: _Warm, t: float, b: BBox) -> None:
        c = bbox_center(b)
        h = max(1e-3, b[3] - b[1])
        w.n += 1
        w.last_t, w.bbox = t, b
        w.max_disp = max(w.max_disp, math.hypot(c[0] - w.c0[0], c[1] - w.c0[1]))
        w.max_size = max(w.max_size, abs(h - w.h0) / w.h0)
        if not _cut_lr(b):
            w.widths.append(b[2] - b[0])
        if not _cut_tb(b):
            w.heights.append(h)
        w.areas.append(bbox_area(b))

    def filter(self, cues: Sequence[Cue], tracks: List[Track], t: float, ch=None) -> List[Cue]:
        """The first cue that passes the gate, as a one-element list, or [].

        A cue held only because its track is not confirmed yet is deferred, not dropped: it is said at the next
        decision (within `pending_s`) once the track is confirmed, if the scheduler has nothing better. The
        scheduler already marked it as announced, so it would not offer it again for recent_block_s. When nothing
        passes, the gate says a held or earlier cue itself if its object now qualifies for safety (rescue).
        `ch` (the ChannelState given to decide) only tells a deferred or rescued cue to wait while the channel is busy."""
        self.observe(tracks, t)
        self._prune(t)
        by_id = {trk.track_id: trk for trk in tracks}
        visible = {trk.track_id: trk for trk in tracks if not trk.remembered and trk.misses == 0}
        self._update_closing(visible, t)
        self._track_records(visible, t)
        idents: Optional[Dict[int, Tuple[str, str]]] = None
        for es in self._mem.values():
            for e in es:
                own = visible.get(e.track_id)
                if own is None:
                    continue
                e.seen = t
                # another track of the same identity in view now, as a separate box: a second object, so it must
                # never inherit this memory as an id switch later (a duplicate box of the same object overlaps it)
                for o in visible.values():
                    if o is own or o.track_id in e.apart or bbox_iou(o.bbox, own.bbox) >= 0.3:
                        continue
                    if idents is None:
                        idents = {tid: self._identity_of(tr) for tid, tr in visible.items()}
                    if idents[o.track_id] == e.identity:
                        e.apart.add(o.track_id)
        for cue in cues or ():
            out = self._judge(cue, by_id.get(cue.track_id), visible, t)
            if out is not None:
                return out
        out = self._flush_pending(visible, t, ch)
        if out:
            return out
        return self._rescue(visible, t, ch)

    def _judge(self, cue: Cue, trk: Optional[Track], visible: Dict[int, Track], t: float, deferred: bool = False) -> Optional[List[Cue]]:
        ok, reason, entry, info = self._check(cue, trk, visible, t)
        self.reasons[reason + (" (deferred)" if deferred else "")] += 1
        self.last_reason = f"{'pass' if ok else 'hold'} {reason}{' (deferred)' if deferred else ''}: {info}"
        self.last_verdict = {"t": t, "track_id": cue.track_id, "label": cue.label, "action": cue.action,
                             "passed": ok, "reason": reason, "info": info, "deferred": deferred}
        if ok:
            self.stats["passed"] += 1
            prev = self._pending.pop(cue.track_id, None)
            if prev is not None and prev[0] is not cue:
                self._refund(prev[0])            # the older held cue for this track will never be said
            if cue.action in SPEECH_ACTIONS:
                self._held.pop(cue.track_id, None)
            self._commit(cue, trk, entry, t, reason)
            return [cue]
        self.stats[_HOLD_STAT.get(reason, "suppressed_background")] += 1
        if reason == "unconfirmed":
            prev = self._pending.get(cue.track_id)
            if prev is not None and prev[0] is not cue:
                self._refund(prev[0])            # superseded by the scheduler's newer cue for the same track
            self._pending[cue.track_id] = (cue, t)
        elif not self._rescuing:                 # (a rescued cue was refunded when it was first held)
            self._refund(cue)
            if cue.action in SPEECH_ACTIONS:     # the gate may say it itself once the object qualifies (rescue)
                angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
                self._held[cue.track_id] = (cue, t, reason, angle)
        return None

    def _flush_pending(self, visible: Dict[int, Track], t: float, ch) -> List[Cue]:
        g = self.gate_cfg
        busy = bool(getattr(ch, "busy", False))
        for tid, (cue, t_held) in sorted(self._pending.items(), key=lambda kv: -kv[1][0].utility):
            trk = visible.get(tid)
            if trk is None or t - t_held > g.pending_s:
                del self._pending[tid]           # the track flickered out: exactly what min_hits is for
                self.reasons["unconfirmed (dropped)"] += 1
                self._refund(cue)
                continue
            if not self._confirmed(trk):
                continue
            action = cue.action
            if action == "interrupt" and not busy:
                action = "speak"
            elif action == "speak" and busy:
                continue                         # wait for the channel (until pending_s runs out)
            del self._pending[tid]
            out = self._judge(replace(cue, t=t, action=action), trk, visible, t, deferred=True)
            if out is not None:
                return out
        return []

    def _rescue(self, visible: Dict[int, Track], t: float, ch) -> List[Cue]:
        """Say a cue the gate held (or one it said earlier) itself when its object now qualifies for safety and the
        scheduler did not offer it again (module docstring: Rescue). At most one, the most urgent."""
        g = self.gate_cfg
        best = None
        for tid, trk in visible.items():
            if not self._confirmed(trk):
                continue
            h = self._held.get(tid)
            if h is not None and 1e-9 < t - h[1] <= g.held_keep_s + 1e-9:
                src, angle0, held_for = h[0], h[3], h[2]
            else:
                e = self._own_entry(tid)
                if e is None or e.cue is None:
                    continue
                src, angle0, held_for = e.cue, e.angle, None
            cue = self._refreshed(src, angle0, trk, t)
            ok, reason, _, _ = self._check(cue, trk, visible, t)     # (may release it as approaching)
            if held_for is not None:
                allowed = _RESCUE_HELD
            else:                                # a said object: only what is new since it was said
                allowed = _RESCUE_SAID_APPROACH if tid in self._approach else _RESCUE_SAID
            if not ok or reason not in allowed or (reason == "new" and held_for not in _SILENCE):
                continue
            if reason == "near" and held_for is None and not self._in_corridor(trk.label, trk.bbox, self._range_now(trk), True):
                continue                         # said by the gate unasked only when the user walks straight into it
            key = (allowed[reason], float(cue.utility or 0.0))
            if best is None or key > best[0]:
                best = (key, cue, trk, reason)
        if best is None:
            return []
        _, cue, trk, reason = best
        if bool(getattr(ch, "busy", False)):
            if reason not in ("critical", "approach", "near"):
                return []
            cue = replace(cue, action="interrupt")
        else:
            cue = replace(cue, action="speak")
        self._rescuing = True
        try:
            out = self._judge(cue, trk, visible, t, deferred=True)
        finally:
            self._rescuing = False
        if out:
            self.stats["rescued"] += 1
            self._charge(out[0], trk, t)
        return out or []

    # ---- the decision ---------------------------------------------------------------------------------------
    def _check(self, cue: Cue, trk: Optional[Track], visible: Dict[int, Track], t: float):
        g = self.gate_cfg
        ident = self._identity_of(trk, cue)
        angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
        sec = sector_of(angle, g.sector_deg)
        desc = f"{ident[1] or cue.label} {sec}"
        # background / operator bookkeeping first (it follows the user's box and releases approaching objects), so
        # a track the gate holds as the user or as background is critical only when it measurably closes in
        silenced = self._background_reason(trk, t, visible) if trk is not None else None
        if silenced in ("warmup", "background") and self._is_goal(cue, trk):
            silenced = None                      # a FIND goal is never background (the operator, the user, stays held)
        if (trk is not None and self._critical(trk)
                and (silenced is None or -trk.rel_speed_mps >= g.approach_min_mps)):
            e = self._lookup(ident, angle, cue.track_id, trk, visible, t)
            if cue.action == "haptic":
                last = self._crit_haptic.get((ident, sec))
            else:
                last = (e.crit_t, e.crit_ttc, e.crit_dist) if e is not None and e.crit_t is not None else None
            held = self._critical_gap(last, trk, t)
            if held:
                return False, "critical_gap", e, f"{desc}, ttc {trk.ttc_s:.1f} s, {held}"
            return True, "critical", e, f"{desc}, ttc {trk.ttc_s:.1f} s"
        if trk is not None and not self._confirmed(trk):
            return False, "unconfirmed", None, f"{desc}, {trk.hits} hits"
        if silenced:
            return False, silenced, None, desc
        if cue.action == "haptic":
            last = self._haptic.get((ident, sec))
            if last is not None and t - last < g.haptic_gap_s:
                return False, "haptic_gap", None, f"{desc}, buzzed {t - last:.1f} s ago"
            return True, "haptic", None, desc
        e = self._lookup(ident, angle, cue.track_id, trk, visible, t)
        released = self._approach.get(cue.track_id)
        if released is not None and t - released <= g.approach_keep_s:
            return True, "approach", e, f"{desc}, closing in (released {t - released:.1f} s ago)"
        if e is None:
            return True, "new", None, desc
        ago = t - e.t
        info = f"{desc}, said {ago:.1f} s ago"
        if e.track_id == cue.track_id and not in_sector(angle, e.sector, g.sector_deg, g.sector_hyst_deg):
            return (True, "sector", e, f"{desc}, was {e.sector}") if ago >= g.min_gap_s else (False, "repeat", e, info)
        d_now = trk.distance_m if trk is not None else cue.distance_m
        if materially_closer(e.dist, d_now, g.closer_m, g.closer_frac) and ago >= g.min_gap_s:
            return True, "closer", e, f"{desc}, {e.dist:.1f} -> {d_now:.1f} m"
        if trk is not None and ago >= g.min_gap_s and self._near_without_depth(trk):
            rng = self._range_now(trk)
            return True, "near", e, (f"{desc}, {rng:.1f} m, closing (no measured depth), said {ago:.1f} s ago")
        if trk is not None and ago >= g.min_gap_s and self._closing_in_path(trk, e):
            return True, "closing", e, f"{desc}, {self._range_now(trk):.1f} m and closing (said at {e.rng or 0.0:.1f} m)"
        goal = self._is_goal(cue, trk)
        moving = trk is not None and self._moving(trk)
        hold = g.hold_goal_s if goal else (g.hold_moving_s if moving else g.hold_static_s)
        if ago >= hold:
            return True, "reminder", e, f"{info} (hold {hold:.0f} s)"
        if (cue.action == "interrupt" and ago >= g.min_gap_s and trk is not None
                and (moving or self._approaching(trk, e.dist))):
            return True, "interrupt", e, info
        return False, "repeat", e, f"{info} (hold {hold:.0f} s)"

    def _confirmed(self, trk: Track) -> bool:
        g = self.gate_cfg
        return trk.hits >= g.min_hits or (trk.hits >= 2 and trk.last_seen - trk.first_seen >= g.confirm_age_s)

    def _critical(self, trk: Track) -> bool:
        g = self.gate_cfg
        return measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.critical_ttc_s and trk.path_prob >= g.critical_path

    def _critical_gap(self, last: Optional[Tuple[float, float, Optional[float]]], trk: Track, t: float) -> str:
        """'' when a critical cue may pass, else why it is held: one critical cue per object every critical_gap_s,
        unless the hazard worsened since the last one (TTC down by critical_worse_ttc_s or distance by critical_worse_m)."""
        g = self.gate_cfg
        if last is None or t - last[0] >= g.critical_gap_s:
            return ""
        t0, ttc0, d0 = last
        if math.isfinite(ttc0) and trk.ttc_s <= ttc0 - g.critical_worse_ttc_s:
            return ""
        if d0 is not None and trk.distance_m is not None and trk.distance_m <= d0 - g.critical_worse_m:
            return ""
        return f"critical {t - t0:.1f} s ago and not worse"

    # ---- range and closing evidence -------------------------------------------------------------------------
    def _box_ranges(self, bbox: BBox, label: str) -> Tuple[Optional[float], Optional[float]]:
        """Range from the box height and from the box width (typical object size), None for a dimension the frame
        edge cuts off."""
        H, W = TYPICAL_SIZE_M.get(label, DEFAULT_SIZE_M)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rh = self.cam.fy * H / h if h > 1e-3 and not _cut_tb(bbox) else None
        rw = self.cam.fx * W / w if w > 1e-3 and not _cut_lr(bbox) else None
        return rh, rw

    def _range_now(self, trk: Track) -> Optional[float]:
        """Measured depth when there is any (depth_conf >= 0.4), else the box: its width when the frame edge does not
        cut it (a floor obstacle a step away is cut off at the bottom; its box-height range grows as it comes closer),
        else its height, else the tracker's estimate."""
        if trk.depth_conf >= 0.4 and trk.distance_m is not None:
            return trk.distance_m
        rh, rw = self._box_ranges(trk.bbox, trk.label)
        r = rw if rw is not None else rh
        return r if r is not None else trk.distance_m

    def _in_corridor(self, label: str, bbox: BBox, dist: float, strict: bool = False) -> bool:
        """Does the object overlap the walking corridor (`strict`: is its box centre inside it, straight ahead)?"""
        _, width = TYPICAL_SIZE_M.get(label, DEFAULT_SIZE_M)
        half = self.gate_cfg.corridor_half_m + (0.0 if strict else width / 2)
        return abs(self.cam.lateral_from_u(bbox_center(bbox)[0], dist)) <= half

    def _box_closing(self, trk: Track, rng: float) -> float:
        """Closing speed (m/s) the box growth shows over the last ~0.5 s and ~1 s, from the sides the frame edge does
        not cut (box size ~ 1 / range); 0 unless the box grew by at least nodepth_grow."""
        g = self.gate_cfg
        hist = list(trk.bbox_history)
        if len(hist) < 2:
            return 0.0
        t1, b1 = hist[-1]
        best = 0.0
        for lag in (0.5, 1.0):
            prev = [(tt, b) for tt, b in hist if tt <= t1 - lag + 0.05]
            if not prev:
                continue
            t0, b0 = prev[-1]
            dt = t1 - t0
            if dt < 0.3:
                continue
            logs = []
            w0, w1, h0, h1 = b0[2] - b0[0], b1[2] - b1[0], b0[3] - b0[1], b1[3] - b1[1]
            if not _cut_lr(b0) and not _cut_lr(b1) and w0 > 1e-3 and w1 > 1e-3:
                logs.append(math.log(w1 / w0))
            if not _cut_tb(b0) and not _cut_tb(b1) and h0 > 1e-3 and h1 > 1e-3:
                logs.append(math.log(h1 / h0))
            if not logs:
                continue
            grow = math.exp(sum(logs) / len(logs))
            if grow >= 1.0 + g.nodepth_grow:
                best = max(best, rng * (grow - 1.0) / dt)
        return best

    def _near_without_depth(self, trk: Track) -> bool:
        """No measured depth (TTC is assumed, so nothing is critical): an object in the walking path or corridor that
        is near (range < nodepth_near_m, or its box bottom in the lowest nodepth_bottom_frac of the frame) and closing
        at >= nodepth_approach_mps, by the range rate over >= 1 s or by the box growth (sides the frame does not cut)."""
        g = self.gate_cfg
        if measured(trk):
            return False
        rng = self._range_now(trk)
        if rng is None:
            return False
        if not (trk.path_prob >= g.critical_path or self._in_corridor(trk.label, trk.bbox, rng)):
            return False
        if not (rng < g.nodepth_near_m or trk.bbox[3] >= 1.0 - g.nodepth_bottom_frac):
            return False
        if (trk.distance_m is not None and history_span_ok(trk) and trk.distance_m < g.nodepth_near_m
                and trk.path_prob >= g.critical_path and -trk.rel_speed_mps >= g.nodepth_approach_mps):
            return True
        return self._box_closing(trk, rng) >= g.nodepth_approach_mps

    def _smoothed_depth(self, trk: Track) -> Optional[float]:
        """Median of the range samples over the last release_smooth_s (one noisy frame does not count)."""
        h = trk.dist_history
        if not h:
            return trk.distance_m
        t1 = h[-1][0]
        return _med([z for tt, z, _ in h if tt >= t1 - self.gate_cfg.release_smooth_s - 1e-9])

    def _closing_in_path(self, trk: Track, e: _Said) -> bool:
        """closing: an object in the walking path or corridor, nearer than bg_path_m, at least max(release_m,
        release_frac x range) nearer than when it was last said and closing now (>= release_mps; or, by its latest sample,
        nearer by that much while closing at >= release_fast_mps over the last 0.5 s). Measured depth when there is any,
        else the box (sides the frame does not cut). Finer than `closer` (1 m) for what is a few steps away: a person
        or a car that starts coming at a user who stands still is said again before its TTC turns critical."""
        g = self.gate_cfg
        if e.rng is None or trk.distance_m is None:
            return False
        rng = self._range_now(trk)
        if rng is None or rng >= g.bg_path_m:
            return False
        if not (trk.path_prob >= g.critical_path or self._in_corridor(trk.label, trk.bbox, rng)):
            return False
        if measured(trk):
            d, d_last = self._smoothed_depth(trk), trk.dist_history[-1][1]
        else:
            d = d_last = rng
        thr = max(g.release_m, g.release_frac * e.rng)
        fast = self._short_closing(trk)
        if e.rng - d >= thr - 1e-9 and (fast >= g.release_mps or (measured(trk) and -trk.rel_speed_mps >= g.release_mps)):
            return True
        return e.rng - d_last >= thr - 1e-9 and fast >= g.release_fast_mps

    def _moving(self, trk: Track) -> bool:
        """Lateral speed above moving_speed_mps (+0.05 m/s per metre: it is noisier at range), or a measured range
        rate above moving_speed_mps. A TTC under 6 s from a slower drift (0.12-0.3 m/s: depth jitter, the user
        leaning toward the laptop) does not count, or the user at a desk would be 'moving' and repeated every 8 s."""
        g = self.gate_cfg
        if len(trk.dist_history) < 4:
            return False
        z = trk.distance_m if trk.distance_m is not None else 3.0
        if abs(trk.lateral_speed_mps) > g.moving_speed_mps + 0.05 * z:
            return True
        return measured(trk) and abs(trk.rel_speed_mps) > g.moving_speed_mps

    def _approaching(self, trk: Track, dist_ref: Optional[float]) -> bool:
        g = self.gate_cfg
        if materially_closer(dist_ref, trk.distance_m, g.closer_m, g.closer_frac):
            return True
        return (measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.approach_ttc_s
                and -trk.rel_speed_mps >= g.approach_min_mps)

    def _is_goal(self, cue: Cue, trk: Optional[Track]) -> bool:
        if "goal" in (cue.slots or ()):
            return True
        lf = getattr(self._inner, "last_features", None)
        if isinstance(lf, dict):
            f = lf.get(cue.track_id)
            if f is not None and getattr(f, "goal", 0.0) > 0:
                return True
        tags = [k for k in (normalize_text(g) for g in self._current_goal_tags()) if k]
        if not tags:
            return False
        label = (trk.label if trk is not None else cue.label or "").lower()
        keys = {label, LABEL_ALIAS.get(label, label)}
        txt = normalize_text(trk.text if trk is not None and trk.text else cue.text)
        if txt:
            keys.add(txt)
        return any(self._names(gk, k) for gk in tags for k in keys)

    # ---- identity memory ------------------------------------------------------------------------------------
    def _identity_of(self, trk: Optional[Track], cue: Optional[Cue] = None) -> Tuple[str, str]:
        kind = trk.kind if trk is not None else (cue.kind if cue is not None else "object")
        if kind == "text":
            raw = (trk.text if trk is not None and trk.text else None) or (cue.text if cue is not None else None)
            txt = normalize_text(raw)
            if txt and ("text", txt) not in self._mem:
                for ident in self._mem:            # OCR variants of the same sign ("ROOM 204" / "R0OM 2O4")
                    if ident[0] == "text" and same_sign(ident[1], txt):
                        return ident
            return ("text", txt)
        label = (trk.label if trk is not None else cue.label if cue is not None else "object").lower()
        return ("object", LABEL_ALIAS.get(label, label))

    def _lookup(self, ident, angle: float, track_id: int, trk: Optional[Track], visible: Dict[int, Track], t: float) -> Optional[_Said]:
        """The memory entry this cue is about: the entry of its own track, else (tracker id switch) the most recent
        entry of the same identity in the same sector whose track vanished shortly before this one appeared. The
        matched entry adopts the new track id, so a chain of id switches stays one object."""
        g = self.gate_cfg
        entries = self._mem.get(ident)
        if not entries:
            return None
        own = [e for e in entries if e.track_id == track_id]
        if own:
            return max(own, key=lambda e: e.t)
        cands = []
        for e in entries:
            if not in_sector(angle, e.sector, g.sector_deg, g.sector_hyst_deg):
                continue
            other = visible.get(e.track_id)
            # the remembered object is still in view as a separate track: this cue is about a second object,
            # unless the two boxes are the same thing (duplicate detection)
            if other is not None and trk is not None and other is not trk and bbox_iou(other.bbox, trk.bbox) < 0.3:
                continue
            if other is None and trk is not None:
                if trk.first_seen - e.seen > g.switch_gap_s:
                    continue                 # appeared long after the old track was last seen: a different object
                if trk.track_id in e.apart:
                    continue                 # was in view together with the old track: a second object, not a switch
            cands.append(e)
        if not cands:
            return None
        e = max(cands, key=lambda e: e.t)
        if visible.get(e.track_id) is None:
            e.track_id, e.seen = track_id, t
        return e

    def _own_entry(self, track_id: int) -> Optional[_Said]:
        own = [e for es in self._mem.values() for e in es if e.track_id == track_id]
        return max(own, key=lambda e: e.t) if own else None

    def _commit(self, cue: Cue, trk: Optional[Track], e: Optional[_Said], t: float, reason: str = "") -> None:
        g = self.gate_cfg
        ident = self._identity_of(trk, cue)
        angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
        sec = sector_of(angle, g.sector_deg)
        if cue.action == "haptic":
            self._haptic[(ident, sec)] = t
            if reason == "critical" and trk is not None:
                self._crit_haptic[(ident, sec)] = (t, trk.ttc_s, trk.distance_m)
            return
        if cue.action not in SPEECH_ACTIONS:
            return
        self._approach.pop(cue.track_id, None)   # the approach was said
        for b in self._bg:                       # an in-path object said once is background from now on
            if cue.track_id in b.track_ids:
                b.said = True
        dist = trk.distance_m if trk is not None and trk.distance_m is not None else cue.distance_m
        goal = self._is_goal(cue, trk)
        if e is None:
            e = _Said(ident, sec, angle, t, dist, cue.track_id, trk.bbox if trk is not None else None, goal, t)
            self._mem.setdefault(ident, []).append(e)
        else:
            if not in_sector(angle, e.sector, g.sector_deg, g.sector_hyst_deg):
                e.sector = sec
            e.angle, e.t, e.dist, e.track_id, e.goal, e.seen = angle, t, dist, cue.track_id, goal, t
            e.bbox = trk.bbox if trk is not None else e.bbox
        e.cue = cue
        e.rng = self._range_now(trk) if trk is not None else dist
        if reason == "critical" and trk is not None:
            e.crit_t, e.crit_ttc, e.crit_dist = t, trk.ttc_s, trk.distance_m

    def _prune(self, t: float) -> None:
        g = self.gate_cfg
        keep = 2.0 * max(g.hold_static_s, g.hold_moving_s, g.hold_goal_s)   # older entries pass anyway (as reminders)
        for ident in list(self._mem):
            # the 8 entries said most recently (an id switch adopts an old entry, so list order is creation order)
            es = sorted((e for e in self._mem[ident] if t - e.t <= keep), key=lambda e: e.t)[-8:]
            if es:
                self._mem[ident] = es
            else:
                del self._mem[ident]
        for k in [k for k, v in self._haptic.items() if t - v > g.haptic_gap_s]:
            del self._haptic[k]
        for k in [k for k, v in self._crit_haptic.items() if t - v[0] > g.critical_gap_s]:
            del self._crit_haptic[k]
        for k in [k for k, v in self._held.items() if t - v[1] > g.held_keep_s]:
            del self._held[k]
        for k in [k for k, v in self._approach.items() if t - v > g.approach_keep_s]:
            del self._approach[k]

    # ---- background and operator ----------------------------------------------------------------------------
    def _in_path_near(self, label: str, bbox: BBox, dist: Optional[float], path_prob: float) -> bool:
        """In the walking path (path_prob >= critical_path, or the box inside the walking corridor) and nearer than
        bg_path_m. The corridor test matters with depth: a static object straight ahead has path_prob ~0.3 (the
        dynamics halve it when nothing approaches), but the user will walk into it."""
        g = self.gate_cfg
        if dist is None or dist >= g.bg_path_m:
            return False
        return path_prob >= g.critical_path or self._in_corridor(label, bbox, dist)

    @staticmethod
    def _is_vehicle(identity: Tuple[str, str]) -> bool:
        return identity[0] == "object" and identity[1] in VEHICLES

    def _warm_closing(self, w: _Warm) -> bool:
        """Did the track measurably close in during the warm-up? A measured closing speed of release_mps, its depth
        dropping by max(release_m, release_frac x depth), or its box (unclipped width or height) growing by nodepth_grow
        (first third of the samples against the last third)."""
        g = self.gate_cfg
        if w.closing:
            return True
        if len(w.depths) >= 3:               # measured depth: trust it over the box
            k = max(1, len(w.depths) // 3)
            first, last = _med(w.depths[:k]), _med(w.depths[-k:])
            return first - last >= max(g.release_m, g.release_frac * first)
        for xs in (w.widths, w.heights):
            if len(xs) >= 3:
                k = max(1, len(xs) // 3)
                if _med(xs[-k:]) >= (1.0 + g.nodepth_grow) * _med(xs[:k]):
                    return True
        return False

    def _seen_enough(self, w: _Warm, span_s: float) -> bool:
        """Seen long enough to be judged: never on its first sample, warm_min_samples frames (box samples, or tracker
        matches for a track without box history) spanning span_s."""
        g = self.gate_cfg
        return w.n >= 2 and max(w.n, w.hits) >= g.warm_min_samples and w.last_t - w.first_t >= span_s - 1e-9

    def _warm_static(self, w: _Warm, span_s: float) -> bool:
        g = self.gate_cfg
        return (self._seen_enough(w, span_s) and not self._is_vehicle(w.identity)
                and w.max_disp <= g.static_disp and w.max_size <= g.static_size_frac and not self._warm_closing(w))

    def _operator_like(self, w: _Warm, bbox: BBox) -> bool:
        """The user in front of a desk webcam: a large, centred, still person at most operator_max_m away (measured
        depth; without it the nearer of the box-height and box-width ranges: the desk user's box is cut off at the
        bottom, so its height says 'far'; a box cut off on every side fills the frame: near)."""
        g = self.gate_cfg
        if w.kind != "object" or w.identity[1] != "person" or not self._seen_enough(w, g.warm_hold_span_s):
            return False
        if (bbox[3] - bbox[1]) <= g.operator_min_height or abs(bbox_center(bbox)[0] - 0.5) > g.operator_center_tol:
            return False
        if w.max_disp > g.operator_disp:
            return False
        if w.depths:
            dist = _med(w.depths)
        else:
            rs = [r for r in self._box_ranges(bbox, "person") if r is not None]
            dist = min(rs) if rs else 0.0
        return dist <= g.operator_max_m

    def _record(self, w: _Warm, tid: int, say_once: bool = False, said: bool = False) -> _Background:
        half = max(1, w.n // 2)              # a reference size only from a side the frame rarely cut
        return _Background(w.identity, w.kind, w.label, w.bbox, w.bbox, _med(w.depths) if w.depths else None,
                           _med(w.widths) if len(w.widths) >= half else None,
                           _med(w.heights) if len(w.heights) >= half else None,
                           _med(w.areas) if w.areas else None, {tid}, say_once=say_once, said=said)

    def _finish_warmup(self) -> None:
        g = self.gate_cfg
        self._warm_done = True
        t_end = self._t0 + g.warmup_s
        best_op: Optional[Tuple[float, _Warm, int]] = None
        bg: List[Tuple[int, _Warm, bool, bool]] = []
        for tid, w in self._warm.items():
            present = self._seen_enough(w, max(g.bg_min_span_s, 0.5 * g.warmup_s)) and w.last_t >= t_end - 1.0
            if not present:
                continue
            if g.operator_filter and self._operator_like(w, w.bbox):
                h = w.bbox[3] - w.bbox[1]
                if best_op is None or h > best_op[0]:
                    best_op = (h, w, tid)
            if self._warm_static(w, g.bg_min_span_s):
                dref = _med(w.dists) if w.dists else None
                once = self._in_path_near(w.label, w.bbox, dref, w.path)
                said = once and any(e.track_id == tid for es in self._mem.values() for e in es)
                bg.append((tid, w, once, said))
        op_tid = best_op[2] if best_op is not None else None
        for tid, w, once, said in bg:
            if tid != op_tid:                    # the user is the operator, not background
                self._bg.append(self._record(w, tid, say_once=once, said=said))
        if best_op is not None:
            self._operator = self._record(best_op[1], best_op[2])
        self._warm.clear()

    def _update_closing(self, visible: Dict[int, Track], t: float) -> None:
        """Since when each visible track closes in at >= release_mps (measured), at every decision."""
        g = self.gate_cfg
        for tid in [k for k in self._closing_since if k not in visible]:
            del self._closing_since[tid]
        for tid, trk in visible.items():
            if measured(trk) and -trk.rel_speed_mps >= g.release_mps:
                self._closing_since.setdefault(tid, t)
            else:
                self._closing_since.pop(tid, None)

    def _drop(self, trk: Track, rec: _Background, latest: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """(how much nearer than its warm-up reference, the reference) in metres: measured depth (median over
        release_smooth_s) when there is depth now and was in the warm-up, else the box (median of the last 3 frames)
        against its warm-up median: unclipped width, else height, else area, scaled by the reference range.
        `latest`: the last sample only (for a fast approach, where a median lags)."""
        if trk.depth_conf >= 0.4 and rec.dist_ref is not None and trk.dist_history:
            d = trk.dist_history[-1][1] if latest else self._smoothed_depth(trk)
            return rec.dist_ref - d, rec.dist_ref
        boxes = [b for _, b in list(trk.bbox_history)[-(1 if latest else 3):]] or [trk.bbox]
        H, W = TYPICAL_SIZE_M.get(rec.label, DEFAULT_SIZE_M)
        if rec.w_ref and not any(_cut_lr(b) for b in boxes):
            ratio, ref = _med([b[2] - b[0] for b in boxes]) / rec.w_ref, self.cam.fx * W / rec.w_ref
        elif rec.h_ref and not any(_cut_tb(b) for b in boxes):
            ratio, ref = _med([b[3] - b[1] for b in boxes]) / rec.h_ref, self.cam.fy * H / rec.h_ref
        elif rec.a_ref:
            # every side cut off now: the area of a cut box grows slower than the object's, so this under-reads
            ratio = math.sqrt(_med([bbox_area(b) for b in boxes]) / rec.a_ref)
            rs = [r for r in self._box_ranges(rec.bbox0, rec.label) if r is not None]
            ref = min(rs) if rs else None
        else:
            return None, None
        if rec.dist_ref is not None:
            ref = rec.dist_ref
        if ref is None or ratio <= 1e-6:
            return None, None
        return ref * (1.0 - 1.0 / ratio), ref

    def _short_closing(self, trk: Track) -> float:
        """Closing speed over the last ~0.5 s: measured depth (no EMA lag beyond the tracker's), else the box growth."""
        if trk.depth_conf >= 0.4 and len(trk.dist_history) >= 2:
            t1, z1, _ = trk.dist_history[-1]
            prev = [(tt, z) for tt, z, _ in trk.dist_history if tt <= t1 - 0.45]
            if not prev:
                return 0.0
            t0, z0 = prev[-1]
            return (z0 - z1) / max(1e-3, t1 - t0)
        rng = self._range_now(trk)
        return self._box_closing(trk, rng) if rng is not None else 0.0

    def _track_records(self, visible: Dict[int, Track], t: float) -> None:
        """Rest tracking for the background objects and the operator (module docstring: Release). At every decision the
        record's track (if in view and not closing in) adds its smoothed depth and box size; once it has rested for
        recapture_s (depth within rest_spread_m, box size within rest_size_frac, box centre within static_disp x 2) the
        record's references move to where it rests, so a lean during the warm-up or the user settling a little nearer
        is not an approach later, and a record released as approaching is held again: it came up and stopped, and says
        nothing new until it moves again (a false release costs one cue, not a "Person." loop)."""
        g = self.gate_cfg
        if not self._warm_done:
            return
        for rec in ([self._operator] if self._operator is not None else []) + self._bg:
            if rec.released and not rec.approach:
                continue                         # moved away from its place: an ordinary object for good
            trk = next((visible[tid] for tid in rec.track_ids if tid in visible), None)
            if trk is None or trk.track_id in self._closing_since or self._short_closing(trk) >= g.release_mps:
                rec.rest.clear()
                continue
            b = trk.bbox
            rec.rest.append((t, self._smoothed_depth(trk) if trk.depth_conf >= 0.4 else None,
                             None if _cut_lr(b) else b[2] - b[0], None if _cut_tb(b) else b[3] - b[1], bbox_area(b),
                             bbox_center(b)))
            if t - rec.rest[0][0] < g.recapture_s - 1e-9:
                continue
            while len(rec.rest) > 1 and t - rec.rest[1][0] >= g.recapture_s - 1e-9:
                rec.rest.pop(0)                  # keep exactly one sample at or before t - recapture_s
            rest = self._at_rest(rec.rest)
            if rest is None:
                continue
            d, w, h, a = rest
            if d is not None:
                rec.dist_ref = d
            rec.w_ref, rec.h_ref, rec.a_ref = (w or rec.w_ref), (h or rec.h_ref), a
            rec.bbox0 = b
            if rec.released:
                rec.released, rec.approach = False, False
                if rec is self._operator:
                    rec.bbox = b
                for tid in rec.track_ids:
                    self._approach.pop(tid, None)
                where = f"{d:.2f} m" if d is not None else "its box"
                self.releases.append((t, trk.track_id, ("operator" if rec is self._operator else "background")
                                      + f" re-captured: at rest for {g.recapture_s:.0f} s at {where}"))

    def _at_rest(self, samples: List[Tuple]) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], float]]:
        """(depth, width, height, area) medians when the samples show the object at rest, else None."""
        g = self.gate_cfg
        n = len(samples)
        if n < 3:
            return None
        cs = [x[5] for x in samples]
        if max(math.hypot(c[0] - cs[0][0], c[1] - cs[0][1]) for c in cs) > 2 * g.static_disp:
            return None
        meds: List[Optional[float]] = []
        for k in (1, 2, 3):
            xs = [x[k] for x in samples if x[k] is not None]
            if len(xs) < n:                      # a value missing in part of the window (depth dropped, edge cut)
                meds.append(None)
                continue
            spread = max(xs) - min(xs) if k == 1 else max(xs) / max(1e-6, min(xs)) - 1.0
            if spread > (g.rest_spread_m if k == 1 else g.rest_size_frac):
                return None
            meds.append(_med(xs))
        if meds[0] is None and meds[1] is None and meds[2] is None:
            areas = [x[4] for x in samples]
            if max(areas) / max(1e-6, min(areas)) - 1.0 > 2 * g.rest_size_frac:
                return None
        return meds[0], meds[1], meds[2], _med([x[4] for x in samples])

    def _release_why(self, trk: Track, rec: _Background, t: float) -> str:
        """Why a background / operator track is released as approaching, or '' (module docstring: Release)."""
        g = self.gate_cfg
        drop, ref = self._drop(trk, rec)
        if drop is not None and ref is not None and drop >= max(g.release_m, g.release_frac * ref) - 1e-9:
            return f"{drop:.2f} m nearer than its {ref:.1f} m reference"
        if ref is not None:
            last, _ = self._drop(trk, rec, latest=True)
            fast = self._short_closing(trk)
            if (last is not None and last >= max(g.release_m, g.release_frac * ref) - 1e-9
                    and fast >= g.release_fast_mps):
                return f"{last:.2f} m nearer than its {ref:.1f} m reference, closing {fast:.1f} m/s"
        if not measured(trk) and self._near_without_depth(trk):
            return "looming in the walking path (no measured depth)"
        since = self._closing_since.get(trk.track_id)
        if since is not None and t - since >= g.release_span_s - 1e-9:
            return f"closing {-trk.rel_speed_mps:.2f} m/s for {t - since:.1f} s"
        if (measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.release_ttc_s
                and -trk.rel_speed_mps >= g.approach_min_mps):
            return f"ttc {trk.ttc_s:.1f} s closing {-trk.rel_speed_mps:.2f} m/s"
        return ""

    def _release(self, rec: _Background, trk: Track, t: float, approach: bool, why: str = "") -> None:
        rec.released, rec.approach = True, approach
        rec.rest.clear()
        self.releases.append((t, trk.track_id, ("operator " if rec is self._operator else "background ") + why))
        if approach:
            for tid in rec.track_ids | {trk.track_id}:
                self._approach[tid] = t

    def _match_background(self, trk: Track) -> Optional[_Background]:
        ident = self._identity_of(trk)
        for b in self._bg:
            if trk.track_id in b.track_ids:
                return b
        for b in self._bg:
            if not b.released and b.identity == ident and bbox_iou(b.bbox, trk.bbox) >= 0.3:
                b.track_ids.add(trk.track_id)
                return b
        return None

    def _match_operator(self, trk: Track, visible: Optional[Dict[int, Track]] = None) -> bool:
        g, op = self.gate_cfg, self._operator
        if op is None or op.released or trk.kind != "object" or self._identity_of(trk)[1] != "person":
            return False
        if trk.track_id not in op.track_ids:
            mine = [visible[tid] for tid in op.track_ids if tid in visible] if visible else []
            if mine:
                # the user's own track is in view: this is another person (a visitor leaning in beside or behind the
                # user), unless the two boxes are one detection twice; the user's box never moves onto it
                return any(bbox_iou(m.bbox, trk.bbox) >= g.operator_dup_iou for m in mine)
            h = trk.bbox[3] - trk.bbox[1]
            cx = bbox_center(trk.bbox)[0]
            if not (h >= 0.8 * g.operator_min_height and abs(cx - 0.5) <= g.operator_center_tol + 0.1
                    and bbox_iou(op.bbox, trk.bbox) >= 0.3):
                return False
            op.track_ids.add(trk.track_id)   # a tracker id switch of the user
        op.bbox = trk.bbox          # follow the user as they shift in the chair
        return True

    def _background_reason(self, trk: Track, t: float, visible: Optional[Dict[int, Track]] = None) -> Optional[str]:
        """'warmup' / 'background' / 'operator' when the cue must be held for that reason, else None. Releases a
        background / operator track that closes in (an approach release: its next cue passes)."""
        g = self.gate_cfg
        if not (g.background or g.operator_filter):
            return None
        if not self._warm_done:
            w = self._warm.get(trk.track_id)
            if w is None:
                return None
            static = (g.background and self._warm_static(w, g.warm_hold_span_s)
                      and not self._in_path_near(trk.label, trk.bbox, trk.distance_m, trk.path_prob))
            op_like = g.operator_filter and self._operator_like(w, trk.bbox)
            return "warmup" if (static or op_like) else None
        if g.operator_filter and self._match_operator(trk, visible):
            why = self._release_why(trk, self._operator, t)
            if why:
                self._release(self._operator, trk, t, approach=True, why=why)
                return None
            return "operator"
        if g.background:
            b = self._match_background(trk)
            if b is not None and not b.released:
                why = self._release_why(trk, b, t)
                if why:
                    self._release(b, trk, t, approach=True, why=why)
                    return None
                c, c0 = bbox_center(trk.bbox), bbox_center(b.bbox0)
                if math.hypot(c[0] - c0[0], c[1] - c0[1]) > g.bg_release_disp:
                    self._release(b, trk, t, approach=False, why="moved")
                    return None
                b.bbox = trk.bbox
                if b.say_once and not b.said:
                    return None              # in the walking path and near: said once before it is silenced
                return "background"
        return None

    # ---- inner-policy bookkeeping ---------------------------------------------------------------------------
    def _refund(self, cue: Cue) -> None:
        """A held cue is never spoken: give its seconds back to the scheduler's speech budget (P-CACS token bucket).
        The scheduler's 'announced' marks on the track stay, so its own recent-block and novelty decay keep it from
        offering the same repeat or background object at every step and starving the other candidates."""
        inner = self._inner
        if cue.action in SPEECH_ACTIONS and isinstance(getattr(inner, "budget", None), (int, float)):
            cap = getattr(getattr(inner, "cfg", None), "budget_capacity_s", None)
            b = inner.budget + float(cue.duration_s or 0.0)
            inner.budget = min(cap, b) if isinstance(cap, (int, float)) else b

    def _charge(self, cue: Cue, trk: Track, t: float) -> None:
        """The gate said a cue itself (rescue): charge the scheduler's speech budget again and mark the track as
        announced, as the scheduler does for its own cues."""
        inner = self._inner
        if cue.action in SPEECH_ACTIONS and isinstance(getattr(inner, "budget", None), (int, float)):
            cap = getattr(getattr(inner, "cfg", None), "budget_capacity_s", None)
            b = inner.budget - float(cue.duration_s or 0.0)
            inner.budget = max(-cap, b) if isinstance(cap, (int, float)) else b
        trk.last_announced, trk.last_announced_bbox = t, trk.bbox
        trk.announce_count += 1
        ad = getattr(inner, "announced_dist", None)
        if isinstance(ad, dict) and trk.distance_m is not None:
            ad[trk.track_id] = trk.distance_m

    def _refreshed(self, cue: Cue, angle0: float, trk: Track, t: float) -> Cue:
        """A copy of an earlier cue for saying now: its distance and direction phrases follow the object."""
        from ..cues.compose import direction_phrase, distance_phrase
        msg = cue.message or ""
        units = getattr(self.cfg, "units", "m") or "m"
        d_now = self._range_now(trk)
        if msg and cue.distance_m is not None and d_now is not None:
            old, new = distance_phrase(cue.distance_m, trk.depth_conf, units), distance_phrase(d_now, trk.depth_conf, units)
            if old and new and old != new and f", {old}" in msg:
                msg = msg.replace(f", {old}", f", {new}", 1)
        if msg:
            angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0])
            old, new = direction_phrase(angle0), direction_phrase(angle)
            if old != new and f", {old}" in msg:
                msg = msg.replace(f", {old}", f", {new}", 1)
        ttc = trk.ttc_s if measured(trk) and math.isfinite(trk.ttc_s) else cue.ttc_s
        return replace(cue, t=t, message=msg, distance_m=d_now if d_now is not None else cue.distance_m, ttc_s=ttc)

    # ---- reporting ------------------------------------------------------------------------------------------
    def describe(self) -> str:
        s, g = self.stats, self.gate_cfg
        held = sum(v for k, v in s.items() if k.startswith("suppressed_"))
        bg = sum(1 for b in self._bg if not b.released)
        op = "on" if g.operator_filter else "off"
        if g.operator_filter:
            op += " (user found)" if self._operator is not None and not self._operator.released else " (no user)"
        return (f"gate[{self.name}]: {s['passed']} passed ({s.get('rescued', 0)} said by the gate itself), {held} held "
                f"(repeat {s['suppressed_repeat']}, background {s['suppressed_background']}, unconfirmed "
                f"{s['suppressed_unconfirmed']}, haptic {s['suppressed_haptic']}); "
                f"hold {g.hold_static_s:.0f}/{g.hold_moving_s:.0f} s, sector +-{g.sector_deg:.0f} deg, "
                f"background {'on' if g.background else 'off'} ({bg} objects), operator {op}; "
                f"remembers {sum(len(v) for v in self._mem.values())}")
