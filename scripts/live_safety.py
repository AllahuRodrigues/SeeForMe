"""Live safety harness: does anything that decides what the live demo says warn later than the bare scheduler?

The live demo puts the AnnouncementGate (seeforme/live/gate.py) and the LiveController (activity modes, read-mode
muting, desk detection) between the scheduler and the speaker. Their safety principle: never warn later about an
approaching, in-path, vehicle or FIND-goal object than the bare scheduler would (allowance: one 0.5 s decision step,
or at least 1.5 s before contact when the bare cue carries news), and never say or imply a path is clear. Unit tests
pin single cases; this harness fuzzes whole scenes, frame by frame, through the real pieces:

    synthetic detections   objects with a trajectory (range z, lateral x) projected through geometry.CameraModel into
                           boxes (optional box jitter, per-frame dropouts, bursts, 2-frame sightings), at 10 fps
    depth                  "model" (range + Gaussian noise growing with range, depth_conf 0.85), "bbox" (no depth
                           model: perception.depth.BBoxDepthEstimator, depth_conf 0.25) or "none" (no range at all)
    tracking               tracking.IoUTracker + geometry.TrackDynamics (PipelineConfig().tracker)
    schedulers             make_live_policy("pcacs", load_cfg()) and "pcacs_learned:rl", a decision every 0.5 s, on a
                           speech channel like LivePipeline's (eval.runner.SpeechChannel: a message holds the channel
                           for its spoken length, an interrupt cuts it). A "speak" on a busy channel is dropped, never
                           heard, as the live speakers do (feedback/speech.py Speaker.say); the count is reported
    configurations         bare (the scheduler alone), gate (AnnouncementGate(GateConfig())), gate_nobg (background
                           off), gate_op (operator_filter on), and LiveController in setups auto / handheld / desk and
                           with the mode pinned to scene / read / find / navigate (ctl:auto, ctl:handheld, ctl:desk,
                           ctl:scene, ctl:read, ctl:find, ctl:navigate); "+cane" (cane family): the controller with the
                           cane on, fed the scene's metric depth maps (ctl:handheld+cane, ctl:navigate+cane,
                           ctl:auto+cane, ctl:find+cane); "+reply" (reply_cut family): spoken replies to voice commands
                           on the real channel by scripts/run_live.py speak_reply's rules (a reply cuts what plays
                           unless the last cue, under 6 s ago, is commands.is_safety_cue; then it waits), at random
                           times and right after cues (ctl:handheld+reply, ctl:navigate+reply, ctl:auto+reply,
                           ctl:find+reply). Every run gets its own deep copy of the scheduler and its config, and its
                           own copy of each track, kept while the tracker holds it (a missed frame keeps the
                           scheduler's marks and the gate's object identity, as in the live pipeline)
    depth maps (cane)      a floor and boxes (the obstacles, and the detected objects at their typical size) rendered for
                           geometry.CameraModel's camera (1.35 m up, 15 degrees down, 70 degrees), 160 x 120, 2 % noise,
                           every 2nd frame (make_depth)
    camera frames (cane)   a texture scrolling down by 0.2 frame widths per metre walked, so the activity detector's
                           frame path sees the walk (make_motion); the other families feed no frames (its track path)

Families (each seeded: `family:seed` builds the same scene every time):

    approach         person, car, bicycle or chair (the user walking into it) at random ranges, speeds and lateral
                     offsets; present and static from the start (the gate's background) or coming into view
    static_path      a static object ahead; the user stands, then walks into it (everything else in view closes too)
    side_entry       a person, bicycle, dog or car steps in from the side and then comes at the user or stops in the path
    two_hazards      two approaching objects, sometimes of one label
    flicker          approaches under heavy detection dropouts: per-frame misses, bursts, 2-frame sightings
    id_switch        dropouts longer than the tracker's max_age and box glitches (new track ids), the approach starting
                     while out of view
    cross_switch     object A (said) leaves; B of the same label appears where A was, nearer or farther, and approaches
    twins            two objects of one label side by side (two cars a car's width apart, the parked one off the
                     corridor); one stays, the other walks at the user
    lost_background  a background object drops out (or leaves) and it, or another of its label at its place, approaches
    find             a FIND goal (object or sign; a same-label distractor or a sign with another number) set mid-run,
                     in view since the start (background) or coming into view, with or without the user walking to it
    desk             a desk webcam: still, fidget, leans 0.15-0.6 m (one or several), drift, an arm (a second person box),
                     stand up and sit down, a stranger walking up after the user leaves (after 0-3 leans), a visitor
                     walking up in front of the user
    read_scene       read / scene mode with a sign in view and an object that starts to approach
  the second verification's probe families (kept so the sweep covers them):
    deskwalk         2-4 deep leans at a desk, then a stranger (also to the user's very seat) or a visitor
    swap             two objects of one label cross (the tracker may swap ids); the one in the path then approaches
    find_behind      a FIND goal behind a background object (occluded while it covers it), which may step aside
    read_car         a sign and a fast car / bicycle / person in read and scene modes; the depth may turn off mid-run
    rescue_switch    an object said (background or a repeat) starts to approach, with dropouts right after the start
    depth_flip       an approach whose depth source changes mid-approach (model, box only, none)
    bgref            a background object drops out for 0.1-0.7 s; another of its label steps in 0.6-3 m farther
    handover         the occlusion handover: another person steps in where one stood within 0-0.4 s (often in the
                     same frame), 1-2.5 m farther, and walks at the user at 1-1.8 m/s
    reacq            a static person or chair in the path with periodic dropouts: no hazard, counts repetition (the
                     by-family table's "max cues per run")
  the fourth verification's probe families (ext4.py, unchanged; v4_<name>:N builds its scene):
    swap_fast        FIND: the goal object H in the path, said, starts to walk at the user; another goal object D off
                     the path comes into view unsaid just before (the FIND swap at motion onset)
    flip_size        an object not of its label's typical size (0.65-0.85 or 1.15-1.35) stands in the path, then comes
                     at the user; the depth switches (model to box or none, or back) as it does
    small_static     no measured depth (mostly), a small object (0.55-0.85 of the typical size) static in the path
                     1.8-3 m away for 4-20 s, then it comes at the user (long static spells)
    desk_still       a still or fidgeting desk user with the depth switched off and on (still_flip), or a flickering
                     box (flicker), or a stranger after the user leaves, or a visitor; a rebaseline in some
    reply_cut        replies that cut gated cues right after they start (+reply configurations)
  the third verification's probe families (ext3.py, unchanged; v3_find_disp:N and v3_displaced:N build its scenes):
    find_disp        FIND a sign set while it is in view (the gate says it itself, a step late), another sign appears
                     (a routine cue that waits behind that message), a person, dog or bicycle comes at the user
    displaced        the gate's own message about A (lost, deferred, rescued), a routine cue about B offered while it
                     plays, a hazard H as B would have ended alone
    cane             the cane on: the user walks a corridor with 2-4 obstacles the detector cannot name (pole, barrier
                     panel, head-height sign box, low box), routine objects to the side, and a hazard (head-on person,
                     bicycle or dog, a side entry, a chair in the path) timed around a cane cue; in half the scenes
                     the user stands still for 1-8 s first (auto setup may take the camera for a desk until the walk
                     ends that)

Each hazard H (the approaching, in-path or goal object of the scene) has a window [t_lo, contact) (contact: range
under 0.6 m inside the walking corridor; no contact: to the end of the run). For every gated configuration:

    news           a bare cue about H in the window carries news unless the gated user heard H in the 2 s before, or
                   heard H in the 8 s before while it was less than max(0.35 m, 15 %) farther than it is now, or heard H
                   with its range at any time before and H stayed within that tolerance of where it was then (a static
                   object), or heard, in the 8 s before, another object of its label (a sign: its text) at H's place now
                   with its range said: within 1 m laterally, H not nearer than it was by that tolerance nor farther by
                   the gate's `closer` step, max(1 m, 25 %) (a chair where the said one stood is the same thing to the
                   user; a person 3 m behind the one said is not), never the desk user (naming the user says nothing
                   about a visitor in front of them). None of these counts if H has started to come at the user since
                   (its approach speed, ground truth with the user's own walk taken out, up by 0.5 m/s: a person who
                   starts to walk at the user is news before they are any nearer). A gated message an interrupt or a
                   reply cut counts only up to the cut: its object if the words before its first comma were said, its
                   range if its distance phrase was. Read mode: only bare cues the read-mode filter keeps (text,
                   haptic / interrupt, severity >= 0.9, TTC under 6 s)
    bare warning   the first bare cue that carries news; gated warning: the first gated cue (spoken or haptic) about
                   H in the window that named H before any cut
    severe         bare warns at least 1.5 s before contact (or there is no contact) and the gated system is silent,
                   or warns later than bare + 0.5 s and less than 1.5 s before contact; and, for every later bare cue
                   with news at least 1.5 s before contact, the same against the first gated cue about H from then on
                   (a system that names H once early and then stays silent while it closes in is severe)
    late           a gated warning later than bare + 0.5 s (or missing), for any of those: the softer statistic
    LiveController rows found severe or late against the bare scheduler are re-judged against the same controller
    without the gate (ctl:...:nogate): "gate" when still severe there, else "controller" (the adapter's own effect)

Desk noise (desk, deskwalk, desk_still): cues about the desk user after 8 s (past the warm-up and auto setup's desk
detection and re-warm-up). Acceptance (operator configurations gate_op, ctl:desk, ctl:auto): no cue from a still,
fidgeting, drifting, arm, leaning (< 0.35 m), depth-switching or flickering user, at most one per deep-lean or
standing-up run, the first warning about a stranger or visitor never late.

    OMP_NUM_THREADS=2 .venv/bin/python scripts/live_safety.py --families all --seeds 3000 --policies pcacs,rl --jobs 4
    OMP_NUM_THREADS=2 .venv/bin/python scripts/live_safety.py --repro cross_switch:17 --policies rl --configs gate
    OMP_NUM_THREADS=2 .venv/bin/python scripts/live_safety.py --families desk,approach --seeds 40 --no-write

The scheduler's view (--diverge): in every configuration (SAME_VIEW_CONFIGS: no mode sets the scheduler's settings,
and neither a cane cue nor a reply is on the channel it is shown), the scheduler must be shown at every decision exactly
the channel it is shown alone (busy, remaining time, current track) and make exactly the offers it makes alone;
--diverge lists every run where it is not.

    OMP_NUM_THREADS=2 .venv/bin/python scripts/live_safety.py --diverge --families all --seeds 1200 --jobs 4

--seeds N is the number of scenes, dealt round-robin over the families (seed i goes to family i mod #families, so
`family:seed` ids stay the same whatever N is). Writes results/live/safety_harness.{md,json} unless --no-write.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for p in (str(Path(__file__).resolve().parent), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

FAMILIES = ("approach", "static_path", "side_entry", "two_hazards", "flicker", "id_switch", "cross_switch", "twins",
            "lost_background", "find", "desk", "read_scene",
            # second verification's probe families (gate_verify_r2), and the occlusion handover
            "deskwalk", "swap", "find_behind", "read_car", "rescue_switch", "depth_flip", "bgref", "handover", "reacq",
            # third verification (gate_verify_r3): routine cues waiting behind the gate's own message; and the cane
            "find_disp", "displaced", "cane",
            # fourth verification (gate_verify_r4): FIND swaps at motion onset, sizes and depth switching, small static
            # objects, desk depth flips and flicker, spoken replies that cut cues
            "swap_fast", "flip_size", "small_static", "desk_still", "reply_cut")
# the verifier's names of the families kept here (the same builder; the name seeds the scene, so ids reproduce)
ALIASES = {"v3_find_disp": "find_disp", "v3_displaced": "displaced", "v4_swap_fast": "swap_fast",
           "v4_flip_size": "flip_size", "v4_small_static": "small_static", "v4_desk_still": "desk_still",
           "v4_cut_label": "reply_cut"}
WALK_CONFIGS = ("gate", "gate_nobg", "gate_op", "ctl:auto", "ctl:handheld", "ctl:desk", "ctl:scene", "ctl:read",
                "ctl:find", "ctl:navigate")
DESK_CONFIGS = ("gate", "gate_op", "ctl:auto", "ctl:desk", "ctl:scene")
READ_CONFIGS = ("gate", "gate_nobg", "ctl:auto", "ctl:handheld", "ctl:scene", "ctl:read", "ctl:find", "ctl:navigate")
# the cane on (+cane: LiveController(cane=True), fed the scene's depth maps); ctl:handheld without it for comparison
CANE_CONFIGS = ("gate", "ctl:handheld", "ctl:handheld+cane", "ctl:navigate+cane", "ctl:auto+cane", "ctl:find+cane")
# spoken replies to voice commands on the real channel (+reply: run_live speak_reply's rules); the reference stays bare
# without replies, and a row is blamed on the gate against the same controller with the same replies
REPLY_CONFIGS = ("gate", "ctl:handheld", "ctl:handheld+reply", "ctl:navigate+reply", "ctl:auto+reply", "ctl:find+reply")
# every configuration shows the scheduler exactly what it sees alone (bare): no mode sets the scheduler's settings, and
# neither the cane nor a reply is on the channel it is shown (--diverge checks all of a scene's configurations)
SAME_VIEW_CONFIGS = tuple(dict.fromkeys(WALK_CONFIGS + DESK_CONFIGS + READ_CONFIGS + CANE_CONFIGS + REPLY_CONFIGS))
OPERATOR_CONFIGS = ("gate_op", "ctl:desk", "ctl:auto")
POLICY_NAMES = {"pcacs": "pcacs", "rl": "pcacs_learned:rl", "il": "pcacs_learned:il"}
FPS = 10
CONTACT_M = 0.6                  # contact: range under this, inside the walking corridor
CORRIDOR_HALF_M = 0.45           # (GateConfig.corridor_half_m)
SEVERE_S = 1.5                   # a warning at least this long before contact is in time
STEP_S = 0.5                     # one decision step of allowance
NEWS_S = 8.0                     # a gated cue about H this recently, at about the range, makes a bare cue no news ...
JUST_S = 2.0                     # ... and so does one about H this recently, at any range (and, with no time limit, one
                                 # about H heard at its range when H has stayed at that place since) ...
ONSET_MPS = 0.5                  # ... unless H has started to come at the user since: its approach speed (ground truth,
                                 # the user's own walk taken out) is this much higher than when the user heard of it
PLACE_M = 1.0                    # ... or one about another object of H's label (sign: text) this near H laterally ...
CLOSER_M, CLOSER_FRAC = 1.0, 0.25   # ... and H not farther than it by this step (GateConfig.closer_m / closer_frac)
DESK_AFTER_S = 8.0               # desk noise counts from here (warm-up, desk detection, re-warm-up)
DEEP_LEAN_M = 0.35


# ---- scenes ----------------------------------------------------------------------------------------------------------
@dataclass
class Obj:
    """One object of a scene. z(t) is its range along the walking direction (m, camera frame: the user's own walking
    is folded in), x(t) its lateral offset (m, + right); `on(t)` whether it is physically there; `drops` frame indices
    without a detection. `box(t)` overrides the projection (the desk user, whose box the frame cuts off)."""
    name: str
    label: str
    z: Callable[[float], float]
    x: Callable[[float], float] = lambda t: 0.0
    on: Callable[[float], bool] = lambda t: True
    kind: str = "object"
    text: Optional[str] = None
    base: float = 0.0
    box: Optional[Callable[[float], Optional[Tuple[float, float, float, float]]]] = None
    drops: frozenset = frozenset()
    hazard: bool = False
    t_lo: float = 0.0
    role: str = ""
    contact: Optional[float] = None


@dataclass
class Scene:
    family: str
    seed: int
    T: float
    objs: List[Obj]
    depth: str = "model"              # model | bbox | none
    noise: float = 0.03               # depth noise sigma (m) at 2 m; grows with range
    jitter: float = 0.0               # box jitter sigma (normalised units)
    goal: Optional[List[str]] = None
    t_goal: float = 0.0
    rebaseline_at: Optional[float] = None   # the gate rebaselines here (gate configurations)
    variant: str = ""
    params: Dict[str, object] = field(default_factory=dict)
    configs: Tuple[str, ...] = WALK_CONFIGS
    # obstacles the detector cannot name (cane family): t -> (x0, x1, y0, y1, z0, z1) in metres (x right, y up from the
    # floor, z forward, the user's walk folded in) or None; with them the scene has metric depth maps (make_depth)
    obstacles: List[Callable[[float], Optional[Tuple[float, float, float, float, float, float]]]] = field(default_factory=list)
    # the user's walk (m walked by t): with it the controller gets camera frames that move as a walk does (make_motion;
    # the activity detector's frame path), else none (its track path)
    walk: Optional[Callable[[float], float]] = None

    @property
    def sid(self) -> str:
        return f"{self.family}:{self.seed}"


def _ramp(z0: float, v: float, t0: float, zmin: float = 0.25) -> Callable[[float], float]:
    return lambda t: max(zmin, z0 - v * max(0.0, t - t0))


def _slide(x0: float, x1: float, t0: float, dur: float) -> Callable[[float], float]:
    return lambda t: x0 + (x1 - x0) * min(1.0, max(0.0, t - t0) / max(1e-6, dur))


def _window(a: float, b: float = math.inf) -> Callable[[float], bool]:
    return lambda t: a - 1e-9 <= t < b - 1e-9


def _size(label: str) -> Tuple[float, float]:
    from seeforme.geometry.camera import DEFAULT_SIZE_M, TYPICAL_SIZE_M
    return TYPICAL_SIZE_M.get(label, DEFAULT_SIZE_M)


def _speed(rng: random.Random, label: str) -> float:
    return {"person": rng.uniform(0.5, 1.8), "bicycle": rng.uniform(2.0, 6.0), "car": rng.uniform(2.5, 11.0),
            "dog": rng.uniform(0.8, 2.5)}.get(label, rng.uniform(0.5, 1.4))   # (static things: the user walks)


def _range0(rng: random.Random, label: str) -> float:
    return {"person": rng.uniform(3.0, 13.0), "bicycle": rng.uniform(5.0, 20.0), "car": rng.uniform(8.0, 30.0),
            "dog": rng.uniform(3.0, 9.0)}.get(label, rng.uniform(2.5, 9.0))


def _depth_mode(rng: random.Random, sc: Scene, p_model: float = 0.55) -> None:
    u = rng.random()
    if u < p_model:
        sc.depth, sc.noise = "model", rng.choice([0.02, 0.03, 0.05, 0.08, 0.12])
    else:
        sc.depth = "bbox" if u < p_model + (1 - p_model) / 2 else "none"
    sc.jitter = rng.choice([0.0, 0.0, 0.002, 0.004, 0.008])


def _frame_drops(rng: random.Random, T: float, p: float = 0.0, bursts: int = 0, burst_s: Tuple[float, float] = (0.3, 1.5),
                 t_min: float = 0.0, every: int = 0) -> frozenset:
    """Frame indices without a detection: per-frame misses (p), `bursts` dropouts of burst_s seconds after t_min, or
    `every` k > 0: only 2 frames out of every k are seen (2-frame sightings)."""
    n = int(T * FPS) + 1
    out = {i for i in range(n) if rng.random() < p}
    for _ in range(bursts):
        a = rng.uniform(t_min, max(t_min + 0.1, T - 1.0))
        out |= set(range(int(a * FPS), int((a + rng.uniform(*burst_s)) * FPS) + 1))
    if every > 2:
        ph = rng.randrange(every)
        out |= {i for i in range(n) if (i + ph) % every >= 2}
    return frozenset(out)


def _contact(o: Obj, T: float) -> Optional[float]:
    _, W = _size(o.label)
    for i in range(int(o.t_lo * FPS), int(T * FPS) + 1):
        t = i / FPS
        if o.on(t) and o.z(t) <= CONTACT_M + 1e-9 and abs(o.x(t)) <= CORRIDOR_HALF_M + W / 2:
            return t
    return None


def _finish(sc: Scene, T_max: float = 90.0, tail: float = 1.0) -> Scene:
    for o in sc.objs:
        if o.hazard:
            o.contact = _contact(o, T_max)
    ends = [o.contact for o in sc.objs if o.hazard and o.contact is not None]
    if ends and len(ends) == sum(o.hazard for o in sc.objs):
        sc.T = min(T_max, max(ends) + tail)
    else:
        sc.T = min(T_max, sc.T)
    return sc


def _distractors(rng: random.Random, n: int, ego: Callable[[float], float], labels=("person", "chair", "potted plant",
                 "bench")) -> List[Obj]:
    out = []
    for k in range(n):
        x0, z0 = rng.choice([-1, 1]) * rng.uniform(1.6, 3.0), rng.uniform(2.5, 10.0)
        out.append(Obj(f"d{k}", rng.choice(labels), (lambda z0: lambda t: max(0.25, z0 - ego(t)))(z0), (lambda x0: lambda t: x0)(x0)))
    return out


def _mover(rng: random.Random, name: str, label: str, t_go: float, d0: float, v: float, x0: float, x1: float,
           on: Callable[[float], bool] = lambda t: True, ego: float = 0.0) -> Obj:
    vv = v + ego
    dur = max(0.5, (d0 - CONTACT_M) / max(0.1, vv))
    return Obj(name, label, _ramp(d0, vv, t_go), _slide(x0, x1, t_go, dur), on, hazard=True, t_lo=t_go)


def fam_approach(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["person", "person", "car", "bicycle", "chair"])
    d0, v = _range0(rng, label), _speed(rng, label)
    ego = rng.uniform(0.5, 1.2) if label not in ("chair",) and rng.random() < 0.3 else 0.0
    if rng.random() < 0.5:                       # in view, static, from the start (the gate's background)
        t_go, on = rng.uniform(4.0, 14.0), (lambda t: True)
        t_lo = t_go
    else:                                        # comes into view, then moves
        t_app = rng.uniform(0.5, 10.0)
        t_go, on = t_app + rng.choice([0.0, rng.uniform(0.0, 2.0)]), _window(t_app)
        t_lo = t_app
    x0, x1 = rng.uniform(-1.5, 1.5), rng.uniform(-0.35, 0.35)
    if rng.random() < 0.4:
        x0 = x1                                  # straight down the corridor
    h = _mover(rng, "H", label, t_go, d0, v, x0, x1, on, ego)
    h.t_lo = t_lo
    h.drops = _frame_drops(rng, 120.0, p=rng.choice([0.0, 0.0, 0.03, 0.08]))
    egof = (lambda t: ego * max(0.0, t - t_go))
    sc.objs = [h] + _distractors(rng, rng.choice([0, 0, 1, 2]), egof)
    _depth_mode(rng, sc)
    if rng.random() < 0.15:
        sc.rebaseline_at = round(max(0.5, t_go + rng.uniform(-2.0, 2.0)) * 2) / 2
    sc.T = 60.0
    sc.params = dict(label=label, d0=round(d0, 2), v=round(v, 2), ego=round(ego, 2), t_go=round(t_go, 2),
                     x0=round(x0, 2), x1=round(x1, 2))
    return _finish(sc, 60.0)


def fam_static_path(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["chair", "person", "bench", "potted plant", "fire hydrant", "chair"])
    d0, x0, v = rng.uniform(2.0, 9.0), rng.uniform(-0.35, 0.35), rng.uniform(0.5, 1.4)
    t_go = 0.0 if rng.random() < 0.4 else rng.uniform(3.0, 14.0)
    egof = (lambda t: v * max(0.0, t - t_go))
    h = Obj("H", label, lambda t: max(0.25, d0 - egof(t)), lambda t: x0, hazard=True, t_lo=t_go)
    h.drops = _frame_drops(rng, 120.0, p=rng.choice([0.0, 0.0, 0.05]))
    sc.objs = [h] + _distractors(rng, rng.choice([0, 1, 2, 3]), egof)
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(label=label, d0=round(d0, 2), x0=round(x0, 2), v=round(v, 2), t_go=round(t_go, 2))
    return _finish(sc, 60.0)


def fam_side_entry(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["person", "person", "bicycle", "dog", "car"])
    z_e, side = rng.uniform(1.5, 7.0), rng.choice([-1.0, 1.0])
    x0, x1 = side * rng.uniform(1.8, 3.5), rng.uniform(-0.3, 0.3)
    vx = {"bicycle": rng.uniform(2.0, 4.0), "car": rng.uniform(3.0, 6.0)}.get(label, rng.uniform(0.5, 2.0))
    t_app = rng.uniform(0.5, 10.0)
    t_in = t_app + abs(x1 - x0) / vx
    vz = rng.choice([0.0, rng.uniform(0.3, 1.5)]) if label != "car" else rng.uniform(0.0, 3.0)
    ego = rng.choice([0.0, 0.0, rng.uniform(0.5, 1.2)])
    z = lambda t: max(0.25, z_e - ego * max(0.0, t - t_app) - vz * max(0.0, t - t_in))
    h = Obj("H", label, z, _slide(x0, x1, t_app, t_in - t_app), _window(t_app), hazard=True, t_lo=t_app)
    egof = (lambda t: ego * max(0.0, t - t_app))
    sc.objs = [h] + _distractors(rng, rng.choice([0, 1]), egof)
    _depth_mode(rng, sc)
    sc.T = min(60.0, t_in + 12.0)
    sc.params = dict(label=label, z_e=round(z_e, 2), x0=round(x0, 2), x1=round(x1, 2), vx=round(vx, 2), vz=round(vz, 2),
                     ego=round(ego, 2), t_app=round(t_app, 2))
    return _finish(sc, sc.T)


def fam_two_hazards(rng: random.Random, sc: Scene) -> Scene:
    la = rng.choice(["person", "car", "bicycle", "chair"])
    lb = la if rng.random() < 0.4 else rng.choice(["person", "car", "bicycle", "chair"])
    objs = []
    for name, label, t0 in (("H", la, rng.uniform(1.0, 8.0)), ("H2", lb, rng.uniform(4.0, 16.0))):
        d0, v = _range0(rng, label), _speed(rng, label)
        static = rng.random() < 0.4
        o = _mover(rng, name, label, t0, d0, v, rng.uniform(-1.2, 1.2), rng.uniform(-0.3, 0.3),
                   (lambda t: True) if static else _window(max(0.0, t0 - rng.uniform(0.0, 2.0))))
        objs.append(o)
    sc.objs = objs
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(labels=[la, lb])
    return _finish(sc, 60.0)


def fam_flicker(rng: random.Random, sc: Scene) -> Scene:
    sc2 = fam_approach(rng, Scene(sc.family, sc.seed, 60.0, []))
    h = sc2.objs[0]
    mode = rng.choice(["frames", "bursts", "sightings", "mixed"])
    if mode == "frames":
        h.drops = _frame_drops(rng, 120.0, p=rng.uniform(0.2, 0.6))
    elif mode == "bursts":
        h.drops = _frame_drops(rng, sc2.T + 1, bursts=rng.randint(2, 5), burst_s=(0.3, 1.5), t_min=max(0.0, h.t_lo - 2.0))
    elif mode == "sightings":
        h.drops = _frame_drops(rng, sc2.T + 1, every=rng.randint(4, 12))
    else:
        h.drops = _frame_drops(rng, sc2.T + 1, p=rng.uniform(0.1, 0.4), bursts=rng.randint(1, 3), t_min=h.t_lo)
    sc2.family, sc2.variant = sc.family, mode
    sc2.params["flicker"] = mode
    return sc2


def fam_id_switch(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["person", "car", "chair", "bicycle", "person"])
    d0, v = _range0(rng, label), _speed(rng, label)
    if label == "car":
        d0 = rng.uniform(6.0, 18.0)
    t_go = rng.uniform(4.0, 14.0)
    x0 = rng.uniform(-0.8, 0.8)
    h = _mover(rng, "H", label, t_go, d0, v, x0, rng.uniform(-0.3, 0.3))
    drops: set = set()
    glitch: Dict[int, float] = {}
    for _ in range(rng.randint(1, 3)):
        a = rng.uniform(max(1.0, t_go - 3.0), t_go + 3.0)
        if rng.random() < 0.7:                   # out of view longer than the tracker's max_age (0.8 s): a new id
            drops |= set(range(int(a * FPS), int((a + rng.uniform(0.85, 1.6)) * FPS)))
        else:                                    # one frame with a box far off: the tracker loses and re-finds it
            glitch[int(a * FPS)] = rng.choice([-1, 1]) * rng.uniform(0.25, 0.4)
    h.drops = frozenset(drops)
    if glitch:
        H, W = _size(label)
        from seeforme.geometry.camera import CameraModel
        cam = CameraModel()

        def box(t, h=h, glitch=glitch):
            b = cam.clip(cam.project_box(h.x(t), h.z(t), H, W))
            dx = glitch.get(int(round(t * FPS)))
            if b is None or dx is None:
                return b
            return cam.clip((b[0] + dx, b[1], b[2] + dx, b[3]))
        h.box = box
    sc.objs = [h]
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(label=label, d0=round(d0, 2), v=round(v, 2), t_go=round(t_go, 2), glitches=len(glitch))
    return _finish(sc, 60.0)


def fam_cross_switch(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["person", "car", "chair", "bicycle", "person"])
    xa, za = rng.uniform(-1.0, 1.0), (rng.uniform(5.0, 12.0) if label == "car" else rng.uniform(2.0, 8.0))
    t_leave = rng.uniform(6.0, 15.0)
    walk_off = rng.random() < 0.5
    side = rng.choice([-1.0, 1.0])
    xa_f = (lambda t: xa + side * 1.5 * max(0.0, t - t_leave)) if walk_off else (lambda t: xa)
    a = Obj("A", label, lambda t: za, xa_f, _window(rng.choice([0.0, rng.uniform(0.0, 4.0)]), t_leave + (3.0 if walk_off else 0.0)))
    gap = rng.uniform(0.0, 2.0)
    t_b = t_leave + gap
    zb = max(1.5, za + rng.uniform(-2.5, 4.0))
    v = _speed(rng, label)
    b = _mover(rng, "B", label, t_b + rng.choice([0.0, rng.uniform(0.0, 1.0)]), zb, v, xa + rng.uniform(-0.4, 0.4),
               rng.uniform(-0.3, 0.3), _window(t_b))
    b.t_lo = t_b
    sc.objs = [a, b]
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(label=label, xa=round(xa, 2), za=round(za, 2), t_leave=round(t_leave, 2), walk_off=walk_off,
                     gap=round(gap, 2), zb=round(zb, 2), v=round(v, 2))
    return _finish(sc, 60.0)


def fam_twins(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["person", "person", "chair", "car"])
    xa, za = rng.uniform(-1.0, 1.0), rng.uniform(2.0, 7.0) if label != "car" else rng.uniform(5.0, 12.0)
    t_b = rng.choice([0.0, rng.uniform(3.0, 12.0)])
    zb = max(1.5, za + rng.uniform(-1.0, 1.5))
    side, off = rng.choice([-1.0, 1.0]), rng.uniform(0.2, 0.7)
    xb = xa + side * off
    if label == "car":                           # two cars side by side: A parked off the corridor, B a car's width
        xa = math.copysign(2.2 + abs(xa), xa)    # nearer the middle, so B never drives through A
        xb = xa - math.copysign(1.8 + off, xa)
    a = Obj("A", label, lambda t: za, lambda t: xa)
    t_go = t_b + rng.uniform(0.0, 3.0) + (rng.uniform(4.0, 10.0) if t_b == 0.0 else 0.0)
    b = _mover(rng, "B", label, t_go, zb, _speed(rng, label), xb, rng.uniform(-0.3, 0.3), _window(t_b))
    b.t_lo = max(t_b, t_go) if t_b == 0.0 else t_b
    sc.objs = [a, b]
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(label=label, xa=round(xa, 2), za=round(za, 2), xb=round(xb, 2), zb=round(zb, 2), t_b=round(t_b, 2),
                     t_go=round(t_go, 2))
    return _finish(sc, 60.0)


def fam_lost_background(rng: random.Random, sc: Scene) -> Scene:
    label = rng.choice(["person", "person", "chair", "dog"])
    xa, za = rng.uniform(-1.2, 1.2), rng.uniform(2.0, 7.0)
    variant = rng.choice(["dropout", "replace", "late_warmup"])
    v = _speed(rng, label) if label in ("person", "dog") else rng.uniform(0.5, 1.4)
    if variant == "replace":
        t_d = rng.uniform(5.0, 15.0)
        a = Obj("A", label, lambda t: za, lambda t: xa, _window(0.0, t_d))
        t_b = t_d + rng.uniform(0.1, 1.5)
        zb = max(1.5, za + rng.uniform(-1.0, 3.0))
        b = _mover(rng, "B", label, t_b + rng.choice([0.0, rng.uniform(0.0, 1.0)]), zb, v, xa + rng.uniform(-0.15, 0.15),
                   rng.uniform(-0.3, 0.3), _window(t_b))
        b.t_lo = t_b
        sc.objs = [a, b]
        sc.params = dict(zb=round(zb, 2), t_b=round(t_b, 2))
    else:
        t_d = rng.uniform(5.0, 15.0) if variant == "dropout" else rng.uniform(1.8, 2.9)
        gap = rng.uniform(0.2, 1.6) if variant == "dropout" else rng.uniform(0.5, 3.0)
        t_go = t_d + rng.uniform(0.0, gap)       # starts to approach while out of view
        h = _mover(rng, "A", label, t_go, za, v, xa, rng.uniform(-0.3, 0.3))
        h.drops = frozenset(range(int(t_d * FPS), int((t_d + gap) * FPS)))
        h.t_lo = t_d + gap
        sc.objs = [h]
        sc.params = dict(t_d=round(t_d, 2), gap=round(gap, 2), t_go=round(t_go, 2))
    sc.variant = variant
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params.update(label=label, variant=variant, xa=round(xa, 2), za=round(za, 2), v=round(v, 2))
    return _finish(sc, 60.0)


SIGNS = ("EXIT", "ROOM 204", "ROOM 312", "RESTROOM", "GATE 12")


def fam_find(rng: random.Random, sc: Scene) -> Scene:
    sign = rng.random() < 0.4
    xg, zg = rng.uniform(-2.0, 2.0), rng.uniform(2.0, 9.0)
    t_app = 0.0 if rng.random() < 0.5 else rng.uniform(2.0, 12.0)
    t_goal = rng.uniform(4.0, 15.0)
    walk = rng.random() < 0.5
    v = rng.uniform(0.5, 1.2)
    t_walk = max(t_goal, t_app) + rng.uniform(0.0, 3.0)
    egof = (lambda t: v * max(0.0, t - t_walk)) if walk else (lambda t: 0.0)
    if walk:
        xg = rng.uniform(-0.3, 0.3)
    p = rng.choice([0.0, 0.0, 0.1, 0.4])
    if sign:
        text = rng.choice(SIGNS)
        g = Obj("H", "text", lambda t: max(0.25, zg - egof(t)), lambda t: xg, _window(t_app), kind="text", text=text,
                base=1.0, hazard=True)
        goal = [text.lower()]
        other = text[:-1] + str((int(text[-1]) + 2) % 10) if text[-1].isdigit() else "EXIT 2"
        xo, zo = rng.uniform(-2.0, 2.0), rng.uniform(2.0, 9.0)
        d = Obj("D", "text", lambda t: max(0.25, zo - egof(t)), lambda t: xo, kind="text", text=other, base=1.0)
    else:
        label = rng.choice(["chair", "bench", "person", "bicycle"])
        g = Obj("H", label, lambda t: max(0.25, zg - egof(t)), lambda t: xg, _window(t_app), hazard=True)
        goal = [label]
        xo, zo = rng.uniform(-2.5, 2.5), rng.uniform(2.0, 9.0)
        d = Obj("D", label, lambda t: max(0.25, zo - egof(t)), lambda t: xo)
    g.drops = _frame_drops(rng, 120.0, p=p)
    g.t_lo = max(t_goal, t_app)
    sc.objs = [g] + ([d] if rng.random() < 0.6 else [])
    sc.goal, sc.t_goal = goal, round(t_goal * FPS) / FPS
    _depth_mode(rng, sc)
    sc.T = min(60.0, g.t_lo + 25.0)
    sc.params = dict(goal=goal, xg=round(xg, 2), zg=round(zg, 2), t_app=round(t_app, 2), t_goal=round(t_goal, 2),
                     walk=walk, p_drop=p)
    return _finish(sc, sc.T)


def _desk_box(z0: float, zf: Callable[[float], float], fid: float, xs: Callable[[float], float] = lambda t: 0.0,
              top_f: Callable[[float], float] = lambda t: 0.0):
    """The desk user's box: centred, cut off at the bottom of the frame, larger as they come nearer."""
    def box(t):
        z = zf(t)
        half = min(0.5, 0.22 * 0.8 / max(0.2, z))
        top = max(0.0, 0.12 - 0.2 * (0.8 / max(0.2, z) - 1.0) - top_f(t))
        cx = 0.5 + fid * math.sin(0.7 * t) + xs(t)
        b = (cx - half, top, cx + half, 0.98)
        x1, x2 = max(0.0, b[0]), min(1.0, b[2])
        return (x1, b[1], x2, b[3]) if x2 - x1 > 0.02 else None
    return box


def fam_desk(rng: random.Random, sc: Scene) -> Scene:
    variant = rng.choice(["still", "fidget", "lean", "lean", "lean_multi", "drift", "arm", "stand", "stranger",
                          "stranger", "visitor", "visitor"])
    z0 = rng.uniform(0.7, 0.85) if variant in ("stranger", "visitor") else rng.uniform(0.5, 1.0)
    fid = 0.0 if variant == "still" else rng.uniform(0.005, 0.04)
    T = rng.uniform(60.0, 100.0)
    leans: List[Tuple[float, float]] = []                # (start, depth)
    n_leans = {"lean": 1, "lean_multi": rng.randint(2, 4), "stranger": rng.randint(0, 3),
               "visitor": rng.randint(0, 2)}.get(variant, 0)
    lean_m = rng.uniform(0.15, 0.6) if variant in ("lean", "lean_multi") else rng.uniform(0.25, 0.5)
    t = rng.uniform(8.0, 15.0)
    for _ in range(n_leans):
        leans.append((t, lean_m * (rng.uniform(0.9, 1.0) if variant == "lean_multi" else 1.0)))
        t += rng.uniform(12.0, 22.0)
    hold = rng.uniform(1.0, 3.0)

    def lean(t):
        for a, m in leans:
            if a <= t < a + 1.0:
                return m * (t - a)
            if a + 1.0 <= t < a + 1.0 + hold:
                return m
            if a + 1.0 + hold <= t < a + 2.0 + hold:
                return m * (a + 2.0 + hold - t)
        return 0.0
    drift = rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.2) if variant == "drift" else 0.0
    zu = lambda t: z0 - lean(t) + drift * min(1.0, t / 40.0)
    objs: List[Obj] = []
    user_on: Callable[[float], bool] = lambda t: True
    xs: Callable[[float], float] = lambda t: 0.0
    top_f: Callable[[float], float] = lambda t: 0.0
    if variant == "stand":
        t_up, away = rng.uniform(15.0, 40.0), rng.uniform(4.0, 15.0)
        side = rng.choice([-1.0, 1.0])
        top_f = lambda t: 0.3 * min(1.0, max(0.0, t - t_up)) * (0.0 if t > t_up + 3.0 + away else 1.0)
        xs = lambda t: side * (0.7 * min(1.0, max(0.0, t - t_up - 1.0) / 2.0) if t < t_up + 3.0 + away else
                               0.7 * max(0.0, 1.0 - (t - t_up - 3.0 - away) / 2.0))
        user_on = lambda t: not (t_up + 3.0 <= t < t_up + 3.0 + away)
        sc.params.update(t_up=round(t_up, 2), away=round(away, 2))
    if variant == "stranger":
        t_leave = (leans[-1][0] + hold + 4.0 if leans else rng.uniform(10.0, 20.0)) + rng.uniform(0.0, 10.0)
        side = rng.choice([-1.0, 1.0])
        xs = lambda t: side * 0.8 * min(1.0, max(0.0, t - t_leave) / 1.5)
        user_on = _window(0.0, t_leave + 1.5)
        t_s = t_leave + 1.5 + rng.uniform(0.0, 5.0)
        zs0, vs = rng.uniform(3.0, 5.0), rng.uniform(0.4, 1.0)
        z_s = _ramp(zs0, vs, t_s + rng.uniform(0.0, 2.0), 0.3)
        sx = rng.uniform(-0.15, 0.15)
        objs.append(Obj("S", "person", z_s, lambda t: sx, _window(t_s), hazard=True, t_lo=t_s))
        T = min(T + 40.0, 150.0)
        sc.params.update(t_leave=round(t_leave, 2), t_s=round(t_s, 2), zs0=round(zs0, 2), vs=round(vs, 2))
    if variant == "visitor":
        t_v = (leans[-1][0] + hold + 4.0 if leans else rng.uniform(10.0, 30.0)) + rng.uniform(0.0, 10.0)
        zv0, vv = rng.uniform(2.5, 4.0), rng.uniform(0.4, 1.0)
        zv = _ramp(zv0, vv, t_v + rng.uniform(0.0, 2.0), 0.3)
        vx = rng.uniform(-0.15, 0.15)
        objs.append(Obj("S", "person", zv, lambda t: vx, _window(t_v), hazard=True, t_lo=t_v))
        user_on = lambda t: zv(t) > zu(t) - 0.1 or t < t_v       # hidden once the visitor stands in front
        T = min(T + 40.0, 150.0)
        sc.params.update(t_v=round(t_v, 2), zv0=round(zv0, 2), vv=round(vv, 2))
    user = Obj("user", "person", zu, lambda t: 0.0, user_on, box=_desk_box(z0, zu, fid, xs, top_f), role="user")
    objs.insert(0, user)
    if variant == "arm":                         # a hand / arm detected as a second, smaller person box over the user
        p_arm = rng.uniform(0.1, 0.5)
        zarm = rng.uniform(0.35, 0.6)
        cx = 0.5 + rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.25)
        arm_drops = _frame_drops(rng, T + 1, p=1.0 - p_arm)
        objs.append(Obj("arm", "person", lambda t: zarm, lambda t: 0.0, drops=arm_drops, role="user",
                        box=lambda t: (cx - 0.08, 0.45, cx + 0.08, 0.98)))
    sc.objs = objs
    u = rng.random()
    if u < 0.7:
        sc.depth, sc.noise = "model", rng.uniform(0.02, 0.07)
    else:
        sc.depth = "bbox" if u < 0.85 else "none"
    sc.jitter = rng.choice([0.0, 0.002, 0.004])
    sc.variant = variant if variant not in ("lean", "lean_multi") else (
        ("deep_" if lean_m >= DEEP_LEAN_M else "") + variant)
    sc.configs = DESK_CONFIGS
    sc.T = T
    sc.params.update(variant=variant, z0=round(z0, 2), fidget=round(fid, 3), leans=[(round(a, 1), round(m, 2)) for a, m in leans],
                     drift=round(drift, 2))
    return _finish(sc, T, tail=1.0)


def fam_read_scene(rng: random.Random, sc: Scene) -> Scene:
    text = rng.choice(SIGNS)
    zs, xs = rng.uniform(0.8, 2.5), rng.uniform(-0.3, 0.3)
    sign = Obj("sign", "text", lambda t: zs, lambda t: xs, kind="text", text=text, base=1.0)
    label = rng.choice(["chair", "person", "person", "car"])
    z0 = rng.uniform(3.0, 7.0) if label != "car" else rng.uniform(6.0, 15.0)
    v = rng.uniform(0.2, 0.8) if label != "car" else rng.uniform(1.0, 4.0)
    t_go = rng.uniform(4.0, 15.0)
    h = _mover(rng, "H", label, t_go, z0, v, rng.uniform(-1.0, 1.0), rng.uniform(-0.3, 0.3),
               (lambda t: True) if rng.random() < 0.6 else _window(rng.uniform(0.5, t_go)))
    h.t_lo = t_go
    h.drops = _frame_drops(rng, 120.0, p=rng.choice([0.0, 0.05]))
    sc.objs = [sign, h]
    _depth_mode(rng, sc, p_model=0.7)
    sc.configs = READ_CONFIGS
    sc.T = 60.0
    sc.params = dict(text=text, zs=round(zs, 2), label=label, z0=round(z0, 2), v=round(v, 2), t_go=round(t_go, 2))
    return _finish(sc, 60.0)


# ---- the second verification's probe families (gate_verify_r2), kept so the sweep covers them ------------------------
def fam_deskwalk(rng: random.Random, sc: Scene) -> Scene:
    """A desk user leans in 2-4 times (0.3-0.55 m), then a stranger walks up after the user leaves (to the very place
    the user sat: stranger_back), or a visitor walks up in front of the user."""
    variant = rng.choice(["stranger", "visitor", "stranger_back"])
    z0 = rng.uniform(0.6, 0.9)
    fid = rng.uniform(0.0, 0.03)
    leans: List[Tuple[float, float]] = []
    t = rng.uniform(8.0, 12.0)
    for _ in range(rng.randint(2, 4)):
        leans.append((t, rng.uniform(0.3, 0.55)))
        t += rng.uniform(6.0, 14.0)
    hold = rng.uniform(0.5, 3.0)

    def lean(t):
        for a, m in leans:
            if a <= t < a + 1.0:
                return m * (t - a)
            if a + 1.0 <= t < a + 1.0 + hold:
                return m
            if a + 1.0 + hold <= t < a + 2.0 + hold:
                return m * (a + 2.0 + hold - t)
        return 0.0
    zu = lambda t: z0 - lean(t)
    objs: List[Obj] = []
    xs: Callable[[float], float] = lambda t: 0.0
    user_on: Callable[[float], bool] = lambda t: True
    t_end = leans[-1][0] + 2.0 + hold
    if variant.startswith("stranger"):
        t_leave = t_end + rng.uniform(1.0, 8.0)
        side = rng.choice([-1.0, 1.0])
        xs = lambda t: side * 0.8 * min(1.0, max(0.0, t - t_leave) / 1.5)
        user_on = _window(0.0, t_leave + 1.5)
        t_s = t_leave + 1.5 + rng.uniform(0.0, 4.0)
        zs0, vs = rng.uniform(2.0, 5.0), rng.uniform(0.3, 1.2)
        if variant == "stranger_back":
            zs0 = rng.uniform(1.5, 2.5)
        z_s = _ramp(zs0, vs, t_s + rng.uniform(0.0, 2.0), 0.3)
        sx = rng.uniform(-0.1, 0.1)
        objs.append(Obj("S", "person", z_s, lambda t: sx, _window(t_s), hazard=True, t_lo=t_s))
        sc.params.update(t_leave=round(t_leave, 2), t_s=round(t_s, 2), zs0=round(zs0, 2), vs=round(vs, 2))
    else:
        t_v = t_end + rng.uniform(1.0, 8.0)
        zv0, vv = rng.uniform(2.0, 4.0), rng.uniform(0.3, 1.2)
        zv = _ramp(zv0, vv, t_v + rng.uniform(0.0, 2.0), 0.3)
        vx = rng.uniform(-0.12, 0.12)
        objs.append(Obj("S", "person", zv, lambda t: vx, _window(t_v), hazard=True, t_lo=t_v))
        user_on = lambda t: zv(t) > zu(t) - 0.1 or t < t_v
        sc.params.update(t_v=round(t_v, 2), zv0=round(zv0, 2), vv=round(vv, 2))
    user = Obj("user", "person", zu, lambda t: 0.0, user_on, box=_desk_box(z0, zu, fid, xs), role="user")
    sc.objs = [user] + objs
    u = rng.random()
    sc.depth, sc.noise = ("model", rng.uniform(0.02, 0.07)) if u < 0.6 else (("bbox" if u < 0.8 else "none"), 0.03)
    sc.jitter = rng.choice([0.0, 0.002, 0.004])
    sc.variant = variant
    sc.configs = DESK_CONFIGS
    sc.T = 150.0
    sc.params.update(variant=variant, z0=round(z0, 2), leans=[(round(a, 1), round(m, 2)) for a, m in leans])
    return _finish(sc, 150.0)


def fam_swap(rng: random.Random, sc: Scene) -> Scene:
    """Two objects of one label stand apart (background), cross each other (the boxes overlap: the tracker may swap
    their ids), then the one that ends in the path walks at the user."""
    label = rng.choice(["person", "person", "dog", "chair"])
    za, zb = rng.uniform(2.5, 6.0), rng.uniform(2.5, 7.0)
    xa, xb = rng.uniform(-1.4, -0.4), rng.uniform(0.4, 1.4)
    t_x, dur = rng.uniform(4.0, 14.0), rng.uniform(1.0, 3.0)
    t_go = t_x + dur + rng.uniform(0.0, 3.0)
    v = rng.uniform(0.5, 1.6)
    end_b = rng.uniform(-0.3, 0.3)
    a = Obj("A", label, lambda t: za + (zb - za) * min(1.0, max(0.0, t - t_x) / dur), _slide(xa, xb, t_x, dur))
    zbf = lambda t: zb + (za - zb) * min(1.0, max(0.0, t - t_x) / dur)
    b = Obj("B", label, lambda t: max(0.25, zbf(t) - v * max(0.0, t - t_go)),
            lambda t: (xb + (xa - xb) * min(1.0, max(0.0, t - t_x) / dur)) if t < t_go
            else xa + (end_b - xa) * min(1.0, (t - t_go) / 1.5), hazard=True, t_lo=t_go)
    if rng.random() < 0.5:
        b.drops = _frame_drops(rng, 120.0, p=rng.choice([0.0, 0.05, 0.15]))
    sc.objs = [a, b]
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(label=label, za=round(za, 2), zb=round(zb, 2), xa=round(xa, 2), xb=round(xb, 2), t_x=round(t_x, 2),
                     dur=round(dur, 2), t_go=round(t_go, 2), v=round(v, 2))
    return _finish(sc, 60.0)


def fam_find_behind(rng: random.Random, sc: Scene) -> Scene:
    """A FIND goal behind a background object (overlapping boxes; occluded while the other covers its centre),
    sometimes of the same label; the user walks to it, or the object in front steps aside."""
    glabel = rng.choice(["chair", "bench", "person", "bicycle"])
    blabel = glabel if rng.random() < 0.4 else rng.choice(["person", "chair", "potted plant", "bench"])
    zb, xb = rng.uniform(1.8, 4.0), rng.uniform(-0.4, 0.4)
    zg = zb + rng.uniform(0.6, 3.0)
    xg = xb + rng.uniform(-0.3, 0.3)
    t_goal = rng.uniform(4.0, 14.0)
    walk = rng.random() < 0.6
    t_walk = t_goal + rng.uniform(0.0, 3.0)
    vw = rng.uniform(0.5, 1.2)
    egof = (lambda t: vw * max(0.0, t - t_walk)) if walk else (lambda t: 0.0)
    move_away = rng.random() < 0.4
    t_away = t_goal + rng.uniform(0.5, 5.0)
    bg = Obj("Bg", blabel, lambda t: max(0.25, zb - egof(t)),
             (lambda t: xb + (2.0 if xb >= 0 else -2.0) * min(1.0, max(0.0, t - t_away) / 1.5)) if move_away
             else (lambda t: xb))
    g = Obj("H", glabel, lambda t: max(0.25, zg - egof(t)), lambda t: xg, hazard=True, t_lo=t_goal)
    g.drops = frozenset(i for i in range(1200) if (abs(xg - bg.x(i / 10)) < 0.25 and rng.random() < 0.6))
    sc.objs = [bg, g]
    sc.goal, sc.t_goal = [glabel], round(t_goal * 10) / 10
    _depth_mode(rng, sc)
    sc.T = 50.0
    sc.params = dict(glabel=glabel, blabel=blabel, zb=round(zb, 2), zg=round(zg, 2), walk=walk, move_away=move_away)
    return _finish(sc, 50.0)


def fam_read_car(rng: random.Random, sc: Scene) -> Scene:
    """A sign near the user (read / scene modes) and a car, bicycle or person coming, sometimes from out of view;
    the depth source may turn to box-only or off mid-run (params['flip'])."""
    text = rng.choice(SIGNS)
    zs, xs = rng.uniform(0.8, 2.5), rng.uniform(-0.3, 0.3)
    sign = Obj("sign", "text", lambda t: zs, lambda t: xs, kind="text", text=text, base=1.0)
    label = rng.choice(["car", "car", "bicycle", "person"])
    z0 = {"car": rng.uniform(8.0, 30.0), "bicycle": rng.uniform(6.0, 20.0)}.get(label, rng.uniform(3.0, 8.0))
    v = {"car": rng.uniform(3.0, 12.0), "bicycle": rng.uniform(2.0, 6.0)}.get(label, rng.uniform(0.5, 1.5))
    t_go = rng.uniform(3.0, 15.0)
    h = _mover(rng, "H", label, t_go, z0, v, rng.uniform(-1.5, 1.5), rng.uniform(-0.3, 0.3),
               (lambda t: True) if rng.random() < 0.5 else _window(rng.uniform(0.5, t_go)))
    h.t_lo = t_go
    h.drops = _frame_drops(rng, 120.0, p=rng.choice([0.0, 0.05, 0.1]))
    sc.objs = [sign, h]
    _depth_mode(rng, sc, p_model=0.6)
    if rng.random() < 0.3:
        sc.params["flip"] = [(t_go + rng.uniform(-1.0, 2.0), rng.choice(["bbox", "none"]))]
        sc.depth = "model"
    sc.configs = READ_CONFIGS
    sc.T = 60.0
    sc.params.update(text=text, label=label, z0=round(z0, 2), v=round(v, 2), t_go=round(t_go, 2))
    return _finish(sc, 60.0)


def fam_rescue_switch(rng: random.Random, sc: Scene) -> Scene:
    """An object in view (said, then background or a repeat) starts to approach; dropouts (new track ids) fall right
    after the start, when the gate's held cues and releases are pending."""
    label = rng.choice(["person", "person", "car", "bicycle", "dog", "chair"])
    d0 = {"car": rng.uniform(6.0, 18.0), "bicycle": rng.uniform(5.0, 14.0)}.get(label, rng.uniform(3.0, 9.0))
    v = _speed(rng, label)
    t_go = rng.uniform(3.0, 14.0)
    h = _mover(rng, "H", label, t_go, d0, v, rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3))
    contact_eta = (d0 - 0.6) / max(0.1, v)
    drops: set = set()
    for _ in range(rng.randint(1, 3)):
        a = t_go + rng.uniform(0.0, max(0.3, 0.7 * contact_eta))
        drops |= set(range(int(a * FPS), int((a + rng.uniform(0.3, 1.4)) * FPS)))
    h.drops = frozenset(drops)
    sc.objs = [h] + _distractors(rng, rng.choice([0, 1]), lambda t: 0.0)
    _depth_mode(rng, sc)
    if rng.random() < 0.15:
        sc.rebaseline_at = round(max(0.5, t_go + rng.uniform(-1.0, 1.0)) * 2) / 2
    sc.T = 60.0
    sc.params = dict(label=label, d0=round(d0, 2), v=round(v, 2), t_go=round(t_go, 2), drops=len(drops))
    return _finish(sc, 60.0)


def fam_depth_flip(rng: random.Random, sc: Scene) -> Scene:
    """An approach whose depth source changes mid-approach (model -> bbox / none, or back), once or several times."""
    sc2 = fam_approach(rng, Scene(sc.family, sc.seed, 60.0, []))
    h = sc2.objs[0]
    end = h.contact if h.contact is not None else sc2.T
    flips = []
    t = h.t_lo + rng.uniform(-1.0, max(0.5, 0.6 * (end - h.t_lo)))
    for _ in range(rng.randint(1, 3)):
        flips.append((round(t, 1), rng.choice(["bbox", "none", "model"])))
        t += rng.uniform(0.5, 3.0)
    sc2.depth, sc2.noise = "model", rng.choice([0.02, 0.05, 0.1])
    sc2.params["flip"] = flips
    sc2.family = sc.family
    return sc2


def fam_bgref(rng: random.Random, sc: Scene) -> Scene:
    """A person / dog / chair at za (said, background) drops out for 0.1-0.7 s; another of its label appears at its
    place 0.6-3 m farther (the IoU tracker keeps the id) and approaches. Mostly without measured depth."""
    label = rng.choice(["person", "person", "dog", "chair"])
    za = rng.uniform(1.8, 4.5)
    zb = za + rng.uniform(0.6, 3.0)
    xa = rng.uniform(-0.5, 0.5)
    t_d = rng.uniform(4.0, 14.0)
    gap = rng.uniform(0.1, 0.7)
    a = Obj("A", label, lambda t: za, lambda t: xa, _window(0.0, t_d))
    v = rng.uniform(0.6, 1.8) if label != "chair" else rng.uniform(0.5, 1.2)
    b = _mover(rng, "B", label, t_d + gap + rng.uniform(0.0, 1.0), zb, v, xa + rng.uniform(-0.1, 0.1),
               rng.uniform(-0.2, 0.2), _window(t_d + gap))
    b.t_lo = t_d + gap
    sc.objs = [a, b]
    u = rng.random()
    sc.depth = "none" if u < 0.4 else ("bbox" if u < 0.75 else "model")
    sc.noise = 0.05
    sc.jitter = rng.choice([0.0, 0.002])
    sc.T = 60.0
    sc.params = dict(label=label, za=round(za, 2), zb=round(zb, 2), t_d=round(t_d, 2), gap=round(gap, 2), v=round(v, 2))
    return _finish(sc, 60.0)


def fam_handover(rng: random.Random, sc: Scene) -> Scene:
    """Occlusion handover: a person (said, background) leaves and another steps in where they stood within 0-0.4 s
    (often no gap at all: the IoU tracker keeps the id, and no dropout shows the range jump), 1-2.5 m farther, and
    walks at the user at 1-1.8 m/s."""
    za, dz = rng.uniform(2.5, 4.5), rng.uniform(1.0, 2.5)
    gap = rng.choice([0.0, 0.0, 0.1, rng.uniform(0.0, 0.4)])
    v, x, t_d = rng.uniform(1.0, 1.8), rng.uniform(-0.3, 0.3), rng.uniform(6.0, 14.0)
    a = Obj("A", "person", lambda t: za, lambda t: x, _window(0.0, t_d))
    b = Obj("B", "person", _ramp(za + dz, v, t_d + gap + rng.uniform(0.0, 0.5)), lambda t: x - 0.05,
            _window(t_d + gap), hazard=True, t_lo=t_d + gap)
    sc.objs = [a, b]
    sc.depth, sc.noise = rng.choice(["none", "bbox", "model"]), 0.05
    sc.jitter = rng.choice([0.0, 0.0, 0.002])
    sc.T = 60.0
    sc.params = dict(za=round(za, 2), dz=round(dz, 2), gap=round(gap, 2), v=round(v, 2), t_d=round(t_d, 2))
    return _finish(sc, 60.0)


def fam_reacq(rng: random.Random, sc: Scene) -> Scene:
    """A static person or chair in the path with a 0.9-1.5 s dropout every 2.5-5 s: no hazard; the cue counts (max per
    run) show how often the gate repeats it."""
    label = rng.choice(["person", "chair"])
    z, x = rng.uniform(1.5, 4.0), rng.uniform(-0.3, 0.3)
    per, dur = rng.uniform(2.5, 5.0), rng.uniform(0.9, 1.5)
    h = Obj("H", label, lambda t: z, lambda t: x)
    h.drops = frozenset(i for i in range(700) if (i / 10.0) % per > per - dur and i > 20)
    sc.objs = [h]
    _depth_mode(rng, sc)
    sc.T = 60.0
    sc.params = dict(label=label, z=round(z, 2), per=round(per, 2), dur=round(dur, 2))
    return sc


# ---- the third verification's probe families (gate_verify_r3: ext3.py, unchanged so its ids reproduce) -------------
def fam_find_disp(rng: random.Random, sc: Scene) -> Scene:
    """FIND a sign: the goal is set while the sign is in view (the gate forgets it and says it itself, deferred, a step
    after the scheduler), another sign appears (a routine cue that may wait behind the gate's message), and a person,
    dog or bicycle comes at the user right then."""
    goal = rng.choice(["GATE 12", "ROOM 204", "EXIT"])
    other = {"GATE 12": "GATE 14", "ROOM 204": "ROOM 206", "EXIT": "EXIT 2"}[goal]
    zg, xg = rng.uniform(3.0, 7.0), rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)
    t_goal = rng.uniform(4.0, 10.0)
    g = Obj("G", "text", lambda t: zg, lambda t: xg, kind="text", text=goal, base=1.0)
    zd, xd = rng.uniform(3.0, 7.0), -xg + rng.uniform(-0.4, 0.4)
    t_d = t_goal + rng.uniform(-0.5, 2.5)
    d = Obj("D", "text", lambda t: zd, lambda t: xd, _window(t_d), kind="text", text=other, base=1.0)
    lh = rng.choice(["person", "person", "dog", "bicycle"])
    t_h = t_goal + rng.uniform(0.5, 4.5)
    zh = {"bicycle": rng.uniform(5.0, 9.0)}.get(lh, rng.uniform(2.8, 5.0))
    vh = {"bicycle": rng.uniform(2.0, 3.5)}.get(lh, rng.uniform(1.0, 1.8))
    h = _mover(rng, "H", lh, t_h, zh, vh, rng.choice([-1, 1]) * rng.uniform(0.2, 1.0), rng.uniform(-0.2, 0.2),
               _window(t_h - rng.choice([0.0, 0.0, 0.5])))
    sc.objs = [h, g, d]
    sc.goal, sc.t_goal = [goal.lower()], round(t_goal * 10) / 10
    u = rng.random()
    sc.depth, sc.noise = ("model", 0.03) if u < 0.5 else (("bbox" if u < 0.8 else "none"), 0.03)
    sc.T = 60.0
    sc.params = dict(goal=goal, t_goal=round(t_goal, 2), t_d=round(t_d, 2), lh=lh, t_h=round(t_h, 2), zh=round(zh, 2),
                     vh=round(vh, 2))
    return _finish(sc, 60.0)


def fam_displaced(rng: random.Random, sc: Scene) -> Scene:
    """The gate's own message (a 'lost' reminder, a deferred or rescued cue about A), a routine cue about an off-path
    object B the scheduler offers while it plays (it waits for the channel), then hazard H coming at the user just as
    B would have ended alone."""
    la = rng.choice(["person", "chair", "bench", "person"])
    za, xa = rng.uniform(2.5, 5.5), rng.uniform(-0.3, 0.3)
    t_l = rng.uniform(5.0, 12.0)
    mode = rng.choice(["leave", "creep", "appear"])
    if mode == "leave":
        a = Obj("A", la, lambda t: za, lambda t: xa, _window(0.0, t_l))
    elif mode == "creep":
        va = rng.uniform(0.3, 0.6)
        a = Obj("A", la, lambda t: max(1.3, za - va * max(0.0, t - t_l)), lambda t: xa)
    else:
        xa2 = rng.choice([-1, 1]) * rng.uniform(1.2, 2.0)
        a = Obj("A", la, lambda t: za, lambda t: xa2, _window(t_l))
    lb = rng.choice(["potted plant", "bench", "tv", "person", "dog"])
    xb, zb = rng.choice([-1, 1]) * rng.uniform(2.0, 3.2), rng.uniform(3.0, 7.0)
    t_b = t_l + rng.uniform(0.3, 2.5)
    b = Obj("B", lb, lambda t: zb, lambda t: xb, _window(t_b))
    lh = rng.choice(["person", "person", "dog", "bicycle", "chair"])
    t_h = t_b + rng.uniform(0.0, 3.0)
    zh = {"bicycle": rng.uniform(6.0, 10.0)}.get(lh, rng.uniform(3.0, 6.0))
    vh = {"bicycle": rng.uniform(2.0, 3.5), "chair": rng.uniform(0.8, 1.4)}.get(lh, rng.uniform(1.0, 1.8))
    h = _mover(rng, "H", lh, t_h, zh, vh, rng.choice([-1, 1]) * rng.uniform(0.3, 1.2), rng.uniform(-0.2, 0.2),
               _window(t_h - rng.choice([0.0, 0.0, 1.0, 3.0])))
    sc.objs = [a, b, h]
    u = rng.random()
    sc.depth, sc.noise = ("model", rng.choice([0.03, 0.05])) if u < 0.5 else (("bbox" if u < 0.8 else "none"), 0.03)
    sc.jitter = rng.choice([0.0, 0.002])
    sc.T = 60.0
    sc.params = dict(mode=mode, la=la, za=round(za, 2), t_l=round(t_l, 2), lb=lb, t_b=round(t_b, 2), lh=lh,
                     t_h=round(t_h, 2), zh=round(zh, 2), vh=round(vh, 2))
    return _finish(sc, 60.0)


# obstacles the cane layer finds in the depth map and the detector cannot name: (x half-width, y0, y1, z depth) in m
CANE_OBSTACLES = {"pole": (0.08, 0.0, 2.2, 0.16), "panel": (0.5, 0.0, 1.4, 0.1), "head": (0.4, 1.35, 1.8, 0.3),
                  "low": (0.3, 0.0, 0.35, 0.4)}


def fam_cane(rng: random.Random, sc: Scene) -> Scene:
    """The cane on: the user walks a corridor with 2-4 obstacles in the walking path that the detector cannot name (a
    pole, a barrier panel, a head-height sign box, a low box: the cane layer says them from the depth map), routine
    objects to the side (the scheduler names them), and a hazard (a person, bicycle or dog coming head-on, a person or
    dog stepping in from the side, or a chair in the path) timed around the cane's first cue. Cane cues and scheduler
    cues compete for the channel: a scheduler cue the cane displaced is said late, which must not change what the
    scheduler is shown. Depth model always (the cane needs the metric depth map)."""
    v = rng.uniform(0.6, 1.2)
    t_walk = rng.choice([0.0, rng.uniform(1.0, 8.0)])
    ego = lambda t: v * max(0.0, t - t_walk)
    obs, z = [], rng.uniform(4.0, 7.0)
    for _ in range(rng.randint(2, 4)):
        kind = rng.choice(["pole", "pole", "panel", "head", "low"])
        obs.append((kind, rng.uniform(-0.25, 0.25), z))
        z += rng.uniform(3.0, 7.0)

    def obstacle(kind, x, z0):
        hw, y0, y1, dz = CANE_OBSTACLES[kind]
        return lambda t: ((x - hw, x + hw, y0, y1, z0 - ego(t), z0 - ego(t) + dz) if z0 - ego(t) > 0.35 else None)
    sc.obstacles = [obstacle(k, x, z0) for k, x, z0 in obs]
    # when the cane says one of them (body level at 2.5 m: the cane aid's detour range); the hazard comes around then
    t_c = t_walk + max(0.0, obs[rng.randrange(len(obs))][2] - 2.5) / v
    objs: List[Obj] = []
    for k in range(rng.choice([1, 2, 3])):              # routine: objects to the side (people stroll), passed by the user
        label = rng.choice(["person", "person", "bench", "potted plant", "chair"])
        xd, zd = rng.choice([-1, 1]) * rng.uniform(1.3, 2.6), rng.uniform(3.0, 12.0)
        vx = rng.uniform(-0.4, 0.4) if label == "person" else 0.0
        zf = (lambda z0: lambda t: max(0.25, z0 - ego(t)))(zd)
        t_in = max(0.0, t_c + rng.uniform(-3.0, 2.0)) if rng.random() < 0.6 else 0.0
        objs.append(Obj(f"d{k}", label, zf, (lambda x0, vx, t_in: lambda t: x0 + vx * max(0.0, t - t_in))(xd, vx, t_in),
                        (lambda zf, t_in: lambda t: t >= t_in and zf(t) > 0.5)(zf, t_in)))
    kind = rng.choice(["headon", "headon", "side", "chair"])
    t_h = max(0.5, t_c + rng.uniform(-2.0, 3.0))
    if kind == "chair":
        label, vh = "chair", 0.0
        zh = rng.uniform(2.5, 5.0) + ego(t_h)
        x0 = x1 = rng.uniform(-0.3, 0.3)
        t_go, on = t_h, _window(t_h)
    elif kind == "side":
        label = rng.choice(["person", "person", "dog"])
        zh, vh = rng.uniform(2.5, 6.0) + ego(t_h), rng.choice([0.0, rng.uniform(0.3, 1.2)])
        x0, x1 = rng.choice([-1, 1]) * rng.uniform(1.8, 3.0), rng.uniform(-0.3, 0.3)
        t_go, on = t_h + abs(x1 - x0) / rng.uniform(0.6, 1.8), _window(t_h)
    else:
        label = rng.choice(["person", "person", "bicycle", "dog"])
        zh = {"bicycle": rng.uniform(8.0, 14.0), "dog": rng.uniform(4.0, 8.0)}.get(label, rng.uniform(5.0, 10.0))
        vh = {"bicycle": rng.uniform(2.0, 4.0), "dog": rng.uniform(0.8, 2.0)}.get(label, rng.uniform(0.6, 1.5))
        zh += ego(t_h)
        x0, x1 = rng.choice([-1, 1]) * rng.uniform(0.3, 1.2), rng.uniform(-0.25, 0.25)
        t_go, on = t_h + rng.choice([0.0, rng.uniform(0.0, 2.0)]), _window(t_h - rng.choice([0.0, 0.0, 1.0]))
    zfun = lambda t: max(0.25, zh - ego(t) - vh * max(0.0, t - t_go))
    dur = max(0.5, (zfun(t_go) - CONTACT_M) / max(0.3, vh + v)) if kind != "side" else max(0.5, t_go - t_h)
    xs = _slide(x0, x1, t_h if kind == "side" else t_go, dur)
    h = Obj("H", label, zfun, xs, on, hazard=True, t_lo=t_h)
    h.drops = _frame_drops(rng, 120.0, p=rng.choice([0.0, 0.0, 0.05]))
    sc.objs = [h] + objs
    sc.walk = ego
    sc.depth, sc.noise = "model", rng.choice([0.02, 0.03, 0.05])
    sc.jitter = rng.choice([0.0, 0.0, 0.002])
    sc.configs = CANE_CONFIGS
    sc.variant = kind
    sc.T = 60.0
    sc.params = dict(v=round(v, 2), t_walk=round(t_walk, 2), obstacles=[(k, round(x, 2), round(z0, 1)) for k, x, z0 in obs],
                     t_cane=round(t_c, 2), hazard=kind, label=label, t_h=round(t_h, 2), zh=round(zh, 2), vh=round(vh, 2))
    return _finish(sc, 60.0)


# ---- the fourth verification's probe families (gate_verify_r4, ext4.py; v4_<name>:N builds its scene) ---------------
def fam_swap_fast(rng: random.Random, sc: Scene) -> Scene:
    """FIND swap against a goal object that starts to come at the user: FIND person (or dog / chair), H in the walking
    path, said, then it walks at the user at 0.8-1.8 m/s; D, another goal object off the path, comes into view unsaid
    just before. The scheduler may offer H again as it starts to move, before its range is the `closing` step nearer."""
    label = rng.choice(["person", "person", "person", "dog", "chair"])
    zh, xh = rng.uniform(3.0, 6.5), rng.uniform(-0.25, 0.25)
    t_goal = rng.uniform(1.0, 4.0)
    t_go = t_goal + rng.uniform(2.0, 9.0)
    v = rng.uniform(0.8, 1.8) if label != "chair" else rng.uniform(0.8, 1.3)
    h = Obj("H", label, _ramp(zh, v, t_go), lambda t: xh, hazard=True, t_lo=t_goal)
    xd, zd = rng.choice([-1, 1]) * rng.uniform(1.2, 3.0), rng.uniform(4.0, 9.0)
    t_d = t_go - rng.uniform(0.0, 2.5)
    d = Obj("D", label, lambda t: zd, lambda t: xd, _window(t_d))
    sc.objs = [h, d]
    sc.goal, sc.t_goal = [label], round(t_goal * 10) / 10
    u = rng.random()
    sc.depth, sc.noise = ("model", rng.choice([0.05, 0.08, 0.12])) if u < 0.75 else (rng.choice(["bbox", "none"]), 0.03)
    sc.jitter = rng.choice([0.0, 0.002, 0.004])
    sc.T = 60.0
    sc.params = dict(label=label, zh=round(zh, 2), t_go=round(t_go, 2), v=round(v, 2), xd=round(xd, 2), zd=round(zd, 2),
                     t_d=round(t_d, 2))
    return _finish(sc, 60.0)


def _sized_box(cam, x0: float, zf: Callable[[float], float], H: float, W: float):
    return lambda t: (lambda b: cam.clip(b) if b is not None else None)(cam.project_box(x0, zf(t), H, W))


def fam_flip_size(rng: random.Random, sc: Scene) -> Scene:
    """Depth switching with objects not of the typical size the box range assumes (a child or a tall person, a small or
    large chair, a dog: size factor 0.65-1.35). It is seen with measured depth while it stands in the path (the user
    stands: auto setup takes the camera for a desk), then depth turns off (box only, or none: the 'd' key) and it comes
    at the user; or the reverse."""
    from seeforme.geometry.camera import CameraModel
    cam = CameraModel()
    label = rng.choice(["person", "person", "chair", "dog"])
    k = rng.choice([rng.uniform(0.65, 0.85), rng.uniform(1.15, 1.35)])
    H, W = _size(label)
    z0, x0 = rng.uniform(2.5, 6.0), rng.uniform(-0.3, 0.3)
    t_f = rng.uniform(4.0, 12.0)
    t_go = t_f + rng.uniform(-1.0, 3.0)
    v = rng.uniform(0.5, 1.5)
    zf = _ramp(z0, v, t_go)
    h = Obj("H", label, zf, lambda t: x0, hazard=True, t_lo=0.0, box=_sized_box(cam, x0, zf, H * k, W * k))
    sc.objs = [h]
    first, then = ("model", rng.choice(["bbox", "none"])) if rng.random() < 0.7 else (rng.choice(["bbox", "none"]), "model")
    sc.depth, sc.noise = first, rng.choice([0.03, 0.05])
    sc.params["flip"] = [(round(t_f, 1), then)]
    sc.jitter = rng.choice([0.0, 0.002])
    sc.T = 60.0
    sc.params.update(label=label, k=round(k, 2), z0=round(z0, 2), t_f=round(t_f, 1), t_go=round(t_go, 2), v=round(v, 2),
                     first=first, then=then)
    return _finish(sc, 60.0)


def fam_small_static(rng: random.Random, sc: Scene) -> Scene:
    """No measured depth (mostly) and an object smaller than the box range assumes (a child, a small dog, a low stool:
    size factor 0.55-0.85), standing in the walking path 1.8-3.0 m away from the start (its box range reads beyond
    bg_path_m) while the user stands, then it comes at the user (or the user walks at it). Long static periods: the
    gate holds its repeats for hold_static_s."""
    from seeforme.geometry.camera import CameraModel
    cam = CameraModel()
    label = rng.choice(["person", "person", "dog", "chair"])
    k = rng.uniform(0.55, 0.85)
    H, W = _size(label)
    z0, x0 = rng.uniform(1.8, 3.0), rng.uniform(-0.3, 0.3)
    t_go = rng.uniform(4.0, 20.0)
    v = rng.uniform(0.4, 1.6)
    zf = _ramp(z0, v, t_go)
    t_app = rng.choice([0.0, 0.0, rng.uniform(0.5, 6.0)])
    h = Obj("H", label, zf, lambda t: x0, _window(t_app), hazard=True, t_lo=t_app,
            box=_sized_box(cam, x0, zf, H * k, W * k))
    sc.objs = [h] + _distractors(rng, rng.choice([0, 1]), lambda t: 0.0)
    sc.depth, sc.noise = rng.choice(["bbox", "bbox", "none"]), 0.03
    if rng.random() < 0.3:
        sc.depth, sc.noise = "model", 0.05
        sc.params["flip"] = [(round(rng.uniform(1.0, 8.0), 1), rng.choice(["bbox", "none"]))]
    sc.jitter = rng.choice([0.0, 0.002])
    sc.T = 60.0
    sc.params.update(label=label, k=round(k, 2), z0=round(z0, 2), t_go=round(t_go, 2), v=round(v, 2), t_app=round(t_app, 2))
    return _finish(sc, 60.0)


def fam_desk_still(rng: random.Random, sc: Scene) -> Scene:
    """Desk acceptance on more variants: a still or fidgeting user with the depth switching off and on (run_live's 'd'
    key: still_flip), or the user's box flickering (flicker), plus a stranger after the user leaves or a visitor; a
    rebaseline mid-run in some."""
    variant = rng.choice(["still", "still_flip", "flicker", "stranger", "visitor"])
    z0 = rng.uniform(0.55, 0.95)
    fid = 0.0 if variant == "still" else rng.uniform(0.0, 0.04)
    zu = lambda t: z0
    user_on: Callable[[float], bool] = lambda t: True
    xs: Callable[[float], float] = lambda t: 0.0
    objs: List[Obj] = []
    if variant == "stranger":
        t_leave = rng.uniform(15.0, 40.0)
        side = rng.choice([-1.0, 1.0])
        xs = lambda t: side * 0.8 * min(1.0, max(0.0, t - t_leave) / 1.5)
        user_on = _window(0.0, t_leave + 1.5)
        t_s = t_leave + 1.5 + rng.uniform(0.0, 4.0)
        zs0, vs = rng.uniform(1.5, 5.0), rng.uniform(0.3, 1.3)
        sx = rng.uniform(-0.1, 0.1)
        objs.append(Obj("S", "person", _ramp(zs0, vs, t_s + rng.uniform(0.0, 2.0), 0.3), lambda t: sx, _window(t_s),
                        hazard=True, t_lo=t_s))
    elif variant == "visitor":
        t_v = rng.uniform(12.0, 40.0)
        zv0, vv = rng.uniform(1.8, 4.0), rng.uniform(0.3, 1.2)
        zv = _ramp(zv0, vv, t_v + rng.uniform(0.0, 1.5), 0.3)
        vx = rng.uniform(-0.12, 0.12)
        objs.append(Obj("S", "person", zv, lambda t: vx, _window(t_v), hazard=True, t_lo=t_v))
        user_on = lambda t, zv=zv: zv(t) > z0 - 0.1 or t < t_v
    user = Obj("user", "person", zu, lambda t: 0.0, user_on, box=_desk_box(z0, zu, fid, xs), role="user")
    if variant == "flicker":
        user.drops = _frame_drops(rng, 200.0, p=rng.choice([0.05, 0.15]), bursts=rng.randint(1, 4), burst_s=(0.5, 2.5))
    sc.objs = [user] + objs
    u = rng.random()
    sc.depth, sc.noise = ("model", rng.uniform(0.02, 0.1)) if u < 0.6 else (("bbox" if u < 0.8 else "none"), 0.03)
    if variant == "still_flip" or rng.random() < 0.3:
        tf = rng.uniform(10.0, 40.0)
        sc.params["flip"] = [(round(tf, 1), rng.choice(["bbox", "none"])), (round(tf + rng.uniform(2.0, 15.0), 1), "model")]
        sc.depth, sc.noise = "model", rng.uniform(0.02, 0.1)
    if rng.random() < 0.3:
        sc.rebaseline_at = round(rng.uniform(15.0, 50.0) * 2) / 2
    sc.jitter = rng.choice([0.0, 0.002, 0.004])
    sc.variant = variant
    sc.configs = DESK_CONFIGS
    sc.T = 100.0
    sc.params.update(variant=variant, z0=round(z0, 2))
    return _finish(sc, 100.0)


def fam_reply_cut(rng: random.Random, sc: Scene) -> Scene:
    """Spoken replies to voice commands (+reply configurations, run_live speak_reply) that cut gated cues right after
    they start (0.1-1.5 s: before the object or the distance phrase was said): H is said once, off the path or far, and
    later comes at the user. A message a reply cut is heard only up to the cut."""
    label = rng.choice(["person", "person", "dog", "bicycle", "car", "chair"])
    z0 = {"bicycle": rng.uniform(8.0, 14.0), "car": rng.uniform(12.0, 25.0)}.get(label, rng.uniform(4.0, 9.0))
    x0 = rng.uniform(-2.0, 2.0)
    t_app = rng.uniform(1.0, 6.0)
    t_go = t_app + rng.uniform(2.0, 10.0)
    v = _speed(rng, label) if label != "chair" else rng.uniform(0.5, 1.2)
    h = _mover(rng, "H", label, t_go, z0, v, x0, rng.uniform(-0.3, 0.3), _window(t_app))
    h.t_lo = t_app
    sc.objs = [h] + _distractors(rng, rng.choice([0, 1, 2]), lambda t: 0.0)
    _depth_mode(rng, sc, p_model=0.6)
    sc.params = dict(label=label, z0=round(z0, 2), x0=round(x0, 2), t_app=round(t_app, 2), t_go=round(t_go, 2),
                     v=round(v, 2), oncue_p=rng.choice([0.5, 0.8, 1.0]), replies_only=True, replies=[])
    sc.T = 60.0
    sc.configs = REPLY_CONFIGS
    return _finish(sc, 60.0)


BUILDERS = {"approach": fam_approach, "static_path": fam_static_path, "side_entry": fam_side_entry,
            "two_hazards": fam_two_hazards, "flicker": fam_flicker, "id_switch": fam_id_switch,
            "cross_switch": fam_cross_switch, "twins": fam_twins, "lost_background": fam_lost_background,
            "find": fam_find, "desk": fam_desk, "read_scene": fam_read_scene, "deskwalk": fam_deskwalk, "swap": fam_swap,
            "find_behind": fam_find_behind, "read_car": fam_read_car, "rescue_switch": fam_rescue_switch,
            "depth_flip": fam_depth_flip, "bgref": fam_bgref, "handover": fam_handover, "reacq": fam_reacq,
            "find_disp": fam_find_disp, "displaced": fam_displaced, "cane": fam_cane, "swap_fast": fam_swap_fast,
            "flip_size": fam_flip_size, "small_static": fam_small_static, "desk_still": fam_desk_still,
            "reply_cut": fam_reply_cut}
DESK_FAMILIES = ("desk", "deskwalk", "desk_still")


def build_scene(family: str, seed: int) -> Scene:
    """The scene `family:seed` (the name seeds it: an alias in ALIASES builds the verifier's scene of that id)."""
    rng = random.Random(f"{family}:{seed}")
    sc = Scene(family, seed, 60.0, [])
    sc = BUILDERS[ALIASES.get(family, family)](rng, sc)
    sc.family = family
    return sc


# ---- frames --------------------------------------------------------------------------------------------------------
def make_frames(sc: Scene):
    """[(t, [(label, box, kind, text, distance, depth_conf)], [(name, box)])] at 10 fps: the same detections for every
    configuration of the scene."""
    from seeforme.geometry.camera import CameraModel
    from seeforme.perception.depth import BBoxDepthEstimator
    from seeforme.types import Detection
    cam = CameraModel()
    bbd = BBoxDepthEstimator(cam)
    rng = random.Random(f"frames:{sc.sid}")
    flips = sorted(sc.params.get("flip") or [])   # [(t, depth mode from then on)]
    frames = []
    for i in range(int(round(sc.T * FPS)) + 1):
        t = round(i / FPS, 3)
        depth = sc.depth
        for tf, m in flips:
            if t >= tf:
                depth = m
        dets, named = [], []
        for o in sc.objs:
            if not o.on(t):
                continue
            if o.box is not None:
                b = o.box(t)
            else:
                H, W = (0.3, 0.8) if o.kind == "text" else _size(o.label)
                b = cam.clip(cam.project_box(o.x(t), o.z(t), H, W, base_height=o.base))
            if b is None:
                continue
            named.append((o.name, b))
            if i in o.drops:
                continue
            if sc.jitter > 0:
                b = cam.clip(tuple(c + rng.gauss(0.0, sc.jitter) for c in b))
                if b is None:
                    continue
            z = o.z(t)
            if depth == "model":
                dist, conf = max(0.2, z + rng.gauss(0.0, sc.noise * max(1.0, z / 2.0))), 0.85
            else:
                dist, conf = None, 0.0
            if depth == "bbox":
                d = Detection(o.label, 0.9, b, kind=o.kind, text=o.text)
                bbd.annotate([d])
                dist, conf = d.distance_m, d.depth_conf
            dets.append((o.label, b, o.kind, o.text, dist, conf))
        frames.append((t, dets, named))
    return frames


DEPTH_EVERY = 2                  # a metric depth map every 2nd frame (5 per second, as the live depth model)
DEPTH_W, DEPTH_H = 160, 120


def render_depth(boxes, cam, noise: float = 0.02, seed: int = 0):
    """Metric depth (m, camera z) of a floor and boxes (x0, x1, y0, y1, z0, z1: x right, y up from the floor, z forward)
    seen by `cam` (geometry.CameraModel: its height, pitch and field of view, so the detector's boxes and the depth map
    agree), DEPTH_W x DEPTH_H pixels, with relative Gaussian noise."""
    import numpy as np
    f = 0.5 * DEPTH_W / math.tan(math.radians(cam.hfov_deg) / 2)
    xc, yc = np.meshgrid((np.arange(DEPTH_W) + 0.5 - DEPTH_W / 2) / f, (np.arange(DEPTH_H) + 0.5 - DEPTH_H / 2) / f)
    th, height = math.radians(cam.pitch_deg), cam.height_m
    dx, dy, dz = xc, -(yc * math.cos(th) + math.sin(th)), math.cos(th) - yc * math.sin(th)   # world ray per unit depth
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(dy < -1e-6, height / -dy, np.inf)
        for (x0, x1, y0, y1, z0, z1) in boxes:
            lo, hi = np.full(xc.shape, -np.inf), np.full(xc.shape, np.inf)
            for o, d, a, b in ((0.0, dx, x0, x1), (height, dy, y0, y1), (0.0, dz, z0, z1)):
                ta, tb = (a - o) / d, (b - o) / d
                par = np.abs(d) < 1e-9
                inside = a <= o <= b
                lo = np.maximum(lo, np.where(par, -np.inf if inside else np.inf, np.minimum(ta, tb)))
                hi = np.minimum(hi, np.where(par, np.inf if inside else -np.inf, np.maximum(ta, tb)))
            T = np.where((hi >= lo) & (lo > 0), np.minimum(T, lo), T)
    T = np.where(np.isfinite(T), T, 30.0)
    return (T * (1 + noise * np.random.default_rng(seed).standard_normal(T.shape))).astype(np.float32)


def make_motion(sc: Scene) -> Dict[int, object]:
    """{frame index: grayscale camera frame} for a scene with a walk: a fixed smooth texture that scrolls down the image
    by 0.2 frame widths per metre walked, with sensor noise (a still camera: the texture and the noise only), for the
    activity detector's frame path (walking: flow, shift and bob; standing: still). Empty without a walk."""
    if sc.walk is None:
        return {}
    import cv2
    import numpy as np
    rng = np.random.default_rng(zlib.crc32(sc.sid.encode()))
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (DEPTH_H, DEPTH_W)).astype(np.float32), (0, 0), 2.0)
    tex = (tex - tex.mean()) * (60.0 / max(1e-6, tex.std())) + 128.0
    out = {}
    for i in range(int(round(sc.T * FPS)) + 1):
        dy = int(round(0.2 * DEPTH_W * sc.walk(i / FPS)))
        out[i] = np.clip(np.roll(tex, dy, axis=0) + rng.normal(0, 2.0, tex.shape), 0, 255).astype(np.uint8)
    return out


def make_depth(sc: Scene) -> Dict[int, object]:
    """{frame index: depth map} every DEPTH_EVERY frames for a scene with cane obstacles: the floor, the obstacles, and
    the detected objects (as boxes of their typical size: a person in the path is in the depth map too, and the cane
    leaves it to the scheduler when a track explains it). Empty for other scenes (no cane configuration runs there)."""
    if not sc.obstacles:
        return {}
    from seeforme.geometry.camera import CameraModel
    cam = CameraModel()
    out = {}
    for i in range(0, int(round(sc.T * FPS)) + 1, DEPTH_EVERY):
        t = round(i / FPS, 3)
        boxes = [b for b in (f(t) for f in sc.obstacles) if b is not None]
        for o in sc.objs:
            if o.kind != "object" or not o.on(t) or o.box is not None:
                continue
            z, x = o.z(t), o.x(t)
            if z <= 0.35:
                continue
            H, W = _size(o.label)
            boxes.append((x - W / 2, x + W / 2, 0.0, H, z, z + min(0.4, W)))
        out[i] = render_depth(boxes, cam, noise=0.02, seed=i)
    return out


# ---- runs ----------------------------------------------------------------------------------------------------------
_CACHE: Dict[str, object] = {}


def _base_cfg():
    if "cfg" not in _CACHE:
        from _common import load_cfg
        _CACHE["cfg"] = load_cfg()
    return _CACHE["cfg"]


def fresh_policy(name: str):
    """A new scheduler with its own config: deep copy of a cached one, sharing only the loaded models."""
    full = POLICY_NAMES.get(name, name)
    tmpl = _CACHE.get(full)
    if tmpl is None:
        from seeforme.pipeline import make_live_policy
        tmpl = _CACHE[full] = make_live_policy(full, copy.deepcopy(_base_cfg()))
    memo = {}
    for k in ("text_model", "model"):
        m = getattr(tmpl, k, None)
        if m is not None:
            memo[id(m)] = m
    pol = copy.deepcopy(tmpl, memo)
    pol.reset()
    return pol


@dataclass
class Said:
    t: float
    owner: str
    action: str
    kind: str
    severity: float
    ttc: Optional[float]
    message: str
    why: str = ""
    end: float = math.inf                # when it stopped playing: its spoken length, or cut by an interrupt (heard_until)


# spoken replies (+reply configurations): what run_live's command router answers, short and long (the help text holds
# the channel for 9 s); the same list and seeds as the fourth verification's ext4.py, so its scenes reproduce
LONG_HELP = ("You can say: what is around me, read this, find a door, is the path clear, where is the exit, "
             "repeat that, quieter, louder, pause, resume, or help to hear this again.")
REPLIES = ("Mode navigate.", "Goal set: exit.", "Nothing new ahead.", "Paused.", "Resumed.",
           "Around you: a person ahead, about 3 meters; a chair on the left, about 2 meters.",
           "The text says: ROOM 204.", LONG_HELP)


# fields of a Track the schedulers and the gate write (the tracker never reads them); everything else is the tracker's
_POLICY_FIELDS = ("last_announced", "last_announced_bbox", "announce_count", "last_haptic")


class _Shadows:
    """One configuration's view of the shared tracker's tracks: a persistent copy per track (object identity matters to
    the gate) that takes the tracker's fields every frame and keeps its own scheduler fields. So every configuration
    sees exactly the tracks it would see with its own tracker (the tracker never reads what the schedulers write),
    and the scene is tracked once instead of once per configuration."""

    def __init__(self) -> None:
        self.by_id: Dict[int, Tuple[object, object]] = {}
        self.absent: Dict[int, int] = {}         # frames a track has not been in the tracker's list

    def view(self, tracks):
        """The configuration's copies of the tracker's tracks this frame. A track the tracker leaves out of its list for a
        few frames (a missed detection: active_tracks lists only matched ones) keeps its copy, as the live pipeline's
        Track object persists while the tracker holds it: the scheduler's marks and the gate's object identity survive a
        dropout (IoUTracker never reuses an id; a new object under an id gets a new copy)."""
        out, seen = [], set()
        for m in tracks:
            seen.add(m.track_id)
            pair = self.by_id.get(m.track_id)
            if pair is None or pair[0] is not m:
                sh = copy.copy(m)
                self.by_id[m.track_id] = (m, sh)
            else:
                sh = pair[1]
                d = sh.__dict__
                keep = {f: d[f] for f in _POLICY_FIELDS}
                d.update(m.__dict__)
                d.update(keep)
            out.append(sh)
        for tid in list(self.by_id):
            self.absent[tid] = 0 if tid in seen else self.absent.get(tid, 0) + 1
            if self.absent[tid] > 300:           # (long gone: the tracker dropped it after max_age)
                del self.by_id[tid], self.absent[tid]
        return out


class _Run:
    """One configuration being simulated: its scheduler, gate / controller, speech channel and track view."""

    def __init__(self, policy: str, config: str):
        from seeforme.config import PipelineConfig
        from seeforme.eval.runner import SpeechChannel
        from seeforme.geometry import CameraModel
        from seeforme.live.controller import LiveController
        from seeforme.live.gate import AnnouncementGate, GateConfig
        pcfg = PipelineConfig()
        self.config = config
        self.pol = fresh_policy(policy)
        cam = CameraModel()
        self.channel = SpeechChannel(pcfg.speech.min_gap, pcfg.speech.seconds_per_word)
        self.gate = self.ctl = None
        if config == "bare":
            self.dec = self.pol
        elif config.startswith("gate"):
            gcfg = {"gate": GateConfig(), "gate_nobg": GateConfig(background=False),
                    "gate_op": GateConfig(operator_filter=True)}[config]
            self.dec = self.gate = AnnouncementGate(self.pol, gcfg, camera=cam)
        else:
            parts = config.split(":")
            opts = parts[1].split("+")
            what, cane = opts[0], "cane" in opts[1:]
            setup, mode = (what, "auto") if what in ("auto", "handheld", "desk") else ("auto", what)
            self.ctl = LiveController(self.pol, mode=mode, setup=setup, camera=cam, gate=parts[-1] != "nogate", cane=cane)
            self.dec, self.gate = self.ctl.policy, self.ctl.gate
            self._rp_on = "reply" in opts[1:]
        self.dec.reset()
        self.shadows = _Shadows()
        self.next_t = 0.0
        self.out: List[Said] = []
        self.dropped = 0                         # spoken ("speak") on a busy channel: the live speaker drops it
        self.depth = None                        # the latest depth map (the cane: a map passed again is ignored)
        self._rp_on = getattr(self, "_rp_on", False)   # +reply: spoken replies on the channel
        self._rp_plan: Optional[List[Tuple[float, str]]] = None   # replies to come (t, text)
        self._rp_pending: List[str] = []                   # replies waiting for the channel
        self._rp_last: Tuple[object, float, object] = (None, -1e9, None)   # the last pipeline cue, when, its track
        self.replies: List[Tuple[float, str, float]] = []    # (start, text, end) of every reply spoken
        self._rp_oncue: Optional[random.Random] = None

    # ---- spoken replies (run_live speak_reply) ------------------------------------------------------------------------
    def _reply_plan(self, sc: Scene) -> List[Tuple[float, str]]:
        """The scene's replies (sc.params["replies"]) plus, unless replies_only, one every 4-16 s from 1-8 s on."""
        rng = random.Random(f"replies:{sc.sid}")
        self._rp_oncue = random.Random(f"oncue:{sc.sid}")
        out = list(sc.params.get("replies") or [])
        if not sc.params.get("replies_only"):
            t = rng.uniform(1.0, 8.0)
            while t < sc.T:
                out.append((round(t * FPS) / FPS, rng.choice(REPLIES)))
                t += rng.uniform(4.0, 16.0)
        return sorted(out)

    def _occupy(self, t: float, text: str) -> None:
        """A reply on the channel (controller.reply_cue, interrupt=True): it cuts the message playing."""
        from seeforme.live.controller import reply_cue
        if not self.channel.free(t):
            last = next((s for s in reversed(self.out) if s.action != "haptic"), None)
            if last is not None and last.end > t:
                last.end = t
        rc = reply_cue(text, t)
        self.channel.speak(rc, interrupt=True)
        self.replies.append((t, text, round(t + rc.duration_s, 2)))

    def _reply(self, t: float, text: str) -> None:
        """run_live speak_reply: a hazard warning still playing (the last cue, under 6 s ago, is_safety_cue with its track
        and the goal) is not cut: the reply waits for the channel; anything else is cut."""
        from seeforme.live.commands import is_safety_cue
        cue, t_cue, trk = self._rp_last
        if (not self.channel.free(t) and cue is not None and t - t_cue < 6.0
                and is_safety_cue(cue, trk, self.ctl.goal_tags if self.ctl is not None else [])):
            self._rp_pending.append(text)
            return
        self._occupy(t, text)

    def _replies_before(self, sc: Scene, t: float) -> None:
        if self._rp_plan is None:
            self._rp_plan = self._reply_plan(sc)
        while self._rp_plan and self._rp_plan[0][0] <= t + 1e-9:
            self._reply(t, self._rp_plan.pop(0)[1])
        if self._rp_pending and self.channel.free(t):
            self._occupy(t, self._rp_pending.pop(0))

    def _replies_after(self, sc: Scene, t: float, tracks, n0: int) -> None:
        """After a cue: it is the last pipeline cue (speak_reply's test), and a command may follow it (oncue_p)."""
        if len(self.out) <= n0:
            return
        s, cur = self.out[-1], self.channel.current
        if s.action == "haptic":
            from types import SimpleNamespace
            cur = SimpleNamespace(action="haptic", policy="", track_id=0, severity=1.0, ttc_s=None, label="", text=None, t=t)
        cue = cur if cur is not None and abs(cur.t - t) < 1e-6 else None
        trk = next((x for x in tracks if x.track_id == cue.track_id), None) if cue is not None else self._rp_last[2]
        self._rp_last = (cue, t, trk)
        r = self._rp_oncue
        if s.action != "haptic" and r.random() < sc.params.get("oncue_p", 0.15):
            d = r.choice([r.uniform(0.1, 0.4), r.uniform(0.3, 1.5)])
            self._rp_plan.append((round((t + d) * FPS) / FPS, r.choice(REPLIES[:5] + (LONG_HELP,))))
            self._rp_plan.sort()

    def step(self, sc: Scene, t: float, tracks, owner: Dict[int, str], depth=None, frame=None) -> None:
        if self._rp_on:
            self._replies_before(sc, t)
        n0 = len(self.out)
        self._step(sc, t, tracks, owner, depth, frame)
        if self._rp_on:
            self._replies_after(sc, t, tracks, n0)

    def _step(self, sc: Scene, t: float, tracks, owner: Dict[int, str], depth=None, frame=None) -> None:
        from seeforme.cues.pcacs import ChannelState
        if sc.goal and abs(t - sc.t_goal) < 1e-6:
            (self.ctl or self.gate or self.pol).set_goal(list(sc.goal))
        if sc.rebaseline_at is not None and abs(t - sc.rebaseline_at) < 1e-6 and self.gate is not None and self.ctl is None:
            self.gate.rebaseline()
        view = self.shadows.view(tracks)
        if depth is not None:
            self.depth = depth
        if self.ctl is not None:
            self.ctl.on_frame(frame, t, view, depth=self.depth if self.ctl.cane is not None else None)
        if t + 1e-9 < self.next_t:
            return
        self.next_t = t + self.dec.decision_interval
        ch_ = self.channel
        busy = not ch_.free(t)
        cur = ch_.current if busy else None
        ch = ChannelState(busy=busy, remaining_s=ch_.remaining(t), current_utility=(cur.utility if cur else 0.0) or 0.0,
                          current_severity=(cur.severity if cur else 0.0) or 0.0,
                          current_track=(cur.track_id if cur else None))
        cues = self.dec.decide(view, t, ch)
        if not cues:
            return
        c = cues[0]
        if c.action == "speak" and busy:
            # the live speakers (feedback/speech.py, elevenlabs_tts.py) drop a message they get while they speak: it is
            # never heard. The schedulers speak only on a free channel; the gate and the controller wait for it
            self.dropped += 1
            return
        end = t
        if c.action != "haptic":
            if busy and self.out:                # an interrupt cuts the message playing: heard only up to here
                last = next((s for s in reversed(self.out) if s.action != "haptic"), None)
                if last is not None and last.end > t:
                    last.end = t
            ch_.speak(c, interrupt=(c.action == "interrupt"))
            end = t + float(c.duration_s or 0.0)
        ttc = c.ttc_s if c.ttc_s is not None and math.isfinite(c.ttc_s) else None
        g = self.gate
        why = g.last_reason if g is not None and g.last_verdict is not None and abs(g.last_verdict.get("t", -1) - t) < 1e-6 else ""
        owner_ = "cane" if c.policy == "cane" else owner.get(c.track_id, "?")
        self.out.append(Said(t, owner_, c.action, c.kind, float(c.severity or 0.0), ttc, c.message, why, end))

    def trace(self) -> None:
        g = self.gate
        print(f"  [{self.config}] {len(self.out)} cues" + (f", {self.dropped} dropped on a busy channel" if self.dropped else "")
              + (f" | {g.describe()}" if g is not None else ""))
        for s in self.out:
            print(f"    {s.t:6.1f} {s.owner:>5} {s.action:9} {s.message!r} {s.why}"
                  + (f" [cut at {s.end:.1f}]" if s.action != "haptic"
                     and s.end < s.t + max(1.2, SPW * len(s.message.split())) - 1e-6 else ""))
        if self.replies:
            print("    replies:", self.replies[:12])
        if g is not None and g.releases:
            print("    releases:", g.releases[:12])


def run_configs(sc: Scene, frames, policy: str, configs: Sequence[str], trace: bool = False,
                dropped: Optional[Dict[str, int]] = None, views: Optional[Dict[str, list]] = None) -> Dict[str, List[Said]]:
    """The configurations in lockstep over the scene's frames, one shared tracker; the cues each let through, with the
    object each is about (the named box the cue's track overlapped last). `dropped` gets, per configuration, the cues
    spoken onto a busy channel (never heard); `views`, per configuration, what its scheduler was shown and offered at
    every decision (t, busy, remaining, current track, offers): scheduler_view."""
    from seeforme.config import PipelineConfig
    from seeforme.geometry import CameraModel, TrackDynamics
    from seeforme.live.gate import bbox_iou
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.types import Detection
    tracker = IoUTracker(PipelineConfig().tracker, dynamics=TrackDynamics(CameraModel()))
    runs = [_Run(policy, c) for c in configs]
    if views is not None:
        for r in runs:
            scheduler_view(r, views.setdefault(r.config, []))
    depth = make_depth(sc) if any("+cane" in c for c in configs) else {}
    motion = make_motion(sc) if any(c.startswith("ctl:") for c in configs) else {}
    owner: Dict[int, str] = {}
    for i, (t, dets, named) in enumerate(frames):
        tracks = tracker.update([Detection(lb, 0.9, b, kind=k, text=tx, distance_m=d, depth_conf=c)
                                 for lb, b, k, tx, d, c in dets], t)
        for trk in tracks:
            if trk.misses:
                continue
            best = max(named, key=lambda nb: bbox_iou(nb[1], trk.bbox), default=None)
            if best is not None and bbox_iou(best[1], trk.bbox) > 0.3:
                owner[trk.track_id] = best[0]
        for r in runs:
            r.step(sc, t, tracks, owner, depth.get(i), motion.get(i))
    if trace:
        for r in runs:
            r.trace()
    if dropped is not None:
        dropped.update({r.config: r.dropped for r in runs})
    return {r.config: r.out for r in runs}


def scheduler_view(run: "_Run", log: list) -> None:
    """Log what the run's scheduler (inside its gate and controller) is shown and offers at every decision: (t, channel
    busy, remaining s, current track, [(track, action, message)]). The gate and the controller must show it exactly what
    it is shown alone (module docstring of seeforme/live/gate.py: The scheduler's view)."""
    inner = run.pol.decide

    def watch(tracks, t, ch=None, *a, **kw):
        out = inner(tracks, t, ch, *a, **kw)
        busy = bool(getattr(ch, "busy", False))
        log.append((round(t, 3), busy, round(getattr(ch, "remaining_s", 0.0) or 0.0, 3) if busy else 0.0,
                    getattr(ch, "current_track", None) if busy else None,
                    tuple((c.track_id, c.action, c.message) for c in out)))
        return out
    run.pol.decide = watch


def diverge(sc: Scene, frames, policy: str, configs: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """The configurations whose scheduler was shown or offered anything other than alone (bare), with the first
    decision that differs. By default every configuration of the scene (SAME_VIEW_CONFIGS: all of them)."""
    cfgs = [c for c in (configs or SAME_VIEW_CONFIGS) if c in sc.configs or configs is not None]
    views: Dict[str, list] = {}
    run_configs(sc, frames, policy, ["bare"] + cfgs, views=views)
    b, out = views["bare"], []
    for c in cfgs:
        g = views[c]
        k = next((i for i, (x, y) in enumerate(zip(b, g)) if x != y), None)
        if k is None and len(b) != len(g):
            k = min(len(b), len(g))
        if k is not None:
            out.append(dict(sid=sc.sid, policy=policy, config=c, decisions=len(b), k=k,
                            bare=b[k] if k < len(b) else None, gated=g[k] if k < len(g) else None))
    return out


def run_config(sc: Scene, frames, policy: str, config: str, trace: bool = False) -> List[Said]:
    """One configuration over the scene's frames (run_configs)."""
    return run_configs(sc, frames, policy, [config], trace)[config]


def _range_at(o: Obj, t: float) -> float:
    return o.z(t)


def read_kept(s: Said) -> bool:
    """Would the TaskAdapter's read-mode filter keep this cue (activity.TaskAdapter.filter_cues)?"""
    return (s.kind == "text" or s.action in ("haptic", "interrupt") or s.severity >= 0.9
            or (s.ttc is not None and s.ttc < 6.0))


SPW = 0.35                       # seconds per spoken word (PipelineConfig().speech.seconds_per_word)
_RANGE_WORDS = re.compile(r"\b(meters?|feet|close|front of you)\b")


def _words_heard(s: Said) -> float:
    """How many words of the message the user heard: all, unless an interrupt cut it (Said.end)."""
    return math.inf if s.action == "haptic" or not math.isfinite(s.end) else (s.end - s.t) / SPW + 1e-6


def heard_label(s: Said) -> bool:
    """Did the user hear what the cue is about (a buzz; or the words before its first comma, the object or the sign's
    text, before an interrupt cut it)?"""
    return _words_heard(s) >= len(s.message.split(",")[0].split())


def said_range(s: Said) -> bool:
    """Did the message say how far it is (it has a distance phrase) and did the user hear it?"""
    n = 0
    for part in s.message.split(","):
        n += len(part.split())
        if _RANGE_WORDS.search(part.lower()):
            return _words_heard(s) >= n
    return False


def heard_range(s: Said) -> bool:
    """... and how far it is (the words through its distance phrase; a cue without one: its object)?"""
    return said_range(s) if _RANGE_WORDS.search(s.message.lower()) else heard_label(s)


def approach_mps(sc: Scene, o: Obj, t: float, dt: float = 0.1) -> float:
    """How fast o comes at the user at t (m/s, ground truth over the last frame), the user's own walk (sc.walk) taken
    out: the user knows they walk; an object that starts to come at them is news."""
    t0 = max(0.0, t - dt)
    if t - t0 < 1e-6:
        return 0.0
    v = (o.z(t0) - o.z(t)) / (t - t0)
    if sc.walk is not None:
        v -= (sc.walk(t) - sc.walk(t0)) / (t - t0)
    return v


def _stayed(o: Obj, t0: float, t1: float, tol: float, step: float = 0.25) -> bool:
    """Was o within tol of where it was at t0 (range and lateral) all the way to t1?"""
    z0, x0 = o.z(t0), o.x(t0)
    k = 1
    while t0 + k * step < t1 + 1e-9:
        tt = t0 + k * step
        if abs(o.z(tt) - z0) >= tol or abs(o.x(tt) - x0) >= tol:
            return False
        k += 1
    return abs(o.z(t1) - z0) < tol and abs(o.x(t1) - x0) < tol


def heard(sc: Scene, h: Obj, t: float, gated: List[Said], bare: Optional[Said] = None) -> bool:
    """Had the gated user heard about H at about its range now, before t? A gated cue about H in the JUST_S before; or
    in the NEWS_S before about H while it was less than max(0.35 m, 15 %) farther than it is now; or about H at its
    range at any time before, H having stayed within that tolerance of the place it was then (a static object the user
    heard of is no news when the scheduler names it again, however long ago: the gate holds such repeats for
    hold_static_s); or, in the NEWS_S before, about another object of its label (a sign: its text) at H's place now,
    the message saying how far (or, when the bare cue does not say how far either, not: the user heard as much as it
    says; bgref:906, "Dog." about a dog that left and "Dog." about one stepping in behind its place): within PLACE_M laterally, H not nearer than it was by that tolerance and not farther by
    the gate's `closer` step (max(1 m, 25 %): a chair where the said one stood, or a dog 0.6 m behind the place of the
    dog said, is the same thing to the user; a person 3 m behind the one said is not), never the desk user (naming the
    user says nothing about a visitor in front of them). None of these if H has started to come at the user since
    (approach_mps up by ONSET_MPS: a person who starts to walk at the user is news before they are any nearer). A
    message an interrupt or a reply cut counts only for what was said before the cut: its object, and its range only
    if the distance phrase was said."""
    objs = {o.name: o for o in sc.objs}
    r, x = _range_at(h, t), h.x(t)
    v_now = approach_mps(sc, h, t)
    last_h = None                                # the latest cue about H itself, heard with its range
    for g in gated:
        if not g.t < t or not heard_label(g):
            continue
        if v_now - approach_mps(sc, h, g.t) >= ONSET_MPS - 1e-9:
            continue                             # (H started to come at the user since)
        if g.owner == h.name:
            if t - g.t <= JUST_S:
                return True
            if heard_range(g):
                last_h = g
        if t - g.t > NEWS_S:
            continue
        o = objs.get(g.owner)
        if o is None or o.label != h.label or o.kind != h.kind or (o.kind == "text" and o.text != h.text):
            continue
        rg = _range_at(o, g.t)
        tol = max(0.35, 0.15 * rg)
        if o is h:
            if heard_range(g) and rg - r < tol:
                return True
        elif ((said_range(g) or (bare is not None and not _RANGE_WORDS.search(bare.message.lower())))
              and o.role != "user" and abs(o.x(g.t) - x) < PLACE_M and rg - r < tol
              and r - rg < max(CLOSER_M, CLOSER_FRAC * rg)):
            return True
    if last_h is not None:
        rg = _range_at(h, last_h.t)
        return _stayed(h, last_h.t, t, max(0.35, 0.15 * rg))
    return False


def judge(sc: Scene, h: Obj, ref: List[Said], gated: List[Said], read_mode: bool = False) -> Dict[str, object]:
    """Severity of one hazard in one gated run against a reference run (module docstring): the first bare cue that
    carries news against the first gated cue in the window, and every bare cue that carries news at least SEVERE_S
    before contact against the first gated cue about H from then on (a system that names H once early and then stays
    silent while it closes in is severe)."""
    lo = h.t_lo
    hi = h.contact if h.contact is not None else sc.T + 1.0
    mine = [s for s in gated if s.owner == h.name and heard_label(s)]     # (a cue cut before it named H warned nobody)
    news = [s for s in ref if s.owner == h.name and lo - 1e-9 <= s.t < hi - 1e-9 and not (read_mode and not read_kept(s))
            and not heard(sc, h, s.t, gated, s)]
    tb = news[0].t if news else None
    rb = _range_at(h, tb) if tb is not None else None
    tg = next((s.t for s in mine if lo - 1e-9 <= s.t < hi - 1e-9), None)
    severe = late = False
    if tb is not None:
        late = tg is None or tg > tb + STEP_S + 1e-6
        if h.contact is None:
            severe = tg is None
        elif h.contact - tb >= SEVERE_S - 1e-9:
            severe = tg is None or (tg > tb + STEP_S + 1e-6 and h.contact - tg < SEVERE_S - 1e-9)
    late_first = late
    every = None                                 # (bare cue, first gated cue from then on, severe): the worst
    if h.contact is not None:
        for s in news:
            if h.contact - s.t < SEVERE_S - 1e-9:
                break
            g = next((m.t for m in mine if s.t - 1e-9 <= m.t < hi - 1e-9), None)
            if g is not None and g <= s.t + STEP_S + 1e-6:
                continue
            sev = g is None or h.contact - g < SEVERE_S - 1e-9
            if every is None or (sev and not every[2]):
                every = (s.t, g, sev)
            if sev:
                break
    if every is not None:
        late = True
        severe = severe or every[2]
    return dict(tb=tb, tg=tg, rb=None if rb is None else round(rb, 2), contact=h.contact, severe=severe, late=late,
                late_first=late_first, every_tb=every[0] if every else None, every_tg=every[1] if every else None)


def run_scene(family: str, seed: int, policies: Sequence[str], configs: Optional[Sequence[str]] = None,
              trace: bool = False, exact: bool = False) -> Dict[str, object]:
    """All configurations of one scene for each policy: hazard rows, desk noise, cue counts. `configs` restricts them to
    those of the scene's; `exact` (--repro): exactly those (e.g. a +reply configuration on any scene)."""
    sc = build_scene(family, seed)
    frames = make_frames(sc)
    cfgs = list(configs) if exact and configs else [c for c in sc.configs if configs is None or c in configs]
    rows, desk, counts = [], [], []
    if trace:
        print(f"{sc.sid} T={sc.T:.1f} depth={sc.depth} noise={sc.noise} jitter={sc.jitter} goal={sc.goal}@{sc.t_goal} "
              f"rebaseline={sc.rebaseline_at} {sc.params}")
        for o in sc.objs:
            if o.hazard:
                print(f"  hazard {o.name} {o.label} window {o.t_lo:.1f}-{o.contact} ")
    for pol in policies:
        dropped: Dict[str, int] = {}
        runs = run_configs(sc, frames, pol, ["bare"] + cfgs, trace, dropped)
        for c in ["bare"] + cfgs:
            counts.append(dict(sid=sc.sid, family=family, policy=pol, config=c, n=len(runs[c]), T=sc.T,
                               dropped=dropped.get(c, 0)))
        for h in (o for o in sc.objs if o.hazard):
            for c in cfgs:
                j = judge(sc, h, runs["bare"], runs[c], read_mode=c == "ctl:read")
                j["blame"] = ""
                if (j["severe"] or j["late"]) and c.startswith("ctl:"):
                    key = c + ":nogate"
                    if key not in runs:
                        runs[key] = run_config(sc, frames, pol, key, trace)
                    jn = judge(sc, h, runs[key], runs[c])
                    j["blame"] = "gate" if (jn["severe"] if j["severe"] else jn["late"]) else "controller"
                    j["nogate_tb"] = jn["tb"] if jn["tb"] is not None else jn["every_tb"]
                elif j["severe"] or j["late"]:
                    j["blame"] = "gate"
                rows.append(dict(sid=sc.sid, family=family, seed=seed, variant=sc.variant, policy=pol, config=c,
                                 hazard=h.name, label=h.label, depth=sc.depth, **j))
        if ALIASES.get(family, family) in DESK_FAMILIES:
            for c in ["bare"] + cfgs:
                n_user = sum(1 for s in runs[c] if s.t >= DESK_AFTER_S and s.owner in ("user", "arm"))
                desk.append(dict(sid=sc.sid, seed=seed, variant=sc.variant, policy=pol, config=c, user_cues=n_user,
                                 depth=sc.depth))
    return dict(rows=rows, desk=desk, counts=counts)


# ---- the sweep -----------------------------------------------------------------------------------------------------
def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _work(args):
    family, seed, policies, configs = args
    try:
        return run_scene(family, seed, policies, configs)
    except Exception as e:                       # a broken scene is reported, not fatal for the sweep
        import traceback
        return dict(rows=[], desk=[], counts=[], error=f"{family}:{seed}: {type(e).__name__}: {e}\n{traceback.format_exc()}")


def sweep(families: Sequence[str], n_seeds: int, policies: Sequence[str], jobs: int = 1, seed_start: int = 0,
          configs: Optional[Sequence[str]] = None, progress: bool = True) -> Dict[str, object]:
    """n_seeds scenes dealt round-robin over the families (seed i -> family i mod #families)."""
    tasks = [(families[i % len(families)], seed_start + i, tuple(policies), tuple(configs) if configs else None)
             for i in range(n_seeds)]
    rows, desk, counts, errors = [], [], [], []
    t0 = time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs, initializer=_init_worker) as pool:
            for k, res in enumerate(pool.imap_unordered(_work, tasks, chunksize=2), 1):
                rows += res["rows"]; desk += res["desk"]; counts += res["counts"]
                if res.get("error"):
                    errors.append(res["error"])
                if progress and k % 50 == 0:
                    print(f"  {k}/{len(tasks)} scenes, {time.time() - t0:.0f} s, severe so far "
                          f"{sum(r['severe'] and r['blame'] == 'gate' for r in rows)}", flush=True)
    else:
        for k, task in enumerate(tasks, 1):
            res = _work(task)
            rows += res["rows"]; desk += res["desk"]; counts += res["counts"]
            if res.get("error"):
                errors.append(res["error"])
            if progress and k % 25 == 0:
                print(f"  {k}/{len(tasks)} scenes, {time.time() - t0:.0f} s", flush=True)
    return dict(rows=rows, desk=desk, counts=counts, errors=errors, seconds=time.time() - t0)


def _diverge_work(args):
    family, seed, policies = args
    try:
        sc = build_scene(family, seed)
        frames = make_frames(sc)
        return [d for p in policies for d in diverge(sc, frames, p)], None
    except Exception as e:
        import traceback
        return [], f"{family}:{seed}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def diverge_sweep(families: Sequence[str], n_seeds: int, policies: Sequence[str], jobs: int = 1, seed_start: int = 0):
    """diverge() over n_seeds scenes (round-robin over the families): (divergences, errors, runs checked)."""
    tasks = [(families[i % len(families)], seed_start + i, tuple(policies)) for i in range(n_seeds)]
    out, errors, t0 = [], [], time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs, initializer=_init_worker) as pool:
            for k, (res, err) in enumerate(pool.imap_unordered(_diverge_work, tasks, chunksize=2), 1):
                out += res
                errors += [err] if err else []
                if k % 50 == 0:
                    print(f"  {k}/{len(tasks)} scenes, {time.time() - t0:.0f} s, diverging so far {len(out)}", flush=True)
    else:
        for task in tasks:
            res, err = _diverge_work(task)
            out += res
            errors += [err] if err else []
    runs = sum(len([c for c in SAME_VIEW_CONFIGS if c in build_scene(f, s).configs]) for f, s, _ in tasks) * len(policies)
    return out, errors, runs


def desk_acceptance(desk: List[Dict]) -> Dict[str, object]:
    """Per desk variant and configuration: runs, user cues after DESK_AFTER_S (total, max per run), and the verdict
    for the operator configurations."""
    agg: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for d in desk:
        agg[(d["variant"], d["config"], d["policy"])].append(d["user_cues"])
    table, bad = [], []
    for (v, c, p), xs in sorted(agg.items()):
        # (a user who walked out of view and comes back from the side is someone walking up until they sit down: once)
        quiet = v in ("still", "fidget", "lean", "lean_multi", "drift", "arm", "still_flip", "flicker")
        limit = 0 if quiet else (1 if v.startswith("deep_") or v == "stand" else None)
        ok = None
        if c in OPERATOR_CONFIGS and limit is not None:
            ok = max(xs) <= limit
            if not ok:
                bad += [dict(sid=d["sid"], policy=p, config=c, variant=v, user_cues=d["user_cues"]) for d in desk
                        if d["variant"] == v and d["config"] == c and d["policy"] == p and d["user_cues"] > limit]
        table.append(dict(variant=v, config=c, policy=p, runs=len(xs), cues=sum(xs), max=max(xs),
                          runs_with_cue=sum(1 for x in xs if x), limit=limit, ok=ok))
    return dict(table=table, bad=bad)


def summarize(res: Dict[str, object], args) -> Dict[str, object]:
    rows = res["rows"]
    by = defaultdict(lambda: Counter())
    for r in rows:
        k = (r["family"], r["config"], r["policy"])
        by[k]["rows"] += 1
        by[k]["bare_warns"] += r["tb"] is not None
        by[k]["severe"] += bool(r["severe"] and r["blame"] == "gate")
        by[k]["severe_ctl"] += bool(r["severe"] and r["blame"] == "controller")
        by[k]["late"] += bool(r["late"] and r["blame"] == "gate")
        by[k]["late_ctl"] += bool(r["late"] and r["blame"] == "controller")
        by[k]["silent"] += bool(r["tb"] is not None and r["tg"] is None)
    cpm = defaultdict(lambda: [0, 0.0, 0])      # cues, seconds, most cues in one run
    for c in res["counts"]:
        k = (c["family"], c["config"], c["policy"])
        cpm[k][0] += c["n"]
        cpm[k][1] += c["T"]
        cpm[k][2] = max(cpm[k][2], c["n"])
    fam_table = []
    for (f, c, p) in sorted(set(by) | {k for k in cpm if k[1] != "bare"}):   # (reacq has no hazard rows)
        cnt = by.get((f, c, p), Counter())
        n, T, mx = cpm.get((f, c, p), [0, 0.0, 0])
        nb, Tb, mxb = cpm.get((f, "bare", p), [0, 0.0, 0])
        fam_table.append(dict(family=f, config=c, policy=p, **{k: cnt[k] for k in ("rows", "bare_warns", "severe",
                                                                                  "severe_ctl", "late", "late_ctl",
                                                                                  "silent")},
                              cues_per_min=round(60.0 * n / T, 2) if T else None,
                              bare_cues_per_min=round(60.0 * nb / Tb, 2) if Tb else None, max_cues=mx, bare_max_cues=mxb))
    severe = [r for r in rows if r["severe"] and r["blame"] == "gate"]
    severe_ctl = [r for r in rows if r["severe"] and r["blame"] == "controller"]
    late = [r for r in rows if r["late"] and r["blame"] == "gate"]
    da = desk_acceptance(res["desk"])
    # (the stranger's or visitor's first warning: the acceptance, as before every bare cue was judged)
    stranger = [r for r in rows if ALIASES.get(r["family"], r["family"]) in DESK_FAMILIES and r["config"] in OPERATOR_CONFIGS
                and r.get("late_first", r["late"]) and r["blame"] == "gate"]
    dropped = sum(c.get("dropped", 0) for c in res["counts"])
    scenes = len({r["sid"] for r in rows} | {d["sid"] for d in res["desk"]} | {c["sid"] for c in res["counts"]})
    return dict(
        args=dict(families=list(args.families_list), seeds=args.seeds, seed_start=args.seed_start,
                  policies=list(args.policies_list), configs=args.configs_list, jobs=args.jobs),
        scenes=scenes, rows=len(rows), seconds=round(res["seconds"], 1), errors=res["errors"][:20],
        totals=dict(severe=len(severe), severe_controller=len(severe_ctl), late=len(late),
                    late_controller=sum(1 for r in rows if r["late"] and r["blame"] == "controller"),
                    bare_warns=sum(1 for r in rows if r["tb"] is not None),
                    desk_bad=len(da["bad"]), desk_stranger_late=len(stranger), dropped=dropped),
        by_family=fam_table, severe=severe, severe_controller=severe_ctl[:200],
        late=sorted(late, key=lambda r: -((r["tg"] or 999) - (r["tb"] or 0)))[:300], desk=da,
        desk_stranger_late=stranger)


def repro(r: Dict) -> str:
    pol = "rl" if r["policy"] == "rl" else r["policy"]
    return (f"OMP_NUM_THREADS=2 .venv/bin/python scripts/live_safety.py --repro {r['sid']} --policies {pol} "
            f"--configs {r['config']}")


def write_md(s: Dict[str, object], path: Path) -> None:
    t = s["totals"]
    L = ["# Live safety harness", "",
         "Generated by `scripts/live_safety.py` (module docstring: scenes, configurations, severity). Synthetic scenes "
         "through the real IoUTracker, TrackDynamics, P-CACS and the learned scheduler; no camera, no video data.", "",
         f"- Command: `scripts/live_safety.py --families "
         f"{'all' if tuple(s['args']['families']) == FAMILIES else ','.join(s['args']['families'])} --seeds {s['args']['seeds']} "
         f"--policies {','.join(s['args']['policies'])} --jobs {s['args']['jobs']}`"
         + (f" (seed start {s['args']['seed_start']})" if s['args']['seed_start'] else ""),
         f"- {s['scenes']} scenes, {s['rows']} hazard rows (hazard x gated configuration x policy), {s['seconds']:.0f} s",
         f"- **Severe (gate): {t['severe']}**; severe only against the bare scheduler, not against the same controller "
         f"without the gate (controller): {t['severe_controller']}",
         f"- Late (> bare + 0.5 s, gate): {t['late']}; late (controller only): {t['late_controller']}; bare warned in "
         f"{t['bare_warns']} rows",
         f"- Desk: {t['desk_bad']} operator runs over their user-cue limit; stranger / visitor late in operator "
         f"configurations: {t['desk_stranger_late']}",
         f"- Cues spoken onto a busy channel (the live speaker drops them): {t.get('dropped', 0)}", "",
         "Severity is judged for the first bare cue that carries news and for every later one at least 1.5 s before "
         "contact; a bare cue carries no news when the gated user heard H in the 2 s before, or H, or another object of "
         "its label at H's place with its range said (never the desk user), in the 8 s before at about that range, or "
         "H at its range at any time while it has stayed there since; none of these once H has started to come at the "
         "user (its approach speed up by 0.5 m/s); a message an interrupt or a reply cut counts only up to the cut "
         "(module docstring). The cane family runs the controller with the cane on (+cane), fed rendered depth maps; "
         "the reply_cut family puts spoken replies on the channel (+reply).", ""]
    if s["errors"]:
        L += ["**Errors**", ""] + [f"- `{e.splitlines()[0]}`" for e in s["errors"]] + [""]
    L += ["## By family", "",
          "| family | config | policy | rows | bare warns | severe | late | silent | ctl-only severe / late | cues/min (bare) "
          "| max cues per run (bare) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["by_family"]:
        L.append(f"| {r['family']} | {r['config']} | {r['policy']} | {r['rows']} | {r['bare_warns']} | {r['severe']} | "
                 f"{r['late']} | {r['silent']} | {r['severe_ctl']} / {r['late_ctl']} | {r['cues_per_min']} "
                 f"({r['bare_cues_per_min']}) | {r.get('max_cues')} ({r.get('bare_max_cues')}) |")
    L += ["", "## Desk", "",
          "User cues after 8 s (warm-up, desk detection and re-warm-up done). Limit: 0 for a still, fidgeting, "
          "drifting, standing-up, arm or shallow-lean (< 0.35 m) user, 1 per deep-lean run; checked for the operator "
          "configurations (gate_op, ctl:desk, ctl:auto).", "",
          "| variant | config | policy | runs | user cues | max per run | runs with a cue | limit | ok |", "|---|---|---|---|---|---|---|---|---|"]
    for r in s["desk"]["table"]:
        L.append(f"| {r['variant']} | {r['config']} | {r['policy']} | {r['runs']} | {r['cues']} | {r['max']} | "
                 f"{r['runs_with_cue']} | {'' if r['limit'] is None else r['limit']} | "
                 f"{'' if r['ok'] is None else ('yes' if r['ok'] else '**no**')} |")
    if s["severe"]:
        L += ["", "## Severe rows", "",
              "| scene | policy | config | hazard | bare | gated | every news: bare -> gated | contact | repro |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in s["severe"]:
            L.append(f"| {r['sid']} | {r['policy']} | {r['config']} | {r['hazard']} {r['label']} | {r['tb']} | {r['tg']} | "
                     f"{r.get('every_tb')} -> {r.get('every_tg')} | {r['contact']} | `{repro(r)}` |")
    if s["severe_controller"]:
        L += ["", "## Severe against bare only (controller)", "",
              "The same controller without the gate is as late: the adapter's mode (verbosity, read-mode filter), not the gate.",
              "", "| scene | policy | config | hazard | bare | gated | no-gate ctl | contact |", "|---|---|---|---|---|---|---|---|"]
        for r in s["severe_controller"][:40]:
            L.append(f"| {r['sid']} | {r['policy']} | {r['config']} | {r['hazard']} {r['label']} | {r['tb']} | {r['tg']} | "
                     f"{r.get('nogate_tb')} | {r['contact']} |")
    if s["late"]:
        L += ["", "## Latest non-severe late rows (gate)", "",
              "Later than bare + 0.5 s but at least 1.5 s before contact (or bare itself warned under 1.5 s before contact).",
              "", "| scene | policy | config | hazard | bare | gated | contact |", "|---|---|---|---|---|---|---|"]
        for r in s["late"][:25]:
            L.append(f"| {r['sid']} | {r['policy']} | {r['config']} | {r['hazard']} {r['label']} | {r['tb']} | {r['tg']} | "
                     f"{r['contact']} |")
    path.write_text("\n".join(L) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default="all", help="comma list or 'all': " + ",".join(FAMILIES))
    ap.add_argument("--seeds", type=int, default=240, help="number of scenes, round-robin over the families")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--policies", default="pcacs,rl", help="pcacs, rl (pcacs_learned:rl), il")
    ap.add_argument("--configs", default=None, help="restrict the gated configurations (comma list)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--repro", default=None, help="family:seed: trace one scene (cues, gate reasons, releases)")
    ap.add_argument("--diverge", action="store_true",
                    help="check instead that the scheduler is shown and offers exactly what it does alone (SAME_VIEW_CONFIGS)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "live" / "safety_harness"))
    args = ap.parse_args(argv)
    args.families_list = list(FAMILIES) if args.families == "all" else [f for f in args.families.split(",") if f]
    bad = [f for f in args.families_list if f not in BUILDERS]
    if bad:
        ap.error(f"unknown families {bad}")
    args.policies_list = [p for p in args.policies.split(",") if p]
    args.configs_list = [c for c in args.configs.split(",") if c] if args.configs else None
    if args.repro:
        fam, seed = args.repro.rsplit(":", 1)
        res = run_scene(fam, int(seed), args.policies_list, args.configs_list, trace=True, exact=True)
        for r in res["rows"]:
            flag = "SEVERE" if r["severe"] else ("late" if r["late"] else "ok")
            print(f"{r['policy']:6} {r['config']:18} {r['hazard']:3} bare {r['tb']} gated {r['tg']} every {r['every_tb']} -> "
                  f"{r['every_tg']} contact {r['contact']} {flag} {r['blame']}")
        for d in res["desk"]:
            print(f"desk {d['policy']} {d['config']}: {d['user_cues']} user cues after {DESK_AFTER_S:.0f} s")
        return 0
    if args.diverge:
        out, errors, runs = diverge_sweep(args.families_list, args.seeds, args.policies_list, args.jobs, args.seed_start)
        print(f"{args.seeds} scenes, {runs} gated runs: {len(out)} where the scheduler was shown or offered anything "
              f"other than alone; errors {len(errors)}")
        for d in out[:30]:
            print("  DIVERGE", json.dumps(d, default=str)[:600])
        for e in errors[:5]:
            print("  ERROR", e)
        return 1 if out or errors else 0
    res = sweep(args.families_list, args.seeds, args.policies_list, args.jobs, args.seed_start, args.configs_list)
    s = summarize(res, args)
    t = s["totals"]
    print(f"{s['scenes']} scenes, {s['rows']} rows in {s['seconds']:.0f} s: severe {t['severe']} (controller-only "
          f"{t['severe_controller']}), late {t['late']}, desk over limit {t['desk_bad']}, stranger late "
          f"{t['desk_stranger_late']}, dropped on a busy channel {t['dropped']}, errors {len(s['errors'])}")
    for r in s["severe"][:30]:
        print("  SEVERE", r["sid"], r["policy"], r["config"], r["hazard"], r["label"], "bare", r["tb"], "gated", r["tg"],
              "every", r["every_tb"], "->", r["every_tg"], "contact", r["contact"], "->", repro(r))
    for b in s["desk"]["bad"][:30]:
        print("  DESK", b)
    for e in s["errors"][:5]:
        print("  ERROR", e)
    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(s, indent=1, default=float))
        write_md(s, out.with_suffix(".md"))
        print("wrote", out.with_suffix(".md"), out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
