"""Everything the live demo adds on top of a scheduler, in one object `scripts/run_live.py` can drive.

    ctl = LiveController(policy, mode="auto", setup="auto")
    pipe.policy = ctl.policy                         # the scheduler the pipeline calls, wrapped
    ...
    ctl.on_frame(frame, t, res.tracks, depth=pipe.depth_map)   # every frame: activity, mode, desk detection, cane
    speaker.say(ctl.summary(res.tracks))             # "what is around me?"

Layers, inside out: the scheduler (P-CACS or the learned GRU) proposes a cue; the AnnouncementGate holds repeats,
background objects and unconfirmed tracks; the TaskAdapter's read mode drops non-critical object cues. The
ActivityDetector picks the mode (stationary -> scene, walking or turning -> navigate, steady with text -> read, a FIND
goal -> find) unless a mode is pinned. With setup "auto", a camera that is still at the end of the warm-up is taken
to be a desk webcam, and the large centred person in front of it (the user) is not announced unless they approach.

The cane (seeforme/live/cane.py, on unless cane=False) adds what the detector cannot name: walls, doors, poles, low
and head-height obstacles, from the metric depth map passed to on_frame. aid ("cane" by default, "guide-dog" or
"none": what the user travels with) keeps it quiet about what the user's own cane or dog already covers (cane.py); the
scheduler's cues are the same for every aid. Spoken replies to voice commands (run_live
registers them on the pipeline's channel with reply_cue) are the other message the scheduler did not choose. One cue
per decision at most, and neither may make the scheduler warn later than it would alone:
    - while a cane cue or a reply holds the channel (ChannelState.current_track < 0) and something in view could need
      a safety cue (commands.is_safety_track: in the path, approaching, a vehicle or stairs, the FIND goal), the
      scheduler is shown the channel as it would be without them: busy with its own last message while that would
      still be playing (a reply may have cut it), else free. A safety cue it then offers cuts the cane cue or the reply
      ("interrupt"); a routine one waits for the channel. P-CACS (and the gate) cut a message only for a cue of higher
      severity that is critical or scores far above it, so a spoken "Obstacle ahead" (severity 0.7-0.85) or a 9 s help
      text would otherwise hold an in-path chair or an approaching person back
    - in the same decision a scheduler cue wins, unless the cane cue is more urgent (by more than 0.05) and the
      scheduler cue is not about safety (commands.is_safety_cue); a haptic "Stop." (something within 0.6 m ahead while
      walking) wins over any spoken cue, since it does not occupy the audio channel
    - a scheduler cue that lost its decision or waits for the channel was already recorded as said by the gate and the
      scheduler, so it is said at a later decision (within 3 s): a safety cue at the next decision, cutting whatever
      plays (a stop delays it by one decision at most), a routine one once the channel is free; the most urgent first
    - a cane cue may cut a reply (a reply never holds an obstacle warning back) but never a scheduler message
    - the cane is off in desk setup (also once auto setup detects a desk), without a depth map, and before the desk
      detection can have run in auto setup (warm-up + 1.5 s); quiet while still unless the obstacle approaches; muted in
      read mode except a stop
Cane cues are logged like other cues (policy "cane", track -7). cane_note() is the cane's confirmed obstacle ahead in
words, for answers ("is the path clear?").
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..cues.pcacs import ChannelState
from ..geometry.camera import CameraModel
from ..types import Cue
from .activity import Activity, ActivityDetector, MODES, TaskAdapter
from .cane import AIDS, CANE_POLICY, CANE_TRACK_ID, Cane, CaneConfig
from .commands import is_safety_cue, is_safety_track
from .gate import AnnouncementGate, GateConfig
from .summary import SessionLog, scene_summary

SETUPS = ("auto", "desk", "handheld")
DISPLACED_KEEP_S = 3.0           # a scheduler cue that lost its decision or waits for the channel: said within this long
MAX_WAITING = 3                  # ... at most this many wait (the least urgent beyond is dropped)
CANE_MARGIN = 0.05               # the cane cue must be this much more urgent than the scheduler's to win
REPLY_TRACK_ID = -1              # Cue.track_id of a spoken reply on the channel (reply_cue); scheduler tracks are >= 1
REPLY_POLICY = "reply"


def reply_cue(text: str, t: float) -> Cue:
    """The channel message of a spoken reply to a command (run_live passes it to pipe.channel.speak): no severity and no
    utility, and a track id the controller recognises, so a safety cue cuts it (module docstring)."""
    return Cue(t=t, track_id=REPLY_TRACK_ID, label="reply", kind="text", message=text, score=0.0, policy=REPLY_POLICY)


def held_by(ch) -> Optional[str]:
    """"cane" / "reply" when the busy channel plays a message the scheduler did not choose, else None."""
    if ch is None or not getattr(ch, "busy", False):
        return None
    cur = getattr(ch, "current_track", None)
    if cur is None or cur >= 0:
        return None
    return "cane" if cur == CANE_TRACK_ID else "reply"


class _ModeFilter:
    """The policy the pipeline sees: calls the gate (or the bare scheduler) with the channel it would have without the
    cane and the replies, drops object cues in read mode and merges the cane's cue. Every other attribute is the wrapped
    policy's (name, cfg, decision_interval, set_goal ...)."""

    def __init__(self, inner, controller: "LiveController"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_ctl", controller)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)

    def decide(self, tracks, t, ch=None, *a, **kw):
        ch_in, freed = self._ctl._scheduler_channel(ch, tracks, t)
        return self._ctl._after(self._inner.decide(tracks, t, ch_in, *a, **kw), t, ch=ch, tracks=tracks, freed=freed)

    def select(self, tracks, t, ctx=None, *a, **kw):
        return self._ctl._after(self._inner.select(tracks, t, ctx, *a, **kw), t, tracks=tracks)

    def reset(self):
        self._inner.reset()


def cue_urgency(c: Cue) -> float:
    """0..1: class severity, raised by a short measured time to contact and by haptic / interrupt actions."""
    u = float(getattr(c, "severity", 0.0) or 0.0)
    ttc = getattr(c, "ttc_s", None)
    if ttc is not None and math.isfinite(ttc):
        u = max(u, 1.0 - min(1.0, max(0.0, ttc) / 6.0))
    if getattr(c, "action", "speak") in ("haptic", "interrupt"):
        u = max(u, 0.9)
    return u


class LiveController:
    def __init__(self, policy, mode: str = "auto", setup: str = "auto", gate: bool = True,
                 gate_cfg: Optional[GateConfig] = None, camera: Optional[CameraModel] = None,
                 log: Optional[SessionLog] = None, goal_tags: Optional[Sequence[str]] = None, units: str = "m",
                 cane: bool = True, cane_cfg: Optional[CaneConfig] = None, aid: str = "cane"):
        if setup not in SETUPS:
            raise ValueError(f"setup must be one of {', '.join(SETUPS)}")
        if mode != "auto" and mode not in MODES:
            raise ValueError(f"mode must be auto or one of {', '.join(MODES)}")
        if aid not in AIDS:
            raise ValueError(f"aid must be one of {', '.join(AIDS)}")
        self.aid = aid
        self.camera = camera or CameraModel()
        self.setup = setup
        self.use_gate = gate
        self.use_cane = cane
        self.units = units
        self.log = log
        self.goal_tags: List[str] = list(goal_tags or [])
        self._gate_cfg = gate_cfg or GateConfig()
        if setup == "desk":
            self._gate_cfg.operator_filter = True
        self.desk = setup == "desk"            # a still camera with the user in front of it
        self.activity = ActivityDetector()
        self.adapter = TaskAdapter()
        self.cane: Optional[Cane] = Cane(cane_cfg, aid=aid) if cane else None
        self.gate: Optional[AnnouncementGate] = None
        self.policy = None
        self.set_policy(policy)
        if mode != "auto":
            self.adapter.force_mode(mode)
        self.last_mode = self.adapter.mode
        self.last_activity = self.activity.activity
        self._t = 0.0
        self.muted = 0
        self._waiting: List[Tuple[Cue, float]] = []   # scheduler cues already marked as said, not yet spoken (since when)
        self._own_msg: Optional[Cue] = None             # the scheduler's last spoken cue (its channel without the others)
        self.cane_stats: Dict[str, int] = {"won": 0, "lost": 0, "displaced_said": 0, "displaced_dropped": 0,
                                           "cut_in": 0, "waited": 0}

    # ---- wiring ---------------------------------------------------------------------------------------------
    def set_policy(self, policy) -> None:
        """Wrap a (new) scheduler: keyboard policy switches keep the gate's settings, mode and desk detection."""
        if self.use_gate:
            self.gate = AnnouncementGate(policy, GateConfig(**vars(self._gate_cfg)), camera=self.camera)
            inner = self.gate
        else:
            self.gate = None
            inner = policy
        self.adapter.set_gate(self.gate)
        self.adapter.set_policy(inner)
        if self.goal_tags and hasattr(inner, "set_goal"):
            inner.set_goal(self.goal_tags)
        self.policy = _ModeFilter(inner, self)

    def set_goal(self, tags: Sequence[str]) -> None:
        self.goal_tags = [g for g in tags if g]
        inner = self.gate if self.gate is not None else self.policy._inner
        if hasattr(inner, "set_goal"):
            inner.set_goal(self.goal_tags)

    # ---- the cane -------------------------------------------------------------------------------------------
    @property
    def cane_active(self) -> bool:
        """The cane may speak: on, not a desk, and (auto setup) past the moment the desk detection decides."""
        if self.cane is None or self.desk:
            return False
        if self.setup == "auto":
            warm = self.gate.gate_cfg.warmup_s if self.gate is not None else self._gate_cfg.warmup_s
            return self._t >= warm + 1.5
        return True

    def _safety_cue(self, c: Cue, tracks) -> bool:
        """A scheduler cue the cane and the replies must never delay (commands.is_safety_cue with its track and the
        FIND goal): vehicle, approaching or in-path object, FIND goal, haptic or interrupt, critical severity."""
        trk = next((x for x in (tracks or ()) if x.track_id == c.track_id), None)
        return is_safety_cue(c, trk, self.goal_tags)

    def _scheduler_channel(self, ch, tracks, t: float):
        """(the channel the scheduler is shown, freed): while a cane cue or a reply holds it and something in view could
        need a safety cue, the channel as it would be without them (module docstring): busy with the scheduler's own
        last message while that would still be playing, else free. The real channel otherwise."""
        if held_by(ch) is None or not any(is_safety_track(tr, self.goal_tags) for tr in (tracks or ())):
            return ch, False
        m = self._own_msg
        end = m.t + (m.duration_s or 0.0) if m is not None else -math.inf
        if end > t + 1e-9:
            return ChannelState(busy=True, remaining_s=end - t, current_utility=m.utility or 0.0,
                                current_severity=m.severity or 0.0, current_track=m.track_id), True
        return ChannelState(), True

    # ---- cues that wait (already marked as said upstream) -----------------------------------------------------
    def _wait(self, c: Cue, t: float) -> None:
        self._waiting = [w for w in self._waiting if w[0].track_id != c.track_id] + [(c, t)]
        if len(self._waiting) > MAX_WAITING:
            self._waiting.remove(min(self._waiting, key=lambda w: (cue_urgency(w[0]), w[1])))
            self.cane_stats["displaced_dropped"] += 1

    def _expire(self, t: float, busy: bool, cur_track) -> None:
        keep = []
        for w in self._waiting:
            if t - w[1] > DISPLACED_KEEP_S + 1e-9 or (busy and w[0].track_id == cur_track):
                self.cane_stats["displaced_dropped"] += 1       # stale, or its object is being talked about now
            else:
                keep.append(w)
        self._waiting = keep

    def _ready(self, busy: bool, tracks) -> Optional[Tuple[Cue, float]]:
        """The most urgent waiting cue that may go now: a safety cue always (it cuts what plays), a routine one only on
        a free channel."""
        for w in sorted(self._waiting, key=lambda w: -cue_urgency(w[0])):
            if not busy or self._safety_cue(w[0], tracks):
                return w
        return None

    def _merge(self, kept: List[Cue], t: float, ch, tracks, freed: bool = False) -> List[Cue]:
        """One cue for this decision from the scheduler's (decided on the channel _scheduler_channel showed it), the
        cues waiting for the channel and the cane's (module docstring)."""
        busy = bool(getattr(ch, "busy", False)) if ch is not None else False
        cur_track = getattr(ch, "current_track", None) if busy else None
        sched = kept[0] if kept else None
        late = None
        if sched is not None and freed and sched.action in ("speak", "interrupt"):
            if self._safety_cue(sched, tracks):          # cuts the cane cue or the reply, now
                sched = replace(sched, action="interrupt", interrupted_track=cur_track)
                self.cane_stats["cut_in"] += 1
            else:                                        # routine: waits for the channel
                late, sched = sched, None
        self._expire(t, busy, cur_track)
        if sched is not None and sched.action != "haptic":
            self._waiting = [w for w in self._waiting if w[0].track_id != sched.track_id]   # said again now
        ready = self._ready(busy, tracks)
        cc = None
        if self.cane is not None and self.cane_active:
            moving = self.activity.activity not in (Activity.STATIONARY, Activity.READING)
            cc = self.cane.propose(t, moving=moving, busy=busy and held_by(ch) != "reply",
                                   read_mode=self.adapter.mode == "read", units=self.units)
            if cc is not None and busy and cc.action != "haptic":
                cc.action, cc.interrupted_track = "interrupt", cur_track     # a reply never holds an obstacle back
        out = sched
        if cc is not None and cc.action == "haptic":
            self.cane.commit(cc, t)
            if sched is None or sched.action != "haptic":                    # (both buzz: the scheduler's serves)
                out = cc
                if sched is not None:
                    self.cane_stats["won"] += 1
                    self._wait(sched, t)                                     # one decision later
        elif sched is not None and self._safety_cue(sched, tracks):
            self.cane_stats["lost"] += int(cc is not None)
        elif ready is not None and (sched is None or cue_urgency(ready[0]) >= cue_urgency(sched)):
            self._waiting.remove(ready)
            out = replace(ready[0], t=t, action="interrupt" if busy else "speak", interrupted_track=cur_track)
            self.cane_stats["displaced_said"] += 1
            if sched is not None:
                self._wait(sched, t)                                         # the one that lost goes next
        elif cc is not None and (sched is None or cue_urgency(cc) > cue_urgency(sched) + CANE_MARGIN):
            self.cane.commit(cc, t)
            out = cc
            if sched is not None:                                            # won over a scheduler cue
                self.cane_stats["won"] += 1
                self._wait(sched, t)
        elif cc is not None:
            self.cane_stats["lost"] += 1
        if late is not None:
            self._wait(late, t)
            self.cane_stats["waited"] += 1
        return [out] if out is not None else []

    def _after(self, cues, t: float, ch=None, tracks=None, freed: bool = False):
        kept = self.adapter.filter_cues(cues)
        self.muted += len(cues or []) - len(kept)
        kept = self._merge(kept, t, ch, tracks, freed)
        if kept and kept[0].policy != CANE_POLICY and kept[0].action in ("speak", "interrupt"):
            self._own_msg = kept[0]              # (its duration_s is set by the scheduler, then by the channel)
        if kept and self.log is not None:
            self.log.cue(t, kept[0], mode=self.adapter.mode, activity=self.activity.activity.value)
        return kept

    # ---- per frame ------------------------------------------------------------------------------------------
    def on_frame(self, frame, t: float, tracks=None, depth=None) -> str:
        """Update activity and mode from this frame (and its tracks), and the cane from the metric depth map the
        pipeline computed (LivePipeline.depth_map; the same map passed again is ignored); returns the mode."""
        self._t = t
        act = self.activity.update(frame, t, tracks)
        if self.setup == "auto" and not self.desk and self.gate is not None:
            warm = self.gate.gate_cfg.warmup_s
            if t >= warm + 1.5 and act == Activity.STATIONARY and t < warm + 15.0:
                # still camera right after start: a desk / laptop webcam. The person sitting in front of it is the user.
                self.desk = True
                self._gate_cfg.operator_filter = True
                self.gate.set_mode_params(operator_filter=True)
                self.gate.rebaseline()
                if self.cane is not None:
                    self.cane.reset()                 # a desk webcam does not look where the user walks
                if self.log is not None:
                    self.log.write("setup", t, setup="desk (detected: camera still after warm-up)")
        if self.cane is not None and not self.desk and depth is not None:
            self.cane.update(depth, t, tracks)
        mode = self.adapter.step(act, t, has_goal=bool(self.goal_tags))
        if self.log is not None and (mode != self.last_mode or act != self.last_activity):
            self.log.write("mode", t, mode=mode, activity=act.value, reason=self.adapter.reason)
        self.last_mode, self.last_activity = mode, act
        return mode

    def camera_changed(self) -> None:
        """Another camera (flip): the gate's background, the activity detector's motion history, the cane's floor and
        sector memory and the cues waiting for the channel are all about the old view."""
        if self.gate is not None:
            self.gate.rebaseline()
        self.activity.reset()
        if self.cane is not None:
            self.cane.reset()
        self._waiting = []
        self._own_msg = None

    def cycle_mode(self) -> str:
        """Keyboard: auto -> navigate -> scene -> read -> find -> auto. Returns a short phrase to speak."""
        o = self.adapter.cycle_override()
        self.adapter.step(self.activity.activity, self._t, has_goal=bool(self.goal_tags))
        if self.log is not None:
            self.log.write("key", self._t, key="mode", override=o or "auto")
        return f"Mode {o}." if o else f"Automatic mode, now {self.adapter.mode}."

    # ---- on demand ------------------------------------------------------------------------------------------
    def summary(self, tracks) -> str:
        return scene_summary(tracks, self.camera, units=self.units)

    def cane_note(self, t: Optional[float] = None, units: Optional[str] = None, commit: bool = False) -> str:
        """The cane's confirmed obstacle ahead in words ("The depth map shows an obstacle ahead, about 1 meter."), or ""
        (no cane, a desk, no recent depth map, not a walking view, or a detected track explains it). commit=True when
        a reply says it: the cane then holds that obstacle as said (Cane.note)."""
        if self.cane is None or self.desk:
            return ""
        return self.cane.note(self._t if t is None else t, units or self.units, commit=commit)

    def cane_state(self) -> str:
        if self.cane is None:
            return "off"
        if self.desk:
            return "off (desk)"
        if self.cane.frame is None:
            return "no depth"
        return "on" if self.cane_active else "warming up"

    def status(self) -> dict:
        g = self.gate
        return {"mode": self.adapter.mode + ("*" if self.adapter.override else ""), "activity": self.activity.activity.value,
                "setup": "desk" if self.desk else ("handheld" if self.setup == "handheld" else "auto"),
                "gate": (g.last_reason or "") if g is not None else "off", "cane": self.cane_state()}

    def report(self) -> List[str]:
        lines = [f"mode: {self.adapter.describe()}; activity: {self.activity.describe()}"]
        if self.gate is not None:
            lines.append(f"gate: {self.gate.describe()}")
        if self.cane is not None:
            s = self.cane_stats
            lines.append(f"{self.cane.describe()} [{self.cane_state()}]; merge: won {s['won']}, lost {s['lost']} to the "
                         f"scheduler, displaced cues said {s['displaced_said']}, dropped {s['displaced_dropped']}")
        s = self.cane_stats
        if s["cut_in"] or s["waited"]:
            lines.append(f"channel: {s['cut_in']} safety cue(s) cut a cane cue or a reply, {s['waited']} routine cue(s) "
                         "waited for it")
        if self.muted:
            lines.append(f"read mode muted {self.muted} object cue(s)")
        return lines


__all__ = ["LiveController", "SETUPS", "AIDS", "CANE_POLICY", "REPLY_TRACK_ID", "cue_urgency", "held_by", "reply_cue"]
