"""Everything the live demo adds on top of a scheduler, in one object `scripts/run_live.py` can drive.

    ctl = LiveController(policy, mode="auto", setup="auto")
    pipe.policy = ctl.policy                         # the scheduler the pipeline calls, wrapped
    ...
    ctl.on_frame(frame, t, res.tracks, depth=pipe.depth_map)   # every frame: activity, mode, desk detection, cane
    speaker.say(ctl.summary(res.tracks))             # "what is around me?"

Layers, inside out: the scheduler (P-CACS or the learned GRU) proposes a cue; the AnnouncementGate holds repeats,
background objects and unconfirmed tracks; the TaskAdapter's read mode drops non-critical object cues. The
ActivityDetector picks the mode (stationary -> scene, walking or turning -> navigate, steady with text -> read, a FIND
goal -> find) unless a mode is pinned. A mode sets the gate (hold times, background, read-mode muting), never the
scheduler's own settings: a scheduler with other settings decides other cues than alone (with the scene mode's quiet
verbosity the learned scheduler never named a person walking at the user, harness side_entry:71409), so a quiet mode is
quiet only through the gate, whose holds pass anything with safety news; the verbosity stays the user's (keys, voice
commands). With setup "auto", a camera that is still at the end of the warm-up is taken to be a desk webcam, and the
large centred person in front of it (the user) is not announced unless they approach; once the camera frames show
walking for DESK_END_WALK_S without a break (a phone held still at the start, then carried), the setup is handheld for
good: the operator filter off, the cane on. Walking by the tracks alone does not end it (people walking past a desk
webcam churn tracks).

The cane (seeforme/live/cane.py, on unless cane=False) adds what the detector cannot name: walls, doors, poles, low
and head-height obstacles, from the metric depth map passed to on_frame. aid ("cane" by default, "guide-dog" or
"none": what the user travels with) keeps it quiet about what the user's own cane or dog already covers (cane.py); the
scheduler's cues are the same for every aid. Spoken replies to voice commands (run_live
registers them on the pipeline's channel with reply_cue) are the other message the scheduler did not choose. One cue
per decision at most, and neither may make the scheduler warn later than it would alone:
    - the scheduler (with its gate) is shown the channel it would have without the cane and the replies, built from its
      own cues only: each holds it from the decision it was offered at, for its spoken length, whether it was said
      then, said later (it lost its decision to the cane, or waited for the channel) or not at all; while the real
      channel plays exactly that cue, said when offered, the real channel. So a cane cue, a reply or a cue said late
      never makes its channel busier or freer than alone (a busier one held back an in-path chair behind a spoken
      "Obstacle ahead" or a 9 s help text; one busy with a cue the cane had displaced kept the scheduler off a channel
      it had free alone, gate_verify_r3). A safety cue it offers while the real channel plays another message cuts it
      ("interrupt"); a routine one waits for the channel
    - in the same decision a scheduler cue wins, unless the cane cue is more urgent (by more than 0.05) and the
      scheduler cue is not about safety (commands.is_safety_cue); a haptic "Stop." (something within 0.6 m ahead while
      walking) wins over any spoken cue, since it does not occupy the audio channel
    - a scheduler cue that lost its decision or waits for the channel was already recorded as said by the gate and the
      scheduler, so it is said at a later decision (within 3 s): a safety cue at the next decision, cutting whatever
      plays (a stop delays it by one decision at most), a routine one once the channel is free; the most urgent first.
      A routine cue offered while a long reply plays may wait longer than that and is then dropped
    - a cane cue may cut a reply (a reply never holds an obstacle warning back) but never a scheduler message
    - a message the controller put on the channel that a spoken reply (seen at the next decision: the channel plays a
      reply it did not play before) or a later interrupt cut is reported to the gate (AnnouncementGate.cut), with the
      least the user may have heard: cut before what it is about and how far, the gate takes it back from its memory,
      so the scheduler's next cue about that object is not held as a repeat (v4_flip_size:970079)
    - the cane is off in desk setup (also while auto setup takes the camera for a desk), without a depth map, and before
      the desk detection can have run in auto setup (warm-up + 1.5 s); quiet while still unless the obstacle approaches;
      muted in read mode except a stop
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
from .commands import is_safety_cue
from .gate import AnnouncementGate, GateConfig, _speech_s
from .summary import SessionLog, scene_summary

SETUPS = ("auto", "desk", "handheld")
DISPLACED_KEEP_S = 3.0           # a scheduler cue that lost its decision or waits for the channel: said within this long
DESK_END_WALK_S = 3.0            # auto setup: the camera frames show walking this long without a break: no desk webcam
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
        ch_in = self._ctl._scheduler_channel(ch, t)
        return self._ctl._after(self._inner.decide(tracks, t, ch_in, *a, **kw), t, ch=ch, tracks=tracks, shown=ch_in)

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
        self._desk_ended = False               # auto setup: a desk detected, then the camera walked (handheld for good)
        self._walk_since: Optional[float] = None   # walking (by the camera frames) without a break since
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
        self._own_msg: Optional[Tuple[Cue, float, float]] = None   # the scheduler's last cue on its own channel (without
                                                                     # the cane and the replies): (cue, offered, end)
        self._said: Optional[Tuple[Cue, float]] = None   # the last message put on the channel (the scheduler's cue as
                                                         # offered when said at its time)
        self._on_air: Optional[Tuple[Cue, float]] = None   # the last message put on the channel, as put there, and when
        self._prev_dec: Tuple[Optional[float], bool] = (None, False)   # the previous decision: time, a reply playing
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
        if self.gate is not None:
            self.gate._own_cuts = False         # (it is shown the scheduler's channel; cuts are reported here: _note_cuts)
        self.adapter.set_gate(self.gate)
        # the modes set the gate only, never the scheduler's own settings: a scheduler with other settings decides other
        # cues than alone (harness side_entry:71409: with the scene mode's quiet verbosity the learned scheduler never
        # named a person walking at the user, whom it names at once with its own settings). Verbosity stays the user's
        # (keys, voice commands)
        self.adapter.set_policy(None)
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

    def _scheduler_channel(self, ch, t: float):
        """The channel the scheduler (with its gate) is shown: the one it would have without the cane and the replies
        (module docstring), built from its own cues only. While the real channel plays its last cue, said when it was
        offered, the real channel (the same message; the live speaker may run longer than the estimate). Otherwise its
        last cue (said, said later, or lost to the cane) while that would still be playing alone, from when it was
        offered, else a free channel, whatever the real channel plays (a cane cue, a reply, a cue said late). A channel
        busier than alone holds back a warning it would give alone, a freer one lets it speak where alone it would wait:
        either way its next cues are not the ones it gives alone. ch None (no channel modelled): None."""
        if ch is None:
            return None
        a, last = self._own_msg, self._said
        if (getattr(ch, "busy", False) and a is not None and last is not None and last[0] is a[0]
                and getattr(ch, "current_track", None) == a[0].track_id):
            return ch
        if a is not None and a[2] > t + 1e-9:
            m = a[0]
            return ChannelState(busy=True, remaining_s=a[2] - t, current_utility=m.utility or 0.0,
                                current_severity=m.severity or 0.0, current_track=m.track_id)
        return ChannelState()

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

    def _merge(self, kept: List[Cue], t: float, ch, tracks) -> Tuple[List[Cue], bool]:
        """One cue for this decision from the scheduler's (decided on the channel _scheduler_channel showed it), the
        cues waiting for the channel and the cane's (module docstring); and whether it is the scheduler's cue, said now."""
        busy = bool(getattr(ch, "busy", False)) if ch is not None else False
        cur_track = getattr(ch, "current_track", None) if busy else None
        sched = kept[0] if kept else None
        late = None
        if sched is not None and busy and sched.action == "speak":
            # shown its own channel, free, while the real one plays a cane cue, a reply or a cue said late: the speaker
            # would drop a "speak"
            if self._safety_cue(sched, tracks):          # cuts it, now
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
        return ([out] if out is not None else []), (out is not None and out is sched)

    def _note_reply_cut(self, ch, t: float) -> None:
        """A spoken reply (run_live speak_reply: interrupt=True) plays now and did not at the previous decision, while the
        message the controller put on the channel before would still have been playing then: the reply cut it, somewhere
        after the previous decision. The gate is told (AnnouncementGate.cut) with the words heard up to the previous
        decision, the least the user may have heard: a message cut before what it is about and how far was not heard,
        so the gate does not hold the scheduler's next cue about that object as a repeat (v4_flip_size:970079)."""
        reply = held_by(ch) == "reply"
        prev_t, prev_reply = self._prev_dec
        self._prev_dec = (t, reply)
        on = self._on_air
        if not reply or prev_reply or prev_t is None or on is None:
            return
        cue, ts = on
        if ts <= prev_t + 1e-9 and ts + _speech_s(cue) > prev_t + 1e-9:
            self._on_air = None
            self._tell_cut(cue, ts, prev_t - ts)

    def _tell_cut(self, cue: Cue, ts: float, heard_s: float) -> None:
        if self.gate is None or cue.policy == CANE_POLICY or cue.track_id < 0 or cue.action == "haptic":
            return                               # (the cane keeps its own memory; a reply is no cue of the gate's)
        if self.gate.cut(cue, ts, heard_s):
            self.cane_stats["cut"] = self.cane_stats.get("cut", 0) + 1

    def _after(self, cues, t: float, ch=None, tracks=None, shown=None):
        if ch is not None:
            self._note_reply_cut(ch, t)
        kept = self.adapter.filter_cues(cues, tracks=tracks, goal=self.goal_tags)
        self.muted += len(cues or []) - len(kept)
        c0 = kept[0] if kept else None
        if (c0 is not None and c0.policy != CANE_POLICY and c0.action in ("speak", "interrupt")
                and (c0.action == "interrupt" or not getattr(shown, "busy", False))):
            # without the cane and the replies this cue holds the channel from now for its spoken length, whether it is
            # said now, later or not at all (a "speak" onto a busy channel is dropped by the speaker, alone too)
            self._own_msg = (c0, t, t + _speech_s(c0))
        kept, own = self._merge(kept, t, ch, tracks)
        if kept and kept[0].action in ("speak", "interrupt"):
            on = self._on_air
            if (kept[0].action == "interrupt" and on is not None and getattr(ch, "busy", False)
                    and t < on[1] + _speech_s(on[0]) - 1e-9):
                self._tell_cut(on[0], on[1], t - on[1])   # (this cue cuts the message the controller put there before)
            self._said = (c0 if own else kept[0], t)
            self._on_air = (kept[0], t)
        if kept and self.log is not None:
            self.log.cue(t, kept[0], mode=self.adapter.mode, activity=self.activity.activity.value)
        return kept

    # ---- per frame ------------------------------------------------------------------------------------------
    def on_frame(self, frame, t: float, tracks=None, depth=None) -> str:
        """Update activity and mode from this frame (and its tracks), and the cane from the metric depth map the
        pipeline computed (LivePipeline.depth_map; the same map passed again is ignored); returns the mode."""
        self._t = t
        act = self.activity.update(frame, t, tracks)
        # walking, by the camera frames (the track path is weak evidence: people walking past a desk webcam churn tracks)
        walking = act in (Activity.WALKING, Activity.TURNING) and self.activity.state.source == "frame"
        self._walk_since = (self._walk_since if self._walk_since is not None else t) if walking else None
        if self.setup == "auto" and not self.desk and self.gate is not None and not self._desk_ended:
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
        elif (self.setup == "auto" and self.desk and self._walk_since is not None
              and t - self._walk_since >= DESK_END_WALK_S - 1e-9):
            # the camera walks: not a desk webcam (a phone held still at the start, then carried). Handheld from now on:
            # the operator filter would hold whoever is large and centred, and the cane looks where the user walks
            self.desk, self._desk_ended = False, True
            self._gate_cfg.operator_filter = False
            if self.gate is not None:
                self.gate.set_mode_params(operator_filter=False)
                self.gate.rebaseline()
            if self.cane is not None:
                self.cane.reset()
            if self.log is not None:
                self.log.write("setup", t, setup=f"handheld (detected: walking for {t - self._walk_since:.1f} s)")
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
        self._own_msg = self._said = self._on_air = None

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
