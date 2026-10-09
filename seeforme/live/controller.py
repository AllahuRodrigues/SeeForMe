"""Everything the live demo adds on top of a scheduler, in one object `scripts/run_live.py` can drive.

    ctl = LiveController(policy, mode="auto", setup="auto")
    pipe.policy = ctl.policy                         # the scheduler the pipeline calls, wrapped
    ...
    ctl.on_frame(frame, t, res.tracks)               # every frame: activity, mode, desk detection
    speaker.say(ctl.summary(res.tracks))             # "what is around me?"

Layers, inside out: the scheduler (P-CACS or the learned GRU) proposes a cue; the AnnouncementGate holds repeats,
background objects and unconfirmed tracks; the TaskAdapter's read mode drops non-critical object cues. The
ActivityDetector picks the mode (stationary -> scene, walking or turning -> navigate, steady with text -> read, a FIND
goal -> find) unless a mode is pinned. With setup "auto", a camera that is still at the end of the warm-up is taken
to be a desk webcam, and the large centred person in front of it (the user) is not announced unless they approach.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..geometry.camera import CameraModel
from .activity import Activity, ActivityDetector, MODES, TaskAdapter
from .gate import AnnouncementGate, GateConfig
from .summary import SessionLog, scene_summary

SETUPS = ("auto", "desk", "handheld")


class _ModeFilter:
    """The policy the pipeline sees: calls the gate (or the bare scheduler) and then drops object cues in read mode.
    Every other attribute is the wrapped policy's (name, cfg, decision_interval, set_goal, last_observations ...)."""

    def __init__(self, inner, controller: "LiveController"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_ctl", controller)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)

    def decide(self, tracks, t, ch=None, *a, **kw):
        return self._ctl._after(self._inner.decide(tracks, t, ch, *a, **kw), t)

    def select(self, tracks, t, ctx=None, *a, **kw):
        return self._ctl._after(self._inner.select(tracks, t, ctx, *a, **kw), t)

    def reset(self):
        self._inner.reset()


class LiveController:
    def __init__(self, policy, mode: str = "auto", setup: str = "auto", gate: bool = True,
                 gate_cfg: Optional[GateConfig] = None, camera: Optional[CameraModel] = None,
                 log: Optional[SessionLog] = None, goal_tags: Optional[Sequence[str]] = None, units: str = "m"):
        if setup not in SETUPS:
            raise ValueError(f"setup must be one of {', '.join(SETUPS)}")
        if mode != "auto" and mode not in MODES:
            raise ValueError(f"mode must be auto or one of {', '.join(MODES)}")
        self.camera = camera or CameraModel()
        self.setup = setup
        self.use_gate = gate
        self.units = units
        self.log = log
        self.goal_tags: List[str] = list(goal_tags or [])
        self._gate_cfg = gate_cfg or GateConfig()
        if setup == "desk":
            self._gate_cfg.operator_filter = True
        self.desk = setup == "desk"            # a still camera with the user in front of it
        self.activity = ActivityDetector()
        self.adapter = TaskAdapter()
        self.gate: Optional[AnnouncementGate] = None
        self.policy = None
        self.set_policy(policy)
        if mode != "auto":
            self.adapter.force_mode(mode)
        self.last_mode = self.adapter.mode
        self.last_activity = self.activity.activity
        self._t = 0.0
        self.muted = 0

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

    def _after(self, cues, t: float):
        kept = self.adapter.filter_cues(cues)
        self.muted += len(cues or []) - len(kept)
        if kept and self.log is not None:
            self.log.cue(t, kept[0], mode=self.adapter.mode, activity=self.activity.activity.value)
        return kept

    # ---- per frame ------------------------------------------------------------------------------------------
    def on_frame(self, frame, t: float, tracks=None) -> str:
        """Update activity and mode from this frame (and its tracks); returns the mode."""
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
                if self.log is not None:
                    self.log.write("setup", t, setup="desk (detected: camera still after warm-up)")
        mode = self.adapter.step(act, t, has_goal=bool(self.goal_tags))
        if self.log is not None and (mode != self.last_mode or act != self.last_activity):
            self.log.write("mode", t, mode=mode, activity=act.value, reason=self.adapter.reason)
        self.last_mode, self.last_activity = mode, act
        return mode

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

    def status(self) -> dict:
        g = self.gate
        return {"mode": self.adapter.mode + ("*" if self.adapter.override else ""), "activity": self.activity.activity.value,
                "setup": "desk" if self.desk else ("handheld" if self.setup == "handheld" else "auto"),
                "gate": (g.last_reason or "") if g is not None else "off"}

    def report(self) -> List[str]:
        lines = [f"mode: {self.adapter.describe()}; activity: {self.activity.describe()}"]
        if self.gate is not None:
            lines.append(f"gate: {self.gate.describe()}")
        if self.muted:
            lines.append(f"read mode muted {self.muted} object cue(s)")
        return lines
