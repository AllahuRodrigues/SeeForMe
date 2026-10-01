"""Closed-loop runner for CampusBench: detector + depth simulator -> tracker with dynamics ->
scheduler (speak / interrupt / haptic / silence) -> audio channel. Records everything the
new metrics need: every cue, the channel occupancy log, and per-decision silence records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from ..config import PipelineConfig
from ..cues.pcacs import ChannelState
from ..eval.runner import SpeechChannel
from ..geometry import CameraModel, TrackDynamics
from ..tracking.tracker import IoUTracker
from ..types import Cue, Track
from .simulator import CampusDetectorSimulator
from .world import CampusScenario


@dataclass
class DecisionRecord:
    t: float
    action: str                 # silence | speak | interrupt | haptic | busy
    due_ids: Set[int]           # events in their window, not yet announced, currently tracked
    channel_busy: bool


@dataclass
class RunLog:
    scenario: str
    policy: str
    cues: List[Cue] = field(default_factory=list)            # spoken (speak/interrupt) and haptic cues
    channel_log: List[List] = field(default_factory=list)    # [start, end, cue]
    decisions: List[DecisionRecord] = field(default_factory=list)
    truth_at_cue: Dict[int, Dict] = field(default_factory=dict)   # cue index -> true distance / ttc at cue time

    @property
    def spoken(self) -> List[Cue]:
        return [c for c in self.cues if c.action != "haptic"]


def run_on_campus(policy, scenario: CampusScenario, cfg: Optional[PipelineConfig] = None, seed: Optional[int] = None,
                  on_decision: Optional[Callable] = None, camera: Optional[CameraModel] = None) -> RunLog:
    cfg = cfg or PipelineConfig()
    cam = camera or CameraModel()      # the *assumed* camera; the scenario's real pitch may differ (poor capture)
    tracker = IoUTracker(cfg.tracker, dynamics=TrackDynamics(cam))
    channel = SpeechChannel(cfg.speech.min_gap, cfg.speech.seconds_per_word)
    policy.reset()
    log = RunLog(scenario=scenario.name, policy=policy.name)
    events = {e.obj_id: e for e in scenario.speak_events()}
    announced: Set[int] = set()
    next_decision = 0.0
    supports_actions = getattr(policy, "supports_actions", False)
    for t, dets in CampusDetectorSimulator(scenario, seed=seed).frames():
        tracks = tracker.update(dets, t)
        if t + 1e-9 < next_decision:
            continue
        next_decision = t + policy.decision_interval
        busy = not channel.free(t)
        cur = channel.current if busy else None
        ch = ChannelState(busy=busy, remaining_s=channel.remaining(t),
                          current_utility=(cur.utility if cur else 0.0), current_severity=(cur.severity if cur else 0.0),
                          current_track=(cur.track_id if cur else None))
        if supports_actions:
            cues = policy.decide(tracks, t, ch)
        else:
            cues = policy.select(tracks, t) if not busy else []
        cue = cues[0] if cues else None
        action = "busy" if (busy and cue is None) else (cue.action if cue else "silence")
        if cue is not None:
            if cue.action == "haptic":
                log.cues.append(cue)
            elif cue.action == "interrupt":
                channel.speak(cue, interrupt=True); log.cues.append(cue)
            else:
                channel.speak(cue); log.cues.append(cue)
            if cue.action != "haptic" and cue.source_id is not None:
                announced.add(cue.source_id)
            if cue.source_id is not None:
                obj = next(o for o in scenario.objects if o.obj_id == cue.source_id)
                x, z = obj.rel_pos(t)
                vx, vz = obj.world_vel(t)
                vz_rel = vz - scenario.user_speed(t)
                ttc = (z - 0.6) / (-vz_rel) if vz_rel < -0.1 else None
                log.truth_at_cue[len(log.cues) - 1] = {"z": z, "x": x, "ttc": ttc}
        tracked_ids = {tr.source_id for tr in tracks if tr.source_id is not None}
        due = {oid for oid, e in events.items() if e.in_window(t) and oid not in announced and oid in tracked_ids}
        log.decisions.append(DecisionRecord(t=t, action=action, due_ids=due, channel_busy=busy))
        if on_decision is not None:
            on_decision(t, tracks, policy.last_observations() if hasattr(policy, "last_observations") else [], cue, ch, due)
    log.channel_log = [[a, min(b, scenario.duration), c] for a, b, c in channel.log]
    return log
