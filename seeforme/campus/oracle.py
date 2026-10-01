"""Oracle scheduler for CampusBench: knows the annotation, speaks every event's ideal cue at its
ideal time (or as soon after as the channel allows), interrupts for critical events that would
otherwise be late. Gives the upper bound for communication regret and the labels for imitation."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..cues.compose import speech_seconds
from ..cues.pcacs import ChannelState
from ..cues.policy import CuePolicy
from ..types import Cue, Track
from .world import CampusScenario, EventAnnotation


class CampusOracle(CuePolicy):
    name = "oracle"
    supports_actions = True
    decision_interval = 0.5

    def __init__(self, scenario: CampusScenario, lead: float = 0.25):
        self.events: Dict[int, EventAnnotation] = {e.obj_id: e for e in scenario.speak_events()}
        self.lead = lead
        self.done: set = set()
        self._last_obs: List = []

    def reset(self) -> None:
        self.done = set()

    def last_observations(self):
        return self._last_obs

    def label(self, track: Track, t: float, ch: ChannelState, whole_window: bool = False) -> str:
        """Oracle action for one candidate. The oracle itself acts at the ideal time; as an imitation
        target (`whole_window`) every step of the useful window is a positive."""
        e = self.events.get(track.source_id) if track.source_id is not None else None
        if e is None or track.source_id in self.done:
            return "silence"
        start = e.earliest_useful_s if whole_window else e.ideal_s - self.lead
        if t < start or t > e.latest_safe_s:
            return "silence"
        if not ch.busy:
            return "speak"
        if e.interrupt_current_speech and ch.current_severity < e.severity and t >= e.latest_safe_s - 1.0:
            return "interrupt"
        return "silence"

    def decide(self, tracks: List[Track], t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List[Cue]:
        ch = ch or ChannelState()
        cands = []
        for tr in tracks:
            a = self.label(tr, t, ch)
            if a != "silence":
                e = self.events[tr.source_id]
                cands.append((e.severity, -e.latest_safe_s, a, tr, e))
        if not cands:
            return []
        cands.sort(key=lambda c: (-c[0], -c[1]))
        _, _, action, tr, e = cands[0]
        self.done.add(tr.source_id)
        tr.last_announced = t; tr.announce_count += 1
        return [Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind, message=e.ideal_cue, score=e.severity,
                    policy=self.name, text=tr.text, source_id=tr.source_id, action=action,
                    duration_s=speech_seconds(e.ideal_cue), slots=tuple(e.minimum_information), severity=e.severity,
                    utility=10.0 * e.severity, distance_m=tr.distance_m,
                    interrupted_track=ch.current_track if action == "interrupt" else None)]

    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        return self.decide(tracks, t, ChannelState(), scene_ctx)
