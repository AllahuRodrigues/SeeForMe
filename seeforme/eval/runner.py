"""Run a cue-selection policy over a scenario (or a real detection log).

The runner models the audio channel explicitly: a spoken message occupies the
channel for a duration proportional to its word count, and no policy can speak
while the channel is busy. All policies share the same channel model so audio
load differences come from the policy, not from the runner.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..config import PipelineConfig
from ..cues.policy import CuePolicy
from ..sim.scenario import DetectorSimulator, Scenario
from ..tracking.tracker import IoUTracker
from ..types import Cue, Detection, Track


class SpeechChannel:
    """One message at a time. `log` records (start, end, cue) so occupancy and blocking can be measured;
    an interrupt truncates the current entry."""

    def __init__(self, min_gap: float = 1.2, seconds_per_word: float = 0.35):
        self.min_gap = min_gap
        self.spw = seconds_per_word
        self.busy_until = -1e9
        self.current: Optional[Cue] = None
        self.log: List[List] = []

    def free(self, t: float) -> bool:
        return t >= self.busy_until

    def remaining(self, t: float) -> float:
        return max(0.0, self.busy_until - t)

    def duration(self, cue: Cue) -> float:
        n_words = max(1, len(cue.message.split()))
        return max(self.min_gap, n_words * self.spw)

    def speak(self, cue: Cue, interrupt: bool = False) -> None:
        if interrupt and self.log and not self.free(cue.t):
            self.log[-1][1] = cue.t            # the running message is cut off
        d = self.duration(cue)
        cue.duration_s = d
        self.busy_until = cue.t + d
        self.current = cue
        self.log.append([cue.t, cue.t + d, cue])

    def reset(self) -> None:
        self.busy_until = -1e9
        self.current = None
        self.log = []


def run_policy_on_frames(policy: CuePolicy, frames: Iterable[Tuple[float, List[Detection]]],
                         cfg: Optional[PipelineConfig] = None, scene_context=None,
                         on_step: Optional[Callable[[float, List[Track], List[Cue]], None]] = None) -> List[Cue]:
    cfg = cfg or PipelineConfig()
    tracker = IoUTracker(cfg.tracker)
    channel = SpeechChannel(cfg.speech.min_gap, cfg.speech.seconds_per_word)
    policy.reset()
    spoken: List[Cue] = []
    next_decision = 0.0
    for t, dets in frames:
        tracks = tracker.update(dets, t)
        ctx = scene_context.observe(tracks, t) if scene_context is not None else None
        cues: List[Cue] = []
        if t + 1e-9 >= next_decision:
            next_decision = t + policy.decision_interval
            if channel.free(t):
                cues = policy.select(tracks, t, ctx)
                for c in cues[:1]:   # one message at a time on the audio channel
                    channel.speak(c)
                    spoken.append(c)
                cues = cues[:1]
        if on_step is not None:
            on_step(t, tracks, cues)
    return spoken


def run_policy_on_scenario(policy: CuePolicy, scenario: Scenario, cfg: Optional[PipelineConfig] = None,
                           scene_context=None, seed: Optional[int] = None) -> List[Cue]:
    sim = DetectorSimulator(scenario, seed=seed)
    return run_policy_on_frames(policy, sim.frames(), cfg, scene_context)
