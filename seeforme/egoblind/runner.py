"""Closed-loop replay of a scheduler over a cached EgoBlind video: identical channel semantics to
`campus.runner.run_on_campus` (one message at a time, interrupts cut the running message, haptics
bypass the channel), with recorded detections instead of the simulator."""
from __future__ import annotations

import random
from typing import Callable, Optional, Set

from ..campus.runner import DecisionRecord, RunLog, play_distractors, run_on_campus
from ..config import PipelineConfig
from ..cues.pcacs import ChannelState
from ..eval.runner import SpeechChannel
from ..geometry import CameraModel, TrackDynamics
from ..tracking.tracker import IoUTracker
from .cache import EgoNoise
from .scenario import EgoBlindScenario


def run_on_egoblind(policy, scenario: EgoBlindScenario, cfg: Optional[PipelineConfig] = None, seed: Optional[int] = None,
                    on_decision: Optional[Callable] = None, noise: Optional[EgoNoise] = None, camera: Optional[CameraModel] = None,
                    distractors=None) -> RunLog:
    cfg = cfg or PipelineConfig()
    cam = camera or scenario.camera
    tracker = IoUTracker(cfg.tracker, dynamics=TrackDynamics(cam))
    channel = SpeechChannel(cfg.speech.min_gap, cfg.speech.seconds_per_word)
    policy.reset()
    log = RunLog(scenario=scenario.name, policy=policy.name)
    events = {e.obj_id: e for e in scenario.speak_events()}
    announced: Set[int] = set()
    next_decision = 0.0
    supports_actions = getattr(policy, "supports_actions", False)
    rng = random.Random(0 if seed is None else seed) if noise is not None else None
    distractors = sorted(distractors or [])
    di = 0
    for t, dets in scenario.frames(noise=noise, rng=rng):
        tracks = tracker.update(dets, t)
        di = play_distractors(distractors, di, t, channel, log, cfg.speech)
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
        tracked_ids = {tr.source_id for tr in tracks if tr.source_id is not None}
        due = {oid for oid, e in events.items() if e.in_window(t) and oid not in announced and oid in tracked_ids}
        log.decisions.append(DecisionRecord(t=t, action=action, due_ids=due, channel_busy=busy))
        if on_decision is not None:
            on_decision(t, tracks, policy.last_observations() if hasattr(policy, "last_observations") else [], cue, ch, due)
    log.channel_log = [[a, min(b, scenario.duration), c] for a, b, c in channel.log]
    return log


def make_runner(noise: Optional[EgoNoise] = None) -> Callable:
    """A `run(policy, scenario, cfg, seed=, on_decision=)` that dispatches CampusBench and EgoBlind scenarios,
    so the scheduler trainers can mix both domains."""
    def run(policy, scenario, cfg=None, seed=None, on_decision=None, **kw):
        if isinstance(scenario, EgoBlindScenario):
            return run_on_egoblind(policy, scenario, cfg, seed=seed, on_decision=on_decision, noise=noise)
        return run_on_campus(policy, scenario, cfg, seed=seed, on_decision=on_decision, **kw)
    return run
