"""Baseline cue-selection policies from section 4.5 of the proposal."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config import BaselineConfig
from ..types import Cue, Track
from .messages import compose_message
from .policy import CuePolicy


class ConfidenceThresholdPolicy(CuePolicy):
    """Baseline 1: speak every detection whose confidence exceeds a fixed value.

    Because only one message fits on the audio channel at a time, "speak
    everything" is implemented as a queue: whenever the queue is empty it is
    refilled with every currently visible track above the threshold (highest
    confidence first) and one item is spoken per decision step.
    """

    name = "confidence_threshold"

    def __init__(self, cfg: Optional[BaselineConfig] = None):
        self.cfg = cfg or BaselineConfig()
        self.decision_interval = self.cfg.decision_interval
        self.queue: List[int] = []

    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        by_id = {tr.track_id: tr for tr in tracks if tr.confidence >= self.cfg.confidence_threshold}
        self.queue = [tid for tid in self.queue if tid in by_id]   # drop vanished objects
        if not self.queue:
            self.queue = [tr.track_id for tr in sorted(by_id.values(), key=lambda tr: -tr.confidence)]
        cues = []
        while self.queue and len(cues) < self.cfg.max_cues_per_decision:
            tr = by_id[self.queue.pop(0)]
            tr.last_announced = t
            tr.announce_count += 1
            cues.append(Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind,
                            message=compose_message(tr), score=tr.confidence,
                            policy=self.name, text=tr.text, source_id=tr.source_id))
        return cues

    def reset(self) -> None:
        self.queue.clear()


class ConfidenceCooldownPolicy(CuePolicy):
    """Baseline 2: confident detections, but identical labels are suppressed for a fixed period."""

    name = "confidence_cooldown"

    def __init__(self, cfg: Optional[BaselineConfig] = None):
        self.cfg = cfg or BaselineConfig()
        self.decision_interval = self.cfg.decision_interval
        self.last_spoken: Dict[str, float] = {}

    def _key(self, tr: Track) -> str:
        return f"text:{(tr.text or '').lower()}" if tr.kind == "text" else tr.label

    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        cands = []
        for tr in tracks:
            if tr.confidence < self.cfg.confidence_threshold:
                continue
            last = self.last_spoken.get(self._key(tr))
            if last is not None and t - last < self.cfg.cooldown:
                continue
            cands.append(tr)
        cands.sort(key=lambda tr: -tr.confidence)
        cues = []
        for tr in cands[: self.cfg.max_cues_per_decision]:
            self.last_spoken[self._key(tr)] = t
            tr.last_announced = t
            tr.announce_count += 1
            cues.append(Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind,
                            message=compose_message(tr), score=tr.confidence,
                            policy=self.name, text=tr.text, source_id=tr.source_id))
        return cues

    def reset(self) -> None:
        self.last_spoken.clear()
