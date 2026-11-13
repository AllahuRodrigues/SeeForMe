"""CARE in the replay / live loop: the frozen usefulness scorer supplies p_helpful, CARE decides under a strict ledger.

Candidate legality (busy channel, recent announcement) is shared with the threshold policy, so experiment E2 compares
the decision rule alone: same candidates, same scores. Heads not yet trained (claim reliability, redundancy) are 0.
"""
from __future__ import annotations

from typing import List, Optional

from ..cues.pcacs import ChannelState
from ..events.schema import SceneEvent
from ..online.decision_models import DecisionModel, FrozenScorerPolicy
from .care import CareWeights, choose_action
from .speech_ledger import SpeechLedger


class CarePolicy(FrozenScorerPolicy):
    def __init__(self, model: DecisionModel, capacity_s: float = 6.0, refill_per_s: float = 4.0 / 60.0,
                 weights: Optional[CareWeights] = None, name: Optional[str] = None, **kw):
        super().__init__(model, budget=False, name=name or "care", **kw)
        self.ledger = SpeechLedger(capacity_s, refill_per_s)
        self.weights = weights or CareWeights()
        self.last_decision = None
        self._last_t = None

    def reset(self) -> None:
        super().reset()
        if self._last_t is not None:              # the ledger runs on stream time, which restarts in every video
            self.ledger.reset()
        self._last_t = None

    def decide(self, tracks, t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List:
        ch = ch or ChannelState()
        self._last_t = t
        obs, p, legal = self._scored(tracks, t, ch)
        events = [SceneEvent(event_id=str(obs[i].track.track_id), t=t, category=obs[i].track.label, track_ids=(obs[i].track.track_id,),
                             p_helpful=float(p[i]), duration_s=float(getattr(obs[i], "duration", 1.5)))
                  for i in legal]
        d = choose_action(events, t, self.ledger, busy=ch.busy, w=self.weights)
        self.last_decision = d
        if d.action != "SPEAK":
            return []
        i = next(j for j in legal if str(obs[j].track.track_id) == d.event_id)
        self.last_action = "speak"
        self.last_reasons[obs[i].track.track_id] = "spoken"
        return [self._make_cue(obs[i], t, "speak", ch)]
