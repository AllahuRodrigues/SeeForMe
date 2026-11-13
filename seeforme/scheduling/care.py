"""CARE scheduler: expected-utility choice with silence as an explicit action, under a strict speech ledger.

    U(e) = value * p_helpful - cost_wrong * p_wrong - cost_repeat * p_redundant - cost_speech_per_s * duration
Silence has utility 0. Eligible events: not expired, not already announced, grounded in current evidence, and payable
from the ledger. The learned model only supplies the probabilities; this rule and the ledger are deterministic and
testable. If |U_hat - U| <= eps for every feasible action, the chosen action is within 2 eps of the best one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..events.schema import CommunicationDecision, SceneEvent
from .speech_ledger import SpeechLedger


@dataclass(frozen=True)
class CareWeights:
    cost_wrong: float = 2.0
    cost_repeat: float = 1.0
    cost_speech_per_s: float = 0.1


def utility(e: SceneEvent, w: CareWeights) -> float:
    return e.value * e.p_helpful - w.cost_wrong * e.p_wrong - w.cost_repeat * e.p_redundant - w.cost_speech_per_s * e.duration_s


def choose_action(events: Sequence[SceneEvent], t: float, ledger: SpeechLedger, busy: bool, w: CareWeights) -> CommunicationDecision:
    if busy:
        return CommunicationDecision(action="SILENCE", reason_code="channel_busy")
    skipped, best, best_u = {}, None, 0.0
    for e in events:
        if e.expired or (e.deadline_t is not None and t > e.deadline_t):
            skipped[e.event_id] = "expired"; continue
        if e.announced:
            skipped[e.event_id] = "already_announced"; continue
        if not e.grounded:
            skipped[e.event_id] = "ungrounded"; continue
        if not ledger.can_spend(e.duration_s, t):
            skipped[e.event_id] = "budget"; continue
        u = utility(e, w)
        if u > best_u:
            best, best_u = e, u
    if best is None:
        return CommunicationDecision(action="SILENCE", skipped=skipped,
                                     reason_code="no_eligible_event" if len(skipped) == len(events) and events else "no_positive_utility")
    ledger.try_spend(best.duration_s, t)
    return CommunicationDecision(action="SPEAK", reason_code="max_utility", event_id=best.event_id, utility=best_u, text=best.text, skipped=skipped)
