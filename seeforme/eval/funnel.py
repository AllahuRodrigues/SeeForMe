"""Event-failure funnel: where along the pipeline does each should-speak event get lost?

For every event with a useful window [earliest, latest], the replay records the times at which the pipeline held
evidence linked to that event:
    det         a raw detection was linked to the event
    track       a confirmed tracker track carried it
    cand        the scheduler saw it as a candidate (passed the policy's confidence filter)
    cand_legal  ... and speaking about it was allowed at that decision (channel free, not just announced)
    cue         a cue about it was spoken
The event is attributed to the first stage that fails inside the window:
    perception  never detected in the window             -> better detector / segmentation / open vocabulary
    tracking    detected but never a confirmed track     -> tracker confirmation, flicker
    candidate   tracked but never a scheduler candidate  -> candidate filter, confidence threshold
    scheduler   a candidate but never spoken about       -> relevance model, threshold, budget (blocked = never legal)
    timing      spoken about, but only outside the window-> anticipation, latency
    delivered   spoken about inside the window
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

STAGES = ("perception", "tracking", "candidate", "scheduler", "timing", "delivered")


@dataclass
class EventTrace:
    window: Tuple[float, float]
    det: List[float] = field(default_factory=list)
    track: List[float] = field(default_factory=list)
    cand: List[float] = field(default_factory=list)
    cand_legal: List[float] = field(default_factory=list)
    cue: List[float] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    def _inw(self, ts: Sequence[float]) -> bool:
        a, b = self.window
        return any(a - 1e-6 <= t <= b + 1e-6 for t in ts)

    @property
    def blocked(self) -> bool:
        """A candidate in the window, but speaking about it was never allowed there."""
        return self._inw(self.cand) and not self._inw(self.cand_legal)


def classify(t: EventTrace) -> str:
    if not t._inw(t.det):
        return "perception"
    if not t._inw(t.track):
        return "tracking"
    if not t._inw(t.cand):
        return "candidate"
    if t._inw(t.cue):
        return "delivered"
    if t.cue:
        return "timing"
    return "scheduler"


def funnel_counts(traces: Sequence[EventTrace]) -> Dict[str, int]:
    """N events; P reached detection, T tracking, C candidate, S were spoken about at all, D on time."""
    stages = [classify(t) for t in traces]
    lost_by = {s: stages.count(s) for s in STAGES}
    n = len(traces)
    p = n - lost_by["perception"]
    tt = p - lost_by["tracking"]
    c = tt - lost_by["candidate"]
    s = c - lost_by["scheduler"]
    return {"N": n, "P": p, "T": tt, "C": c, "S": s, "D": lost_by["delivered"], **{f"lost_{k}": v for k, v in lost_by.items()},
            "blocked": sum(1 for t, st in zip(traces, stages) if st == "scheduler" and t.blocked)}
