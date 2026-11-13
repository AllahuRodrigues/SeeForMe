"""Onset-timing metrics in the units used by EgoVoice (EMNLP 2026) and ProactEgoMem (arXiv 2610.30089).

Each event has a reference time `ref` and a source id; cues are (time, source id). Events and cues about the same
source are matched one-to-one (Hungarian, cost |delta|) when the signed offset delta = t_cue - ref lies in
[-early, +late]. Then:
    start@tau        share of *matched* events with |delta| <= tau (ProactEgoMem's Start@tau)
    start@tau_all    share of *all* events with a match within tau (cannot be gamed by matching few events)
    mean_abs_delta, median_abs_delta, sd_abs_delta, mean_signed_delta (negative = early)
    recall           matched / events;  precision = matched / n_scored_cues (cues whose usefulness is knowable)
Decisions are made every 0.5 s, so tau = 0.2 is mostly quantisation; report 0.5, 1 and 2 as primary.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TimedEvent:
    eid: Hashable
    sid: Hashable
    ref: float


def match_onsets(events: Sequence[TimedEvent], cues: Sequence[Tuple[float, Optional[Hashable]]], early: float, late: float) -> Dict[Hashable, float]:
    """Event id -> signed offset of its matched cue."""
    cues = [(t, s) for t, s in cues if s is not None]
    if not events or not cues:
        return {}
    big = 1e9
    cost = np.full((len(events), len(cues)), big)
    for i, e in enumerate(events):
        for j, (t, s) in enumerate(cues):
            d = t - e.ref
            if s == e.sid and -early - 1e-9 <= d <= late + 1e-9:
                cost[i, j] = abs(d)
    rows, cols = linear_sum_assignment(cost)
    return {events[i].eid: cues[j][0] - events[i].ref for i, j in zip(rows, cols) if cost[i, j] < big}


def timing_metrics(events: Sequence[TimedEvent], cues: Sequence[Tuple[float, Optional[Hashable]]], early: float, late: float,
                   taus: Sequence[float] = (0.2, 0.5, 1.0, 2.0), n_scored_cues: Optional[int] = None) -> Dict:
    m = match_onsets(events, cues, early, late)
    d = list(m.values())
    n_ev = len(events)
    r: Dict = {"events": n_ev, "matched": len(d), "recall": (len(d) / n_ev) if n_ev else None}
    for tau in taus:
        key = f"start@{tau:g}"
        hits = sum(abs(x) <= tau + 1e-9 for x in d)
        r[key] = hits / len(d) if d else None
        r[key + "_all"] = hits / n_ev if n_ev else None
    a = [abs(x) for x in d]
    r["mean_abs_delta"] = statistics.mean(a) if a else None
    r["median_abs_delta"] = statistics.median(a) if a else None
    r["sd_abs_delta"] = statistics.pstdev(a) if len(a) > 1 else None
    r["mean_signed_delta"] = statistics.mean(d) if d else None
    if n_scored_cues:
        p = len(d) / n_scored_cues
        r["precision"] = p
        rc = r["recall"] or 0.0
        r["f1"] = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
    return r
