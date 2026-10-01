"""Cue-selection metrics from section 5 of the proposal.

Given the cues a policy actually spoke and the scenario's ground-truth important
events, compute:

  important_cue_recall   announced important events / total important events
  cue_precision          useful cues / all cues
  audio_load             spoken cues per minute
  duplicate_rate         fraction of cues that repeat recently given information
  critical_event_delay   seconds from event onset to first announcement (recalled events)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from ..sim.scenario import Scenario
from ..types import Cue


@dataclass
class CueMetrics:
    policy: str
    scenario: str
    n_cues: int
    n_important_events: int
    important_cue_recall: float
    cue_precision: float
    audio_load_per_min: float
    duplicate_rate: float
    mean_critical_delay: Optional[float]
    median_critical_delay: Optional[float]
    n_hedged: int = 0
    n_false_positive_cues: int = 0

    def as_dict(self) -> Dict:
        return asdict(self)


def _cue_source(cue: Cue) -> Optional[int]:
    return cue.source_id


def evaluate_cues(cues: List[Cue], scenario: Scenario, policy_name: str, dup_window: float = 6.0,
                  move_tolerance: float = 0.0) -> CueMetrics:
    events = scenario.important_events()
    objects = {o.obj_id: o for o in scenario.objects}
    cues = sorted(cues, key=lambda c: c.t)

    # --- recall & delay ------------------------------------------------------
    delays = []
    recalled = 0
    for ev in events:
        hits = [c for c in cues if _cue_source(c) == ev["obj_id"] and ev["t_start"] <= c.t < ev["t_end"]]
        if hits:
            recalled += 1
            delays.append(hits[0].t - ev["t_start"])
    recall = recalled / len(events) if events else 1.0

    # --- precision & duplicates --------------------------------------------
    last_spoken: Dict[int, float] = {}
    useful = 0
    dup = 0
    fp = 0
    for c in cues:
        sid = _cue_source(c)
        is_dup = sid is not None and sid in last_spoken and (c.t - last_spoken[sid]) < dup_window
        if sid is None:
            fp += 1
        elif is_dup:
            dup += 1
        elif objects[sid].important_at(c.t):
            useful += 1
        if sid is not None:
            last_spoken[sid] = c.t
    precision = useful / len(cues) if cues else 1.0
    dup_rate = dup / len(cues) if cues else 0.0
    minutes = scenario.duration / 60.0

    med = None
    if delays:
        s = sorted(delays)
        med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])

    return CueMetrics(
        policy=policy_name, scenario=scenario.name, n_cues=len(cues), n_important_events=len(events),
        important_cue_recall=recall, cue_precision=precision, audio_load_per_min=len(cues) / minutes,
        duplicate_rate=dup_rate,
        mean_critical_delay=(sum(delays) / len(delays)) if delays else None,
        median_critical_delay=med, n_hedged=sum(1 for c in cues if c.hedged), n_false_positive_cues=fp,
    )


def aggregate(rows: List[CueMetrics]) -> Dict[str, Dict[str, float]]:
    """Average each metric per policy across scenarios."""
    import statistics as st
    by_policy: Dict[str, List[CueMetrics]] = {}
    for r in rows:
        by_policy.setdefault(r.policy, []).append(r)
    out = {}
    for p, rs in by_policy.items():
        delays = [r.mean_critical_delay for r in rs if r.mean_critical_delay is not None]
        out[p] = {
            "n_scenarios": len(rs),
            "important_cue_recall": st.mean(r.important_cue_recall for r in rs),
            "cue_precision": st.mean(r.cue_precision for r in rs),
            "audio_load_per_min": st.mean(r.audio_load_per_min for r in rs),
            "duplicate_rate": st.mean(r.duplicate_rate for r in rs),
            "mean_critical_delay": st.mean(delays) if delays else float("nan"),
            "recall_sd": st.pstdev(r.important_cue_recall for r in rs),
            "audio_load_sd": st.pstdev(r.audio_load_per_min for r in rs),
        }
    return out
