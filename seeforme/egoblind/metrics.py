"""EgoBlind-specific metrics on top of `campus.metrics.evaluate_run`:

  quiet_violation_rate      quiet windows (user confirmed "no obstacle") in which the policy spoke anyway / quiet windows
  anticipation_lead_s       mean (t_q - first timely cue): how long before the user had to ask was the information given
  coverable_recall          timely events / events the perception stack can see at all (scheduler quality)
  detected_recall           timely events / events the detector actually saw before t_q (isolates scheduling from perception)
  near_question_precision   precision of cues spoken within 6 s of some question (precision elsewhere is unknowable
                            from the questions alone, so plain cue_precision is a lower bound on EgoBlind)
"""
from __future__ import annotations

import statistics as st
from typing import Dict, List, Optional

from ..campus.runner import RunLog
from .scenario import EgoBlindScenario

NEAR_S = 6.0


def egoblind_extras(log: RunLog, sc: EgoBlindScenario) -> Dict[str, Optional[float]]:
    spoken = sorted(log.spoken, key=lambda c: c.t)
    ev = {e.obj_id: e for e in sc.speak_events()}
    first = {}
    for c in spoken:
        if c.source_id in ev and c.source_id not in first:
            first[c.source_id] = c
    timely = {o: c for o, c in first.items() if ev[o].in_window(c.t)}
    leads = [ev[o].detail["t_q"] - c.t for o, c in timely.items()]
    cover = [e for e in ev.values() if e.detail.get("coverable")]
    detected = [e for e in cover if e.detail.get("detected")]
    quiet = sc.quiet_events()
    viol = sum(1 for q in quiet if any(q.earliest_useful_s <= c.t <= q.latest_safe_s for c in spoken))
    tq = sc.question_times
    near = [c for c in spoken if any(abs(c.t - t) <= NEAR_S for t in tq)]
    useful_near = sum(1 for c in near if c.source_id in ev and ev[c.source_id].in_window(c.t) and first.get(c.source_id) is c)
    late_cover = [e for e in cover if e.obj_id not in timely]
    return {"n_quiet": len(quiet), "quiet_violation_rate": (viol / len(quiet)) if quiet else None,
            "anticipation_lead_s": st.mean(leads) if leads else None,
            "n_coverable": len(cover), "n_detected": len(detected),
            "coverable_recall": (sum(1 for e in cover if e.obj_id in timely) / len(cover)) if cover else None,
            "detected_recall": (sum(1 for e in detected if e.obj_id in timely) / len(detected)) if detected else None,
            "perception_miss_rate": (sum(1 for e in cover if e.detail.get("detected") is False) / len(cover)) if cover else None,
            "near_question_precision": (useful_near / len(near)) if near else None, "n_near_question_cues": len(near),
            "n_uncoverable": len(ev) - len(cover), "n_late_coverable": len(late_cover)}


def aggregate_extras(rows: List[Dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = float("nan")
        elif k.startswith("n_"):
            out[k] = float(sum(vals))
        else:
            out[k] = st.mean(vals)
    return out
