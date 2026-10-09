"""CampusBench metrics: communication quality under a bandwidth constraint.

  action_weighted_recall   sum(severity of events announced inside their window) / sum(severity)
  important_cue_recall     unweighted version
  late_rate                events announced after latest_safe or never, among should-speak events
  late_critical_rate       the same restricted to severity >= 0.7 (sudden events reported separately)
  mean_lead_s              mean (latest_safe - first announcement) over timely events: how early the warning came
  cue_precision            spoken cues inside the window of a should-speak event (first time) / spoken cues
  audio_occupancy          fraction of the scenario during which the policy's speech was playing (external speech excluded)
  cues_per_min, haptics_per_min
  silence_precision        decision steps where the policy stayed silent with a free channel and nothing was due
                           / silent-with-free-channel steps
  critical_blocking_rate   critical events that were late while the channel was busy with a less severe message
  min_message_ratio        Jaccard overlap between spoken slots and the annotated minimum information
  distance_mae_m, ttc_mae_s   error of the distance / TTC inside spoken cues vs the true value
  regret                   oracle communication score - policy score (per minute; see score())
"""
from __future__ import annotations

import statistics as st
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from ..types import Cue
from .runner import RunLog
from .world import CampusScenario


@dataclass
class CampusMetrics:
    policy: str
    scenario: str
    kind: str
    n_events: int
    n_critical: int
    n_spoken: int
    n_haptic: int
    important_cue_recall: float
    action_weighted_recall: float
    late_rate: float
    late_critical_rate: float
    late_critical_rate_nonsudden: float
    mean_lead_s: Optional[float]
    cue_precision: float
    audio_occupancy: float
    cues_per_min: float
    haptics_per_min: float
    silence_precision: float
    critical_blocking_rate: float
    min_message_ratio: Optional[float]
    distance_mae_m: Optional[float]
    ttc_mae_s: Optional[float]
    score: float
    duplicate_rate: float
    n_interrupts: int

    def as_dict(self) -> Dict:
        return asdict(self)


def _is_external(cue) -> bool:
    """Speech the policy did not choose (runner.distractor_cue: a sign read on request, a conversation)."""
    return getattr(cue, "policy", None) == "external"


def _policy_airtime(log: RunLog) -> float:
    """Seconds of channel time taken by the policy's own messages. External speech is not charged to the policy
    (and cutting it short with an interrupt therefore earns nothing); it still counts for critical_blocking_rate."""
    return sum(b - a for a, b, c in log.channel_log if not _is_external(c))


def _timeliness(t: float, e) -> float:
    """1 inside [earliest, ideal], decaying linearly to 0.5 at latest_safe, 0 after."""
    if t < e.earliest_useful_s or t > e.latest_safe_s:
        return 0.0
    if t <= e.ideal_s:
        return 1.0
    span = max(1e-6, e.latest_safe_s - e.ideal_s)
    return 1.0 - 0.5 * (t - e.ideal_s) / span


def score(log: RunLog, sc: CampusScenario, c_fp: float = 0.3, c_occ: float = 0.15, c_late: float = 0.5) -> float:
    """Communication score: timely severity-weighted warnings minus off-window cues, late penalties and occupancy."""
    ev = {e.obj_id: e for e in sc.speak_events()}
    first: Dict[int, float] = {}
    fp = 0
    # a scenario whose ground truth is only known in places (EgoBlind: around the users' questions) says where an
    # off-event cue can be judged a false alarm; CampusBench knows everything and counts them all
    counts_fp = getattr(sc, "counts_fp", None)
    for c in log.spoken:
        if c.source_id in ev:
            if c.source_id not in first:
                first[c.source_id] = c.t
                if not ev[c.source_id].in_window(c.t):
                    fp += 1
            else:
                fp += 1
        else:
            fp += 1 if counts_fp is None else int(counts_fp(c.t))
    s = 0.0
    for oid, e in ev.items():
        if oid in first:
            tl = _timeliness(first[oid], e)
            s += e.severity * tl
            if tl == 0.0 and first[oid] > e.latest_safe_s:
                s -= c_late * e.severity
        else:
            s -= c_late * e.severity
    occ = _policy_airtime(log)
    return s - c_fp * fp - c_occ * occ


def evaluate_run(log: RunLog, sc: CampusScenario, oracle_score: Optional[float] = None, dup_window: float = 6.0) -> CampusMetrics:
    events = sc.speak_events()
    ev = {e.obj_id: e for e in events}
    spoken = sorted(log.spoken, key=lambda c: c.t)
    haptic = [c for c in log.cues if c.action == "haptic"]
    first: Dict[int, Cue] = {}
    for c in spoken:
        if c.source_id in ev and c.source_id not in first:
            first[c.source_id] = c
    timely = {oid: c for oid, c in first.items() if ev[oid].in_window(c.t)}
    # recall
    sev_sum = sum(e.severity for e in events)
    awr = sum(ev[o].severity for o in timely) / sev_sum if sev_sum else 1.0
    recall = len(timely) / len(events) if events else 1.0
    # lateness
    late = [e for e in events if e.obj_id not in timely and (e.obj_id not in first or first[e.obj_id].t > e.latest_safe_s)]
    crit = [e for e in events if e.severity >= 0.7]
    crit_late = [e for e in late if e.severity >= 0.7]
    crit_ns = [e for e in crit if not e.sudden]
    crit_late_ns = [e for e in crit_late if not e.sudden]
    leads = [ev[o].latest_safe_s - c.t for o, c in timely.items()]
    # precision & duplicates
    useful, dup, last_spoken = 0, 0, {}
    for c in spoken:
        sid = c.source_id
        if sid is not None and sid in last_spoken and c.t - last_spoken[sid] < dup_window:
            dup += 1
        elif sid in ev and ev[sid].in_window(c.t) and first.get(sid) is c:
            useful += 1
        if sid is not None:
            last_spoken[sid] = c.t
    precision = useful / len(spoken) if spoken else 1.0
    # occupancy (the policy's own speech; external speech is left out)
    occ = _policy_airtime(log) / sc.duration
    minutes = sc.duration / 60.0
    # silence precision
    silent_free = [d for d in log.decisions if d.action == "silence" and not d.channel_busy]
    sp = sum(1 for d in silent_free if not d.due_ids) / len(silent_free) if silent_free else 1.0
    # blocking: critical late events whose deadline passed while a less severe message occupied the channel
    blocked = 0
    for e in crit_late:
        for a, b, c in log.channel_log:
            if a <= e.latest_safe_s < b and c.severity < e.severity and c.source_id != e.obj_id:
                blocked += 1; break
    # minimum message ratio
    jac = []
    for oid, c in timely.items():
        req = set(ev[oid].minimum_information); got = set(c.slots) or {"object"}
        jac.append(len(req & got) / len(req | got))
    # distance / ttc error
    d_err, t_err = [], []
    for i, c in enumerate(log.cues):
        tr = log.truth_at_cue.get(i)
        if tr is None or c.action == "haptic":
            continue
        if c.distance_m is not None:
            d_err.append(abs(c.distance_m - tr["z"]))
        if c.ttc_s is not None and tr["ttc"] is not None:
            t_err.append(abs(c.ttc_s - tr["ttc"]))
    s = score(log, sc)
    return CampusMetrics(
        policy=log.policy, scenario=sc.name, kind=sc.kind, n_events=len(events), n_critical=len(crit),
        n_spoken=len(spoken), n_haptic=len(haptic), important_cue_recall=recall, action_weighted_recall=awr,
        late_rate=len(late) / len(events) if events else 0.0,
        late_critical_rate=len(crit_late) / len(crit) if crit else 0.0,
        late_critical_rate_nonsudden=len(crit_late_ns) / len(crit_ns) if crit_ns else 0.0,
        mean_lead_s=st.mean(leads) if leads else None, cue_precision=precision, audio_occupancy=occ,
        cues_per_min=len(spoken) / minutes, haptics_per_min=len(haptic) / minutes, silence_precision=sp,
        critical_blocking_rate=blocked / len(crit) if crit else 0.0,
        min_message_ratio=st.mean(jac) if jac else None,
        distance_mae_m=st.mean(d_err) if d_err else None, ttc_mae_s=st.mean(t_err) if t_err else None,
        score=s, duplicate_rate=dup / len(spoken) if spoken else 0.0,
        n_interrupts=sum(1 for c in spoken if c.action == "interrupt"),
    )


def aggregate(rows: List[CampusMetrics], oracle_scores: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    by: Dict[str, List[CampusMetrics]] = {}
    for r in rows:
        by.setdefault(r.policy, []).append(r)
    for p, rs in by.items():
        def mean_of(key, skip_none=True):
            vals = [getattr(r, key) for r in rs if getattr(r, key) is not None]
            return st.mean(vals) if vals else float("nan")
        d = {k: mean_of(k) for k in ["important_cue_recall", "action_weighted_recall", "late_rate", "late_critical_rate",
                                     "late_critical_rate_nonsudden", "mean_lead_s", "cue_precision", "audio_occupancy",
                                     "cues_per_min", "haptics_per_min", "silence_precision", "critical_blocking_rate",
                                     "min_message_ratio", "distance_mae_m", "ttc_mae_s", "score", "duplicate_rate", "n_interrupts"]}
        d["n_runs"] = len(rs)
        d["recall_sd"] = st.pstdev([r.action_weighted_recall for r in rs])
        if oracle_scores:
            regrets = [oracle_scores[r.scenario] - r.score for r in rs if r.scenario in oracle_scores]
            d["regret"] = st.mean(regrets) if regrets else float("nan")
        out[p] = d
    return out
