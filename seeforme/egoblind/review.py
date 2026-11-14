"""Blind human review of the weak EgoBlind events (M3, issue #5).

The weak events are derived automatically from user questions and detector labels. A reviewer looks at the frames
around each question and records one verdict. Rubric v3 (2026-11-13) separates what to say from whether to say it
unprompted: proactive_warn, proactive_clear, on_request (useful only if asked), silent, uncertain (evidence
insufficient; never folded into silent), wrong_object. Rubric v1/v2 verdicts map onto it (`canonical`). Legacy notes:
    speak         warn: the assistant should have told the user about an object in (the possibly corrected) window
    confirm_clear no object needed a warning, but confirming that the way is clear would have helped
    silent        nothing needed (idle curiosity, or the answer was not visual)
    wrong_object  a cue was needed, but about something the derived labels do not name (note says what)
    unclear       the frames do not settle it
Reviews are blind: every model column (bin:*, max_score:*) is removed before display. Verdicts are append-only
records with provenance; only `verification == "human"` rows count as ground truth. Model pre-labels may be stored
with `verification == "model"` and are never reported as human labels.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VERDICTS = ("proactive_warn", "proactive_clear", "on_request", "silent", "uncertain", "wrong_object")
LEGACY = {"speak": "proactive_warn", "confirm_clear": "proactive_clear", "unclear": "uncertain"}
WARN_LIKE = {"proactive_warn", "wrong_object"}
# Rubric v4 (2026-11-13) records two labels per decision moment: whether an unprompted announcement would help
# (USEFULNESS), and whether the statement the assistant would make is true of what was visible (CLAIMS). 'wrong_object'
# mixed the two; it now reads as useful to announce, with a wrong claim.
USEFULNESS = ("proactive_warn", "proactive_clear", "on_request", "silent", "uncertain")
CLAIMS = ("correct", "partially_correct", "wrong", "not_assessable")
EVIDENCE = ("yes", "no", "uncertain")
CONFIDENCE = ("high", "medium", "low")


def canonical(verdict: str) -> str:
    """Map rubric v1 / v2 verdicts onto rubric v3."""
    return LEGACY.get(verdict, verdict)


def decision_time_frames(times, t_q: float):
    """Indices of review frames available at decision time (no frame after the question)."""
    return [i for i, t in enumerate(times) if t <= t_q + 1e-6]
MODEL_PREFIXES = ("bin:", "max_score:", "derived:")   # model outputs and the audit answer key stay hidden


def event_key(row: Dict) -> str:
    return f"{int(float(row['video_id']))}:{int(float(row['obj_id']))}"


def blind_row(row: Dict) -> Dict:
    out = {k: v for k, v in row.items() if not k.startswith(MODEL_PREFIXES)}
    out["key"] = event_key(row)
    return out


def make_verdict(key: str, verdict: str, reviewer: str, window: Optional[Tuple[float, float]] = None, note: str = "",
                 verification: str = "human", t: Optional[float] = None, claim: Optional[str] = None,
                 evidence: Optional[str] = None, confidence: Optional[str] = None) -> Dict:
    verdict = canonical(verdict)
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    for name, val, allowed in (("claim", claim, CLAIMS), ("evidence", evidence, EVIDENCE), ("confidence", confidence, CONFIDENCE)):
        if val not in (None, "") and val not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    if verification not in ("human", "model"):
        raise ValueError("verification must be 'human' or 'model'")
    if window is not None:
        a, b = float(window[0]), float(window[1])
        if not a < b:
            raise ValueError(f"window start must precede its end, got {window}")
        window = [a, b]
    rec = {"key": key, "verdict": verdict, "window": window, "note": note.strip(), "reviewer": reviewer.strip(),
           "verification": verification, "t": time.time() if t is None else t}
    for name, val in (("claim", claim), ("evidence", evidence), ("confidence", confidence)):
        if val:
            rec[name] = val
    return rec


def two_axis(record: Dict) -> Tuple[str, Optional[str]]:
    """(usefulness, claim correctness) of one verdict record, for any rubric version."""
    v = canonical(record["verdict"])
    if v == "wrong_object":
        return "proactive_warn", "wrong"
    return v, record.get("claim") or None


def proactive_target(usefulness: str) -> Optional[int]:
    """Training target for 'speak unprompted now': 1, 0, or None when the reviewer could not tell (never a negative)."""
    if usefulness in ("proactive_warn", "proactive_clear"):
        return 1
    if usefulness in ("on_request", "silent"):
        return 0
    return None


def consensus_targets(a: Dict[str, Dict], b: Dict[str, Dict]) -> Dict[str, Dict]:
    """Per event: the target two reviewers agree on. Disagreement, an unsure reviewer or a single review give None."""
    out: Dict[str, Dict] = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out[k] = {"target": None, "status": "single"}; continue
        ta, tb = proactive_target(two_axis(a[k])[0]), proactive_target(two_axis(b[k])[0])
        if ta is None or tb is None:
            out[k] = {"target": None, "status": "uncertain"}
        elif ta != tb:
            out[k] = {"target": None, "status": "disagree"}
        else:
            out[k] = {"target": ta, "status": "agree"}
    return out


def latest_verdicts(records: Iterable[Dict], reviewer: Optional[str] = None, human_only: bool = True) -> Dict[str, Dict]:
    """Last record per event (optionally for one reviewer); later records overwrite earlier ones."""
    out: Dict[str, Dict] = {}
    for r in records:
        if human_only and r.get("verification") != "human":
            continue
        if reviewer is not None and r.get("reviewer") != reviewer:
            continue
        out[r["key"]] = r
    return out


def rater_kind(records: Iterable[Dict], reviewer: str) -> Optional[str]:
    """'human' or 'model' for a reviewer id, None if it has no records. One id is never both."""
    kinds = {r.get("verification", "human") for r in records if r.get("reviewer") == reviewer}
    if len(kinds) > 1:
        raise ValueError(f"reviewer {reviewer!r} has both human and model records")
    return next(iter(kinds), None)


def model_verdict_records(labels: Sequence[Dict], events: Sequence[Dict], reviewer: str, existing: Iterable[Dict] = ()) -> List[Dict]:
    """Validate a model's pre-labels against a rubric v4 queue and return store records with verification 'model'.
    `events` are blind rows (sample_id, key, claim). The reviewer id must not belong to a human."""
    if rater_kind(existing, reviewer) == "human":
        raise ValueError(f"reviewer {reviewer!r} already has human verdicts; model labels need their own id")
    by_id = {e["sample_id"]: e for e in events}
    out, seen = [], set()
    for lab in labels:
        sid = lab.get("sample_id")
        if sid not in by_id:
            raise ValueError(f"unknown sample id {sid!r}")
        if sid in seen:
            raise ValueError(f"sample {sid} is labelled twice")
        seen.add(sid)
        use, claim = lab.get("usefulness"), lab.get("claim") or None
        if use not in USEFULNESS:
            raise ValueError(f"{sid}: usefulness must be one of {USEFULNESS}, got {use!r}")
        has_sentence = bool((by_id[sid].get("claim") or "").strip())
        if has_sentence and claim is None:
            raise ValueError(f"{sid} has a proposed sentence: the claim label is required")
        if not has_sentence and claim is not None:
            raise ValueError(f"{sid} has no sentence to judge, but a claim label was given")
        out.append(make_verdict(by_id[sid]["key"], use, reviewer, note=lab.get("note", "") or "", verification="model", claim=claim,
                                evidence=lab.get("evidence") or None, confidence=lab.get("confidence") or None))
    return out


def disagreements(a: Dict[str, Dict], b: Dict[str, Dict]) -> List[Dict]:
    """Events both raters labelled where they differ, with what differs: the speak-or-not decision, one rater unsure,
    the usefulness category, or the truth of the sentence."""
    out = []
    for k in sorted(set(a) & set(b)):
        (ua, ca), (ub, cb) = two_axis(a[k]), two_axis(b[k])
        ta, tb = proactive_target(ua), proactive_target(ub)
        differs = []
        if ta is not None and tb is not None and ta != tb:
            differs.append("speak_or_not")
        if (ta is None) != (tb is None):
            differs.append("one_unsure")
        if ua != ub:
            differs.append("category")
        if ca and cb and ca != cb:
            differs.append("claim")
        if differs:
            out.append({"key": k, "differs": differs, "a": ua, "b": ub, "claim_a": ca, "claim_b": cb,
                        "note_a": a[k].get("note", ""), "note_b": b[k].get("note", "")})
    return out


def packet_frames(times: Sequence[float], t_q: float, n: int = 4, gap: int = 2) -> List[int]:
    """Frame indices for a contact sheet: the decision-time frame last, earlier ones `gap` steps apart, none after t_q."""
    idx = decision_time_frames(times, t_q)
    while gap > 1 and (n - 1) * gap > len(idx) - 1:
        gap -= 1
    return idx[::-1][::gap][:n][::-1]


def merge_verdicts(rows: List[Dict], verdicts: Dict[str, Dict]) -> List[Dict]:
    for row in rows:
        v = verdicts.get(event_key(row))
        if v is not None:
            row["human_verdict"] = v["verdict"]
            note = v.get("note", "")
            if v.get("window"):
                note = (note + f" [window {v['window'][0]:.1f}-{v['window'][1]:.1f}]").strip()
            row["human_note"] = note
    return rows


def cohen_kappa(a: Dict[str, str], b: Dict[str, str]) -> Optional[float]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    n = len(keys)
    po = sum(a[k] == b[k] for k in keys) / n
    ca, cb = Counter(a[k] for k in keys), Counter(b[k] for k in keys)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def _kappa_block(a: Dict[str, str], b: Dict[str, str], n_boot: int, seed: int) -> Dict:
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "raw": None, "kappa": None, "kappa_ci": None}
    k = cohen_kappa(a, b)
    rng, ks = random.Random(seed), []
    for _ in range(n_boot):                                      # percentile bootstrap over events
        pick = [rng.choice(keys) for _ in keys]
        ks.append(cohen_kappa({str(i): a[x] for i, x in enumerate(pick)}, {str(i): b[x] for i, x in enumerate(pick)}))
    ks.sort()
    ci = [ks[int(0.025 * len(ks))], ks[max(0, int(0.975 * len(ks)) - 1)]] if ks else None
    return {"n": len(keys), "raw": sum(a[x] == b[x] for x in keys) / len(keys), "kappa": k, "kappa_ci": ci}


def agreement(a: Dict[str, Dict], b: Dict[str, Dict], n_boot: int = 2000, seed: int = 0) -> Dict:
    """Agreement of two reviewers on each axis. The binary block (speak unprompted or not) drops every event that
    either reviewer marked uncertain and says how many; it never counts them as silent."""
    keys = sorted(set(a) & set(b))
    ua, ub = {k: two_axis(a[k])[0] for k in keys}, {k: two_axis(b[k])[0] for k in keys}
    decided = [k for k in keys if proactive_target(ua[k]) is not None and proactive_target(ub[k]) is not None]
    ca = {k: two_axis(a[k])[1] for k in keys if two_axis(a[k])[1] and two_axis(b[k])[1]}
    cb = {k: two_axis(b[k])[1] for k in ca}
    binary = _kappa_block({k: str(proactive_target(ua[k])) for k in decided}, {k: str(proactive_target(ub[k])) for k in decided}, n_boot, seed)
    binary["excluded_uncertain"] = len(keys) - len(decided)
    cons = consensus_targets(a, b)
    return {"n_overlap": len(keys), "usefulness": _kappa_block(ua, ub, n_boot, seed), "binary": binary,
            "claim": _kappa_block(ca, cb, n_boot, seed),
            "consensus": dict(Counter(("target_%d" % c["target"]) if c["target"] is not None else c["status"] for k, c in cons.items() if k in keys))}


def sample_second_rater(keys: Sequence[str], share: float = 0.2, seed: int = 0) -> List[str]:
    keys = sorted(keys)
    k = max(1, round(share * len(keys)))
    return sorted(random.Random(seed).sample(keys, k))


def audit_score(derived: Dict[str, str], verdicts: Dict[str, str]) -> Dict:
    """Compare a reviewer's verdicts with the derived labels on a mixed audit set (should-speak events plus
    user-confirmed quiet moments). 'uncertain' is excluded, never counted as silent. A should-speak event is confirmed
    by a proactive warning; a quiet moment is consistent unless it was marked as a warning."""
    keys = [k for k in derived if k in verdicts]
    v = {k: canonical(verdicts[k]) for k in keys}
    conf: Dict[str, Counter] = {"speak": Counter(), "silent": Counter()}
    for k in keys:
        conf.setdefault(derived[k], Counter())[v[k]] += 1

    def dec(cls):
        c = conf.get(cls, Counter()); return c, sum(n for kk, n in c.items() if kk != "uncertain")
    sc, sn = dec("speak"); qc, qn = dec("silent")
    return {"n": len(keys), "n_decided": sum(x != "uncertain" for x in v.values()), "uncertain": sum(x == "uncertain" for x in v.values()),
            "speak_confirmed": (sc["proactive_warn"] / sn) if sn else None,
            "silent_confirmed": ((qn - sum(qc[w] for w in WARN_LIKE)) / qn) if qn else None,
            "confirm_clear_on_quiet": (qc["proactive_clear"] / qn) if qn else None,
            "on_request_share": (sum(c["on_request"] for c in conf.values()) / max(1, sn + qn)),
            "confusion": {k: dict(c) for k, c in conf.items()}}
