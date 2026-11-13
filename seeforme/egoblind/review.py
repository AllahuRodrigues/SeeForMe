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
                 verification: str = "human", t: Optional[float] = None) -> Dict:
    verdict = canonical(verdict)
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    if verification not in ("human", "model"):
        raise ValueError("verification must be 'human' or 'model'")
    if window is not None:
        a, b = float(window[0]), float(window[1])
        if not a < b:
            raise ValueError(f"window start must precede its end, got {window}")
        window = [a, b]
    return {"key": key, "verdict": verdict, "window": window, "note": note.strip(), "reviewer": reviewer.strip(),
            "verification": verification, "t": time.time() if t is None else t}


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
