"""Blind human review of the weak EgoBlind events (M3, issue #5).

The weak events are derived automatically from user questions and detector labels. A reviewer looks at the frames
around each question and records one verdict:
    speak         the assistant should have spoken about this object in (the possibly corrected) window
    silent        no unprompted cue was needed (the question was idle curiosity, or the answer was not visual)
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

VERDICTS = ("speak", "silent", "wrong_object", "unclear")
MODEL_PREFIXES = ("bin:", "max_score:", "derived:")   # model outputs and the audit answer key stay hidden


def event_key(row: Dict) -> str:
    return f"{int(float(row['video_id']))}:{int(float(row['obj_id']))}"


def blind_row(row: Dict) -> Dict:
    out = {k: v for k, v in row.items() if not k.startswith(MODEL_PREFIXES)}
    out["key"] = event_key(row)
    return out


def make_verdict(key: str, verdict: str, reviewer: str, window: Optional[Tuple[float, float]] = None, note: str = "",
                 verification: str = "human", t: Optional[float] = None) -> Dict:
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
    user-confirmed quiet moments). Rates are over decided verdicts (not 'unclear') per derived class."""
    keys = [k for k in derived if k in verdicts]
    conf: Dict[str, Counter] = {"speak": Counter(), "silent": Counter()}
    for k in keys:
        conf.setdefault(derived[k], Counter())[verdicts[k]] += 1
    def rate(cls, want):
        c = conf.get(cls, Counter()); dec = sum(v for kk, v in c.items() if kk != "unclear")
        return c[want] / dec if dec else None
    return {"n": len(keys), "n_decided": sum(verdicts[k] != "unclear" for k in keys),
            "unclear": sum(verdicts[k] == "unclear" for k in keys),
            "speak_confirmed": rate("speak", "speak"), "silent_confirmed": rate("silent", "silent"),
            "confusion": {k: dict(v) for k, v in conf.items()}}
