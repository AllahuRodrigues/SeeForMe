"""Stratified queue of decision moments for two independent reviewers (issue #5, rubric v4).

The weak EgoBlind labels say when a blind user asked something; they do not say whether an unprompted announcement would
have helped. This queue samples decision moments from the development splits only, across the event types the detector
misses (doors, stairs, poles), whole-scene questions, other detectable events, moments the user confirmed as clear, and
moments nobody asked about at all (which training has so far treated as silent without evidence).

Rules: `train` and `finetune` only; seeded and independent of pool order; at most `max_per_group` items per source
recording; shortages are filled from the other strata and reported, never hidden; the final order is shuffled so the
position of an item does not reveal its stratum.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

DEV_SPLITS = ("train", "finetune")
QUOTAS = {"door": 24, "stairs": 24, "pole": 16, "scene": 28, "covered_other": 28, "quiet": 24, "unasked": 16}
FILL_ORDER = ("covered_other", "scene", "quiet", "unasked", "pole", "door", "stairs")

_DOOR = ("door", "entrance", "exit", "gate", "elevator", "doorway")
_STAIRS = ("stair", "step", "escalator", "ramp", "curb", "kerb")
_POLE = ("pole", "railing", "fence", "bollard", "pillar", "post", "handrail", "barrier")


def stratum_of(label: str, concepts: Iterable[str] = (), should_speak: bool = True, reason: str = "") -> str:
    """Stratum of a weak event. The object is often named only in the concepts (label 'unknown')."""
    if not should_speak:
        return "quiet"
    words = " ".join([label or ""] + [c or "" for c in concepts]).lower()
    for name, keys in (("door", _DOOR), ("stairs", _STAIRS), ("pole", _POLE)):
        if any(k in words for k in keys):
            return name
    return "scene" if (label or "unknown").lower() in ("unknown", "none", "") else "covered_other"


def build_queue(pool: Sequence[Dict], quotas: Dict[str, int] = QUOTAS, seed: int = 0, max_per_group: int = 2) -> Tuple[List[Dict], Dict]:
    """Pick `sum(quotas)` items. Each pool item needs key, stratum, group (source recording) and source_split."""
    bad = sorted({p["source_split"] for p in pool} - set(DEV_SPLITS))
    if bad:
        raise ValueError(f"the label queue may only draw on {DEV_SPLITS}; got {bad}")
    rng = random.Random(seed)
    by: Dict[str, List[Dict]] = {s: [] for s in quotas}
    for p in sorted(pool, key=lambda p: p["key"]):               # order of the pool must not matter
        if p["stratum"] in by:
            by[p["stratum"]].append(p)
    for s in by:
        rng.shuffle(by[s])
    used, picked = Counter(), []
    counts, filled = Counter(), Counter()

    def take(s: str, n: int, bucket: Counter) -> int:
        got = 0
        while got < n and by[s]:
            p = by[s].pop()
            if used[p["group"]] >= max_per_group:
                continue
            used[p["group"]] += 1
            picked.append(dict(p)); bucket[s] += 1; got += 1
        return got

    short = {s: q - take(s, q, counts) for s, q in quotas.items()}
    need = sum(short.values())
    while need > 0:                                               # round-robin over the strata that still have items
        progressed = False
        for s in FILL_ORDER:
            if need > 0 and s in by and take(s, 1, filled):
                need -= 1; progressed = True
        if not progressed:
            break
    rng.shuffle(picked)
    for i, p in enumerate(picked):
        p["sample_id"] = f"SFM-{i + 1:03d}"
    report = {"requested": dict(quotas), "picked": {s: counts[s] for s in quotas}, "shortfall": short,
              "filled": dict(filled), "total": len(picked), "unfilled": need, "seed": seed, "max_per_group": max_per_group,
              "final": dict(Counter(p["stratum"] for p in picked)), "by_split": dict(Counter(p["source_split"] for p in picked))}
    return picked, report
