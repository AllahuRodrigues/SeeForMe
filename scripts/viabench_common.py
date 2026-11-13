"""Shared loading for the VIABench reminder videos (data/viabench/, git-ignored; MIT-tagged annotations, public creator videos)."""
import json
import random
from collections import defaultdict
from typing import Dict, List

from _common import ROOT
from seeforme.egoblind.cache import PerceptionCache
from seeforme.egoblind.scenario import EgoBlindScenario
from seeforme.viabench.data import SPLITS, clip_reminders, events_from_reminders, parse_reminders, split_of

VIA = ROOT / "data" / "viabench"
CACHE = VIA / "cache"
ANNOTATIONS = VIA / "annotations" / "proactive_reminder.json"
ID0 = 100000                      # VIABench ids start here so they can never collide with EgoBlind clip ids
DEV_SPLITS = SPLITS[:2]           # via_test is a lockbox: never cached or loaded by default


def load_index() -> List[Dict]:
    if not ANNOTATIONS.exists():
        raise SystemExit(f"{ANNOTATIONS} missing: hf download MCG-NJU/VIABench --repo-type dataset --local-dir data/viabench")
    parsed = parse_reminders(json.loads(ANNOTATIONS.read_text()))
    return [{"key": k, "id": ID0 + i, "split": split_of(k), "path": VIA / k, "source": k.split("/")[2], **parsed[k]}
            for i, k in enumerate(sorted(parsed))]


def caching_order(index: List[Dict], seed: int = 0) -> List[Dict]:
    """Round-robin over the source platforms, shuffled inside each, so every prefix is a balanced sample."""
    by = defaultdict(list)
    for v in index:
        by[v["source"]].append(v)
    rng = random.Random(seed)
    for s in by:
        rng.shuffle(by[s])
    out, k = [], 0
    while any(by.values()):
        for s in sorted(by):
            if k < len(by[s]):
                out.append(by[s][k])
        k += 1
        if k > max(len(x) for x in by.values()):
            break
    return out


def load_viabench_split(split: str, allow_lockbox: bool = False) -> List[EgoBlindScenario]:
    """Scenarios for the cached videos of one split. Reminders past the cached part of a video are not events."""
    if split not in SPLITS:
        raise SystemExit(f"split must be one of {SPLITS}")
    if split == "via_test" and not allow_lockbox:
        raise SystemExit("via_test is a lockbox: it is only read for a final, pre-registered evaluation")
    out = []
    for v in load_index():
        cp = CACHE / f"{v['id']}.npz"
        if v["split"] != split or not cp.exists():
            continue
        cache = PerceptionCache.load(cp)
        rem = clip_reminders(v["reminders"], cache.duration)
        sc = EgoBlindScenario(name=f"viabench_{v['id']}", video_id=v["id"], split=split, kind="viabench_reminder",
                              duration=float(cache.duration), fps=float(cache.meta.get("eff_fps", 5.0)), seed=v["id"], video=str(v["path"]),
                              goal_tags=[], events_list=events_from_reminders(rem), queries=[], stats={}, cache_path=str(cp),
                              description=f"VIABench {v['uid']} ({split}), {len(rem)} reminders in the cached part")
        sc.attach_cache(cache)
        sc.reminders = rem
        out.append(sc)
    return out
