"""Shared helpers for the EgoBlind scripts (paths, splits, policies)."""
from __future__ import annotations
import os

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from _common import MODELS, ROOT
from campus_common import make_campus_policy, text_model
from seeforme.config import PipelineConfig
from seeforme.egoblind import EgoBlindScenario, Question, group_by_video, load_questions, refine_with_cache

EGO = ROOT / "data" / "egoblind"
META = EGO / "meta"
VIDEOS = EGO / "videos"
CACHE = Path(os.environ["SEEFORME_EGOBLIND_CACHE"]) if os.environ.get("SEEFORME_EGOBLIND_CACHE") else EGO / "cache"
SCENARIOS = EGO / "scenarios"
RESULTS_E = ROOT / "results" / "egoblind"
MODELS_E = MODELS / "egoblind"
# EgoBlind ships train and test. Its official *train* clips are split 70 / 15 / 5 / 10 by source recording
# (data/egoblind/meta/video_source.csv maps clip-id ranges to the original video they were cut from), so no recording
# has clips in two splits:
#   train     70%  imitation learning (DAgger)
#   finetune  15%  reward fine-tuning (REINFORCE) - the "20% development" part used to change weights ...
#   select     5%  ... and the 5% used only to choose checkpoints, thresholds and the RL silence bias
#   holdout   10%  unseen test: never used for training, tuning or selection
# EgoBlind's official `test` split stays untouched as a second, external held-out benchmark.
# Recordings are assigned whole, in a fixed sha256 order of their ids: holdout first, then select, then finetune, each
# until it holds its share of the clips; the rest is train.
SPLIT_SHARES = {"holdout": 0.10, "select": 0.05, "finetune": 0.15}
SPLITS = ("train", "finetune", "select", "holdout", "test")


def csv_for(split: str) -> Path:
    name = "test.csv" if split == "test" else "train.csv"
    for p in (META / name, ROOT / "external" / "EgoBlind" / name, ROOT / "external" / "EgoBlind" / "finetune" / name):
        if p.exists():
            return p
    raise SystemExit(f"{name} not found; run scripts/download_egoblind.py --meta-only (or clone external/EgoBlind)")


def source_recordings() -> Dict[int, str]:
    """clip id -> id of the source recording it was cut from (the modal_id of its row in video_source.csv). Rows that share
    a clip id (overlapping ranges such as 665-667 / 667-669) or a modal_id are merged into one recording (union-find,
    keyed by the smallest id), so one recording can never straddle two splits. Clips without a row are absent."""
    import csv, re
    path = META / "video_source.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing: run scripts/download_egoblind.py --meta-only (it is needed for the leakage-free split)")
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: Dict[int, str] = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf8")):
        m = re.search(r"modal_id=(\d+)", r["video_url"])
        src = m.group(1) if m else r["video_url"].strip()
        find(src)
        for part in r["video_name"].split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                ids = range(int(a), int(b) + 1)
            else:
                ids = [int(part)] if part.isdigit() else []
            for i in ids:
                if i in first:
                    ra, rb = find(first[i]), find(src)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                else:
                    first[i] = src
    return {i: find(s) for i, s in first.items()}


_ASSIGN: Dict[int, str] = {}


def train_pool_assignment() -> Dict[int, str]:
    """clip id -> 'train' | 'finetune' | 'select' | 'holdout' | 'excluded' for every clip of EgoBlind's official train
    split (whole source recordings per split, see SPLIT_SHARES).

    Two kinds of clip are kept out of the evaluated splits: clips with no video_source.csv row cannot be placed by
    recording, and several continue a neighbouring recording (87 ~ 86 backyard, 367/368 ~ 366 horse ride, 443 ~ 444
    waiter), so they go to train; and clips cut from a recording that also has clips in EgoBlind's official test
    (794-797 share modal_id 7299034538745892105 with test 1480-1495) are 'excluded' from every split, so the official
    test stays disjoint from training by recording."""
    if _ASSIGN:
        return _ASSIGN
    import hashlib
    clips = sorted(group_by_video(load_questions(csv_for("train"), "train")))
    test_clips = set(group_by_video(load_questions(csv_for("test"), "test")))
    src = source_recordings()
    test_recordings = {src[v] for v in test_clips if v in src}
    groups: Dict[str, List[int]] = {}
    for v in clips:
        if v not in src:
            _ASSIGN[v] = "train"
        elif src[v] in test_recordings:
            _ASSIGN[v] = "excluded"
        else:
            groups.setdefault(src[v], []).append(v)
    order = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    filled = {k: 0 for k in SPLIT_SHARES}
    for g in order:
        split = next((k for k, share in SPLIT_SHARES.items() if filled[k] < share * len(clips)), "train")
        if split != "train":
            filled[split] += len(groups[g])
        for v in groups[g]:
            _ASSIGN[v] = split
    return _ASSIGN


def load_egoblind_questions(split: str) -> Dict[int, List[Question]]:
    """video_id -> questions for a split: EgoBlind's official `test`, or the train / finetune / select / holdout parts of
    its official train split (grouped by source recording, see SPLIT_SHARES)."""
    if split not in SPLITS:
        raise SystemExit(f"unknown EgoBlind split {split!r}; expected one of {SPLITS}")
    base = "test" if split == "test" else "train"
    qs = load_questions(csv_for(base), base)
    by = group_by_video(qs)
    if split != "test":
        assign = train_pool_assignment()
        by = {v: q for v, q in by.items() if assign.get(v) == split}
    for q in (q for v in by.values() for q in v):
        q.split = split
    return by


_VIDEO_INDEX: Dict[int, Path] = {}


def video_path(video_id: int) -> Optional[Path]:
    """EgoBlind ships videos as zero-padded names (00093.mp4); accept any width."""
    p = _VIDEO_INDEX.get(video_id)
    if p is not None and p.exists():
        return p
    for sub in ("train", "test"):
        for cand in (VIDEOS / sub / f"{video_id}.mp4", VIDEOS / sub / f"{video_id:05d}.mp4", VIDEOS / sub / f"{video_id:04d}.mp4"):
            if cand.exists():
                _VIDEO_INDEX[video_id] = cand
                return cand
    return None


def cache_path(video_id: int) -> Path:
    return CACHE / f"{video_id}.npz"


def load_egoblind_split(split: str, require_cache: bool = True, limit: int = 0, kinds: Optional[Sequence[str]] = None,
                        refresh: bool = True) -> List[EgoBlindScenario]:
    files = sorted((SCENARIOS / split).glob("*.json"), key=lambda p: int(p.stem))
    if split != "test":   # a scenario file left over from an older split must never be evaluated
        assign = train_pool_assignment()
        files = [f for f in files if assign.get(int(f.stem)) == split]
    if not files:
        raise SystemExit(f"no EgoBlind scenarios in {SCENARIOS / split}; run scripts/build_egoblind_scenarios.py")
    out: List[EgoBlindScenario] = []
    for f in files:
        sc = EgoBlindScenario.from_json(str(f), cache_dir=CACHE)
        if require_cache and not sc.has_cache():
            continue
        if kinds and sc.kind not in kinds:
            continue
        if refresh and sc.has_cache():
            c = sc.cache()
            if sc.stats.get("refined_from") != c.meta.get("built"):
                refine_with_cache(sc, c)
                sc.stats["refined_from"] = c.meta.get("built")
        out.append(sc)
        if limit and len(out) >= limit:
            break
    return out


def make_egoblind_policy(name: str, cfg: PipelineConfig, scenario, model_dir: Optional[Path] = None):
    """`egoblind_learned:il|rl` loads the scheduler trained with scripts/train_scheduler_egoblind.py; everything else
    (oracle, baselines, cacs, pcacs, pcacs-<ablation>, pcacs_learned:il|rl) is shared with CampusBench."""
    if name == "pcacs_egoblind":   # linear P-CACS tuned on EgoBlind `train` replays (make egoblind-tune); the pre-audit config is in results/egoblind/exploratory_2026-10-08/
        import yaml
        from seeforme.cues.pcacs import PCACSConfig, PCACSPolicy, PCACSWeights
        path = ROOT / "configs" / "pcacs_egoblind.yaml"
        if not path.exists():
            raise FileNotFoundError("configs/pcacs_egoblind.yaml missing: run scripts/tune_pcacs.py --domain egoblind")
        raw = yaml.safe_load(path.read_text()); w = raw.pop("weights", None)
        pc = PCACSConfig(**raw)
        if w:
            pc.weights = PCACSWeights(**w)
        return PCACSPolicy(pc, text_model=text_model(), goal_tags=scenario.goal_tags, name=name)
    if name.startswith("egoblind_learned"):      # egoblind_learned:rl  or  egoblind_learned:il@models/egoblind_only
        from seeforme.ml.scheduler_model import load_learned_scheduler
        spec = name.split(":", 1)[1] if ":" in name else "rl"
        which, _, at = spec.partition("@")
        return load_learned_scheduler(which, Path(at) if at else (model_dir or MODELS_E), cfg, scenario, text_model=text_model(), name=name)
    return make_campus_policy(name, cfg, scenario)


def parse_noise(spec: str):
    """'p_miss=0.1,depth_rel=0.1' -> EgoNoise (None for an empty spec)."""
    from seeforme.egoblind import EgoNoise
    if not spec:
        return None
    kw = {}
    for item in spec.split(","):
        k, v = item.split("=")
        kw[k.strip()] = float(v)
    return EgoNoise(**kw)
