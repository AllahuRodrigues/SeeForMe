"""Shared helpers for the CampusBench scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from _common import CONFIGS, MODELS, ROOT
from seeforme.campus import CampusScenario
from seeforme.campus.oracle import CampusOracle
from seeforme.campus.runner import run_on_campus
from seeforme.campus.scenes import OOD_KINDS
from seeforme.campus.metrics import score
from seeforme.config import PipelineConfig
from seeforme.cues import CACSPolicy, ConfidenceCooldownPolicy, ConfidenceThresholdPolicy
from seeforme.cues.pcacs import PCACSConfig, PCACSPolicy
from seeforme.cues.text_importance import LearnedTextImportance

import os
_TAG = os.environ.get("SEEFORME_CAMPUS", "")           # e.g. SEEFORME_CAMPUS=v2 -> data/campus_v2, results/campus_v2
CAMPUS = ROOT / "data" / (f"campus_{_TAG}" if _TAG else "campus")
RESULTS_C = ROOT / "results" / (f"campus_{_TAG}" if _TAG else "campus")


def load_campus_split(split: str) -> List[CampusScenario]:
    files = sorted((CAMPUS / split).glob("*.json"))
    if not files:
        raise SystemExit(f"no CampusBench scenarios in {CAMPUS / split}; run scripts/make_campus.py")
    scenarios = [CampusScenario.from_json(str(f)) for f in files]
    if split in ("train", "val"):
        ood = sorted({sc.kind for sc in scenarios if sc.kind in OOD_KINDS})
        if ood:     # an explicit raise, so the check survives python -O
            raise AssertionError(f"{CAMPUS / split} contains out-of-distribution kinds {ood}: they are test-only; "
                                 "regenerate with scripts/make_campus.py (without --ood-in-train)")
    return scenarios


def load_campus_cfg() -> PipelineConfig:
    tuned = CONFIGS / "pcacs_tuned.yaml"
    cfg = PipelineConfig.load(str(CONFIGS / "cacs_tuned.yaml")) if (CONFIGS / "cacs_tuned.yaml").exists() else PipelineConfig()
    if tuned.exists():
        raw = json.loads(tuned.read_text()) if tuned.suffix == ".json" else __import__("yaml").safe_load(tuned.read_text())
        from seeforme.cues.pcacs import PCACSWeights
        w = raw.pop("weights", None)
        cfg.pcacs = PCACSConfig(**raw)
        if w:
            cfg.pcacs.weights = PCACSWeights(**w)
    return cfg


def text_model():
    return LearnedTextImportance(str(MODELS / "text_importance.joblib"))


def make_campus_policy(name: str, cfg: PipelineConfig, scenario: CampusScenario, model_dir: Optional[Path] = None):
    """Policies evaluated on CampusBench, including P-CACS ablations (pcacs-<toggle>) and learned schedulers."""
    from dataclasses import replace
    if name == "oracle":
        return CampusOracle(scenario)
    if name == "confidence_threshold":
        return ConfidenceThresholdPolicy(cfg.baseline)
    if name == "confidence_cooldown":
        return ConfidenceCooldownPolicy(cfg.baseline)
    if name == "cacs":
        return CACSPolicy(cfg.cacs, text_model=text_model())
    if name == "pcacs":
        return PCACSPolicy(cfg.pcacs, text_model=text_model(), goal_tags=scenario.goal_tags)
    if name.startswith("pcacs-"):
        flag = name.split("-", 1)[1]
        toggles = {"no_ttc": dict(use_ttc=False), "no_budget": dict(use_budget=False), "no_interrupt": dict(use_interrupt=False),
                   "no_queue": dict(use_queue_cost=False), "no_haptic": dict(use_haptic=False), "no_depth": dict(use_depth=False),
                   "no_slots": dict(use_slots=False), "quiet": dict(verbosity="quiet"), "detailed": dict(verbosity="detailed")}
        if flag.startswith("thr"):                 # pcacs-thr0.5: speak threshold sweep
            toggles[flag] = dict(threshold_speak=float(flag[3:]))
        if flag.startswith("aware"):               # pcacs-aware1.5: awareness weight sweep (w_aware)
            toggles[flag] = dict(weights=replace(cfg.pcacs.weights, w_aware=float(flag[5:])))
        if flag not in toggles:
            raise ValueError(f"unknown P-CACS ablation {flag}")
        return PCACSPolicy(replace(cfg.pcacs, **toggles[flag]), text_model=text_model(), goal_tags=scenario.goal_tags, name=name)
    if name.startswith("pcacs_learned"):
        from seeforme.ml.scheduler_model import load_learned_scheduler
        which = name.split(":", 1)[1] if ":" in name else "il"
        return load_learned_scheduler(which, model_dir or MODELS, cfg, scenario, text_model=text_model())
    raise ValueError(f"unknown policy {name}")


def oracle_scores(scenarios: List[CampusScenario], cfg: PipelineConfig, repeats: int = 1, perfect: bool = False) -> Dict[str, float]:
    out = {}
    for sc in scenarios:
        vals = [score(run_on_campus(CampusOracle(sc), sc, cfg, seed=sc.seed + 17 * r, perfect_perception=perfect), sc) for r in range(repeats)]
        out[sc.name] = sum(vals) / len(vals)
    return out
