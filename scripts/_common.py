"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seeforme.config import PipelineConfig  # noqa: E402
from seeforme.sim.scenario import Scenario, make_benchmark  # noqa: E402

DATA = ROOT / "data" / "scenarios"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CONFIGS = ROOT / "configs"


def load_split(split: str):
    d = DATA / split
    files = sorted(d.glob("*.json"))
    if not files:
        raise SystemExit(f"no scenarios in {d}; run scripts/make_scenarios.py first")
    return [Scenario.from_json(str(f)) for f in files]


def load_cfg(path: str | None = None) -> PipelineConfig:
    if path:
        return PipelineConfig.load(path)
    tuned = CONFIGS / "cacs_tuned.yaml"
    cfg = PipelineConfig.load(str(tuned)) if tuned.exists() else PipelineConfig()
    ptuned = CONFIGS / "pcacs_tuned.yaml"
    if ptuned.exists():
        import yaml
        from seeforme.cues.pcacs import PCACSConfig, PCACSWeights
        raw = yaml.safe_load(ptuned.read_text())
        w = raw.pop("weights", None)
        cfg.pcacs = PCACSConfig(**raw)
        if w:
            cfg.pcacs.weights = PCACSWeights(**w)
    return cfg


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
