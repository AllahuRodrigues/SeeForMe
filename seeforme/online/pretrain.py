"""Warm start (online-to-batch): pretrain the online learner on CampusBench, where the simulator knows every label.

Layer 1 of the data plan. In simulation every candidate at every decision is labelled: +1 inside its event's window
before the first announcement, -1 otherwise. The learner then deploys on real video with one-sided feedback only.
CampusBench is generated data, so a model pretrained on it stays on the commercially clean licence track.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Sequence, Set

import numpy as np

from ..campus.runner import run_on_campus
from ..config import PipelineConfig
from .policy import N_FEAT, BudgetController, OnlinePolicy, _Normalizer


class _Announced:
    def __init__(self) -> None:
        self.announced: Set[int] = set()


class _SimLabels:
    """Scenario view for full-information labelling: off-event candidates are always known negatives in simulation."""

    def __init__(self, scenario) -> None:
        self._sc = scenario

    def speak_events(self):
        return self._sc.speak_events()

    def counts_fp(self, t: float) -> bool:
        return True


def pretrain_on_campus(policy: OnlinePolicy, scenarios: Sequence, cfg: Optional[PipelineConfig] = None, epochs: int = 1,
                       seed: int = 0, carry: str = "all") -> Dict[str, int]:
    """Full-information passes over simulated scenarios; restores the policy's mode and gives it a fresh budget.

    carry="all" keeps everything the simulator taught (weights, AdaGrad accumulator, posterior precision, feature
    normaliser). carry="weights" keeps only the weights as a prior mean: step sizes and posterior variance start fresh
    and real features are normalised on real data, so the learner can move away from a simulator that is wrong."""
    if carry not in ("all", "weights"):
        raise ValueError(f"carry must be 'all' or 'weights', got {carry!r}")
    mode, ctl = policy.mode, policy.ctl
    n0, p0 = policy.n_labels, policy.n_pos
    policy.mode = "full_info"
    videos = 0
    try:
        for ep in range(epochs):
            for k, sc in enumerate(scenarios):
                ann = _Announced()
                policy.begin_video(_SimLabels(sc), ann)

                def hook(t, tracks, obs, cue, ch, due, _ann=ann):
                    if cue is not None and cue.action != "haptic" and cue.source_id is not None:
                        _ann.announced.add(cue.source_id)

                run_on_campus(policy, sc, cfg, seed=seed + 1000 * ep + k, on_decision=hook)
                videos += 1
    finally:
        policy.mode = mode
        policy.ctl = BudgetController.make(ctl.target_per_min, policy.tau0, ctl.eta)   # deployment budget starts fresh
        policy.begin_video(None, None)
        if carry == "weights":
            st = policy.state
            policy.state = replace(st, g2=np.full_like(st.g2, 1e-8), prec=np.ones_like(st.prec))
            policy.norm = _Normalizer(N_FEAT)
    return {"videos": videos, "labels": policy.n_labels - n0, "positives": policy.n_pos - p0}

