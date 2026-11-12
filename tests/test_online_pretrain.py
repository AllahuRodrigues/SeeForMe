"""M1: warm start (online-to-batch) from CampusBench, where the simulator labels every candidate."""
from pathlib import Path

import numpy as np

from seeforme.campus.world import CampusScenario
from seeforme.online.policy import OnlinePolicy
from seeforme.online.pretrain import pretrain_on_campus

ROOT = Path(__file__).resolve().parents[1]


def _campus(n=3):
    files = sorted((ROOT / "data" / "campus_v2" / "train").glob("*.json"))[:n]
    return [CampusScenario.from_json(str(f)) for f in files]


def test_pretraining_learns_and_restores_mode_and_budget():
    pol = OnlinePolicy(mode="thompson", tau0=0.6)
    w0, tau0 = pol.state.w.copy(), pol.tau
    stats = pretrain_on_campus(pol, _campus())
    assert stats["labels"] > 0 and stats["positives"] > 0 and stats["videos"] == 3
    assert pol.mode == "thompson" and not np.allclose(pol.state.w, w0)
    assert abs(pol.tau - tau0) < 1e-9                                  # the deployment budget starts fresh


def test_pretraining_is_deterministic():
    a, b = OnlinePolicy(mode="greedy"), OnlinePolicy(mode="greedy")
    pretrain_on_campus(a, _campus(2)); pretrain_on_campus(b, _campus(2))
    assert np.allclose(a.state.w, b.state.w)


def test_weights_only_carry_resets_optimizer_posterior_and_normalizer():
    full, wonly = OnlinePolicy(mode="thompson"), OnlinePolicy(mode="thompson")
    pretrain_on_campus(full, _campus(2))
    pretrain_on_campus(wonly, _campus(2), carry="weights")
    assert np.allclose(full.state.w, wonly.state.w)                     # same prior mean
    assert np.allclose(wonly.state.g2, 1e-8) and np.allclose(wonly.state.prec, 1.0)
    assert wonly.norm.n == 0 and full.norm.n > 0                        # real features are normalised on real data
    assert full.state.g2.max() > 1e-3
