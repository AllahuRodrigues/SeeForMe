"""The live pipeline's policy factory: the learned scheduler loads in RL mode and decides, P-CACS stays P-CACS,
and a missing checkpoint falls back to P-CACS (with the reason attached) only when asked to."""
import math
from pathlib import Path

import pytest

from seeforme.config import PipelineConfig
from seeforme.cues.pcacs import ChannelState, PCACSPolicy
from seeforme.geometry import CameraModel
from seeforme.pipeline import make_live_policy
from seeforme.types import Track

MODELS = Path(__file__).resolve().parents[1] / "models"
RL_CKPT = MODELS / "scheduler" / "scheduler_rl.pt"


def mk(tid, label, x, z, conf=0.9, t=0.0, approach=1.0, depth_conf=0.9):
    cam = CameraModel()
    b = cam.project_box(x, z, 0.9 if label != "person" else 1.7, 0.5)
    tr = Track(tid, label, "object", b, conf, t, t, confirmed=True)
    tr.distance_m, tr.depth_conf, tr.lateral_m = z, depth_conf, x
    tr.rel_speed_mps = -approach
    tr.ttc_s = (z - 0.6) / approach if approach > 0.1 else math.inf
    for i in range(5):
        tr.dist_history.append((t - 2 + 0.4 * i, z + (4 - i) * 0.4 * approach, x))
    tr.path_prob = 0.9 if abs(x) < 0.6 else 0.05
    return tr


def scene(t):
    """A chair closing in on the walking path, a person off to the side, a far bench."""
    z = max(0.8, 5.0 - 1.0 * t)
    return [mk(1, "chair", 0.1, z, t=t), mk(2, "person", 1.8, 7.0, t=t, approach=0.2), mk(3, "bench", -2.0, 9.0, t=t, approach=0.0)]


@pytest.mark.skipif(not RL_CKPT.exists(), reason="no trained scheduler (models/scheduler/scheduler_rl.pt)")
def test_learned_rl_policy_loads_and_decides():
    from seeforme.ml.scheduler_model import SchedulerPolicy
    pol = make_live_policy("pcacs_learned:rl", PipelineConfig(), str(MODELS))
    assert isinstance(pol, SchedulerPolicy) and pol.mode == "rl" and pol.name == "pcacs_learned:rl"
    assert pol.checkpoint == str(RL_CKPT) and not hasattr(pol, "fallback_from")
    actions = []
    for k in range(10):                              # several steps so the GRU sees a multi-step window
        t = 0.5 * k
        busy = k % 3 == 2
        ch = ChannelState(busy=busy, remaining_s=1.0 if busy else 0.0, current_utility=1.0 if busy else 0.0,
                          current_severity=0.3 if busy else 0.0, current_track=9 if busy else None)
        cues = pol.decide(scene(t), t, ch)
        assert isinstance(cues, list) and len(cues) <= 1
        for c in cues:
            assert c.action in ("speak", "interrupt", "haptic") and c.policy == "pcacs_learned:rl"
            assert c.action != "speak" or not busy           # legality mask: speak needs a free channel
            assert c.action != "interrupt" or busy
        actions.append(cues[0].action if cues else "silence")
        assert pol.last_action == actions[-1]
    assert any(a != "silence" for a in actions)       # the approaching chair in the path gets announced
    assert len(pol.last_observations()) >= 1 and len(pol.hist[1]) == pol.K   # chair's window filled to K=8
    assert pol.decide([], 6.0, ChannelState()) == []


def test_pcacs_is_the_linear_rule():
    pol = make_live_policy("pcacs", PipelineConfig(), str(MODELS))
    assert type(pol) is PCACSPolicy and pol.name == "pcacs"
    assert isinstance(pol.decide(scene(1.0), 1.0, ChannelState()), list)


def test_missing_checkpoint_falls_back_to_pcacs(tmp_path):
    with pytest.raises(FileNotFoundError):                 # default: fail loudly (evaluation scripts)
        make_live_policy("pcacs_learned:rl", PipelineConfig(), str(tmp_path))
    pol = make_live_policy("pcacs_learned:rl", PipelineConfig(), str(tmp_path), fallback=True)
    assert type(pol) is PCACSPolicy and pol.name == "pcacs"
    assert pol.fallback_from == "pcacs_learned:rl" and "FileNotFoundError" in pol.fallback_reason
    assert isinstance(pol.decide(scene(1.0), 1.0, ChannelState()), list)


def test_unreadable_checkpoint_falls_back(tmp_path):
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_rl.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        make_live_policy("pcacs_learned:rl", PipelineConfig(), str(tmp_path))
    pol = make_live_policy("pcacs_learned:rl", PipelineConfig(), str(tmp_path), fallback=True)
    assert type(pol) is PCACSPolicy and pol.fallback_from == "pcacs_learned:rl"


def test_unknown_learned_variant_is_an_error_even_with_fallback(tmp_path):
    with pytest.raises(ValueError):
        make_live_policy("pcacs_learned:xyz", PipelineConfig(), str(tmp_path), fallback=True)


def _live_cfg():
    """The tuned config run_live.py actually uses (configs/cacs_tuned.yaml + pcacs_tuned.yaml)."""
    import sys
    sys.path.insert(0, str(MODELS.parent / "scripts"))
    from _common import load_cfg
    return load_cfg()


@pytest.mark.skipif(not RL_CKPT.exists(), reason="no trained scheduler (models/scheduler/scheduler_rl.pt)")
def test_learned_policy_with_the_tuned_live_config():
    cfg = _live_cfg()
    pol = make_live_policy("pcacs_learned:rl", cfg, str(MODELS))
    assert pol.mode == "rl" and pol.cfg is cfg.pcacs
    assert isinstance(pol.decide(scene(1.0), 1.0, ChannelState()), list)


@pytest.mark.skipif(not RL_CKPT.exists(), reason="no trained scheduler (models/scheduler/scheduler_rl.pt)")
def test_wrong_shape_checkpoint_falls_back(tmp_path):
    """A real checkpoint of a different size (here: wrong input width) must fall back, not crash mid-run."""
    import torch
    state = torch.load(RL_CKPT, map_location="cpu", weights_only=False)
    def corrupt(obj):
        if isinstance(obj, torch.Tensor) and obj.dim() == 2:
            return torch.zeros(obj.shape[0], obj.shape[1] + 7)
        if isinstance(obj, dict):
            return {k: corrupt(v) for k, v in obj.items()}
        return obj
    (tmp_path / "scheduler").mkdir()
    torch.save(corrupt(state), tmp_path / "scheduler" / "scheduler_rl.pt")
    pol = make_live_policy("pcacs_learned:rl", PipelineConfig(), str(tmp_path), fallback=True)
    assert type(pol) is PCACSPolicy and pol.fallback_from == "pcacs_learned:rl"


@pytest.mark.skipif(not RL_CKPT.exists(), reason="no trained scheduler (models/scheduler/scheduler_rl.pt)")
def test_imitation_without_report_falls_back(tmp_path):
    import shutil
    (tmp_path / "scheduler").mkdir()
    shutil.copy(MODELS / "scheduler" / "scheduler_il.pt", tmp_path / "scheduler" / "scheduler_il.pt")
    with pytest.raises(FileNotFoundError):
        make_live_policy("pcacs_learned:il", PipelineConfig(), str(tmp_path))
    pol = make_live_policy("pcacs_learned:il", PipelineConfig(), str(tmp_path), fallback=True)
    assert type(pol) is PCACSPolicy and "scheduler_report.json" in pol.fallback_reason
