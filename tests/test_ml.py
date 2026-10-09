import numpy as np
from seeforme.cues.text_importance import KeywordTextImportance
from seeforme.ml.priority_model import TorchMLPPriority, collect_training_data
from seeforme.ml.scene_context import SceneContextModel, frames_from_scenarios
from seeforme.ml.text_classifier import build_model, generate_dataset
from seeforme.sim.scenario import make_scenario


def test_keyword_text_importance():
    k = KeywordTextImportance()
    assert k.score("EMERGENCY EXIT") == 1.0
    assert k.score("Room 204") == 1.0
    assert k.score("SALE 50% OFF") < 0.2
    assert k.score(None) == 0.0


def test_text_classifier_learns_signage():
    X, y = generate_dataset(n_per_class=150, seed=1, extra_csv="/nonexistent.csv")
    m = build_model().fit(X, y)
    assert m.predict_proba(["ex1t"])[0, 1] > 0.5
    assert m.predict_proba(["happy hour 5 to 7"])[0, 1] < 0.5


def test_priority_data_and_torch_mlp():
    scs = [make_scenario("hallway", 1, duration=8), make_scenario("office", 2, duration=8)]
    X, y, meta = collect_training_data(scs)
    assert X.shape[1] == 6 and len(y) == len(X) and y.sum() > 0
    m = TorchMLPPriority(epochs=5, device="cpu").fit(X, y)
    p = m.predict_proba(X[:10])
    assert p.shape == (10, 2) and np.allclose(p.sum(1), 1)


def test_attn_encoder_right_aligns_the_current_step():
    """The current step always sits in the last position embedding, whatever the track's age (as in mlpk)."""
    import torch
    from seeforme.ml.scheduler_model import N_IN, SchedulerNet, _pad
    K, H = 4, 8
    net = SchedulerNet(N_IN, H, "attn", K)
    with torch.no_grad():                      # attention lands on whatever sits in the last slot
        net.embed.weight[0].zero_(); net.embed.bias[0] = 0.0
        net.key.weight.copy_(torch.eye(H)); net.key.bias.zero_()
        net.query.zero_(); net.query[0] = 1.0
        net.pos.zero_(); net.pos[K - 1, 0] = 100.0
    rng = np.random.default_rng(0)
    cur = rng.normal(size=N_IN).tolist()
    older = [rng.normal(size=N_IN).tolist() for _ in range(K - 1)]
    X, L = _pad([[cur], older + [cur]], K)
    with torch.no_grad():
        z = net.encode(X, L)
    assert torch.allclose(z[0], z[1], atol=1e-5)   # a new track and an old one with the same current step agree


def test_scene_context_clusters_and_modulates():
    scs = [make_scenario("hallway", 1, duration=6), make_scenario("street", 2, duration=6)]
    frames = frames_from_scenarios(scs)
    m = SceneContextModel(k=2).fit(frames)
    assert len(m.cluster_names) == 2
    ctx = {"cluster": 0}
    r = m.modulate("car", 0.5, ctx)
    assert 0.0 <= r <= 1.0
