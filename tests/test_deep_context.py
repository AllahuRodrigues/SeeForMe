import numpy as np
from seeforme.config import PipelineConfig
from seeforme.cues.cacs import CACSPolicy, LearnedCACS
from seeforme.ml.onpolicy import OraclePolicy, collect, dagger, rollout, select_threshold
from seeforme.ml.sequence_model import SequenceCACS, SequenceModel, build_sequences, train_reinforce
from seeforme.sim.scenario import make_scenario
from sklearn.linear_model import LogisticRegression


def _scenes():
    return [make_scenario("hallway", 1, duration=8), make_scenario("office", 2, duration=8)]


def test_rollout_matches_runner_semantics():
    from seeforme.eval import run_policy_on_scenario
    cfg = PipelineConfig(); sc = _scenes()[0]
    a = rollout(CACSPolicy(cfg.cacs), sc, cfg, seed=5)
    b = run_policy_on_scenario(CACSPolicy(cfg.cacs), sc, cfg, seed=5)
    assert [(c.t, c.track_id) for c in a] == [(c.t, c.track_id) for c in b]


def test_collect_has_13_features_and_both_classes():
    cfg = PipelineConfig()
    d = collect(lambda sc: OraclePolicy(sc, cfg.cacs, seed=1), _scenes(), cfg, repeats=1)
    assert d.X.shape[1] == 13 and len(d) > 50
    assert 0 < d.y.mean() < 1
    assert d.X[:, :6].min() >= -1e-6           # CACS features stay in [0, 1]


def test_dagger_returns_model_and_threshold():
    cfg = PipelineConfig(); sc = _scenes()
    d = collect(lambda sc_: OraclePolicy(sc_, cfg.cacs, seed=1), sc, cfg)
    best, hist = dagger(lambda: LogisticRegression(max_iter=500, class_weight="balanced"),
                        lambda m, thr, s: LearnedCACS(m, cfg.cacs, threshold=thr), d, sc, sc, cfg,
                        n_iter=1, thr_grid=(0.3, 0.6), n_features=6)
    assert best["threshold"] in (0.3, 0.6) and len(hist) == 2
    assert hist[1]["n_train"] > hist[0]["n_train"]


def test_sequence_model_fit_predict_and_policy():
    cfg = PipelineConfig(); sc = _scenes()
    d = collect(lambda s: OraclePolicy(s, cfg.cacs, seed=2), sc, cfg)
    X, L, y = build_sequences(d, K=4)
    assert X.shape == (len(d), 4, 13) and L.max() <= 4
    m = SequenceModel(K=4, epochs=2).fit(d)
    pol = SequenceCACS(m, cfg.cacs, threshold=0.0, mode="imitation")
    cues = rollout(pol, sc[0], cfg)
    assert isinstance(cues, list)
    pol_rl = SequenceCACS(m, cfg.cacs, mode="rl")
    assert isinstance(rollout(pol_rl, sc[0], cfg), list)


def test_reinforce_runs_one_epoch(tmp_path):
    cfg = PipelineConfig(); sc = _scenes()
    d = collect(lambda s: OraclePolicy(s, cfg.cacs, seed=3), sc, cfg)
    m = SequenceModel(K=4, epochs=1).fit(d)
    out = train_reinforce(m, sc, sc[:1], cfg, epochs=1, log=lambda *_: None)
    assert len(out["history"]) == 2
    m.save(str(tmp_path / "s.pt")); m2 = SequenceModel.load(str(tmp_path / "s.pt"))
    assert m2.K == 4
