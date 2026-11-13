"""Frozen decision models: shapes, causal windows, save/load, and the frozen replay policy."""
import numpy as np
import pytest
import torch

from seeforme.config import PipelineConfig
from seeforme.cues.pcacs import ChannelState
from seeforme.online.decision_models import (DecisionModel, FrozenScorerPolicy, Standardizer, causal_windows,
                                             make_net, window_from_history)
from seeforme.online.policy import N_FEAT
from seeforme.online.stream import run_stream
from seeforme.tracking.tracker import IoUTracker

from tests.test_egoblind import synthetic_scenario

STD = Standardizer(np.zeros(N_FEAT), np.ones(N_FEAT))


def _model(family, K, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if family == "logistic":
        return DecisionModel("logistic", K, STD, coef=rng.normal(size=N_FEAT), intercept=-0.3, platt=(1.2, -0.1))
    kw = {"dropout": 0.2}
    return DecisionModel(family, K, STD, net=make_net(family, **kw), net_kw=kw)


def _rows(n_videos=3, n_tracks=4, n_steps=30, seed=0):
    rng = np.random.default_rng(seed)
    vid, trk, step = [], [], []
    for v in range(n_videos):
        for tr in range(n_tracks):
            for s in sorted(rng.choice(n_steps, size=n_steps // 2, replace=False)):
                vid.append(v); trk.append(tr); step.append(int(s))
    X = rng.normal(size=(len(vid), N_FEAT)).astype(np.float32)
    return vid, trk, step, X


@pytest.mark.parametrize("family", ["logistic", "mlp", "gru", "tcn"])
@pytest.mark.parametrize("K", [1, 2, 4, 8, 16])
def test_shapes(family, K):
    m = _model(family, K)
    W = np.random.default_rng(1).normal(size=(7, K, N_FEAT)).astype(np.float32)
    M = np.ones((7, K), dtype=np.float32)
    p = m.predict_proba(W, M)
    assert p.shape == (7,) and np.all((p > 0) & (p < 1))
    assert m.n_params() > 0


def test_window_from_history_ignores_future_and_pads():
    h = {3: np.full(N_FEAT, 3.0), 5: np.full(N_FEAT, 5.0), 6: np.full(N_FEAT, 6.0), 7: np.full(N_FEAT, 7.0)}
    W, M = window_from_history(h, 5, 4)
    assert W.shape == (4, N_FEAT) and M.tolist() == [0, 1, 0, 1]      # steps 2, 3, 4, 5
    assert W[1, 0] == 3.0 and W[3, 0] == 5.0 and W[0].sum() == 0 and W[2].sum() == 0
    assert 6.0 not in W and 7.0 not in W


@pytest.mark.parametrize("K", [2, 4, 8, 16])
def test_causal_windows_use_only_past_steps(K):
    vid, trk, step, X = _rows()
    W, M = causal_windows(vid, trk, step, X, K)
    assert W.shape == (len(X), K, N_FEAT) and M.shape == (len(X), K)
    assert np.allclose(W[:, -1], X) and np.all(M[:, -1] == 1)           # current step is the last slot
    t0 = 12
    X2 = X.copy()
    fut = np.array(step) > t0
    X2[fut] += 100.0                                                     # change only the future
    W2, M2 = causal_windows(vid, trk, step, X2, K)
    past = ~fut
    assert np.array_equal(W[past], W2[past]) and np.array_equal(M[past], M2[past])
    # and the model output at past rows is unchanged too, for every family
    for fam in ("logistic", "mlp", "gru", "tcn"):
        m = _model(fam, K)
        assert np.allclose(m.predict_proba(W[past], M[past]), m.predict_proba(W2[past], M2[past]))


def test_sequence_models_are_causal_inside_the_window():
    """Changing slots before the last one changes GRU / TCN output (they use history); the last-step models ignore it."""
    rng = np.random.default_rng(3)
    W = rng.normal(size=(5, 8, N_FEAT)).astype(np.float32)
    M = np.ones((5, 8), dtype=np.float32)
    W2 = W.copy(); W2[:, :-1] += 1.0
    for fam in ("logistic", "mlp"):
        m = _model(fam, 8)
        assert np.allclose(m.predict_proba(W, M), m.predict_proba(W2, M))
    for fam in ("gru", "tcn"):
        m = _model(fam, 8)
        assert not np.allclose(m.predict_proba(W, M), m.predict_proba(W2, M))


@pytest.mark.parametrize("family", ["logistic", "mlp", "gru", "tcn"])
def test_save_load_round_trip(family, tmp_path):
    m = _model(family, 8)
    m.std = Standardizer(np.arange(N_FEAT) * 0.1, np.arange(N_FEAT) * 0.05 + 1.0)
    rng = np.random.default_rng(2)
    W = rng.normal(size=(11, 8, N_FEAT)).astype(np.float32)
    M = (rng.random((11, 8)) > 0.3).astype(np.float32)
    f = tmp_path / f"{family}.pt"
    m.save(f)
    m2 = DecisionModel.load(f)
    assert m2.family == family and m2.K == 8 and m2.n_params() == m.n_params()
    assert np.allclose(m.predict_proba(W, M), m2.predict_proba(W, M))


def _tracks_at(sc, t_stop):
    tr = IoUTracker(PipelineConfig().tracker)
    tracks = []
    for t, dets in sc.frames():
        tracks = tr.update(dets, t)
        if t > t_stop:
            break
    return tracks


class _Const(DecisionModel):
    """Scores fixed per track id, so the test controls the ranking."""

    def __init__(self, by_track, K=4):
        super().__init__("logistic", K, STD, coef=np.zeros(N_FEAT))
        self.by_track = by_track
        self.order = []

    def predict_proba(self, W, M):
        return np.array([self.by_track.get(tid, 0.5) for tid in self.order])


def test_frozen_policy_speaks_only_legal_top_candidate():
    sc, _ = synthetic_scenario()
    tracks = _tracks_at(sc, 6.0)
    ids = [tk.track_id for tk in tracks]
    assert len(ids) >= 2
    model = _Const({ids[0]: 0.9, ids[1]: 0.8})
    pol = FrozenScorerPolicy(model, tau0=0.5, budget=False)
    pol.reset()
    orig = pol._features

    def feats(tracks_, t, ch):                      # record the candidate order the scorer will see
        obs = orig(tracks_, t, ch)
        model.order = [o.track.track_id for o in obs if o.track.confidence >= pol.cfg.min_confidence]
        return obs
    pol._features = feats
    assert pol.decide(tracks, 6.0, ChannelState(busy=True, remaining_s=1.0)) == []      # busy channel: silent
    cue = pol.decide(tracks, 6.0, ChannelState())
    assert len(cue) == 1 and cue[0].track_id == ids[0]                                  # top candidate
    cue2 = pol.decide(tracks, 6.5, ChannelState())                                      # ids[0] just announced
    assert len(cue2) == 1 and cue2[0].track_id == ids[1]
    assert pol.decide(tracks, 7.0, ChannelState()) == []                                # both recently announced
    hi = FrozenScorerPolicy(_Const({ids[0]: 0.9}), tau0=0.95, budget=False)
    hi.reset(); hi.model.order = model.order
    hi._features = lambda tr_, t, ch: orig(tr_, t, ch)
    assert hi.decide(tracks, 6.0, ChannelState()) == []                                 # below tau: silent


def test_frozen_policy_respects_budget_controller_and_never_learns():
    sc, _ = synthetic_scenario()
    m = _model("gru", 4)
    pol = FrozenScorerPolicy(m, target_per_min=2.0, tau0=0.6, budget=True)
    w0 = [p.detach().clone() for p in m.net.parameters()]
    z0 = pol.ctl.z
    spoke = []
    orig_step = pol._step_budget

    def step(s):
        spoke.append(s)
        orig_step(s)
    pol._step_budget = step
    logs = run_stream(pol, [sc] * 3, p_tap=1.0, seed=0)
    assert pol.ctl.z != z0                                                  # controller moved
    assert all(p0.equal(p1) for p0, p1 in zip(w0, m.net.parameters()))      # model unchanged by feedback
    spoken = [c for _, log in logs for c in log.spoken]
    assert sum(spoke) == len(spoken)                                        # every cue was charged to the controller
    assert len(spoke) > 0


def test_frozen_policy_cues_clear_tau_and_are_legal():
    sc, _ = synthetic_scenario()
    pol = FrozenScorerPolicy(_model("tcn", 8), tau0=0.3, budget=True)
    seen = []
    orig = pol.decide

    def decide(tracks, t, ch=None, scene_ctx=None):
        tau = pol.tau
        last = {tk.track_id: tk.last_announced for tk in tracks}
        out = orig(tracks, t, ch, scene_ctx)
        for c in out:
            seen.append((c, tau, ch, last.get(c.track_id), t))
        return out
    pol.decide = decide
    run_stream(pol, [sc] * 2, p_tap=1.0, seed=0)
    assert seen
    for c, tau, ch, last, t in seen:
        assert c.score >= tau
        assert ch is None or not ch.busy
        assert last is None or t - last >= pol.cfg.recent_block_s
