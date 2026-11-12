"""M1/S2: online logistic regression (AdaGrad), immutable updates, Laplace posterior for Thompson sampling."""
import numpy as np

from seeforme.online.learners import LogisticState, init_state, predict, sample_weights, update


def _stream(n=600, d=5, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = np.where(X @ w_true > 0, 1, -1)
    return X, y


def test_converges_on_separable_stream():                          # AC-7
    X, y = _stream()
    s = init_state(X.shape[1])
    mistakes_first, mistakes_last = 0, 0
    for i, (x, t) in enumerate(zip(X, y)):
        pred = 1 if predict(s, x[None])[0] >= 0.5 else -1
        if i < 100:
            mistakes_first += pred != t
        elif i >= len(X) - 100:
            mistakes_last += pred != t
        s = update(s, x[None], np.array([t]))
    assert mistakes_last < mistakes_first / 2
    assert mistakes_last <= 15


def test_update_is_pure():                                         # AC-8
    s = init_state(3)
    w0 = s.w.copy()
    s2 = update(s, np.ones((1, 3)), np.array([1]))
    assert isinstance(s2, LogisticState) and s2 is not s
    assert np.array_equal(s.w, w0) and not np.array_equal(s2.w, w0)
    assert s.n == 0 and s2.n == 1


def test_sample_weights_shrinks_with_evidence():
    X, y = _stream(400)
    s = init_state(X.shape[1])
    rng = np.random.default_rng(0)
    spread0 = np.std([sample_weights(s, rng) for _ in range(200)], axis=0).mean()
    for x, t in zip(X, y):
        s = update(s, x[None], np.array([t]))
    spread1 = np.std([sample_weights(s, rng) for _ in range(200)], axis=0).mean()
    assert spread1 < spread0 / 3


def test_sample_weights_shape_includes_bias():
    s = init_state(4)
    assert sample_weights(s, np.random.default_rng(0)).shape == s.w.shape == (5,)


def test_weighted_update_counts_more():
    s = init_state(2)
    a = update(s, np.array([[1.0, 0.0]]), np.array([1]), np.array([1.0]))
    b = update(s, np.array([[1.0, 0.0]]), np.array([1]), np.array([4.0]))
    assert b.w[0] > a.w[0] > 0
