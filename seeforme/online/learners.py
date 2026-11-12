"""Online logistic regression for P(should speak | candidate features).

AdaGrad steps on the weighted logistic loss with L2, plus a diagonal Laplace approximation of the posterior
(precision = prior + sum of w * p(1-p) * x^2) used for Thompson sampling. Every update returns a new state.
Labels are +1 (should have spoken) / -1 (should have stayed silent); the last weight is the bias.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LogisticState:
    w: np.ndarray          # (d+1,) weights, bias last
    g2: np.ndarray         # AdaGrad accumulator
    prec: np.ndarray       # diagonal posterior precision
    lr: float = 0.5
    l2: float = 1e-3
    n: int = 0             # number of labelled examples seen


def init_state(d: int, lr: float = 0.5, l2: float = 1e-3, prior_prec: float = 1.0, w0: Optional[np.ndarray] = None) -> LogisticState:
    w = np.zeros(d + 1) if w0 is None else np.asarray(w0, dtype=float).copy()
    if w.shape != (d + 1,):
        raise ValueError(f"w0 must have shape ({d + 1},), got {w.shape}")
    return LogisticState(w=w, g2=np.full(d + 1, 1e-8), prec=np.full(d + 1, float(prior_prec)), lr=lr, l2=l2)


def _aug(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.hstack([X, np.ones((X.shape[0], 1))])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def predict(s: LogisticState, X: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """P(label = +1) for each row of X (optionally under sampled weights w)."""
    return _sigmoid(_aug(X) @ (s.w if w is None else w))


def update(s: LogisticState, X: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None) -> LogisticState:
    """One AdaGrad step on the weighted mean logistic loss of a batch."""
    A = _aug(X)
    y01 = (np.asarray(y, dtype=float).reshape(-1) > 0).astype(float)
    wt = np.ones(len(y01)) if weight is None else np.asarray(weight, dtype=float).reshape(-1)
    if len(y01) == 0:
        return s
    p = _sigmoid(A @ s.w)
    grad = A.T @ (wt * (p - y01)) + s.l2 * s.w
    g2 = s.g2 + grad ** 2
    w = s.w - s.lr * grad / np.sqrt(g2)
    prec = s.prec + (A ** 2).T @ (wt * p * (1 - p))
    return replace(s, w=w, g2=g2, prec=prec, n=s.n + len(y01))


def sample_weights(s: LogisticState, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Thompson sample from the diagonal Laplace posterior N(w, scale^2 / prec)."""
    return s.w + scale * rng.normal(size=s.w.shape) / np.sqrt(s.prec)
