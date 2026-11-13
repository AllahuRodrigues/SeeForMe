"""Frozen "speak or stay silent" decision models over P-CACS candidate features, and a policy that replays them.

Every model scores one confirmed candidate at one decision from a causal window of that track's raw 29-dim P-CACS
vectors at the last K decision steps (slot K-1 = the current step), zero-padded with a 0/1 mask where the track was
not a candidate. Families:

    logistic   last step only (the online learner's model class); optional class weighting and Platt calibration
    mlp        last step only, 2 hidden layers of 64 units with dropout
    gru        GRU over the K-step window
    tcn        3 dilated causal 1D convolutions over the K-step window, read out at the last step

`FrozenScorerPolicy` is an OnlinePolicy that never learns: it scores the confirmed candidates with a fixed model,
picks the top legal one (channel free, not announced in the last `recent_block_s`) and speaks when its score is at
least the BudgetController's tau, exactly like online greedy.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..cues.pcacs import ChannelState
from ..types import Cue, Track
from .policy import N_FEAT, OnlinePolicy

FAMILIES = ("logistic", "mlp", "gru", "tcn")
CLIP = 5.0


# ---------------------------------------------------------------------------------------------- causal windows
def step_of(t: float, dt: float) -> int:
    return int(round(t / dt))


def window_from_history(hist: Dict[int, np.ndarray], step: int, K: int, d: int = N_FEAT) -> Tuple[np.ndarray, np.ndarray]:
    """(K, d) raw vectors and (K,) mask for steps step-K+1 .. step. Only steps <= `step` are ever read."""
    W = np.zeros((K, d), dtype=np.float32)
    M = np.zeros(K, dtype=np.float32)
    for j in range(K):
        s = step - (K - 1 - j)
        v = hist.get(s)
        if v is not None:
            W[j] = v
            M[j] = 1.0
    return W, M


def causal_windows(video: Sequence, track: Sequence, step: Sequence, X: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Windows for a table of candidate rows (one row per candidate per decision, raw vectors X).
    Row i gets the vectors of rows with the same (video, track) at steps step[i]-K+1 .. step[i]."""
    index: Dict[tuple, Dict[int, np.ndarray]] = {}
    for i, (v, tr, s) in enumerate(zip(video, track, step)):
        index.setdefault((v, tr), {})[int(s)] = X[i]
    n = len(X)
    W = np.zeros((n, K, X.shape[1]), dtype=np.float32)
    M = np.zeros((n, K), dtype=np.float32)
    for i, (v, tr, s) in enumerate(zip(video, track, step)):
        W[i], M[i] = window_from_history(index[(v, tr)], int(s), K, X.shape[1])
    return W, M


class Standardizer:
    """z-score with fixed (train) statistics, clipped to +-5; padded slots stay 0."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.where(np.asarray(sd) > 1e-6, sd, 1.0).astype(np.float32)

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        return Standardizer(X.mean(0), X.std(0))

    def __call__(self, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        Z = np.clip((W - self.mu) / self.sd, -CLIP, CLIP)
        return (Z * M[..., None]).astype(np.float32)


# ---------------------------------------------------------------------------------------------- networks
class MLPNet(nn.Module):
    def __init__(self, d: int = N_FEAT, h: int = 64, dropout: float = 0.2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, Z: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        return self.f(Z[:, -1]).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, d: int = N_FEAT, h: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(d + 1, h, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(h, 1)

    def forward(self, Z: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(torch.cat([Z, M.unsqueeze(-1)], -1))
        return self.out(self.drop(y[:, -1])).squeeze(-1)


class TCNNet(nn.Module):
    """Causal dilated convolutions (left padding only), kernel 3, dilations 1, 2, 4: receptive field 15 steps."""

    def __init__(self, d: int = N_FEAT, h: int = 32, dropout: float = 0.2, dilations=(1, 2, 4), k: int = 3):
        super().__init__()
        self.k, self.dil = k, tuple(dilations)
        chans = [d + 1] + [h] * len(self.dil)
        self.convs = nn.ModuleList(nn.Conv1d(chans[i], chans[i + 1], k, dilation=dl) for i, dl in enumerate(self.dil))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(h, 1)

    def forward(self, Z: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        x = torch.cat([Z, M.unsqueeze(-1)], -1).transpose(1, 2)        # (n, d+1, K)
        for i, (conv, dl) in enumerate(zip(self.convs, self.dil)):
            y = torch.relu(conv(nn.functional.pad(x, ((self.k - 1) * dl, 0))))
            x = y + x if (i > 0 and y.shape == x.shape) else y
            x = self.drop(x)
        return self.out(x[:, :, -1]).squeeze(-1)


def make_net(family: str, **kw) -> nn.Module:
    return {"mlp": MLPNet, "gru": GRUNet, "tcn": TCNNet}[family](**kw)


# ---------------------------------------------------------------------------------------------- frozen model
class DecisionModel:
    """A frozen scorer: raw windows (n, K, d) + mask (n, K) -> P(should speak) (n,)."""

    def __init__(self, family: str, K: int, std: Standardizer, net: Optional[nn.Module] = None,
                 coef: Optional[np.ndarray] = None, intercept: float = 0.0, platt: Optional[Tuple[float, float]] = None,
                 net_kw: Optional[dict] = None, meta: Optional[dict] = None):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family, self.K, self.std = family, int(K), std
        self.net, self.net_kw = net, dict(net_kw or {})
        self.coef = None if coef is None else np.asarray(coef, dtype=np.float64).ravel()
        self.intercept = float(intercept)
        self.platt = None if platt is None else (float(platt[0]), float(platt[1]))
        self.meta = dict(meta or {})
        if self.net is not None:
            self.net.eval()

    def logit(self, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=np.float32)
        M = np.asarray(M, dtype=np.float32)
        W, M = W[:, -self.K:], M[:, -self.K:]
        Z = self.std(W, M)
        if self.family == "logistic":
            z = Z[:, -1].astype(np.float64) @ self.coef + self.intercept
        else:
            with torch.no_grad():
                z = self.net(torch.from_numpy(Z), torch.from_numpy(M)).double().numpy()
        if self.platt is not None:
            z = self.platt[0] * z + self.platt[1]
        return z

    def predict_proba(self, W: np.ndarray, M: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.logit(W, M), -50, 50)))

    def n_params(self) -> int:
        if self.family == "logistic":
            return int(self.coef.size + 1 + (2 if self.platt else 0))
        return int(sum(p.numel() for p in self.net.parameters()))

    def save(self, path) -> None:
        torch.save({"family": self.family, "K": self.K, "mu": self.std.mu, "sd": self.std.sd,
                    "state": None if self.net is None else self.net.state_dict(), "net_kw": self.net_kw,
                    "coef": self.coef, "intercept": self.intercept, "platt": self.platt, "meta": self.meta}, str(path))

    @staticmethod
    def load(path) -> "DecisionModel":
        d = torch.load(str(path), weights_only=False)
        net = None
        if d["family"] != "logistic":
            net = make_net(d["family"], **d["net_kw"])
            net.load_state_dict(d["state"])
        return DecisionModel(d["family"], d["K"], Standardizer(d["mu"], d["sd"]), net=net, coef=d["coef"],
                             intercept=d["intercept"], platt=d["platt"], net_kw=d["net_kw"], meta=d["meta"])


# ---------------------------------------------------------------------------------------------- frozen policy
class FrozenScorerPolicy(OnlinePolicy):
    """Greedy speak-the-top-legal-candidate policy with a frozen scorer and the online budget controller.
    No learning: `learn` ignores feedback. Per-track windows are rebuilt from this video's own past decisions."""

    def __init__(self, model: DecisionModel, target_per_min: float = 2.0, tau0: float = 0.6, budget: bool = True,
                 seed: int = 0, name: Optional[str] = None, **kw):
        super().__init__(mode="greedy", target_per_min=target_per_min, tau0=tau0, lr=0.0, budget=budget, seed=seed,
                         name=name or f"frozen_{model.family}_k{model.K}", **kw)
        self.model = model
        self._track_hist: Dict[int, Dict[int, np.ndarray]] = {}
        self.score_seconds = 0.0
        self.n_scored_decisions = 0

    def reset(self) -> None:
        super().reset()
        self._track_hist = {}

    def learn(self, feedback) -> None:      # frozen
        return None

    def decide(self, tracks: List[Track], t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List[Cue]:
        c = self.cfg
        ch = ch or ChannelState()
        self.last_action = "silence"
        allobs = self._features(tracks, t, ch)
        self.last_reasons = {o.track.track_id: "low_confidence" for o in allobs if o.track.confidence < c.min_confidence}
        obs = [o for o in allobs if o.track.confidence >= c.min_confidence]
        self._last_obs = obs
        step = step_of(t, self.decision_interval)
        K = self.model.K
        for o in obs:
            h = self._track_hist.setdefault(o.track.track_id, {})
            h[step] = np.asarray(o.event.as_vector(), dtype=np.float32)   # type: ignore[attr-defined]
            for s in [s for s in h if s <= step - K]:
                del h[s]
        if not obs:
            self._step_budget(False)
            return []
        WM = [window_from_history(self._track_hist[o.track.track_id], step, K) for o in obs]
        t0 = time.perf_counter()
        p = self.model.predict_proba(np.stack([w for w, _ in WM]), np.stack([m for _, m in WM]))
        self.score_seconds += time.perf_counter() - t0
        self.n_scored_decisions += 1
        for o, pi in zip(obs, p):
            o.score = float(pi)
        legal = [i for i, o in enumerate(obs)
                 if not ch.busy and (o.track.last_announced is None or t - o.track.last_announced >= c.recent_block_s)]
        for j, o in enumerate(obs):
            self.last_reasons[o.track.track_id] = ("channel_busy" if ch.busy else "recently_announced" if j not in legal
                                                   else "below_threshold" if p[j] < self.tau else "outranked")
        if not legal:
            self._step_budget(False)
            return []
        i = max(legal, key=lambda j: p[j])
        speak = bool(p[i] >= self.tau)
        self._step_budget(speak)
        if not speak:
            return []
        self.last_action = "speak"
        self.last_reasons[obs[i].track.track_id] = "spoken"
        return [self._make_cue(obs[i], t, "speak", ch)]

    def to_dict(self):
        return {"family": self.model.family, "K": self.model.K, "tau": self.tau,
                "mean_score_ms": 1000 * self.score_seconds / max(1, self.n_scored_decisions)}
