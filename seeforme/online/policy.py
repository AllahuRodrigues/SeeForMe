"""OnlinePolicy: P-CACS features, an online logistic model of P(should speak), a cue-rate controller.

Modes
    greedy     speak the best legal candidate when P >= tau; learns only from the labels it happens to get
    apple      greedy plus exploration below tau (apple tasting): a share eps of the cue budget, importance weighted
    thompson   greedy under weights sampled from the Laplace posterior (exploration from uncertainty)
    full_info  upper bound: every candidate at every decision is labelled from the scenario (not deployable)

Labels arrive through `learn(feedback)`: a tap applies to the spoken candidate, a missed question to every remembered
candidate of that event in [t0, t1], a quiet window to every remembered candidate in [t0, t1] (each question carries
total weight 1, split over its candidates). The model persists across `reset()` (one reset per video).

With `implicit_neg_w > 0`, a remembered candidate that no feedback covered by the time it leaves the lookback window
(or the video ends) becomes a weak negative of that weight: nobody asked about it, so it probably wasn't needed.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from ..cues.pcacs import ChannelState, EventFeatures, PCACSConfig, PCACSPolicy
from ..types import Cue, Track
from .feedback import Feedback
from .learners import LogisticState, init_state, predict, sample_weights, update

MODES = ("greedy", "apple", "thompson", "full_info")
N_FEAT = len(EventFeatures.names())
MAX_IPW = 10.0


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


@dataclass(frozen=True)
class BudgetController:
    """Dual ascent on the logit of the speaking threshold so the long-run cue rate tracks `target_per_min`.
    Working in logit space keeps the step proportional near tau ~ 0.98, where a 2 cues/min budget usually sits."""
    target_per_min: float = 2.0
    z: float = 0.0          # logit(tau)
    eta: float = 0.3
    lo: float = 0.01
    hi: float = 0.999

    @staticmethod
    def make(target_per_min: float = 2.0, tau0: float = 0.5, eta: float = 0.3) -> "BudgetController":
        return BudgetController(target_per_min=target_per_min, z=_logit(tau0), eta=eta)

    @property
    def tau(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.z))

    def step(self, spoke: bool, dt: float) -> "BudgetController":
        g = (1.0 if spoke else 0.0) - self.target_per_min * dt / 60.0
        z = min(_logit(self.hi), max(_logit(self.lo), self.z + self.eta * g))
        return replace(self, z=z)


class _Normalizer:
    """Running mean / variance (Welford) of the raw feature vectors; z-scores clipped to +-5."""

    def __init__(self, d: int):
        self.n, self.mean, self.m2 = 0, np.zeros(d), np.zeros(d)

    def observe(self, X: np.ndarray) -> None:
        for x in X:
            self.n += 1
            delta = x - self.mean
            self.mean = self.mean + delta / self.n
            self.m2 = self.m2 + delta * (x - self.mean)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        sd = np.sqrt(self.m2 / max(1, self.n - 1)) if self.n > 1 else np.ones_like(self.mean)
        return np.clip((X - self.mean) / np.where(sd > 1e-6, sd, 1.0), -5.0, 5.0)


class OnlinePolicy(PCACSPolicy):
    name = "online"
    supports_actions = True

    def __init__(self, mode: str = "greedy", cfg: Optional[PCACSConfig] = None, target_per_min: float = 2.0, tau0: float = 0.6,
                 eps: float = 0.25, lr: float = 0.5, l2: float = 1e-3, budget: bool = True, seed: int = 0, lookback_s: float = 15.0,
                 ts_scale: float = 1.0, implicit_neg_w: float = 0.0, name: Optional[str] = None, **kw):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        kw.pop("oracle_scenario", None)
        super().__init__(cfg, name=name or f"online_{mode}", **kw)
        self.mode, self.eps, self.ts_scale, self.lookback_s = mode, eps, ts_scale, lookback_s
        self.implicit_neg_w = implicit_neg_w
        self.use_budget_ctl = budget
        self.tau0 = tau0
        self.ctl = BudgetController.make(target_per_min, tau0)
        self.state: LogisticState = init_state(N_FEAT, lr=lr, l2=l2)
        self.norm = _Normalizer(N_FEAT)
        self._rng = random.Random(seed)
        self._nrng = np.random.default_rng(seed)
        self._hist: Deque[list] = deque()      # [t, source_id, x, covered by feedback]
        self._spoken: List[Tuple[float, np.ndarray, float]] = []      # (t, x, importance weight) of this video's cues
        self._scenario = None
        self._feedback = None
        self.n_labels = 0
        self.n_pos = 0

    # ------------------------------------------------------------------ video lifecycle
    def begin_video(self, scenario, feedback_model) -> None:
        self._scenario, self._feedback = scenario, feedback_model

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_hist"):
            self._expire(float("inf"))
        self._spoken = []

    @property
    def tau(self) -> float:
        return self.ctl.tau

    # ------------------------------------------------------------------ decisions
    def decide(self, tracks: List[Track], t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List[Cue]:
        c = self.cfg
        ch = ch or ChannelState()
        self.last_action = "silence"
        self._expire(t - self.lookback_s)
        obs = [o for o in self._features(tracks, t, ch) if o.track.confidence >= c.min_confidence]
        self._last_obs = obs
        if not obs:
            self._step_budget(False)
            return []
        raw = np.array([o.event.as_vector() for o in obs], dtype=float)   # type: ignore[attr-defined]
        self.norm.observe(raw)
        X = self.norm(raw)
        w = sample_weights(self.state, self._nrng, self.ts_scale) if self.mode == "thompson" else None
        p = predict(self.state, X, w)
        for o, x, pi in zip(obs, X, p):
            o.score = float(pi)
            self._hist.append([t, o.track.source_id, x, False])
        if self.mode == "full_info":
            self._learn_full_info(obs, X, t)
        legal = [i for i, o in enumerate(obs)
                 if not ch.busy and (o.track.last_announced is None or t - o.track.last_announced >= c.recent_block_s)]
        if not legal:
            self._step_budget(False)
            return []
        i = max(legal, key=lambda j: p[j])
        speak, weight = bool(p[i] >= self.tau), 1.0
        if not speak and self.mode == "apple":
            # eps = share of the cue budget spent on exploration; explored cues count against the budget too, so the
            # controller raises tau to pay for them. Importance weights are capped to keep updates stable.
            p_explore = min(1.0, self.eps * self.ctl.target_per_min * self.decision_interval / 60.0)
            if self._rng.random() < p_explore:
                speak, weight = True, min(MAX_IPW, 1.0 / p_explore)
        self._step_budget(speak)
        if not speak:
            return []
        self.last_action = "speak"
        self._spoken.append((t, X[i], weight))
        return [self._make_cue(obs[i], t, "speak", ch)]

    def _expire(self, before: float) -> None:
        """Forget candidates older than `before`; uncovered ones become weak negatives when implicit_neg_w > 0."""
        neg = []
        while self._hist and self._hist[0][0] < before:
            ts, sid, x, covered = self._hist.popleft()
            if not covered:
                neg.append(x)
        if neg and self.implicit_neg_w > 0:
            self._fit(np.array(neg), np.full(len(neg), -1), np.full(len(neg), self.implicit_neg_w))

    def _step_budget(self, spoke: bool) -> None:
        if self.use_budget_ctl:
            self.ctl = self.ctl.step(spoke, self.decision_interval)

    # ------------------------------------------------------------------ learning
    def learn(self, feedback: Sequence[Feedback]) -> None:
        for f in feedback:
            if f.kind.startswith("tap"):
                hit = [(x, wt) for (ts, x, wt) in self._spoken if abs(ts - f.t) < 1e-6]
                if not hit:
                    continue
                for h in self._hist:
                    if abs(h[0] - f.t) < 1e-6 and any(np.array_equal(h[2], x) for x, _ in hit):
                        h[3] = True
                X = np.array([h[0] for h in hit]); W = np.array([h[1] for h in hit])
            else:
                hits = [h for h in self._hist
                        if f.t0 - 1e-6 <= h[0] <= f.t1 + 1e-6 and (f.source_id is None or h[1] == f.source_id)]
                if not hits:
                    continue
                for h in hits:
                    h[3] = True
                rows = [h[2] for h in hits]
                X = np.array(rows); W = np.full(len(rows), 1.0 / len(rows))
            self._fit(X, np.full(len(X), f.label), W)

    def _learn_full_info(self, obs, X: np.ndarray, t: float) -> None:
        sc, fb = self._scenario, self._feedback
        if sc is None:
            return
        announced = fb.announced if fb is not None else set()
        ev = {e.obj_id: e for e in sc.speak_events()}
        ys, xs = [], []
        for o, x in zip(obs, X):
            sid = o.track.source_id
            e = ev.get(sid) if sid is not None else None
            if e is not None and e.in_window(t) and sid not in announced:
                ys.append(1); xs.append(x)
            elif sc.counts_fp(t):
                ys.append(-1); xs.append(x)
        if xs:
            self._fit(np.array(xs), np.array(ys), np.full(len(xs), 1.0 / len(xs)))

    def _fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        self.state = update(self.state, X, y, w)
        self.n_labels += len(y)
        self.n_pos += int((np.asarray(y) > 0).sum())

    def to_dict(self):
        return {"mode": self.mode, "tau": self.tau, "eps": self.eps, "implicit_neg_w": self.implicit_neg_w, "n_labels": self.n_labels, "w": [round(float(v), 4) for v in self.state.w]}
