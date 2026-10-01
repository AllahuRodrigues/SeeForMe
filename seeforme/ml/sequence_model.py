"""Deep context model: a GRU over each track's recent feature history.

Instead of scoring a single snapshot (C, R, N, P, T, D), the sequence model sees
the last K decision steps of a track (13 signals per step: the six CACS features
plus position, size, growth, age, time since announcement, announcement count)
and outputs a "speak now" logit. Two training modes:

  * imitation:  binary cross-entropy against oracle labels, with DAgger
                aggregation (see onpolicy.py);
  * reinforce:  REINFORCE fine-tuning in closed loop with a reward that pays
                +1 for a useful cue, penalises useless / duplicate cues, and
                slightly penalises silence while a useful candidate is visible.
                The audio channel is part of the environment, so the return
                captures the cost of occupying it.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..config import CACSConfig, PipelineConfig
from ..cues.cacs import CACSPolicy
from ..cues.messages import compose_message
from ..types import Cue, Observation, Track
from .onpolicy import Dataset, is_useful, rollout, closed_loop_scores

N_IN = 13


class SequenceNet(nn.Module):
    def __init__(self, n_in: int = N_IN, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.silence = nn.Parameter(torch.zeros(1))   # logit of the "stay silent" action (RL mode)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1]).squeeze(-1)


def build_sequences(data: Dataset, K: int):
    """Right-padded windows of the previous <=K rows of the same (run, track)."""
    order = np.lexsort((data.t, data.track, data.run))
    n = len(data)
    X = np.zeros((n, K, data.X.shape[1]), dtype=np.float32)
    L = np.ones(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.float32)
    hist: Dict[tuple, deque] = {}
    for i, idx in enumerate(order):
        key = (int(data.run[idx]), int(data.track[idx]))
        d = hist.setdefault(key, deque(maxlen=K))
        d.append(data.X[idx])
        L[i] = len(d)
        X[i, : len(d)] = np.stack(d)
        y[i] = data.y[idx]
    return X, L, y


def _pad(windows: Sequence[Sequence[Sequence[float]]], K: int):
    n = len(windows)
    X = np.zeros((n, K, N_IN), dtype=np.float32)
    L = np.ones(n, dtype=np.int64)
    for i, w in enumerate(windows):
        w = w[-K:]
        L[i] = len(w)
        X[i, : len(w)] = np.asarray(w, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(L)


class SequenceModel:
    """Trainable wrapper (imitation mode) with save/load; `needs_dataset` tells DAgger to pass the Dataset."""

    needs_dataset = True

    def __init__(self, K: int = 8, hidden: int = 32, epochs: int = 30, lr: float = 2e-3, batch_size: int = 512,
                 weight_decay: float = 1e-4, seed: int = 0):
        self.K, self.hidden, self.epochs, self.lr, self.bs, self.wd, self.seed = K, hidden, epochs, lr, batch_size, weight_decay, seed
        torch.manual_seed(seed)
        self.net = SequenceNet(N_IN, hidden)
        self.history: List[float] = []

    def fit(self, data: Dataset) -> "SequenceModel":
        X, L, y = build_sequences(data, self.K)
        Xt, Lt, yt = torch.from_numpy(X), torch.from_numpy(L), torch.from_numpy(y)
        pos = max(1.0, float(y.sum())); neg = max(1.0, float(len(y) - y.sum()))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos]))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        g = torch.Generator().manual_seed(self.seed)
        n = len(yt)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx], Lt[idx]), yt[idx])
                loss.backward(); opt.step()
                tot += float(loss.detach()) * len(idx)
            self.history.append(tot / n)
        self.net.eval()
        return self

    def logits(self, windows, grad: bool = False) -> torch.Tensor:
        X, L = _pad(windows, self.K)
        if grad:
            return self.net(X, L)
        with torch.no_grad():
            return self.net(X, L)

    def save(self, path: str) -> None:
        torch.save({"state": self.net.state_dict(), "K": self.K, "hidden": self.hidden, "history": self.history}, path)

    @staticmethod
    def load(path: str) -> "SequenceModel":
        d = torch.load(path, map_location="cpu")
        m = SequenceModel(K=d["K"], hidden=d["hidden"])
        m.net.load_state_dict(d["state"]); m.net.eval(); m.history = d.get("history", [])
        return m


class SequenceCACS(CACSPolicy):
    """Cue policy driven by the sequence model.

    mode="imitation": speak the top candidate if sigmoid(logit) >= threshold.
    mode="rl":        softmax over [candidates..., silence]; greedy at test time,
                      sampled during REINFORCE training (train_mode=True).
    """

    name = "cacs_seq"

    def __init__(self, model: SequenceModel, cfg: Optional[CACSConfig] = None, threshold: float = 0.5,
                 mode: str = "imitation", name: Optional[str] = None, **kw):
        super().__init__(replace(cfg or CACSConfig(), threshold=threshold), **kw)
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.K = model.K
        self.hist: Dict[int, deque] = {}
        self._seen: Dict[int, int] = {}
        self._step = 0
        self.train_mode = False
        self.last_action: Optional[Observation] = None
        self.last_logp = None
        self.last_entropy = None
        if name:
            self.name = name

    def reset(self) -> None:
        super().reset()
        self.hist.clear(); self._seen.clear(); self._step = 0
        self.last_action = self.last_logp = self.last_entropy = None

    def _announce(self, o: Observation, t: float, hedged: bool) -> Cue:
        self.features.note_announced(o.track, t)
        tr = o.track
        return Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind, message=compose_message(tr, hedged=hedged),
                   score=o.score, policy=self.name, text=tr.text, source_id=tr.source_id, hedged=hedged)

    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        self._step += 1
        self.last_action = self.last_logp = self.last_entropy = None
        obs = self.features.extract(tracks, t, scene_ctx)
        if not obs:
            self._last_obs = []
            return []
        windows = []
        for o in obs:
            d = self.hist.setdefault(o.track.track_id, deque(maxlen=self.K))
            d.append(o.features.as_vector() + list(o.extras))
            windows.append(list(d))
            self._seen[o.track.track_id] = self._step
        if self._step % 20 == 0:   # forget tracks not seen for a while
            for tid in [k for k, s in self._seen.items() if self._step - s > 20]:
                self.hist.pop(tid, None); self._seen.pop(tid, None)

        logits = self.model.logits(windows, grad=self.train_mode)
        gates = [self._gate(o.track) for o in obs]
        allowed = torch.tensor([g is not None for g in gates])

        if self.mode == "imitation":
            probs = torch.sigmoid(logits).detach().numpy()
            for o, p in zip(obs, probs):
                o.score = float(p)
            ranked = sorted(obs, key=lambda o: -o.score)
            self._last_obs = ranked
            for o, g in sorted(zip(obs, gates), key=lambda og: -og[0].score):
                if g is None:
                    continue
                if o.score < self.threshold:
                    return []
                self.last_action = o
                return [self._announce(o, t, g)]
            return []

        # ---- rl mode: categorical over candidates + silence -----------------
        full = torch.cat([logits.masked_fill(~allowed, -1e9), self.model.net.silence])
        if self.train_mode:
            dist = torch.distributions.Categorical(logits=full)
            a = dist.sample()
            self.last_logp = dist.log_prob(a)
            self.last_entropy = dist.entropy()
            a = int(a)
        else:
            a = int(torch.argmax(full))
        sc = full[:-1].detach().numpy()
        for o, s in zip(obs, sc):
            o.score = float(s) if s > -1e8 else float("-inf")
        self._last_obs = sorted(obs, key=lambda o: -o.score)
        if a == len(obs):
            return []
        self.last_action = obs[a]
        return [self._announce(obs[a], t, bool(gates[a]))]


# ---------------------------------------------------------------------------
# REINFORCE fine-tuning
# ---------------------------------------------------------------------------

def train_reinforce(model: SequenceModel, train: Sequence[object], val: Sequence[object], cfg: PipelineConfig,
                    epochs: int = 20, lr: float = 5e-4, gamma: float = 0.9, entropy_beta: float = 0.01,
                    r_useful: float = 1.0, r_useless: float = -0.5, r_duplicate: float = -0.75,
                    r_missed: float = -0.25, dup_window: float = 6.0, batch_episodes: int = 4, seed: int = 0,
                    text_model=None, log=print) -> dict:
    """Fine-tune `model` (in place) with REINFORCE; returns best-by-validation weights and history."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    policy = SequenceCACS(model, cfg.cacs, mode="rl", text_model=text_model)
    policy.train_mode = True
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    baseline: Optional[float] = None

    def evaluate():
        p = SequenceCACS(model, cfg.cacs, mode="rl", text_model=text_model)
        return closed_loop_scores(lambda sc: p, val, cfg, seed=seed)

    best_scores = evaluate()
    best_state = copy.deepcopy(model.net.state_dict())
    history = [{"epoch": 0, "mean_return": None, **best_scores}]
    log(f"  reinforce epoch 0 (imitation init): val F1={best_scores['f1']:.3f} recall={best_scores['important_cue_recall']:.3f} "
        f"prec={best_scores['cue_precision']:.3f} load={best_scores['audio_load_per_min']:.1f}")

    for ep in range(1, epochs + 1):
        order = list(train); rng.shuffle(order)
        ep_returns = []
        losses = []
        for k, sc in enumerate(order):
            objects = {o.obj_id: o for o in sc.objects}
            last_spoken: Dict[int, float] = {}
            logps, ents, rewards = [], [], []

            def on_decision(t, obs, cue):
                if policy.last_logp is None:      # no candidates: nothing was decided
                    return
                if policy.last_action is None:
                    r = r_missed if any(is_useful(o.track, t, objects, last_spoken, dup_window) for o in obs) else 0.0
                else:
                    tr = policy.last_action.track
                    sid = tr.source_id
                    important = sid is not None and objects[sid].important_at(t)
                    recently = sid is not None and sid in last_spoken and (t - last_spoken[sid]) < dup_window
                    r = r_useful if (important and not recently) else (r_duplicate if important else r_useless)
                    if sid is not None:
                        last_spoken[sid] = t
                logps.append(policy.last_logp); ents.append(policy.last_entropy); rewards.append(r)

            rollout(policy, sc, cfg, seed=sc.seed + 31 * ep, on_decision=on_decision)
            if not logps:
                continue
            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            ep_returns.append(float(sum(rewards)))
            if baseline is None:
                baseline = float(returns_t.mean())
            adv = returns_t - baseline
            baseline = 0.9 * baseline + 0.1 * float(returns_t.mean())
            loss = -(torch.stack(logps) * adv).mean() - entropy_beta * torch.stack(ents).mean()
            losses.append(loss)
            if len(losses) == batch_episodes or k == len(order) - 1:
                opt.zero_grad()
                torch.stack(losses).mean().backward()
                nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
                opt.step()
                losses = []
        scores = evaluate()
        history.append({"epoch": ep, "mean_return": float(np.mean(ep_returns)) if ep_returns else None, **scores})
        log(f"  reinforce epoch {ep}: mean return={np.mean(ep_returns):.2f} val F1={scores['f1']:.3f} "
            f"recall={scores['important_cue_recall']:.3f} prec={scores['cue_precision']:.3f} load={scores['audio_load_per_min']:.1f}")
        if scores["f1"] > best_scores["f1"]:
            best_scores, best_state = scores, copy.deepcopy(model.net.state_dict())
    model.net.load_state_dict(best_state); model.net.eval()
    return {"best": best_scores, "history": history}
