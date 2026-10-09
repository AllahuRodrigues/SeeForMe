"""Learned predictive scheduler: a GRU over each track's recent event features that outputs

    action logits   silence / speak / interrupt / haptic
    urgency         regression target: how close the current time is to the latest safe moment
    slot mask       which semantic slots the message needs (object implicit)

Training: imitation of the CampusBench oracle with DAgger-style aggregation (the policy is rolled
out in closed loop, the states it visits are labelled by the oracle, aggregated, retrained), then
REINFORCE fine-tuning in closed loop with the benchmark's own communication score as the reward
(timely severity-weighted warnings, minus off-window cues, late penalties and occupied airtime).
The audio channel, the budget and the other candidates are part of the state, so the model can
learn to stay silent, to wait, and to cut in.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..campus.metrics import score as campus_score, _timeliness
from ..campus.oracle import CampusOracle
from ..campus.runner import run_on_campus
from ..campus.world import CampusScenario
from ..config import PipelineConfig
from ..cues.compose import SLOTS
from ..cues.pcacs import ACTIONS, ChannelState, EventFeatures, PCACSConfig, PCACSPolicy
from ..types import Cue, Observation, Track

N_IN = len(EventFeatures.names())
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
SLOT_IDX = {s: i for i, s in enumerate(SLOTS)}


ARCHS = ("gru", "mlp1", "mlpk", "attn")


def _right_align(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Move each left-aligned window (as built by `_pad`) to the end of the K slots, zero-padding in front,
    so slot K-1 is always the current step."""
    n, K, d = x.shape
    src = torch.arange(K).unsqueeze(0) - (K - lengths.clamp(max=K)).unsqueeze(1)   # (n, K) index in the left-aligned window
    out = x.gather(1, src.clamp(min=0).unsqueeze(-1).expand(n, K, d))
    return torch.where((src >= 0).unsqueeze(-1), out, torch.zeros((), dtype=x.dtype))


class SchedulerNet(nn.Module):
    """Candidate encoder + heads. `arch` selects how the K-step history is summarised (the Irie comparison):

        gru    GRU over the window (the original scheduler)
        mlp1   the current step only (no temporal memory)
        mlpk   the K steps concatenated (fixed-window memory, no recurrence)
        attn   learned-query attention over the K steps with position embeddings (equal-size temporal attention);
               the window is right-aligned like mlpk, so the last position embedding always marks the current step

    `hidden` is chosen by `matched_hidden` so every architecture has about the same number of parameters.
    """

    def __init__(self, n_in: int = N_IN, hidden: int = 48, arch: str = "gru", K: int = 8):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch, self.K, self.n_in = arch, K, n_in
        if arch == "gru":
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
        elif arch == "mlp1":
            self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        elif arch == "mlpk":
            self.enc = nn.Sequential(nn.Linear(n_in * K, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        else:
            self.embed = nn.Linear(n_in, hidden)
            self.pos = nn.Parameter(torch.zeros(K, hidden))
            self.query = nn.Parameter(torch.randn(hidden) * 0.1)
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.action = nn.Linear(hidden, len(ACTIONS))
        self.urgency = nn.Linear(hidden, 1)
        self.slots = nn.Linear(hidden, len(SLOTS))
        self.silence_bias = nn.Parameter(torch.zeros(1))   # global "say nothing" logit used in the categorical over candidates

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.arch == "gru":
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return h[-1]
        if self.arch == "mlp1":
            last = x[torch.arange(x.shape[0]), (lengths - 1).clamp(min=0)]
            return self.enc(last)
        if self.arch == "mlpk":
            # right-align the window so "the current step" is always the last block
            out = _right_align(x, lengths)
            n, K, d = out.shape
            return self.enc(out.reshape(n, K * d))
        # attn: right-align as well, so pos[-1] always marks the current step whatever the track's age
        x = _right_align(x, lengths)
        n, K, _ = x.shape
        e = self.embed(x) + self.pos[-K:].unsqueeze(0)
        scores = (self.key(e) @ self.query) / math.sqrt(e.shape[-1])
        mask = torch.arange(K).unsqueeze(0) >= (K - lengths).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.relu((w * self.value(e)).sum(1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self.trunk(self.encode(x, lengths))
        return self.action(z), self.urgency(z).squeeze(-1), self.slots(z)


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def matched_hidden(arch: str, target: int, n_in: int = N_IN, K: int = 8) -> int:
    """Hidden size for `arch` whose parameter count is closest to `target` (the GRU-48 scheduler has ~14k)."""
    best = None
    for h in range(8, 257, 2):
        n = count_params(SchedulerNet(n_in, h, arch, K))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (h, n)
        if n > 1.5 * target:
            break
    return best[0]


def _pad(windows: Sequence[Sequence[Sequence[float]]], K: int):
    n = len(windows)
    X = np.zeros((n, K, N_IN), dtype=np.float32)
    L = np.ones(n, dtype=np.int64)
    for i, w in enumerate(windows):
        w = w[-K:]
        L[i] = len(w)
        X[i, : len(w)] = np.asarray(w, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(L)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class SchedulerDataset:
    def __init__(self):
        self.windows: List[List[List[float]]] = []
        self.action: List[int] = []
        self.urgency: List[float] = []
        self.slots: List[List[float]] = []

    def add(self, window, action: str, urgency: float, slots: Sequence[str]):
        self.windows.append([list(r) for r in window])
        self.action.append(ACTION_IDX[action])
        self.urgency.append(urgency)
        self.slots.append([1.0 if s in slots else 0.0 for s in SLOTS])

    def extend(self, other: "SchedulerDataset"):
        self.windows += other.windows; self.action += other.action; self.urgency += other.urgency; self.slots += other.slots

    def __len__(self):
        return len(self.action)


def collect(policy_factory, scenarios: Sequence[CampusScenario], cfg: PipelineConfig, K: int, repeats: int = 1,
            seed: int = 0, run=run_on_campus) -> SchedulerDataset:
    """Roll out `policy_factory(scenario)` (a SchedulerPolicy or PCACSPolicy) and label every candidate with the oracle."""
    data = SchedulerDataset()
    for sc in scenarios:
        oracle = CampusOracle(sc)
        events = oracle.events
        for r in range(repeats):
            pol = policy_factory(sc)
            hist: Dict[int, deque] = {}
            announced: set = set()

            def on_decision(t, tracks, obs, cue, ch, due):
                for o in obs:
                    f = o.event if hasattr(o, "event") else None   # type: ignore[attr-defined]
                    if f is None:
                        continue
                    d = hist.setdefault(o.track.track_id, deque(maxlen=K))
                    d.append(f.as_vector())
                    sid = o.track.source_id
                    e = events.get(sid) if sid is not None else None
                    oracle.done = announced
                    a = oracle.label(o.track, t, ch, whole_window=True)
                    if a == "silence" and e is not None and sid not in announced and e.severity >= 0.5 \
                            and f.ttc <= 1.5 and f.path_prob >= 0.6 and ch.busy:
                        a = "haptic"
                    urg = 0.0
                    if e is not None and e.should_speak and e.latest_safe_s is not None:
                        span = max(0.5, e.latest_safe_s - e.earliest_useful_s)
                        urg = float(np.clip(1.0 - (e.latest_safe_s - t) / span, 0.0, 1.0)) if t <= e.latest_safe_s else 1.0
                    data.add(list(d), a, urg, e.minimum_information if e is not None else ())
                if cue is not None and cue.action != "haptic" and cue.source_id in events \
                        and events[cue.source_id].in_window(cue.t):
                    announced.add(cue.source_id)

            run(pol, sc, cfg, seed=sc.seed + 17 * r + seed, on_decision=on_decision)
    return data


# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------

class SchedulerModel:
    def __init__(self, K: int = 8, hidden: int = 48, epochs: int = 25, lr: float = 2e-3, batch_size: int = 512,
                 weight_decay: float = 1e-4, seed: int = 0, arch: str = "gru", match_params: bool = True):
        if arch != "gru" and match_params:
            hidden = matched_hidden(arch, count_params(SchedulerNet(N_IN, 48, "gru", K)), N_IN, K)
        self.K, self.hidden, self.epochs, self.lr, self.bs, self.wd, self.seed, self.arch = K, hidden, epochs, lr, batch_size, weight_decay, seed, arch
        torch.manual_seed(seed)
        self.net = SchedulerNet(N_IN, hidden, arch, K)
        self.history: List[float] = []

    @property
    def n_params(self) -> int:
        return count_params(self.net)

    def fit(self, data: SchedulerDataset) -> "SchedulerModel":
        X, L = _pad(data.windows, self.K)
        ya = torch.tensor(data.action, dtype=torch.long)
        yu = torch.tensor(data.urgency, dtype=torch.float32)
        ys = torch.tensor(data.slots, dtype=torch.float32)
        counts = torch.bincount(ya, minlength=len(ACTIONS)).float().clamp(min=1)
        wts = (counts.sum() / counts) ** 0.5
        ce = nn.CrossEntropyLoss(weight=wts / wts.mean())
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        g = torch.Generator().manual_seed(self.seed)
        n = len(ya)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g); tot = 0.0
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                la, lu, ls = self.net(X[idx], L[idx])
                loss = ce(la, ya[idx]) + 0.5 * nn.functional.mse_loss(lu, yu[idx]) \
                    + 0.3 * nn.functional.binary_cross_entropy_with_logits(ls, ys[idx])
                loss.backward(); opt.step()
                tot += float(loss.detach()) * len(idx)
            self.history.append(tot / n)
        self.net.eval()
        return self

    def forward(self, windows, grad: bool = False):
        X, L = _pad(windows, self.K)
        if grad:
            return self.net(X, L)
        with torch.no_grad():
            return self.net(X, L)

    def save(self, path: str) -> None:
        torch.save({"state": self.net.state_dict(), "K": self.K, "hidden": self.hidden, "history": self.history, "arch": self.arch,
                    "pos_align": "right"}, path)

    @staticmethod
    def load(path: str) -> "SchedulerModel":
        d = torch.load(path, map_location="cpu")
        if d.get("arch") == "attn" and d.get("pos_align") != "right":
            import warnings
            warnings.warn(f"{path}: attn checkpoint trained with left-aligned position embeddings; retrain it "
                          "(the encoder now right-aligns the window)", stacklevel=2)
        m = SchedulerModel(K=d["K"], hidden=d["hidden"], arch=d.get("arch", "gru"), match_params=False)
        m.net.load_state_dict(d["state"]); m.net.eval(); m.history = d.get("history", [])
        return m


# ---------------------------------------------------------------------------
# policy driven by the model
# ---------------------------------------------------------------------------

class SchedulerPolicy(PCACSPolicy):
    """P-CACS whose decision comes from the learned model.

    mode="imitation": per candidate, argmax over the action head (speak/interrupt/haptic need
                      P(action) >= threshold); the highest-probability non-silent candidate wins.
    mode="rl":        one categorical over [candidate x {speak, interrupt, haptic}] + global silence,
                      sampled during REINFORCE training, greedy at test time. Channel semantics are
                      enforced (speak needs a free channel, interrupt a busy one).
    """

    name = "pcacs_learned"

    def __init__(self, model: SchedulerModel, cfg: Optional[PCACSConfig] = None, threshold: float = 0.5,
                 mode: str = "imitation", name: Optional[str] = None, **kw):
        super().__init__(cfg, **kw)
        self.model, self.threshold, self.mode = model, threshold, mode
        self.K = model.K
        self.hist: Dict[int, deque] = {}
        self._seen: Dict[int, int] = {}
        self._step = 0
        self.train_mode = False
        self.last_logp = None
        self.last_entropy = None
        if name:
            self.name = name

    def reset(self) -> None:
        super().reset()
        self.hist.clear(); self._seen.clear(); self._step = 0
        self.last_logp = self.last_entropy = None

    def _apply_slots(self, o: Observation, mask: np.ndarray) -> None:
        from ..cues.compose import compose, speech_seconds
        slots = tuple(["object"] + [s for s, m in zip(SLOTS, mask) if m > 0.5 and s != "object"])
        trk, f = o.track, o.event   # type: ignore[attr-defined]
        o.slots = slots   # type: ignore[attr-defined]
        o.message = compose(slots, trk.label, trk.kind, trk.text, distance_m=f.distance, depth_conf=f.depth_conf,  # type: ignore[attr-defined]
                            angle_deg=f.angle, rel_speed=-f.own_speed, lateral_speed=(f.lateral_speed if f.moving else 0.0),
                            lateral_m=trk.lateral_m, hedged=o.hedged, units=self.cfg.units)  # type: ignore[attr-defined]
        o.duration = speech_seconds(o.message)   # type: ignore[attr-defined]

    def decide(self, tracks: List[Track], t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List[Cue]:
        c = self.cfg
        ch = ch or ChannelState()
        self._refill(t)
        self._step += 1
        self.last_logp = self.last_entropy = None
        self.last_action = "silence"
        obs = self._features(tracks, t, ch)
        obs = [o for o in obs if o.track.confidence >= c.min_confidence]
        if not obs:
            self._last_obs = []
            return []
        windows = []
        for o in obs:
            d = self.hist.setdefault(o.track.track_id, deque(maxlen=self.K))
            d.append(o.event.as_vector())   # type: ignore[attr-defined]
            windows.append(list(d))
            self._seen[o.track.track_id] = self._step
        if self._step % 20 == 0:
            for tid in [k for k, s in self._seen.items() if self._step - s > 20]:
                self.hist.pop(tid, None); self._seen.pop(tid, None)
        la, lu, ls = self.model.forward(windows, grad=self.train_mode)
        masks = torch.sigmoid(ls).detach().numpy()
        for o, m in zip(obs, masks):
            if c.use_slots:
                self._apply_slots(o, m)
        # legality of each action given the channel
        legal = torch.ones(len(obs), len(ACTIONS), dtype=torch.bool)
        for i, o in enumerate(obs):
            trk = o.track
            recent = trk.last_announced is not None and t - trk.last_announced < c.recent_block_s
            legal[i, ACTION_IDX["speak"]] = (not ch.busy) and not recent
            legal[i, ACTION_IDX["interrupt"]] = ch.busy and not recent and ch.current_track != trk.track_id
            legal[i, ACTION_IDX["haptic"]] = c.use_haptic and (trk.last_haptic is None or t - trk.last_haptic >= c.haptic_repeat_s)

        if self.mode == "imitation":
            probs = torch.softmax(la, dim=-1).detach().numpy()
            best = None
            for i, o in enumerate(obs):
                o.score = float(1.0 - probs[i, 0])
                for a in ("interrupt", "speak", "haptic"):
                    j = ACTION_IDX[a]
                    if legal[i, j] and probs[i, j] >= self.threshold and (best is None or probs[i, j] > best[0]):
                        best = (float(probs[i, j]), i, a)
            self._last_obs = sorted(obs, key=lambda o: -o.score)
            if best is None:
                return []
            _, i, a = best
            self.last_action = a
            return [self._make_cue(obs[i], t, a, ch)]

        # rl mode: joint categorical over (candidate, non-silent action) + global silence,
        # using each candidate's log-odds against its own silence logit
        flat = (la[:, 1:] - la[:, :1]).reshape(-1)
        flat_legal = legal[:, 1:].reshape(-1)
        full = torch.cat([flat.masked_fill(~flat_legal, -1e9), self.model.net.silence_bias])
        if self.train_mode:
            dist = torch.distributions.Categorical(logits=full)
            a = dist.sample()
            self.last_logp, self.last_entropy = dist.log_prob(a), dist.entropy()
            a = int(a)
        else:
            a = int(torch.argmax(full))
        sc = full[:-1].detach().reshape(len(obs), -1).max(dim=1).values.numpy()
        for o, s in zip(obs, sc):
            o.score = float(s) if s > -1e8 else float("-inf")
        self._last_obs = sorted(obs, key=lambda o: -o.score)
        if a == len(full) - 1:
            return []
        i, j = divmod(a, len(ACTIONS) - 1)
        action = ACTIONS[j + 1]
        self.last_action = action
        return [self._make_cue(obs[i], t, action, ch)]


# ---------------------------------------------------------------------------
# training loops
# ---------------------------------------------------------------------------

def closed_loop(policy_factory, scenarios: Sequence[CampusScenario], cfg: PipelineConfig, seed: int = 0, run=run_on_campus) -> Dict[str, float]:
    from ..campus.metrics import aggregate, evaluate_run
    from ..campus.oracle import CampusOracle
    rows, osc = [], {}
    for sc in scenarios:
        osc[sc.name] = campus_score(run(CampusOracle(sc), sc, cfg, seed=sc.seed + seed), sc)
        rows.append(evaluate_run(run(policy_factory(sc), sc, cfg, seed=sc.seed + seed), sc))
    a = aggregate(rows, osc)
    return next(iter(a.values()))


def select_threshold(model: SchedulerModel, val: Sequence[CampusScenario], cfg: PipelineConfig, grid=(0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
                     text_model=None, seed: int = 0, run=run_on_campus):
    best = None; hist = []
    for thr in grid:
        s = closed_loop(lambda sc: SchedulerPolicy(model, cfg.pcacs, threshold=thr, goal_tags=sc.goal_tags, text_model=text_model), val, cfg, seed, run=run)
        hist.append({"threshold": thr, **s})
        if best is None or s["score"] > best[1]["score"]:
            best = (thr, s)
    return best[0], best[1], hist


def dagger(train: Sequence[CampusScenario], val: Sequence[CampusScenario], cfg: PipelineConfig, K: int = 8, n_iter: int = 2,
           epochs: int = 25, text_model=None, seed: int = 0, log=print, run=run_on_campus, data0: Optional[SchedulerDataset] = None,
           arch: str = "gru"):
    """Imitation with data aggregation. Iteration 0 collects states under the linear P-CACS and the oracle
    (plus `data0`, e.g. states collected in another domain)."""
    data = collect(lambda sc: PCACSPolicy(cfg.pcacs, goal_tags=sc.goal_tags, text_model=text_model), train, cfg, K, repeats=1, seed=seed, run=run)
    data.extend(collect(lambda sc: CampusOracle(sc), train, cfg, K, repeats=1, seed=seed + 7, run=run))
    if data0 is not None:
        data.extend(data0)
    best, history = None, []
    for it in range(n_iter + 1):
        model = SchedulerModel(K=K, epochs=epochs, seed=seed, arch=arch).fit(data)
        thr, s, _ = select_threshold(model, val, cfg, text_model=text_model, seed=seed, run=run)
        counts = np.bincount(np.array(data.action), minlength=len(ACTIONS)).tolist()
        history.append({"iter": it, "n": len(data), "action_counts": dict(zip(ACTIONS, counts)), "threshold": thr, **s})
        log(f"  dagger iter {it}: n={len(data)} actions={dict(zip(ACTIONS, counts))} thr={thr} val score={s['score']:.3f} "
            f"regret={s['regret']:.3f} AWrecall={s['action_weighted_recall']:.3f} prec={s['cue_precision']:.3f} occ={s['audio_occupancy']:.2f}")
        if best is None or s["score"] > best["score"]:
            best = {"model": model, "threshold": thr, "score": s["score"], "iter": it}
        if it < n_iter:
            new = collect(lambda sc: SchedulerPolicy(model, cfg.pcacs, threshold=thr, goal_tags=sc.goal_tags, text_model=text_model),
                          train, cfg, K, repeats=1, seed=1000 * (it + 1) + seed, run=run)
            data.extend(new)
    return best, history


def train_reinforce(model: SchedulerModel, train: Sequence[CampusScenario], val: Sequence[CampusScenario], cfg: PipelineConfig,
                    epochs: int = 12, lr: float = 4e-4, gamma: float = 0.92, entropy_beta: float = 0.01,
                    batch_episodes: int = 4, seed: int = 0, text_model=None, log=print, run=run_on_campus) -> dict:
    """REINFORCE in closed loop; the per-decision reward is the increment of the CampusBench score."""
    rng = random.Random(seed); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr)
    baseline: Optional[float] = None

    def evaluate():
        return closed_loop(lambda sc: SchedulerPolicy(model, cfg.pcacs, mode="rl", goal_tags=sc.goal_tags, text_model=text_model), val, cfg, seed, run=run)

    # calibrate the global silence bias so the greedy RL policy starts near the imitation operating point
    cal = []
    for b in (-1.0, 0.0, 1.0, 2.0, 3.0):
        with torch.no_grad():
            model.net.silence_bias.fill_(b)
        cal.append((evaluate()["score"], b))
    best_b = max(cal)[1]
    with torch.no_grad():
        model.net.silence_bias.fill_(best_b)
    log(f"  silence bias calibrated: {best_b:+.1f} (val scores {[round(s, 2) for s, _ in cal]})")
    best_scores = evaluate(); best_state = copy.deepcopy(model.net.state_dict())
    history = [{"epoch": 0, "silence_bias": best_b, **best_scores}]
    log(f"  reinforce epoch 0 (imitation init): val score={best_scores['score']:.3f} regret={best_scores['regret']:.3f} "
        f"AWrecall={best_scores['action_weighted_recall']:.3f} prec={best_scores['cue_precision']:.3f} occ={best_scores['audio_occupancy']:.2f}")
    c_fp, c_occ, c_late = 0.3, 0.15, 0.5
    for ep in range(1, epochs + 1):
        order = list(train); rng.shuffle(order)
        losses, ep_returns = [], []
        for k, sc in enumerate(order):
            policy = SchedulerPolicy(model, cfg.pcacs, mode="rl", goal_tags=sc.goal_tags, text_model=text_model)
            policy.train_mode = True
            events = {e.obj_id: e for e in sc.speak_events()}
            counts_fp = getattr(sc, "counts_fp", None)      # EgoBlind: off-event cues count only near a question
            first: Dict[int, float] = {}
            logps, ents, rewards = [], [], []
            deadline_charged: set = set()

            def on_decision(t, tracks, obs, cue, ch, due):
                r = 0.0
                # late penalty the moment an unannounced event passes its latest safe time
                for oid, e in events.items():
                    if oid not in first and oid not in deadline_charged and t > e.latest_safe_s:
                        deadline_charged.add(oid); r -= c_late * e.severity
                if cue is not None and cue.action != "haptic":
                    r -= c_occ * cue.duration_s
                    sid = cue.source_id
                    if sid in events and sid not in first:
                        first[sid] = t
                        tl = _timeliness(t, events[sid])
                        r += events[sid].severity * tl if tl > 0 else -c_fp
                    elif sid in events:     # repeat of an announced event: score() always counts it as a false alarm
                        r -= c_fp
                    else:
                        r -= c_fp * (1.0 if counts_fp is None else float(counts_fp(t)))
                elif cue is not None:   # haptic: small reward if it marks a real imminent hazard
                    e = events.get(cue.source_id)
                    r += 0.1 if (e is not None and e.severity >= 0.5) else -0.05
                if policy.last_logp is not None:
                    logps.append(policy.last_logp); ents.append(policy.last_entropy); rewards.append(r)

            run(policy, sc, cfg, seed=sc.seed + 31 * ep, on_decision=on_decision)
            if not logps:
                continue
            G, rets = 0.0, []
            for r in reversed(rewards):
                G = r + gamma * G; rets.insert(0, G)
            rets_t = torch.tensor(rets, dtype=torch.float32)
            ep_returns.append(float(sum(rewards)))
            if baseline is None:
                baseline = float(rets_t.mean())
            adv = rets_t - baseline
            baseline = 0.9 * baseline + 0.1 * float(rets_t.mean())
            losses.append(-(torch.stack(logps) * adv).mean() - entropy_beta * torch.stack(ents).mean())
            if len(losses) == batch_episodes or k == len(order) - 1:
                opt.zero_grad(); torch.stack(losses).mean().backward()
                nn.utils.clip_grad_norm_(model.net.parameters(), 1.0); opt.step(); losses = []
        s = evaluate()
        history.append({"epoch": ep, "mean_return": float(np.mean(ep_returns)) if ep_returns else None, **s})
        log(f"  reinforce epoch {ep}: return={np.mean(ep_returns):.2f} val score={s['score']:.3f} regret={s['regret']:.3f} "
            f"AWrecall={s['action_weighted_recall']:.3f} prec={s['cue_precision']:.3f} occ={s['audio_occupancy']:.2f} lateC={s['late_critical_rate']:.2f}")
        if s["score"] > best_scores["score"]:
            best_scores, best_state = s, copy.deepcopy(model.net.state_dict())
    model.net.load_state_dict(best_state); model.net.eval()
    return {"best": best_scores, "history": history}


def load_learned_scheduler(which: str, model_dir, cfg: PipelineConfig, scenario: CampusScenario, text_model=None, name: Optional[str] = None):
    import json
    from pathlib import Path
    d = Path(model_dir) / "scheduler"
    rep_path = Path(model_dir) / "scheduler_report.json"
    if not rep_path.exists():
        raise FileNotFoundError("no trained scheduler: run scripts/train_scheduler.py")
    rep = json.loads(rep_path.read_text())
    if which == "il":
        m = SchedulerModel.load(str(d / "scheduler_il.pt"))
        return SchedulerPolicy(m, cfg.pcacs, threshold=rep["il"]["threshold"], mode="imitation", name=name or "pcacs_learned:il",
                               goal_tags=scenario.goal_tags, text_model=text_model)
    m = SchedulerModel.load(str(d / "scheduler_rl.pt"))
    return SchedulerPolicy(m, cfg.pcacs, mode="rl", name=name or "pcacs_learned:rl", goal_tags=scenario.goal_tags, text_model=text_model)
