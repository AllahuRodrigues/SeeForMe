"""On-policy rollouts, an oracle policy, and DAgger-style data aggregation.

The first learned models were trained on data whose announcement history came
from a noisy oracle, not from the model itself. That mismatch ("off-policy"
training) is the usual reason imitation policies under-perform in closed loop.
DAgger fixes it: run the *current* learned policy, record the states it actually
visits, label them with the oracle, aggregate, retrain, repeat.

Everything here uses the same closed-loop semantics as `eval.runner`, so numbers
measured during training are directly comparable with the experiment scripts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import CACSConfig, PipelineConfig
from ..cues.cacs import CACSPolicy
from ..cues.messages import compose_message
from ..cues.policy import CuePolicy
from ..eval.runner import SpeechChannel
from ..sim.scenario import DetectorSimulator, Scenario
from ..tracking.tracker import IoUTracker
from ..types import Cue, CueFeatures, Observation, Track
from ..cues.features import EXTRA_NAMES

FEATURE_NAMES = CueFeatures.names() + EXTRA_NAMES   # 13 inputs


# ---------------------------------------------------------------------------
# closed-loop rollout
# ---------------------------------------------------------------------------

def rollout(policy: CuePolicy, scenario: Scenario, cfg: PipelineConfig, seed: Optional[int] = None,
            on_decision: Optional[Callable[[float, List[Observation], Optional[Cue]], None]] = None,
            scene_context=None) -> List[Cue]:
    """Run `policy` on `scenario` with the shared audio-channel model.

    `on_decision(t, observations, spoken_cue)` fires after every decision step at
    which the channel was free (the only steps where a policy acts).
    """
    tracker = IoUTracker(cfg.tracker)
    channel = SpeechChannel(cfg.speech.min_gap, cfg.speech.seconds_per_word)
    policy.reset()
    spoken: List[Cue] = []
    next_decision = 0.0
    for t, dets in DetectorSimulator(scenario, seed=seed).frames():
        tracks = tracker.update(dets, t)
        ctx = scene_context.observe(tracks, t) if scene_context is not None else None
        if t + 1e-9 < next_decision:
            continue
        next_decision = t + policy.decision_interval
        if not channel.free(t):
            continue
        cues = policy.select(tracks, t, ctx)
        cue = cues[0] if cues else None
        if cue is not None:
            channel.speak(cue)
            spoken.append(cue)
        if on_decision is not None:
            on_decision(t, policy.last_observations(), cue)
    return spoken


def is_useful(track: Track, t: float, objects: Dict[int, object], last_spoken: Dict[int, float],
              dup_window: float) -> bool:
    """Oracle label: object is inside an important window and was not announced within dup_window."""
    sid = track.source_id
    if sid is None:
        return False
    if sid in last_spoken and (t - last_spoken[sid]) < dup_window:
        return False
    return objects[sid].important_at(t)


# ---------------------------------------------------------------------------
# oracle policy (bootstraps imitation)
# ---------------------------------------------------------------------------

class OraclePolicy(CACSPolicy):
    """Speaks a genuinely useful candidate with probability p_useful and a useless
    one with probability p_noise, so recorded states include both announced and
    never-announced objects of both classes."""

    name = "oracle_noisy"

    def __init__(self, scenario: Scenario, cfg: Optional[CACSConfig] = None, p_useful: float = 0.7,
                 p_noise: float = 0.1, dup_window: float = 6.0, seed: int = 0, **kw):
        super().__init__(cfg, **kw)
        self.objects = {o.obj_id: o for o in scenario.objects}
        self.p_useful, self.p_noise, self.dup_window = p_useful, p_noise, dup_window
        self.rng = random.Random(seed)
        self.last_spoken: Dict[int, float] = {}

    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        obs = self.features.extract(tracks, t, scene_ctx)
        for o in obs:
            o.score = 1.0 if is_useful(o.track, t, self.objects, self.last_spoken, self.dup_window) else 0.0
        obs.sort(key=lambda o: -o.score)
        self._last_obs = obs
        cands = [o for o in obs if o.track.confidence >= self.cfg.min_confidence]
        useful = [o for o in cands if o.score > 0]
        useless = [o for o in cands if o.score == 0]
        pick = None
        if useful and self.rng.random() < self.p_useful:
            pick = self.rng.choice(useful)
        elif useless and self.rng.random() < self.p_noise:
            pick = self.rng.choice(useless)
        if pick is None:
            return []
        self.features.note_announced(pick.track, t)
        if pick.track.source_id is not None:
            self.last_spoken[pick.track.source_id] = t
        tr = pick.track
        return [Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind, message=compose_message(tr),
                    score=pick.score, policy=self.name, text=tr.text, source_id=tr.source_id)]

    def reset(self) -> None:
        super().reset()
        self.last_spoken.clear()


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    X: np.ndarray        # (n, 13) CACS features + extras, in FEATURE_NAMES order
    y: np.ndarray        # (n,) 1 = announcing now would be useful
    run: np.ndarray      # rollout id (groups rows of one closed-loop run)
    t: np.ndarray
    track: np.ndarray    # track id within the run

    def concat(self, other: "Dataset") -> "Dataset":
        return Dataset(*(np.concatenate([a, b]) for a, b in zip(self.as_tuple(), other.as_tuple())))

    def as_tuple(self):
        return (self.X, self.y, self.run, self.t, self.track)

    def __len__(self):
        return len(self.y)


def collect(policy_factory: Callable[[Scenario], CuePolicy], scenarios: Sequence[Scenario], cfg: PipelineConfig,
            repeats: int = 1, dup_window: float = 6.0, seed: int = 0, run_offset: int = 0) -> Dataset:
    """Roll out `policy_factory(scenario)` and label every candidate observation with the oracle."""
    X, y, run, ts, trk = [], [], [], [], []
    for k, sc in enumerate(scenarios):
        objects = {o.obj_id: o for o in sc.objects}
        for r in range(repeats):
            run_id = run_offset + 100 * k + r
            last_spoken: Dict[int, float] = {}

            def on_decision(t, obs, cue):
                for o in obs:
                    X.append(o.features.as_vector() + list(o.extras))
                    y.append(int(is_useful(o.track, t, objects, last_spoken, dup_window)))
                    run.append(run_id); ts.append(t); trk.append(o.track.track_id)
                if cue is not None and cue.source_id is not None:
                    last_spoken[cue.source_id] = cue.t

            rollout(policy_factory(sc), sc, cfg, seed=sc.seed + 17 * r + seed, on_decision=on_decision)
    return Dataset(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(run),
                   np.asarray(ts, dtype=np.float32), np.asarray(trk))


# ---------------------------------------------------------------------------
# closed-loop evaluation, threshold selection, DAgger
# ---------------------------------------------------------------------------

def closed_loop_scores(policy_factory: Callable[[Scenario], CuePolicy], scenarios: Sequence[Scenario],
                       cfg: PipelineConfig, repeats: int = 1, seed: int = 0, name: str = "p") -> Dict[str, float]:
    from ..eval.metrics import aggregate, evaluate_cues
    rows = []
    for sc in scenarios:
        for r in range(repeats):
            cues = rollout(policy_factory(sc), sc, cfg, seed=sc.seed + 17 * r + seed)
            rows.append(evaluate_cues(cues, sc, name))
    a = aggregate(rows)[name]
    rec, prec = a["important_cue_recall"], a["cue_precision"]
    a["f1"] = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    return a


def select_threshold(factory_at: Callable[[float, Scenario], CuePolicy], scenarios: Sequence[Scenario],
                     cfg: PipelineConfig, grid: Sequence[float], seed: int = 0):
    """Pick the decision threshold by closed-loop F1 (recall vs precision) on validation scenarios."""
    best_thr, best, history = None, None, []
    for thr in grid:
        s = closed_loop_scores(lambda sc: factory_at(thr, sc), scenarios, cfg, seed=seed)
        history.append({"threshold": thr, **s})
        if best is None or s["f1"] > best["f1"]:
            best_thr, best = thr, s
    return best_thr, best, history


def dagger(model_factory: Callable[[], object], policy_from_model: Callable[[object, float, Scenario], CuePolicy],
           data0: Dataset, train: Sequence[Scenario], val: Sequence[Scenario], cfg: PipelineConfig,
           n_iter: int = 2, thr_grid: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
           n_features: int = 6, seed: int = 0, log=print):
    """Imitation learning with data aggregation.

    iteration 0 trains on oracle-generated data; every later iteration adds the
    states visited by the previous model in closed loop (labelled by the oracle).
    The threshold is re-selected on the validation split each time; the best
    iteration by validation F1 is returned.
    """
    data, best, history = data0, None, []
    for it in range(n_iter + 1):
        model = model_factory()
        if getattr(model, "needs_dataset", False):
            model.fit(data)
        else:
            model.fit(data.X[:, :n_features], data.y)
        thr, scores, _ = select_threshold(lambda th, sc: policy_from_model(model, th, sc), val, cfg, thr_grid, seed=seed)
        rec = {"iter": it, "n_train": int(len(data)), "pos_rate": float(data.y.mean()), "threshold": thr, **scores}
        history.append(rec)
        log(f"  dagger iter {it}: n={len(data)} thr={thr} val F1={scores['f1']:.3f} recall={scores['important_cue_recall']:.3f} "
            f"prec={scores['cue_precision']:.3f} load={scores['audio_load_per_min']:.1f}")
        if best is None or scores["f1"] > best["f1"]:
            best = {"model": model, "threshold": thr, "f1": scores["f1"], "iter": it}
        if it < n_iter:
            new = collect(lambda sc: policy_from_model(model, thr, sc), train, cfg, repeats=1,
                          seed=1000 * (it + 1) + seed, run_offset=100000 * (it + 1))
            data = data.concat(new)
    return best, history
