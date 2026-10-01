"""Supervised cue-priority model: learn the CACS weighting from data.

The proposal asks whether learned weighting beats manually defined weighting.
Here we generate supervision directly from the benchmark: run the tracker over
training scenarios, and at every decision step label each candidate observation
with whether announcing it *now* would be useful (its object is inside an
important window and it was not announced in the last few seconds). The six CACS
features are the inputs.

Two models are trained on identical data:
  * LogisticRegression: its coefficients are directly comparable to
    (alpha, beta, gamma, delta, eta, -lambda), so the learned weights are interpretable.
  * A small PyTorch MLP (deep-learning variant) able to capture interactions
    such as "novelty only matters if proximity is high".
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..config import CACSConfig, PipelineConfig
from ..cues.features import FeatureExtractor
from ..sim.scenario import DetectorSimulator, Scenario
from ..tracking.tracker import IoUTracker
from ..types import CueFeatures


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------

def collect_training_data(scenarios: List[Scenario], cfg: Optional[PipelineConfig] = None,
                          dup_window: float = 6.0, announce_policy: str = "oracle_noisy", seed: int = 0,
                          text_model=None) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Roll out scenarios and produce (features, label) pairs.

    To obtain announcement history (needed for N and D) without depending on any
    particular policy, a noisy oracle announces useful observations with prob 0.7
    and non-useful ones with prob 0.1 at each decision step. This yields a mix of
    "recently announced" and "never announced" states for both classes.
    """
    rng = np.random.default_rng(seed)
    cfg = cfg or PipelineConfig()
    X, y, meta = [], [], []
    for sc in scenarios:
        objects = {o.obj_id: o for o in sc.objects}
        tracker = IoUTracker(cfg.tracker)
        fx = FeatureExtractor(cfg.cacs, text_model=text_model)
        sim = DetectorSimulator(sc)
        last_spoken = {}
        next_decision = 0.0
        for t, dets in sim.frames():
            tracks = tracker.update(dets, t)
            if t + 1e-9 < next_decision:
                continue
            next_decision = t + cfg.cacs.decision_interval
            for obs in fx.extract(tracks, t):
                sid = obs.track.source_id
                important = sid is not None and objects[sid].important_at(t)
                recently = sid is not None and sid in last_spoken and (t - last_spoken[sid]) < dup_window
                label = int(important and not recently)
                X.append(obs.features.as_vector()); y.append(label)
                meta.append({"scenario": sc.name, "t": t, "label_name": obs.track.label, "kind": obs.track.kind})
                p = 0.7 if label else 0.1
                if rng.random() < p:
                    fx.note_announced(obs.track, t)
                    if sid is not None:
                        last_spoken[sid] = t
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class TorchMLPPriority:
    """Small MLP with a scikit-learn-like interface (fit / predict_proba)."""

    def __init__(self, hidden: int = 32, epochs: int = 60, lr: float = 3e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, seed: int = 0, device: Optional[str] = None):
        self.hidden, self.epochs, self.lr, self.wd, self.bs, self.seed = hidden, epochs, lr, weight_decay, batch_size, seed
        self.device = device
        self.net = None
        self.mean = None
        self.std = None
        self.history: List[float] = []

    def _build(self, n_in: int):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(n_in, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                             nn.Linear(self.hidden, 1))

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        dev = self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        Xn = torch.tensor((X - self.mean) / self.std, device=dev)
        yt = torch.tensor(y, device=dev).unsqueeze(1)
        self.net = self._build(X.shape[1]).to(dev)
        pos = max(1.0, float(y.sum())); neg = max(1.0, float(len(y) - y.sum()))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=dev))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(Xn)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for i in range(0, n, self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                loss = loss_fn(self.net(Xn[idx]), yt[idx])
                loss.backward(); opt.step()
                tot += float(loss.detach()) * len(idx)
            self.history.append(tot / n)
        self.net.eval()
        self._dev = dev
        return self

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            z = self.net(torch.tensor((X - self.mean) / self.std, device=self._dev))
            p = torch.sigmoid(z).cpu().numpy().ravel()
        return np.stack([1 - p, p], axis=1)

    # joblib-friendly state
    def __getstate__(self):
        import io, torch
        buf = io.BytesIO()
        torch.save(self.net.state_dict() if self.net is not None else None, buf)
        d = self.__dict__.copy(); d["net"] = buf.getvalue(); d.pop("_dev", None)
        return d

    def __setstate__(self, d):
        import io, torch
        blob = d.pop("net"); self.__dict__.update(d)
        self._dev = "cpu"
        state = torch.load(io.BytesIO(blob), map_location="cpu")
        if state is not None:
            self.net = self._build(len(self.mean)); self.net.load_state_dict(state); self.net.eval()


def train_priority_models(train_scenarios: List[Scenario], val_scenarios: List[Scenario],
                          out_dir: str = "models", cfg: Optional[PipelineConfig] = None, seed: int = 0,
                          text_model=None) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

    Xtr, ytr, _ = collect_training_data(train_scenarios, cfg, seed=seed, text_model=text_model)
    Xva, yva, _ = collect_training_data(val_scenarios, cfg, seed=seed + 1, text_model=text_model)
    report = {"n_train": int(len(ytr)), "pos_rate_train": float(ytr.mean()), "n_val": int(len(yva)),
              "features": CueFeatures.names(), "models": {}}

    models = {
        "logreg": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "gbm": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=seed),
        "mlp_torch": TorchMLPPriority(seed=seed),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        r = {"val_auc": float(roc_auc_score(yva, p)), "val_ap": float(average_precision_score(yva, p)),
             "val_f1@0.5": float(f1_score(yva, (p >= 0.5).astype(int)))}
        if name == "logreg":
            coef = m.coef_[0]
            r["learned_weights"] = {n: float(c) for n, c in zip(CueFeatures.names(), coef)}
            r["intercept"] = float(m.intercept_[0])
        if name == "mlp_torch":
            r["final_train_loss"] = float(m.history[-1])
        # choose the decision threshold on the validation split (never on test)
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(yva, p)
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
        best = int(np.argmax(f1))
        r["val_threshold"] = float(thr[best]); r["val_f1@best"] = float(f1[best])
        report["models"][name] = r
        joblib.dump(m, Path(out_dir) / f"priority_{name}.joblib")
    (Path(out_dir) / "priority_report.json").write_text(json.dumps(report, indent=1))
    return report


def load_priority_model(name: str = "logreg", model_dir: str = "models"):
    import joblib
    return joblib.load(Path(model_dir) / f"priority_{name}.joblib")
