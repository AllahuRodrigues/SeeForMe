"""Unsupervised scene-context model (k-means over detection histograms).

Every window of tracked detections is summarised as a normalised class histogram.
K-means over these histograms discovers scene types (hallway-like, office-like,
street-like, ...) without labels. Two contextual signals are derived:

  * surprise: a class that is rare in the current scene cluster is more worth
    mentioning (a car in an indoor scene) -> boosts relevance R_i.
  * scene change: when the assigned cluster changes, the environment changed, so
    the decision layer can treat the scene as novel.

This is the unsupervised component of the project; it feeds R_i in CACS.
"""
from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..types import Track


class SceneContextModel:
    def __init__(self, vocab: Optional[List[str]] = None, k: int = 4, window: int = 20, surprise_gain: float = 0.6):
        self.vocab = vocab or []
        self.k = k
        self.window = window
        self.surprise_gain = surprise_gain
        self.kmeans = None
        self.cluster_class_freq: Optional[np.ndarray] = None  # (k, |vocab|)
        self.cluster_names: List[str] = []
        self._buffer: deque = deque(maxlen=window)
        self._last_cluster: Optional[int] = None
        self._last_change_t: float = -1e9

    # ---- features -----------------------------------------------------------
    def _hist(self, label_lists: List[List[str]]) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        idx = {l: i for i, l in enumerate(self.vocab)}
        for labels in label_lists:
            for l in labels:
                if l in idx:
                    v[idx[l]] += 1
        s = v.sum()
        return v / s if s > 0 else v

    # ---- training -----------------------------------------------------------
    def fit(self, frames_labels: List[List[str]], seed: int = 0) -> "SceneContextModel":
        """frames_labels: list over frames of the list of labels present in that frame."""
        from sklearn.cluster import KMeans
        if not self.vocab:
            counts = Counter(l for fl in frames_labels for l in fl)
            self.vocab = sorted(counts)
        X = np.stack([self._hist(frames_labels[max(0, i - self.window):i + 1]) for i in range(len(frames_labels))])
        self.kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=seed).fit(X)
        assign = self.kmeans.labels_
        freq = np.zeros((self.k, len(self.vocab)), dtype=np.float32)
        for c in range(self.k):
            member = X[assign == c]
            if len(member):
                freq[c] = member.mean(0)
        self.cluster_class_freq = freq
        self.cluster_names = []
        for c in range(self.k):
            top = np.argsort(-freq[c])[:3]
            self.cluster_names.append("/".join(self.vocab[i] for i in top if freq[c, i] > 0) or f"cluster{c}")
        self.inertia_ = float(self.kmeans.inertia_)
        return self

    # ---- online use ---------------------------------------------------------
    def observe(self, tracks: List[Track], t: float) -> Dict:
        self._buffer.append([tr.label for tr in tracks])
        if self.kmeans is None:
            return {"cluster": None, "changed": False}
        h = self._hist(list(self._buffer))
        c = int(self.kmeans.predict(h[None])[0])
        changed = self._last_cluster is not None and c != self._last_cluster
        if changed:
            self._last_change_t = t
        self._last_cluster = c
        return {"cluster": c, "cluster_name": self.cluster_names[c], "changed": changed,
                "since_change": t - self._last_change_t}

    def surprise(self, label: str, cluster: int) -> float:
        if self.cluster_class_freq is None or label not in self.vocab:
            return 0.5
        f = float(self.cluster_class_freq[cluster, self.vocab.index(label)])
        return 1.0 - min(1.0, f / (self.cluster_class_freq[cluster].max() + 1e-6))

    def modulate(self, label: str, base_relevance: float, ctx: Dict) -> float:
        c = ctx.get("cluster")
        if c is None:
            return base_relevance
        s = self.surprise(label, c)
        r = base_relevance * (1.0 + self.surprise_gain * (s - 0.5))
        return float(max(0.0, min(1.0, r)))

    def reset(self) -> None:
        self._buffer.clear(); self._last_cluster = None; self._last_change_t = -1e9

    # ---- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SceneContextModel":
        import joblib
        return joblib.load(path)

    def summary(self) -> Dict:
        return {"k": self.k, "vocab_size": len(self.vocab), "clusters": self.cluster_names,
                "inertia": getattr(self, "inertia_", None)}


def frames_from_scenarios(scenarios, cfg=None) -> List[List[str]]:
    """Collect per-frame label lists from simulated scenarios (for unsupervised training)."""
    from ..config import PipelineConfig
    from ..sim.scenario import DetectorSimulator
    from ..tracking.tracker import IoUTracker
    cfg = cfg or PipelineConfig()
    out = []
    for sc in scenarios:
        tr = IoUTracker(cfg.tracker)
        for t, dets in DetectorSimulator(sc).frames():
            out.append([x.label for x in tr.update(dets, t)])
    return out
