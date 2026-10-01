"""Context-Aware Cue Selection (CACS).

For every candidate observation i the policy computes

    S_i = alpha*C_i + beta*R_i + gamma*N_i + delta*P_i + eta*T_i - lambda*D_i

and speaks the highest-scoring observations that exceed a decision threshold.
An uncertainty gate suppresses low-confidence predictions entirely and hedges
("possibly a chair") predictions in the uncertain band. Feature toggles allow
the ablation study to switch individual signals off.

`LearnedCACS` replaces the linear weighting with a supervised model trained on
the same six features (see ml/priority_model.py).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from ..config import CACSConfig
from ..types import Cue, CueFeatures, Observation, Track
from .features import FeatureExtractor
from .messages import compose_message
from .policy import CuePolicy


class CACSPolicy(CuePolicy):
    name = "cacs"

    def __init__(self, cfg: Optional[CACSConfig] = None, text_model=None, scene_context=None):
        self.cfg = cfg or CACSConfig()
        self.decision_interval = self.cfg.decision_interval
        self.features = FeatureExtractor(self.cfg, text_model=text_model, scene_context=scene_context)
        self._last_obs: List[Observation] = []

    # ------------------------------------------------------------------
    def _apply_ablation(self, f: CueFeatures) -> CueFeatures:
        c = self.cfg
        return replace(
            f,
            novelty=f.novelty if c.use_novelty else 1.0,
            proximity=f.proximity if c.use_proximity else 0.5,
            relevance=f.relevance if c.use_relevance else 0.5,
            text_importance=f.text_importance if c.use_text else (0.5 if f.text_importance > 0 else 0.0),
            duplicate=f.duplicate if c.use_temporal_memory else 0.0,
        )

    def score(self, f: CueFeatures) -> float:
        w = self.cfg.weights
        return (w.alpha * f.confidence + w.beta * f.relevance + w.gamma * f.novelty
                + w.delta * f.proximity + w.eta * f.text_importance - w.lam * f.duplicate)

    def score_batch(self, obs: List[Observation]) -> List[float]:
        """Score all candidates of one decision at once (learned models override this)."""
        return [self.score(o.features) for o in obs]

    def _gate(self, track: Track) -> Optional[bool]:
        """Returns None if suppressed, True if hedged, False if confident."""
        if not self.cfg.use_uncertainty:
            return False
        if track.confidence < self.cfg.min_confidence:
            return None
        return track.confidence < self.cfg.sure_confidence

    # ------------------------------------------------------------------
    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        if not self.cfg.use_temporal_memory:
            # without temporal memory every decision sees fresh objects
            for tr in tracks:
                tr.last_announced = None
                tr.last_announced_bbox = None
                tr.announce_count = 0
        obs = self.features.extract(tracks, t, scene_ctx)
        for o in obs:
            o.features = self._apply_ablation(o.features)
        for o, s in zip(obs, self.score_batch(obs)):
            o.score = float(s)
        scored = sorted(obs, key=lambda o: -o.score)
        self._last_obs = scored

        cues: List[Cue] = []
        for o in scored:
            if len(cues) >= self.cfg.max_cues_per_decision:
                break
            if o.score < self.cfg.threshold:
                break
            gate = self._gate(o.track)
            if gate is None:
                continue
            self.features.note_announced(o.track, t)
            cues.append(Cue(t=t, track_id=o.track.track_id, label=o.track.label, kind=o.track.kind,
                            message=compose_message(o.track, hedged=gate), score=o.score,
                            policy=self.name, text=o.track.text, source_id=o.track.source_id,
                            hedged=gate))
        return cues

    def last_observations(self) -> List[Observation]:
        return self._last_obs

    def reset(self) -> None:
        self.features.reset()
        self._last_obs = []


class LearnedCACS(CACSPolicy):
    """CACS whose scoring function is a supervised model over the six CACS features
    (n_features=6) or over the six features plus the motion/history extras (n_features=13)."""

    name = "cacs_learned"

    def __init__(self, model, cfg: Optional[CACSConfig] = None, threshold: float = 0.5, n_features: int = 6,
                 name: Optional[str] = None, **kw):
        super().__init__(replace(cfg or CACSConfig(), threshold=threshold), **kw)
        self.model = model
        self.n_features = n_features
        if name:
            self.name = name

    def _vec(self, o: Observation) -> List[float]:
        v = o.features.as_vector()
        return v + list(o.extras) if self.n_features > 6 else v

    def score_batch(self, obs: List[Observation]) -> List[float]:
        if not obs:
            return []
        return self.model.predict_proba([self._vec(o) for o in obs])[:, 1].tolist()

    def score(self, f: CueFeatures) -> float:
        return float(self.model.predict_proba([f.as_vector()])[0, 1])
