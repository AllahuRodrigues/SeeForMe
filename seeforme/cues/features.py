"""Compute the CACS feature vector (C, R, N, P, T, D) for every active track."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..config import CACSConfig
from ..types import CueFeatures, Observation, Track, bbox_area, bbox_center
from .relevance import relevance_prior
from .text_importance import KeywordTextImportance


EXTRA_NAMES = ["cx", "cy", "size", "growth", "age", "since_announced", "announce_count"]


def extra_features(track: Track, t: float) -> List[float]:
    """Motion and history signals beyond the six CACS features (all roughly in [-1, 1])."""
    cx, cy = bbox_center(track.bbox)
    size = math.sqrt(bbox_area(track.bbox))
    growth = max(-1.0, min(1.0, track.area_growth_rate()))
    age = min(1.0, (t - track.first_seen) / 10.0)
    since = 1.0 if track.last_announced is None else min(1.0, (t - track.last_announced) / 10.0)
    count = min(1.0, track.announce_count / 3.0)
    return [cx, cy, size, growth, age, since, count]


class FeatureExtractor:
    def __init__(self, cfg: CACSConfig, text_model=None, scene_context=None):
        self.cfg = cfg
        self.text_model = text_model or KeywordTextImportance()
        self.scene_context = scene_context  # optional unsupervised context model
        self.label_last_announced: Dict[str, float] = {}

    # ---- individual signals -------------------------------------------------
    @staticmethod
    def proximity(track: Track) -> float:
        """Navigation importance from geometry: big, low, central boxes are close and in the path."""
        x1, y1, x2, y2 = track.bbox
        area = bbox_area(track.bbox)
        cx, _ = bbox_center(track.bbox)
        size_term = min(1.0, math.sqrt(area) / 0.6)         # sqrt(area)=0.6 -> fills most of frame
        bottom_term = min(1.0, max(0.0, (y2 - 0.35) / 0.65))  # bottom edge low in frame -> near
        center_term = 1.0 - min(1.0, abs(cx - 0.5) * 2.0)    # in the walking path
        approach = max(0.0, min(1.0, track.area_growth_rate()))  # growing -> approaching
        p = 0.4 * size_term + 0.3 * bottom_term + 0.2 * center_term + 0.1 * approach
        return max(0.0, min(1.0, p))

    def novelty(self, track: Track, t: float) -> float:
        if track.last_announced is None:
            return 1.0
        n = 0.0
        moved = track.displacement_since_announced()
        if moved >= self.cfg.move_reset:
            n = max(n, min(1.0, moved / (2 * self.cfg.move_reset)))
        if track.area_growth_rate() > 0.15:
            n = max(n, self.cfg.approach_boost)
        # object re-appearing after a long absence from speech is somewhat novel again
        n = max(n, 1.0 - math.exp(-(t - track.last_announced) / (3 * self.cfg.duplicate_tau)))
        return n / (1 + 0.5 * track.announce_count)

    def duplicate(self, track: Track, t: float) -> float:
        d = 0.0
        if track.last_announced is not None:
            d = math.exp(-(t - track.last_announced) / self.cfg.duplicate_tau)
        last_label = self.label_last_announced.get(track.label)
        if last_label is not None and last_label != track.last_announced:
            # a *different* track with the same label was announced recently
            d = max(d, 0.6 * math.exp(-(t - last_label) / self.cfg.label_duplicate_tau))
        return d

    def relevance(self, track: Track, scene_ctx: Optional[dict]) -> float:
        base = relevance_prior(track.label) if track.kind == "object" else 0.3
        if self.scene_context is not None and scene_ctx is not None and self.cfg.use_scene_context:
            base = self.scene_context.modulate(track.label, base, scene_ctx)
        return base

    # ---- assemble -----------------------------------------------------------
    def extract(self, tracks: List[Track], t: float, scene_ctx: Optional[dict] = None) -> List[Observation]:
        obs = []
        for trk in tracks:
            f = CueFeatures(
                confidence=trk.confidence,
                relevance=self.relevance(trk, scene_ctx),
                novelty=self.novelty(trk, t),
                proximity=self.proximity(trk),
                text_importance=self.text_model.score(trk.text) if trk.kind == "text" else 0.0,
                duplicate=self.duplicate(trk, t),
            )
            obs.append(Observation(track=trk, t=t, features=f, extras=extra_features(trk, t)))
        return obs

    def note_announced(self, track: Track, t: float) -> None:
        track.last_announced = t
        track.last_announced_bbox = track.bbox
        track.announce_count += 1
        self.label_last_announced[track.label] = t

    def reset(self) -> None:
        self.label_last_announced.clear()
