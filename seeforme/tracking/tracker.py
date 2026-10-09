"""Lightweight multi-object tracker providing temporal memory for cue selection.

Detections are associated with existing tracks by IoU (Hungarian assignment via
scipy when available, greedy otherwise). Each track keeps the state described in
section 4.3 of the proposal so that CACS can decide whether an object is new,
whether it moved, and whether it was recently announced.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np

from ..config import TrackerConfig
from ..types import Detection, Track, bbox_center, bbox_iou

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _text_sim(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class IoUTracker:
    def __init__(self, cfg: Optional[TrackerConfig] = None, dynamics=None):
        self.cfg = cfg or TrackerConfig()
        self.dynamics = dynamics          # optional geometry.TrackDynamics (distance / TTC / path probability)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = 0

    # ------------------------------------------------------------------
    def _affinity(self, det: Detection, trk: Track) -> float:
        if det.kind != trk.kind:
            return 0.0
        iou = bbox_iou(det.bbox, trk.bbox)
        if det.kind == "text":
            sim = _text_sim(det.text, trk.text)
            # text boxes jitter a lot between OCR passes; let string similarity help
            return 0.5 * iou + 0.5 * sim if sim >= self.cfg.text_similarity else iou
        if det.label != trk.label:
            # allow re-association on strong overlap but discourage label switching
            return 0.5 * iou
        return iou

    def update(self, detections: List[Detection], t: float) -> List[Track]:
        self.frame_index += 1
        track_ids = list(self.tracks.keys())
        n_t, n_d = len(track_ids), len(detections)
        matched_t, matched_d = set(), set()

        if n_t and n_d:
            aff = np.zeros((n_t, n_d), dtype=float)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    aff[i, j] = self._affinity(det, self.tracks[tid])
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(-aff)
                pairs = list(zip(rows, cols))
            else:  # greedy fallback
                pairs = []
                order = np.dstack(np.unravel_index(np.argsort(-aff, axis=None), aff.shape))[0]
                used_r, used_c = set(), set()
                for r, c in order:
                    if r in used_r or c in used_c:
                        continue
                    pairs.append((r, c))
                    used_r.add(r); used_c.add(c)
            for r, c in pairs:
                if aff[r, c] >= self.cfg.iou_threshold:
                    self._update_track(self.tracks[track_ids[r]], detections[c], t)
                    matched_t.add(r); matched_d.add(c)

        # second pass: a reappearing object may no longer overlap the box a remembered track last had
        if self.cfg.memory_ttl_s > 0:
            for j, det in enumerate(detections):
                if j in matched_d:
                    continue
                best, best_d = None, 0.25
                for i, tid in enumerate(track_ids):
                    trk = self.tracks[tid]
                    if i in matched_t or not trk.remembered or trk.label != det.label or trk.kind != det.kind:
                        continue
                    (ax, ay), (bx, by) = bbox_center(det.bbox), bbox_center(trk.bbox)
                    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                    if d < best_d:
                        best, best_d = i, d
                if best is not None:
                    self._update_track(self.tracks[track_ids[best]], det, t)
                    matched_t.add(best); matched_d.add(j)

        for j, det in enumerate(detections):
            if j not in matched_d:
                self._create_track(det, t)

        for i, tid in enumerate(track_ids):
            if i not in matched_t:
                trk = self.tracks[tid]
                trk.misses += 1
                trk.age += 1
                if trk.misses > self.cfg.max_age:
                    if self.cfg.memory_ttl_s > 0 and trk.confirmed and (t - trk.last_seen) <= self.cfg.memory_ttl_s \
                            and self._remember(trk, t):
                        continue
                    del self.tracks[tid]

        return self.active_tracks()

    def _remember(self, trk: Track, t: float) -> bool:
        """Memory ledger step: decay the evidence and extrapolate the range as if the user kept walking.
        Returns False once the remembered object would have been reached (the user walked past it)."""
        t_prev = trk.memory_t if trk.memory_t is not None else trk.last_seen
        dt = max(0.0, t - t_prev)
        trk.remembered = True
        trk.memory_t = t
        decay = self.cfg.memory_decay ** dt
        trk.confidence *= decay
        trk.depth_conf *= decay
        if trk.distance_m is not None:
            trk.distance_m = trk.distance_m - self.cfg.memory_user_speed * dt
            if trk.distance_m < 0.5:
                return False
            trk.dist_history.append((t, trk.distance_m, trk.lateral_m))
            if self.dynamics is not None:
                self.dynamics._ttc_and_path(trk)
        return True

    # ------------------------------------------------------------------
    def _update_track(self, trk: Track, det: Detection, t: float) -> None:
        a = self.cfg.conf_ema
        trk.confidence = a * det.confidence + (1 - a) * trk.confidence
        trk.bbox = det.bbox
        trk.bbox_history.append((t, det.bbox))
        trk.last_seen = t
        trk.age += 1
        trk.hits += 1
        trk.misses = 0
        trk.remembered = False
        trk.memory_t = None
        if det.kind == "text" and det.text:
            # keep the higher-confidence reading of the string
            if trk.text is None or det.confidence >= trk.confidence:
                trk.text = det.text
        if det.label != trk.label and det.confidence > trk.confidence:
            trk.label = det.label
        if det.source_id is not None:
            trk.source_votes[det.source_id] = trk.source_votes.get(det.source_id, 0) + 1
        if trk.hits >= self.cfg.n_init:
            trk.confirmed = True
        if self.dynamics is not None:
            self.dynamics.update(trk, det, t)

    def _create_track(self, det: Detection, t: float) -> Track:
        trk = Track(
            track_id=self._next_id, label=det.label, kind=det.kind, bbox=det.bbox,
            confidence=det.confidence, first_seen=t, last_seen=t, text=det.text,
            confirmed=self.cfg.n_init <= 1,
        )
        trk.bbox_history.append((t, det.bbox))
        if det.source_id is not None:
            trk.source_votes[det.source_id] = 1
        self.tracks[trk.track_id] = trk
        self._next_id += 1
        if self.dynamics is not None:
            self.dynamics.update(trk, det, t)
        return trk

    def active_tracks(self) -> List[Track]:
        """Tracks that were matched in the current frame and are confirmed, plus remembered ones when the ledger is on."""
        return [t for t in self.tracks.values() if (t.misses == 0 or t.remembered) and t.confirmed]

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.frame_index = 0
