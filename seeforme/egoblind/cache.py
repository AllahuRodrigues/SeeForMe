"""Perception cache: run the detector and the metric depth model once per EgoBlind video and store
the per-frame detections, so that schedulers can be replayed, trained (DAgger, REINFORCE) and
evaluated thousands of times at tracker + policy speed.

One .npz per video:
    meta            JSON: video, fps, size, stride, effective fps, duration processed, models, labels, version
    frame_t         (F,)   timestamp of each processed frame
    frame_quality   (F, 2) blur score, mean brightness (degraded-capture diagnostics)
    det_frame       (N,)   index into frames
    det_label       (N,)   index into meta["labels"]
    det_conf        (N,)
    det_bbox        (N, 4) normalised x1, y1, x2, y2
    det_dist        (N,)   metric distance (m), NaN when depth was unavailable
    det_depth_conf  (N,)

`PerceptionCache.frames(events)` rebuilds `Detection` lists and assigns `source_id`s from the weak
events (a detection whose label the event names, inside the event's window plus a margin), so the
CampusBench oracle, metrics and trainers work unchanged. `EgoNoise` perturbs the replay (missed
detections, confidence and depth noise) for counterfactual robustness tests and training diversity.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Detection

CACHE_VERSION = 1
SOURCE_MARGIN_S = 2.0


@dataclass
class EgoNoise:
    """Replay perturbation (all off by default)."""
    p_miss: float = 0.0          # probability of dropping a detection
    conf_sd: float = 0.0         # gaussian noise on confidence
    depth_rel: float = 0.0       # relative gaussian noise on distance
    p_depth_gross: float = 0.0   # probability of a gross depth error (x0.5 .. x1.8)
    p_blackout: float = 0.0      # probability that a whole frame is lost (camera covered / pointed away)


class PerceptionCache:
    def __init__(self, meta: Dict, frame_t: np.ndarray, frame_quality: np.ndarray, det_frame: np.ndarray, det_label: np.ndarray,
                 det_conf: np.ndarray, det_bbox: np.ndarray, det_dist: np.ndarray, det_depth_conf: np.ndarray):
        self.meta = meta
        self.frame_t = np.asarray(frame_t, dtype=np.float64)
        self.frame_quality = np.asarray(frame_quality, dtype=np.float32).reshape(-1, 2)
        self.det_frame = np.asarray(det_frame, dtype=np.int64)
        self.det_label = np.asarray(det_label, dtype=np.int64)
        self.det_conf = np.asarray(det_conf, dtype=np.float32)
        self.det_bbox = np.asarray(det_bbox, dtype=np.float32).reshape(-1, 4)
        self.det_dist = np.asarray(det_dist, dtype=np.float32)
        self.det_depth_conf = np.asarray(det_depth_conf, dtype=np.float32)
        self.labels: List[str] = list(meta.get("labels", []))
        # frame index -> slice of detections (detections are stored frame-ordered)
        self._starts = np.searchsorted(self.det_frame, np.arange(len(self.frame_t) + 1))

    # ---- persistence --------------------------------------------------------
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=json.dumps(self.meta), frame_t=self.frame_t, frame_quality=self.frame_quality,
                            det_frame=self.det_frame, det_label=self.det_label, det_conf=self.det_conf, det_bbox=self.det_bbox,
                            det_dist=self.det_dist, det_depth_conf=self.det_depth_conf)

    @staticmethod
    def load(path) -> "PerceptionCache":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return PerceptionCache(meta, z["frame_t"], z["frame_quality"], z["det_frame"], z["det_label"], z["det_conf"],
                                   z["det_bbox"], z["det_dist"], z["det_depth_conf"])

    # ---- properties -----------------------------------------------------------
    @property
    def duration(self) -> float:
        return float(self.meta.get("duration", self.frame_t[-1] + 1.0 / max(1e-6, self.meta.get("eff_fps", 5.0)) if len(self.frame_t) else 0.0))

    @property
    def n_frames(self) -> int:
        return len(self.frame_t)

    @property
    def n_detections(self) -> int:
        return len(self.det_frame)

    def label_of(self, i: int) -> str:
        return self.labels[int(self.det_label[i])]

    def detections_of_label(self, labels: Sequence[str], t0: float = -math.inf, t1: float = math.inf, min_conf: float = 0.0):
        """Indices of detections with one of `labels` between t0 and t1."""
        ids = {self.labels.index(l) for l in labels if l in self.labels}
        if not ids:
            return np.zeros(0, dtype=np.int64)
        t = self.frame_t[self.det_frame]
        mask = np.isin(self.det_label, list(ids)) & (t >= t0) & (t <= t1) & (self.det_conf >= min_conf)
        return np.nonzero(mask)[0]

    # ---- replay --------------------------------------------------------------
    def frames(self, events: Optional[Sequence] = None, margin: float = SOURCE_MARGIN_S, noise: Optional[EgoNoise] = None,
               rng: Optional[random.Random] = None) -> List[Tuple[float, List[Detection]]]:
        """(t, detections) per processed frame; `events` (should-speak weak events) give the source ids."""
        ev = [e for e in (events or []) if getattr(e, "should_speak", False)]
        by_label: Dict[str, List] = {}
        for e in ev:
            for lab in e.detail.get("det_labels") or e.detail.get("labels", []):
                by_label.setdefault(lab, []).append(e)
        rng = rng or random.Random(0)
        out: List[Tuple[float, List[Detection]]] = []
        for fi, t in enumerate(self.frame_t):
            t = float(t)
            if noise is not None and noise.p_blackout > 0 and rng.random() < noise.p_blackout:
                out.append((t, []))
                continue
            dets: List[Detection] = []
            for i in range(int(self._starts[fi]), int(self._starts[fi + 1])):
                if noise is not None and noise.p_miss > 0 and rng.random() < noise.p_miss:
                    continue
                lab = self.labels[int(self.det_label[i])]
                conf = float(self.det_conf[i])
                dist = float(self.det_dist[i]); dconf = float(self.det_depth_conf[i])
                if not math.isfinite(dist) or dconf <= 0:
                    dist, dconf = None, 0.0
                if noise is not None:
                    if noise.conf_sd > 0:
                        conf = min(1.0, max(0.0, conf + rng.gauss(0.0, noise.conf_sd)))
                    if dist is not None and noise.p_depth_gross > 0 and rng.random() < noise.p_depth_gross:
                        dist *= rng.uniform(0.5, 1.8)
                    elif dist is not None and noise.depth_rel > 0:
                        dist = max(0.2, dist * (1.0 + rng.gauss(0.0, noise.depth_rel)))
                sid = None
                cands = [e for e in by_label.get(lab, []) if e.earliest_useful_s - margin <= t <= e.latest_safe_s + margin]
                if cands:
                    sid = min(cands, key=lambda e: abs(e.detail.get("t_q", e.latest_safe_s) - t)).obj_id
                x1, y1, x2, y2 = (float(v) for v in self.det_bbox[i])
                dets.append(Detection(label=lab, confidence=conf, bbox=(x1, y1, x2, y2), kind="object", source_id=sid,
                                      distance_m=dist, depth_conf=dconf))
            out.append((t, dets))
        return out


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def build_cache(video_path, detector, depth=None, target_fps: float = 5.0, frame_width: int = 640, max_seconds: Optional[float] = None,
                progress=None, depth_name: str = "", detector_name: str = "") -> PerceptionCache:
    """Decode `video_path`, run `detector` (+ `depth`) every `stride` frames and return the cache."""
    import cv2
    from ..perception.preprocess import assess_quality, enhance_dark, resize_for_inference
    from ..perception.depth import object_distances
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not math.isfinite(fps) or fps <= 1:
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(round(fps / target_fps)))
    labels: List[str] = []; lab_idx: Dict[str, int] = {}
    frame_t, quality, det_frame, det_label, det_conf, det_bbox, det_dist, det_dconf = [], [], [], [], [], [], [], []
    i = 0; t0 = time.perf_counter(); last_t = 0.0
    failures = streak = 0          # unreadable frames in total / in a row (a corrupt frame is skipped, the end of the stream stops)
    while True:
        if i % stride:
            if cap.grab():
                streak = 0
            else:
                failures += 1; streak += 1
                if streak > 30 or (n_total and i >= n_total):
                    break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            failures += 1; streak += 1
            if streak > 30 or (n_total and i >= n_total):
                break
            i += 1
            continue
        streak = 0
        t = i / fps
        if max_seconds is not None and t > max_seconds:
            break
        small = resize_for_inference(frame, frame_width)
        q = assess_quality(small)
        if q.is_dark:
            small = enhance_dark(small)
        dets = detector.detect(small)
        if q.confidence_scale < 1.0:
            for d in dets:
                d.confidence *= q.confidence_scale
        if depth is not None and dets:
            object_distances(depth(small), dets)
        fi = len(frame_t)
        frame_t.append(t); quality.append((q.blur_score, q.brightness)); last_t = t
        for d in dets:
            if d.label not in lab_idx:
                lab_idx[d.label] = len(labels); labels.append(d.label)
            det_frame.append(fi); det_label.append(lab_idx[d.label]); det_conf.append(d.confidence); det_bbox.append(d.bbox)
            det_dist.append(float("nan") if d.distance_m is None else d.distance_m); det_dconf.append(d.depth_conf)
        if progress is not None and fi % 50 == 0:
            progress(fi, t)
        i += 1
    cap.release()
    meta = {"version": CACHE_VERSION, "video": str(video_path), "fps": fps, "width": w, "height": h, "n_frames_total": n_total,
            "stride": stride, "eff_fps": fps / stride, "frame_width": frame_width, "duration": last_t + stride / fps,
            "max_seconds": max_seconds, "detector": detector_name or type(detector).__name__, "depth": depth_name or (type(depth).__name__ if depth else ""),
            "labels": labels, "vocab": list(getattr(detector, "vocab_labels", []) or []), "read_failures": failures,
            "truncated": bool(n_total and fps and (max_seconds is None or last_t + 1.0 < max_seconds) and last_t + 1.0 < n_total / fps - 1.0),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - t0, 1)}
    return PerceptionCache(meta, np.array(frame_t), np.array(quality, dtype=np.float32).reshape(-1, 2), np.array(det_frame, dtype=np.int64),
                           np.array(det_label, dtype=np.int64), np.array(det_conf, dtype=np.float32), np.array(det_bbox, dtype=np.float32).reshape(-1, 4),
                           np.array(det_dist, dtype=np.float32), np.array(det_dconf, dtype=np.float32))


def synthetic_cache(tracks: Sequence[Dict], duration: float, eff_fps: float = 5.0, meta: Optional[Dict] = None) -> PerceptionCache:
    """Cache from scripted tracks (tests / examples). Each track: label, t0, t1, bbox(t) callable or fixed, conf, dist(t) callable or fixed."""
    frame_t = np.arange(0.0, duration, 1.0 / eff_fps)
    labels: List[str] = []; lab_idx: Dict[str, int] = {}
    det_frame, det_label, det_conf, det_bbox, det_dist, det_dconf = [], [], [], [], [], []
    for fi, t in enumerate(frame_t):
        for tr in tracks:
            if not (tr["t0"] <= t <= tr["t1"]):
                continue
            lab = tr["label"]
            if lab not in lab_idx:
                lab_idx[lab] = len(labels); labels.append(lab)
            bbox = tr["bbox"](t) if callable(tr["bbox"]) else tr["bbox"]
            dist = tr.get("dist", None)
            dist = dist(t) if callable(dist) else dist
            det_frame.append(fi); det_label.append(lab_idx[lab]); det_conf.append(tr.get("conf", 0.9)); det_bbox.append(bbox)
            det_dist.append(float("nan") if dist is None else dist); det_dconf.append(tr.get("depth_conf", 0.9) if dist is not None else 0.0)
    m = {"version": CACHE_VERSION, "video": "synthetic", "fps": eff_fps, "stride": 1, "eff_fps": eff_fps, "duration": float(duration),
         "labels": labels, "detector": "synthetic", "depth": "synthetic"}
    m.update(meta or {})
    return PerceptionCache(m, frame_t, np.zeros((len(frame_t), 2), dtype=np.float32), np.array(det_frame, dtype=np.int64),
                           np.array(det_label, dtype=np.int64), np.array(det_conf, dtype=np.float32), np.array(det_bbox, dtype=np.float32).reshape(-1, 4),
                           np.array(det_dist, dtype=np.float32), np.array(det_dconf, dtype=np.float32))
