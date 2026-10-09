"""EgoBlindScenario: one EgoBlind video with its weak events, duck-typed to CampusScenario so that
`run_on_*`, `campus.metrics`, `CampusOracle` and the scheduler trainers work on it unchanged."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..campus.world import LABEL_EVENT, EventAnnotation
from ..cues.compose import compose
from ..geometry.camera import CameraModel
from .cache import PerceptionCache, EgoNoise
from .data import Question, events_for_video, LOOKBACK_S, IDEAL_BEFORE_S, LATEST_AFTER_S, SUDDEN_S, coverability, detector_labels


def _kind(events: Sequence[EventAnnotation]) -> str:
    speak = [e for e in events if e.should_speak]
    if any(e.detail.get("qtype") == "safety" for e in speak):
        return "egoblind_safety"
    if speak:
        return "egoblind_navigation"
    if any(e.reason == "user_confirmed_clear" for e in events):
        return "egoblind_quiet"
    return "egoblind_query"


@dataclass
class EgoBlindScenario:
    name: str
    video_id: int
    split: str
    kind: str
    duration: float
    fps: float
    seed: int
    video: str = ""
    goal_tags: List[str] = field(default_factory=list)
    events_list: List[EventAnnotation] = field(default_factory=list)
    queries: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    cache_path: Optional[str] = None
    description: str = ""
    camera: CameraModel = field(default_factory=CameraModel)
    _cache: Optional[PerceptionCache] = field(default=None, repr=False, compare=False)

    # --- CampusScenario interface ------------------------------------------------
    def events(self) -> List[EventAnnotation]:
        return self.events_list

    def speak_events(self) -> List[EventAnnotation]:
        return [e for e in self.events_list if e.should_speak]

    def quiet_events(self) -> List[EventAnnotation]:
        return [e for e in self.events_list if e.reason == "user_confirmed_clear"]

    def event_for(self, obj_id: Optional[int]) -> Optional[EventAnnotation]:
        if obj_id is None:
            return None
        return next((e for e in self.events_list if e.obj_id == obj_id), None)

    @property
    def question_times(self) -> List[float]:
        return sorted({e.detail["t_q"] for e in self.events_list} | {q["t_q"] for q in self.queries})

    NEAR_QUESTION_S = 6.0

    @property
    def labelled_times(self) -> List[float]:
        """Moments with hazard ground truth: the safety / navigation questions that became a should-speak event or a quiet
        window. Information, tool-use and communication questions say nothing about hazards and are excluded."""
        return sorted({e.detail["t_q"] for e in self.events_list})

    def counts_fp(self, t: float) -> bool:
        """An off-event cue is a false alarm for the communication score (and the REINFORCE reward) only within
        NEAR_QUESTION_S of a labelled moment; elsewhere nobody knows whether it was useful."""
        qt = getattr(self, "_qtimes", None)
        if qt is None:
            qt = self._qtimes = self.labelled_times
        return any(abs(t - q) <= self.NEAR_QUESTION_S for q in qt)

    # --- perception cache -------------------------------------------------------------
    def has_cache(self) -> bool:
        return bool(self.cache_path) and Path(self.cache_path).exists()

    def cache(self) -> PerceptionCache:
        if self._cache is None:
            if not self.has_cache():
                raise FileNotFoundError(f"no perception cache for {self.name} ({self.cache_path}); run scripts/cache_egoblind_perception.py")
            self._cache = PerceptionCache.load(self.cache_path)
        return self._cache

    def attach_cache(self, cache: PerceptionCache) -> None:
        self._cache = cache

    def frames(self, noise: Optional[EgoNoise] = None, rng=None):
        return self.cache().frames(self.speak_events(), noise=noise, rng=rng)

    # --- serialisation ---------------------------------------------------------
    def to_json(self, path) -> None:
        d = {"name": self.name, "video_id": self.video_id, "split": self.split, "kind": self.kind, "duration": self.duration,
             "fps": self.fps, "seed": self.seed, "video": self.video, "goal_tags": self.goal_tags, "description": self.description,
             "cache_path": self.cache_path, "stats": self.stats, "queries": self.queries,
             "events": [e.as_dict() for e in self.events_list]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def from_json(path, cache_dir=None) -> "EgoBlindScenario":
        d = json.loads(Path(path).read_text())
        events = []
        for e in d["events"]:
            e = dict(e); e["minimum_information"] = tuple(e.get("minimum_information", ()))
            events.append(EventAnnotation(**e))
        cache_path = d.get("cache_path")
        if cache_dir is not None:
            cache_path = str(Path(cache_dir) / f"{d['video_id']}.npz")
        return EgoBlindScenario(name=d["name"], video_id=d["video_id"], split=d["split"], kind=d["kind"], duration=d["duration"],
                                fps=d["fps"], seed=d["seed"], video=d.get("video", ""), goal_tags=d.get("goal_tags", []),
                                events_list=events, queries=d.get("queries", []), stats=d.get("stats", {}), cache_path=cache_path,
                                description=d.get("description", ""))


def build_scenario(video_id: int, questions: Sequence[Question], split: str, video: str = "", cache: Optional[PerceptionCache] = None,
                   cache_path: Optional[str] = None) -> EgoBlindScenario:
    events, queries, stats = events_for_video(questions)
    duration = max(5.0, max(q.t_q for q in questions) + 3.0) if questions else 5.0
    sc = EgoBlindScenario(name=f"egoblind_{video_id}", video_id=video_id, split=split, kind=_kind(events), duration=duration,
                          fps=5.0, seed=int(video_id), video=video, events_list=events, queries=queries, stats=stats,
                          cache_path=cache_path, description=f"EgoBlind video {video_id} ({split}), {len(questions)} questions")
    if cache is not None:
        sc.attach_cache(cache)
        refine_with_cache(sc, cache)
    return sc


def refine_with_cache(sc: EgoBlindScenario, cache: PerceptionCache, min_conf: float = 0.35) -> Dict:
    """Use the cached detections to (1) set the real duration, (2) open each window at the first detection of a
    matching label before t_q, (3) mark events the detector never saw, (4) render the ideal cue with the observed
    distance / direction. Returns detection statistics."""
    sc.duration = float(cache.duration)
    sc.fps = float(cache.meta.get("eff_fps", sc.fps))
    sc.cache_path = sc.cache_path or cache.meta.get("path")
    n_det = 0
    vocab = list(cache.meta.get("vocab") or [])          # open-vocabulary caches name objects differently from COCO
    cache_end = float(cache.duration)
    for e in sc.speak_events():
        b = e.detail.get("base")
        if b:   # start from the cache-independent state so the result does not depend on an earlier refine
            e.label, e.event_type = b["label"], b["event_type"]
            e.earliest_useful_s, e.ideal_s, e.latest_safe_s = b["window"]
            e.minimum_information, e.ideal_cue, e.sudden = tuple(b["slots"]), b["cue"], b["sudden"]
            e.distance_at_ideal_m = e.direction_at_ideal_deg = None
            e.detail.update({"labels": list(b["labels"]), "coverable": b["coverable"], "coverable_reason": b["coverable_reason"],
                             "detected": None, "first_seen_s": None})
            for k in ("det_labels", "vocab_extended"):
                e.detail.pop(k, None)
        labels = list(e.detail.get("labels", []))
        if vocab:
            labels = detector_labels(labels, e.detail.get("concepts", []), vocab)
            coverable, why, primary = coverability(labels, e.detail.get("concepts", []))
            e.detail.update({"det_labels": labels, "coverable": coverable, "coverable_reason": why})
            if coverable and primary != e.label:
                e.label = primary
                e.event_type = LABEL_EVENT.get(primary, e.event_type)
        t_q = e.detail["t_q"]
        if e.detail.get("coverable") and t_q - LOOKBACK_S > cache_end:
            # the cache stops before this question (truncated decode): nothing can be said about the event
            e.detail.update({"coverable": False, "coverable_reason": "cache_truncated"})
        if not labels or not e.detail.get("coverable"):
            e.detail["detected"] = None
            continue
        idx = cache.detections_of_label(labels, t_q - LOOKBACK_S, t_q + LATEST_AFTER_S, min_conf)
        if len(idx) == 0:
            e.detail["detected"] = False
            continue
        n_det += 1
        times = cache.frame_t[cache.det_frame[idx]]
        first = float(times.min())
        e.detail["detected"] = True
        e.detail["first_seen_s"] = round(first, 2)
        e.earliest_useful_s = round(max(first, t_q - LOOKBACK_S), 2)
        e.latest_safe_s = round(max(t_q + LATEST_AFTER_S, e.earliest_useful_s + 0.3), 2)
        e.ideal_s = round(min(max(t_q - IDEAL_BEFORE_S, e.earliest_useful_s), e.latest_safe_s), 2)
        e.sudden = bool(t_q < SUDDEN_S or (e.latest_safe_s - e.earliest_useful_s) < 1.0)
        # state near the ideal moment: the closest matching detection within +-0.75 s
        near = idx[np.abs(times - e.ideal_s) <= 0.75]
        if len(near) == 0:
            near = idx
        dists = cache.det_dist[near]
        ok = np.isfinite(dists)
        j = near[ok][np.argmin(dists[ok])] if ok.any() else near[0]
        z = float(cache.det_dist[j]) if np.isfinite(cache.det_dist[j]) else None
        x1, _, x2, _ = cache.det_bbox[j]
        angle = sc.camera.angle_deg_from_u(0.5 * float(x1 + x2))
        e.distance_at_ideal_m = None if z is None else round(z, 2)
        e.direction_at_ideal_deg = round(angle, 1)
        slots = list(e.minimum_information)
        if abs(angle) > 12 and "direction" not in slots:
            slots.append("direction"); e.minimum_information = tuple(slots)
        e.ideal_cue = compose(slots, e.label, "object", None, distance_m=z, depth_conf=float(cache.det_depth_conf[j]) if z is not None else 0.0,
                              angle_deg=angle)
    cover = [e for e in sc.speak_events() if e.detail.get("coverable")]
    sc.stats["detected"] = n_det
    sc.stats["coverable_total"] = len(cover)
    return {"detected": n_det, "coverable": len(cover)}
