"""End-to-end live pipeline: frame -> detection + OCR -> tracking -> cue policy -> speech."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from .config import PipelineConfig
from .cues import CACSPolicy, ConfidenceCooldownPolicy, ConfidenceThresholdPolicy, CuePolicy, LearnedCACS
from .cues.text_importance import LearnedTextImportance
from .eval.runner import SpeechChannel
from .perception.preprocess import assess_quality, enhance_dark, resize_for_inference
from .tracking.tracker import IoUTracker
from .types import Cue, Detection, Track


@dataclass
class StepResult:
    t: float
    tracks: List[Track]
    detections: List[Detection]
    cue: Optional[Cue]
    latency_ms: dict
    quality: object


@dataclass
class LatencyStats:
    detect: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    ocr: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    track: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    policy: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    end_to_end: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def summary(self) -> dict:
        import statistics as st
        def s(d):
            return {"mean_ms": round(st.mean(d), 3), "p95_ms": round(sorted(d)[int(0.95 * (len(d) - 1))], 3)} if d else {}
        return {"detect": s(self.detect), "ocr": s(self.ocr), "track": s(self.track), "policy": s(self.policy),
                "end_to_end": s(self.end_to_end)}


def make_policy(name: str, cfg: PipelineConfig, model_dir: str = "models", scene_context=None) -> CuePolicy:
    if name == "confidence_threshold":
        return ConfidenceThresholdPolicy(cfg.baseline)
    if name == "confidence_cooldown":
        return ConfidenceCooldownPolicy(cfg.baseline)
    text_model = LearnedTextImportance(f"{model_dir}/text_importance.joblib")
    if name == "cacs":
        return CACSPolicy(cfg.cacs, text_model=text_model, scene_context=scene_context)
    if name.startswith("cacs_learned"):
        import json
        from pathlib import Path
        from .ml.priority_model import load_priority_model
        which = name.split(":", 1)[1] if ":" in name else "logreg"
        rep_path = Path(model_dir) / "priority_report.json"
        thr = 0.5
        if rep_path.exists():
            thr = json.loads(rep_path.read_text())["models"].get(which, {}).get("val_threshold", 0.5)
        return LearnedCACS(load_priority_model(which, model_dir), cfg.cacs, threshold=thr, n_features=6,
                           name=f"cacs_learned:{which}", text_model=text_model, scene_context=scene_context)
    if name.startswith("cacs_dagger") or name in ("cacs_seq_il", "cacs_seq_rl"):
        import json
        from pathlib import Path
        rep = json.loads((Path(model_dir) / "deep_context_report.json").read_text())
        d = Path(model_dir) / "deep_context"
        if name.startswith("cacs_dagger"):
            import joblib
            which = name.split(":", 1)[1]
            r = rep["dagger"][which]
            return LearnedCACS(joblib.load(d / f"{which}.joblib"), cfg.cacs, threshold=r["threshold"],
                               n_features=r["n_features"], name=name, text_model=text_model, scene_context=scene_context)
        from .ml.sequence_model import SequenceCACS, SequenceModel
        if name == "cacs_seq_il":
            return SequenceCACS(SequenceModel.load(str(d / "seq_il.pt")), cfg.cacs, threshold=rep["seq_il"]["threshold"],
                                mode="imitation", name=name, text_model=text_model, scene_context=scene_context)
        return SequenceCACS(SequenceModel.load(str(d / "seq_rl.pt")), cfg.cacs, mode="rl", name=name,
                            text_model=text_model, scene_context=scene_context)
    raise ValueError(f"unknown policy {name}")


class Pipeline:
    def __init__(self, detector, ocr, policy: CuePolicy, cfg: Optional[PipelineConfig] = None,
                 speaker=None, scene_context=None):
        self.cfg = cfg or PipelineConfig()
        self.detector = detector
        self.ocr = ocr
        self.policy = policy
        self.speaker = speaker
        self.scene_context = scene_context
        self.tracker = IoUTracker(self.cfg.tracker)
        self.channel = SpeechChannel(self.cfg.speech.min_gap, self.cfg.speech.seconds_per_word)
        self.latency = LatencyStats()
        self.frame_index = 0
        self._next_decision = 0.0
        self._text_cache: List[Detection] = []
        self.cue_log: List[Cue] = []

    def process(self, frame: np.ndarray, t: Optional[float] = None) -> StepResult:
        t = time.perf_counter() if t is None else t
        t_start = time.perf_counter()
        small = resize_for_inference(frame, self.cfg.perception.frame_width)
        quality = assess_quality(small)
        if quality.is_dark:
            small = enhance_dark(small)

        t0 = time.perf_counter()
        dets = self.detector.detect(small)
        self.latency.detect.append((time.perf_counter() - t0) * 1000)
        if quality.confidence_scale < 1.0:
            for d in dets:
                d.confidence *= quality.confidence_scale

        if self.frame_index % self.cfg.perception.ocr_every_n_frames == 0:
            t0 = time.perf_counter()
            self._text_cache = self.ocr.read(small)
            self.latency.ocr.append((time.perf_counter() - t0) * 1000)
        dets = dets + self._text_cache

        t0 = time.perf_counter()
        tracks = self.tracker.update(dets, t)
        self.latency.track.append((time.perf_counter() - t0) * 1000)
        ctx = self.scene_context.observe(tracks, t) if self.scene_context is not None else None

        cue: Optional[Cue] = None
        t0 = time.perf_counter()
        if t + 1e-9 >= self._next_decision:
            self._next_decision = t + self.policy.decision_interval
            if self.channel.free(t) and (self.speaker is None or not self.speaker.busy()):
                cues = self.policy.select(tracks, t, ctx)
                if cues:
                    cue = cues[0]
                    self.channel.speak(cue)
                    self.cue_log.append(cue)
                    if self.speaker is not None:
                        self.speaker.say(cue.message)
        self.latency.policy.append((time.perf_counter() - t0) * 1000)
        self.latency.end_to_end.append((time.perf_counter() - t_start) * 1000)
        self.frame_index += 1
        return StepResult(t=t, tracks=tracks, detections=dets, cue=cue,
                          latency_ms={"detect": self.latency.detect[-1], "end_to_end": self.latency.end_to_end[-1]},
                          quality=quality)


# ---------------------------------------------------------------------------
# P-CACS live pipeline: adds metric depth, track dynamics and the four-action scheduler
# ---------------------------------------------------------------------------

def make_live_policy(name: str, cfg: PipelineConfig, model_dir: str = "models", goal_tags=None, scene_context=None) -> CuePolicy:
    from .cues.pcacs import PCACSPolicy
    text_model = LearnedTextImportance(f"{model_dir}/text_importance.joblib")
    if name == "pcacs":
        return PCACSPolicy(cfg.pcacs, text_model=text_model, goal_tags=goal_tags)
    if name.startswith("pcacs_learned"):
        import json
        from pathlib import Path
        from .ml.scheduler_model import SchedulerModel, SchedulerPolicy
        rep_path = Path(model_dir) / "scheduler_report.json"
        if not rep_path.exists():
            raise FileNotFoundError("no trained scheduler: run scripts/train_scheduler.py")
        rep = json.loads(rep_path.read_text())
        which = name.split(":", 1)[1] if ":" in name else "rl"
        d = Path(model_dir) / "scheduler"
        if which == "il":
            return SchedulerPolicy(SchedulerModel.load(str(d / "scheduler_il.pt")), cfg.pcacs, threshold=rep["il"]["threshold"],
                                   mode="imitation", name=name, goal_tags=goal_tags, text_model=text_model)
        return SchedulerPolicy(SchedulerModel.load(str(d / "scheduler_rl.pt")), cfg.pcacs, mode="rl", name=name,
                               goal_tags=goal_tags, text_model=text_model)
    return make_policy(name, cfg, model_dir, scene_context=scene_context)


class LivePipeline(Pipeline):
    """Pipeline with metric depth and the predictive scheduler. Works with the old policies too."""

    def __init__(self, detector, ocr, policy: CuePolicy, cfg: Optional[PipelineConfig] = None, speaker=None, depth=None,
                 camera=None, depth_every_n_frames: int = 1):
        super().__init__(detector, ocr, policy, cfg, speaker=speaker)
        from .geometry import CameraModel, TrackDynamics
        from .perception.depth import BBoxDepthEstimator
        self.cam = camera or CameraModel()
        self.tracker = IoUTracker(self.cfg.tracker, dynamics=TrackDynamics(self.cam))
        self.depth = depth
        self.use_depth = depth is not None
        self.depth_every = depth_every_n_frames
        self.bbox_depth = BBoxDepthEstimator(self.cam)
        self._depth_map = None
        self.latency.__dict__.setdefault("depth", deque(maxlen=200))

    def process(self, frame: np.ndarray, t: Optional[float] = None) -> StepResult:
        from .cues.pcacs import ChannelState
        from .perception.depth import object_distances
        t = time.perf_counter() if t is None else t
        t_start = time.perf_counter()
        small = resize_for_inference(frame, self.cfg.perception.frame_width)
        quality = assess_quality(small)
        if quality.is_dark:
            small = enhance_dark(small)

        t0 = time.perf_counter()
        dets = self.detector.detect(small)
        self.latency.detect.append((time.perf_counter() - t0) * 1000)
        if quality.confidence_scale < 1.0:
            for d in dets:
                d.confidence *= quality.confidence_scale

        if self.frame_index % self.cfg.perception.ocr_every_n_frames == 0:
            t0 = time.perf_counter()
            self._text_cache = self.ocr.read(small)
            self.latency.ocr.append((time.perf_counter() - t0) * 1000)
        dets = dets + self._text_cache

        t0 = time.perf_counter()
        depth_ms = 0.0
        if self.use_depth and self.depth is not None:
            if self.frame_index % self.depth_every == 0 or self._depth_map is None:
                self._depth_map = self.depth(small)
            object_distances(self._depth_map, dets)
            depth_ms = (time.perf_counter() - t0) * 1000
        else:
            self.bbox_depth.annotate(dets)
        self.latency.__dict__["depth"].append(depth_ms)

        t0 = time.perf_counter()
        tracks = self.tracker.update(dets, t)
        self.latency.track.append((time.perf_counter() - t0) * 1000)

        cue: Optional[Cue] = None
        t0 = time.perf_counter()
        if t + 1e-9 >= self._next_decision:
            self._next_decision = t + self.policy.decision_interval
            busy = (not self.channel.free(t)) or (self.speaker is not None and self.speaker.busy())
            cur = self.channel.current if busy else None
            if getattr(self.policy, "supports_actions", False):
                ch = ChannelState(busy=busy, remaining_s=self.channel.remaining(t),
                                  current_utility=(cur.utility if cur else 0.0), current_severity=(cur.severity if cur else 0.0),
                                  current_track=(cur.track_id if cur else None))
                cues = self.policy.decide(tracks, t, ch)
            else:
                cues = self.policy.select(tracks, t) if not busy else []
            if cues:
                cue = cues[0]
                self.cue_log.append(cue)
                if cue.action == "haptic":
                    print("[haptic] buzz", flush=True)
                else:
                    self.channel.speak(cue, interrupt=(cue.action == "interrupt"))
                    if self.speaker is not None:
                        if cue.action == "interrupt" and hasattr(self.speaker, "interrupt"):
                            self.speaker.interrupt(cue.message)
                        else:
                            if cue.action == "interrupt":
                                self.speaker.stop()
                            self.speaker.say(cue.message)
        self.latency.policy.append((time.perf_counter() - t0) * 1000)
        self.latency.end_to_end.append((time.perf_counter() - t_start) * 1000)
        self.frame_index += 1
        return StepResult(t=t, tracks=tracks, detections=dets, cue=cue,
                          latency_ms={"detect": self.latency.detect[-1], "depth": depth_ms, "end_to_end": self.latency.end_to_end[-1]},
                          quality=quality)
