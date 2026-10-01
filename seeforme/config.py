"""Configuration dataclasses for every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict
import yaml


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_age: int = 8            # frames without a match before a track is dropped
    n_init: int = 2             # matches required before a track is confirmed
    conf_ema: float = 0.6       # weight of new confidence in the exponential moving average
    text_similarity: float = 0.6


@dataclass
class CACSWeights:
    """S_i = alpha*C + beta*R + gamma*N + delta*P + eta*T - lambda*D"""

    alpha: float = 0.5   # confidence
    beta: float = 1.5    # contextual relevance
    gamma: float = 1.0   # novelty
    delta: float = 2.0   # proximity / navigation importance
    eta: float = 2.0     # text importance
    lam: float = 2.5     # duplicate penalty

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CACSConfig:
    weights: CACSWeights = field(default_factory=CACSWeights)
    threshold: float = 3.3          # minimum score before a cue is spoken
    min_confidence: float = 0.35    # below this a prediction is never spoken
    sure_confidence: float = 0.6    # below this a prediction is spoken with hedging
    duplicate_tau: float = 8.0      # seconds; time constant of the duplicate penalty
    label_duplicate_tau: float = 4.0  # penalty for same-label announcements from other tracks
    move_reset: float = 0.25        # normalised displacement that makes a track "new" again
    approach_boost: float = 0.5     # extra novelty when an announced object is approaching
    decision_interval: float = 0.5  # seconds between cue-selection decisions
    max_cues_per_decision: int = 1
    # feature toggles used by the ablation study
    use_novelty: bool = True
    use_proximity: bool = True
    use_uncertainty: bool = True
    use_temporal_memory: bool = True
    use_relevance: bool = True
    use_text: bool = True
    use_scene_context: bool = True


@dataclass
class BaselineConfig:
    confidence_threshold: float = 0.5
    cooldown: float = 5.0           # seconds a label is suppressed after being spoken
    decision_interval: float = 0.5
    max_cues_per_decision: int = 1


@dataclass
class SpeechConfig:
    min_gap: float = 1.2            # seconds a message occupies the audio channel (minimum)
    seconds_per_word: float = 0.35


@dataclass
class PerceptionConfig:
    yolo_model: str = "yolo11n.pt"
    yolo_conf: float = 0.25
    yolo_imgsz: int = 640
    device: str = "auto"
    ocr_every_n_frames: int = 8
    ocr_min_conf: float = 0.3
    ocr_languages: tuple = ("en",)
    frame_width: int = 640


@dataclass
class PipelineConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    cacs: CACSConfig = field(default_factory=CACSConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    pcacs: "PCACSConfig" = None   # type: ignore[assignment]  # filled in __post_init__ (avoids an import cycle)

    def __post_init__(self):
        if self.pcacs is None:
            from .cues.pcacs import PCACSConfig
            self.pcacs = PCACSConfig()

    @staticmethod
    def load(path: str) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = PipelineConfig()
        for section, values in raw.items():
            target = getattr(cfg, section)
            for k, v in (values or {}).items():
                if k == "weights" and section == "cacs":
                    target.weights = CACSWeights(**v)
                elif k == "weights" and section == "pcacs":
                    from .cues.pcacs import PCACSWeights
                    target.weights = PCACSWeights(**v)
                else:
                    setattr(target, k, v)
        return cfg

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
