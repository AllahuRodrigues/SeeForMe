"""Core data types shared across the pipeline."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 normalised to [0, 1]


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_center(b: BBox) -> Tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def bbox_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """A single model prediction for one frame."""

    label: str
    confidence: float
    bbox: BBox
    kind: str = "object"          # "object" or "text"
    text: Optional[str] = None    # recognised string for kind == "text"
    source_id: Optional[int] = None  # ground-truth object id (simulation only)
    distance_m: Optional[float] = None   # metric depth of the object (depth model / simulated depth sensor)
    depth_conf: float = 0.0              # 0..1 confidence of distance_m (0 = no depth available)


@dataclass
class Track:
    """Persistent state for one tracked object or piece of text.

    This is the "temporal memory" from section 4.3 of the proposal:
    (class, confidence, location, age, last announced).
    """

    track_id: int
    label: str
    kind: str
    bbox: BBox
    confidence: float
    first_seen: float
    last_seen: float
    text: Optional[str] = None
    age: int = 0                      # frames since creation
    hits: int = 1                     # matched detections
    misses: int = 0                   # consecutive frames without a match
    confirmed: bool = False
    last_announced: Optional[float] = None
    last_announced_bbox: Optional[BBox] = None
    announce_count: int = 0
    bbox_history: Deque[Tuple[float, BBox]] = field(default_factory=lambda: deque(maxlen=15))
    source_votes: dict = field(default_factory=dict)
    # --- metric dynamics (filled by geometry.dynamics.TrackDynamics) ---------
    distance_m: Optional[float] = None    # forward range to the object (m)
    depth_conf: float = 0.0               # confidence of distance_m (0 = bbox-height guess only)
    lateral_m: float = 0.0                # lateral offset, +right (m)
    rel_speed_mps: float = 0.0            # d(distance)/dt, negative = approaching
    lateral_speed_mps: float = 0.0
    ttc_s: float = float("inf")           # time to contact if approaching
    path_prob: float = 0.0                # P(object will be inside the walking corridor at arrival)
    dist_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=25))  # (t, z, x)
    last_haptic: Optional[float] = None
    remembered: bool = False              # kept alive by the memory ledger (not currently detected)
    memory_t: Optional[float] = None      # last time the memory extrapolation ran

    @property
    def source_id(self) -> Optional[int]:
        if not self.source_votes:
            return None
        return max(self.source_votes.items(), key=lambda kv: kv[1])[0]

    def displacement_since_announced(self) -> float:
        if self.last_announced_bbox is None:
            return 1.0
        (cx0, cy0) = bbox_center(self.last_announced_bbox)
        (cx1, cy1) = bbox_center(self.bbox)
        return ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5

    def area_growth_rate(self) -> float:
        """Relative bbox-area growth per second over the recent history (>0 means approaching)."""
        if len(self.bbox_history) < 2:
            return 0.0
        t0, b0 = self.bbox_history[0]
        t1, b1 = self.bbox_history[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        a0, a1 = bbox_area(b0), bbox_area(b1)
        if a0 <= 1e-6:
            return 0.0
        return (a1 - a0) / a0 / dt


@dataclass
class CueFeatures:
    """The signals combined by CACS. All in [0, 1] except where noted."""

    confidence: float = 0.0   # C
    relevance: float = 0.0    # R
    novelty: float = 0.0      # N
    proximity: float = 0.0    # P
    text_importance: float = 0.0  # T
    duplicate: float = 0.0    # D

    def as_vector(self):
        return [self.confidence, self.relevance, self.novelty,
                self.proximity, self.text_importance, self.duplicate]

    @staticmethod
    def names():
        return ["confidence", "relevance", "novelty", "proximity", "text_importance", "duplicate"]


@dataclass
class Observation:
    """A candidate visual event at one decision step."""

    track: Track
    t: float
    features: CueFeatures
    score: float = 0.0
    extras: list = field(default_factory=list)   # motion / age signals used by the deep context model


@dataclass
class Cue:
    """A spoken message actually emitted by a policy."""

    t: float
    track_id: int
    label: str
    kind: str
    message: str
    score: float
    policy: str
    text: Optional[str] = None
    source_id: Optional[int] = None
    hedged: bool = False
    action: str = "speak"                 # speak | interrupt | haptic
    duration_s: float = 0.0               # how long the message occupies the audio channel
    slots: Tuple[str, ...] = ()           # semantic slots in the message (object, distance, direction, motion, action, detail)
    distance_m: Optional[float] = None
    ttc_s: Optional[float] = None
    severity: float = 0.0
    utility: float = 0.0
    interrupted_track: Optional[int] = None  # track id of the message this cue cut off
