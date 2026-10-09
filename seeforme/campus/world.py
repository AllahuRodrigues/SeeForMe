"""World model and ground-truth annotation for CampusBench."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..cues.compose import compose
from ..geometry.camera import CameraModel, TYPICAL_SIZE_M, DEFAULT_SIZE_M
from ..types import BBox

# ---------------------------------------------------------------------------
# event taxonomy: what each kind of visual event needs from the communication channel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventSpec:
    severity: float          # how bad it is to miss (0 = never speak)
    notice_m: float          # static: range at which information first becomes useful
    notice_ttc: float        # moving: TTC at which information first becomes useful
    react_s: float           # seconds before contact the user needs to act
    ideal_m: float           # static: range of the ideal announcement
    ideal_ttc: float         # moving: ideal TTC of the announcement
    slots: Tuple[str, ...]   # minimum information (object is implicit)
    urgency: str


EVENT_SPECS: Dict[str, EventSpec] = {
    "stairs":         EventSpec(1.0, 8.0, 6.0, 2.5, 4.8, 3.0, ("distance", "action"), "high"),
    "curb":           EventSpec(0.8, 5.5, 5.0, 1.5, 3.4, 2.5, ("distance", "action"), "high"),
    "vehicle":        EventSpec(1.0, 10.0, 5.0, 1.5, 4.0, 3.0, ("motion", "direction"), "high"),
    "bike":           EventSpec(0.9, 8.0, 4.5, 1.2, 3.5, 2.5, ("motion", "direction"), "high"),
    "hazard_static":  EventSpec(0.7, 6.0, 5.0, 1.5, 3.2, 2.5, ("distance",), "medium"),
    "barrier":        EventSpec(0.75, 6.0, 5.0, 1.5, 3.2, 2.5, ("distance",), "medium"),
    "person_moving":  EventSpec(0.5, 6.0, 4.0, 1.2, 3.0, 2.0, ("motion",), "medium"),
    "person_static":  EventSpec(0.5, 5.0, 4.0, 1.5, 3.0, 2.5, ("distance",), "medium"),
    "door":           EventSpec(0.45, 6.0, 5.0, 1.0, 3.5, 2.5, ("distance", "detail"), "low"),
    "landmark":       EventSpec(0.4, 7.0, 5.0, 1.0, 4.0, 3.0, ("distance", "direction"), "low"),
    "sign_nav":       EventSpec(0.45, 8.0, 5.0, 0.8, 5.0, 3.0, ("direction",), "low"),
    "goal":           EventSpec(0.7, 9.0, 5.0, 1.0, 5.0, 3.0, ("distance", "direction"), "medium"),
    "unknown_obstacle": EventSpec(0.6, 5.0, 4.0, 1.5, 3.0, 2.5, ("distance",), "medium"),   # something unfamiliar in the path (OOD)
    "decor":          EventSpec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (), "none"),
    "sign_decor":     EventSpec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (), "none"),
}

# labels an open-vocabulary detector could produce for things the taxonomy has no class for; large and low in
# the walking corridor they are treated as an unknown obstacle and announced hedged ("possibly debris")
UNKNOWN_OBSTACLE_LABELS = {"debris", "broken glass", "fallen sign", "wreckage", "rubble", "unknown object", "obstruction", "spill"}


def default_event_type(label: str, kind: str = "object") -> str:
    if kind == "text":
        return "sign_decor"
    return LABEL_EVENT.get(label, "unknown_obstacle" if label in UNKNOWN_OBSTACLE_LABELS else "decor")

# label -> default event type when none is given
LABEL_EVENT = {
    "stairs": "stairs", "curb": "curb", "car": "vehicle", "bus": "vehicle", "truck": "vehicle",
    "motorcycle": "vehicle", "bicycle": "bike", "scooter": "bike", "door": "door", "elevator": "landmark",
    "person": "person_static", "cone": "barrier", "barrier": "barrier", "wet floor sign": "barrier",
    "chair": "hazard_static", "bench": "hazard_static", "box": "hazard_static", "cart": "hazard_static",
    "backpack": "hazard_static", "suitcase": "hazard_static", "trash can": "hazard_static",
    "potted plant": "hazard_static", "pole": "hazard_static", "dining table": "hazard_static",
    # open-vocabulary labels (perception.NAV_VOCAB)
    "fence": "barrier", "tree": "hazard_static", "shelf": "hazard_static", "machine": "hazard_static",
    "ground hazard": "hazard_static", "fire hydrant": "hazard_static", "animal": "person_moving",
    "sign": "landmark", "crossing": "landmark",
}


@dataclass
class WorldObject:
    obj_id: int
    label: str
    kind: str = "object"                           # object | text
    text: Optional[str] = None
    event_type: str = ""                           # key of EVENT_SPECS ("" -> from label / decor)
    keyframes: List[Tuple[float, float, float]] = field(default_factory=list)   # (t, x_world, z_world)
    t_appear: float = 0.0
    t_disappear: float = 1e9
    height: float = 0.0
    width: float = 0.0
    base_height: float = 0.0                       # metres above ground (signs are mounted high)
    detail: Dict = field(default_factory=dict)     # handrail / handle side / stairs direction ...
    goal_tags: List[str] = field(default_factory=list)
    detectability: float = 0.9
    base_conf: float = 0.85
    conf_noise: float = 0.06
    occluded_windows: List[Tuple[float, float]] = field(default_factory=list)   # hidden behind something (not detectable)
    _scene: Optional["CampusScenario"] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.height <= 0 or self.width <= 0:
            h, w = TYPICAL_SIZE_M.get(self.label, DEFAULT_SIZE_M)
            self.height = self.height or h
            self.width = self.width or w
        if not self.event_type:
            self.event_type = default_event_type(self.label, self.kind)

    # --- motion in the world frame -----------------------------------------
    def world_pos(self, t: float) -> Tuple[float, float]:
        kf = self.keyframes
        if not kf:
            return (0.0, 0.0)
        if t <= kf[0][0] or len(kf) == 1:
            return (kf[0][1], kf[0][2])
        for (t0, x0, z0), (t1, x1, z1) in zip(kf, kf[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / max(1e-6, t1 - t0)
                return (x0 + (x1 - x0) * u, z0 + (z1 - z0) * u)
        return (kf[-1][1], kf[-1][2])

    def world_vel(self, t: float) -> Tuple[float, float]:
        h = 0.1
        (x0, z0), (x1, z1) = self.world_pos(t - h), self.world_pos(t + h)
        return ((x1 - x0) / (2 * h), (z1 - z0) / (2 * h))

    # --- interface used by the detector simulator ---------------------------
    def rel_pos(self, t: float) -> Tuple[float, float]:
        x, z = self.world_pos(t)
        return (x, z - self._scene.user_z(t))

    def bbox_at(self, t: float) -> Optional[BBox]:
        sc = self._scene
        x, z = self.rel_pos(t)
        pitch, yaw = sc.camera_pose(t)
        b = sc.camera.project_box(x, z, self.height, self.width, self.base_height, pitch_deg=pitch, yaw_deg=yaw)
        return sc.camera.clip(b) if b is not None else None

    def occluded(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.occluded_windows)

    def visible(self, t: float) -> bool:
        if not (self.t_appear <= t < self.t_disappear) or self.occluded(t):
            return False
        x, z = self.rel_pos(t)
        if z < 0.3 or z > 40:
            return False
        return self.bbox_at(t) is not None

    def spec(self) -> EventSpec:
        return EVENT_SPECS[self.event_type]

    def to_dict(self) -> Dict:
        return {"obj_id": self.obj_id, "label": self.label, "kind": self.kind, "text": self.text,
                "event_type": self.event_type, "keyframes": [list(k) for k in self.keyframes],
                "t_appear": self.t_appear, "t_disappear": self.t_disappear, "height": self.height, "width": self.width,
                "base_height": self.base_height, "detail": dict(self.detail), "goal_tags": list(self.goal_tags),
                "detectability": self.detectability, "base_conf": self.base_conf, "conf_noise": self.conf_noise,
                "occluded_windows": [list(w) for w in self.occluded_windows]}

    @property
    def moving(self) -> bool:
        return len(self.keyframes) > 1 and any(abs(a[1] - b[1]) + abs(a[2] - b[2]) > 1e-6 for a, b in zip(self.keyframes, self.keyframes[1:]))


@dataclass
class EventAnnotation:
    obj_id: int
    label: str
    kind: str
    event_type: str
    text: Optional[str]
    should_speak: bool
    reason: str
    severity: float
    urgency: str
    interrupt_current_speech: bool
    earliest_useful_s: Optional[float]
    ideal_s: Optional[float]
    latest_safe_s: Optional[float]
    t_contact: Optional[float]
    path_intersection: float
    distance_at_ideal_m: Optional[float]
    direction_at_ideal_deg: Optional[float]
    moving: bool
    sudden: bool
    minimum_information: Tuple[str, ...]
    ideal_cue: str
    detail: Dict = field(default_factory=dict)

    def in_window(self, t: float) -> bool:
        return self.should_speak and self.earliest_useful_s is not None and self.earliest_useful_s <= t <= self.latest_safe_s

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CampusScenario:
    name: str
    kind: str
    duration: float
    fps: float
    objects: List[WorldObject]
    user_keyframes: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (30.0, 33.0)])  # (t, z_user)
    goal: Optional[str] = None
    goal_tags: List[str] = field(default_factory=list)
    camera: CameraModel = field(default_factory=CameraModel)
    pitch_jitter_deg: float = 0.0
    yaw_jitter_deg: float = 0.0
    blackout_windows: List[Tuple[float, float]] = field(default_factory=list)  # camera covered / pointed away
    lighting: str = "normal"
    motion_blur: float = 0.0
    clutter_fp_rate: float = 0.05
    depth_noise_rel: float = 0.06
    seed: int = 0
    description: str = ""
    yaw_keyframes: List[Tuple[float, float]] = field(default_factory=list)          # (t, yaw_deg): deliberate camera pans
    distractor_speech: List[Tuple[float, float, float]] = field(default_factory=list)  # (t_start, duration_s, severity): external audio
    reaction_scale: float = 1.0                                                      # multiplies every event type's reaction time
    memory_windows: bool = False     # True: an object seen once stays action-relevant while hidden (the window follows the user's
                                     # need, not the camera); False: the v1 rule, a window needs the object visible in range
    _events: Optional[List[EventAnnotation]] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for o in self.objects:
            o._scene = self

    # --- user motion ----------------------------------------------------------
    def user_z(self, t: float) -> float:
        kf = self.user_keyframes
        if t <= kf[0][0]:
            return kf[0][1]
        for (t0, z0), (t1, z1) in zip(kf, kf[1:]):
            if t0 <= t <= t1:
                return z0 + (z1 - z0) * (t - t0) / max(1e-6, t1 - t0)
        return kf[-1][1]

    def user_speed(self, t: float) -> float:
        h = 0.1
        return (self.user_z(t + h) - self.user_z(t - h)) / (2 * h)

    def camera_pose(self, t: float) -> Tuple[float, float]:
        """(pitch_deg, yaw_deg) at time t: nominal pitch plus deterministic hand/body sway."""
        pitch = self.camera.pitch_deg
        yaw = 0.0
        if self.pitch_jitter_deg > 0:
            pitch += self.pitch_jitter_deg * (0.6 * math.sin(2 * math.pi * 0.9 * t + self.seed) + 0.4 * math.sin(2 * math.pi * 0.23 * t))
        if self.yaw_jitter_deg > 0:
            yaw = self.yaw_jitter_deg * (0.7 * math.sin(2 * math.pi * 0.31 * t + 0.5 * self.seed) + 0.3 * math.sin(2 * math.pi * 1.3 * t))
        return pitch, yaw + self.pan_deg(t)

    def pan_deg(self, t: float) -> float:
        """Piecewise-linear deliberate pan from yaw_keyframes (0 outside them)."""
        kf = self.yaw_keyframes
        if not kf or t <= kf[0][0] or t >= kf[-1][0]:
            return 0.0
        for (t0, y0), (t1, y1) in zip(kf, kf[1:]):
            if t0 <= t <= t1:
                return y0 + (y1 - y0) * (t - t0) / max(1e-6, t1 - t0)
        return 0.0

    def blacked_out(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.blackout_windows)

    # --- ground truth -----------------------------------------------------------
    def events(self) -> List[EventAnnotation]:
        if self._events is None:
            self._events = [self._annotate(o) for o in self.objects]
        return self._events

    def speak_events(self) -> List[EventAnnotation]:
        return [e for e in self.events() if e.should_speak]

    def event_for(self, obj_id: Optional[int]) -> Optional[EventAnnotation]:
        if obj_id is None:
            return None
        for e in self.events():
            if e.obj_id == obj_id:
                return e
        return None

    def _goal_match(self, o: WorldObject) -> bool:
        if not self.goal_tags:
            return False
        tags = set(t.lower() for t in o.goal_tags)
        if o.kind == "text" and o.text:
            tags |= set(o.text.lower().split())
            tags.add(o.text.lower())
        return any(g.lower() in tags for g in self.goal_tags)

    def _annotate(self, o: WorldObject, dt: float = 0.1) -> EventAnnotation:
        spec = o.spec()
        half = 0.45 + o.width / 2
        contact_m = 0.6
        goal = self._goal_match(o)
        etype = "goal" if goal else o.event_type
        spec = EVENT_SPECS[etype]
        none = lambda reason: EventAnnotation(o.obj_id, o.label, o.kind, etype, o.text, False, reason, 0.0, "none", False,
                                              None, None, None, None, 0.0, None, None, o.moving, False, (), "", dict(o.detail))
        if spec.severity <= 0:
            return none("not_action_relevant")

        # sample the true trajectory while the object exists
        t0, t1 = max(0.0, o.t_appear), min(self.duration, o.t_disappear)
        samples = []
        t = t0
        while t <= t1 + 1e-9:
            x, z = o.rel_pos(t)
            vx, vz = o.world_vel(t)
            vz_rel = vz - self.user_speed(t)
            samples.append((t, x, z, vx, vz_rel, o.visible(t)))
            t += dt
        if not samples:
            return none("never_visible")

        # path intersection / contact: the object is in the corridor when it reaches the user,
        # or sweeps through the corridor close in front of the user
        t_contact = None
        intersect = 0.0
        for (t, x, z, vx, vz_rel, vis) in samples:
            in_corridor = abs(x) < half
            if in_corridor and z <= contact_m + 0.2 and z > -0.5:
                t_contact = t; intersect = 1.0; break
            if in_corridor and o.moving and 0.0 < z < 3.0 and abs(vx) > 0.3:
                t_contact = t; intersect = 1.0; break
        first_vis = next((s for s in samples if s[5]), None)
        if first_vis is None:
            return none("never_visible")

        legible = lambda s: s[5] and s[2] <= spec.notice_m and abs(math.degrees(math.atan2(s[1], max(0.1, s[2])))) < 30
        if etype in ("sign_nav", "goal", "landmark"):
            # informational: useful once legible/visible ahead, until it leaves the view or is passed
            vis_s = [s for s in samples if legible(s)]
            if not vis_s:
                return none("never_legible")
            earliest = vis_s[0][0]
            latest = max(vis_s[-1][0], earliest + 0.5)
            ideal = min(latest, max(earliest + 0.5, next((s[0] for s in vis_s if s[2] <= spec.ideal_m), earliest + 0.5)))
            if etype == "goal" and o.kind != "text" and intersect:
                latest = min(latest, t_contact - spec.react_s * self.reaction_scale)
            sev = spec.severity
            sudden = False
        else:
            if not intersect:
                return none("off_path")
            moving = o.moving
            seen = (lambda s: s[5]) if not self.memory_windows else (lambda s: s[0] >= first_vis[0])   # seen now, or seen before
            if moving:
                cand = [s for s in samples if seen(s) and s[4] < -0.1 and (s[2] - contact_m) / (-s[4]) <= spec.notice_ttc]
                cand = cand or [s for s in samples if seen(s) and s[2] <= spec.notice_m]
            else:
                cand = [s for s in samples if seen(s) and s[2] <= spec.notice_m]
            if not cand:
                return none("never_visible_in_range")
            earliest = cand[0][0]
            latest = t_contact - spec.react_s * self.reaction_scale
            sudden = latest < earliest
            if sudden:
                earliest = first_vis[0]
                latest = max(earliest + 0.3, latest)
            if moving:
                ideal = next((s[0] for s in cand if s[4] < -0.1 and (s[2] - contact_m) / (-s[4]) <= spec.ideal_ttc), None)
            else:
                ideal = next((s[0] for s in cand if s[2] <= spec.ideal_m), None)
            if ideal is None:
                ideal = 0.5 * (earliest + latest)
            ideal = min(max(ideal, earliest), latest)
            sev = spec.severity
            if o.event_type == "person_static" and moving:
                etype = "person_moving"; spec = EVENT_SPECS[etype]

        # state at the ideal moment
        s_ideal = min(samples, key=lambda s: abs(s[0] - ideal))
        _, xi, zi, vxi, vzi, _ = s_ideal
        angle = math.degrees(math.atan2(xi, max(0.1, zi)))
        slots = ["object"] + list(spec.slots)
        if abs(angle) > 12 and "direction" not in slots:
            slots.append("direction")
        if o.moving and "motion" not in slots and etype not in ("sign_nav", "goal", "landmark"):
            slots.append("motion")
        if etype in ("stairs", "curb") and "action" not in slots:
            slots.append("action")
        cue = compose(slots, o.label, o.kind, o.text, distance_m=zi, depth_conf=1.0, angle_deg=angle,
                      rel_speed=vzi, lateral_speed=vxi, lateral_m=xi, detail=o.detail)
        return EventAnnotation(o.obj_id, o.label, o.kind, etype, o.text, True, "", sev, spec.urgency,
                               sev >= 0.75, round(earliest, 2), round(ideal, 2), round(latest, 2),
                               None if t_contact is None else round(t_contact, 2), intersect, round(zi, 2),
                               round(angle, 1), o.moving, sudden, tuple(slots), cue, dict(o.detail))

    # --- serialisation (the annotation format from docs/campus_annotation.md) --------
    def to_json(self, path: str) -> None:
        d = {
            "name": self.name, "kind": self.kind, "duration": self.duration, "fps": self.fps, "seed": self.seed,
            "description": self.description, "goal": self.goal, "goal_tags": self.goal_tags,
            "user_keyframes": self.user_keyframes,
            "camera": asdict(self.camera), "pitch_jitter_deg": self.pitch_jitter_deg, "yaw_jitter_deg": self.yaw_jitter_deg,
            "blackout_windows": self.blackout_windows, "lighting": self.lighting, "motion_blur": self.motion_blur,
            "clutter_fp_rate": self.clutter_fp_rate, "depth_noise_rel": self.depth_noise_rel,
            "yaw_keyframes": [list(k) for k in self.yaw_keyframes], "distractor_speech": [list(d) for d in self.distractor_speech],
            "reaction_scale": self.reaction_scale, "memory_windows": self.memory_windows,
            "objects": [o.to_dict() for o in self.objects],
            "events": [e.as_dict() for e in self.events()],
        }
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def from_json(path: str) -> "CampusScenario":
        d = json.loads(Path(path).read_text())
        objs = []
        for o in d["objects"]:
            o["keyframes"] = [tuple(k) for k in o["keyframes"]]
            o["occluded_windows"] = [tuple(w) for w in o.get("occluded_windows", [])]
            objs.append(WorldObject(**o))
        sc = CampusScenario(name=d["name"], kind=d["kind"], duration=d["duration"], fps=d["fps"], objects=objs,
                            user_keyframes=[tuple(k) for k in d["user_keyframes"]], goal=d.get("goal"),
                            goal_tags=d.get("goal_tags", []), camera=CameraModel(**d["camera"]),
                            pitch_jitter_deg=d.get("pitch_jitter_deg", 0.0), yaw_jitter_deg=d.get("yaw_jitter_deg", 0.0),
                            blackout_windows=[tuple(w) for w in d.get("blackout_windows", [])], lighting=d["lighting"],
                            motion_blur=d["motion_blur"], clutter_fp_rate=d["clutter_fp_rate"],
                            depth_noise_rel=d.get("depth_noise_rel", 0.06), seed=d["seed"], description=d.get("description", ""),
                            yaw_keyframes=[tuple(k) for k in d.get("yaw_keyframes", [])],
                            distractor_speech=[tuple(x) for x in d.get("distractor_speech", [])],
                            reaction_scale=d.get("reaction_scale", 1.0), memory_windows=d.get("memory_windows", False))
        return sc
