"""Synthetic evaluation scenarios with ground-truth important events.

Real annotated assistive video is scarce, so the benchmark generates controlled
scenes: a set of scene objects with trajectories in normalised image space, plus a
noisy detector model (missed detections, confidence noise, label flips, false
positives, motion-blur and low-light degradations). Every object carries a
ground-truth `important` window so cue-selection metrics can be computed exactly.

Scenarios are serialisable to JSON so the same benchmark can be re-run and shared.
The same JSON format can be produced by hand-annotating real video (see
`docs/annotation_format.md`).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..types import BBox, Detection

# ----------------------------------------------------------------------------
# scene description
# ----------------------------------------------------------------------------

@dataclass
class SceneObject:
    obj_id: int
    label: str
    kind: str = "object"                # "object" | "text"
    text: Optional[str] = None
    t_appear: float = 0.0
    t_disappear: float = 1e9
    # trajectory: bbox at t_appear and at t_disappear, linearly interpolated
    bbox_start: BBox = (0.4, 0.4, 0.6, 0.7)
    bbox_end: BBox = (0.4, 0.4, 0.6, 0.7)
    motion: str = "static"              # static | approach | pass | drift
    # ground truth: windows during which announcing this object is useful
    important_windows: List[Tuple[float, float]] = field(default_factory=list)
    detectability: float = 0.9          # base probability the detector sees it per frame
    base_conf: float = 0.85
    conf_noise: float = 0.06

    def bbox_at(self, t: float) -> BBox:
        if self.t_disappear >= 1e8:
            u = min(1.0, max(0.0, (t - self.t_appear) / 10.0))
        else:
            span = max(1e-6, self.t_disappear - self.t_appear)
            u = min(1.0, max(0.0, (t - self.t_appear) / span))
        if self.motion == "approach":
            u = u ** 1.6   # objects grow faster as they get close
        return tuple(a + (b - a) * u for a, b in zip(self.bbox_start, self.bbox_end))  # type: ignore

    def visible(self, t: float) -> bool:
        return self.t_appear <= t < self.t_disappear

    def important_at(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.important_windows)


@dataclass
class Scenario:
    name: str
    duration: float
    fps: float
    objects: List[SceneObject]
    seed: int = 0
    lighting: str = "normal"     # normal | low
    motion_blur: float = 0.0     # 0..1, degrades confidence/detectability
    clutter_fp_rate: float = 0.05  # false positives per frame
    description: str = ""

    def important_events(self) -> List[Dict]:
        ev = []
        for o in self.objects:
            for (a, b) in o.important_windows:
                ev.append({"obj_id": o.obj_id, "label": o.label, "kind": o.kind,
                           "text": o.text, "t_start": a, "t_end": b})
        return ev

    def to_json(self, path: str) -> None:
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def from_json(path: str) -> "Scenario":
        d = json.loads(Path(path).read_text())
        objs = []
        for o in d.pop("objects"):
            o["bbox_start"] = tuple(o["bbox_start"]); o["bbox_end"] = tuple(o["bbox_end"])
            o["important_windows"] = [tuple(w) for w in o["important_windows"]]
            objs.append(SceneObject(**o))
        return Scenario(objects=objs, **d)


# ----------------------------------------------------------------------------
# noisy detector model
# ----------------------------------------------------------------------------

CONFUSIONS = {
    "chair": ["bench", "couch"], "person": ["mannequin"], "door": ["window"],
    "backpack": ["handbag", "suitcase"], "trash can": ["potted plant"], "bicycle": ["motorcycle"],
    "dog": ["cat"], "car": ["truck"], "poster": ["tv"], "potted plant": ["vase"],
}
FP_LABELS = ["bottle", "cup", "cell phone", "book", "clock", "handbag", "remote", "vase", "chair", "person"]
OCR_CORRUPTIONS = [("O", "0"), ("I", "1"), ("E", "F"), ("S", "5"), ("T", "7"), ("B", "8")]


class DetectorSimulator:
    """Produces per-frame Detections from a Scenario with realistic noise."""

    def __init__(self, scenario: Scenario, seed: Optional[int] = None):
        self.sc = scenario
        self.rng = random.Random(scenario.seed if seed is None else seed)
        self.degrade = 0.0
        if scenario.lighting == "low":
            self.degrade += 0.25
        self.degrade += 0.35 * scenario.motion_blur

    def _jitter(self, b: BBox, amount: float) -> BBox:
        j = lambda: self.rng.uniform(-amount, amount)
        x1, y1, x2, y2 = b[0] + j(), b[1] + j(), b[2] + j(), b[3] + j()
        x1, x2 = sorted((max(0.0, x1), min(1.0, x2)))
        y1, y2 = sorted((max(0.0, y1), min(1.0, y2)))
        if x2 - x1 < 0.01: x2 = min(1.0, x1 + 0.01)
        if y2 - y1 < 0.01: y2 = min(1.0, y1 + 0.01)
        return (x1, y1, x2, y2)

    def _corrupt_text(self, s: str) -> str:
        if self.rng.random() > 0.15 + self.degrade * 0.5:
            return s
        chars = list(s)
        for _ in range(self.rng.randint(1, 2)):
            i = self.rng.randrange(len(chars))
            for a, b in OCR_CORRUPTIONS:
                if chars[i].upper() == a:
                    chars[i] = b
                    break
        return "".join(chars)

    def detections_at(self, t: float) -> List[Detection]:
        dets: List[Detection] = []
        for o in self.sc.objects:
            if not o.visible(t):
                continue
            bbox = o.bbox_at(t)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # small and far objects are missed more often; text needs to be fairly large
            size_factor = min(1.0, math.sqrt(area) / (0.12 if o.kind == "object" else 0.18))
            p_det = o.detectability * (0.55 + 0.45 * size_factor) * (1 - self.degrade)
            if self.rng.random() > p_det:
                continue
            conf = o.base_conf * (0.7 + 0.3 * size_factor) - self.degrade * 0.25
            conf += self.rng.gauss(0, o.conf_noise)
            conf = max(0.05, min(0.99, conf))
            label = o.label
            if o.kind == "object" and self.rng.random() < 0.04 + self.degrade * 0.1:
                label = self.rng.choice(CONFUSIONS.get(o.label, [o.label]))
            text = self._corrupt_text(o.text) if o.kind == "text" and o.text else None
            dets.append(Detection(label=label, confidence=conf, bbox=self._jitter(bbox, 0.012 + 0.02 * self.degrade),
                                  kind=o.kind, text=text, source_id=o.obj_id))
        # false positives (clutter)
        n_fp = 0
        lam = self.sc.clutter_fp_rate * (1 + 2 * self.degrade)
        while self.rng.random() < lam and n_fp < 3:
            n_fp += 1
            lam *= 0.5
            w, h = self.rng.uniform(0.03, 0.15), self.rng.uniform(0.03, 0.2)
            x, y = self.rng.uniform(0, 1 - w), self.rng.uniform(0, 1 - h)
            dets.append(Detection(label=self.rng.choice(FP_LABELS), confidence=self.rng.uniform(0.25, 0.7),
                                  bbox=(x, y, x + w, y + h), kind="object", source_id=None))
        return dets

    def frames(self):
        n = int(self.sc.duration * self.sc.fps)
        for i in range(n):
            t = i / self.sc.fps
            yield t, self.detections_at(t)


# ----------------------------------------------------------------------------
# scenario templates
# ----------------------------------------------------------------------------

def _approach(label, t0, t1, x=0.5, w0=0.06, w1=0.45, h_ratio=1.4, y_floor=0.95, **kw) -> SceneObject:
    """Object starting small near the horizon and growing as the user walks toward it."""
    h0, h1 = w0 * h_ratio, w1 * h_ratio
    bs = (x - w0 / 2, 0.55 - h0 / 2, x + w0 / 2, 0.55 + h0 / 2)
    be = (x - w1 / 2, y_floor - h1, x + w1 / 2, y_floor)
    return SceneObject(obj_id=-1, label=label, t_appear=t0, t_disappear=t1, bbox_start=bs, bbox_end=be,
                       motion="approach", **kw)


def _static(label, t0, t1, bbox, **kw) -> SceneObject:
    return SceneObject(obj_id=-1, label=label, t_appear=t0, t_disappear=t1, bbox_start=bbox, bbox_end=bbox, **kw)


def _pass(label, t0, t1, y=0.5, h=0.3, left_to_right=True, **kw) -> SceneObject:
    w = h * 0.6
    xs, xe = (-w, 1.0) if left_to_right else (1.0, -w)
    return SceneObject(obj_id=-1, label=label, t_appear=t0, t_disappear=t1,
                       bbox_start=(max(0, xs), y - h / 2, max(0, xs) + w, y + h / 2),
                       bbox_end=(min(1 - w, xe), y - h / 2, min(1 - w, xe) + w, y + h / 2), motion="pass", **kw)


def _sign(text, t0, t1, bbox, important: bool, **kw) -> SceneObject:
    windows = [(t0 + 0.3, min(t1, t0 + 6.0))] if important else []
    return SceneObject(obj_id=-1, label="text", kind="text", text=text, t_appear=t0, t_disappear=t1,
                       bbox_start=bbox, bbox_end=bbox, important_windows=windows,
                       detectability=kw.pop("detectability", 0.75), base_conf=kw.pop("base_conf", 0.7), **kw)


IMPORTANT_SIGNS = ["EXIT", "STOP", "CAUTION WET FLOOR", "ROOM 204", "RESTROOM", "STAIRS", "ELEVATOR",
                   "DO NOT ENTER", "EMERGENCY EXIT", "PUSH", "PULL", "ROOM 117B", "FIRE EXIT", "GATE 12"]
DECOR_SIGNS = ["SALE 50% OFF", "WELCOME", "COFFEE $2", "FOLLOW US ONLINE", "SPRING CONCERT", "HAPPY HOUR",
               "JOIN THE CLUB", "NEW MENU", "BUY ONE GET ONE", "ART SHOW FRIDAY", "www.example.com"]
DECOR_OBJECTS = ["poster", "tv", "clock", "laptop", "book", "cup", "bottle", "vase", "cell phone", "painting"]
OBSTACLES = ["chair", "trash can", "potted plant", "suitcase", "bench", "box", "cart", "cone"]


def make_scenario(kind: str, seed: int, duration: float = 30.0, fps: float = 10.0) -> Scenario:
    rng = random.Random(seed)
    objs: List[SceneObject] = []

    def add(o: SceneObject):
        o.obj_id = len(objs) + 1
        objs.append(o)

    lighting = "normal"
    blur = 0.0
    fp_rate = 0.05

    if kind == "hallway":
        # background clutter visible the whole time, never important
        for _ in range(rng.randint(2, 4)):
            x = rng.uniform(0.05, 0.8); y = rng.uniform(0.05, 0.4)
            add(_static(rng.choice(DECOR_OBJECTS), 0, duration, (x, y, x + 0.15, y + 0.15), base_conf=0.9))
        add(_sign(rng.choice(DECOR_SIGNS), 0, duration, (0.6, 0.15, 0.9, 0.28), important=False, base_conf=0.85))
        # far person standing still: visible early, only important if it approaches
        add(_static("person", 0, duration, (0.7, 0.42, 0.76, 0.6), base_conf=0.95))
        # obstacles approaching in the walking path
        t = rng.uniform(2, 5)
        while t < duration - 6:
            lab = rng.choice(OBSTACLES)
            o = _approach(lab, t, t + 6, x=rng.uniform(0.35, 0.65))
            o.important_windows = [(t + 1.5, t + 6)]   # becomes important once it is reasonably near
            add(o)
            t += rng.uniform(7, 11)
        # an important sign appears mid-way
        ts = rng.uniform(8, duration - 8)
        add(_sign(rng.choice(IMPORTANT_SIGNS), ts, ts + 7, (0.45, 0.1, 0.72, 0.22), important=True))
        # a person walks across
        tp = rng.uniform(5, duration - 6)
        p = _pass("person", tp, tp + 4, y=0.6, h=0.45, left_to_right=rng.random() < 0.5)
        p.important_windows = [(tp + 0.5, tp + 3.5)]
        add(p)

    elif kind == "office":
        for _ in range(rng.randint(4, 7)):
            x = rng.uniform(0.0, 0.85); y = rng.uniform(0.0, 0.5)
            add(_static(rng.choice(DECOR_OBJECTS + ["laptop", "keyboard", "mouse"]), 0, duration,
                        (x, y, x + rng.uniform(0.06, 0.18), y + rng.uniform(0.06, 0.18)), base_conf=0.88))
        add(_sign(rng.choice(DECOR_SIGNS), 0, duration, (0.1, 0.05, 0.4, 0.15), important=False))
        # chair in path immediately, then a door with room number later
        c = _approach("chair", 1.0, 7.0, x=0.5)
        c.important_windows = [(2.0, 7.0)]
        add(c)
        td = rng.uniform(10, duration - 9)
        d = _approach("door", td, td + 8, x=0.5, w0=0.1, w1=0.6, h_ratio=1.5)
        d.important_windows = [(td + 2, td + 8)]
        add(d)
        add(_sign(rng.choice(["ROOM 204", "ROOM 117B", "RESTROOM", "EXIT"]), td + 1, td + 8,
                  (0.55, 0.2, 0.8, 0.3), important=True))
        t = 8.0
        while t < duration - 5:
            o = _approach(rng.choice(["trash can", "box", "potted plant"]), t, t + 5, x=rng.uniform(0.3, 0.7), w1=0.35)
            o.important_windows = [(t + 1.5, t + 5)]
            add(o)
            t += rng.uniform(8, 12)

    elif kind == "street":
        fp_rate = 0.08
        add(_static("traffic light", 0, duration, (0.5, 0.05, 0.56, 0.2), base_conf=0.8))
        t = 1.0
        while t < duration - 4:
            car = _pass("car", t, t + rng.uniform(2, 3.5), y=0.55, h=0.35, left_to_right=rng.random() < 0.5)
            car.important_windows = [(t + 0.3, t + 2.0)]
            add(car)
            t += rng.uniform(4, 8)
        ts = rng.uniform(6, duration - 8)
        add(_sign("STOP", ts, ts + 7, (0.15, 0.15, 0.32, 0.32), important=True, base_conf=0.8))
        tb = rng.uniform(4, duration - 7)
        b = _approach("bicycle", tb, tb + 4, x=rng.uniform(0.4, 0.6), w1=0.5)
        b.important_windows = [(tb + 0.5, tb + 4)]
        add(b)
        for _ in range(rng.randint(2, 4)):
            x = rng.uniform(0.0, 0.8)
            add(_static("person", 0, duration, (x, 0.45, x + 0.05, 0.62), base_conf=0.9))  # far pedestrians
        add(_sign(rng.choice(DECOR_SIGNS), 0, duration, (0.7, 0.1, 0.95, 0.25), important=False))
        tp = rng.uniform(5, duration - 6)
        pole = _approach("pole", tp, tp + 5, x=rng.uniform(0.4, 0.6), w0=0.02, w1=0.12, h_ratio=6, y_floor=1.0)
        pole.important_windows = [(tp + 1.5, tp + 5)]
        add(pole)

    elif kind == "cluttered_room":
        fp_rate = 0.15
        for _ in range(rng.randint(8, 12)):
            x = rng.uniform(0.0, 0.85); y = rng.uniform(0.0, 0.6)
            add(_static(rng.choice(DECOR_OBJECTS + ["backpack", "handbag", "bottle"]), 0, duration,
                        (x, y, x + rng.uniform(0.05, 0.2), y + rng.uniform(0.05, 0.2)), base_conf=rng.uniform(0.6, 0.95)))
        for _ in range(2):
            add(_sign(rng.choice(DECOR_SIGNS), 0, duration, (rng.uniform(0, 0.6), rng.uniform(0, 0.3), 0, 0), important=False))
            objs[-1].bbox_start = (objs[-1].bbox_start[0], objs[-1].bbox_start[1], objs[-1].bbox_start[0] + 0.25, objs[-1].bbox_start[1] + 0.1)
            objs[-1].bbox_end = objs[-1].bbox_start
        t = rng.uniform(1, 3)
        while t < duration - 5:
            o = _approach(rng.choice(OBSTACLES), t, t + 5, x=rng.uniform(0.35, 0.65), w1=0.4)
            o.important_windows = [(t + 1.2, t + 5)]
            add(o)
            t += rng.uniform(6, 9)
        ts = rng.uniform(6, duration - 8)
        add(_sign(rng.choice(IMPORTANT_SIGNS), ts, ts + 6, (0.4, 0.08, 0.7, 0.2), important=True))

    elif kind == "low_light":
        lighting = "low"; blur = 0.3
        base = make_scenario("hallway", seed, duration, fps)
        objs = base.objects
        for o in objs:
            o.detectability *= 0.9

    elif kind == "text_corridor":
        for i in range(rng.randint(3, 5)):
            x = rng.uniform(0.0, 0.7)
            add(_sign(rng.choice(DECOR_SIGNS), 0, duration, (x, rng.uniform(0.02, 0.3), x + 0.25, 0, ), important=False))
            objs[-1].bbox_start = (x, objs[-1].bbox_start[1], x + 0.25, objs[-1].bbox_start[1] + 0.1)
            objs[-1].bbox_end = objs[-1].bbox_start
        t = rng.uniform(2, 4)
        while t < duration - 7:
            add(_sign(rng.choice(IMPORTANT_SIGNS), t, t + 6, (rng.uniform(0.3, 0.6), 0.1, 0, 0), important=True))
            objs[-1].bbox_start = (objs[-1].bbox_start[0], 0.1, objs[-1].bbox_start[0] + 0.3, 0.22)
            objs[-1].bbox_end = objs[-1].bbox_start
            t += rng.uniform(8, 12)
        c = _approach(rng.choice(OBSTACLES), duration * 0.4, duration * 0.4 + 6, x=0.5)
        c.important_windows = [(duration * 0.4 + 1.5, duration * 0.4 + 6)]
        add(c)
        add(_static("person", 0, duration, (0.2, 0.42, 0.26, 0.6), base_conf=0.95))
    else:
        raise ValueError(f"unknown scenario kind {kind}")

    return Scenario(name=f"{kind}_{seed}", duration=duration, fps=fps, objects=objs, seed=seed,
                    lighting=lighting, motion_blur=blur, clutter_fp_rate=fp_rate,
                    description=f"{kind} scenario, seed {seed}")


SCENARIO_KINDS = ["hallway", "office", "street", "cluttered_room", "low_light", "text_corridor"]


def make_benchmark(n_per_kind: int = 5, seed0: int = 100, duration: float = 30.0, fps: float = 10.0) -> List[Scenario]:
    out = []
    for k, kind in enumerate(SCENARIO_KINDS):
        for i in range(n_per_kind):
            out.append(make_scenario(kind, seed0 + 1000 * k + i, duration, fps))
    return out
