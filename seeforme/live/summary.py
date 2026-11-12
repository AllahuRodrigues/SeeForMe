"""On-demand spoken summaries for the live demo ("what is around me?") and a local session log.

    scene_summary(tracks, camera, units)   up to three confirmed objects, nearest first, with direction and distance
    SessionLog(path)                       one JSON line per cue / mode change / key press, for reviewing a walk later

The tracker sometimes keeps two overlapping tracks of one person for a while (a re-detection that did not match the
old track yet), which made "what's around me" say "A person slightly right, about 2.5 meters." twice. Objects that
read the same (label or sign text, direction and distance as spoken) are merged: overlapping boxes are one object
said once, separate boxes are counted ("2 people slightly right, about 2.5 meters.").
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ..geometry.camera import CameraModel

_PLURAL = {"person": "people", "bus": "buses", "bench": "benches", "couch": "couches", "dining table": "tables"}


def _direction(angle: float) -> str:
    if abs(angle) < 10:
        return "ahead"
    if abs(angle) < 25:
        return "slightly right" if angle > 0 else "slightly left"
    return "on your right" if angle > 0 else "on your left"


def _distance(d: Optional[float], depth_conf: float, units: str) -> str:
    if d is None or not math.isfinite(d) or depth_conf < 0.45:
        return ""
    if units == "ft":
        return f", about {round(d * 3.28)} feet"
    v = round(d * 2) / 2 if d < 3 else round(d)
    return f", about {v:g} meters"


def _overlap(a, b) -> float:
    """Intersection over the smaller box: 1 when one box lies inside the other (a split track), 0 when apart."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / max(small, 1e-9)


SAME_OBJECT_OVERLAP = 0.5        # two tracks that read the same and overlap this much are one object


def scene_summary(tracks: Iterable, camera: Optional[CameraModel] = None, units: str = "m", limit: int = 3) -> str:
    """'A person ahead, about 1.5 meters. A chair on your left. 2 more people.' from confirmed, currently visible tracks.
    Tracks that read the same are merged (module docstring): one object when their boxes overlap, else counted."""
    cam = camera or CameraModel()
    seen = [t for t in tracks if getattr(t, "confirmed", True) and getattr(t, "misses", 0) == 0 and not getattr(t, "remembered", False)]
    if not seen:
        return "I don't see anything I can name right now. That does not mean the way is clear."
    seen.sort(key=lambda t: t.distance_m if t.distance_m is not None and math.isfinite(t.distance_m) else 99.0)
    # (what, what for several, where, label, boxes): one entry per phrase, the boxes of the distinct objects it stands for
    groups: List[Tuple[str, str, str, str, list]] = []
    for t in seen:
        angle = cam.angle_deg_from_u(0.5 * (t.bbox[0] + t.bbox[2]))
        if t.kind == "text" and t.text:
            what, many, label = f"sign that says {t.text}", f"signs that say {t.text}", "sign"
        else:
            what, many, label = t.label, _PLURAL.get(t.label, t.label + "s"), t.label
        where = f"{_direction(angle)}{_distance(t.distance_m, getattr(t, 'depth_conf', 0.0), units)}"
        g = next((g for g in groups if g[0] == what and g[2] == where), None)
        if g is None:
            groups.append((what, many, where, label, [t.bbox]))
        elif not any(_overlap(t.bbox, b) >= SAME_OBJECT_OVERLAP for b in g[4]):
            g[4].append(t.bbox)                       # another object that reads the same
    parts = [f"A {what} {where}." if len(boxes) == 1 else f"{len(boxes)} {many} {where}."
             for what, many, where, _label, boxes in groups[:limit]]
    rest = groups[limit:]
    if rest:
        n = sum(len(g[4]) for g in rest)
        labels = {g[3] for g in rest}
        if len(labels) == 1:
            lab = next(iter(labels))
            parts.append(f"{n} more {_PLURAL.get(lab, lab + 's') if n > 1 else lab}.")
        else:
            parts.append(f"{n} more objects.")
    return " ".join(parts)


class SessionLog:
    """Append-only JSON lines in results/live_sessions/ (git-ignored): when, what was said or done, and why."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            root = Path(__file__).resolve().parents[2] / "results" / "live_sessions"
            root.mkdir(parents=True, exist_ok=True)
            path = str(root / time.strftime("%Y%m%d-%H%M%S.jsonl"))
        self.path = path
        self._f = open(path, "a", encoding="utf8")

    def write(self, kind: str, t: float, **fields) -> None:
        rec = {"kind": kind, "t": round(float(t), 3)}
        rec.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fields.items()})
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()

    def cue(self, t: float, cue, mode: str = "", activity: str = "") -> None:
        self.write("cue", t, action=cue.action, message=cue.message, label=cue.label, track=cue.track_id,
                   distance_m=cue.distance_m, ttc_s=cue.ttc_s if cue.ttc_s is None or math.isfinite(cue.ttc_s) else None,
                   severity=cue.severity, policy=cue.policy, mode=mode, activity=activity)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
