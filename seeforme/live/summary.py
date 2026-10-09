"""On-demand spoken summaries for the live demo ("what is around me?") and a local session log.

    scene_summary(tracks, camera, units)   up to three confirmed objects, nearest first, with direction and distance
    SessionLog(path)                       one JSON line per cue / mode change / key press, for reviewing a walk later
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional

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


def scene_summary(tracks: Iterable, camera: Optional[CameraModel] = None, units: str = "m", limit: int = 3) -> str:
    """'A person ahead, about 1.5 meters. A chair on your left. 2 more people.' from confirmed, currently visible tracks."""
    cam = camera or CameraModel()
    seen = [t for t in tracks if getattr(t, "confirmed", True) and getattr(t, "misses", 0) == 0 and not getattr(t, "remembered", False)]
    if not seen:
        return "I don't see anything I can name right now. That does not mean the way is clear."
    seen.sort(key=lambda t: t.distance_m if t.distance_m is not None and math.isfinite(t.distance_m) else 99.0)
    parts: List[str] = []
    for t in seen[:limit]:
        angle = cam.angle_deg_from_u(0.5 * (t.bbox[0] + t.bbox[2]))
        what = f"a sign that says {t.text}" if t.kind == "text" and t.text else f"a {t.label}"
        parts.append(f"{what[0].upper() + what[1:]} {_direction(angle)}{_distance(t.distance_m, getattr(t, 'depth_conf', 0.0), units)}.")
    rest = seen[limit:]
    if rest:
        labels = {t.label for t in rest}
        if len(labels) == 1:
            lab = next(iter(labels))
            parts.append(f"{len(rest)} more {_PLURAL.get(lab, lab + 's') if len(rest) > 1 else lab}.")
        else:
            parts.append(f"{len(rest)} more objects.")
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
