"""Slot-based minimal message composer shared by P-CACS and the CampusBench annotations.

A message is built from semantic slots so the scheduler can decide *how much* to say:

    object     "stairs down", "chair", "sign reads ROOM 204"
    distance   "about 3 meters" / "right in front of you" (hedged when depth is uncertain)
    direction  "slightly right", "on your left"
    motion     "approaching", "crossing from the right"
    action     "handrail left", "step down"
    detail     "handle on the right", "push"

The same function renders the ground-truth ideal cue in the simulator and the live message,
so the "minimum message ratio" metric compares like with like.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

SLOTS = ("object", "distance", "direction", "motion", "action", "detail")

STAIRS_WORDS = {"down": "stairs down", "up": "stairs up"}


def object_phrase(label: str, kind: str = "object", text: Optional[str] = None, detail: Optional[Dict] = None,
                  hedged: bool = False) -> str:
    detail = detail or {}
    if kind == "text":
        txt = (text or "").strip()
        core = f"sign reads {txt}" if txt else "a sign"
        return f"possibly a sign, {txt}" if hedged and txt else core
    label = label.replace("_", " ")
    if label in ("unknown object", "obstruction"):
        return "possibly an obstacle" if hedged else "an unknown obstacle"
    if label == "stairs":
        label = STAIRS_WORDS.get(str(detail.get("direction", "")), "stairs")
    elif label == "curb":
        label = "curb"
    elif label == "person" and detail.get("count", 1) > 1:
        label = "people"
    return f"possibly a {label}" if hedged else label


def distance_phrase(distance_m: Optional[float], depth_conf: float = 1.0, units: str = "m") -> str:
    if distance_m is None:
        return ""
    z = float(distance_m)
    if depth_conf < 0.45:                       # only a bbox guess: say it coarsely, never a number
        if z < 1.2:
            return "right in front of you"
        return "very close" if z < 2.5 else ("close" if z < 4.5 else "")
    if z < 1.0:
        return "right in front of you"
    if units == "ft":
        ft = z * 3.28084
        ft = round(ft) if ft < 10 else 5 * round(ft / 5)
        return f"about {int(ft)} feet"
    m = round(z * 2) / 2 if z < 3.0 else round(z)
    m_txt = f"{m:.1f}".rstrip("0").rstrip(".") if z < 3.0 else f"{int(m)}"
    return f"about {m_txt} meter{'s' if m_txt != '1' else ''}"


def direction_phrase(angle_deg: float) -> str:
    a = float(angle_deg)
    if abs(a) < 10:
        return "ahead"
    side = "right" if a > 0 else "left"
    return f"slightly {side}" if abs(a) < 25 else f"on your {side}"


def motion_phrase(rel_speed: float, lateral_speed: float, lateral_m: float, approaching_fast: float = 1.5) -> str:
    if abs(lateral_speed) > 0.4 and lateral_m * lateral_speed < 0:
        return f"crossing from the {'right' if lateral_m > 0 else 'left'}"
    if rel_speed < -approaching_fast:
        return "approaching"
    if abs(lateral_speed) > 0.4:
        return f"passing on your {'right' if lateral_m > 0 else 'left'}"
    return ""


def action_phrase(label: str, detail: Optional[Dict]) -> str:
    detail = detail or {}
    if label == "stairs":
        parts = []
        if detail.get("handrail"):
            parts.append(f"handrail {detail['handrail']}")
        return ". ".join(parts)
    if label == "curb":
        return f"step {detail.get('direction', 'down')}"
    if label == "door" and detail.get("open") is False:
        return "closed"
    return str(detail.get("action", ""))


def detail_phrase(label: str, detail: Optional[Dict]) -> str:
    detail = detail or {}
    if label == "door":
        bits = []
        if detail.get("handle"):
            bits.append(f"handle on the {detail['handle']}")
        if detail.get("push_pull"):
            bits.append(detail["push_pull"])
        if detail.get("automatic"):
            bits.append("automatic")
        return ", ".join(bits)
    return str(detail.get("detail", ""))


def compose(slots: Iterable[str], label: str, kind: str = "object", text: Optional[str] = None,
            distance_m: Optional[float] = None, depth_conf: float = 1.0, angle_deg: float = 0.0,
            rel_speed: float = 0.0, lateral_speed: float = 0.0, lateral_m: float = 0.0,
            detail: Optional[Dict] = None, hedged: bool = False, units: str = "m") -> str:
    """Render the requested slots (missing/empty slots are skipped)."""
    slots = set(slots)
    first = []
    if "object" in slots or True:   # the object is always named
        first.append(object_phrase(label, kind, text, detail, hedged))
    if "distance" in slots:
        d = distance_phrase(distance_m, depth_conf, units)
        if d:
            first.append(d)
    if "direction" in slots:
        first.append(direction_phrase(angle_deg))
    if "motion" in slots:
        m = motion_phrase(rel_speed, lateral_speed, lateral_m)
        if m:
            first.append(m)
    sentences = [", ".join(first)]
    if "action" in slots:
        a = action_phrase(label, detail)
        if a:
            sentences.append(a)
    if "detail" in slots:
        d = detail_phrase(label, detail)
        if d:
            sentences.append(d)
    msg = ". ".join(s[0].upper() + s[1:] for s in sentences if s)
    return msg + "." if not msg.endswith(".") else msg


def speech_seconds(message: str, min_gap: float = 1.2, seconds_per_word: float = 0.35) -> float:
    return max(min_gap, len(message.split()) * seconds_per_word)
