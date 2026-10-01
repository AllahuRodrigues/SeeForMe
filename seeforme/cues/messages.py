"""Turn a track into a short spoken message with a coarse spatial hint."""
from __future__ import annotations

from ..types import Track, bbox_center, bbox_area


def direction_word(track: Track) -> str:
    cx, _ = bbox_center(track.bbox)
    if cx < 0.33:
        return "on your left"
    if cx > 0.67:
        return "on your right"
    return "ahead"


def distance_word(track: Track) -> str:
    a = bbox_area(track.bbox)
    if a > 0.18:
        return "very close"
    if a > 0.06:
        return "close"
    return ""


def compose_message(track: Track, hedged: bool = False) -> str:
    if track.kind == "text":
        txt = (track.text or "").strip()
        core = f"sign reads {txt}" if txt else "a sign"
        return (f"possibly a sign, {txt}" if hedged and txt else core) + f", {direction_word(track)}"
    dist = distance_word(track)
    label = track.label.replace("_", " ")
    lead = f"possibly a {label}" if hedged else label
    parts = [lead, direction_word(track)]
    if dist:
        parts.append(dist)
    return ", ".join(parts)
