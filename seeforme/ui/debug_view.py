"""OpenCV debugging overlay: bounding boxes, per-track CACS features/scores, and
the cue that was selected. Makes the algorithm's decisions visible (section 4.1)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from ..types import Cue, Observation, Track

COLORS = {"object": (60, 200, 60), "text": (230, 160, 30)}


def draw_overlay(frame: np.ndarray, tracks: List[Track], observations: List[Observation],
                 selected: Optional[Cue], stats: dict, threshold: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    score_by_track = {o.track.track_id: o for o in observations}
    for tr in tracks:
        x1, y1, x2, y2 = int(tr.bbox[0] * w), int(tr.bbox[1] * h), int(tr.bbox[2] * w), int(tr.bbox[3] * h)
        col = COLORS.get(tr.kind, (200, 200, 200))
        if selected is not None and tr.track_id == selected.track_id:
            col = (40, 60, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"#{tr.track_id} {tr.text if tr.kind == 'text' and tr.text else tr.label} {tr.confidence:.2f}"
        o = score_by_track.get(tr.track_id)
        if o is not None:
            f = o.features
            label += f" S={o.score:.2f}"
            detail = f"C{f.confidence:.2f} R{f.relevance:.2f} N{f.novelty:.2f} P{f.proximity:.2f} T{f.text_importance:.2f} D{f.duplicate:.2f}"
            cv2.putText(out, detail, (x1, min(h - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # header
    cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), -1)
    line = " | ".join(f"{k}: {v}" for k, v in stats.items())
    cv2.putText(out, line, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if selected is not None:
        cv2.putText(out, f"SPEAK: {selected.message}  (score {selected.score:.2f} >= {threshold})",
                    (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 60, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, f"silent (no cue above threshold {threshold})", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 160), 1, cv2.LINE_AA)
    # ranked list on the right
    y = 60
    for o in observations[:6]:
        txt = f"{o.score:5.2f} {o.track.text if o.track.kind=='text' and o.track.text else o.track.label}"
        cv2.putText(out, txt, (w - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return out


def draw_overlay_pcacs(frame: np.ndarray, tracks: List[Track], observations: list, selected: Optional[Cue], stats: dict,
                       policy) -> np.ndarray:
    """Overlay for P-CACS: distance / TTC / path probability / utility per track, the action taken, budget bar."""
    if not hasattr(policy, "last_features"):
        thr = getattr(getattr(policy, "cfg", None), "threshold", 0)
        return draw_overlay(frame, tracks, observations, selected, stats, thr)
    out = frame.copy()
    h, w = out.shape[:2]
    by_track = {o.track.track_id: o for o in observations}
    for tr in tracks:
        x1, y1, x2, y2 = int(tr.bbox[0] * w), int(tr.bbox[1] * h), int(tr.bbox[2] * w), int(tr.bbox[3] * h)
        col = COLORS.get(tr.kind, (200, 200, 200))
        if selected is not None and tr.track_id == selected.track_id:
            col = (40, 60, 255) if selected.action != "haptic" else (255, 120, 40)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        name = tr.text if tr.kind == "text" and tr.text else tr.label
        d = f"{tr.distance_m:.1f}m" if tr.distance_m is not None else "?"
        ttc = f"{tr.ttc_s:.1f}s" if tr.ttc_s != float("inf") else "-"
        o = by_track.get(tr.track_id)
        label = f"#{tr.track_id} {name} {tr.confidence:.2f} | {d} ttc {ttc} path {tr.path_prob:.2f}"
        if o is not None:
            label += f" | U={o.score:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(out, " | ".join(f"{k}: {v}" for k, v in stats.items()), (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if selected is not None:
        txt = f"{selected.action.upper()}: {selected.message or '(buzz)'}  U={selected.utility:.2f}"
        cv2.putText(out, txt, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 60, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "silent", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    # speech budget bar
    cap = getattr(policy.cfg, "budget_capacity_s", 1.0)
    frac = max(0.0, min(1.0, getattr(policy, "budget", cap) / cap))
    cv2.rectangle(out, (w - 130, 50), (w - 10, 60), (80, 80, 80), 1)
    cv2.rectangle(out, (w - 130, 50), (w - 130 + int(120 * frac), 60), (60, 200, 60), -1)
    cv2.putText(out, "speech budget", (w - 130, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    y = 92
    for o in observations[:6]:
        f = getattr(o, "event", None)
        name = o.track.text if o.track.kind == "text" and o.track.text else o.track.label
        extra = f" H{f.hazard:.2f} u{f.urgency:.2f}" if f is not None else ""
        cv2.putText(out, f"{o.score:5.2f} {name}{extra}", (w - 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return out
