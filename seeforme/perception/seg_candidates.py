"""Scene segmentation as an event generator (doors, stairs, railings the object detector cannot name).

Input: per-second path-region shares from scripts/cache_segmentation.py (`t`, `share`, `groups`). At a frame time t
the generator looks only at samples at or before t (causal). A group becomes a candidate when its share is at least
`min_share` in the last `min_consecutive` samples (stability). The candidate is a detection covering the path region
(central 60% of the width, lower 70% of the height: where the share was measured), with confidence rising with the
share, and it is linked to weak events exactly like detector output (same label, inside the window +- margin,
nearest question time). Direction and distance are not estimated: depth is left empty.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Detection

SEG_LABELS = {"stairs": "stairs", "door": "door", "railing": "fence"}
PATH_BOX = (0.2, 0.3, 0.8, 1.0)
SOURCE_MARGIN_S = 2.0


def seg_detections(t: float, z, min_share: float = 0.02, min_consecutive: int = 2,
                   labels: Optional[Dict[str, str]] = None) -> List[Detection]:
    labels = labels or SEG_LABELS
    ts = np.asarray(z["t"], float)
    i = int(np.searchsorted(ts, t + 1e-6)) - 1
    if i + 1 < min_consecutive:
        return []
    groups = [str(g) for g in z["groups"]]
    recent = np.asarray(z["share"])[i - min_consecutive + 1:i + 1]
    out = []
    for g, lab in labels.items():
        if g not in groups:
            continue
        col = recent[:, groups.index(g)]
        if (col >= min_share).all():
            share = float(col[-1])
            out.append(Detection(label=lab, confidence=float(min(0.9, 0.4 + 2.0 * share)), bbox=PATH_BOX))
    return out


def _link(lab: str, t: float, events: Sequence, margin: float = SOURCE_MARGIN_S) -> Optional[int]:
    cands = [e for e in events if getattr(e, "should_speak", False)
             and lab in ((e.detail or {}).get("det_labels") or (e.detail or {}).get("labels", []))
             and e.earliest_useful_s - margin <= t <= e.latest_safe_s + margin]
    if not cands:
        return None
    return min(cands, key=lambda e: abs((e.detail or {}).get("t_q", e.latest_safe_s) - t)).obj_id


def inject_segmentation(frames: List[Tuple[float, List[Detection]]], z, events: Sequence, **kw) -> List[Tuple[float, List[Detection]]]:
    """Append segmentation candidates to each frame's detections (detector output is left untouched)."""
    out = []
    for t, dets in frames:
        extra = seg_detections(t, z, **kw)
        for d in extra:
            d.source_id = _link(d.label, t, events)
        out.append((t, list(dets) + extra))
    return out
