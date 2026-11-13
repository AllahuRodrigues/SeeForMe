"""M5: ingest the owner's own chest-mounted recordings (issue #9).

Pure parts, tested in tests/test_ingest_own.py:
- `parse_narration`: the walker says a short keyword at the moment a warning would be wanted ("stairs down",
  "door", "curb", "crossing", "bike left", "clear"). Word-level speech recognition (Vosk, offline) gives each word a
  time; the keywords become timed, in-situ labels: the thing EgoBlind lacks. "clear" marks a real quiet moment.
- `head_region` / `blur_regions`: privacy. Every detected person gets the top of their box blurred before a frame is
  written anywhere; raw video never leaves data/own/raw/ (git-ignored).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# keyword (lower case) -> (label, default detail); a following direction word becomes the detail
NARRATION: Dict[str, Tuple[str, str]] = {
    "stairs": ("stairs", ""), "steps": ("stairs", ""), "escalator": ("stairs", "escalator"),
    "door": ("door", ""), "doorway": ("door", ""), "entrance": ("door", ""), "gate": ("door", "gate"),
    "curb": ("curb", ""), "ramp": ("curb", "ramp"),
    "crossing": ("crossing", ""), "crosswalk": ("crossing", ""), "intersection": ("crossing", ""),
    "clear": ("clear", ""),
    "bike": ("bicycle", ""), "bicycle": ("bicycle", ""), "scooter": ("bicycle", "scooter"),
    "car": ("car", ""), "bus": ("car", "bus"), "truck": ("car", "truck"),
    "pole": ("pole", ""), "post": ("pole", ""), "bollard": ("pole", ""), "sign": ("pole", "sign"),
    "elevator": ("elevator", ""), "bench": ("bench", ""), "chair": ("chair", ""), "table": ("table", ""),
    "person": ("person", ""), "people": ("person", "crowd"), "crowd": ("person", "crowd"), "dog": ("animal", "dog"),
    "construction": ("barrier", "construction"), "barrier": ("barrier", ""), "cone": ("barrier", "cone"),
    "puddle": ("ground hazard", ""), "hole": ("ground hazard", ""), "wet": ("ground hazard", "wet floor"),
    "wall": ("wall", ""), "railing": ("fence", "railing"), "fence": ("fence", ""),
}
DETAILS = {"up", "down", "left", "right", "ahead", "front", "open", "closed"}


def parse_narration(words: Sequence[Dict]) -> List[Dict]:
    """[{word, start, end}] -> [{t, label, detail, phrase}] in time order; unrelated speech is ignored."""
    out: List[Dict] = []
    ws = [(str(x["word"]).lower().strip(".,!?"), float(x["start"]), float(x.get("end", x["start"]))) for x in words]
    i = 0
    while i < len(ws):
        word, t, _ = ws[i]
        if word in NARRATION:
            label, detail = NARRATION[word]
            phrase = word
            if i + 1 < len(ws) and ws[i + 1][0] in DETAILS and ws[i + 1][1] - t <= 1.5:
                detail = ws[i + 1][0]
                phrase = f"{word} {detail}"
                i += 1
            out.append({"t": t, "label": label, "detail": detail, "phrase": phrase})
        i += 1
    return out


def head_region(bbox: Tuple[float, float, float, float], frac: float = 0.3, pad: float = 0.25) -> Tuple[float, float, float, float]:
    """Normalised box of the top `frac` of a person box, widened by `pad` of its width, clamped to the image."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (max(0.0, x1 - pad * w), max(0.0, y1 - 0.05 * h), min(1.0, x2 + pad * w), min(1.0, y1 + frac * h))


def blur_regions(img: np.ndarray, boxes: Sequence[Tuple[float, float, float, float]], k: int = 51) -> np.ndarray:
    """Gaussian-blur each normalised box of an HxWx3 image (returns a copy)."""
    import cv2
    out = img.copy()
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        a, b, c, d = int(x1 * W), int(y1 * H), int(np.ceil(x2 * W)), int(np.ceil(y2 * H))
        if c - a < 2 or d - b < 2:
            continue
        kk = max(3, (min(k, (c - a) | 1, (d - b) | 1)) | 1)
        out[b:d, a:c] = cv2.GaussianBlur(out[b:d, a:c], (kk, kk), 0)
    return out


BATCHES = ("dev", "lockbox")


def batch_dirs(root, batch: str):
    """(raw, processed) folders of one recording batch. The development batch may be looked at and tuned on; the
    lockbox batch is recorded on other routes and is scored once, after models and thresholds are frozen
    (docs/recording-protocol.md). Keeping them in separate folders means no script reads both by accident."""
    from pathlib import Path
    if batch not in BATCHES:
        raise ValueError(f"batch must be one of {BATCHES}, got {batch!r}")
    own = Path(root) / "data" / "own"
    return own / "raw" / batch, own / "processed" / batch
