"""Appearance embeddings for detections (CLIP ViT-B/32 image tower, unit-normalised, float16).

Used by the long-term re-identification memory (seeforme.tracking.reid): an object that left the frame minutes ago can
be recognised when it comes back, so the assistant does not announce it as new. Runs on the GPU (MPS / CUDA) when one
is available. `crop_boxes` is pure and testable without the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "clip"
EMB_DIM = 512
MIN_SIDE_PX = 8


def crop_boxes(frame_bgr: np.ndarray, boxes: Sequence[Tuple[float, float, float, float]], pad: float = 0.1) -> List[np.ndarray]:
    """RGB crops for normalised x1,y1,x2,y2 boxes, padded by `pad` of the box size and clamped to the frame.
    Boxes smaller than MIN_SIDE_PX on a side are grown around their centre so every box yields a crop."""
    h, w = frame_bgr.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = (x2 - x1) * w, (y2 - y1) * h
        cx, cy = (x1 + x2) * 0.5 * w, (y1 + y2) * 0.5 * h
        half_w = max(MIN_SIDE_PX, bw * (1 + 2 * pad)) / 2
        half_h = max(MIN_SIDE_PX, bh * (1 + 2 * pad)) / 2
        a, b = int(max(0, cx - half_w)), int(min(w, cx + half_w))
        c, d = int(max(0, cy - half_h)), int(min(h, cy + half_h))
        crop = frame_bgr[c:max(d, c + 1), a:max(b, a + 1)]
        out.append(crop[:, :, ::-1].copy())
    return out


class AppearanceEncoder:
    def __init__(self, device: Optional[str] = None, model: str = "ViT-B/32", batch: int = 256):
        import clip
        import torch
        from .detector import pick_device
        self.torch = torch
        self.device = device or pick_device("auto")
        self.model, self.preprocess = clip.load(model, device=self.device, download_root=str(WEIGHTS))
        self.model.eval()
        self.batch = batch
        self.name = f"clip-{model}"

    def embed(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """(N, 512) float16 unit vectors."""
        from PIL import Image
        if not len(crops_rgb):
            return np.zeros((0, EMB_DIM), dtype=np.float16)
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(crops_rgb), self.batch):
                x = self.torch.stack([self.preprocess(Image.fromarray(c)) for c in crops_rgb[i:i + self.batch]]).to(self.device)
                e = self.model.encode_image(x).float()
                e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                outs.append(e.cpu().numpy())
        return np.concatenate(outs).astype(np.float16)
