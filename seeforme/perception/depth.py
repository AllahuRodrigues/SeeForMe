"""Metric depth for the live pipeline.

`DepthAnythingEstimator` runs Depth Anything V2 (metric, small: ~25M parameters) through
transformers on MPS / CUDA / CPU and returns a dense depth map in metres. `object_distances`
assigns every detection a robust distance (lower-quartile depth inside the central part of the
box, so the object rather than the background behind it wins) and a confidence that falls
with range and with the spread of depth inside the box. `BBoxDepthEstimator` is the fallback
that uses projected height and a typical object size; it reports low confidence so the message
composer hedges ("close" instead of "about 2 meters").
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ..geometry.camera import CameraModel
from ..types import Detection

INDOOR = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
OUTDOOR = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def object_distances(depth: np.ndarray, dets: List[Detection], max_range: float = 20.0) -> None:
    """Fill det.distance_m / det.depth_conf in place from a dense metric depth map (metres)."""
    h, w = depth.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        # central 60% of the box: avoids background pixels along the edges
        cx1, cx2 = x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)
        cy1, cy2 = y1 + 0.2 * (y2 - y1), y2 - 0.2 * (y2 - y1)
        px1, px2 = int(cx1 * w), max(int(cx1 * w) + 1, int(cx2 * w))
        py1, py2 = int(cy1 * h), max(int(cy1 * h) + 1, int(cy2 * h))
        patch = depth[py1:py2, px1:px2]
        patch = patch[np.isfinite(patch) & (patch > 0.05)]
        if patch.size < 4:
            d.distance_m, d.depth_conf = None, 0.0
            continue
        z = float(np.percentile(patch, 30))          # nearest plausible surface in the box
        spread = float(np.percentile(patch, 70) - np.percentile(patch, 30)) / max(z, 0.3)
        conf = 0.95 - 0.03 * z - 0.6 * min(1.0, spread)
        if z > max_range:
            conf *= 0.5
        d.distance_m, d.depth_conf = z, float(max(0.05, min(0.98, conf)))


class DepthAnythingEstimator:
    def __init__(self, model: str = INDOOR, device: Optional[str] = None, input_size: int = 308):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(device).eval()
        self.input_size = input_size
        self.last_latency_ms = 0.0
        self.name = model
        self(np.zeros((240, 320, 3), dtype=np.uint8))   # warm-up

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Dense metric depth (metres) at the frame's resolution."""
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt", size={"height": self.input_size, "width": self.input_size},
                                do_resize=True, keep_aspect_ratio=False, ensure_multiple_of=14)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        pred = out.predicted_depth
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = self.torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        depth = pred.float().cpu().numpy()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return depth


class BBoxDepthEstimator:
    """No depth model: range from projected height and a typical object height, low confidence."""

    name = "bbox"
    last_latency_ms = 0.0

    def __init__(self, camera: Optional[CameraModel] = None):
        self.cam = camera or CameraModel()

    def annotate(self, dets: List[Detection]) -> None:
        for d in dets:
            d.distance_m, d.depth_conf = self.cam.distance_from_bbox(d.bbox, d.label), 0.25
