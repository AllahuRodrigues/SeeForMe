"""Object detection wrappers producing normalised `Detection`s.

`YoloDetector` uses Ultralytics YOLO (lightweight n/s variants) on CPU, CUDA or
Apple MPS. `MockDetector` replays scripted detections for tests and offline runs.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import Detection


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector:
    def __init__(self, cfg: Optional[PerceptionConfig] = None):
        from ultralytics import YOLO
        self.cfg = cfg or PerceptionConfig()
        self.device = pick_device(self.cfg.device)
        self.model = YOLO(self.cfg.yolo_model)
        self.names = self.model.names
        self.last_latency_ms = 0.0
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((self.cfg.yolo_imgsz, self.cfg.yolo_imgsz, 3), dtype=np.uint8),
                           device=self.device, verbose=False, imgsz=self.cfg.yolo_imgsz)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        res = self.model.predict(frame, device=self.device, conf=self.cfg.yolo_conf, imgsz=self.cfg.yolo_imgsz,
                                 verbose=False)[0]
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        h, w = frame.shape[:2]
        out: List[Detection] = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(Detection(label=self.names[int(k)], confidence=float(c),
                                 bbox=(float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h))))
        return out


class MockDetector:
    """Returns scripted detections; `script` maps frame index -> list of Detection."""

    def __init__(self, script=None):
        self.script = script or {}
        self.i = 0
        self.last_latency_ms = 0.0

    def detect(self, frame) -> List[Detection]:
        dets = self.script.get(self.i, [])
        self.i += 1
        return list(dets)
