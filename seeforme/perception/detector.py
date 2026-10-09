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


# Open-vocabulary navigation vocabulary: (text prompt, label the rest of the system sees). Labels reuse the
# names of the P-CACS taxonomy (campus.world.LABEL_EVENT) and of the EgoBlind concept matcher (egoblind.data).
NAV_VOCAB = [
    ("person", "person"), ("bicycle", "bicycle"), ("car", "car"), ("motorcycle", "motorcycle"), ("bus", "bus"), ("truck", "truck"),
    ("train", "train"), ("scooter", "scooter"), ("dog", "animal"), ("cat", "animal"),
    ("chair", "chair"), ("bench", "bench"), ("table", "dining table"), ("backpack", "backpack"), ("suitcase", "suitcase"),
    ("potted plant", "potted plant"), ("traffic light", "traffic light"), ("stop sign", "stop sign"), ("fire hydrant", "fire hydrant"),
    ("stairs", "stairs"), ("escalator", "stairs"), ("door", "door"), ("curb", "curb"), ("pole", "pole"), ("bollard", "pole"),
    ("fence", "fence"), ("railing", "fence"), ("tree trunk", "tree"), ("box", "box"), ("shopping cart", "cart"), ("trash can", "trash can"),
    ("traffic cone", "cone"), ("barrier", "barrier"), ("elevator", "elevator"), ("shelf", "shelf"), ("sign", "sign"),
    ("puddle", "ground hazard"), ("pothole", "ground hazard"), ("crosswalk", "crossing"), ("turnstile", "machine"), ("kiosk", "machine"),
]


class OpenVocabDetector:
    """YOLO-World (Ultralytics) with a fixed navigation vocabulary: same `Detection` output as `YoloDetector`, but it also
    sees stairs, doors, poles, curbs, fences, cones ... that the COCO-trained model cannot name."""

    def __init__(self, cfg: Optional[PerceptionConfig] = None, vocab=None):
        from ultralytics import YOLOWorld
        self.cfg = cfg or PerceptionConfig()
        self.device = pick_device(self.cfg.device)
        self.vocab = list(vocab or NAV_VOCAB)
        self.model = YOLOWorld(self.cfg.open_vocab_model)
        self.model.set_classes([p for p, _ in self.vocab])
        self.names = {i: lab for i, (_, lab) in enumerate(self.vocab)}
        self.vocab_labels = sorted({lab for _, lab in self.vocab})
        self.last_latency_ms = 0.0
        self.model.predict(np.zeros((self.cfg.yolo_imgsz, self.cfg.yolo_imgsz, 3), dtype=np.uint8),
                           device=self.device, verbose=False, imgsz=self.cfg.yolo_imgsz)

    detect = YoloDetector.detect


def make_detector(cfg: Optional[PerceptionConfig] = None):
    """`cfg.detector`: "yolo" (COCO YOLO11) or "yolo-world" (open vocabulary, NAV_VOCAB)."""
    cfg = cfg or PerceptionConfig()
    return OpenVocabDetector(cfg) if cfg.detector == "yolo-world" else YoloDetector(cfg)


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
