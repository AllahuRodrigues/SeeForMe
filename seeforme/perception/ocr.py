"""OCR wrapper (EasyOCR) producing text `Detection`s.

OCR is much slower than detection, so it runs on every N-th frame; the tracker
keeps text tracks alive between OCR passes.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from ..config import PerceptionConfig
from ..types import Detection


class EasyOcrReader:
    def __init__(self, cfg: Optional[PerceptionConfig] = None, gpu: Optional[bool] = None):
        import easyocr
        self.cfg = cfg or PerceptionConfig()
        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
            except Exception:
                gpu = False
        self.reader = easyocr.Reader(list(self.cfg.ocr_languages), gpu=gpu, verbose=False)
        self.last_latency_ms = 0.0

    def read(self, frame: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        results = self.reader.readtext(frame, paragraph=False)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        out: List[Detection] = []
        for box, text, conf in results:
            if conf < self.cfg.ocr_min_conf or not text.strip():
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            out.append(Detection(label="text", confidence=float(conf), kind="text", text=text.strip(),
                                 bbox=(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)))
        return out


class NullOcr:
    last_latency_ms = 0.0

    def read(self, frame) -> List[Detection]:
        return []
