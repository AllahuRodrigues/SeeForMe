"""Indoor / outdoor scene classification from the frames alone (CLIP ViT-B/32 zero-shot).

Used to choose between the indoor (Hypersim) and outdoor (Virtual KITTI 2) metric-depth models without looking at any
annotation: a deployed system only has the camera. `classify_video` averages the outdoor probability over the frames
of the first `seconds` of a video; `classify_frames` takes BGR frames directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

INDOOR_PROMPTS = ["a photo taken indoors", "a photo of the inside of a building", "a photo of a room", "a photo of a hallway",
                  "a photo inside a shop", "a photo inside a vehicle"]
OUTDOOR_PROMPTS = ["a photo taken outdoors", "a photo of a street", "a photo of a sidewalk", "a photo of a road with cars",
                   "a photo of a park", "a photo of a train station platform outside"]
WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "clip"


class SceneClassifier:
    def __init__(self, device: Optional[str] = None, model: str = "ViT-B/32"):
        import clip
        import torch
        from .detector import pick_device
        self.torch = torch
        self.device = device or pick_device("auto")
        self.model, self.preprocess = clip.load(model, device=self.device, download_root=str(WEIGHTS))
        self.model.eval()
        with torch.no_grad():
            tok = clip.tokenize(INDOOR_PROMPTS + OUTDOOR_PROMPTS).to(self.device)
            t = self.model.encode_text(tok).float()
            self.text = t / t.norm(dim=-1, keepdim=True)
        self.n_in = len(INDOOR_PROMPTS)

    def outdoor_prob(self, frames_bgr: Sequence[np.ndarray]) -> float:
        """Mean P(outdoor) over the frames (softmax over all prompts, outdoor prompts summed)."""
        from PIL import Image
        if not len(frames_bgr):
            return 0.5
        ims = self.torch.stack([self.preprocess(Image.fromarray(f[:, :, ::-1])) for f in frames_bgr]).to(self.device)
        with self.torch.no_grad():
            v = self.model.encode_image(ims).float()
            v = v / v.norm(dim=-1, keepdim=True)
            p = (100.0 * v @ self.text.T).softmax(dim=-1)
        return float(p[:, self.n_in:].sum(dim=-1).mean())

    def classify_video(self, video_path, seconds: float = 2.0, n_frames: int = 5) -> float:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[np.ndarray] = []
        for t in np.linspace(0.0, seconds, n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        cap.release()
        return self.outdoor_prob(frames)
