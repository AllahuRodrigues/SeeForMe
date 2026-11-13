"""Per-second scene-segmentation shares for EgoBlind videos (input to the feature gate).

    python scripts/cache_segmentation.py --splits train,finetune --fps 1

Decodes each video at --fps, segments with SegFormer (ADE20K) on the GPU, and stores, per sampled time, the share of
the path region (central 60% of the width, lower 70% of the height) covered by each group below.
Output: data/egoblind_seg/<video_id>.npz with `t` (N,) and `share` (N, G), plus `groups`. Resumable; git-ignored.
"""
import argparse, json, time

import cv2
import numpy as np
import torch
from PIL import Image

from _common import ROOT

GROUPS = {
    "stairs": ["stairs", "stairway", "step", "escalator"],
    "door": ["door", "screen door"],
    "railing": ["railing", "fence", "bannister"],
    "walkable": ["floor", "sidewalk", "road", "path", "earth", "grass", "rug", "field", "dirt track", "runway"],
    "vehicle": ["car", "bus", "truck", "van", "minibike", "bicycle"],
    "person": ["person"],
    "obstacle": ["pole", "column", "streetlight", "signboard", "box", "chair", "table", "plant", "trash can", "bench"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="train,finetune")
ap.add_argument("--fps", type=float, default=1.0)
ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-ade-512-512")
args = ap.parse_args()

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation   # noqa: E402

dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(args.model)
model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(dev).eval()
name_to_id = {str(n).strip().lower(): int(i) for i, n in model.config.id2label.items()}
gids = [[name_to_id[c] for c in cs if c in name_to_id] for cs in GROUPS.values()]
out_dir = ROOT / "data" / "egoblind_seg"
out_dir.mkdir(parents=True, exist_ok=True)

vids = []
for s in args.splits.split(","):
    for f in sorted((ROOT / "data" / "egoblind" / "scenarios" / s).glob("*.json")):
        d = json.loads(f.read_text())
        p = ROOT / "data" / "egoblind" / "videos" / s.replace("finetune", "train").replace("select", "train").replace("holdout", "train") / f"{d['video_id']:05d}.mp4"
        if not p.exists():
            p = ROOT / "data" / "egoblind" / "videos" / "train" / f"{d['video_id']:05d}.mp4"
        vids.append((d["video_id"], p))
print(f"{len(vids)} videos, {args.fps} fps, {dev}", flush=True)
t0, frames = time.time(), 0
for k, (vid, path) in enumerate(vids):
    dst = out_dir / f"{vid}.npz"
    if dst.exists() or not path.exists():
        continue
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.fps)))
    ts, shares, i = [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                img.thumbnail((640, 640))
                with torch.no_grad():
                    logits = model(**proc(images=img, return_tensors="pt").to(dev)).logits
                seg = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
                h, w = seg.shape
                reg = seg[int(0.3 * h):, int(0.2 * w):int(0.8 * w)]
                shares.append([float(np.isin(reg, g).mean()) for g in gids])
                ts.append(i / fps)
                frames += 1
        i += 1
    cap.release()
    np.savez_compressed(dst, t=np.array(ts, np.float32), share=np.array(shares, np.float32), groups=np.array(list(GROUPS)))
    if (k + 1) % 25 == 0:
        el = time.time() - t0
        print(f"   {k + 1}/{len(vids)}  {frames} frames  {frames / max(el, 1e-6):.1f} f/s  {el / 60:.1f} min", flush=True)
print(f"done: {frames} frames in {(time.time() - t0) / 60:.1f} min")
