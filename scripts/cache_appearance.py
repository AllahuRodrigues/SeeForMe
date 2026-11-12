"""Embed every cached EgoBlind detection with CLIP on the GPU (side-car .npz next to the perception cache).

    python scripts/cache_appearance.py --splits train,finetune,select          # development pool first
    python scripts/cache_appearance.py --splits holdout,test                    # evaluation splits (embedding only, no labels used)

Writes data/egoblind_app/<video_id>.npz with det_emb (N, 512) float16 aligned with the perception cache's
detections, plus meta (encoder, cache build stamp). Resumable: existing side-cars built from the same cache are skipped.
"""
import argparse, json, time
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT
from egoblind_common import CACHE, SCENARIOS, train_pool_assignment
from seeforme.egoblind.cache import PerceptionCache
from seeforme.perception.appearance import AppearanceEncoder, crop_boxes

OUT = ROOT / "data" / "egoblind_app"   # worktree-local (data/egoblind is shared with the main checkout)

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="train,finetune,select")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default=None)
args = ap.parse_args()

assign = train_pool_assignment()
ids = []
for s in args.splits.split(","):
    if s == "test":
        ids += sorted(int(p.stem) for p in (SCENARIOS / "test").glob("*.json"))
    else:
        ids += sorted(v for v, sp in assign.items() if sp == s)
ids = [v for v in ids if (CACHE / f"{v}.npz").exists()]
if args.limit:
    ids = ids[:args.limit]
OUT.mkdir(parents=True, exist_ok=True)
enc = AppearanceEncoder(device=args.device)
print(f"{len(ids)} videos, encoder {enc.name} on {enc.device}", flush=True)


def frames_at(video: str, wanted: dict):
    """Yield (frame_index, bgr) for the wanted frame indices, decoding sequentially."""
    cap = cv2.VideoCapture(video)
    i, last = 0, max(wanted) if wanted else -1
    try:
        while i <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if i in wanted:
                yield i, fr
            i += 1
    finally:
        cap.release()


t_start, n_det, done = time.time(), 0, 0
for k, vid in enumerate(ids):
    out = OUT / f"{vid}.npz"
    cache = PerceptionCache.load(CACHE / f"{vid}.npz")
    built = cache.meta.get("built", "")
    if out.exists():
        with np.load(out, allow_pickle=False) as z:
            if json.loads(str(z["meta"])).get("cache_built") == built:
                continue
    fps = float(cache.meta.get("fps", 30.0))
    want = {int(round(t * fps)): fi for fi, t in enumerate(cache.frame_t)}
    emb = np.zeros((cache.n_detections, 512), dtype=np.float16)
    seen = np.zeros(cache.n_detections, dtype=bool)
    for idx, fr in frames_at(cache.meta["video"], want):
        fi = want[idx]
        lo, hi = int(cache._starts[fi]), int(cache._starts[fi + 1])
        if hi > lo:
            emb[lo:hi] = enc.embed(crop_boxes(fr, cache.det_bbox[lo:hi]))
            seen[lo:hi] = True
    np.savez_compressed(out, det_emb=emb, has_emb=seen,
                        meta=json.dumps({"encoder": enc.name, "cache_built": built, "video_id": vid, "n": int(cache.n_detections),
                                         "n_missing": int((~seen).sum())}))
    n_det += cache.n_detections; done += 1
    if (k + 1) % 25 == 0:
        el = time.time() - t_start
        print(f"  {k + 1}/{len(ids)}  {n_det} detections  {n_det / max(el, 1e-6):.0f} det/s  {el / 60:.1f} min", flush=True)
print(f"done: {done} videos embedded, {n_det} detections, {(time.time() - t_start) / 60:.1f} min")
