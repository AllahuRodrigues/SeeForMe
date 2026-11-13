"""Per-frame camera ego-motion for EgoBlind videos (input to the feature gate, `feature_auc.py --ego`).

    OMP_NUM_THREADS=2 python scripts/cache_egomotion.py --splits train,finetune --fps 2 [--shard 0/3]

Decodes each video at --fps, resizes to 640 px wide, and runs seeforme.perception.egomotion.estimate_pair on each
consecutive sampled pair (Shi-Tomasi + pyramidal LK + RANSAC homography). The sample at t uses the frames at the
previous sampled time and t only.
Output: data/egoblind_ego/<video_id>.npz with `t` (N,), `feats` (N, F), `fields`, `fps`, and timing
(`ms_pairs`: summed estimate time, `ms_decode`). Resumable; git-ignored (info/exclude).
"""
import argparse, json, time

import cv2
import numpy as np

from _common import ROOT
from seeforme.perception.egomotion import EgoSample, estimate_pair, intrinsics, prepare_frame

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="train,finetune")
ap.add_argument("--fps", type=float, default=2.0)
ap.add_argument("--shard", default="0/1", help="k/n: process every n-th video starting at k")
args = ap.parse_args()
cv2.setNumThreads(2)
k_sh, n_sh = (int(x) for x in args.shard.split("/"))

out_dir = ROOT / "data" / "egoblind_ego"
out_dir.mkdir(parents=True, exist_ok=True)
vids = []
for s in args.splits.split(","):
    for f in sorted((ROOT / "data" / "egoblind" / "scenarios" / s).glob("*.json")):
        d = json.loads(f.read_text())
        p = ROOT / "data" / "egoblind" / "videos" / "train" / f"{d['video_id']:05d}.mp4"
        vids.append((d["video_id"], p))
vids = vids[k_sh::n_sh]
print(f"{len(vids)} videos (shard {args.shard}), {args.fps} fps", flush=True)
t0, pairs, ms_est_all = time.time(), 0, 0.0
for k, (vid, path) in enumerate(vids):
    dst = out_dir / f"{vid}.npz"
    if dst.exists() or not path.exists():
        continue
    td = time.time()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    step = max(1, int(round(fps / args.fps)))
    ts, rows, i, prev, prev_t, K, ms_est = [], [], 0, None, None, None, 0.0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                g = prepare_frame(fr)
                t = i / fps
                if K is None:
                    K = intrinsics(g.shape[1], g.shape[0])
                if prev is not None and prev.shape == g.shape:
                    te = time.perf_counter()
                    s = estimate_pair(prev, g, t - prev_t, K)
                    ms_est += 1000 * (time.perf_counter() - te)
                    ts.append(t)
                    rows.append(s.as_row())
                prev, prev_t = g, t
        i += 1
    cap.release()
    ms_total = 1000 * (time.time() - td)
    np.savez_compressed(dst, t=np.array(ts, np.float32), feats=np.array(rows, np.float32).reshape(-1, len(EgoSample.FIELDS)),
                        fields=np.array(EgoSample.FIELDS), fps=np.float32(fps / step), ms_pairs=np.float32(ms_est),
                        ms_decode=np.float32(ms_total - ms_est))
    pairs += len(ts)
    ms_est_all += ms_est
    if (k + 1) % 25 == 0:
        el = time.time() - t0
        print(f"   {k + 1}/{len(vids)}  {pairs} pairs  {ms_est_all / max(pairs, 1):.1f} ms/pair  {el / 60:.1f} min", flush=True)
print(f"done: {pairs} pairs in {(time.time() - t0) / 60:.1f} min, {ms_est_all / max(pairs, 1):.1f} ms/pair (estimate only)")
