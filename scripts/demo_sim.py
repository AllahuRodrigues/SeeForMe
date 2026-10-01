"""Render a synthetic scenario as a video with the debug overlay for each policy.

Useful for the poster/demo when no camera is available: shows exactly why CACS
stays silent where the baselines keep talking.
"""
import argparse
import cv2, numpy as np
from _common import MODELS, RESULTS, load_cfg
from seeforme.sim.scenario import DetectorSimulator, make_scenario
from seeforme.pipeline import make_policy
from seeforme.eval.runner import run_policy_on_frames
from seeforme.ui import draw_overlay

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="hallway"); ap.add_argument("--seed", type=int, default=9000)
ap.add_argument("--policy", default="cacs"); ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = load_cfg()
sc = make_scenario(args.kind, args.seed)
pol = make_policy(args.policy, cfg, str(MODELS))
W, H = 640, 480
out = args.out or str(RESULTS / f"demo_{args.kind}_{args.policy}.mp4")
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), sc.fps, (W, H))
last = [None, -1e9]

def on_step(t, tracks, cues):
    frame = np.full((H, W, 3), 235, dtype=np.uint8)
    for o in sc.objects:  # faint ground-truth boxes
        if o.visible(t):
            b = o.bbox_at(t); col = (60, 60, 200) if o.important_at(t) else (190, 190, 190)
            cv2.rectangle(frame, (int(b[0]*W), int(b[1]*H)), (int(b[2]*W), int(b[3]*H)), col, 1)
    if cues: last[0], last[1] = cues[0], t
    thr = pol.cfg.threshold if hasattr(pol.cfg, "threshold") else 0
    vis = draw_overlay(frame, tracks, pol.last_observations(), last[0] if t - last[1] < 2.0 else None,
                       {"policy": pol.name, "t": f"{t:.1f}s", "scenario": sc.name}, thr)
    writer.write(vis)

cues = run_policy_on_frames(pol, DetectorSimulator(sc).frames(), cfg, on_step=on_step)
writer.release()
print(f"{len(cues)} cues -> {out}")
for c in cues: print(f"  {c.t:5.1f}s  {c.message}")
