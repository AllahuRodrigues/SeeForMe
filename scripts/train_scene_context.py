"""Fit the unsupervised scene-context model (k-means over detection histograms)."""
import argparse, json
from _common import MODELS, RESULTS, dump_json, load_cfg, load_split
from seeforme.ml.scene_context import SceneContextModel, frames_from_scenarios

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=4); args = ap.parse_args()
frames = frames_from_scenarios(load_split("train"), load_cfg())
report = {"n_frames": len(frames), "candidates": []}
best = None
for k in range(2, 8):
    m = SceneContextModel(k=k).fit(frames)
    report["candidates"].append({"k": k, "inertia": m.inertia_, "clusters": m.cluster_names})
    if k == args.k:
        best = m
best.save(str(MODELS / "scene_context.joblib"))
report["chosen"] = best.summary()
dump_json(report, RESULTS / "scene_context_report.json")
print(json.dumps(report, indent=1))
