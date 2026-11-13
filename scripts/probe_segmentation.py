"""Does dense scene segmentation see what the detectors miss, and does it say anything about "is the path clear"?

    python scripts/probe_segmentation.py --model nvidia/segformer-b2-finetuned-ade-512-512

Part 1, architectural objects. For the `train` / `finetune` weak events whose object no detector can name (door,
stairs, pole, fence, wall, ...), segment the review frames inside the event window and call the event "seen" if the
concept's ADE20K classes cover at least --min-frac of the path region (central 60% of the width, lower 70% of the
height) in some frame. Control: the same test on review frames of the other events, which did not ask about that
concept, gives the base rate. A concept is only informative if its hit rate clearly exceeds its base rate.

Part 2, scene-level questions. For events with no object named ("is the path clear?", "any obstacles?") and for the
user-confirmed quiet moments of the blind audit, report the walkable share (floor, sidewalk, road, path, ...) of the
path region. Writes results/perception/segmentation_probe_<tag>.{json,md}.
"""
import argparse, csv, json, statistics
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from _common import dump_json, ROOT

CONCEPT_CLASSES = {
    "door": ["door", "double door", "screen door"],
    "stairs": ["stairs", "stairway", "step", "escalator"],
    "pole": ["pole", "column", "streetlight"],
    "fence": ["fence", "railing", "bannister"],
    "wall": ["wall"],
    "tree": ["tree", "palm"],
    "sign": ["signboard", "poster"],
    "shelf": ["shelf", "case", "cabinet"],
    "animal": ["animal"],
}
WALKABLE = ["floor", "sidewalk", "road", "path", "earth", "grass", "rug", "field", "dirt track", "runway"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-ade-512-512")
ap.add_argument("--min-frac", type=float, default=0.02)
ap.add_argument("--tag", default="segformer_b2")
args = ap.parse_args()

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation   # noqa: E402

dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(args.model)
model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(dev).eval()
name_to_id = {}
for i, n in model.config.id2label.items():
    for part in str(n).split(";"):
        name_to_id.setdefault(part.strip().lower(), int(i))
missing = sorted(({c for cs in CONCEPT_CLASSES.values() for c in cs} | set(WALKABLE)) - set(name_to_id))
if missing:
    print("ADE20K has no class named:", missing)
cls_ids = {k: [name_to_id[c] for c in v if c in name_to_id] for k, v in CONCEPT_CLASSES.items()}
walk_ids = [name_to_id[c] for c in WALKABLE if c in name_to_id]


def segment(path):
    img = Image.open(path).convert("RGB")
    inp = proc(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = model(**inp).logits
    up = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    seg = up.argmax(1)[0].cpu().numpy()
    h, w = seg.shape
    return seg[int(0.3 * h):, int(0.2 * w):int(0.8 * w)]        # path region


def window_frames(split, r):
    key = f"{int(float(r['video_id']))}_{int(float(r['obj_id']))}"
    fdir = ROOT / "data" / "review" / split / "frames" / key
    if not (fdir / "meta.json").exists():
        return []
    times = json.loads((fdir / "meta.json").read_text())["times"]
    try:
        a, b = (float(x) for x in r["window"].split("-"))
    except ValueError:
        a, b = -1e9, 1e9
    idx = [i for i, t in enumerate(times) if a - 1e-6 <= t <= b + 1e-6] or list(range(len(times)))
    return [fdir / f"{i:02d}.jpg" for i in idx]


rows = []
for split in ("train", "finetune"):
    rows += [(split, r) for r in csv.DictReader(open(ROOT / "results" / "egoblind" / f"review_events_{split}.csv"))]
print(f"{len(rows)} events; segmenting review frames on {dev}", flush=True)

frac_cache = {}
for k, (split, r) in enumerate(rows):
    fr = window_frames(split, r)
    per = []
    for f in fr:
        reg = segment(f)
        per.append({**{c: float(np.isin(reg, ids).mean()) for c, ids in cls_ids.items()}, "walkable": float(np.isin(reg, walk_ids).mean())})
    frac_cache[(split, r["video_id"], r["obj_id"])] = per
    if (k + 1) % 50 == 0:
        print(f"   {k + 1}/{len(rows)}", flush=True)

res = {"model": args.model, "min_frac": args.min_frac, "concepts": {}, "scene": {}}
for c in CONCEPT_CLASSES:
    hit_n = hit_k = base_n = base_k = 0
    for split, r in rows:
        per = frac_cache[(split, r["video_id"], r["obj_id"])]
        if not per:
            continue
        seen = max(p[c] for p in per) >= args.min_frac
        concepts = (r["concepts"] or "").split(",")
        if c in concepts and r["coverable_reason"] == "not_detectable":
            hit_n += 1; hit_k += seen
        elif c not in concepts:
            base_n += 1; base_k += seen
    if hit_n:
        res["concepts"][c] = {"hit": hit_k, "n": hit_n, "hit_rate": hit_k / hit_n, "base_rate": base_k / max(1, base_n), "base_n": base_n}

# scene-level: walkable share for no-object questions vs everything else; quiet moments from the audit frames
scene = defaultdict(list)
for split, r in rows:
    per = frac_cache[(split, r["video_id"], r["obj_id"])]
    if per:
        grp = "no_object_named" if r["coverable_reason"] == "no_object_named" else "other_speak_events"
        scene[grp].append(statistics.mean(p["walkable"] for p in per))
audit = ROOT / "results" / "egoblind" / "review_events_audit.csv"
if audit.exists():
    for r in csv.DictReader(open(audit)):
        if r.get("derived:label") != "silent":
            continue
        fr = window_frames("audit", r)
        if fr:
            scene["user_confirmed_quiet"].append(statistics.mean(float(np.isin(segment(f), walk_ids).mean()) for f in fr))
res["scene"] = {g: {"n": len(v), "mean_walkable": statistics.mean(v), "median_walkable": statistics.median(v)} for g, v in scene.items() if v}

out = ROOT / "results" / "perception"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / f"segmentation_probe_{args.tag}.json")
lines = [f"Segmentation probe ({args.model}); concept seen if its classes cover >= {args.min_frac:.0%} of the path region in a window frame\n",
         "| concept | seen on its not-detectable events | base rate on events not about it |", "|---|---|---|"]
for c, v in sorted(res["concepts"].items(), key=lambda kv: -kv[1]["n"]):
    lines.append(f"| {c} | {v['hit']}/{v['n']} ({v['hit_rate']:.0%}) | {v['base_rate']:.0%} (n={v['base_n']}) |")
lines += ["", "Walkable share of the path region (floor, sidewalk, road, path, ...):", "", "| group | n | mean | median |", "|---|---|---|---|"]
for g, v in res["scene"].items():
    lines.append(f"| {g} | {v['n']} | {v['mean_walkable']:.2f} | {v['median_walkable']:.2f} |")
(out / f"segmentation_probe_{args.tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
