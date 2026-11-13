"""Does a stronger open-vocabulary detector see what the current detectors miss?

    python scripts/probe_openvocab.py --model google/owlv2-base-patch16-ensemble --thr 0.2

For every `train` / `finetune` weak event that the current perception cannot cover because the named object is not
detectable (door, stairs, pole, fence, wall, ...), run the detector on the review frames inside the event window
(data/review/<split>/frames, from scripts/review_events.py frames) with prompts for that concept, and count the
events where it fires at least once. Writes results/perception/openvocab_probe_<tag>.{json,md} and saves the best
box per event for a visual spot check (data/review/probe/, git-ignored). Firing is not correctness: spot-check.
"""
import argparse, csv, json
from collections import Counter, defaultdict

import torch
from PIL import Image, ImageDraw

from _common import dump_json, ROOT

PROMPTS = {
    "door": ["a door", "a doorway", "an entrance door", "a glass door"],
    "stairs": ["stairs", "a staircase", "steps", "an escalator"],
    "pole": ["a pole", "a lamp post", "a bollard", "a sign post"],
    "fence": ["a fence", "a railing", "a barrier"],
    "wall": ["a wall"],
    "tree": ["a tree trunk", "a tree"],
    "sign": ["a sign", "a street sign"],
    "shelf": ["a shelf", "a store shelf"],
    "elevator": ["an elevator door", "an elevator"],
    "animal": ["a dog", "a horse", "an animal"],
    "ground hazard": ["a puddle", "a hole in the ground", "a wet floor"],
    "machine": ["a machine", "a kiosk", "a vending machine"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
ap.add_argument("--thr", type=float, default=0.2)
ap.add_argument("--tag", default="owlv2")
args = ap.parse_args()

from transformers import Owlv2ForObjectDetection, Owlv2Processor   # noqa: E402

dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = Owlv2Processor.from_pretrained(args.model)
model = Owlv2ForObjectDetection.from_pretrained(args.model).to(dev).eval()

events = []
for split in ("train", "finetune"):
    for r in csv.DictReader(open(ROOT / "results" / "egoblind" / f"review_events_{split}.csv")):
        if r["coverable"] == "True" or r["coverable_reason"] != "not_detectable":
            continue
        concepts = [c for c in (r["concepts"] or "").split(",") if c in PROMPTS]
        if concepts:
            events.append((split, r, concepts))
print(f"{len(events)} not-detectable events with a promptable concept", flush=True)

spot = ROOT / "data" / "review" / "probe"
spot.mkdir(parents=True, exist_ok=True)
per_concept = defaultdict(lambda: [0, 0])
rows = []
for split, r, concepts in events:
    key = f"{int(float(r['video_id']))}_{int(float(r['obj_id']))}"
    fdir = ROOT / "data" / "review" / split / "frames" / key
    meta = fdir / "meta.json"
    if not meta.exists():
        continue
    times = json.loads(meta.read_text())["times"]
    a, b = (float(x) for x in r["window"].split("-"))
    idx = [i for i, t in enumerate(times) if a - 1e-6 <= t <= b + 1e-6] or list(range(len(times)))
    texts = [p for c in concepts for p in PROMPTS[c]]
    best = (0.0, None, None)
    for i in idx:
        img = Image.open(fdir / f"{i:02d}.jpg").convert("RGB")
        inp = proc(text=[texts], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        W, H = img.size
        res = proc.post_process_grounded_object_detection(out, threshold=0.01, target_sizes=[(max(W, H), max(W, H))])[0]
        if len(res["scores"]):
            j = int(res["scores"].argmax())
            s = float(res["scores"][j])
            if s > best[0]:
                best = (s, i, ([float(v) for v in res["boxes"][j]], texts[int(res["labels"][j])]))
    fired = best[0] >= args.thr
    for c in concepts:
        per_concept[c][0] += fired; per_concept[c][1] += 1
    rows.append({"split": split, "key": key, "concepts": ",".join(concepts), "question": r["question"], "best_score": round(best[0], 3),
                 "fired": fired, "prompt": best[2][1] if best[2] else ""})
    if best[1] is not None:
        img = Image.open(fdir / f"{best[1]:02d}.jpg").convert("RGB")
        ImageDraw.Draw(img).rectangle(best[2][0], outline=(255, 140, 0), width=4)
        img.save(spot / f"{key}_{'hit' if fired else 'miss'}_{best[0]:.2f}.jpg")

n_fired = sum(r["fired"] for r in rows)
res = {"model": args.model, "thr": args.thr, "n_events": len(rows), "n_fired": n_fired,
       "per_concept": {c: {"fired": v[0], "n": v[1]} for c, v in per_concept.items()},
       "score_hist": dict(Counter(round(r["best_score"], 1) for r in rows)), "rows": rows}
out = ROOT / "results" / "perception"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / f"openvocab_probe_{args.tag}.json")
lines = [f"Open-vocabulary probe ({args.model}, threshold {args.thr}) on {len(rows)} not-detectable train/finetune events\n",
         f"Fires on {n_fired}/{len(rows)} events ({n_fired / max(1, len(rows)):.0%}). Firing is not correctness; see the spot check.\n",
         "| concept | fired / events |", "|---|---|"] + [f"| {c} | {v[0]}/{v[1]} |" for c, v in sorted(per_concept.items(), key=lambda kv: -kv[1][1])]
(out / f"openvocab_probe_{args.tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
