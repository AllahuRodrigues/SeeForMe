"""Can a permissively licensed detector replace YOLO11 (AGPL-3.0) for the product track? (issue #10)

    python scripts/compare_detectors.py

Same frames for every detector: the review frames inside each `train` / `finetune` weak event's window
(data/review/<split>/frames, from scripts/review_events.py). An event is "seen" by a detector if, in some window frame,
it outputs one of the event's object labels at confidence >= --conf (the P-CACS minimum). Only events whose label is a
COCO class are scored (doors, stairs, poles are outside every COCO detector: see the segmentation and OWLv2 probes).
Also reports mean detections per frame and wall time per frame on this Mac.
Writes results/perception/detector_comparison.{json,md}.
"""
import argparse, csv, json, os, time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from _common import dump_json, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--models", default="yolo11,rtdetr_r18,dfine_s")
args = ap.parse_args()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# the Hugging Face COCO configs use Pascal-VOC-style names for six classes; map them to the COCO names the rest of the
# system uses, so every detector is scored on the same events
NORMALISE = {"motorbike": "motorcycle", "diningtable": "dining table", "sofa": "couch", "tvmonitor": "tv",
             "pottedplant": "potted plant", "aeroplane": "airplane"}
HF = {"rtdetr_r18": ("PekingU/rtdetr_r18vd", "Apache-2.0"), "dfine_s": ("ustc-community/dfine-small-coco", "Apache-2.0")}


def load(name):
    if name == "yolo11":
        from seeforme.perception.detector import YoloDetector
        here = os.getcwd(); os.chdir(ROOT / "weights")
        try:
            det = YoloDetector()
        finally:
            os.chdir(here)
        names = set(det.names.values())

        def run(img):
            arr = np.asarray(img)[:, :, ::-1].copy()
            return [(d.label, d.confidence) for d in det.detect(arr)]
        return run, names, "AGPL-3.0 (Ultralytics)"
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    rid, lic = HF[name]
    proc = AutoImageProcessor.from_pretrained(rid)
    model = AutoModelForObjectDetection.from_pretrained(rid).to(dev).eval()
    names = {NORMALISE.get(v, v) for v in model.config.id2label.values()}

    def run(img):
        inp = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(out, threshold=0.05, target_sizes=[img.size[::-1]])[0]
        return [(NORMALISE.get(model.config.id2label[int(l)], model.config.id2label[int(l)]), float(s))
                for l, s in zip(res["labels"], res["scores"])]
    return run, names, lic


events = []
for split in ("train", "finetune"):
    for r in csv.DictReader(open(ROOT / "results" / "egoblind" / f"review_events_{split}.csv")):
        key = f"{int(float(r['video_id']))}_{int(float(r['obj_id']))}"
        fdir = ROOT / "data" / "review" / split / "frames" / key
        if not (fdir / "meta.json").exists():
            continue
        times = json.loads((fdir / "meta.json").read_text())["times"]
        a, b = (float(x) for x in r["window"].split("-"))
        idx = [i for i, t in enumerate(times) if a - 1e-6 <= t <= b + 1e-6] or list(range(len(times)))
        labels = set((r["labels"] or "").split(",")) - {"", "none"}
        events.append({"key": key, "split": split, "labels": labels, "frames": [fdir / f"{i:02d}.jpg" for i in idx]})

res = {}
loaded = {name: load(name) for name in [m for m in args.models.split(",") if m]}
common = set.intersection(*(v[1] for v in loaded.values()))
for name, (run, names, lic) in loaded.items():
    scored = [e for e in events if e["labels"] & common]          # identical event set for every detector
    seen, ndet, nfr, secs = 0, 0, 0, 0.0
    per_label = defaultdict(lambda: [0, 0])
    for e in scored:
        hit = False
        for f in e["frames"]:
            img = Image.open(f).convert("RGB")
            t0 = time.perf_counter(); out = run(img); secs += time.perf_counter() - t0
            nfr += 1
            keep = [(l, c) for l, c in out if c >= args.conf]
            ndet += len(keep)
            hit = hit or any(l in e["labels"] & common for l, _ in keep)
        seen += hit
        for l in e["labels"] & common:
            per_label[l][0] += hit; per_label[l][1] += 1
    res[name] = {"licence": lic, "events_scored": len(scored), "events_seen": seen, "event_recall": seen / max(1, len(scored)),
                 "dets_per_frame": ndet / max(1, nfr), "ms_per_frame": 1000 * secs / max(1, nfr), "frames": nfr,
                 "per_label": {k: v for k, v in sorted(per_label.items(), key=lambda kv: -kv[1][1])[:10]}}
    print(name, {k: v for k, v in res[name].items() if k != "per_label"}, flush=True)

# all detectors must be compared on the same events: intersection of COCO-label events (identical label sets here)
out = ROOT / "results" / "perception"
dump_json(res, out / "detector_comparison.json")
L = [f"Detector comparison on review frames inside train/finetune event windows (confidence >= {args.conf}); device {dev} for HF models\n",
     "| detector | licence | events scored | seen | event recall | detections / frame | ms / frame |", "|---|---|---|---|---|---|---|"]
for n, r in res.items():
    L.append(f"| {n} | {r['licence']} | {r['events_scored']} | {r['events_seen']} | {r['event_recall']:.3f} | {r['dets_per_frame']:.1f} | {r['ms_per_frame']:.0f} |")
L += ["", "Per label (seen / events): " + "; ".join(f"{n}: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in r["per_label"].items()) for n, r in res.items())]
(out / "detector_comparison.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
