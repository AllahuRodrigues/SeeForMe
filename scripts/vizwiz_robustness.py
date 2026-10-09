"""Robustness of the SeeForMe detector (YOLO11n) on photos taken by blind people, grouped by VizWiz image-quality flags.

VizWiz-Classification (Bafghi & Gurari, CVPR 2023, CC BY 4.0) is 8,900 photos taken by blind users, each labelled
with the ImageNet-1k categories present (200 categories) and, for the 7,147 photos drawn from the VizWiz train / val
pools, with boolean quality flags from VizWiz-QualityIssues: BLR blur, BRT too bright, DRK too dark, FRM improper
framing, OBS obstruction (finger, lens cover), ROT rotation, OTH other, NON no flaw. The 1,753 photos drawn from the
VizWiz test pool carry no flags (null) and are skipped. The full image zips are 18.8 GB (train 11.3, val 3.5, test
4.0), so this script fetches only the sampled photos one by one from the public per-image URL (no login).

    python scripts/vizwiz_robustness.py                  # sample <= 1,500 flagged photos (stratified by flag), download, run YOLO11n
    python scripts/vizwiz_robustness.py --n 300          # quick pass
    python scripts/vizwiz_robustness.py --no-download    # only the photos already under data/vizwiz_cls/images

Outputs: results/vizwiz/summary.{json,md} (detections per image, zero-detection rate, mean confidence and, where the
ImageNet label has a COCO equivalent, detector recall of the labelled object, per quality group) and
results/vizwiz/per_image.csv. Groups: flag:X (a photo counts in every flag it carries), only:X (photos with exactly
that one flag), issues:k (number of non-NON flags), all. Nothing here trains anything; VizWiz photos are single
frames of hand-held objects, so they say nothing about timing, only about what the detector sees when the camera is
held by someone who cannot check the frame.
"""
import argparse, csv, json, math, random, shutil, statistics as st, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _common import RESULTS, ROOT, dump_json, load_cfg

FLAGS = ["BLR", "BRT", "DRK", "FRM", "OBS", "ROT", "OTH", "NON"]
FLAG_NAMES = {"BLR": "blur", "BRT": "too bright", "DRK": "too dark", "FRM": "improper framing", "OBS": "obstruction",
              "ROT": "rotation", "OTH": "other", "NON": "no flaw"}
ISSUES = [f for f in FLAGS if f != "NON"]
IMG_URL = "https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/{name}"
ANN_URLS = {"annotations.json": "https://storage.googleapis.com/vw-c/annotations.json",
            "classification_gt.json": "https://vizwiz.cs.colorado.edu/VizWiz_all_answers/classification_gt.json"}
# ImageNet-1k index -> the COCO class the detector would have to emit for the labelled object. Strict entries name the
# same object; LOOSE entries usually contain it (a desktop computer shows a monitor, a stove is annotated as oven in
# COCO, a book jacket / comic book is a book). Categories without a COCO counterpart (packet, jersey, lotion, ...) are
# counted as detector targets only in the "any detection" statistics.
IMAGENET_TO_COCO = {620: "laptop", 664: "tv", 851: "tv", 673: "mouse", 761: "remote", 508: "keyboard", 487: "cell phone",
                    651: "microwave", 859: "toaster", 760: "refrigerator", 892: "clock", 409: "clock", 530: "clock",
                    883: "vase", 850: "teddy bear", 879: "umbrella", 414: "backpack", 748: "handbag", 898: "bottle",
                    907: "bottle", 737: "bottle", 440: "bottle", 720: "bottle", 968: "cup", 504: "cup", 572: "wine glass",
                    659: "bowl", 809: "bowl", 910: "spoon", 954: "banana", 950: "orange", 937: "broccoli", 963: "pizza",
                    934: "hot dog", 559: "chair", 765: "chair", 831: "couch", 861: "toilet", 896: "sink", 817: "car",
                    751: "car", 717: "truck", 867: "truck", 805: "sports ball", 430: "sports ball", 429: "sports ball",
                    296: "bear", 527: "tv", 827: "oven", 921: "book", 917: "book"}
LOOSE = {527, 827, 921, 917}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "vizwiz_cls"))
ap.add_argument("--out", default=str(RESULTS / "vizwiz"))
ap.add_argument("--n", type=int, default=1500, help="maximum number of flagged photos to use")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no-download", action="store_true", help="use only photos already on disk")
ap.add_argument("--min-free-gb", type=float, default=45.0, help="never let free disk drop under this")
ap.add_argument("--timeout", type=float, default=60.0)
args = ap.parse_args()

DATA = Path(args.data); IMAGES = DATA / "images"; OUT = Path(args.out)
DATA.mkdir(parents=True, exist_ok=True); IMAGES.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)


def fetch(url: str, target: Path, timeout: float) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "seeforme-vizwiz-robustness/1.0"})
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(target)
    return target.stat().st_size


# ---- annotations ------------------------------------------------------------------------------------------------
for name, url in ANN_URLS.items():
    if not (DATA / name).exists():
        print(f"fetching {name} from {url}")
        fetch(url, DATA / name, args.timeout)
gt = json.loads((DATA / "classification_gt.json").read_text())
ann = json.loads((DATA / "annotations.json").read_text())
wnid = {c["id"]: c["wordnet_id"] for c in ann["categories"]}
names = {}
if (DATA / "imagenet_class_index.json").exists():      # optional: human-readable ImageNet names (Keras class index)
    names = {int(k): v[1] for k, v in json.loads((DATA / "imagenet_class_index.json").read_text()).items()}

records = {}
for image, rec in gt.items():
    flaws = rec.get("flaws")
    if not flaws:
        continue                                       # VizWiz test-pool photos: quality flags are hidden
    flags = [f for f in FLAGS if flaws.get(f)]
    labels = list(rec.get("labels", []))
    records[image] = {"image": image, "pool": image.split("_")[1], "flags": flags,
                      "n_issues": sum(f in ISSUES for f in flags), "labels": labels,
                      "coco_targets": sorted({IMAGENET_TO_COCO[l] for l in labels if l in IMAGENET_TO_COCO}),
                      "coco_strict": sorted({IMAGENET_TO_COCO[l] for l in labels if l in IMAGENET_TO_COCO and l not in LOOSE})}
pop_counts = {f: sum(f in r["flags"] for r in records.values()) for f in FLAGS}
print(f"{len(gt)} photos, {len(records)} with quality flags; flag counts {pop_counts}")

# ---- stratified sample: every flag gets up to n/8 photos (rarest first), the rest is filled at random ---------------
rng = random.Random(args.seed)
on_disk = {p.name for p in IMAGES.glob("*.jpg") if p.stat().st_size > 0}
if args.no_download:
    selected = [r for r in records.values() if r["image"] in on_disk][: args.n]
else:
    selected, chosen = [], set()
    quota = max(1, args.n // len(FLAGS))
    for f in sorted(FLAGS, key=lambda f: pop_counts[f]):
        cands = [r for r in records.values() if f in r["flags"] and r["image"] not in chosen]
        rng.shuffle(cands)
        # photos already on disk first, so re-runs with a smaller --n reuse what was fetched
        cands.sort(key=lambda r: r["image"] not in on_disk)
        for r in cands[:quota]:
            if len(selected) >= args.n:
                break
            selected.append(r); chosen.add(r["image"])
    rest = [r for r in records.values() if r["image"] not in chosen]
    rng.shuffle(rest); rest.sort(key=lambda r: r["image"] not in on_disk)
    for r in rest[: max(0, args.n - len(selected))]:
        selected.append(r); chosen.add(r["image"])
print(f"sample: {len(selected)} photos; per flag {dict((f, sum(f in r['flags'] for r in selected)) for f in FLAGS)}")

# ---- download the sampled photos (skips what is on disk, refuses to breach the free-disk floor) -------------------
if not args.no_download:
    todo = [r["image"] for r in selected if r["image"] not in on_disk]
    free_gb = shutil.disk_usage(DATA).free / 1e9
    need_gb = len(todo) * 0.6 / 1000                     # VizWiz photos are ~0.3-0.6 MB each
    if free_gb - need_gb < args.min_free_gb:
        raise SystemExit(f"refusing to download {len(todo)} photos: free disk {free_gb:.1f} GB would drop under {args.min_free_gb} GB")
    t0 = time.time(); got = fail = 0; nbytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, IMG_URL.format(name=n), IMAGES / n, args.timeout): n for n in todo}
        for fut in as_completed(futs):
            try:
                nbytes += fut.result(); got += 1
            except Exception as e:                       # noqa: BLE001 - one bad photo must not stop the run
                fail += 1
                if fail <= 5:
                    print(f"  failed {futs[fut]}: {type(e).__name__}: {str(e)[:80]}")
    print(f"downloaded {got} photos ({nbytes / 1e6:.0f} MB) in {time.time() - t0:.0f}s, {fail} failures")
    selected = [r for r in selected if (IMAGES / r["image"]).exists() and (IMAGES / r["image"]).stat().st_size > 0]

# ---- detector --------------------------------------------------------------------------------------------------
import cv2
from seeforme.perception import YoloDetector

cfg = load_cfg()
det = YoloDetector(cfg.perception)
print(f"YOLO {cfg.perception.yolo_model} on {det.device}, imgsz {cfg.perception.yolo_imgsz}, conf {cfg.perception.yolo_conf}")
rows = []; t0 = time.time()
for i, r in enumerate(selected):
    img = cv2.imread(str(IMAGES / r["image"]))
    if img is None:
        print(f"  unreadable: {r['image']}"); continue
    dets = det.detect(img)
    confs = [d.confidence for d in dets]
    labs = [d.label for d in dets]
    strict_hit = any(l in r["coco_strict"] for l in labs); any_hit = any(l in r["coco_targets"] for l in labs)
    rows.append({**r, "n_dets": len(dets), "mean_conf": st.mean(confs) if confs else None, "max_conf": max(confs) if confs else None,
                 "top": ";".join(f"{d.label}:{d.confidence:.2f}" for d in sorted(dets, key=lambda d: -d.confidence)[:3]),
                 "labels_detected": sorted(set(labs)), "coco_hit": any_hit, "coco_hit_strict": strict_hit,
                 "latency_ms": det.last_latency_ms, "width": img.shape[1], "height": img.shape[0]})
    if (i + 1) % 250 == 0:
        print(f"  {i + 1}/{len(selected)} photos, {time.time() - t0:.0f}s")
print(f"ran the detector on {len(rows)} photos in {time.time() - t0:.0f}s")


# ---- grouping ----------------------------------------------------------------------------------------------------
def summarise(group):
    n = len(group)
    all_confs = [c for r in group for c in [r["mean_conf"]] if c is not None]
    labels = Counter(l for r in group for l in r["labels_detected"])
    targets = [r for r in group if r["coco_targets"]]; strict = [r for r in group if r["coco_strict"]]
    return {"n_images": n,
            "dets_per_image": round(st.mean(r["n_dets"] for r in group), 3) if n else None,
            "zero_det_rate": round(sum(r["n_dets"] == 0 for r in group) / n, 3) if n else None,
            "mean_conf": round(st.mean(all_confs), 3) if all_confs else None,
            "max_conf_mean": round(st.mean(r["max_conf"] for r in group if r["max_conf"] is not None), 3) if all_confs else None,
            "coco_targets": len(targets), "coco_recall": round(sum(r["coco_hit"] for r in targets) / len(targets), 3) if targets else None,
            "coco_targets_strict": len(strict), "coco_recall_strict": round(sum(r["coco_hit_strict"] for r in strict) / len(strict), 3) if strict else None,
            "top_labels": labels.most_common(5)}


groups = {"all": rows}
for f in FLAGS:
    groups[f"flag:{f}"] = [r for r in rows if f in r["flags"]]
    groups[f"only:{f}"] = [r for r in rows if r["flags"] == [f]]
for k in (0, 1, 2):
    groups[f"issues:{k}"] = [r for r in rows if r["n_issues"] == k]
groups["issues:3+"] = [r for r in rows if r["n_issues"] >= 3]
summary = {"dataset": "VizWiz-Classification (Bafghi & Gurari, CVPR 2023), quality flags from VizWiz-QualityIssues",
           "n_photos_total": len(gt), "n_photos_flagged": len(records), "flag_counts_population": pop_counts,
           "n_sampled": len(rows), "seed": args.seed,
           "detector": {"model": cfg.perception.yolo_model, "imgsz": cfg.perception.yolo_imgsz, "conf": cfg.perception.yolo_conf, "device": det.device,
                        "latency_ms_median": round(st.median(r["latency_ms"] for r in rows), 1) if rows else None},
           "groups": {k: summarise(v) for k, v in groups.items()},
           "imagenet_to_coco": {str(k): v for k, v in IMAGENET_TO_COCO.items()}, "loose_mappings": sorted(LOOSE),
           "sources": {"images": IMG_URL, **ANN_URLS}}
dump_json(summary, OUT / "summary.json")
with open(OUT / "per_image.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "pool", "flags", "n_issues", "imagenet_ids", "imagenet_names", "coco_targets", "n_dets", "mean_conf", "max_conf",
                "top_detections", "coco_hit", "coco_hit_strict", "latency_ms", "width", "height"])
    for r in rows:
        w.writerow([r["image"], r["pool"], "|".join(r["flags"]), r["n_issues"], "|".join(map(str, r["labels"])),
                    "|".join(names.get(l, "?") for l in r["labels"]), "|".join(r["coco_targets"]), r["n_dets"],
                    "" if r["mean_conf"] is None else f"{r['mean_conf']:.3f}", "" if r["max_conf"] is None else f"{r['max_conf']:.3f}",
                    r["top"], int(r["coco_hit"]), int(r["coco_hit_strict"]), f"{r['latency_ms']:.1f}", r["width"], r["height"]])


def fmt(x):
    return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


md = [f"# YOLO11n on VizWiz-Classification photos, by image-quality flag", "",
      f"{len(rows)} of the {len(records)} flagged photos (8,900 total; the 1,753 test-pool photos have no flags), sampled with seed "
      f"{args.seed} so that every flag has up to {args.n // len(FLAGS)} photos. Detector: {cfg.perception.yolo_model}, imgsz "
      f"{cfg.perception.yolo_imgsz}, conf {cfg.perception.yolo_conf}, {det.device}, median latency "
      f"{summary['detector']['latency_ms_median']} ms. `COCO recall` = share of photos whose ImageNet label has a COCO equivalent "
      f"({len(IMAGENET_TO_COCO)} of the 200 categories; strict = same object, loose adds desktop computer->tv, stove->oven, "
      f"book jacket / comic book->book) in which the detector emitted that class at any confidence above the threshold.", "",
      "| group | photos | detections / photo | zero-detection rate | mean confidence | COCO targets | COCO recall | strict targets | strict recall | top labels |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for k, g in summary["groups"].items():
    label = k
    for code, nm in FLAG_NAMES.items():
        label = label.replace(f":{code}", f":{code} ({nm})")
    md.append(f"| {label} | {g['n_images']} | {fmt(g['dets_per_image'])} | {fmt(g['zero_det_rate'])} | {fmt(g['mean_conf'])} | {g['coco_targets']} | "
              f"{fmt(g['coco_recall'])} | {g['coco_targets_strict']} | {fmt(g['coco_recall_strict'])} | "
              f"{', '.join(f'{l} {c}' for l, c in g['top_labels'])} |")
md += ["", "Reading: `flag:X` counts a photo in every flag it carries (most photos carry two: 3,111 of 7,147 in the population), "
       "`only:X` keeps the photos with exactly that flag, `issues:k` groups by the number of non-NON flags. VizWiz photos are "
       "hand-held close-ups of objects, not walking scenes, so the detections-per-photo column measures what the detector sees of "
       "a blind user's framing, not hazards; the recall columns are the robustness signal (same object, worse capture).",
       "", "Sources: " + ", ".join(f"{k} {v}" for k, v in summary["sources"].items()) + "; licence CC BY 4.0; cite Bafghi & Gurari, CVPR 2023."]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md[:5 + len(summary["groups"])]))
print(f"wrote {OUT / 'summary.json'}, {OUT / 'summary.md'}, {OUT / 'per_image.csv'}")
