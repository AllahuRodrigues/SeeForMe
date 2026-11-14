"""Choose the live demo's detector on this Mac: yolo11n.pt or yolo11s.pt (both in the repo root; nothing is downloaded).

    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_detector.py               # -> results/live/detector_choice.{json,md}
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_detector.py --skip-blvnav  # latency + agreement only (~1 min)
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_detector.py --no-depth     # no measured full-pipeline run

1. Latency. `YoloDetector.detect` (the call the live pipeline makes: letterbox to imgsz 640, inference, NMS, boxes to
   CPU) on MPS for 100 frames of EgoBlind *development* clips (finetune + select splits; holdout and test are never read),
   resized to the live inference width (640 px) first, after a warm-up, interleaved frame by frame between the models
   (alternating order) so thermal and clock drift hit both alike; median and p95 over all rounds. Then the whole
   `LivePipeline.process` per frame on the same frames: quality check, detector, Depth Anything V2 Metric Indoor Small
   (from the local Hugging Face cache, offline), object distances, IoU tracker with dynamics and the live scheduler
   (`pcacs_learned:rl` unless --policy says otherwise), plus the measured cost of resizing a camera-size frame to 640 px.
2. Accuracy. Presence precision / recall / F1 on the BLV-Road-Nav-Accessibility labelled keyframes for the COCO classes
   that map *directly* onto a taxonomy object, reusing scripts/blvnav_taxonomy.py (load_objects, classify, load_labels,
   score), at conf 0.25 and 0.35, with a paired bootstrap over segments for the difference between the two models.
3. Agreement. On the EgoBlind frames (the 100 timing frames, plus 3 frames of every development clip for broader class
   coverage): how many person / car / chair / bicycle boxes each model finds, how many match
   between the models (same class, IoU >= 0.5), and on the timing frames how often a box is still there in the next sampled frame (0.1 s
   later, same class, IoU >= 0.3 = the tracker's association threshold): a proxy for track continuity, which matters
   because a person who drops out and comes back as a new track is announced again.

Recommendation rule (from the task): the live default is the more accurate model (higher direct-class micro F1 at every
threshold) if its median per-frame pipeline time stays under --budget-ms (120); otherwise yolo11n.pt.
"""
import os

# no downloads: Ultralytics skips its online checks, Hugging Face only reads the local cache
os.environ["YOLO_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse, dataclasses, json, platform, random, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import MODELS, RESULTS, ROOT, dump_json, load_cfg

DEV_SPLITS = ("finetune", "select")          # EgoBlind development splits: the only ones this script may read
CLASSES = ("person", "car", "chair", "bicycle")
OUT = RESULTS / "live"


# ---------------------------------------------------------------------------------------------------------------------
# small pure helpers (no model, no files)
# ---------------------------------------------------------------------------------------------------------------------

def pct(xs, q):
    return float(np.percentile(np.asarray(xs, float), q)) if len(xs) else None


def stats(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median_ms": round(pct(xs, 50), 2), "p95_ms": round(pct(xs, 95), 2),
            "mean_ms": round(float(np.mean(xs)), 2), "min_ms": round(min(xs), 2), "max_ms": round(max(xs), 2)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_matches(A, B, thr):
    """Number of one-to-one matches between box lists A and B with IoU >= thr (highest IoU first)."""
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(A) for j, b in enumerate(B)), reverse=True)
    used_a, used_b, n = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); n += 1
    return n


def boxes_of(frame_dets, label, thr):
    return [d[2] for d in frame_dets if d[0] == label and d[1] >= thr]


def agreement(boxes, a, b, thr, classes=CLASSES, iou_thr=0.5):
    """Per class: boxes each model finds, boxes matched between them, frames where each sees the class."""
    out = {}
    for lab in classes:
        c = Counter()
        for fa, fb in zip(boxes[a], boxes[b]):
            A, B = boxes_of(fa, lab, thr), boxes_of(fb, lab, thr)
            m = greedy_matches(A, B, iou_thr)
            c["boxes_" + a] += len(A); c["boxes_" + b] += len(B); c["matched"] += m
            c["only_" + a] += len(A) - m; c["only_" + b] += len(B) - m
            c["frames_" + a] += bool(A); c["frames_" + b] += bool(B)
            c["frames_both"] += bool(A) and bool(B); c["frames_neither"] += not A and not B
        n_frames = len(boxes[a])
        out[lab] = {**dict(c), "frames": n_frames,
                    "frame_presence_agreement": round((c["frames_both"] + c["frames_neither"]) / n_frames, 3) if n_frames else None,
                    "box_match_rate": round(c["matched"] / max(1, max(c["boxes_" + a], c["boxes_" + b])), 3)}
    return out


def persistence(boxes, clip_of, name, thr, classes=CLASSES, iou_thr=0.3):
    """Per class: share of boxes that still have a same-class box with IoU >= iou_thr in the next sampled frame of the
    same clip, and how often the class's presence flips between consecutive frames."""
    out = {}
    seq = boxes[name]
    for lab in classes:
        kept = total = flips = transitions = 0
        for k in range(len(seq) - 1):
            if clip_of[k] != clip_of[k + 1]:
                continue
            A, B = boxes_of(seq[k], lab, thr), boxes_of(seq[k + 1], lab, thr)
            transitions += 1
            flips += bool(A) != bool(B)
            total += len(A)
            kept += sum(any(iou(x, y) >= iou_thr for y in B) for x in A)
        out[lab] = {"boxes": total, "persisted": kept, "persistence": round(kept / total, 3) if total else None,
                    "transitions": transitions, "presence_flips": flips,
                    "flip_rate": round(flips / transitions, 3) if transitions else None}
    return out


def _prf(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        r = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f = np.where(p + r > 0, 2 * p * r / (p + r), np.nan)
    return {"precision": p, "recall": r, "f1": f}


def paired_bootstrap(ca, cb, n_boot=2000, seed=0):
    """ca, cb: {segment: Counter(tp, fp, fn, tn)} for models a and b on the same frames. Resamples whole segments (frames
    within a segment are strongly correlated) and returns b - a for micro precision, recall and F1 with a 95% CI."""
    segs = sorted(set(ca) | set(cb))
    A = np.array([[ca.get(s, Counter())[k] for k in ("tp", "fp", "fn")] for s in segs], float)
    B = np.array([[cb.get(s, Counter())[k] for k in ("tp", "fp", "fn")] for s in segs], float)
    idx = np.random.default_rng(seed).integers(0, len(segs), size=(n_boot, len(segs)))
    SA, SB = A[idx].sum(1), B[idx].sum(1)
    pa, pb = _prf(*A.sum(0)), _prf(*B.sum(0))
    ba, bb = _prf(SA[:, 0], SA[:, 1], SA[:, 2]), _prf(SB[:, 0], SB[:, 1], SB[:, 2])
    out = {"segments": len(segs), "n_boot": n_boot}
    for k in ("precision", "recall", "f1"):
        d = bb[k] - ba[k]
        lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
        out[k] = {"diff": round(float(pb[k] - pa[k]), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "p_b_not_better": round(float(np.nanmean(d <= 0)), 4)}
    return out


def segment_counts(segs, maxconf, direct_objs, thr):
    """{segment: Counter(tp, fp, fn, tn)} over the direct objects: same rule as blvnav_taxonomy.score (a frame is a
    detection of an object when any of its COCO classes has a box at or above thr)."""
    out = {}
    for seg, (idx, table) in segs.items():
        c = Counter()
        for o in direct_objs:
            vals = table.get(o["object"])
            if vals is None:
                continue
            for i, v in zip(idx, vals):
                if v is None or (seg, i) not in maxconf:
                    continue
                pred = max(maxconf[(seg, i)].get(l, 0.0) for l in o["coco_labels"]) >= thr
                c["tp" if (v and pred) else "fp" if pred else "fn" if v else "tn"] += 1
        out[seg] = c
    return out


def recommend(models, f1_by_thr, pipeline_ms, budget_ms, fallback):
    """models: ordered names; f1_by_thr: {thr: {model: direct micro F1}}; pipeline_ms: {model: median ms per frame of the
    full pipeline}. The more accurate model is the one with the highest F1 at every threshold (none if the thresholds
    disagree or tie); it becomes the default when its pipeline time fits the budget, otherwise the fallback does."""
    more = None
    for m in models:
        if all(f1_by_thr[t][m] is not None and all(f1_by_thr[t][m] > (f1_by_thr[t][x] or 0.0) for x in models if x != m)
               for t in f1_by_thr):
            more = m
    if more is None:
        return {"model": fallback, "more_accurate": None, "reason": "no model has the higher direct-class F1 at every threshold"}
    ms = pipeline_ms.get(more)
    if more == fallback:
        return {"model": fallback, "more_accurate": more, "reason": f"{more} is both the more accurate and the cheaper model"}
    if ms is not None and ms <= budget_ms:
        return {"model": more, "more_accurate": more,
                "reason": f"{more} has the higher direct-class F1 at every threshold and its full pipeline runs at {ms:.0f} ms per frame (median) <= {budget_ms:.0f} ms"}
    return {"model": fallback, "more_accurate": more,
            "reason": f"{more} is more accurate but its full pipeline needs {ms if ms is None else round(ms)} ms per frame (median) > {budget_ms:.0f} ms"}


# ---------------------------------------------------------------------------------------------------------------------
# frames, models, measurements
# ---------------------------------------------------------------------------------------------------------------------

def sample_dev_frames(splits, n_clips, per_clip, step_s, seed, width):
    """`per_clip` consecutive frames, `step_s` apart, from 40% into each of `n_clips` EgoBlind development clips (random
    order, fixed seed). Returns frames resized to the live inference width, their clip index, the clip metadata and the
    time to resize each original frame (what the live pipeline pays for a camera frame)."""
    import cv2
    from egoblind_common import SCENARIOS, video_path
    from seeforme.perception import resize_for_inference
    bad = [s for s in splits if s not in DEV_SPLITS]
    if bad:
        raise SystemExit(f"refusing to read EgoBlind split(s) {bad}: only the development splits {DEV_SPLITS} are allowed here")
    pool = [(s, int(p.stem)) for s in splits for p in sorted((SCENARIOS / s).glob("*.json"), key=lambda p: int(p.stem))]
    pool = [(s, v) for s, v in pool if video_path(v) is not None]
    if not pool:
        raise SystemExit(f"no local EgoBlind videos for {splits}: run scripts/download_egoblind.py first")
    random.Random(seed).shuffle(pool)
    frames, clip_of, clips, resize_ms = [], [], [], []
    for s, v in pool:
        if len(clips) >= n_clips:
            break
        cap = cv2.VideoCapture(str(video_path(v)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(step_s * fps))
        need = (per_clip - 1) * step + 1
        if n < need:
            cap.release(); continue
        start = int(max(0, min(n - need, 0.4 * n)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        got = []
        for k in range(need):
            ok, f = cap.read()
            if not ok:
                break
            if k % step == 0:
                t0 = time.perf_counter()
                small = resize_for_inference(f, width)
                resize_ms.append((time.perf_counter() - t0) * 1000)
                got.append(small)
        cap.release()
        if len(got) < per_clip:
            continue
        clips.append({"split": s, "video_id": v, "fps": round(fps, 3), "start_frame": start, "step_frames": step,
                      "source_wh": [int(f.shape[1]), int(f.shape[0])], "frames": len(got)})
        frames += got
        clip_of += [len(clips) - 1] * len(got)
    return frames, clip_of, clips, resize_ms


def sample_wide_frames(splits, positions, width):
    """One frame at each relative position (e.g. 0.25 / 0.5 / 0.75 of the duration) of every development clip: a broad
    sample for the class counts and the model-to-model agreement (the 100 timing frames come from only 10 clips)."""
    import cv2
    from egoblind_common import SCENARIOS, video_path
    from seeforme.perception import resize_for_inference
    bad = [s for s in splits if s not in DEV_SPLITS]
    if bad:
        raise SystemExit(f"refusing to read EgoBlind split(s) {bad}: only the development splits {DEV_SPLITS} are allowed here")
    frames, ids = [], []
    for s in splits:
        for p in sorted((SCENARIOS / s).glob("*.json"), key=lambda p: int(p.stem)):
            v = int(p.stem)
            path = video_path(v)
            if path is None:
                continue
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for q in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * max(0, n - 1)))
                ok, f = cap.read()
                if ok:
                    frames.append(resize_for_inference(f, width)); ids.append((s, v))
            cap.release()
    return frames, ids


def detect_all(dets, frames):
    return {n: [[(d.label, float(d.confidence), tuple(float(x) for x in d.bbox)) for d in det.detect(f)] for f in frames]
            for n, det in dets.items()}


def load_detectors(model_names, cfg, device, conf):
    from seeforme.perception import YoloDetector
    dets = {}
    for m in model_names:
        path = ROOT / m
        if not path.exists():   # Ultralytics would download a missing *.pt: refuse instead
            raise SystemExit(f"{path} not found; this script never downloads weights")
        pc = dataclasses.replace(cfg.perception, yolo_model=str(path), device=device, yolo_conf=conf, detector="yolo")
        dets[m] = YoloDetector(pc)
        if dets[m].device != device:
            raise SystemExit(f"{m} loaded on {dets[m].device}, expected {device}")
    return dets


def time_detectors(dets, frames, rounds, warmup):
    """Per-frame wall time of detector.detect (including the boxes' transfer to CPU), interleaved between models with
    alternating order; boxes from the first round are kept for the agreement analysis."""
    names = list(dets)
    # warm-up: the first frames plus one frame of every distinct input shape (portrait and landscape clips letterbox to
    # different tensor shapes, and MPS compiles a graph per shape on first use)
    shapes, extra = set(), []
    for f in frames:
        if f.shape not in shapes:
            shapes.add(f.shape); extra.append(f)
    for n in names:
        for f in list(frames[:warmup]) + extra * 2:
            dets[n].detect(f)
    times = {n: [] for n in names}
    per_round = {n: [] for n in names}
    boxes = {n: [None] * len(frames) for n in names}
    for r in range(rounds):
        rt = {n: [] for n in names}
        for i, f in enumerate(frames):
            for n in (names if (i + r) % 2 == 0 else names[::-1]):
                t0 = time.perf_counter()
                out = dets[n].detect(f)
                ms = (time.perf_counter() - t0) * 1000
                times[n].append(ms); rt[n].append(ms)
                if r == 0:
                    boxes[n][i] = [(d.label, float(d.confidence), tuple(float(x) for x in d.bbox)) for d in out]
        for n in names:
            per_round[n].append(round(pct(rt[n], 50), 2))
    return times, per_round, boxes


class _Silent:
    """Speaker stand-in: the pipeline's channel bookkeeping runs, nothing is spoken."""
    def say(self, msg): pass
    def stop(self): pass
    def interrupt(self, msg): pass
    def haptic(self): pass
    def busy(self): return False


def time_pipeline(dets, frames, clip_of, clips, cfg, policy_name, depth, rounds, warmup):
    """Full LivePipeline.process per frame for each detector (one pipeline per detector, interleaved per frame)."""
    from seeforme.perception import NullOcr
    from seeforme.pipeline import LivePipeline, make_live_policy
    names = list(dets)

    def new_pipe(n):
        pol = make_live_policy(policy_name, cfg, str(MODELS), fallback=True)
        return LivePipeline(dets[n], NullOcr(), pol, cfg, speaker=_Silent(), depth=depth), pol

    shapes, extra = set(), []
    for f in frames:
        if f.shape not in shapes:
            shapes.add(f.shape); extra.append(f)
    for n in names:   # warm-up (depth interpolation to every frame size, scheduler first call)
        p, _ = new_pipe(n)
        for k, f in enumerate(list(frames[:warmup]) + extra * 2):
            p.process(f, k * 0.1)
    rec = {n: defaultdict(list) for n in names}
    cues = {n: Counter() for n in names}
    policy_used = None
    for r in range(rounds):
        pipes = {}
        for n in names:
            pipes[n], pol = new_pipe(n)
            policy_used = {"name": getattr(pol, "name", policy_name), "fallback_from": getattr(pol, "fallback_from", None),
                           "checkpoint": getattr(pol, "checkpoint", None)}
        for i, f in enumerate(frames):
            c = clips[clip_of[i]]
            j = i - clip_of.index(clip_of[i])
            t = clip_of[i] * 30.0 + j * c["step_frames"] / c["fps"]
            for n in (names if (i + r) % 2 == 0 else names[::-1]):
                p = pipes[n]
                res = p.process(f, t)
                L = p.latency
                for k in ("detect", "track", "policy", "end_to_end"):
                    rec[n][k].append(getattr(L, k)[-1])
                rec[n]["depth"].append(L.__dict__["depth"][-1])
                if r == 0 and res.cue is not None:
                    cues[n][res.cue.action] += 1
    out = {n: {k: stats(v) for k, v in rec[n].items()} for n in names}
    for n in names:
        out[n]["cues_round0"] = dict(cues[n])
    return out, policy_used


def blvnav_eval(dets, thresholds, width, n_boot):
    """Run every model on every labelled BLV-Road-Nav keyframe (resized to the live width like a camera frame) and score
    presence with blvnav_taxonomy.score; detections go to a temporary directory that is deleted afterwards."""
    import cv2
    import blvnav_taxonomy as bt
    from seeforme.perception import resize_for_inference
    if not bt.REPO.exists():
        raise SystemExit(f"{bt.REPO} missing (git clone https://github.com/Shohan29531/BLV-Road-Nav-Accessibility)")
    names = bt.load_objects()
    objs = [bt.classify(n) for n in names]
    direct = [o for o in objs if o["coco_match"] == "direct"]
    segs, odd = bt.load_labels(names)
    todo = [(seg, i, bt.REPO / "Video Frames" / f"{seg}-frame-{i}.jpeg") for seg, (idx, _) in segs.items() for i in idx]
    models = list(dets)
    maxconf = {m: {} for m in models}
    lat = {m: [] for m in models}
    out = {"frames_labelled": len(todo), "segments": len(segs), "direct_objects": [o["object"] for o in direct],
           "direct_coco_classes": sorted({l for o in direct for l in o["coco_labels"]}), "label_oddities": dict(odd)}
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="seeforme_det_") as td:
        paths = {m: Path(td) / f"{Path(m).stem}.jsonl" for m in models}
        fhs = {m: open(paths[m], "w") for m in models}
        try:
            for k, (seg, i, p) in enumerate(todo):
                if not p.exists():
                    continue
                img = cv2.imread(str(p))
                small = resize_for_inference(img, width)
                for m in (models if k % 2 == 0 else models[::-1]):
                    t0 = time.perf_counter()
                    ds = dets[m].detect(small)
                    lat[m].append((time.perf_counter() - t0) * 1000)
                    mc = defaultdict(float)
                    for d in ds:
                        mc[d.label] = max(mc[d.label], round(float(d.confidence), 4))   # same rounding as the JSONL score() reads
                    maxconf[m][(seg, i)] = mc
                    fhs[m].write(json.dumps({"segment": seg, "frame": i, "file": p.name, "w": img.shape[1], "h": img.shape[0],
                                             "dets": [[d.label, round(d.confidence, 4)] + [round(x, 4) for x in d.bbox] for d in ds]}) + "\n")
                if (k + 1) % 200 == 0:
                    print(f"  blvnav {k + 1}/{len(todo)} keyframes", flush=True)
        finally:
            for fh in fhs.values():
                fh.close()
        ev = {m: bt.score(segs, paths[m], objs, thresholds) for m in models}
    out["seconds"] = round(time.perf_counter() - t_start, 1)
    out["frames_scored"] = ev[models[0]]["frames_scored"]
    out["latency"] = {m: stats(lat[m]) for m in models}
    out["thresholds"] = {}
    for thr in thresholds:
        key = str(thr)
        blk = {}
        for m in models:
            b = ev[m]["thresholds"][key]
            per = {n: {k: r.get(k) for k in ("support", "tp", "fp", "fn", "precision", "recall", "f1")}
                   for n, r in b["per_object"].items() if r.get("annotated") and r.get("match") == "direct"}
            sup10 = [r for r in per.values() if (r["support"] or 0) >= 10]
            seg_c = segment_counts(segs, maxconf[m], direct, thr)
            tot = sum(seg_c.values(), Counter())
            # the per-segment counts must reproduce blvnav_taxonomy.score's direct micro counts exactly
            assert all(tot[k] == b["micro"]["direct"][k] for k in ("tp", "fp", "fn", "tn")), (m, thr, tot, b["micro"]["direct"])
            blk[m] = {"direct_micro": b["micro"]["direct"], "mappable_micro": b["micro"]["mappable"],
                      "paper_style_direct": b["micro"]["all_annotated_direct"],
                      "high_priority_frames": b["high_priority_frames"],
                      "direct_macro_f1_support_ge_10": round(float(np.mean([r["f1"] or 0.0 for r in sup10])), 3) if sup10 else None,
                      "direct_macro_objects": sorted(n for n, r in per.items() if (r["support"] or 0) >= 10),
                      "per_object_direct": per, "_seg": seg_c}
        a, b2 = models[0], models[-1]
        blk["bootstrap_" + f"{Path(b2).stem}_minus_{Path(a).stem}"] = paired_bootstrap(blk[a]["_seg"], blk[b2]["_seg"], n_boot=n_boot)
        for m in models:
            blk[m].pop("_seg")
        out["thresholds"][key] = blk
    return out


def depth_estimator(outdoor=False):
    from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
    name = OUTDOOR if outdoor else INDOOR
    try:
        return DepthAnythingEstimator(name), name, None
    except Exception as e:   # not in the local cache (offline mode): fall back to the conservative estimate
        return None, name, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------------

def write_md(rep, path: Path):
    f2 = lambda x: "" if x is None else f"{x:.2f}"
    f3 = lambda x: "" if x is None else f"{x:.3f}"
    models = rep["models"]
    rec = rep["recommendation"]
    L = ["# Live detector choice: " + " vs ".join(f"`{m}`" for m in models), "",
         f"Host: {rep['host']['machine']}, torch {rep['host']['torch']}, ultralytics {rep['host']['ultralytics']}, device "
         f"`{rep['host']['device']}`, imgsz {rep['protocol']['imgsz']}, generated {rep['generated']} by `scripts/eval_live_detector.py`.", "",
         "## Recommendation", "",
         f"**Live default: `{rec['model']}`** (detector confidence floor `perception.yolo_conf = {rep['protocol']['live_conf']}`, unchanged). {rec['reason']}.", ""]
    L += [f"Rule: the live default is the more accurate model (higher direct-class micro F1 on the BLV-Road-Nav keyframes at every "
          f"threshold scored) if its full per-frame pipeline time (median) stays under {rep['protocol']['budget_ms']:.0f} ms, otherwise "
          f"`{rep['protocol']['fallback']}`.", ""]
    for line in rec.get("notes", []):
        L.append(f"* {line}")
    L.append("")

    lat = rep["latency"]
    L += ["## 1. Latency on MPS", "",
          f"Detector only (`YoloDetector.detect`, wall time including NMS and the copy of the boxes to the CPU) on "
          f"{lat['frames']} EgoBlind development frames ({lat['clips']} clips of the finetune + select splits, resized to "
          f"{rep['protocol']['width']} px wide like a camera frame), {lat['warmup']} warm-up frames per model, {lat['rounds']} rounds "
          "interleaved frame by frame with alternating order.", "",
          "| model | frames x rounds | median ms | p95 ms | mean ms | per-round medians | BLV keyframes median / p95 ms |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        s = lat["detector"][m]
        b = (rep.get("blvnav") or {}).get("latency", {}).get(m, {})
        L.append(f"| `{m}` | {s['n']} | {f2(s['median_ms'])} | {f2(s['p95_ms'])} | {f2(s['mean_ms'])} | "
                 f"{', '.join(f2(x) for x in lat['per_round_median'][m])} | {f2(b.get('median_ms'))} / {f2(b.get('p95_ms'))} |")
    L += ["", f"Resizing a camera-size frame to {rep['protocol']['width']} px: median {f2(lat['resize_ms']['median_ms'])} ms "
          f"(p95 {f2(lat['resize_ms']['p95_ms'])})."]
    pipe = rep.get("pipeline")
    if pipe and pipe.get("measured"):
        L += ["", f"Full `LivePipeline.process` per frame on the same frames: quality check, detector, depth "
              f"(`{pipe['depth_model']}`, input {pipe['depth_input']} px), object distances, IoU tracker with dynamics, scheduler "
              f"`{pipe['policy']['name']}`{' (fallback from ' + str(pipe['policy']['fallback_from']) + ')' if pipe['policy'].get('fallback_from') else ''}, "
              f"no OCR, no speech; {pipe['rounds']} rounds interleaved. The budget column adds the resize.", "",
              "| model | detect med / p95 | depth med / p95 | track med | scheduler med / p95 | end-to-end med / p95 | + resize (median) | fits " + f"{rep['protocol']['budget_ms']:.0f} ms |",
              "|---|---|---|---|---|---|---|---|"]
        for m in models:
            s = pipe["per_model"][m]
            L.append(f"| `{m}` | {f2(s['detect']['median_ms'])} / {f2(s['detect']['p95_ms'])} | {f2(s['depth']['median_ms'])} / {f2(s['depth']['p95_ms'])} | "
                     f"{f2(s['track']['median_ms'])} | {f2(s['policy']['median_ms'])} / {f2(s['policy']['p95_ms'])} | "
                     f"{f2(s['end_to_end']['median_ms'])} / {f2(s['end_to_end']['p95_ms'])} | {f2(pipe['budget_ms'][m])} | "
                     f"{'yes' if pipe['budget_ms'][m] <= rep['protocol']['budget_ms'] else 'no'} |")
    L += ["", "Conservative bound without the measured depth (detector median + Depth Anything at the top of its 30-60 ms range + "
          "tracker and scheduler + resize):", ""]
    for m in models:
        L.append(f"* `{m}`: {f2(rep['conservative_ms'][m])} ms")
    L.append("")

    bl = rep.get("blvnav")
    if bl:
        L += ["## 2. Accuracy: BLV-Road-Nav-Accessibility keyframes", "",
              f"{bl['frames_scored']} labelled keyframes from {bl['segments']} segments of blind pedestrians' street videos; presence "
              "labels (0 / 1 per object and keyframe, no boxes). A frame counts as a detection of an object when one of its COCO classes "
              f"has a box at or above the threshold. Direct objects ({len(bl['direct_objects'])}): {', '.join(bl['direct_objects'])}. "
              "Scoring is `blvnav_taxonomy.score` (same code as `results/blvnav/detector_eval.*`); the frames are resized to 640 px wide "
              "first, like a camera frame. The difference column is a paired bootstrap over segments (95% CI).", ""]
        for thr, blk in bl["thresholds"].items():
            bkey = next(k for k in blk if k.startswith("bootstrap_"))
            bs = blk[bkey]
            L += [f"### confidence >= {thr}", "",
                  "| model | TP | FP | FN | precision | recall | F1 | macro F1 (objects with >= 10 frames) | paper-style recall (N = 12) |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for m in models:
                d, ps = blk[m]["direct_micro"], blk[m]["paper_style_direct"]
                L.append(f"| `{m}` | {d['tp']} | {d['fp']} | {d['fn']} | {f3(d['precision'])} | {f3(d['recall'])} | {f3(d['f1'])} | "
                         f"{f3(blk[m]['direct_macro_f1_support_ge_10'])} | {f3(ps['recall'])} |")
            L.append(f"| {bkey.replace('bootstrap_', '').replace('_minus_', ' - ')} | | | | "
                     + " | ".join(f"{bs[k]['diff']:+.3f} [{bs[k]['ci95'][0]:+.3f}, {bs[k]['ci95'][1]:+.3f}]" for k in ("precision", "recall", "f1"))
                     + " | | |")
            L += ["", "| object | frames present | " + " | ".join(f"`{Path(m).stem}` P / R / F1" for m in models) + " |",
                  "|---|---|" + "---|" * len(models)]
            per0 = blk[models[0]]["per_object_direct"]
            for obj in sorted(per0, key=lambda o: -(per0[o]["support"] or 0)):
                cells = []
                for m in models:
                    r = blk[m]["per_object_direct"][obj]
                    cells.append(f"{f2(r['precision'])} / {f2(r['recall'])} / {f2(r['f1'])}")
                L.append(f"| {obj} | {per0[obj]['support']} | " + " | ".join(cells) + " |")
            L.append("")

    ag = rep["agreement"]
    a, b = models[0], models[-1]
    L += ["## 3. Agreement on the EgoBlind development frames", "",
          f"First the {lat['frames']} timing frames (consecutive frames of {lat['clips']} clips), then a broad sample of every "
          "development clip. Matched = same class, IoU >= 0.5, one-to-one. Persistence = share of a "
          "model's boxes that still have a same-class box with IoU >= 0.3 (the tracker's association threshold) in the next sampled "
          f"frame of the clip, {rep['protocol']['step_s']} s later; flip rate = share of consecutive frame pairs where the class appears "
          "or disappears. Head-mounted walking footage, so some misses are real motion.", ""]
    for thr, blk in ag.items():
        L += [f"### The {lat['frames']} timing frames, confidence >= {thr}", "",
              f"| class | boxes `{Path(a).stem}` | boxes `{Path(b).stem}` | matched | only `{Path(a).stem}` | only `{Path(b).stem}` | "
              f"frames `{Path(a).stem}` | frames `{Path(b).stem}` | frame agreement | persistence `{Path(a).stem}` / `{Path(b).stem}` | "
              f"flip rate `{Path(a).stem}` / `{Path(b).stem}` |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for lab in CLASSES:
            r = blk["pairwise"][lab]
            pa, pb = rep["persistence"][thr][a][lab], rep["persistence"][thr][b][lab]
            L.append(f"| {lab} | {r.get('boxes_' + a, 0)} | {r.get('boxes_' + b, 0)} | {r.get('matched', 0)} | {r.get('only_' + a, 0)} | "
                     f"{r.get('only_' + b, 0)} | {r.get('frames_' + a, 0)} | {r.get('frames_' + b, 0)} | {f3(r['frame_presence_agreement'])} | "
                     f"{f3(pa['persistence'])} / {f3(pb['persistence'])} | {f3(pa['flip_rate'])} / {f3(pb['flip_rate'])} |")
        L.append("")
    wide = rep.get("agreement_wide")
    if wide:
        L += [f"### Broad sample: {wide['frames']} frames ({', '.join(str(x) for x in wide['positions'])} of the duration) of all "
              f"{wide['clips']} development clips", "",
              f"| conf | class | boxes `{Path(a).stem}` | boxes `{Path(b).stem}` | matched (IoU >= 0.5) | only `{Path(a).stem}` | "
              f"only `{Path(b).stem}` | frames `{Path(a).stem}` | frames `{Path(b).stem}` | frame agreement |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for thr, blk in wide["per_threshold"].items():
            for lab in CLASSES:
                r = blk["pairwise"][lab]
                L.append(f"| {thr} | {lab} | {r.get('boxes_' + a, 0)} | {r.get('boxes_' + b, 0)} | {r.get('matched', 0)} | "
                         f"{r.get('only_' + a, 0)} | {r.get('only_' + b, 0)} | {r.get('frames_' + a, 0)} | {r.get('frames_' + b, 0)} | "
                         f"{f3(r['frame_presence_agreement'])} |")
        L.append("")
    L += ["## Frames used", "",
          "EgoBlind development clips only (" + ", ".join(rep["protocol"]["splits"]) + "); the holdout and official test splits are "
          "never opened. " + ", ".join(f"{c['split']}:{c['video_id']}" for c in rep["frames"]["clips"]) + ".", "",
          "## Caveats", ""]
    for c in rep["caveats"]:
        L.append(f"* {c}")
    L += ["", "Reproduce: `OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_detector.py`.", ""]
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="yolo11n.pt,yolo11s.pt", help="weights in the repo root; the first is the fallback")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--splits", default=",".join(DEV_SPLITS), help="EgoBlind development splits to sample frames from")
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--per-clip", type=int, default=10)
    ap.add_argument("--step-s", type=float, default=0.1, help="spacing of the sampled frames inside a clip")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--thresholds", default="0.25,0.35")
    ap.add_argument("--policy", default="pcacs_learned:rl", help="scheduler for the full-pipeline timing (the make live default)")
    ap.add_argument("--budget-ms", type=float, default=120.0)
    ap.add_argument("--depth-hi-ms", type=float, default=60.0, help="Depth Anything Small cost for the conservative bound")
    ap.add_argument("--no-depth", action="store_true", help="skip the measured full-pipeline run")
    ap.add_argument("--skip-blvnav", action="store_true")
    ap.add_argument("--wide-positions", default="0.25,0.5,0.75",
                    help="relative positions of the frames taken from every development clip for the agreement counts ('' = skip)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "detector_choice"), help="output stem (.json and .md are written)")
    args = ap.parse_args()

    import torch
    import ultralytics
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cfg = load_cfg()
    width, imgsz, live_conf = cfg.perception.frame_width, cfg.perception.yolo_imgsz, cfg.perception.yolo_conf
    if imgsz != 640:
        print(f"note: config imgsz {imgsz}; this evaluation uses 640", flush=True)
        cfg.perception.yolo_imgsz = imgsz = 640
    t_all = time.perf_counter()

    print(f"sampling {args.clips} x {args.per_clip} frames from EgoBlind {splits} ...", flush=True)
    frames, clip_of, clips, resize_ms = sample_dev_frames(splits, args.clips, args.per_clip, args.step_s, args.seed, width)
    print(f"  {len(frames)} frames from {len(clips)} clips", flush=True)

    dets = load_detectors(models, cfg, args.device, min(min(thresholds), live_conf))
    print("timing detectors ...", flush=True)
    times, per_round, boxes = time_detectors(dets, frames, args.rounds, args.warmup)
    for m in models:
        s = stats(times[m])
        print(f"  {m}: median {s['median_ms']} ms, p95 {s['p95_ms']} ms over {s['n']} calls; per-round medians {per_round[m]}", flush=True)

    pipeline = {"measured": False}
    if not args.no_depth:
        depth, depth_name, err = depth_estimator()
        if depth is None:
            print(f"  depth model unavailable offline ({err}); using the conservative bound only", flush=True)
            pipeline["error"] = err
        else:
            print("timing the full live pipeline ...", flush=True)
            per_model, pol = time_pipeline(dets, frames, clip_of, clips, cfg, args.policy, depth, args.rounds, args.warmup)
            rs = pct(resize_ms, 50)
            pipeline = {"measured": True, "depth_model": depth_name, "depth_input": depth.input_size, "policy": pol,
                        "rounds": args.rounds, "per_model": per_model,
                        "budget_ms": {m: round(per_model[m]["end_to_end"]["median_ms"] + rs, 2) for m in models}}
            for m in models:
                print(f"  {m}: end-to-end median {per_model[m]['end_to_end']['median_ms']} ms (p95 {per_model[m]['end_to_end']['p95_ms']}), "
                      f"depth {per_model[m]['depth']['median_ms']} ms, scheduler {per_model[m]['policy']['median_ms']} ms", flush=True)
            del depth
    tp_ms = 1.0   # tracker + scheduler when not measured (results/latency_pcacs.json: ~0.13 ms + ~0.01 ms)
    if pipeline["measured"]:
        tp_ms = max(pipeline["per_model"][m]["track"]["median_ms"] + pipeline["per_model"][m]["policy"]["median_ms"] for m in models)
    conservative = {m: round(pct(times[m], 50) + args.depth_hi_ms + tp_ms + pct(resize_ms, 50), 2) for m in models}

    agree, persist = {}, {}
    for thr in thresholds:
        agree[str(thr)] = {"counts": {m: {lab: sum(len(boxes_of(f, lab, thr)) for f in boxes[m]) for lab in CLASSES} for m in models},
                           "pairwise": agreement(boxes, models[0], models[-1], thr)}
        persist[str(thr)] = {m: persistence(boxes, clip_of, m, thr) for m in models}
    wide = None
    positions = [float(x) for x in args.wide_positions.split(",") if x.strip()]
    if positions:
        print(f"agreement on {len(positions)} frames of every development clip ...", flush=True)
        wframes, wids = sample_wide_frames(splits, positions, width)
        wboxes = detect_all(dets, wframes)
        wide = {"frames": len(wframes), "clips": len(set(wids)), "positions": positions, "per_threshold": {}}
        for thr in thresholds:
            wide["per_threshold"][str(thr)] = {
                "counts": {m: {lab: sum(len(boxes_of(f, lab, thr)) for f in wboxes[m]) for lab in CLASSES} for m in models},
                "pairwise": agreement(wboxes, models[0], models[-1], thr)}
        del wframes

    bl = None
    if not args.skip_blvnav:
        print("BLV-Road-Nav keyframes ...", flush=True)
        bl = blvnav_eval(dets, thresholds, width, args.n_boot)
        for thr, blk in bl["thresholds"].items():
            bkey = next(k for k in blk if k.startswith("bootstrap_"))
            print(f"  conf>={thr}: " + " | ".join(f"{m} P {blk[m]['direct_micro']['precision']} R {blk[m]['direct_micro']['recall']} "
                                                  f"F1 {blk[m]['direct_micro']['f1']}" for m in models)
                  + f" | F1 diff {blk[bkey]['f1']['diff']:+.3f} CI {blk[bkey]['f1']['ci95']}", flush=True)

    # decision
    budget_ms = pipeline["budget_ms"] if pipeline["measured"] else conservative
    if bl:
        f1 = {thr: {m: bl["thresholds"][thr][m]["direct_micro"]["f1"] for m in models} for thr in bl["thresholds"]}
        rec = recommend(models, f1, budget_ms, args.budget_ms, models[0])
    else:
        rec = {"model": models[0], "more_accurate": None, "reason": "accuracy not measured (--skip-blvnav): keeping the fallback"}
    rec["pipeline_ms_used"] = "measured LivePipeline end-to-end median + resize" if pipeline["measured"] else "conservative bound"
    rec["pipeline_ms"] = budget_ms
    rec["conservative_ms"] = conservative
    notes = []
    if bl:
        for thr, blk in bl["thresholds"].items():
            bkey = next(k for k in blk if k.startswith("bootstrap_"))
            bs = blk[bkey]
            sig = "excludes" if (bs["f1"]["ci95"][0] > 0 or bs["f1"]["ci95"][1] < 0) else "includes"
            notes.append(f"conf {thr}: direct-class F1 {' vs '.join(f'{blk[m]['direct_micro']['f1']:.3f}' for m in models)} "
                         f"({bkey.replace('bootstrap_', '').replace('_minus_', ' - ')} = {bs['f1']['diff']:+.3f}, 95% CI "
                         f"[{bs['f1']['ci95'][0]:+.3f}, {bs['f1']['ci95'][1]:+.3f}] {sig} 0); recall "
                         f"{' vs '.join(f'{blk[m]['direct_micro']['recall']:.3f}' for m in models)}, precision "
                         f"{' vs '.join(f'{blk[m]['direct_micro']['precision']:.3f}' for m in models)}")
    for m in models:
        notes.append(f"`{m}`: detector median {pct(times[m], 50):.1f} ms (p95 {pct(times[m], 95):.1f}); full pipeline "
                     + (f"median {pipeline['budget_ms'][m]:.1f} ms incl. resize (end-to-end p95 {pipeline['per_model'][m]['end_to_end']['p95_ms']:.1f}); "
                        if pipeline["measured"] else "")
                     + f"conservative bound {conservative[m]:.1f} ms")
    lc = str(live_conf) if str(live_conf) in persist else str(thresholds[0])
    hi = str(max(thresholds))
    pp = {m: persist[lc][m]["person"] for m in models}
    notes.append("person boxes on consecutive EgoBlind frames 0.1 s apart (conf " + lc + "): persistence "
                 + " vs ".join(f"{pp[m]['persistence']:.3f} ({pp[m]['persisted']}/{pp[m]['boxes']})" for m in models)
                 + ", presence flip rate " + " vs ".join(f"{pp[m]['flip_rate']:.3f} ({pp[m]['presence_flips']}/{pp[m]['transitions']})" for m in models)
                 + " (" + ", ".join(f"`{m}`" for m in models) + "; 10 clips, so indicative only); a person who drops out and comes "
                 "back as a new track can be announced again")
    if hi != lc:
        notes.append(f"raising the detector floor from {lc} to {hi} does not help that: person flip rate "
                     + ", ".join(f"`{m}` {persist[lc][m]['person']['flip_rate']:.3f} -> {persist[hi][m]['person']['flip_rate']:.3f}" for m in models)
                     + f"; keep perception.yolo_conf = {live_conf} (the scheduler already ignores tracks under its own min_confidence)")
    rec["notes"] = notes
    rec["wire_up"] = {"cli": f"scripts/run_live.py --model {rec['model']}",
                      "config": {"perception.yolo_model": rec["model"], "perception.yolo_conf": live_conf, "perception.yolo_imgsz": imgsz}}
    print(f"\nrecommendation: {rec['model']} - {rec['reason']}", flush=True)

    caveats = [
        "BLV-Road-Nav keyframes are outdoor street scenes; the live complaint is a webcam on a desk indoors. Indoor classes "
        "(chair, dining table) have only a handful of labelled frames there, so the per-object rows for them are not informative; "
        "the EgoBlind agreement and persistence tables are the indoor-relevant evidence.",
        "Presence labels have no boxes, so a detection anywhere in the frame counts; precision is therefore an upper bound "
        "for box-level precision.",
        "The full-pipeline timing excludes OCR (EasyOCR every 8th frame adds roughly 85-130 ms on those frames, results/latency.json) "
        "and speech; both cost the same for either detector.",
        "Latency was measured on frames already in memory; camera capture runs in parallel with inference in the live loop.",
        "The EgoBlind perception cache (data/egoblind/cache, the replays behind run_on_egoblind and the holdout numbers) was "
        "built with yolo11n.pt; changing the live detector does not change any replay result, but the live demo then runs a "
        "different detector from the replays (yolo11s finds somewhat more boxes, mainly people, chairs and cars).",
        f"All timings are one machine ({platform.machine()}, {platform.mac_ver()[0] or platform.platform()}) on one day; MPS timings "
        "vary by a few ms between runs, so the per-round medians are reported to show the spread.",
    ]
    host = {"machine": f"{platform.machine()}, macOS {platform.mac_ver()[0]}",
            "processor": platform.processor(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "device": args.device, "python": platform.python_version()}
    try:
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
        if chip:
            host["machine"] = f"{chip}, macOS {platform.mac_ver()[0]}"
    except Exception:
        pass
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M"), "host": host, "models": models,
           "protocol": {"imgsz": imgsz, "width": width, "live_conf": live_conf, "detector_conf_floor": min(min(thresholds), live_conf),
                        "thresholds": thresholds, "splits": splits, "clips": args.clips, "per_clip": args.per_clip,
                        "step_s": args.step_s, "rounds": args.rounds, "warmup": args.warmup, "seed": args.seed,
                        "budget_ms": args.budget_ms, "depth_hi_ms": args.depth_hi_ms, "fallback": models[0],
                        "policy": args.policy},
           "recommendation": rec,
           "latency": {"frames": len(frames), "clips": len(clips), "rounds": args.rounds, "warmup": args.warmup,
                       "detector": {m: stats(times[m]) for m in models}, "per_round_median": per_round,
                       "resize_ms": stats(resize_ms)},
           "pipeline": pipeline, "conservative_ms": conservative,
           "blvnav": bl, "agreement": agree, "agreement_wide": wide, "persistence": persist,
           "frames": {"clips": clips}, "caveats": caveats,
           "seconds": round(time.perf_counter() - t_all, 1)}
    out = Path(args.out)
    dump_json(rep, out.with_suffix(".json"))
    write_md(rep, out.with_suffix(".md"))
    print(f"wrote {out.with_suffix('.json')} and .md in {rep['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
