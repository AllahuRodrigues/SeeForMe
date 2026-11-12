"""Does the cane (seeforme/live/cane.py) speak where EgoBlind users asked about obstacles, and how often otherwise?

EgoBlind development videos only (finetune + select; holdout and test are refused): frames are decoded at --fps
(2 per second) for the cane, the metric depth model the live pipeline would use for that video runs on each (the
label-free CLIP indoor / outdoor choice cached in data/egoblind/cache/<id>.npz when there is one, else the indoor
model; input size 308 as in scripts/run_live.py), and the cane runs as LiveController drives it (persistence, identity
memory, stop, path choice). Two streams per video:

    walking      every decision treated as walking: an upper bound on what the cane says (the clips are hand-held /
                 head-worn and mostly walking)
    activity     ActivityDetector on the same video at --activity-fps (4 per second): still -> only approaching
                 obstacles speak, as in the live demo
and, for selective attention, the cached YOLO detections (data/egoblind/cache, yolo11n; the live demo runs yolo11s) at
the nearest cached frame as tracks: an obstacle a detection explains is left to the scheduler. Without a cache, no
tracks (said in the report).

Measured:
    cues/min       spoken cane cues and stops per minute of processed video (a false-alarm proxy: most of the time
                   nobody asked about anything, and most of it is not a hazard)
    event hits     the share of EgoBlind should-speak events whose target is an obstacle type for which the cane says
                   something inside the event's useful window [t_q - 6 s, t_q + 0.5 s] (the event extraction of
                   seeforme/egoblind/data.py via the scenario files), split into targets the COCO detector covers
                   (bench, chair, table, ...) and targets it does not (door, stairs, pole, wall, fence, curb, tree,
                   sign, ...); also "sees": a confirmed obstacle ahead within 2 m at any decision in the window
    lift           the event hit rate over the hit rate of the same windows shifted to random times of the same video
                   (--shifts per event, not overlapping the event's window): 1 = no better than chance
    latency        cane analysis per depth map (ms), the depth model per frame (ms)

Proxy labels, honestly: an EgoBlind "event" is a question a blind user asked while recording, weakly labelled from the
question and its answers; "door" is often the destination ("how do I find the entrance?"), not something in the way,
and "fence" or "stairs" may be metres off to a side. The window is around the question, not around the object. So the
hit rate is a weak, noisy signal and the lift against random windows is the part worth reading.

    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_cane.py                     # ~15 min budget (about 17 min in all), writes results/live
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_cane.py --budget-min 2 --no-write   # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (str(SCRIPTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

DEV_SPLITS = ("finetune", "select")
OUT = ROOT / "results" / "live"
# obstacle-type targets: concepts the COCO detector cannot name (seeforme/egoblind/data.py NON_COCO_KEYWORDS) ...
NON_COCO_OBSTACLES = ("stairs", "door", "curb", "pole", "wall", "fence", "tree", "sign", "box", "cart", "trash can", "cone",
                      "shelf", "machine", "obstacle")
# ... and COCO classes that stand in the way (static furniture and street furniture)
COCO_OBSTACLES = ("bench", "chair", "couch", "dining table", "potted plant", "fire hydrant", "parking meter", "stop sign",
                  "suitcase", "bed", "refrigerator", "toilet")


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def obstacle_kind(e) -> Optional[str]:
    """'not_coco' | 'coco' | None for a should-speak event, from its weak labels (labels = COCO classes, concepts =
    the rest; the generic word "obstacle" is dropped from concepts by the extraction, so it is read back from the
    answers with negations stripped)."""
    from seeforme.egoblind.data import match_concepts, strip_negated
    labels = set(e.detail.get("labels") or [])
    concepts = set(e.detail.get("concepts") or [])
    answers = " ".join(e.detail.get("answers") or [])
    if "obstacle" in match_concepts(strip_negated(answers.lower()))[1]:
        concepts.add("obstacle")
    if labels & set(COCO_OBSTACLES):
        return "coco"
    if concepts & set(NON_COCO_OBSTACLES):
        return "not_coco"
    return None


def cached_meta(vid: int) -> Tuple[Optional[dict], Optional[object]]:
    from egoblind_common import cache_path
    from seeforme.egoblind.cache import PerceptionCache
    p = cache_path(vid)
    if not p.exists():
        return None, None
    c = PerceptionCache.load(str(p))
    return c.meta, c


def letterbox(path: Path, n: int = 12, dark: int = 20) -> Tuple[float, float, float, float]:
    """(x0, y0, x1, y1), normalised: the picture inside black bars (a portrait phone video letterboxed into a landscape
    file, or the reverse), from the columns and rows that stay dark in n frames spread over the first minute. A camera
    has no bars, and the depth model invents surfaces in them."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mx = None
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(total - 1, k * min(total, 60 * fps) / n)) if total else k)
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (160, int(160 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
        mx = g if mx is None else np.maximum(mx, g)
    cap.release()
    if mx is None:
        return 0.0, 0.0, 1.0, 1.0
    cols, rows = np.nonzero(mx.max(0) >= dark)[0], np.nonzero(mx.max(1) >= dark)[0]
    if not len(cols) or not len(rows):
        return 0.0, 0.0, 1.0, 1.0
    h, w = mx.shape
    box = [cols[0] / w, rows[0] / h, (cols[-1] + 1) / w, (rows[-1] + 1) / h]
    # bars narrower than 3% are compression borders, not letterboxing
    if box[0] < 0.03 and box[2] > 0.97:
        box[0], box[2] = 0.0, 1.0
    if box[1] < 0.03 and box[3] > 0.97:
        box[1], box[3] = 0.0, 1.0
    return tuple(float(v) for v in box)


def tracks_at(cache, t: float, depth: Optional[np.ndarray] = None, max_dt: float = 0.25,
              crop: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> List:
    """The cached detections of the frame nearest t (within max_dt) as visible, confirmed tracks; with a depth map their
    distances come from it (object_distances, as LivePipeline does), so the cane compares like with like."""
    from seeforme.perception.depth import object_distances
    from seeforme.types import Detection, Track
    if cache is None or not len(cache.frame_t):
        return []
    i = int(np.argmin(np.abs(cache.frame_t - t)))
    if abs(cache.frame_t[i] - t) > max_dt:
        return []
    x0, y0, x1, y1 = crop                 # cached boxes are in the full (letterboxed) frame: map them into the picture
    dets = []
    for j in range(cache._starts[i], cache._starts[i + 1]):
        b = cache.det_bbox[j]
        nb = (float(np.clip((b[0] - x0) / (x1 - x0), 0, 1)), float(np.clip((b[1] - y0) / (y1 - y0), 0, 1)),
              float(np.clip((b[2] - x0) / (x1 - x0), 0, 1)), float(np.clip((b[3] - y0) / (y1 - y0), 0, 1)))
        if nb[2] - nb[0] > 0.005 and nb[3] - nb[1] > 0.005:
            dets.append(Detection(cache.label_of(j), float(cache.det_conf[j]), nb))
    if depth is not None:
        object_distances(depth, dets)
    out = []
    for j, d in enumerate(dets):
        tr = Track(1000 + j, d.label, "object", d.bbox, d.confidence, t, t, confirmed=True)
        tr.distance_m, tr.depth_conf = d.distance_m, d.depth_conf
        out.append(tr)
    return out


def run_video(vid: int, path: Path, max_s: float, depth_est, cfg, args, cache) -> Dict:
    """Decode, depth, cane: per-decision records for the two streams."""
    import cv2
    from seeforme.live.activity import Activity, ActivityDetector
    from seeforme.live.cane import Cane
    from seeforme.perception.preprocess import assess_quality, enhance_dark, resize_for_inference
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    step_c = max(1, int(round(fps / args.fps)))
    step_a = max(1, int(round(fps / args.activity_fps)))
    crop = letterbox(path)
    canes = {"walking": Cane(), "activity": Cane(), "no_tracks": Cane()}
    act = ActivityDetector()
    rec: Dict[str, list] = {k: [] for k in canes}
    sees: List[Tuple[float, bool]] = []
    ms_cane: List[float] = []
    ms_depth: List[float] = []
    i = 0
    t = 0.0
    while True:
        need_a, need_c = i % step_a == 0, i % step_c == 0
        if not (need_a or need_c):
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if t > max_s:
            break
        i += 1
        if crop != (0.0, 0.0, 1.0, 1.0):
            fh, fw = frame.shape[:2]
            frame = frame[int(crop[1] * fh):int(round(crop[3] * fh)), int(crop[0] * fw):int(round(crop[2] * fw))]
        small = resize_for_inference(frame, cfg.perception.frame_width)
        q = assess_quality(small)
        if q.is_dark:
            small = enhance_dark(small)
        a = act.update(small, t, tracks_at(cache, t, crop=crop))
        if not need_c:
            continue
        t0 = time.perf_counter()
        depth = depth_est(small)
        ms_depth.append((time.perf_counter() - t0) * 1000)
        trks = tracks_at(cache, t, depth, crop=crop)
        moving_now = a not in (Activity.STATIONARY, Activity.READING)
        for k, cane in canes.items():
            t1 = time.perf_counter()
            cane.update(depth, t, None if k == "no_tracks" else trks)
            if k == "walking":
                ms_cane.append((time.perf_counter() - t1) * 1000)
            cue = cane.propose(t, moving=(moving_now if k == "activity" else True))
            if cue is not None:
                cane.commit(cue, t)
                rec[k].append({"t": round(t, 2), "action": cue.action, "message": cue.message, "d": cue.distance_m,
                               "path": getattr(cue, "cane_path", None), "conf": round(cane.sectors["ahead"].confidence, 2)})
        sa = canes["walking"].sectors["ahead"]
        sees.append((round(t, 2), bool(sa.confirmed(t, canes["walking"].cfg) and sa.distance <= canes["walking"].cfg.speak_m)))
    cap.release()
    return {"cues": rec, "sees": sees, "duration": t, "ms_cane": ms_cane, "ms_depth": ms_depth,
            "stats": {k: dict(c.stats) for k, c in canes.items()}, "crop": [round(v, 3) for v in crop]}


def fired(times: Sequence[float], a: float, b: float) -> bool:
    return any(a - 1e-6 <= x <= b + 1e-6 for x in times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default=",".join(DEV_SPLITS))
    ap.add_argument("--fps", type=float, default=2.0, help="depth maps (and cane decisions) per second of video")
    ap.add_argument("--activity-fps", type=float, default=4.0, help="frames per second for the activity detector")
    ap.add_argument("--depth-size", type=int, default=308, help="Depth Anything input size (run_live.py uses 308)")
    ap.add_argument("--budget-min", type=float, default=15.0, help="stop starting new videos after this many minutes")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--videos", default="", help="comma-separated video ids to process (development debugging)")
    ap.add_argument("--others", type=float, default=1.0, help="share of the remaining budget for videos without obstacle events")
    ap.add_argument("--shifts", type=int, default=50, help="random windows per event for the chance rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every cane cue")
    ap.add_argument("--report-only", action="store_true", help="re-render results/live/cane_eval_dev.md from the .json")
    args = ap.parse_args()
    if args.report_only:
        (OUT / "cane_eval_dev.md").write_text(report(json.loads((OUT / "cane_eval_dev.json").read_text())))
        print(f"wrote {OUT / 'cane_eval_dev.md'}")
        return
    splits = [s for s in args.splits.split(",") if s]
    bad = [s for s in splits if s not in DEV_SPLITS]
    if bad:
        raise SystemExit(f"development splits only ({', '.join(DEV_SPLITS)}); refused: {', '.join(bad)}")

    from _common import load_cfg
    from egoblind_common import load_egoblind_split, video_path
    from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
    cfg = load_cfg()
    t_start = time.perf_counter()
    rng = random.Random(args.seed)

    # ---- videos and events ----------------------------------------------------------------------------------
    items = []
    for split in splits:
        for sc in load_egoblind_split(split, require_cache=False, refresh=False):
            vp = video_path(sc.video_id)
            if vp is None:
                continue
            evs = [(e, obstacle_kind(e)) for e in sc.speak_events() if e.earliest_useful_s is not None]
            items.append({"split": split, "vid": sc.video_id, "path": vp, "events": evs, "sc": sc,
                          "has_obstacle": any(k for _, k in evs)})
    n_all = len(items)
    obst = sorted([it for it in items if it["has_obstacle"]], key=lambda it: it["vid"])
    others = [it for it in items if not it["has_obstacle"]]
    rng.shuffle(others)
    queue = obst + others
    if args.videos:
        want = {int(v) for v in args.videos.split(",")}
        queue = [it for it in queue if it["vid"] in want]
    if args.max_videos:
        queue = queue[:args.max_videos]

    models: Dict[str, object] = {}

    def depth_for(kind: str):
        if kind not in models:
            models[kind] = DepthAnythingEstimator(OUTDOOR if kind == "outdoor" else INDOOR, input_size=args.depth_size)
        return models[kind]

    done: List[Dict] = []
    skipped = 0
    for it in queue:
        el = (time.perf_counter() - t_start) / 60
        if el > args.budget_min:
            skipped += 1
            continue
        meta, cache = cached_meta(it["vid"])
        kind = "indoor"
        if meta is not None and meta.get("scene_method") == "clip" and meta.get("depth") in ("indoor", "outdoor"):
            kind = meta["depth"]
        qmax = max([e.detail["t_q"] for e, _ in it["events"]] + [q["t_q"] for q in it["sc"].queries] + [0.0])
        max_s = (meta or {}).get("max_seconds") or (qmax + 3.0)
        tv = time.perf_counter()
        r = run_video(it["vid"], it["path"], float(max_s), depth_for(kind), cfg, args, cache)
        r.update({"vid": it["vid"], "split": it["split"], "depth": kind, "depth_choice": "clip" if meta and meta.get("scene_method") == "clip" else "default indoor",
                  "has_cache": cache is not None, "seconds": round(time.perf_counter() - tv, 1), "item": it})
        done.append(r)
        print(f"[{el:5.1f} min] {it['split']}/{it['vid']}: {r['duration']:.0f}s video ({kind}), walking {len(r['cues']['walking'])} cues, "
              f"activity {len(r['cues']['activity'])}, {r['seconds']}s", flush=True)

    # ---- metrics --------------------------------------------------------------------------------------------
    streams = ("walking", "activity", "no_tracks")
    minutes = sum(r["duration"] for r in done) / 60
    per_min = {k: round(sum(len(r["cues"][k]) for r in done) / max(1e-6, minutes), 2) for k in streams}
    stops_min = {k: round(sum(sum(c["action"] == "haptic" for c in r["cues"][k]) for r in done) / max(1e-6, minutes), 2) for k in streams}
    msg_counts: Dict[str, int] = {}
    for r in done:
        for c in r["cues"]["walking"]:
            key = c["message"].split(",")[0].rstrip(".")
            msg_counts[key] = msg_counts.get(key, 0) + 1

    groups = {"not_coco": [], "coco": [], "other": []}
    for r in done:
        for e, k in r["item"]["events"]:
            a, b = float(e.earliest_useful_s), float(e.latest_safe_s)
            if b > r["duration"] + 0.01:
                continue                                   # the window runs past what was processed
            groups[k or "other"].append((r, e, a, b))

    def rates(evs, stream: str) -> Dict:
        hits, sees, chance, sees_chance, rows = 0, 0, [], [], []
        for r, e, a, b in evs:
            times = [c["t"] for c in r["cues"][stream]]
            st = [t for t, s in r["sees"] if s]
            h = fired(times, a, b)
            sv = fired(st, a, b)
            hits += h
            sees += sv
            L, D = b - a, r["duration"]
            draws = []
            for _ in range(args.shifts):
                for _try in range(20):
                    s0 = rng.uniform(0.0, max(0.0, D - L))
                    if s0 + L < a or s0 > b or D - L <= 0:
                        break
                draws.append((fired(times, s0, s0 + L), fired(st, s0, s0 + L)))
            chance.append(float(np.mean([d[0] for d in draws])) if draws else 0.0)
            sees_chance.append(float(np.mean([d[1] for d in draws])) if draws else 0.0)
            rows.append({"vid": r["vid"], "t_q": e.detail["t_q"], "window": [a, b], "labels": e.detail.get("labels"),
                         "concepts": e.detail.get("concepts"), "question": e.detail.get("question"), "hit": bool(h), "sees": bool(sv)})
        n = len(evs)
        out = {"n": n, "hits": hits, "hit_rate": round(hits / n, 3) if n else None, "hit_ci95": wilson(hits, n),
               "chance_rate": round(float(np.mean(chance)), 3) if n else None,
               "sees": sees, "sees_rate": round(sees / n, 3) if n else None, "sees_ci95": wilson(sees, n),
               "sees_chance_rate": round(float(np.mean(sees_chance)), 3) if n else None}
        out["lift"] = round(out["hit_rate"] / out["chance_rate"], 2) if n and out["chance_rate"] else None
        out["sees_lift"] = round(out["sees_rate"] / out["sees_chance_rate"], 2) if n and out["sees_chance_rate"] else None
        if n and out["chance_rate"]:     # bootstrap over events for the lift
            hv, cv = np.array([x["hit"] for x in rows], float), np.array(chance)
            br = np.random.default_rng(args.seed)
            boot = []
            for _ in range(2000):
                ix = br.integers(0, n, n)
                cm = cv[ix].mean()
                if cm > 0:
                    boot.append(hv[ix].mean() / cm)
            out["lift_ci95"] = [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)] if boot else None
        out["events"] = rows
        return out

    ev = {g: {s: rates(v, s) for s in streams} for g, v in groups.items()}
    ms_c = [x for r in done for x in r["ms_cane"]]
    ms_d = [x for r in done for x in r["ms_depth"]]

    def lat(v):
        return {"n": len(v), "mean_ms": round(float(np.mean(v)), 2), "p50_ms": round(float(np.median(v)), 2),
                "p95_ms": round(float(np.percentile(v, 95)), 2)} if v else {}
    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "splits": splits, "seconds": round(time.perf_counter() - t_start, 1),
        "settings": {"fps": args.fps, "activity_fps": args.activity_fps, "depth_size": args.depth_size, "budget_min": args.budget_min,
                     "max_videos": args.max_videos, "shifts": args.shifts, "seed": args.seed},
        "videos": {"available": n_all, "with_obstacle_events": len(obst), "processed": len(done), "skipped_budget": skipped,
                   "processed_with_obstacle_events": sum(1 for r in done if r["item"]["has_obstacle"]),
                   "minutes": round(minutes, 2), "depth_models": {k: sum(1 for r in done if r["depth"] == k) for k in ("indoor", "outdoor")},
                   "depth_choice": {k: sum(1 for r in done if r["depth_choice"] == k) for k in ("clip", "default indoor")},
                   "with_cached_detections": sum(1 for r in done if r["has_cache"]),
                   "letterboxed": sum(1 for r in done if r["crop"] != [0.0, 0.0, 1.0, 1.0])},
        "cues_per_min": per_min, "stops_per_min": stops_min, "messages_walking": dict(sorted(msg_counts.items(), key=lambda kv: -kv[1])),
        "events": ev,
        "latency": {"cane_update": lat(ms_c), "depth_model": lat(ms_d)},
        "cane_stats": {k: {s: sum(r["stats"][k][s] for r in done) for s in next(iter(done))["stats"][k]} for k in streams} if done else {},
        "per_video": [{"vid": r["vid"], "split": r["split"], "seconds_video": round(r["duration"], 1), "depth": r["depth"], "crop": r["crop"],
                       "cues": {k: len(r["cues"][k]) for k in streams}, "cue_log_walking": r["cues"]["walking"]} for r in done],
    }
    print(json.dumps({k: res[k] for k in ("videos", "cues_per_min", "stops_per_min", "latency")}))
    if args.verbose:
        for r in res["per_video"]:
            print(r["vid"], r["depth"], [(c["t"], c["message"], c["d"], c.get("conf")) for c in r["cue_log_walking"]])
    for g in ("not_coco", "coco", "other"):
        e = ev[g]["walking"]
        print(f"{g}: n={e['n']} hit {e['hit_rate']} chance {e['chance_rate']} lift {e['lift']} | sees {e['sees_rate']} chance {e['sees_chance_rate']}")
    if not args.no_write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "cane_eval_dev.json").write_text(json.dumps(res, indent=1, default=str))
        (OUT / "cane_eval_dev.md").write_text(report(res))
        print(f"wrote {OUT / 'cane_eval_dev.md'}")


def _verdict(name: str, e: Dict) -> str:
    lci = e.get("lift_ci95")
    if not e["n"] or not lci:
        return f"{name}: too few events to say anything."
    if lci[0] > 1.0:
        how = "more often than at random times (the interval excludes 1)"
    elif lci[1] < 1.0:
        how = "less often than at random times (the interval excludes 1)"
    else:
        how = "about as often as at random times (the interval includes 1): no evidence either way"
    return f"{name} ({e['n']} events): lift {e['lift']} ({lci[0]}-{lci[1]}), the cane speaks in their windows {how}."


def report(res: Dict) -> str:
    v, ev = res["videos"], res["events"]
    pv = res.get("per_video", [])
    rate = np.array([p["cues"]["walking"] / max(p["seconds_video"], 1.0) * 60 for p in pv]) if pv else np.zeros(1)

    def row(name, e):
        ci = e["hit_ci95"]
        lci = e.get("lift_ci95")
        return (f"| {name} | {e['n']} | {e['hits']} | {e['hit_rate']} ({ci[0]}-{ci[1]}) | {e['chance_rate']} | {e['lift']}"
                + (f" ({lci[0]}-{lci[1]})" if lci else "") + f" | {e['sees_rate']} | {e['sees_chance_rate']} | {e['sees_lift']} |"
                if e["n"] else f"| {name} | 0 | - | - | - | - | - | - | - |")
    L = [f"# Cane on EgoBlind development videos",
         "",
         f"Generated {res['generated']} by `python scripts/eval_live_cane.py` ({res['seconds'] / 60:.1f} min). Splits: "
         f"{', '.join(res['splits'])} (development only; holdout and test were not loaded).",
         "",
         f"Processed {v['processed']} of {v['available']} development videos with a local file ({v['minutes']:.1f} min of video, up to "
         f"3 s after each video's last question): all {v['processed_with_obstacle_events']} of the {v['with_obstacle_events']} videos with an "
         f"obstacle-type event first, then videos without one in a seeded random order until "
         + (f"{res['settings']['max_videos']} videos (--max-videos, a fixed set) or " if res['settings'].get('max_videos') else "")
         + f"the {res['settings']['budget_min']:.0f}-minute budget ran out ({v['available'] - v['processed']} videos not processed). Depth: Depth Anything V2 metric small at input "
         f"{res['settings']['depth_size']} (as `scripts/run_live.py`), {v['depth_models']['indoor']} indoor / {v['depth_models']['outdoor']} "
         f"outdoor: the cached label-free CLIP choice for {v['depth_choice']['clip']} videos, the indoor model (the live demo's default) for the "
         f"{v['depth_choice']['default indoor']} whose cache predates that record (their choice may have come from the labels, so it is not "
         f"reused). {v.get("letterboxed", 0)} letterboxed video(s) (a portrait phone picture inside a landscape file) were cropped to the picture "
         f"before depth, as a camera would deliver it: the depth model invents surfaces in black bars. The cane sees {res['settings']['fps']:g} depth maps per second and decides at each; the activity "
         f"detector sees {res['settings']['activity_fps']:g} frames per second. Selective attention uses the cached yolo11n detections "
         f"({v['with_cached_detections']} videos have them) as tracks.",
         "",
         "## Reading",
         "",
         f"- Cane cues per minute: **{res['cues_per_min']['walking']}** if the user were always walking (upper bound), "
         f"**{res['cues_per_min']['activity']}** with the activity detector (quiet while still unless something approaches), "
         f"{res['cues_per_min']['no_tracks']} walking without selective attention. Stops (haptic) per minute: walking "
         f"{res['stops_per_min']['walking']}, activity {res['stops_per_min']['activity']}.",
         f"- Per video (walking stream): median {np.median(rate):.1f} cues/min (quartiles {np.percentile(rate, 25):.1f}-"
         f"{np.percentile(rate, 75):.1f}); {np.mean(rate == 0):.0%} of the videos get none, {np.mean(rate > 10):.0%} more than 10 per minute. "
         "The busy ones are hand-held walks through shops, stations and crowds with counters, companions and escalators within a "
         "metre or two (several of the busiest looked at frame by frame); EgoBlind has no labels to tell real obstacles from noise there.",
         "- What it said (walking stream): " + ", ".join(f"{k} {n}" for k, n in res["messages_walking"].items()) + ".",
         "- " + _verdict("Obstacles the detector cannot name", ev["not_coco"]["walking"]),
         "- " + _verdict("Obstacles the detector covers", ev["coco"]["walking"]) + " A tracked chair or table is left to the scheduler "
         "(selective attention), so the cane is not expected to speak more here than elsewhere.",
         "- " + _verdict("Other should-speak events (control)", ev["other"]["walking"]),
         f"- Latency: cane update (analysis + persistence) {res['latency']['cane_update'].get('mean_ms')} ms mean, "
         f"{res['latency']['cane_update'].get('p95_ms')} ms p95 per depth map; the depth model itself "
         f"{res['latency']['depth_model'].get('mean_ms')} ms mean, {res['latency']['depth_model'].get('p95_ms')} ms p95.",
         "",
         "## Obstacle questions (walking stream, the upper bound)",
         "",
         "An event counts as a hit when the cane said something (a cue or a stop) inside its useful window [t_q - 6 s, t_q + 0.5 s]; "
         "**chance** is the same for windows of the same length at random times of the same video (not overlapping the event's), "
         "**lift** = hit rate / chance (bootstrap 95% interval over events). **Sees**: a confirmed obstacle within 2 m ahead at any decision "
         "in the window, whether or not the identity memory let it speak.",
         "",
         "| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |",
         "|---|---|---|---|---|---|---|---|---|",
         row("obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...)", ev["not_coco"]["walking"]),
         row("obstacle, COCO-covered (bench, chair, table, ...)", ev["coco"]["walking"]),
         row("other should-speak events (people, vehicles, unnamed)", ev["other"]["walking"]),
         "",
         "Same with the activity detector (what the live demo would say):",
         "",
         "| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |",
         "|---|---|---|---|---|---|---|---|---|",
         row("obstacle, not COCO", ev["not_coco"]["activity"]),
         row("obstacle, COCO-covered", ev["coco"]["activity"]),
         row("other should-speak events", ev["other"]["activity"]),
         "",
         "## What these labels are",
         "",
         "EgoBlind events are questions blind users asked while filming, weakly labelled from the question and the answers "
         "(`seeforme/egoblind/data.py`). An obstacle-type event names a door, stairs, a pole, a fence ... somewhere in the question "
         "or answer, which is often a destination (\"how can I find the entrance?\") or something to the side, not an obstacle in the "
         "walking corridor; its window is set by when the question was asked, not by when the object was close. The samples are small. "
         "Read the hit rates as weak evidence and the lift against random windows as the useful part; neither is a recall. Cues per minute "
         "count every cane cue as a potential false alarm, which overstates them (some are real walls and doors).",
         "",
         "## Obstacle events",
         "",
         "| video | t_q | names | question | hit | sees |",
         "|---|---|---|---|---|---|"]
    for g in ("not_coco", "coco"):
        for e in ev[g]["walking"]["events"]:
            names = ", ".join((e["concepts"] or []) + (e["labels"] or []))
            q = (e["question"] or "").replace("|", "/")[:70]
            L.append(f"| {e['vid']} | {e['t_q']:.1f} | {names} | {q} | {'yes' if e['hit'] else 'no'} | {'yes' if e['sees'] else 'no'} |")
    cs = res["cane_stats"].get("walking", {})
    L += ["", f"Cane counters (walking stream, summed): {', '.join(f'{k} {v}' for k, v in cs.items())}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
