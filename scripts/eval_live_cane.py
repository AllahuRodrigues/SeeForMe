"""Does the cane (seeforme/live/cane.py) speak where EgoBlind users asked about obstacles, and how often otherwise,
for each travel aid (cane, guide-dog, none)?

EgoBlind development videos only (finetune + select; holdout and test are refused): frames are decoded at --fps
(2 per second) for the cane, the metric depth model the live pipeline would use for that video runs on each (the
label-free CLIP indoor / outdoor choice cached in data/egoblind/cache/<id>.npz when there is one, else the indoor
model; input size 308 as in scripts/run_live.py), the depth map is analysed once (floor, obstacles per sector), and
one Cane per aid and stream runs on that analysis as LiveController drives it (persistence, identity memory, stop, path
choice, the aid's rules). Streams per aid:

    walking      every decision treated as walking: an upper bound on what the cane says (the clips are hand-held /
                 head-worn and mostly walking)
    activity     ActivityDetector on the same video at --activity-fps (4 per second): still -> only approaching
                 obstacles speak, as in the live demo
    no_tracks    walking without selective attention (aids none and cane only)
and, for selective attention, the cached YOLO detections (data/egoblind/cache, yolo11n; the live demo runs yolo11s) at
the nearest cached frame as tracks: an obstacle a detection explains is left to the scheduler. Without a cache, no
tracks (said in the report). The aid changes only what the cane layer says; the analysis, and the scheduler's cues
about detected objects, are the same for every aid. Aid "none" is the behaviour before the aid existed.

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

    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_cane.py        # the reported 84 videos (make eval-cane; about
                                                                        # 16 min on a quiet machine, over an hour loaded)
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_cane.py --videos 710,183 --no-write       # smoke test
    ... --maps-cache <scratch>/maps.pkl        # keep the per-map analysis; a second run replays it in a minute, and
    ... --cane-set reach_m=1.0,detour_m=3.0    # ... tries other speaking thresholds on it (analysis settings need a
                                               #     fresh analysis); --no-write for such sweeps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
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
AIDS = ("none", "cane", "guide-dog")          # report order: before, the default, the dog
DEFAULT_AID = "cane"
# thresholds of the default aid tried one at a time on the same analysis (report only; the defaults follow the focus group)
SENSITIVITY = ("reach_m=1.0", "detour_m=2.0", "detour_m=3.0", "aid_closer_frac=0.15", "steady_resid_m=0.3",
               "reach_m=1.0,detour_m=3.0")
STREAMS = ("walking", "activity", "no_tracks")
NO_TRACKS_AIDS = ("none", DEFAULT_AID)
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


@dataclass
class SlimTrack:
    """What explained_by needs of a track (a picklable stand-in for the analysis cache)."""
    track_id: int
    label: str
    bbox: Tuple[float, float, float, float]
    distance_m: Optional[float]
    kind: str = "object"
    misses: int = 0
    remembered: bool = False


def tracks_at(cache, t: float, depth: Optional[np.ndarray] = None, max_dt: float = 0.25,
              crop: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> List[SlimTrack]:
    """The cached detections of the frame nearest t (within max_dt) as visible, confirmed tracks; with a depth map their
    distances come from it (object_distances, as LivePipeline does), so the cane compares like with like."""
    from seeforme.perception.depth import object_distances
    from seeforme.types import Detection
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
    return [SlimTrack(1000 + j, d.label, d.bbox, d.distance_m) for j, d in enumerate(dets)]


def analyse_video(vid: int, path: Path, max_s: float, depth_est, cfg, args, cache) -> Dict:
    """Decode, depth and the cane's analysis once per video: per depth map (t, CaneFrame, moving by the activity
    detector, tracks). The reference Cane (no aid: the aid changes no analysis) carries the floor prior."""
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
    ref = Cane(aid="none")
    act = ActivityDetector()
    maps: List[Dict] = []
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
        t1 = time.perf_counter()
        ref.update(depth, t, trks)                       # analysis + persistence, as live
        ms_cane.append((time.perf_counter() - t1) * 1000)
        maps.append({"t": t, "frame": ref.frame, "moving": a not in (Activity.STATIONARY, Activity.READING), "tracks": trks})
    cap.release()
    return {"maps": maps, "duration": t, "ms_cane": ms_cane, "ms_depth": ms_depth, "crop": [round(v, 3) for v in crop]}


def run_canes(maps: List[Dict], cane_set: Dict[str, float]) -> Dict:
    """One Cane per (aid, stream) on the shared analysis, deciding at every map as LiveController would."""
    from seeforme.live.cane import Cane, CaneConfig
    canes = {}
    for aid in AIDS:
        for st in STREAMS:
            if st != "no_tracks" or aid in NO_TRACKS_AIDS:
                canes[(aid, st)] = Cane(CaneConfig(**cane_set), aid=aid)
    rec: Dict[Tuple[str, str], list] = {k: [] for k in canes}
    sees: List[Tuple[float, bool]] = []
    for m in maps:
        t = m["t"]
        for (aid, st), cane in canes.items():
            cane.update(None, t, None if st == "no_tracks" else m["tracks"], frame=m["frame"])
            cue = cane.propose(t, moving=(m["moving"] if st == "activity" else True))
            if cue is not None:
                cane.commit(cue, t)
                rec[(aid, st)].append({"t": round(t, 2), "action": cue.action, "message": cue.message, "d": cue.distance_m,
                                       "path": getattr(cue, "cane_path", None), "conf": round(cane.sectors["ahead"].confidence, 2)})
        ref = canes[("none", "walking")]
        sa = ref.sectors["ahead"]
        sees.append((round(t, 2), bool(sa.confirmed(t, ref.cfg) and sa.distance <= ref.cfg.speak_m)))
    return {"cues": rec, "sees": sees, "stats": {k: dict(c.stats) for k, c in canes.items()}}


def fired(times: Sequence[float], a: float, b: float) -> bool:
    return any(a - 1e-6 <= x <= b + 1e-6 for x in times)


def _parse_set(spec: str) -> Dict[str, float]:
    from dataclasses import fields
    from seeforme.live.cane import CaneConfig
    names = {f.name for f in fields(CaneConfig)}
    out: Dict[str, float] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        if k.strip() not in names:
            raise SystemExit(f"--cane-set: CaneConfig has no field {k.strip()!r}")
        out[k.strip()] = float(v)
    return out


def _previous_run() -> Optional[Dict]:
    """Headline numbers of the report this run replaces, when it predates the travel aid (kept across re-runs)."""
    try:
        old = json.loads((OUT / "cane_eval_dev.json").read_text())
    except (OSError, ValueError):
        return None
    if "aids" in old:
        return old.get("previous_run")
    e = old.get("events", {})
    return {"generated": old.get("generated"), "note": "before the travel aid (every cue, speak_m 2 m): the aid 'none' behaviour",
            "videos": old.get("videos", {}).get("processed"), "minutes": old.get("videos", {}).get("minutes"),
            "cues_per_min": old.get("cues_per_min"), "stops_per_min": old.get("stops_per_min"),
            "messages_walking": old.get("messages_walking"),
            "lift": {g: {"lift": e.get(g, {}).get("walking", {}).get("lift"), "lift_ci95": e.get(g, {}).get("walking", {}).get("lift_ci95"),
                         "n": e.get(g, {}).get("walking", {}).get("n")} for g in ("not_coco", "coco", "other")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default=",".join(DEV_SPLITS))
    ap.add_argument("--fps", type=float, default=2.0, help="depth maps (and cane decisions) per second of video")
    ap.add_argument("--activity-fps", type=float, default=4.0, help="frames per second for the activity detector")
    ap.add_argument("--depth-size", type=int, default=308, help="Depth Anything input size (run_live.py uses 308)")
    ap.add_argument("--budget-min", type=float, default=120.0, help="stop starting new videos after this many minutes")
    ap.add_argument("--max-videos", type=int, default=84, help="the reported set: the 34 videos with obstacle events, then "
                    "50 others in a seeded order (0: as many as the budget allows)")
    ap.add_argument("--videos", default="", help="comma-separated video ids to process (development debugging)")
    ap.add_argument("--shifts", type=int, default=50, help="random windows per event for the chance rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps-cache", default="", help="pickle of the per-map analysis: read when present, else written "
                    "(keep it in a scratch folder; it holds no images)")
    ap.add_argument("--cane-set", default="", help="CaneConfig overrides for the speaking rules, e.g. reach_m=1.0,detour_m=3.0")
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the replay of the default aid with other thresholds")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every cane cue of the default aid")
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
    cane_set = _parse_set(args.cane_set)

    from _common import load_cfg
    from egoblind_common import load_egoblind_split, video_path
    from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
    cfg = load_cfg()
    t_start = time.perf_counter()
    rng = random.Random(args.seed)
    load_start = round(os.getloadavg()[0], 1)

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

    cache_path = Path(args.maps_cache) if args.maps_cache else None
    analysed: Dict[int, Dict] = {}
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            analysed = pickle.load(f)
        print(f"replaying the analysis of {len(analysed)} videos from {cache_path}", flush=True)
    models: Dict[str, object] = {}

    def depth_for(kind: str):
        if kind not in models:
            models[kind] = DepthAnythingEstimator(OUTDOOR if kind == "outdoor" else INDOOR, input_size=args.depth_size)
        return models[kind]

    done: List[Dict] = []
    skipped = 0
    n_new = 0
    for it in queue:
        el = (time.perf_counter() - t_start) / 60
        meta, cache = cached_meta(it["vid"])
        kind = "indoor"
        if meta is not None and meta.get("scene_method") == "clip" and meta.get("depth") in ("indoor", "outdoor"):
            kind = meta["depth"]
        tv = time.perf_counter()
        if it["vid"] in analysed:
            a = analysed[it["vid"]]
        elif el > args.budget_min:
            skipped += 1
            continue
        else:
            qmax = max([e.detail["t_q"] for e, _ in it["events"]] + [q["t_q"] for q in it["sc"].queries] + [0.0])
            max_s = (meta or {}).get("max_seconds") or (qmax + 3.0)
            a = analyse_video(it["vid"], it["path"], float(max_s), depth_for(kind), cfg, args, cache)
            analysed[it["vid"]] = a
            n_new += 1
        r = run_canes(a["maps"], cane_set)
        r.update({"vid": it["vid"], "split": it["split"], "depth": kind,
                  "depth_choice": "clip" if meta and meta.get("scene_method") == "clip" else "default indoor",
                  "has_cache": cache is not None, "seconds": round(time.perf_counter() - tv, 1), "item": it,
                  "duration": a["duration"], "ms_cane": a["ms_cane"], "ms_depth": a["ms_depth"], "crop": a["crop"]})
        done.append(r)
        print(f"[{el:5.1f} min] {it['split']}/{it['vid']}: {r['duration']:.0f}s video ({kind}), cues walking "
              + ", ".join(f"{aid} {len(r['cues'][(aid, 'walking')])}" for aid in AIDS) + f", {r['seconds']}s", flush=True)
    meta_path = cache_path.with_suffix(".meta.json") if cache_path is not None else None
    analysis = {"seconds": round(time.perf_counter() - t_start, 1), "load_average": [load_start, round(os.getloadavg()[0], 1)],
                "cpus": os.cpu_count(), "videos": n_new}
    if cache_path is not None and n_new:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(analysed, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta_path.write_text(json.dumps(analysis))
        print(f"wrote the analysis of {len(analysed)} videos to {cache_path}", flush=True)
    elif meta_path is not None and meta_path.exists():
        analysis = json.loads(meta_path.read_text())       # the run that made the replayed analysis
    if not done:
        raise SystemExit("no video processed")

    # ---- metrics --------------------------------------------------------------------------------------------
    minutes = sum(r["duration"] for r in done) / 60

    def grouped(runs: List[Dict]) -> Dict[str, list]:
        g: Dict[str, list] = {"not_coco": [], "coco": [], "other": []}
        for r in runs:
            for e, k in r["item"]["events"]:
                a, b = float(e.earliest_useful_s), float(e.latest_safe_s)
                if b > r["duration"] + 0.01:
                    continue                               # the window runs past what was processed
                g[k or "other"].append((r, e, a, b))
        return g
    groups = grouped(done)

    def rates(evs, key: Tuple[str, str], group: str) -> Dict:
        rr = random.Random(f"{args.seed}:{group}:{key[0]}:{key[1]}")     # draws do not depend on the other streams
        hits, sees, chance, sees_chance, rows = 0, 0, [], [], []
        for r, e, a, b in evs:
            times = [c["t"] for c in r["cues"][key]]
            st = [t for t, s in r["sees"] if s]
            h = fired(times, a, b)
            sv = fired(st, a, b)
            hits += h
            sees += sv
            L, D = b - a, r["duration"]
            draws = []
            for _ in range(args.shifts):
                for _try in range(20):
                    s0 = rr.uniform(0.0, max(0.0, D - L))
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
        out["lift_ci95"] = None
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

    by_aid: Dict[str, Dict] = {}
    for aid in AIDS:
        streams = [st for st in STREAMS if st != "no_tracks" or aid in NO_TRACKS_AIDS]
        msgs: Dict[str, int] = {}
        for r in done:
            for c in r["cues"][(aid, "walking")]:
                k = c["message"].split(",")[0].rstrip(".")
                msgs[k] = msgs.get(k, 0) + 1
        by_aid[aid] = {
            "cues_per_min": {st: round(sum(len(r["cues"][(aid, st)]) for r in done) / max(1e-6, minutes), 2) for st in streams},
            "stops_per_min": {st: round(sum(sum(c["action"] == "haptic" for c in r["cues"][(aid, st)]) for r in done) / max(1e-6, minutes), 2)
                              for st in streams},
            "messages_walking": dict(sorted(msgs.items(), key=lambda kv: -kv[1])),
            "events": {g: {st: rates(v, (aid, st), g) for st in streams} for g, v in groups.items()},
            "cane_stats": {st: {k: sum(r["stats"][(aid, st)][k] for r in done) for k in next(iter(done))["stats"][(aid, st)]}
                           for st in streams},
        }
    # sensitivity: the default aid's speaking thresholds varied one at a time on the same analysis (no new depth)
    sens = []
    for spec in ([] if args.no_sensitivity else SENSITIVITY):
        dv = [{**r, **run_canes(analysed[r["vid"]]["maps"], {**cane_set, **_parse_set(spec)})} for r in done]
        key = (DEFAULT_AID, "walking")
        e = rates(grouped(dv)["not_coco"], key, "not_coco")
        sens.append({"set": spec, "cues_per_min": round(sum(len(r["cues"][key]) for r in dv) / max(1e-6, minutes), 2),
                     "stops_per_min": round(sum(sum(c["action"] == "haptic" for c in r["cues"][key]) for r in dv) / max(1e-6, minutes), 2),
                     "not_coco": {k: e[k] for k in ("n", "hits", "hit_rate", "chance_rate", "lift", "lift_ci95")}})
        print(f"sensitivity {spec}: {sens[-1]['cues_per_min']} cues/min, not-COCO lift {e['lift']} {e['lift_ci95']}", flush=True)
    ms_c = [x for r in done for x in r["ms_cane"]]
    ms_d = [x for r in done for x in r["ms_depth"]]

    def lat(v):
        return {"n": len(v), "mean_ms": round(float(np.mean(v)), 2), "p50_ms": round(float(np.median(v)), 2),
                "p95_ms": round(float(np.percentile(v, 95)), 2)} if v else {}
    d = by_aid[DEFAULT_AID]
    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "splits": splits, "seconds": round(time.perf_counter() - t_start, 1),
        "settings": {"fps": args.fps, "activity_fps": args.activity_fps, "depth_size": args.depth_size, "budget_min": args.budget_min,
                     "max_videos": args.max_videos, "shifts": args.shifts, "seed": args.seed, "cane_set": cane_set,
                     "replayed_analysis": len(done) - n_new,
                     # the depth analysis: how long it took and the machine's load meanwhile (from the cache's
                     # sidecar when it was replayed)
                     "analysis": analysis if (n_new or analysis.get("videos")) else None,
                     "load_average": (analysis or {}).get("load_average") if (n_new or analysis.get("videos")) else None,
                     "cpus": os.cpu_count()},
        "videos": {"available": n_all, "with_obstacle_events": len(obst), "processed": len(done), "skipped_budget": skipped,
                   "processed_with_obstacle_events": sum(1 for r in done if r["item"]["has_obstacle"]),
                   "minutes": round(minutes, 2), "depth_models": {k: sum(1 for r in done if r["depth"] == k) for k in ("indoor", "outdoor")},
                   "depth_choice": {k: sum(1 for r in done if r["depth_choice"] == k) for k in ("clip", "default indoor")},
                   "with_cached_detections": sum(1 for r in done if r["has_cache"]),
                   "letterboxed": sum(1 for r in done if r["crop"] != [0.0, 0.0, 1.0, 1.0])},
        "default_aid": DEFAULT_AID,
        "aids": by_aid,
        "previous_run": _previous_run(),
        "sensitivity": sens,
        # the default aid's numbers at the top level, as before the aid existed
        "cues_per_min": d["cues_per_min"], "stops_per_min": d["stops_per_min"], "messages_walking": d["messages_walking"],
        "latency": {"cane_update": lat(ms_c), "depth_model": lat(ms_d)},
        "per_video": [{"vid": r["vid"], "split": r["split"], "seconds_video": round(r["duration"], 1), "depth": r["depth"], "crop": r["crop"],
                       "cues": {f"{aid}/{st}": len(v) for (aid, st), v in r["cues"].items()},
                       "cue_log_walking": {aid: r["cues"][(aid, "walking")] for aid in AIDS}} for r in done],
    }
    print(json.dumps({k: res[k] for k in ("videos", "latency")}))
    for aid in AIDS:
        a = by_aid[aid]
        e = a["events"]
        print(f"{aid:9s}: cues/min {a['cues_per_min']} stops/min {a['stops_per_min']} | " + " | ".join(
            f"{g} n={e[g]['walking']['n']} lift {e[g]['walking']['lift']} {e[g]['walking']['lift_ci95']}" for g in ("not_coco", "coco", "other")))
    if args.verbose:
        for r in res["per_video"]:
            print(r["vid"], r["depth"], [(c["t"], c["message"], c["d"], c.get("conf")) for c in r["cue_log_walking"][DEFAULT_AID]])
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


AID_RULES = {
    "none": "every cue, as before the aid existed: obstacles of any level within 2 m, the stop for any level",
    "cane": "no low obstacle within the cane's reach (1.2 m), beyond it only in a steady approach; body level from 1.2 "
            "to 2.5 m, said again only once 40% nearer; head level within 2.5 m always; the stop only for body or head "
            "level in a steady approach; path choice (\"... and on your left\") only for a body obstacle",
    "guide-dog": "head level within 2.5 m and the stop (body or head level, steady approach) only: the dog leads around "
                 "ground and body obstacles",
}


def _lift(e: Dict) -> str:
    if e.get("n") and e.get("hits") == 0:
        return f"no hits (0/{e['n']})"
    if not e.get("n") or e.get("lift") is None:
        return "-"
    lci = e.get("lift_ci95")
    return f"{e['lift']}" + (f" ({lci[0]}-{lci[1]})" if lci else "")


def report(res: Dict) -> str:
    v = res["videos"]
    aids = res["aids"]
    default = res.get("default_aid", "cane")
    order = [a for a in ("none", "cane", "guide-dog") if a in aids]
    pv = res.get("per_video", [])

    def rate_of(aid):
        return np.array([p["cues"][f"{aid}/walking"] / max(p["seconds_video"], 1.0) * 60 for p in pv]) if pv else np.zeros(1)

    def row(name, e):
        ci = e["hit_ci95"]
        return (f"| {name} | {e['n']} | {e['hits']} | {e['hit_rate']} ({ci[0]}-{ci[1]}) | {e['chance_rate']} | {_lift(e)}"
                f" | {e['sees_rate']} | {e['sees_chance_rate']} | {e['sees_lift']} |" if e["n"] else f"| {name} | 0 | - | - | - | - | - | - | - |")
    st = res["settings"]
    L = ["# Cane on EgoBlind development videos",
         "",
         f"Generated {res['generated']} by `python scripts/eval_live_cane.py` ({res['seconds'] / 60:.1f} min"
         + (f"; the depth analysis it replays took {st['analysis']['seconds'] / 60:.0f} min" if st.get("analysis") and st.get("replayed_analysis") else "")
         + "). Splits: "
         f"{', '.join(res['splits'])} (development only; holdout and test were not loaded).",
         "",
         f"Processed {v['processed']} of {v['available']} development videos with a local file ({v['minutes']:.1f} min of video, up to "
         f"3 s after each video's last question): all {v['processed_with_obstacle_events']} of the {v['with_obstacle_events']} videos with an "
         f"obstacle-type event first, then videos without one in a seeded random order until "
         + (f"{st['max_videos']} videos (--max-videos, a fixed set) or " if st.get('max_videos') else "")
         + f"the {st['budget_min']:.0f}-minute budget ran out ({v['available'] - v['processed']} videos not processed). Depth: Depth Anything V2 metric small at input "
         f"{st['depth_size']} (as `scripts/run_live.py`), {v['depth_models']['indoor']} indoor / {v['depth_models']['outdoor']} "
         f"outdoor: the cached label-free CLIP choice for {v['depth_choice']['clip']} videos, the indoor model (the live demo's default) for the "
         f"{v['depth_choice']['default indoor']} whose cache predates that record (their choice may have come from the labels, so it is not "
         f"reused). {v.get('letterboxed', 0)} letterboxed video(s) (a portrait phone picture inside a landscape file) were cropped to the picture "
         f"before depth, as a camera would deliver it: the depth model invents surfaces in black bars. The cane sees {st['fps']:g} depth maps per "
         f"second and decides at each; the activity detector sees {st['activity_fps']:g} frames per second. Selective attention uses the cached "
         f"yolo11n detections ({v['with_cached_detections']} videos have them) as tracks. Each depth map is analysed once and every travel aid "
         "runs on that analysis, so the aids differ only in what they say."
         + (f" Speaking thresholds overridden: {st['cane_set']}." if st.get("cane_set") else ""),
         "",
         "## Travel aid",
         "",
         "What the cane layer says depends on what the user travels with (`--aid`, default cane; the web app's \"I travel with\" "
         "setting; the server's `aid` field). It follows the BLV-Road-Nav focus group (Islam et al., arXiv 2407.16777): stay quiet "
         "about what the user's cane can already touch, warn about \"objects that pick you before you pick them\" (head height) "
         "and about what a cane finds too late. The scheduler's cues about detected objects (people, vehicles, chairs, a FIND goal) "
         "are not affected and are not counted here.",
         ""]
    for aid in order:
        L.append(f"- **{aid}**{' (default)' if aid == default else ''}: {AID_RULES.get(aid, '')}.")
    L += ["",
          "| aid | cues/min walking | cues/min activity | stops/min walking | stops/min activity | lift, not-COCO obstacles | lift, COCO obstacles | lift, other events (control) | what it said (walking) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for aid in order:
        a = aids[aid]
        ev = a["events"]
        L.append(f"| {aid}{' (default)' if aid == default else ''} | {a['cues_per_min']['walking']} | {a['cues_per_min']['activity']} | "
                 f"{a['stops_per_min']['walking']} | {a['stops_per_min']['activity']} | {_lift(ev['not_coco']['walking'])} | "
                 f"{_lift(ev['coco']['walking'])} | {_lift(ev['other']['walking'])} | "
                 + (", ".join(f"{k} {n}" for k, n in a["messages_walking"].items()) or "nothing") + " |")
    prev = res.get("previous_run")
    if prev:
        pl = prev.get("lift", {}).get("not_coco", {})
        L += ["", f"Before the aid existed (the report of {prev.get('generated')}, the same {prev.get('videos')} videos): "
              f"{prev.get('cues_per_min', {}).get('walking')} cues/min walking, {prev.get('cues_per_min', {}).get('activity')} with the "
              f"activity detector, {prev.get('stops_per_min', {}).get('walking')} stops/min, not-COCO lift {pl.get('lift')} "
              f"({(pl.get('lift_ci95') or ['?', '?'])[0]}-{(pl.get('lift_ci95') or ['?', '?'])[1]}); messages "
              + ", ".join(f"{k} {n}" for k, n in (prev.get("messages_walking") or {}).items())
              + ". The row \"none\" is that behaviour re-run (its chance windows are drawn per stream now, so its lift can differ a little)."]
    base, dflt = aids.get("none"), aids.get(default)
    if base and dflt:
        cb, cd = base["cues_per_min"]["walking"], dflt["cues_per_min"]["walking"]
        lb, ld = base["events"]["not_coco"]["walking"], dflt["events"]["not_coco"]["walking"]
        lci = ld.get("lift_ci95") or [None, None]
        if ld.get("lift") is None or lci[0] is None:
            verdict = "too few events to judge the lift."
        elif lci[0] > 1.0:
            verdict = (f"the lift for obstacles the detector cannot name holds ({_lift(lb)} without an aid, {_lift(ld)} with a cane): "
                       "the cues the aid drops are not disproportionately the ones near the obstacle questions.")
        elif ld["lift"] >= 1.0:
            verdict = (f"the lift for obstacles the detector cannot name falls from {_lift(lb)} to {_lift(ld)}: still above 1, but the "
                       "interval now includes 1, so with a cane there is no longer evidence that it speaks near those questions more than "
                       "elsewhere. The proxy labels are weak and the sample small (see below), so this is a drop in evidence, not proof of harm.")
        else:
            verdict = (f"the lift for obstacles the detector cannot name collapses from {_lift(lb)} to {_lift(ld)}: with a cane the layer "
                       "speaks no more often near those questions than elsewhere.")
        L += ["", f"**Reading.** With a cane (the default) the layer says {cd} cues/min instead of {cb} "
              f"({(1 - cd / cb) * 100 if cb else 0:.0f}% fewer) on the same videos; {verdict}"]
    sens = res.get("sensitivity") or []
    if sens:
        d0 = aids[default]
        L += ["", f"Sensitivity (aid {default}, walking stream; each row changes the speaking thresholds named, on the same analysis):", "",
              "| thresholds | cues/min | stops/min | not-COCO hits | lift, not-COCO obstacles |", "|---|---|---|---|---|",
              f"| as shipped (reach 1.2 m, detour 2.5 m, again 40% nearer) | {d0['cues_per_min']['walking']} | {d0['stops_per_min']['walking']} | "
              f"{d0['events']['not_coco']['walking']['hits']}/{d0['events']['not_coco']['walking']['n']} | {_lift(d0['events']['not_coco']['walking'])} |"]
        for x in sens:
            L.append(f"| {x['set'].replace(',', ', ')} | {x['cues_per_min']} | {x['stops_per_min']} | {x['not_coco']['hits']}/{x['not_coco']['n']} | "
                     f"{_lift(x['not_coco'])} |")
        L += ["", "The shipped thresholds come from the focus group's account of what a cane covers, not from this table; it shows "
              "how much the counts move with them on these videos (development splits only)."]
    a = aids[default]
    ev = a["events"]
    rate = rate_of(default)
    L += ["",
          f"## Reading (default aid: {default})",
          "",
          f"- Cane cues per minute: **{a['cues_per_min']['walking']}** if the user were always walking (upper bound), "
          f"**{a['cues_per_min']['activity']}** with the activity detector (quiet while still unless something approaches)"
          + (f", {a['cues_per_min']['no_tracks']} walking without selective attention" if "no_tracks" in a["cues_per_min"] else "")
          + f". Stops (haptic) per minute: walking {a['stops_per_min']['walking']}, activity {a['stops_per_min']['activity']}.",
          f"- Per video (walking stream): median {np.median(rate):.1f} cues/min (quartiles {np.percentile(rate, 25):.1f}-"
          f"{np.percentile(rate, 75):.1f}); {np.mean(rate == 0):.0%} of the videos get none, {np.mean(rate > 10):.0%} more than 10 per minute "
          + (f"(without an aid: median {np.median(rate_of('none')):.1f}, {np.mean(rate_of('none') > 10):.0%} above 10). " if "none" in aids else ". ")
          + "The busy ones are hand-held walks through shops, stations and crowds with counters, companions and escalators within a "
          "metre or two; EgoBlind has no labels to tell real obstacles from noise there.",
          "- What it said (walking stream): " + (", ".join(f"{k} {n}" for k, n in a["messages_walking"].items()) or "nothing") + ".",
          "- " + _verdict("Obstacles the detector cannot name", ev["not_coco"]["walking"]),
          "- " + _verdict("Obstacles the detector covers", ev["coco"]["walking"]) + " A tracked chair or table is left to the scheduler "
          "(selective attention), so the cane is not expected to speak more here than elsewhere.",
          "- " + _verdict("Other should-speak events (control)", ev["other"]["walking"]),
          f"- Latency: cane update (analysis + persistence) {res['latency']['cane_update'].get('mean_ms')} ms mean, "
          f"{res['latency']['cane_update'].get('p95_ms')} ms p95 per depth map; the depth model itself "
          f"{res['latency']['depth_model'].get('mean_ms')} ms mean, {res['latency']['depth_model'].get('p95_ms')} ms p95"
          + (f" (a shared machine: load average {st['load_average'][0]} at the start, {st['load_average'][1]} at the end, on "
             f"{st.get('cpus')} cores; the earlier report measured 3.6 ms and 46 ms on a quieter one)" if st.get("load_average") else "")
          + (". The analysis was replayed from --maps-cache, so these are the latencies of the run that made it" if st.get("replayed_analysis") else "")
          + ".",
          "",
          "## Obstacle questions",
          "",
          "An event counts as a hit when the cane said something (a cue or a stop) inside its useful window [t_q - 6 s, t_q + 0.5 s]; "
          "**chance** is the same for windows of the same length at random times of the same video (not overlapping the event's), "
          "**lift** = hit rate / chance (bootstrap 95% interval over events). **Sees**: a confirmed obstacle within 2 m ahead at any decision "
          "in the window, whether or not the identity memory or the aid let it speak (the same for every aid).",
          ""]
    for aid in order:
        e = aids[aid]["events"]
        for stream, what in (("walking", "walking stream, the upper bound"), ("activity", "with the activity detector, what the live demo would say")):
            L += [f"Aid {aid}, {what}:", "",
                  "| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |",
                  "|---|---|---|---|---|---|---|---|---|",
                  row("obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...)", e["not_coco"][stream]),
                  row("obstacle, COCO-covered (bench, chair, table, ...)", e["coco"][stream]),
                  row("other should-speak events (people, vehicles, unnamed)", e["other"][stream]), ""]
    L += ["## What these labels are",
          "",
          "EgoBlind events are questions blind users asked while filming, weakly labelled from the question and the answers "
          "(`seeforme/egoblind/data.py`). An obstacle-type event names a door, stairs, a pole, a fence ... somewhere in the question "
          "or answer, which is often a destination (\"how can I find the entrance?\") or something to the side, not an obstacle in the "
          "walking corridor; its window is set by when the question was asked, not by when the object was close. The samples are small. "
          "Read the hit rates as weak evidence and the lift against random windows as the useful part; neither is a recall. Cues per minute "
          "count every cane cue as a potential false alarm, which overstates them (some are real walls and doors). None of this says "
          "whether a cue was useful to a cane or guide-dog user: that needs a walking study.",
          "",
          "## Obstacle events (walking stream)",
          "",
          "| video | t_q | names | question | " + " | ".join(f"hit, {aid}" for aid in order) + " | sees |",
          "|---|---|---|---|" + "---|" * len(order) + "---|"]
    for g in ("not_coco", "coco"):
        rows = {aid: aids[aid]["events"][g]["walking"]["events"] for aid in order}
        for i, e in enumerate(rows[order[0]]):
            names = ", ".join((e["concepts"] or []) + (e["labels"] or []))
            q = (e["question"] or "").replace("|", "/")[:70]
            hits = " | ".join("yes" if rows[aid][i]["hit"] else "no" for aid in order)
            L.append(f"| {e['vid']} | {e['t_q']:.1f} | {names} | {q} | {hits} | {'yes' if e['sees'] else 'no'} |")
    L.append("")
    for aid in order:
        cs = aids[aid]["cane_stats"].get("walking", {})
        L.append(f"Cane counters, aid {aid} (walking stream, summed): {', '.join(f'{k} {v}' for k, v in cs.items())}.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
