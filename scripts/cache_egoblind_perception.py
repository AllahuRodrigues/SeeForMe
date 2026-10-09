"""Run YOLO11 + Depth Anything V2 once over every downloaded EgoBlind video and cache the detections
(data/egoblind/cache/<video_id>.npz). Resumable; safe to run in the background while videos keep arriving:

    nohup .venv/bin/python scripts/cache_egoblind_perception.py --splits holdout,select,finetune,train,test > data/egoblind/cache_stdout.log 2>&1 &

Each video is processed at --fps frames per second up to --after-last-question seconds after its last
question (the part that carries labels). The indoor or outdoor metric-depth model is chosen per video
by CLIP zero-shot scene classification of its first 2 s of frames (label-free; --scene labels picks from its questions
instead, as an upper-bound ablation; --depth indoor|outdoor forces one). OCR is off: EasyOCR would
multiply the cost and EgoBlind questions about text are not scheduler events.
"""
import argparse, time
from pathlib import Path
from _common import load_cfg
from egoblind_common import CACHE, EGO, SPLITS, cache_path, load_egoblind_questions, video_path
from seeforme.egoblind import build_cache
from seeforme.egoblind.data import PROACTIVE_SEVERITY, guess_outdoor

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="holdout,select,finetune,train,test")
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--after-last-question", type=float, default=3.0)
ap.add_argument("--frame-width", type=int, default=640)
ap.add_argument("--depth", default="auto", choices=["auto", "indoor", "outdoor", "none"])
ap.add_argument("--scene", default="clip", choices=["clip", "labels"],
                help="how 'auto' picks the indoor or outdoor depth model: clip = CLIP zero-shot on the first 2 s of frames (label-free, "
                     "what a deployed system can do); labels = from the video's EgoBlind questions and answers (annotation leak, "
                     "kept only as an upper-bound ablation)")
ap.add_argument("--depth-size", type=int, default=252)
ap.add_argument("--limit", type=int, default=0, help="stop after this many newly cached videos")
ap.add_argument("--order", default="proactive", choices=["proactive", "id"], help="videos with safety/navigation questions first")
ap.add_argument("--force", action="store_true")
ap.add_argument("--detector", default="", choices=["", "yolo", "yolo-world"], help="override cfg.perception.detector (yolo-world = open vocabulary)")
ap.add_argument("--shard", default="", help="k/n: process only every n-th video (k-th residue) so several cachers can share a split")
ap.add_argument("--watch-minutes", type=float, default=0.0, help="re-scan for newly downloaded videos every N minutes (0 = one pass)")
ap.add_argument("--max-hours", type=float, default=12.0)
args = ap.parse_args()

cfg = load_cfg()
from seeforme.perception import make_detector
if args.detector:
    cfg.perception.detector = args.detector
det = make_detector(cfg.perception)
det_name = cfg.perception.open_vocab_model if cfg.perception.detector == "yolo-world" else cfg.perception.yolo_model
depth_models = {}
scene = None
if args.depth == "auto" and args.scene == "clip":
    from seeforme.perception.scene import SceneClassifier
    scene = SceneClassifier()


def depth_for(outdoor: bool):
    if args.depth == "none":
        return None, ""
    key = "outdoor" if (args.depth == "outdoor" or (args.depth == "auto" and outdoor)) else "indoor"
    if key not in depth_models:
        from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
        depth_models[key] = DepthAnythingEstimator(OUTDOOR if key == "outdoor" else INDOOR, input_size=args.depth_size)
    return depth_models[key], key


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(EGO / "cache.log", "a") as f:
        f.write(line + "\n")


CACHE.mkdir(parents=True, exist_ok=True)
t_all = time.perf_counter()
questions = {split: load_egoblind_questions(split) for split in args.splits.split(",")}


def one_pass():
    done = skipped_missing = 0
    for split, by_video in questions.items():
        items = sorted(by_video.items())
        if args.order == "proactive":
            items.sort(key=lambda kv: (-sum(q.qtype in PROACTIVE_SEVERITY for q in kv[1]), kv[0]))
        if args.shard:
            k, n = (int(x) for x in args.shard.split("/"))
            if not 0 <= k < n:
                raise SystemExit(f"--shard {args.shard}: need 0 <= k < n")
            items = [kv for i, kv in enumerate(items) if i % n == k]
        for vid, qs in items:
            out = cache_path(vid)
            if out.exists() and not args.force:
                continue
            vp = video_path(vid)
            if vp is None:
                skipped_missing += 1
                continue
            max_s = max(q.t_q for q in qs) + args.after_last_question
            p_out = scene.classify_video(vp) if scene is not None else None
            outdoor = (p_out > 0.5) if p_out is not None else guess_outdoor(qs)
            depth, dname = depth_for(outdoor)
            t0 = time.perf_counter()
            try:
                cache = build_cache(vp, det, depth, target_fps=args.fps, frame_width=args.frame_width, max_seconds=max_s,
                                    depth_name=dname, detector_name=det_name)
            except Exception as e:
                log(f"{split}/{vid}: FAILED {type(e).__name__}: {e}")
                continue
            cache.meta.update({"scene_method": "clip" if p_out is not None else ("labels" if args.depth == "auto" else args.depth),
                               "scene_outdoor_prob": None if p_out is None else round(p_out, 4),
                               "video_id": vid, "split": split, "path": str(out), "n_questions": len(qs),
                               "n_proactive": sum(q.qtype in PROACTIVE_SEVERITY for q in qs)})
            cache.save(out)
            done += 1
            if cache.meta.get("truncated"):
                log(f"{split}/{vid}: WARNING decode stopped at {cache.duration:.0f}s of {cache.meta.get('n_frames_total', 0) / max(1e-6, cache.meta.get('fps', 30)):.0f}s "
                    f"({cache.meta.get('read_failures', 0)} unreadable frames); events after it are marked cache_truncated")
            log(f"{split}/{vid}: {cache.n_frames} frames ({cache.duration:.0f}s of video, {dname or 'no depth'}), {cache.n_detections} detections, "
                f"{time.perf_counter() - t0:.1f}s")
            if args.limit and done >= args.limit:
                return done, skipped_missing
    return done, skipped_missing


total = 0
while True:
    done, missing = one_pass()
    total += done
    log(f"pass: cached {done} videos ({total} this run, {(time.perf_counter() - t_all) / 60:.1f} min); {missing} videos not downloaded yet; "
        f"{len(list(CACHE.glob('*.npz')))} caches total in {CACHE}")
    if args.limit and total >= args.limit:
        break
    if args.watch_minutes <= 0 or missing == 0 or (time.perf_counter() - t_all) / 3600 > args.max_hours:
        break
    time.sleep(args.watch_minutes * 60)
