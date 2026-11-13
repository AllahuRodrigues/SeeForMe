"""Cache YOLO11 + metric depth for VIABench reminder videos (data/viabench/cache/<id>.npz), the same perception as EgoBlind.

    python scripts/cache_viabench_perception.py --limit 200 --max-seconds 100 --shard 0/3

Each video is first re-timed to 5 fps at 640 px wide with ffmpeg (software decode is about 8x real time; decoding
every 1080p frame in OpenCV is slower than real time), then passed to the EgoBlind cache builder. Only the first
--max-seconds of each video are processed, so more distinct sources fit in the same compute. Videos are taken in a
seeded order that is balanced over the source platforms. `via_test` is a lockbox and is never cached here. Resumable.
"""
import argparse, os, subprocess, tempfile, time
from pathlib import Path

from _common import load_cfg, ROOT
from viabench_common import CACHE, DEV_SPLITS, VIA, caching_order, load_index
from seeforme.egoblind import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=200, help="position in the seeded order up to which videos are cached")
ap.add_argument("--max-seconds", type=float, default=100.0)
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--shard", default="", help="k/n: every n-th video of the order")
ap.add_argument("--depth", default="auto", choices=["auto", "none"])
args = ap.parse_args()

cfg = load_cfg()
from seeforme.perception import make_detector
here = os.getcwd(); os.chdir(ROOT / "weights")              # the detector resolves its weights relative to the working directory
try:
    det = make_detector(cfg.perception)
finally:
    os.chdir(here)
scene, depth_models = None, {}
if args.depth == "auto":
    from seeforme.perception.scene import SceneClassifier
    scene = SceneClassifier()


def depth_for(outdoor: bool):
    if args.depth == "none":
        return None, ""
    key = "outdoor" if outdoor else "indoor"
    if key not in depth_models:
        from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
        depth_models[key] = DepthAnythingEstimator(OUTDOOR if outdoor else INDOOR, input_size=252)
    return depth_models[key], key


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(VIA / "cache.log", "a") as f:
        f.write(line + "\n")


CACHE.mkdir(parents=True, exist_ok=True)
todo = [v for v in caching_order(load_index()) if v["split"] in DEV_SPLITS][:args.limit]
if args.shard:
    k, n = (int(x) for x in args.shard.split("/"))
    todo = [v for i, v in enumerate(todo) if i % n == k]
t_all, done = time.perf_counter(), 0
for v in todo:
    out = CACHE / f"{v['id']}.npz"
    if out.exists():
        continue
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        small = Path(td) / "v.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", f"{args.max_seconds:g}", "-i", str(v["path"]), "-vf",
                            f"fps={args.fps:g},scale=640:-2", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15", str(small)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not small.exists():
            log(f"{v['key']}: FAILED ffmpeg: {r.stderr.strip()[:200]}"); continue
        try:
            p_out = scene.classify_video(small) if scene is not None else None
            depth, dname = depth_for(bool(p_out is None or p_out > 0.5))
            cache = build_cache(small, det, depth, target_fps=args.fps, frame_width=640, depth_name=dname, detector_name=cfg.perception.yolo_model)
        except Exception as e:
            log(f"{v['key']}: FAILED {type(e).__name__}: {e}"); continue
    cache.meta.update({"dataset": "viabench", "video_key": v["key"], "video_id": v["id"], "split": v["split"], "path": str(out),
                       "max_seconds": args.max_seconds, "scene_outdoor_prob": None if p_out is None else round(float(p_out), 4)})
    cache.save(out)
    done += 1
    log(f"{v['split']}/{v['id']} {v['key'].split('/', 2)[2]}: {cache.n_frames} frames, {cache.n_detections} detections, {time.perf_counter() - t0:.0f}s "
        f"({done} this run, {(time.perf_counter() - t_all) / 60:.0f} min)")
log(f"shard {args.shard or 'all'} finished: {done} cached, {len(list(CACHE.glob('*.npz')))} caches in {CACHE}")
