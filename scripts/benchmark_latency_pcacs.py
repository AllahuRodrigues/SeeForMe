"""Latency of the P-CACS live stack on this machine: YOLO + Depth Anything V2 (metric, small) +
tracker with dynamics + scheduler, on the Ultralytics sample photos shifted frame to frame."""
import argparse, json, resource, time
import numpy as np
from _common import RESULTS, dump_json, load_cfg
from seeforme.perception import YoloDetector, NullOcr, EasyOcrReader, pick_device
from seeforme.perception.depth import DepthAnythingEstimator, INDOOR
from seeforme.pipeline import LivePipeline, make_live_policy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--depth-size", type=int, default=252)
ap.add_argument("--ocr", action="store_true")
ap.add_argument("--policy", default="pcacs")
args = ap.parse_args()

cfg = load_cfg(); cfg.perception.yolo_model = args.model
import cv2
from ultralytics.utils import ASSETS
base = [cv2.resize(cv2.imread(str(ASSETS / n)), (1280, 720)) for n in ("bus.jpg", "zidane.jpg")]
frames = [np.roll(base[(i // 40) % 2], shift=(i % 40) * 3, axis=1) for i in range(args.frames + 5)]
report = {"device": pick_device("auto"), "model": args.model, "depth_input": args.depth_size, "configs": {}}
for name, use_depth in (("yolo+depth+pcacs", True), ("yolo+pcacs(bbox depth)", False)):
    depth = DepthAnythingEstimator(INDOOR, input_size=args.depth_size) if use_depth else None
    pipe = LivePipeline(YoloDetector(cfg.perception), EasyOcrReader(cfg.perception) if args.ocr else NullOcr(),
                        make_live_policy(args.policy, cfg, "models"), cfg, depth=depth)
    for i, f in enumerate(frames[:5]):
        pipe.process(f, t=i / 30)
    pipe.latency = type(pipe.latency)(); pipe.latency.__dict__["depth"] = __import__("collections").deque(maxlen=200)
    t0 = time.perf_counter(); n_cues = 0
    for i, f in enumerate(frames[5:]):
        r = pipe.process(f, t=(i + 5) / 30); n_cues += r.cue is not None
    wall = time.perf_counter() - t0
    s = pipe.latency.summary()
    d = list(pipe.latency.__dict__["depth"])
    s["depth"] = {"mean_ms": round(float(np.mean(d)), 2), "p95_ms": round(float(np.percentile(d, 95)), 2)} if d else {}
    s["fps_end_to_end"] = round(args.frames / wall, 2); s["n_cues"] = n_cues
    s["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    report["configs"][name] = s
    print(name, json.dumps(s))
dump_json(report, RESULTS / "latency_pcacs.json")
print("->", RESULTS / "latency_pcacs.json")
