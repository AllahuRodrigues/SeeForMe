"""Measure real inference latency of the perception stack on this machine.

Runs YOLO (and optionally OCR) on the sample photographs that ship with
Ultralytics (bus.jpg, zidane.jpg: people, a bus, text), resized to 720p and
shifted a few pixels per frame so the tracker and the cue policy do real work.
The first frames are treated as warm-up and excluded; reports steady-state
per-stage latency, end-to-end FPS and peak memory.
"""
import argparse, json, resource, time
import numpy as np
from _common import RESULTS, dump_json, load_cfg
from seeforme.perception import YoloDetector, EasyOcrReader, NullOcr, pick_device
from seeforme.cues import CACSPolicy
from seeforme.pipeline import Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="yolo11n.pt,yolo11s.pt")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--ocr", action="store_true")
ap.add_argument("--device", default="auto")
args = ap.parse_args()

cfg = load_cfg()
cfg.perception.device = args.device
report = {"device": pick_device(args.device), "models": {}}
import cv2
from ultralytics.utils import ASSETS
base = [cv2.resize(cv2.imread(str(ASSETS / n)), (1280, 720)) for n in ("bus.jpg", "zidane.jpg")]
frames = []
for i in range(args.frames + 5):
    img = base[(i // 40) % 2]                       # switch scene every 40 frames
    frames.append(np.roll(img, shift=(i % 40) * 3, axis=1))   # small lateral shift = camera motion
ocr = EasyOcrReader(cfg.perception) if args.ocr else NullOcr()

for m in args.models.split(","):
    cfg.perception.yolo_model = m
    det = YoloDetector(cfg.perception)
    pipe = Pipeline(det, ocr, CACSPolicy(cfg.cacs), cfg)
    for i, f in enumerate(frames[:5]):              # warm-up (graph compilation, OCR model load)
        pipe.process(f, t=i / 30)
    from seeforme.pipeline import LatencyStats
    pipe.latency = LatencyStats()
    t0 = time.perf_counter()
    for i, f in enumerate(frames[5:]):
        pipe.process(f, t=(i + 5) / 30)
    wall = time.perf_counter() - t0
    lat = pipe.latency.summary()
    report["models"][m] = {**lat, "fps_end_to_end": round(args.frames / wall, 2),
                           "n_tracks_last_frame": len(pipe.tracker.tracks), "n_cues": len(pipe.cue_log),
                           "cues": [c.message for c in pipe.cue_log][:8],
                           "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 1)}
    print(m, json.dumps(report["models"][m]))
dump_json(report, RESULTS / "latency.json")
