"""Run the full pipeline on a video file; write an annotated video and a cue log (JSONL).

    python scripts/run_video.py input.mp4 --policy cacs --out results/demo_cacs.mp4
The cue log can be scored against a hand-made annotation file with scripts/score_video.py.
"""
import argparse, json
import cv2
from _common import MODELS, RESULTS, load_cfg
from seeforme.perception import YoloDetector, EasyOcrReader, NullOcr
from seeforme.pipeline import LivePipeline, make_live_policy
from seeforme.ui import draw_overlay_pcacs

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--policy", default="pcacs")
ap.add_argument("--out", default=None)
ap.add_argument("--no-ocr", action="store_true")
ap.add_argument("--no-depth", action="store_true")
ap.add_argument("--outdoor", action="store_true")
ap.add_argument("--goal", default="")
ap.add_argument("--max-frames", type=int, default=0)
args = ap.parse_args()

cfg = load_cfg()
depth = None
if not args.no_depth:
    from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
    depth = DepthAnythingEstimator(OUTDOOR if args.outdoor else INDOOR, input_size=252)
goal_tags = [g.strip() for g in args.goal.split(",") if g.strip()]
pipe = LivePipeline(YoloDetector(cfg.perception), NullOcr() if args.no_ocr else EasyOcrReader(cfg.perception),
                    make_live_policy(args.policy, cfg, str(MODELS), goal_tags=goal_tags), cfg, depth=depth)
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
out_path = args.out or str(RESULTS / f"{args.policy}_annotated.mp4")
writer = None
i = 0; last_cue = None; last_t = -1e9
log_path = out_path.rsplit(".", 1)[0] + "_cues.jsonl"
with open(log_path, "w") as log:
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        t = i / fps
        res = pipe.process(frame, t)
        if res.cue is not None:
            last_cue, last_t = res.cue, t
            log.write(json.dumps({"t": t, "message": res.cue.message, "label": res.cue.label, "score": res.cue.score,
                                  "track_id": res.cue.track_id, "hedged": res.cue.hedged, "action": res.cue.action,
                                  "duration_s": res.cue.duration_s, "slots": list(res.cue.slots), "distance_m": res.cue.distance_m,
                                  "ttc_s": res.cue.ttc_s, "utility": res.cue.utility}) + "\n")
        small = cv2.resize(frame, (cfg.perception.frame_width, int(frame.shape[0] * cfg.perception.frame_width / frame.shape[1])))
        vis = draw_overlay_pcacs(small, res.tracks, pipe.policy.last_observations(), last_cue if t - last_t < 2.5 else None,
                                 {"policy": pipe.policy.name, "t": f"{t:.1f}s", "det": f"{res.latency_ms['detect']:.0f}ms",
                                  "depth": f"{res.latency_ms.get('depth', 0):.0f}ms"}, pipe.policy)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        i += 1
if writer: writer.release()
print(f"{i} frames, {len(pipe.cue_log)} cues -> {out_path}, log {log_path}")
print("latency:", pipe.latency.summary())
