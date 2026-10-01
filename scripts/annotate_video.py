"""Draft a CampusBench-style annotation for a real recording.

    python scripts/annotate_video.py walk.mp4 --out data/real/walk.json [--outdoor] [--no-ocr]

Runs YOLO + metric depth + OCR + the tracker with dynamics over the video and writes, for every
track that lasted >= 1 s, a draft event with the times at which it was first seen, came within
6 m, and was closest, plus the slots and ideal cue the composer would produce. Edit the draft by
hand (should_speak, earliest_useful_s, ideal_s, latest_safe_s, minimum_information, ideal_cue),
then score a cue log against it with `scripts/score_video.py --campus`.
"""
import argparse, json, math
from pathlib import Path
import cv2
from _common import load_cfg
from seeforme.cues.compose import compose
from seeforme.cues.pcacs import event_type_for
from seeforme.campus.world import EVENT_SPECS
from seeforme.geometry import CameraModel, TrackDynamics
from seeforme.perception import YoloDetector, EasyOcrReader, NullOcr
from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR, object_distances
from seeforme.perception.preprocess import resize_for_inference
from seeforme.tracking.tracker import IoUTracker

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--out", required=True)
ap.add_argument("--outdoor", action="store_true"); ap.add_argument("--no-ocr", action="store_true")
ap.add_argument("--every", type=int, default=3, help="process every n-th frame")
args = ap.parse_args()

cfg = load_cfg()
cam = CameraModel()
det = YoloDetector(cfg.perception); ocr = NullOcr() if args.no_ocr else EasyOcrReader(cfg.perception)
depth = DepthAnythingEstimator(OUTDOOR if args.outdoor else INDOOR, input_size=252)
tracker = IoUTracker(cfg.tracker, dynamics=TrackDynamics(cam))
cap = cv2.VideoCapture(args.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
hist = {}   # track id -> list of (t, z, x, label, text)
i = 0; text_cache = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % args.every:
        i += 1; continue
    t = i / fps
    small = resize_for_inference(frame, cfg.perception.frame_width)
    dets = det.detect(small)
    if not args.no_ocr and (i // args.every) % 4 == 0:
        text_cache = ocr.read(small)
    dets += text_cache
    object_distances(depth(small), dets)
    for tr in tracker.update(dets, t):
        hist.setdefault(tr.track_id, []).append((t, tr.distance_m, tr.lateral_m, tr.label, tr.text, tr.path_prob))
    i += 1
duration = i / fps
events = []
for tid, h in hist.items():
    if h[-1][0] - h[0][0] < 1.0:
        continue
    label, text = h[-1][3], h[-1][4]
    kind = "text" if label == "text" else "object"
    etype = "sign_nav" if kind == "text" else event_type_for(type("T", (), {"kind": kind, "label": label})())
    spec = EVENT_SPECS.get(etype, EVENT_SPECS["decor"])
    near = [s for s in h if s[1] is not None and s[1] <= spec.notice_m] if spec.severity > 0 else []
    closest = min(h, key=lambda s: s[1] if s[1] is not None else 1e9)
    z, x = closest[1] or 0.0, closest[2]
    angle = math.degrees(math.atan2(x, max(0.1, z)))
    slots = ["object"] + list(spec.slots)
    cue = compose(slots, label, kind, text, distance_m=z, angle_deg=angle) if spec.severity > 0 else ""
    events.append({"track_id": tid, "label": label, "kind": kind, "text": text, "event_type": etype,
                   "should_speak": bool(near) and max(s[5] for s in h) > 0.5, "reason": "" if near else "draft: never in range / off path",
                   "severity": spec.severity, "first_seen_s": round(h[0][0], 2), "last_seen_s": round(h[-1][0], 2),
                   "earliest_useful_s": round(near[0][0], 2) if near else None, "ideal_s": None,
                   "latest_safe_s": None, "closest_s": round(closest[0], 2), "closest_distance_m": round(z, 2),
                   "minimum_information": slots, "ideal_cue": cue, "note": "DRAFT - edit times, should_speak and cue"})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({"video": args.video, "duration": duration, "fps": fps, "events": events}, indent=1))
print(f"{len(events)} draft events -> {args.out}")
