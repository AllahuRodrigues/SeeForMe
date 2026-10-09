"""How well does the open-vocabulary detector see doors and stairs in blind users' egocentric video?

    python scripts/probe_openvocab_prompts.py [--videos 30] [--conf 0.05]

Takes the EgoBlind train videos whose questions or answers mention a door or stairs, grabs two frames around each
question (t_q - 3 s and t_q), runs YOLO-World with a set of alternative prompts and counts detections per prompt at
any confidence and at the caching threshold (0.25). Writes results/egoblind/openvocab_prompt_probe.md.
"""
import argparse, sys
from collections import Counter
import cv2
from _common import ROOT
import egoblind_common as ec
from seeforme.config import PerceptionConfig
from seeforme.perception.detector import OpenVocabDetector

ap = argparse.ArgumentParser()
ap.add_argument("--videos", type=int, default=30)
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--prompts", default="door,doorway,open door,entrance,exit,glass door,sliding door,gate,elevator door,"
                                     "stairs,staircase,steps,stairway,escalator,handrail,wall,person")
args = ap.parse_args()
prompts = [p.strip() for p in args.prompts.split(",")]
qs = ec.load_egoblind_questions("train")
cands = []
for vid, ql in sorted(qs.items()):
    for q in ql:
        txt = (q.question + " " + " ".join(q.answers)).lower()
        if ("door" in txt or "stair" in txt) and ec.video_path(vid) is not None:
            cands.append((vid, q.t_q)); break
det = OpenVocabDetector(PerceptionConfig(yolo_conf=args.conf), vocab=[(p, p) for p in prompts])
hits, hits25, frames = Counter(), Counter(), 0
for vid, t_q in cands[:args.videos]:
    cap = cv2.VideoCapture(str(ec.video_path(vid))); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    for dt in (-3, 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t_q + dt) * fps))); ok, fr = cap.read()
        if not ok:
            continue
        frames += 1
        small = cv2.resize(fr, (640, int(fr.shape[0] * 640 / fr.shape[1])))
        for d in det.detect(small):
            hits[d.label] += 1
            if d.confidence >= 0.25:
                hits25[d.label] += 1
    cap.release()
lines = [f"# YOLO-World prompts on {frames} frames from {min(args.videos, len(cands))} EgoBlind videos that mention a door or stairs",
         "", f"Detector `yolov8s-worldv2.pt`, imgsz 640, confidence floor {args.conf}; frames at t_q - 3 s and t_q.", "",
         "| prompt | detections at any confidence | at the caching threshold (0.25) |", "|---|---|---|"]
lines += [f"| {p} | {hits[p]} | {hits25[p]} |" for p in prompts]
out = ROOT / "results" / "egoblind" / "openvocab_prompt_probe.md"
out.write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print("->", out)
