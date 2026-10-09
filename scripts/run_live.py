"""Live camera demo with P-CACS: YOLO + (optional) OCR + metric depth -> tracker with dynamics ->
predictive scheduler (speak / interrupt / haptic / silence) -> speech.

    python scripts/run_live.py --policy pcacs --speak                 # macOS `say`
    python scripts/run_live.py --policy pcacs --speak --elevenlabs    # ElevenLabs voice (key from .env; default speaker if missing)
    python scripts/run_live.py --policy pcacs --no-depth              # bbox-height distance guess only
Keys: q quit | 1 threshold  2 cooldown  3 cacs  4 pcacs  5 learned | s speech | v verbosity | f FIND goal | d depth on/off
"""
import argparse, time
import cv2
from _common import MODELS, load_cfg
from seeforme.feedback import Speaker
from seeforme.perception import YoloDetector, EasyOcrReader, NullOcr
from seeforme.pipeline import LivePipeline, make_live_policy
from seeforme.ui import draw_overlay_pcacs

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--policy", default="pcacs")
ap.add_argument("--speak", action="store_true")
ap.add_argument("--elevenlabs", action="store_true")
ap.add_argument("--no-ocr", action="store_true")
ap.add_argument("--no-depth", action="store_true")
ap.add_argument("--outdoor", action="store_true", help="use the outdoor metric depth model")
ap.add_argument("--model", default=None)
ap.add_argument("--units", default="m", choices=["m", "ft"])
ap.add_argument("--goal", default="", help='FIND goal, e.g. "exit" or "room 204"')
args = ap.parse_args()

cfg = load_cfg()
if args.model:
    cfg.perception.yolo_model = args.model
cfg.pcacs.units = args.units
speaker = None
if args.elevenlabs:
    from seeforme.feedback.elevenlabs_tts import ElevenLabsSpeaker
    try:
        speaker = ElevenLabsSpeaker(enabled=args.speak)
    except RuntimeError as e:  # no API key: keep the demo running with the default voice
        print(f"warning: {e}; falling back to the default speaker", flush=True)
if speaker is None:
    speaker = Speaker(enabled=args.speak)
depth = None
if not args.no_depth:
    from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
    print("loading depth model ...")
    depth = DepthAnythingEstimator(OUTDOOR if args.outdoor else INDOOR)
policy_names = ["confidence_threshold", "confidence_cooldown", "cacs", "pcacs", "pcacs_learned:rl"]
goal_tags = [g.strip() for g in args.goal.split(",") if g.strip()]
policy = make_live_policy(args.policy, cfg, str(MODELS), goal_tags=goal_tags)
detector = YoloDetector(cfg.perception)
ocr = NullOcr() if args.no_ocr else EasyOcrReader(cfg.perception)
pipe = LivePipeline(detector, ocr, policy, cfg, speaker=speaker, depth=depth)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("could not open camera")
t_start = time.perf_counter()
last_cue = None; last_cue_t = -1e9
verbosities = ["quiet", "balanced", "detailed"]
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = time.perf_counter() - t_start
    res = pipe.process(frame, t)
    if res.cue is not None:
        last_cue, last_cue_t = res.cue, t
    shown = last_cue if t - last_cue_t < 2.5 else None
    stats = {"policy": pipe.policy.name, "det": f"{res.latency_ms['detect']:.0f}ms", "depth": f"{res.latency_ms.get('depth', 0):.0f}ms",
             "e2e": f"{res.latency_ms['end_to_end']:.0f}ms", "tracks": len(res.tracks), "cues": len(pipe.cue_log),
             "verb": getattr(pipe.policy.cfg, "verbosity", "-"), "goal": ",".join(goal_tags) or "-",
             "speech": speaker.backend if args.speak else "off"}
    small = cv2.resize(frame, (cfg.perception.frame_width, int(frame.shape[0] * cfg.perception.frame_width / frame.shape[1])))
    cv2.imshow("SeeForMe", draw_overlay_pcacs(small, res.tracks, pipe.policy.last_observations(), shown, stats, pipe.policy))
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
        try:
            pipe.policy = make_live_policy(policy_names[k - ord("1")], cfg, str(MODELS), goal_tags=goal_tags)
            pipe.tracker.reset()
        except FileNotFoundError as e:
            print(e)
    if k == ord("s"):
        speaker.enabled = not speaker.enabled
    if k == ord("v") and hasattr(pipe.policy.cfg, "verbosity"):
        v = verbosities[(verbosities.index(pipe.policy.cfg.verbosity) + 1) % 3]
        pipe.policy.cfg.verbosity = v; print("verbosity:", v)
    if k == ord("d"):
        pipe.use_depth = not pipe.use_depth; print("depth:", pipe.use_depth)
    if k == ord("f") and hasattr(pipe.policy, "set_goal"):
        goal = input("FIND goal (comma-separated tags, empty to clear): ").strip()
        goal_tags = [g.strip() for g in goal.split(",") if g.strip()]
        pipe.policy.set_goal(goal_tags)
cap.release(); cv2.destroyAllWindows()
print("latency summary:", pipe.latency.summary())
print(f"{len(pipe.cue_log)} cues ({sum(1 for c in pipe.cue_log if c.action == 'haptic')} haptic)")
