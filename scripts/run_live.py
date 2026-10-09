"""Live camera demo: YOLO + (optional) OCR + metric depth -> tracker with dynamics -> scheduler -> announcement gate
-> speech, with the activity detector choosing the mode.

The default scheduler is the learned one `pcacs_learned:rl` (models/scheduler/scheduler_rl.pt); if it cannot be loaded the
demo says so (printed and spoken) and runs the linear P-CACS rule instead (`--strict` exits instead). Around it:
  * the announcement gate (seeforme/live/gate.py) does not repeat an object it already announced in the same direction
    unless it comes materially closer, moves to another side, becomes critical, or 20-60 s pass; it waits until a track
    is confirmed; objects that stay still from the start are background (they speak only if they start approaching);
  * the activity detector (seeforme/live/activity.py) picks the mode from camera motion and visible text:
    walking or turning -> navigate, standing still -> scene (only arrivals, approaching objects, critical events),
    still with text in view -> read (object cues muted unless critical), a FIND goal -> find;
  * --setup auto (default) treats a camera that is still after the first seconds as a desk webcam: the person sitting in
    front of it (you) is not announced unless they approach. --setup desk forces this, --setup handheld disables it.

    python scripts/run_live.py --speak                                # learned scheduler, auto mode, macOS `say`
    python scripts/run_live.py --speak --setup desk                   # laptop webcam on a desk
    python scripts/run_live.py --speak --mode navigate                # pin a mode (auto, navigate, scene, read, find)
    python scripts/run_live.py --speak --goal "door"                  # FIND a door
    python scripts/run_live.py --speak --no-gate                      # the bare scheduler (to compare)
    python scripts/run_live.py --policy pcacs --speak                 # linear P-CACS rule
    python scripts/run_live.py --speak --elevenlabs                   # ElevenLabs voice (key from .env; default speaker if missing)
    python scripts/run_live.py --video clip.mp4 --fast --headless --frames 300   # replay a file, no window
    python scripts/run_live.py --list-cameras
Keys (window focused): h help (spoken) | w what is around me | r read the text in view | m mode | space or s sound on/off
                       f FIND goal | v verbosity | g gate on/off | d depth on/off | 1-5 scheduler | q quit
A session log (every cue, mode change and key, no images) goes to results/live_sessions/ (git-ignored; --no-log to skip).
"""
import argparse, math, time
from pathlib import Path
from collections import Counter
import cv2
from _common import MODELS, load_cfg
from seeforme.feedback import Speaker
from seeforme.perception import YoloDetector, EasyOcrReader, NullOcr
from seeforme.live.controller import LiveController, SETUPS
from seeforme.live.activity import MODES
from seeforme.live.summary import SessionLog
from seeforme.pipeline import LivePipeline, make_live_policy
from seeforme.ui import draw_overlay_pcacs

HELP = ("Keys. H, help. W, what is around me. R, read the text in view. M, change mode. Space or S, sound on or off. "
        "F, find something. V, how much I say. G, repeat filter on or off. Q, quit.")


def resolve_model(name: str) -> str:
    """An absolute path for a local weights file (repo root first), so Ultralytics never downloads one because the
    demo was started from another directory. Unknown names are passed through unchanged."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for cand in (Path(name), root / name):
        if cand.exists():
            return str(cand.resolve())
    return name


def recommended_detector() -> str:
    """The live detector chosen by scripts/eval_live_detector.py (results/live/detector_choice.json) if its weights are
    present, else yolo11n.pt."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    try:
        m = json.loads((root / "results" / "live" / "detector_choice.json").read_text())["recommendation"]["model"]
        if (root / m).exists():
            return str(root / m)
    except Exception:
        pass
    return resolve_model("yolo11n.pt")


def list_cameras(n: int = 6) -> None:
    for i in range(n):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        if ok:
            print(f"camera {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()

ap = argparse.ArgumentParser(description="SeeForMe live demo: speaks what matters, when it matters.",
                             formatter_class=argparse.RawDescriptionHelpFormatter, epilog=HELP)
ap.add_argument("--camera", type=int, default=0, help="camera index (see --list-cameras)")
ap.add_argument("--video", default=None, help="replay this video file instead of the camera (same pipeline)")
ap.add_argument("--fast", action="store_true", help="with --video: process every frame as fast as possible "
                "(timestamps still come from the file) instead of pacing in real time and dropping late frames like a camera")
ap.add_argument("--headless", action="store_true", help="no window and no keys; stop at the end of the video, after --frames, or on Ctrl-C")
ap.add_argument("--frames", type=int, default=0, help="stop after this many processed frames (0 = no limit)")
ap.add_argument("--policy", default="pcacs_learned:rl", help="scheduler: pcacs_learned:rl (default, learned), pcacs (linear rule), cacs, confidence_cooldown, confidence_threshold")
ap.add_argument("--strict", action="store_true", help="exit if the policy cannot be loaded instead of falling back to P-CACS")
ap.add_argument("--speak", action="store_true", help="speak the cues (macOS say; prints them otherwise)")
ap.add_argument("--elevenlabs", action="store_true", help="ElevenLabs voice (needs ELEVENLABS_API_KEY; falls back to the default voice)")
ap.add_argument("--no-ocr", action="store_true", help="no sign reading (faster)")
ap.add_argument("--no-depth", action="store_true", help="no metric depth: distances from box size only (faster, less accurate)")
ap.add_argument("--outdoor", action="store_true", help="use the outdoor metric depth model")
ap.add_argument("--model", default=None, help="detector weights (default: the one chosen in results/live/detector_choice.json, yolo11s.pt)")
ap.add_argument("--units", default="m", choices=["m", "ft"], help="meters or feet in spoken distances")
ap.add_argument("--goal", default="", help='FIND goal, e.g. "exit" or "room 204"')
ap.add_argument("--mode", default="auto", choices=["auto", *MODES], help="auto: from the activity detector; or pin one mode")
ap.add_argument("--setup", default="auto", choices=list(SETUPS),
                help="desk: a still webcam with you in front of it (you are not announced); auto: detected in the first seconds")
ap.add_argument("--no-gate", action="store_true", help="speak whatever the scheduler says (no repeat filter, no background)")
ap.add_argument("--depth-every", type=int, default=2, help="run metric depth every N frames (distances are tracked in between)")
ap.add_argument("--no-log", action="store_true", help="do not write a session log to results/live_sessions/")
ap.add_argument("--list-cameras", action="store_true", help="list the cameras OpenCV can open and exit")
args = ap.parse_args()
if args.list_cameras:
    list_cameras(); raise SystemExit(0)
if args.fast and not args.video:
    ap.error("--fast needs --video (the camera always runs in real time)")

cfg = load_cfg()
cfg.perception.yolo_model = resolve_model(args.model) if args.model else recommended_detector()
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


def policy_label(p) -> str:
    return f"{p.name} (fallback)" if getattr(p, "fallback_from", None) else p.name
goal_tags = [g.strip() for g in args.goal.split(",") if g.strip()]
try:
    policy = make_live_policy(args.policy, cfg, str(MODELS), goal_tags=goal_tags, fallback=not args.strict)
except Exception as e:
    raise SystemExit(f"could not load policy {args.policy}: {type(e).__name__}: {e}")
if getattr(policy, "fallback_from", None):
    print(f"WARNING: could not load the learned scheduler {policy.fallback_from} ({policy.fallback_reason}); "
          "using the linear P-CACS rule instead", flush=True)
    speaker.say("Warning: the learned scheduler is not available. Using the basic scheduler.")
detector = YoloDetector(cfg.perception)
ocr = NullOcr() if args.no_ocr else EasyOcrReader(cfg.perception)
log = None if args.no_log else SessionLog()
ctl = LiveController(policy, mode=args.mode, setup=args.setup, gate=not args.no_gate, log=log, goal_tags=goal_tags, units=args.units)
pipe = LivePipeline(detector, ocr, ctl.policy, cfg, speaker=speaker, depth=depth, depth_every_n_frames=max(1, args.depth_every))
print(f"SeeForMe live | scheduler {policy_label(policy)} | detector {Path(cfg.perception.yolo_model).name} | "
      f"mode {args.mode} | setup {args.setup} | repeat filter {'off' if args.no_gate else 'on'} | depth "
      f"{'off' if depth is None else f'every {max(1, args.depth_every)} frames'} | log {log.path if log else 'off'}", flush=True)
if not args.headless:
    print(HELP, flush=True)
    speaker.say("SeeForMe ready. Press H for help.")


def camera_frames(cap, t_start):
    """(t, frame) from the camera; t is wall-clock time since the start."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield time.perf_counter() - t_start, frame


def video_frames(cap, fps, t_start, realtime, counts):
    """(t, frame) from a file; t is the frame's time in the file. Real time drops the frames the pipeline is too late
    for and waits when it is early, as a camera would; otherwise every frame is yielded as fast as possible."""
    i = 0
    while True:
        if realtime:
            due = int((time.perf_counter() - t_start) * fps)
            while i < due:
                if not cap.grab():
                    return
                i += 1; counts["dropped"] += 1
        ok, frame = cap.read()
        if not ok:
            return
        # the frame's own timestamp (variable-frame-rate phone video does not drift), i / fps if the file has none
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = pos / 1000.0 if pos and math.isfinite(pos) and pos > 0 else i / fps
        if realtime:
            time.sleep(max(0.0, t - (time.perf_counter() - t_start)))
        yield t, frame
        i += 1


replay = {"dropped": 0}
if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
else:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("could not open camera")
t_start = time.perf_counter()
frames = video_frames(cap, fps, t_start, not args.fast, replay) if args.video else camera_frames(cap, t_start)
last_cue = None; last_cue_t = -1e9
verbosities = ["quiet", "balanced", "detailed"]
n_frames = 0; t = 0.0
decision_ms = []; actions = Counter()
try:
    for t, frame in frames:
        tick = pipe._next_decision      # the pipeline moves it forward on every decision step
        res = pipe.process(frame, t)
        ctl.on_frame(frame, t, res.tracks)
        n_frames += 1
        if pipe._next_decision != tick:
            decision_ms.append(pipe.latency.policy[-1])
            actions[res.cue.action if res.cue is not None else "silence"] += 1
        if res.cue is not None:
            last_cue, last_cue_t = res.cue, t
        if not args.headless:
            shown = last_cue if t - last_cue_t < 2.5 else None
            st = ctl.status()
            stats = {"policy": policy_label(pipe.policy), "mode": st["mode"], "activity": st["activity"], "setup": st["setup"],
                     "gate": st["gate"][:40], "det": f"{res.latency_ms['detect']:.0f}ms", "depth": f"{res.latency_ms.get('depth', 0):.0f}ms",
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
                    ctl.set_policy(make_live_policy(policy_names[k - ord("1")], cfg, str(MODELS), goal_tags=goal_tags))
                    pipe.policy = ctl.policy
                    pipe.tracker.reset()
                except Exception as e:      # e.g. no trained scheduler: keep the current policy
                    print(e)
            if k in (ord("s"), ord(" ")):
                speaker.enabled = not speaker.enabled
                print("sound:", "on" if speaker.enabled else "off", flush=True)
                if speaker.enabled:
                    speaker.say("Sound on.")
            if k == ord("h"):
                speaker.interrupt(HELP) if hasattr(speaker, "interrupt") else speaker.say(HELP)
            if k == ord("w"):
                msg = ctl.summary(res.tracks)
                speaker.interrupt(msg) if hasattr(speaker, "interrupt") else speaker.say(msg)
                if log: log.write("key", t, key="what is around me", said=msg)
            if k == ord("r"):
                texts = [tr.text for tr in res.tracks if tr.kind == "text" and tr.text and tr.misses == 0]
                msg = ("The text says: " + ". ".join(dict.fromkeys(texts))) if texts else (
                    "I don't see readable text. Hold the sign steady, closer." if not args.no_ocr else "Reading is off; start without --no-ocr.")
                speaker.interrupt(msg) if hasattr(speaker, "interrupt") else speaker.say(msg)
                if log: log.write("key", t, key="read", said=msg)
            if k == ord("m"):
                msg = ctl.cycle_mode(); print(msg, flush=True)
                speaker.interrupt(msg) if hasattr(speaker, "interrupt") else speaker.say(msg)
            if k == ord("g"):
                ctl.use_gate = not ctl.use_gate
                ctl.set_policy(ctl.gate.inner if ctl.gate is not None else pipe.policy._inner)
                pipe.policy = ctl.policy
                msg = f"Repeat filter {'on' if ctl.use_gate else 'off'}."; print(msg, flush=True); speaker.say(msg)
            if k == ord("v") and hasattr(pipe.policy.cfg, "verbosity"):
                v = verbosities[(verbosities.index(pipe.policy.cfg.verbosity) + 1) % 3]
                pipe.policy.cfg.verbosity = v; print("verbosity:", v)
            if k == ord("d"):
                pipe.use_depth = not pipe.use_depth; print("depth:", pipe.use_depth)
            if k == ord("f"):
                goal = input("FIND goal (comma-separated tags, empty to clear): ").strip()
                goal_tags = [g.strip() for g in goal.split(",") if g.strip()]
                ctl.set_goal(goal_tags)
                msg = f"Looking for {', '.join(goal_tags)}." if goal_tags else "Find cleared."
                speaker.say(msg)
                if log: log.write("key", t, key="find", goal=goal_tags)
        if args.frames and n_frames >= args.frames:
            break
except KeyboardInterrupt:
    pass
wall = time.perf_counter() - t_start
cap.release()
if not args.headless:
    cv2.destroyAllWindows()

print("latency summary:", pipe.latency.summary())
for line in ctl.report():
    print(line)
if log is not None:
    log.close(); print(f"session log: {log.path}")
print(f"{len(pipe.cue_log)} cues ({sum(1 for c in pipe.cue_log if c.action == 'haptic')} haptic)")
spoken = [c for c in pipe.cue_log if c.action in ("speak", "interrupt")]
n_dec = len(decision_ms)
src = f"video {args.video} ({'fast' if args.fast else 'real time'}, {replay['dropped']} frames dropped)" if args.video else f"camera {args.camera}"
ckpt = getattr(pipe.policy, "checkpoint", None)
print(f"policy: {policy_label(pipe.policy)}" + (f" [mode {pipe.policy.mode}, {ckpt}]" if ckpt else "")
      + (f" [requested {pipe.policy.fallback_from}: {pipe.policy.fallback_reason}]" if getattr(pipe.policy, "fallback_from", None) else ""))
print(f"source: {src}")
stream_s = t + (1.0 / fps if args.video and n_frames else 0.0)     # the last frame lasts one frame interval
print(f"frames: {n_frames} in {wall:.1f}s wall ({n_frames / max(wall, 1e-9):.1f} fps), {stream_s:.1f}s of stream")
if n_dec:
    dm = sorted(decision_ms)
    print(f"decisions: {n_dec} ({n_dec / max(wall, 1e-9):.2f}/s wall, {n_dec / max(stream_s, 1e-9):.2f}/s stream) | "
          f"decision step latency mean {sum(dm) / n_dec:.2f} ms, p95 {dm[min(n_dec - 1, math.ceil(0.95 * n_dec) - 1)]:.2f} ms")
print("actions: " + ", ".join(f"{a} {actions.get(a, 0)}" for a in ("silence", "speak", "interrupt", "haptic")))
print(f"cues spoken: {len(spoken)}")
for c in pipe.cue_log:                       # every action, haptics included
    print(f"  {c.t:6.2f}s  {c.action:<9} {c.message or '(vibration)'}")
if args.strict and (n_frames == 0 or n_dec == 0):
    raise SystemExit(f"--strict: nothing was tested ({n_frames} frames, {n_dec} decisions)")
