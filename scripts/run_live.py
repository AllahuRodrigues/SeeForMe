"""Live camera demo: YOLO + (optional) OCR + metric depth -> tracker with dynamics -> scheduler -> announcement gate
-> speech, with the activity detector choosing the mode, driven by voice commands.

The default scheduler is the learned one `pcacs_learned:rl` (models/scheduler/scheduler_rl.pt); if it cannot be loaded the
demo says so (printed and spoken) and runs the linear P-CACS rule instead (`--strict` exits instead). Around it:
  * the announcement gate (seeforme/live/gate.py) does not repeat an object it already announced in the same direction
    unless it comes materially closer, moves to another side, becomes critical, or 20-60 s pass; it waits until a track
    is confirmed; objects that stay still from the start are background (they speak only if they start approaching);
  * the activity detector (seeforme/live/activity.py) picks the mode from camera motion and visible text:
    walking or turning -> navigate, standing still -> scene (only arrivals, approaching objects, critical events),
    still with text in view -> read (object cues muted unless critical), a FIND goal -> find;
  * --setup auto (default) treats a camera that is still after the first seconds as a desk webcam: the person sitting in
    front of it (you) is not announced unless they approach. --setup desk forces this, --setup handheld disables it;
  * the cane layer (seeforme/live/cane.py) names what the detector cannot (walls, poles, low and head-height obstacles)
    from the depth map, quiet about what your own aid covers: --aid cane (default: nothing low within the cane's reach,
    body-level obstacles 1.2-2.5 m ahead, head height always), guide-dog (head height and the stop only), none (all).

Voice commands (seeforme/live/voice.py, commands.py) replace the buttons: say "SeeForMe, find a door", "SeeForMe,
what's around me", "SeeForMe, pause" / "start", "SeeForMe, help". "SeeForMe" alone answers with a short sound and
then takes one command without the wake word for 5 s. --voice auto (default) uses the offline Vosk recogniser when it
is installed (`make voice-setup`); commands typed in this terminal always work (wake word optional), and --commands
"2:find a chair;4:describe" fires commands at stream time in headless replays; lines piped into stdin are commands
too (`printf "what's around me\nquit\n" | ... --headless`), and a typed single letter is the window key (q, h, w, r, p,
s); a headless camera run whose only commands come from such a pipe stops at its end (nothing else could stop it).
Pause stops routine announcements but never hazard warnings; mute silences everything. Microphone results are
ignored while the demo itself speaks. A reply holds the audio channel, but a hazard warning cuts it (the controller
shows the scheduler a free channel while a reply plays and something in view could need a warning), and a reply
waits while a hazard warning plays.

    python scripts/run_live.py --speak                                # learned scheduler, auto mode, voice auto, macOS `say`
    python scripts/run_live.py --speak --setup desk                   # laptop webcam on a desk
    python scripts/run_live.py --speak --mode navigate                # pin a mode (auto, navigate, scene, read, find)
    python scripts/run_live.py --speak --goal "door"                  # FIND a door
    python scripts/run_live.py --speak --aid guide-dog                # travelling with a guide dog
    python scripts/run_live.py --speak --no-gate                      # the bare scheduler (to compare)
    python scripts/run_live.py --policy pcacs --speak                 # linear P-CACS rule
    python scripts/run_live.py --speak --elevenlabs                   # ElevenLabs voice (key from .env; default speaker if missing)
    python scripts/run_live.py --video clip.mp4 --fast --headless --frames 300 --commands "2:find a chair;5:pause"
    python scripts/run_live.py --list-cameras
Keys (window focused): h help (spoken) | w what is around me | r read the text in view | m mode | space or s sound on/off
                       p pause / resume | l listen for one command | f FIND goal (typed) | v verbosity | g gate on/off
                       d depth on/off | i debug overlay | 1-5 scheduler | q quit
The window shows the camera image exactly as captured (never mirrored: left in the picture is left in speech).
A session log (every cue, command, mode change and key, no images) goes to results/live_sessions/ (git-ignored; --no-log).
"""
import argparse, inspect, math, os, stat, sys, threading, time
from pathlib import Path
from collections import Counter
import cv2
from _common import MODELS, load_cfg
from seeforme.feedback import Speaker
from seeforme.perception import YoloDetector, EasyOcrReader, NullOcr
from seeforme.live.controller import AIDS, LiveController, SETUPS, reply_cue
from seeforme.live.activity import MODES
from seeforme.live.commands import LABELS, CommandRouter, goal_tags as parse_goal, is_safety_cue
from seeforme.live.summary import SessionLog
from seeforme.live.voice import (DEFAULT_RECOGNISER, RECOGNISERS, Command, TypedBackend, VoiceListener, VoskBackend,
                                 input_end_stops_run, parse_script, vosk_status)
from seeforme.pipeline import LivePipeline, make_live_policy
from seeforme.ui import draw_overlay_pcacs

HELP = ("Keys. H, help. W, what is around me. R, read the text in view. M, change mode. Space or S, sound on or off. "
        "P, pause or resume. L, listen for one command. F, find something. V, how much I say. G, repeat filter. Q, quit.")
VOICE_HINT = ("Voice commands: run make voice-setup (installs vosk and a 40 MB offline model). "
              "You can type commands here meanwhile.")
VOICE_LABEL = {"listening": "Listening...", "wake": "Say 'SeeForMe' + command", "typed": "Typed commands only", "off": "Keys only"}


def resolve_model(name: str) -> str:
    """An absolute path for a local weights file (repo root first), so Ultralytics never downloads one because the
    demo was started from another directory. Unknown names are passed through unchanged."""
    root = Path(__file__).resolve().parents[1]
    for cand in (Path(name), root / name):
        if cand.exists():
            return str(cand.resolve())
    return name


def recommended_detector() -> str:
    """The live detector chosen by scripts/eval_live_detector.py (results/live/detector_choice.json) if its weights are
    present, else yolo11n.pt."""
    import json
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
ap.add_argument("--voice", default="auto", choices=["auto", "on", "off"],
                help="spoken commands: auto = the offline Vosk recogniser when it is installed (make voice-setup) and the run "
                     "is not headless; on = require it; off = no microphone. Typed commands always work in a terminal")
ap.add_argument("--commands", default="", help='scripted commands at stream time (s), e.g. "2:find a chair;4:describe;6:pause;9:start"')
ap.add_argument("--mic", default=None, help="microphone for --voice (sounddevice index or name; default: the system input)")
ap.add_argument("--vosk-model", default=None, help="Vosk model directory (default: $SEEFORME_VOSK_MODEL or models/vosk/vosk-model-small-en-us-0.15)")
ap.add_argument("--recogniser", default=DEFAULT_RECOGNISER, choices=list(RECOGNISERS),
                help=f"how Vosk decodes speech (default {DEFAULT_RECOGNISER}): grammar = the command vocabulary only, open = any "
                     "words, hybrid = the grammar, with the open vocabulary for words it does not know, snap = the open "
                     "vocabulary, near-misses snapped to the command words (results/live/voice_eval.md)")
ap.add_argument("--no-ocr", action="store_true", help="no sign reading (faster)")
ap.add_argument("--no-depth", action="store_true", help="no metric depth: distances from box size only (faster, less accurate)")
ap.add_argument("--no-cane", action="store_true", help="turn off the cane / guide-dog layer of the live controller")
ap.add_argument("--aid", default="cane", choices=list(AIDS),
                help="what you travel with: cane (default: no cue for low obstacles the cane can touch, body-level ones 1.2-2.5 m "
                     "ahead, head height always), guide-dog (head height and the stop only), none (every cane-layer cue)")
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
try:
    script = parse_script(args.commands)
except ValueError as e:
    ap.error(str(e))
mic = int(args.mic) if args.mic is not None and str(args.mic).isdigit() else args.mic

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
    speaker = Speaker(enabled=args.speak, log_to_stdout=False)     # this script prints one aligned line per message
depth = None
if not args.no_depth:
    from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR
    print("loading depth model ...")
    depth = DepthAnythingEstimator(OUTDOOR if args.outdoor else INDOOR)
policy_names = ["confidence_threshold", "confidence_cooldown", "cacs", "pcacs", "pcacs_learned:rl"]


def policy_label(p) -> str:
    return f"{p.name} (fallback)" if getattr(p, "fallback_from", None) else p.name


goal_tags = list(dict.fromkeys(t for part in args.goal.split(",") for t in parse_goal(part)))
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
ctl_kw = dict(mode=args.mode, setup=args.setup, gate=not args.no_gate, log=log, goal_tags=goal_tags, units=args.units)
if "cane" in inspect.signature(LiveController).parameters:
    ctl_kw["cane"] = not args.no_cane
    ctl_kw["aid"] = args.aid
elif args.no_cane:
    print("note: this LiveController has no cane layer; --no-cane changes nothing", flush=True)
ctl = LiveController(policy, **ctl_kw)
on_frame_takes_depth = "depth" in inspect.signature(ctl.on_frame).parameters
labels = set(getattr(detector, "names", {}).values()) or set(LABELS)
router = CommandRouter(ctl, speaker, ocr=not args.no_ocr, labels=labels, log=log)
pipe = LivePipeline(detector, ocr, router.wrap(ctl.policy), cfg, speaker=speaker, depth=depth,
                    depth_every_n_frames=max(1, args.depth_every))


def set_pipeline_policy() -> None:
    """The controller's policy object changed (keys 1-5, g): the pipeline calls it through the pause filter."""
    pipe.policy = router.wrap(ctl.policy)


# ---- commands: microphone, terminal, script ----------------------------------------------------------------------
listener = VoiceListener(speaker)
if args.voice == "on" or (args.voice == "auto" and not args.headless):
    ok, why = vosk_status(args.vosk_model)
    if ok:
        listener.add(VoskBackend(args.vosk_model, device=mic, recogniser=args.recogniser))
    elif args.voice == "on":
        raise SystemExit(f"--voice on: {why}. Run make voice-setup (installs vosk and a 40 MB offline model).")
    else:
        print(VOICE_HINT, flush=True)


def stdin_has_commands() -> bool:
    """A terminal, or a pipe / file of commands (`printf "describe\nquit\n" | run_live.py --headless ...`); not
    /dev/null or a closed stdin (a test harness, a service)."""
    try:
        if sys.stdin is None or sys.stdin.closed:
            return False
        if sys.stdin.isatty():
            return True
        mode = os.fstat(sys.stdin.fileno()).st_mode
        return stat.S_ISFIFO(mode) or stat.S_ISREG(mode)
    except (OSError, ValueError):
        return False


typed = listener.add(TypedBackend()) if stdin_has_commands() else None
if script:
    listener.set_script(script)
listener.start()
router.voice = listener.voice_running
inputs = [b.name for b in listener.backends] + (["script"] if script else [])
stdin_tty = bool(typed is not None and sys.stdin.isatty())
eof_stops = typed is not None and input_end_stops_run(args.headless, bool(args.video), args.frames, stdin_tty, inputs)

print(f"SeeForMe live | scheduler {policy_label(policy)} | detector {Path(cfg.perception.yolo_model).name} | "
      f"mode {args.mode} | setup {args.setup} | aid {'off' if args.no_cane else args.aid} | "
      f"repeat filter {'off' if args.no_gate else 'on'} | depth "
      f"{'off' if depth is None else f'every {max(1, args.depth_every)} frames'} | commands "
      f"{'+'.join(inputs) or 'keys only'} | log {log.path if log else 'off'}", flush=True)


# ---- output -------------------------------------------------------------------------------------------------------
ui = {"banner": "", "banner_t": -1e9, "debug": False, "last_cue": None, "last_cue_track": None, "last_cue_t": -1e9,
      "pending": [], "after": False}


def line(t: float, kind: str, text: str) -> None:
    """One aligned terminal line per spoken cue, command and reply."""
    print(f"{t:7.1f}s  {kind:<9} {text}", flush=True)


def _occupy_channel(text: str, t: float) -> None:
    """A reply holds the audio channel (controller.reply_cue: no severity, no utility, a track id the controller knows):
    while it plays, a hazard warning still cuts it (seeforme/live/controller.py), a routine cue waits."""
    pipe.channel.speak(reply_cue(text, t), interrupt=True)


def speak_reply(text: str, t: float) -> None:
    """Speak a reply now, cutting a routine cue; a hazard warning that is still playing (an in-path, approaching,
    vehicle, FIND-goal or cane cue: is_safety_cue with its track and the goal) is never cut: the reply waits."""
    ui["banner"], ui["banner_t"] = text, t
    line(t, "reply", text)
    cue = ui["last_cue"]
    if (speaker.busy() and cue is not None and t - ui["last_cue_t"] < 6.0
            and is_safety_cue(cue, ui["last_cue_track"], ctl.goal_tags)):
        ui["pending"].append(text)
        return
    _occupy_channel(text, t)
    speaker.interrupt(text) if hasattr(speaker, "interrupt") else speaker.say(text)


def flush_pending(t: float) -> None:
    if ui["pending"] and not speaker.busy():
        text = ui["pending"].pop(0)
        _occupy_channel(text, t)
        speaker.say(text)
    if ui["after"] and not ui["pending"]:
        ui["after"] = False
        router.after_reply()


def run_command(cmd: Command, tracks, t: float) -> None:
    line(t, "heard", cmd.describe() + (f'  ({cmd.source}: "{cmd.text}")' if cmd.text else f"  ({cmd.source})"))
    try:
        reply = router.handle(cmd, tracks, t)
    except Exception as e:          # a live demo keeps running; the error is printed for the developer
        print(f"command {cmd.describe()!r} failed: {type(e).__name__}: {e}", flush=True)
        reply = "Sorry, that did not work."
    if reply:
        speak_reply(reply, t)
    ui["after"] = True               # a mute waits until its "Sound off." is on its way
    flush_pending(t)


def voice_label() -> str:
    return "Paused" if router.paused else VOICE_LABEL[listener.state()]


def wrap_text(text: str, width: int, scale: float, thick: int):
    rows, cur = [], ""
    for w in text.split():
        cand = f"{cur} {w}".strip()
        if cur and cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > width:
            rows.append(cur); cur = w
        else:
            cur = cand
    return rows + ([cur] if cur else [])


def draw_view(img, tracks, highlight, banner: str, banner_age: float, status: str):
    """The camera image as captured (never mirrored), thin boxes for confirmed objects, one small status line on
    top and the last spoken sentence large at the bottom."""
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for tr in tracks:
        if tr.misses or not getattr(tr, "confirmed", True):
            continue
        x1, y1, x2, y2 = int(tr.bbox[0] * w), int(tr.bbox[1] * h), int(tr.bbox[2] * w), int(tr.bbox[3] * h)
        hl = tr.track_id == highlight
        col = (0, 210, 255) if hl else (215, 215, 215)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if hl else 1)
        name = tr.text if tr.kind == "text" and tr.text else tr.label
        d = f" {tr.distance_m:.1f} m" if tr.distance_m is not None and math.isfinite(tr.distance_m) else ""
        cv2.putText(out, name + d, (x1 + 3, max(40, y1 - 5)), font, 0.5, col, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, status, (8, 18), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    if banner:
        scale, thick, step = 0.9, 2, 34
        rows = wrap_text(banner, w - 24, scale, thick)[-3:]
        top = h - 14 - step * len(rows)
        shade = out.copy()
        cv2.rectangle(shade, (0, top), (w, h), (0, 0, 0), -1)
        out = cv2.addWeighted(shade, 0.72, out, 0.28, 0)
        col = (255, 255, 255) if banner_age < 6.0 else (150, 150, 150)
        for i, row in enumerate(rows):
            cv2.putText(out, row, (12, top + step * (i + 1) - 6), font, scale, col, thick, cv2.LINE_AA)
    return out


probe = {"thread": None, "found": None}


def _probe_cameras(cur: int) -> None:
    """In a thread: opening a camera index that does not exist can take seconds on macOS, and the frame loop must keep
    warning meanwhile."""
    for idx in dict.fromkeys(i for i in (cur + 1, cur + 2, 0) if i != cur):
        c = cv2.VideoCapture(idx)
        if c.isOpened() and c.read()[0]:
            probe["found"] = (idx, c)
            return
        c.release()
    probe["found"] = (None, None)


def flip_camera() -> str:
    """"Flip camera": look for another camera in the background (finish_flip swaps it in when found)."""
    if args.video:
        return "This is a video file, so there is no camera to switch."
    if probe["thread"] is not None:
        return "Still looking for another camera."
    probe["found"] = None
    probe["thread"] = threading.Thread(target=_probe_cameras, args=(src["index"],), name="seeforme-flip", daemon=True)
    probe["thread"].start()
    return "Looking for another camera."


def finish_flip(t: float) -> None:
    """Every frame: a finished camera search swaps the camera in (tracker, gate, activity and cane start over)."""
    th = probe["thread"]
    if th is None or th.is_alive():
        return
    probe["thread"] = None
    idx, c = probe["found"] or (None, None)
    if c is None:
        speak_reply("There is no other camera.", t)
        return
    src["cap"].release()
    src["cap"], src["index"] = c, idx
    pipe.tracker.reset()
    ctl.camera_changed()
    speak_reply(f"Camera {idx}.", t)


router.flip = flip_camera
if not args.headless:
    print(HELP, flush=True)
    if typed is not None:
        print("Type a command and press return (for example: find door, what's around me, pause, help; q quits).", flush=True)
    speak_reply("SeeForMe ready. Say 'SeeForMe, help' for commands." if listener.voice_running
                else "SeeForMe ready. Press H for help.", 0.0)


# ---- frames -------------------------------------------------------------------------------------------------------
def camera_frames(src, t_start):
    """(t, frame) from the camera; t is wall-clock time since the start. src["cap"] may be swapped (flip)."""
    while True:
        ok, frame = src["cap"].read()
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
fps = 30.0
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
src = {"cap": cap, "index": args.camera}
t_start = time.perf_counter()
frames = video_frames(cap, fps, t_start, not args.fast, replay) if args.video else camera_frames(src, t_start)
verbosities = ["quiet", "balanced", "detailed"]
n_frames = 0; t = 0.0
decision_ms = []; actions = Counter()
try:
    for t, frame in frames:
        tick = pipe._next_decision      # the pipeline moves it forward on every decision step
        res = pipe.process(frame, t)
        if on_frame_takes_depth:
            ctl.on_frame(frame, t, res.tracks, depth=getattr(pipe, "depth_map", None))
        else:
            ctl.on_frame(frame, t, res.tracks)
        n_frames += 1
        if pipe._next_decision != tick:
            decision_ms.append(pipe.latency.policy[-1])
            actions[res.cue.action if res.cue is not None else "silence"] += 1
        for th, held in router.take_held():           # routine cues the pause kept quiet (hazards are never held)
            line(th, "paused", f"(not said) {held.message}")
        if res.cue is not None:
            ui["last_cue"], ui["last_cue_t"] = res.cue, t
            ui["last_cue_track"] = next((tr for tr in res.tracks if tr.track_id == res.cue.track_id), None)
            router.note_spoken(res.cue)
            line(t, res.cue.action, res.cue.message or "(vibration)")
            if res.cue.message:
                ui["banner"], ui["banner_t"] = res.cue.message, t
        for msg in listener.notices():                 # e.g. the microphone could not be opened
            router.voice = listener.voice_running
            speak_reply(msg, t)
        input_ended = eof_stops and not typed.running   # read before poll: every line before the end is queued by now
        for cmd in listener.poll(t):
            run_command(cmd, res.tracks, t)
            if router.quit_requested:
                break
        if router.quit_requested:
            break
        if input_ended:
            line(t, "stdin", "end of the piped commands: stopping (a headless camera run has nothing else to stop it)")
            break
        finish_flip(t)
        flush_pending(t)
        if not args.headless:
            scale = min(cfg.perception.frame_width / frame.shape[1], 720 / frame.shape[0])     # portrait clips fit a laptop
            small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            st = ctl.status()
            if ui["debug"]:
                stats = {"policy": policy_label(pipe.policy), "mode": st["mode"], "activity": st["activity"], "setup": st["setup"],
                         "gate": st["gate"][:40], "det": f"{res.latency_ms['detect']:.0f}ms", "depth": f"{res.latency_ms.get('depth', 0):.0f}ms",
                         "e2e": f"{res.latency_ms['end_to_end']:.0f}ms", "tracks": len(res.tracks), "cues": len(pipe.cue_log),
                         "verb": getattr(pipe.policy.cfg, "verbosity", "-"), "goal": ",".join(ctl.goal_tags) or "-",
                         "speech": speaker.backend if args.speak else "off"}
                shown = ui["last_cue"] if t - ui["last_cue_t"] < 2.5 else None
                view = draw_overlay_pcacs(small, res.tracks, pipe.policy.last_observations(), shown, stats, pipe.policy)
            else:
                status = f"{st['mode']} | {st['activity']} | {st['setup']} | {voice_label()}"
                if ctl.goal_tags:
                    status += f" | find: {ctl.goal_tags[0]}"
                if args.speak and not speaker.enabled:
                    status += " | sound off"
                hl = ui["last_cue"].track_id if ui["last_cue"] is not None and t - ui["last_cue_t"] < 2.5 else None
                view = draw_view(small, res.tracks, hl, ui["banner"], t - ui["banner_t"], status)
            cv2.imshow("SeeForMe", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                try:
                    ctl.set_policy(make_live_policy(policy_names[k - ord("1")], cfg, str(MODELS), goal_tags=ctl.goal_tags))
                    set_pipeline_policy()
                    pipe.tracker.reset()
                except Exception as e:      # e.g. no trained scheduler: keep the current policy
                    print(e)
            if k in (ord("s"), ord(" ")):
                run_command(Command("mute" if speaker.enabled else "unmute", source="key"), res.tracks, t)
            if k == ord("h"):
                speak_reply(HELP + (" Or say SeeForMe, help." if listener.voice_running else ""), t)
            if k == ord("w"):
                run_command(Command("describe", source="key"), res.tracks, t)
            if k == ord("r"):
                run_command(Command("read", source="key"), res.tracks, t)
            if k == ord("p"):
                run_command(Command("start" if router.paused else "pause", source="key"), res.tracks, t)
            if k == ord("l"):
                if listener.voice_running:
                    listener.listen_once()
                    run_command(Command("wake", source="key"), res.tracks, t)
                else:
                    speak_reply("Voice commands are off. Type a command in the terminal.", t)
            if k == ord("m"):
                speak_reply(ctl.cycle_mode(), t)
            if k == ord("g"):
                ctl.use_gate = not ctl.use_gate
                ctl.set_policy(ctl.gate.inner if ctl.gate is not None else ctl.policy._inner)   # the bare scheduler
                set_pipeline_policy()
                speak_reply(f"Repeat filter {'on' if ctl.use_gate else 'off'}.", t)
            if k == ord("v") and hasattr(pipe.policy.cfg, "verbosity"):
                v = verbosities[(verbosities.index(pipe.policy.cfg.verbosity) + 1) % 3]
                pipe.policy.cfg.verbosity = v; print("verbosity:", v, flush=True)
            if k == ord("d"):
                pipe.use_depth = not pipe.use_depth; print("depth:", pipe.use_depth, flush=True)
            if k == ord("i"):
                ui["debug"] = not ui["debug"]
            if k == ord("f"):
                if typed is not None:
                    typed.expect_goal()
                    print("FIND goal: type it in this terminal and press return (an empty line clears it)", flush=True)
                    speak_reply("Type what to find in the terminal.", t)
                else:
                    speak_reply("Say SeeForMe, find, and what to look for.", t)
                if log: log.write("key", t, key="find")
        if args.frames and n_frames >= args.frames:
            break
except KeyboardInterrupt:
    pass
wall = time.perf_counter() - t_start
listener.stop(timeout=2.0)
if probe["thread"] is not None:
    probe["thread"].join(5.0)
    if probe["found"] and probe["found"][1] is not None:
        probe["found"][1].release()
src["cap"].release()
if not args.headless:
    cv2.destroyAllWindows()

print("latency summary:", pipe.latency.summary())
for ln in ctl.report() + router.report():
    print(ln)
if listener.ignored_echo or listener.ignored_no_wake:
    print(f"voice: {listener.ignored_echo} result(s) ignored while the demo spoke, "
          f"{listener.ignored_no_wake} without the wake word")
if listener.script is not None and listener.script.remaining:
    print(f"scripted commands not reached: {listener.script.remaining}")
if log is not None:
    log.close(); print(f"session log: {log.path}")
print(f"{len(pipe.cue_log)} cues ({sum(1 for c in pipe.cue_log if c.action == 'haptic')} haptic)")
spoken = [c for c in pipe.cue_log if c.action in ("speak", "interrupt")]
n_dec = len(decision_ms)
src_label = f"video {args.video} ({'fast' if args.fast else 'real time'}, {replay['dropped']} frames dropped)" if args.video else f"camera {src['index']}"
ckpt = getattr(pipe.policy, "checkpoint", None)
print(f"policy: {policy_label(pipe.policy)}" + (f" [mode {pipe.policy.mode}, {ckpt}]" if ckpt else "")
      + (f" [requested {pipe.policy.fallback_from}: {pipe.policy.fallback_reason}]" if getattr(pipe.policy, "fallback_from", None) else ""))
print(f"source: {src_label}")
stream_s = t + (1.0 / fps if args.video and n_frames else 0.0)     # the last frame lasts one frame interval
print(f"frames: {n_frames} in {wall:.1f}s wall ({n_frames / max(wall, 1e-9):.1f} fps), {stream_s:.1f}s of stream")
if n_dec:
    dm = sorted(decision_ms)
    print(f"decisions: {n_dec} ({n_dec / max(wall, 1e-9):.2f}/s wall, {n_dec / max(stream_s, 1e-9):.2f}/s stream) | "
          f"decision step latency mean {sum(dm) / n_dec:.2f} ms, p95 {dm[min(n_dec - 1, math.ceil(0.95 * n_dec) - 1)]:.2f} ms")
print("actions: " + ", ".join(f"{a} {actions.get(a, 0)}" for a in ("silence", "speak", "interrupt", "haptic")))
print(f"cues spoken: {len(spoken)} (each printed above when it was said)")
if args.strict and (n_frames == 0 or n_dec == 0):
    raise SystemExit(f"--strict: nothing was tested ({n_frames} frames, {n_dec} decisions)")
