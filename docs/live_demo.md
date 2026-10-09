# The live demo: what it says, when, and why

`make live` runs SeeForMe on the Mac's camera: YOLO detections and metric depth feed a tracker, the learned scheduler
(`pcacs_learned:rl`, a GRU trained with imitation and reinforcement learning on CampusBench) proposes what to say, and
the layer in `seeforme/live/` decides whether it is worth saying now. This page describes that layer, the commands,
and what was measured. It is a demo for development and supervised testing, not a mobility aid: it can miss objects
and cannot tell you that a path is clear.

## Commands

```bash
make live          # webcam, learned scheduler, repeat filter, activity-driven modes, spoken with the macOS voice
make live-desk     # laptop webcam on a desk: you, in front of it, are never announced unless you come closer
make live-raw      # the bare scheduler (no repeat filter, navigate mode), to hear the difference
make live-pcacs    # the linear P-CACS rule instead of the learned scheduler
make live-elevenlabs   # ElevenLabs voice (needs ELEVENLABS_API_KEY; falls back to the macOS voice)
make live-help     # every option
make live-check    # headless replay of a local clip: policy, decisions, cues, gate and mode summary
make eval-live     # re-measure the detector choice and the gate on EgoBlind development replays
```

Useful options (`scripts/run_live.py --help` lists all): `--setup desk|handheld|auto`, `--mode auto|navigate|scene|read|find`,
`--goal "door"`, `--units ft`, `--no-gate`, `--no-depth`, `--depth-every N`, `--camera N` (`--list-cameras`), `--video clip.mp4`,
`--no-log`.

Keys while the window has focus:

| key | what it does |
|---|---|
| `h` | speaks the list of keys |
| `w` | "what is around me": up to three named objects, nearest first, with direction and distance |
| `r` | reads the text in view (needs OCR, the default) |
| `m` | cycles the mode: automatic, navigate, scene, read, find, back to automatic (spoken) |
| `space` or `s` | sound on or off (muting now really silences the macOS voice) |
| `f` | asks for a FIND goal in the terminal |
| `v` | verbosity: quiet, balanced, detailed |
| `g` | repeat filter on or off |
| `d` | metric depth on or off |
| `1`-`5` | scheduler: threshold, cooldown, CACS, P-CACS, learned |
| `q` | quit |

## What decides whether a cue is spoken

The scheduler proposes at most one cue every 0.5 s. Three things can still stop it:

1. **The announcement gate** (`seeforme/live/gate.py`) remembers what it has said about each object and direction
   (left, ahead, right), keyed so that a track that is lost and found again is the same object, while a second object
   of the same label seen at the same time is a different one; signs are told apart by their numbers ("room 204" and
   "room 206" are two signs). It holds a cue when the same object in the same direction was announced recently (20 s
   while walking, 60 s while standing, 8 s for a moving object) and nothing new happened. New means: the object came
   materially closer (by at least 1 m and 25%), is still closing in the path within 3 m (0.35 m or 15% nearer than when
   last said), changed side, or became critical (a measured time to contact under 2 s with the object in the walking
   path; a critical object may be repeated every 2 s, or sooner if it gets worse). A track must be confirmed (seen in
   several frames) before its first cue, so a one-frame flicker is never spoken. Without measured depth (depth off or
   too uncertain), closing is read from the growth of the box sides the frame edge does not cut, and an object whose
   box reaches the bottom of the frame in the walking corridor counts as near.
2. **Background and the user's own body.** When standing still, objects that stay where they are for at least a second
   become background and speak only once they measurably close in (a distance drop of 0.35 m or 15%, or a sustained
   closing speed, or a short measured time to contact); the first cue after that passes. Vehicles and bikes never
   become background, nothing is held on its first sample, a FIND goal is never held as background, and an object in
   the walking path within 3 m is said once before it becomes background. With a desk webcam, a person within 1.2 m in
   front of the camera at start-up is you: you are not announced, even when you lean in, unless you measurably come
   closer. If the scheduler will not offer a held cue again in time once its object qualifies, the gate says it itself.
3. **Read mode** mutes object cues unless they are critical or approaching (whatever their class), so a sign can be
   read without interruptions about furniture.

An acceptance harness ran every safety scenario found in review (a person walking up from 2 m, a person standing
ahead at start-up then approaching, a car that is the first thing seen, a FIND goal already in view, depth-off
obstacles, background vehicles creeping in) through the bare scheduler and seven gated setups: in all 694 runs the
gated demo warns no later than the bare scheduler plus one decision step (0.5 s), while a still or leaning desk user
produced no cue in 217 runs.

The mode comes from the activity detector (`seeforme/live/activity.py`), which measures camera motion on a
160-pixel grey copy of each frame (phase correlation, frame difference, a forward-zoom cue and a walking bob):

| activity | mode | what changes |
|---|---|---|
| walking, turning, or unclear | navigate | balanced verbosity, 20 s repeat hold, no background |
| standing still | scene | quiet verbosity, 60 s hold, static objects are background: only arrivals, approaching objects and critical events |
| standing still with text in view for 1 s | read | object cues muted unless critical or approaching |
| a FIND goal is set | find | the goal is said as soon as it is seen and reminded sooner |

Leaving a still mode is the safe direction: any sign of walking (including slow walking), mixed signals, or a stalled
camera switches back to navigate within about 1-2 s. With `--setup auto` (the default) a camera that is still in the
first seconds after start is taken to be a desk webcam.

## What was measured

* **Detector** (`results/live/detector_choice.md`, `scripts/eval_live_detector.py`): YOLO11s instead of YOLO11n.
  On the 671 labelled BLV-Road-Nav keyframes its recall of the directly mapped COCO classes is 0.85 against 0.80
  (difference +0.05, 95% interval +0.03 to +0.08) at the same precision (0.94), with clear gains on cars, buses,
  bicycles and traffic lights. The whole pipeline (detector, Depth Anything Small, tracker, scheduler) takes 54 ms per
  frame against 51 ms on an M1 Max; with depth every second frame the demo runs at about 20 frames per second. On the
  `make live-check` clip YOLO11n confidently calls a decorated door "refrigerator, approaching"; YOLO11s is less sure
  of that wrong label and stays silent. Doors are not a COCO class at all, which is the vocabulary gap the research
  side documents (`docs/egoblind.md`).
* **Gate and modes on real blind-user video** (`results/live/gate_eval_dev.md`, `scripts/eval_live_gate.py`):
  on 138 development videos (103 minutes, the finetune and select splits; never the holdout or test), wrapping the
  learned scheduler in the gate cuts its cues per minute by 15-19% (1.02 to 0.87 on the COCO cache, 1.07 to 0.87 on
  the open-vocabulary cache) and removes every repeat of the same object within 6 s (dup rate 0.019 to 0), while
  losing no timely event and leaving quiet-window violations unchanged; the same holds for P-CACS and the
  EgoBlind-trained scheduler in all six detector-by-scheduler comparisons (cues per minute -4% to -19%). 86% of the
  label-and-direction repeats the bare schedulers make are about a different track id than the cue they repeat: the
  tracker lost and re-found the object (or a second object with the same label stands in the same direction), and the
  scheduler's per-track memory cannot see that. That is the mechanism the gate's identity memory addresses, and the
  likely cause of the "Person." loop at a desk. The replays are walking clips, so they do not contain a desk session.
* **The desk case.** Replays of a still camera with a person in front confirm that the activity detector reaches
  scene mode within 2 s and detects the desk setup; simulated fidgeting and leaning (with the real tracker and depth
  noise) do not re-announce the user. The exact repetition reported from a real webcam could not be reproduced
  offline, because the learned scheduler stays silent on still synthetic clips; the session log of the next real run
  (`results/live_sessions/`) records every cue and every held cue's reason, which is the check to do.

## Also fixed in this round

* An interrupt cue was printed but never spoken with the macOS voice (the old `say` process had not exited when the
  new one was started); `Speaker.interrupt()` now waits for it.
* Haptic cues only printed `[haptic] buzz`; a laptop has no vibration motor, so they now play a short system sound.
* Muting with `s` did not silence the macOS voice.
* Replayed video uses the file's own timestamps (no drift on variable-frame-rate phone video); the end-of-run summary
  lists haptic cues; `--strict` fails when nothing was tested; the learned scheduler's import failure falls back to
  P-CACS out loud instead of exiting; the detector weights are resolved inside the repository, so starting the demo
  from another directory cannot trigger a download.

## Limits

* The gate and the activity thresholds were calibrated on EgoBlind development clips and synthetic desk scenes, not
  on a recorded walk with this Mac's camera. `describe()` in the end-of-run summary and the session log show what the
  detector concluded; check them on a short real recording before relying on the desk detection.
* Everything spoken depends on the 80 COCO classes; doors, stairs, poles, curbs and kerbs are mostly invisible to it.
* Screen-reader use of the window has not been tested; the demo is driven by keys and speech, the window is for
  sighted developers.
