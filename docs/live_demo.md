# The live demo: what it says, when, and why

`make live` runs SeeForMe on the Mac's camera: YOLO detections and metric depth feed a tracker, the learned scheduler
(`pcacs_learned:rl`, a GRU trained with imitation and reinforcement learning on CampusBench) proposes what to say, and
the layer in `seeforme/live/` decides whether it is worth saying now. You talk to it ("SeeForMe, find a door"), and a
cane / guide-dog layer adds the obstacles the detector cannot name. This page describes that layer, the commands, and
what was measured. It is a demo for development and supervised testing, not a mobility aid: it can miss objects and
never tells you that a path is clear.

## Commands

```bash
make voice-setup   # once: vosk + sounddevice and the 40 MB offline English model in models/vosk/ (git-ignored)
make live          # webcam, learned scheduler, repeat filter, modes, cane layer, voice commands, macOS voice
make live-desk     # laptop webcam on a desk: you, in front of it, are never announced unless you come closer
make live-raw      # the bare scheduler (no repeat filter, navigate mode), to hear the difference
make live-pcacs    # the linear P-CACS rule instead of the learned scheduler
make live-elevenlabs   # ElevenLabs voice (needs ELEVENLABS_API_KEY; falls back to the macOS voice)
make live-help     # every option
make live-check    # headless replay of a local clip with scripted voice commands: policy, cues, replies, summary
make live-safety   # the safety harness: 3,000 seeded scenes, gated vs bare (results/live/safety_harness.md)
make eval-live     # re-measure the detector choice and the gate on EgoBlind development replays
make eval-cane     # the cane layer per travel aid on EgoBlind development videos
make eval-voice    # voice commands through the real recogniser (needs voice-setup and macOS say)
```

Useful options (`scripts/run_live.py --help` lists all): `--aid cane|guide-dog|none`, `--voice auto|on|off`, `--mic N`,
`--recogniser snap|open|grammar|hybrid`, `--setup desk|handheld|auto`, `--mode auto|navigate|scene|read|find`,
`--goal "door"`, `--units ft`, `--no-gate`, `--no-cane`, `--no-depth`, `--depth-every N`, `--camera N`
(`--list-cameras`), `--video clip.mp4`, `--commands "2:find a chair;6:pause"`, `--no-log`.

The window shows the camera picture exactly as captured, never mirrored: an object on your left is on the left of the
picture and is announced as "left".

## Talking to it

Say **"SeeForMe"** and a command, in one breath or after a pause: a bare "SeeForMe" opens a 5 s window for one command.
Common mis-hearings of the wake phrase ("see for me", "sea for me", "see four me", "C for me", "hey SeeForMe") count.
Without the wake phrase nothing happens, so ordinary conversation near the laptop does not trigger commands. The
microphone is ignored while SeeForMe is speaking and for 0.4 s after, so its own voice cannot command it.

| say "SeeForMe, ..." | what happens |
|---|---|
| find a door / find me a chair / where is the exit / where can I find a seat / show me the way out / take me to room 204 | sets a FIND goal; spoken numbers become digits ("room two oh four" is room 204) |
| stop looking / cancel / never mind | clears the FIND goal |
| pause / stop / wait / camera off, and start / resume / continue / camera on | in the terminal, pauses routine cues and resumes them; the camera stays on (it says so for "camera off"), and anything that carries safety information (in the path, approaching, a vehicle, the FIND goal, the cane layer) is still said. In the web app, pause turns the camera off and says so |
| what's around me / describe / look around | up to three named objects, nearest first, with direction and distance |
| read / read the sign / what does it say / what's written here | reads the text in view (needs OCR, the default) |
| is there a chair / is the path clear / what's on my left | a rule-based answer from the current tracks and the cane layer; never "the path is clear" |
| repeat / say that again | the last cue again |
| mute / be quiet, and unmute / sound on | sound off and on |
| more detail / less detail / normal detail | verbosity |
| navigation mode / scene mode / reading mode / automatic mode | pins a mode, or returns to the activity detector |
| switch camera / front camera / back camera | the next camera; in the web app, the named side, and it says when the browser cannot tell which way a camera faces |
| help / what can I say | the list of commands |
| quit / goodbye | ends the session |

The recogniser is Vosk (`vosk-model-small-en-us-0.15`), fully offline at 16 kHz. Typed commands work in any terminal,
with or without the microphone (the same phrases, the wake phrase optional), and `--commands` scripts them for replays.
The parser (`seeforme/live/voice.py`) and the web app's port (`web/src/voice.ts`) share their test vectors
(`tests/data/voice_commands.json`) and agree on every one of 31,242 generated phrases.

Keys, while the window has focus:

| key | what it does |
|---|---|
| `h` | speaks the list of keys |
| `l` | listens for one command without the wake phrase |
| `w` | what is around me |
| `r` | reads the text in view |
| `p` | pause or resume |
| `m` | cycles the mode: automatic, navigate, scene, read, find, back to automatic (spoken) |
| `space` or `s` | sound on or off |
| `f` | asks for a FIND goal in the terminal |
| `v` | verbosity: quiet, balanced, detailed |
| `g` | repeat filter on or off |
| `d` | metric depth on or off |
| `i` | debug overlay on or off |
| `1`-`5` | scheduler: threshold, cooldown, CACS, P-CACS, learned |
| `q` | quit |

## What decides whether a cue is spoken

The scheduler proposes at most one cue every 0.5 s. Then:

1. **The repeat filter** (`seeforme/live/gate.py`) remembers what it has said about each object and direction (left,
   ahead, right). Identity survives a tracker that loses and re-finds the object, and a newcomer that takes over a
   lost track's id is not mistaken for it when its range or position jumps; signs are told apart by their numbers
   ("room 204" and "room 206" are two signs). A cue is held when the same object in the same direction was announced
   recently and nothing new happened. New means: the object came materially closer, is still closing in the path,
   changed side, came back nearer after being lost, or became critical. Objects in the walking path, vehicles and
   FIND goals never wait for confirmation; other tracks must be seen in several frames first, so a one-frame
   flicker is never spoken. Without measured depth, closing is read from the growth of the box sides the frame edge
   does not cut.
2. **Background and the user's own body.** When standing still, objects that stay where they are become background and
   speak only once they measurably close in. Vehicles never become background, nothing is held on its first sample, and
   a FIND goal is never background. With a desk webcam, the person in front of the camera at start-up is you: you are
   not announced when you lean in, but a second person who walks up is.
3. **The safety rule.** If the scheduler will not offer a held cue again in time once its object qualifies, the filter
   says it itself. The design rule, checked by the harness below, is that the gated demo warns no later than the bare
   scheduler would (one decision step of slack), or at least 1.5 s before contact.
4. **Read mode** mutes object cues unless they are critical or approaching, so a sign can be read without interruptions
   about furniture.
5. **One audio channel.** A spoken reply or a cane cue never holds back a safety cue from the scheduler (something in
   the path, approaching, a vehicle, stairs or the FIND goal): the safety cue cuts it. A cane cue may cut a reply, never
   a scheduler warning. A haptic "Stop." wins over any spoken cue, since it does not occupy the audio channel.

## The cane and guide-dog layer

The 80 COCO classes have no wall, door, pole, post, kerb, barrier or stairs, so the scheduler cannot warn about them.
`seeforme/live/cane.py` looks only at the metric depth map: it fits the floor (RANSAC on inverse depth, which also gives
camera height and pitch), finds surfaces that rise out of it in the walking corridor, and sorts them into sectors (left,
ahead, right) and levels (ground, body, head). It speaks short cues ("Obstacle ahead, about 1 meter.", "Something at
head height ahead.") and, while walking, a haptic "Stop." for something within 0.6 m ahead in a steady approach. It is
off at a desk, without a depth map, and quiet while you stand still unless the obstacle approaches.

What it says depends on what you travel with (`--aid`, the web app's "I travel with" setting, the server's `aid`
field). The rules follow the BLV-Road-Nav focus group (Islam et al., arXiv 2407.16777): stay quiet about what the cane
already touches, and warn about "objects that pick you before you pick them".

| aid | what the cane layer says |
|---|---|
| cane (default) | no low obstacle within the cane's reach (1.2 m), beyond it only in a steady approach; body level from 1.2 to 2.5 m, said again only once 40% nearer; head level within 2.5 m always; the stop only for body or head level |
| guide-dog | head level within 2.5 m and the stop only: the dog leads around ground and body obstacles |
| none | every cue: obstacles of any level within 2 m, the stop for any level |

The scheduler's cues about detected objects are the same for every aid.

## Modes from what you are doing

The mode comes from the activity detector (`seeforme/live/activity.py`), which measures camera motion on a 160-pixel
grey copy of each frame (phase correlation, frame difference, a forward-zoom cue and a walking bob):

| activity | mode | what changes |
|---|---|---|
| walking, turning, or unclear | navigate | balanced verbosity, repeats held while nothing changes, no background |
| standing still | scene | quiet verbosity, longer hold, static objects are background: only arrivals, approaching objects and critical events |
| standing still with text in view for 1 s | read | object cues muted unless critical or approaching |
| a FIND goal is set | find | the goal is said as soon as it is seen and reminded sooner |

Leaving a still mode is the safe direction: any sign of walking, mixed signals, or a stalled camera switches back to
navigate within about 1-2 s. With `--setup auto` (the default) a camera that is still in the first seconds after start
is taken to be a desk webcam.

## What was measured

* **Announcement gate** (`results/live/gate_eval_dev.md`, `scripts/eval_live_gate.py`): on EgoBlind development
  replays (`finetune` and `select`, two detector caches, three schedulers) the gate lost no timely event, delayed
  none and left quiet-window violations unchanged in all six comparisons; cues per minute moved by -2% to +2%. It
  also removed no repeats there (115 label-and-direction repeats before and after): these are walking clips where
  objects leave the view quickly, so they measure what the gate costs, not what it saves at a desk. The desk case
  is covered by `tests/test_live_gate.py`.

* **Speech actually played** (`results/live/playback_browser.md`, `web/src/playback.ts`): the scheduler charges its
  speech budget with an estimate (0.35 s per word, at least 1.2 s). On ten typical cues in a desktop Chromium with
  the Samantha voice, the voice ran 34% longer than that in total (21.5 s against 16.0 s, largest overrun 1.1 s).
  The web app now settles every cue against the seconds it really played: a cue that is cut off, cancelled or fails
  is refunded for what was not heard, a slow voice is charged the difference, and the channel is not free while a
  cue is still sounding. One run, one browser, one voice; Safari, phones and the ElevenLabs voice are not measured.
* **Voice commands** (`results/live/voice_eval.md`, `scripts/eval_voice.py`): every hands-free command vector plus 30
  natural variants, synthesised with four macOS voices (US, UK, Australian and Irish English), clean and with pink
  noise at 20 dB, then decoded by the real Vosk recogniser through the live code path: 736 command recordings and 184
  negatives. The default "snap" recogniser (open vocabulary, near-misses snapped to the command words) gets 89.9% of
  commands fully right (intent and argument; 95% interval 88-92%) against 81.8% for the open vocabulary alone. No plain
  sentence woke it (0 of 128), but sentences built to sound like the wake phrase ("Let's see, for me it's fine") did in
  15 of 32. The recogniser was chosen on these recordings, so the 89.9% is in-sample. On three voices that played no
  part in the choice (Indian, South African and US English; the same phrases), it gets 86.1% (475 of 552; 83-89%)
  against 72.3% for the open vocabulary, with the same false-wake rate. Synthetic speech is still cleaner than a real
  voice in a real room.
* **Cane layer** (`results/live/cane_eval_dev.md`, `scripts/eval_live_cane.py`): 84 EgoBlind development videos
  (70 minutes; never the holdout or test splits). With every cue (aid none) the layer speaks 6.1 times a minute while
  walking and stops the user 1.2 times a minute; the cane default halves that to 3.2 cues and 0.41 stops a minute, and
  the guide-dog setting to 0.6 cues. Its cues come about 1.5 times more often than chance just before the blind users'
  own questions about obstacles the detector cannot name (lift 1.53, 95% interval 0.93-2.38 with the cane rules, 1.56
  and 1.14-2.04 with every cue), and not more often before other events (the control, 0.92).
* **Detector** (`results/live/detector_choice.md`, `scripts/eval_live_detector.py`): YOLO11s instead of YOLO11n.
  On the 671 labelled BLV-Road-Nav keyframes its recall of the directly mapped COCO classes is 0.85 against 0.80
  (difference +0.05, 95% interval +0.03 to +0.08) at the same precision (0.94), with clear gains on cars, buses,
  bicycles and traffic lights. The whole pipeline takes 54 ms per frame against 51 ms on an M1 Max; with depth every
  second frame the demo runs at about 20 frames per second.

## Limits

* Every threshold was calibrated on EgoBlind development clips, synthetic scenes and synthetic speech, not on a recorded
  walk with this Mac's camera or on real voices in a real room. The session log (`results/live_sessions/`) records every
  cue, every held cue's reason and every command heard: check it on a short real recording before relying on the demo.
* The scheduler names only the 80 COCO classes; the cane layer finds walls, poles and other surfaces by geometry but
  cannot say what they are, and depth from one camera fails on glass, mirrors and in the dark.
* Sentences that sound like the wake phrase can trigger a command (15 of 32 such test sentences did).
* Screen-reader use of the window has not been tested; the demo is driven by voice, keys and speech, and the window is
  for sighted developers.
