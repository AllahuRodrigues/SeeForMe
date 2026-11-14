# SeeForMe web app

Audio-first camera assistant that runs **in the browser**: YOLO11 (ONNX Runtime Web: YOLO11s on WebGPU, YOLO11n
on WASM) in a Web Worker, an IoU tracker with metric dynamics (distance, relative speed, time-to-contact,
path-intersection probability), and a scheduler that decides speak / interrupt / haptic / silence under a
speech budget. The default scheduler is the **learned GRU + RL scheduler**, the same model and the same decision
rule as the Python pipeline (`pcacs_learned:rl`, `models/scheduler/scheduler_rl.pt`, trained on CampusBench
synthetic campus scenes, not on EgoBlind video); the linear **P-CACS** is the fallback. In front of either sits
the **repeat filter** (`src/gate.ts`, a port of `seeforme/live/gate.py`): it holds what was just said, the desk
user and still background, and never what approaches, is critical or is a Find goal. Works on phones, tablets
and laptops; front and rear cameras; video files; spoken commands. Version 0.5.0.

```bash
cd web && npm install && npm run dev        # http://localhost:5173  (Vite dev server)
npm run build                                # web/dist, served by server/app.py
```

The optional Python server adds what the browser cannot run yet:

```bash
cd .. && make serve        # http://localhost:8000  -> metric depth (Depth Anything V2), OCR (EasyOCR), ElevenLabs TTS
make serve-ssl             # https://<mac-ip>:8000 (self-signed) -> phones/tablets can open the camera
```

Without the server the app estimates distance from box height (and says "close" instead of "about 2
meters", because the composer hedges low-confidence depth). With the server, distances come from Depth
Anything V2 at ~4 Hz and READ mode works.

## Using it on a phone or iPad

1. Start the server with `make serve-ssl` and note your Mac's LAN address (System Settings → Network).
2. On the phone open `https://<mac-ip>:8000`, accept the self-signed certificate, allow the camera.
3. **Flip** switches rear ↔ front (`facingMode: environment | user`); the camera picker lists every device
   (Continuity Camera and external webcams appear here on a Mac). No camera is mirrored, the front one
   included: the picture is shown as captured, and "left" / "right" in the cues are the picture's left and right.
4. Add to Home Screen: it is a PWA (manifest + service worker). The app shell, configs, benchmark JSON, the
   scheduler model, YOLO11n and the ORT loaders are precached with content revisions, so a release replaces
   exactly the files it changed; the ORT wasm binaries are cached on first use under a per-ORT-version cache, and
   YOLO11s (36 MB, WebGPU only) on first use under a cache named by its content hash. A new release
   activates at once (autoUpdate, skipWaiting + clientsClaim); the page reloads when it takes over, but while
   the camera runs the reload waits until Stop. After the reload the status says "Updated to version X" once.

## Modes

| mode | what it does |
|---|---|
| **Navigate** | push: P-CACS speaks only what changes your next action ("Stairs down, about 3 meters. Handrail left."), buzzes for imminent obstacles, cuts in for critical hazards |
| **Find** | pull: "exit", "room 204", "a seat" become goal tags (articles and words under 3 letters are dropped; "seat" also matches chair/bench); matching objects/signs are announced even off the walking path. Without the server only detector objects (COCO classes) can match, so for signs and room numbers the app says they need the server |
| **Read** | reads the text in view (server OCR) |
| **Ask** | rule-based answers over the live event graph: "what's around me", "any stairs?", "what's on my left", "where is the door" (sets a Find goal). Typed text goes through the voice command parser first, so "help", "unmute" or "where is room two oh four" do what they say |

## Voice

- **Talk** (the A key, or the microphone button in Ask): one command, no wake phrase needed. The first press only
  says that the browser's speech recognition may send the audio to its speech service (Chrome: Google); the next
  press listens.
- **Hands-free voice** (the V key; off by default, remembered): continuous listening for "SeeForMe, ..." (also the
  usual mis-hearings, "sea for me", "c for me"). The command is the text after the last wake phrase; the wake phrase
  alone plays a short tone and opens 5 s for one command. It restarts itself with a growing back-off and stops while
  the page is hidden.
- Commands: start, pause, quit, find ..., stop looking, read, what's around me, repeat, mute / unmute, more / less
  detail, flip camera, help, or any question. The parser (`src/voice.ts`) is a rule-for-rule port of the Python one
  (`seeforme/live/voice.py`): a FIND verb wins over mode and detail words, which must be the whole command ("find
  the scene mode button" is a FIND, "switch to reading mode" a mode), and spoken numbers become digits ("room two oh
  four" -> room 204, "room oh four" -> room 04). `tests/data/voice_commands.json` holds the cases both pass.
- The app never acts on its own voice: results that arrive while it speaks (and 0.4 s after), or that repeat what it
  or the status line (read aloud by screen readers) said a moment ago, are ignored. Of a result's alternatives the
  best decides; a lower one may only refine a command addressed to the app, never start one.

## Repeat filter, activity and cane cues

- **Repeat filter** (`src/gate.ts`, Settings checkbox, on by default and remembered): remembers what was said per
  object identity (label, or sign text whose numbers must match) and direction, across tracker id switches, and
  holds repeats; closer, closing in, a side change, critical (measured TTC < 2 s in the path), a Find goal and the
  hold running out (20 s walking, 60 s still) pass. A new track only continues a remembered object when it
  appears where that one vanished, at about its range, and that one had not walked away since it was said; one
  back in the path after a dropout, or closing in, is said again. The same continuity decides whether a new track
  joins a background object or is the desk user, so a stranger who walks up to a laptop, or a visitor in front of
  the user, is warned about like anyone else. A detection that flickers (a new track id at every sighting) is
  confirmed by its sightings together. A held cue's seconds go back to the speech budget, and the filter says a held
  object itself when it later closes in (up to 20 s later for one held as background or as the user). A buzz and a
  spoken cue can pass at the same decision. `#action-why` in the Algorithm panel shows each verdict.
- **Activity** (`src/activity.ts`): camera motion on a 160 px grey frame (phase correlation). Still: 60 s hold and
  still objects as background; a camera still in the first seconds after Start is a laptop on a desk, and the
  person in front of it is the user (held until they measurably come closer). 2 s of walking ends that.
- **Cane cues** (`src/cane.ts`): with the server, `/api/depth` also returns a class-agnostic obstacle summary
  (the app sends `walking` from the activity detector and the travel aid); while navigating (not still, not at a
  desk) an obstacle no detected object explains is said ("Obstacle ahead, about 1 meter, and on your right.":
  path choice names what blocks, never a side that looks open, since drop-offs are not detected) and goes through
  the repeat filter; very close straight ahead it is a buzz. The app never says or implies a path is clear.
- **I travel with** (Settings; White cane by default, remembered): what the obstacle cues leave to the user's own
  aid, after the BLV-Road-Nav focus group (Islam et al., arXiv 2407.16777): quiet about what the cane can already
  touch, always warn at head height.
  - White cane: no cue for a low obstacle within the cane's reach (1.2 m), farther ones only in a steady approach;
    body height from 1.2 to 2.5 m ahead; head height always (within 2.5 m); the buzz only at body or head height
    within 0.6 m in a steady approach.
  - Guide dog: head height and the buzz only (the dog leads around the rest).
  - Neither: every obstacle cue, as before.
  With a cane or a dog only the corridor ahead is spoken about. Cues about detected objects (people, vehicles,
  chairs, the Find goal) are the same for every choice.

Settings: **I travel with** White cane (default) / Guide dog / Neither (remembered; see above); **Scheduler**
Learned (default) / Linear P-CACS (remembered in this browser); verbosity QUIET /
BALANCED / DETAILED changes which message slots the linear P-CACS speaks (the learned scheduler picks its own
slots, verbosity only changes its speech-cost input, as in Python); units metres / feet; voice Browser
(default) / ElevenLabs (through the server, used only when the server reports an ElevenLabs API key). An
ElevenLabs cue whose audio is not back within 1.5 s is spoken by the browser voice instead, a cue that cannot
start within 2 s is dropped rather than spoken late, and interrupts use the browser voice unless their audio is cached.

## Learned scheduler

The browser runs `public/models/scheduler_rl.onnx` (~100 KB) with ONNX Runtime Web on the main thread
(single-threaded wasm, loaded after the detector so both share the runtime download). Per decision step,
exactly as `SchedulerPolicy` (mode `rl`) in `seeforme/ml/scheduler_model.py`:

1. the 29 P-CACS event features per track, in Python's `EventFeatures` order (`src/pcacs.ts`);
2. candidates with confidence ≥ `minConfidence`; each track keeps its last K = 8 feature vectors;
3. the GRU reads each candidate's window (right-aligned, zero-padded in front, with a mask) and returns action
   logits, slot logits and the learned silence bias;
4. legality: speak needs a free channel, interrupt a busy one (not the track being spoken), neither within
   `recentBlock` of the track's last announcement; haptic needs `hapticRepeat` since the track's last buzz;
5. one argmax over the speak / interrupt / haptic log-odds against each candidate's own silence logit
   (illegal = -1e9), followed by the silence bias: the first maximum wins;
6. the message names the object plus the slots the slot head turns on (sigmoid > 0.5).

If the model or ONNX Runtime fails to load (or a run fails) the app switches to the linear P-CACS and says so
in the status line; the Settings option then reads "Learned (unavailable)". The Algorithm panel shows, per
candidate, the best legal log-odds `L` instead of the utility `U`.

Re-export after retraining (checks ONNX Runtime against PyTorch, max abs diff < 1e-4, and writes
`public/config/scheduler.json` with the feature names, K, the silence bias and the model's sha256):

```bash
python scripts/export_web_scheduler.py
```

**Parity check** (Python and TypeScript must decide the same): `scripts/web_parity.py` records Python decision
steps (track states before each step, channel, Python's features, logits and decision) on CampusBench test
scenarios, synthetic busy-channel sequences and random-logit rule cases; `parity/run.mjs` replays them through
the TypeScript code (bundled with esbuild) and the ONNX model (onnxruntime-web in node):

```bash
cd web && node parity/run.mjs                      # committed fixture parity/fixtures/parity_small.json.gz
python scripts/web_parity.py --all --synthetic 3 --synthetic-steps 200 --out /tmp/parity.json.gz --check
pytest tests/test_web_parity.py                    # skipped without node / web/node_modules
```

On all 66 CampusBench test scenarios plus 3 × 200 synthetic steps (4,560 decision steps, 17,768 candidates)
every decision, cue message, slot list and duration is identical; the 29 features agree to 7e-15 (identical in
float32), the logits to 8e-6; 400 random-logit rule cases (79 interrupts, exact ties) agree as well.

Known differences: text tracks use the keyword text-importance rules (Python's CampusBench evaluation uses the
learned n-gram classifier, which the browser does not ship); Find also matches a multi-word goal as a whole
phrase inside a longer sign ("gate 7" in "Gate 7 →"), which Python's exact-word match does not.

Keyboard: **Space** start/stop, **F** flip camera, **M** mute, **A** Talk (one spoken command), **V** hands-free
voice on/off. Shortcuts are ignored while focus is on a button, link, form field or a disclosure summary, and
whenever Cmd, Ctrl or Alt is held.

The research view sits in the collapsed **How SeeForMe decides** section below the main screen: the
**Algorithm** panel shows the live decision and, per candidate track, distance, TTC, path probability, hazard,
urgency, novelty, speech cost and the score, plus the repeat filter's verdict, the speech-budget bar, channel
state and the last cues; **This session** and **Benchmark** follow. This is the live view for the thesis demo.

Accessibility:

- Status changes (detector ready, camera on/off, errors) go to a single polite `role="status"` region;
  detector loading stages and per-frame diagnostics are visual only.
- Cues and answers are spoken by the app. While the app voice is muted (the **Mute** button, exposed with
  `aria-pressed`), speak cues and answers go to a polite `aria-live` region and interrupt cues go to an
  assertive one, so a screen reader reads each one once and urgent hazards are not queued.
- A skip link jumps to the controls, focus is always visible, and every control is at least 44 px tall (Start
  80 px, Talk and Hands-free 64 px). Start, Talk, Hands-free, Mute and Flip declare their keys
  (`aria-keyshortcuts`). The two disclosures (Settings, How SeeForMe decides) have their headings just before
  them, since a heading inside a `<summary>` is lost to heading navigation in some screen readers.
- The camera overlay and cue card are for low-vision users, developers and demos (hidden from screen readers).

## Where things are

### On an iPhone

Open the site in Safari, then Share, Add to Home Screen: it then runs full screen like an app. Start must be tapped once
(iOS only lets a page use the camera and play sound after a touch). Speech plays with the ring/silent switch on silent.
There is no vibration on iOS; a short tone stands in for it. Not yet tried on a device: see the pull request.

### The voice

Cues are spoken by a recorded voice pack (Kokoro-82M, voice `af_heart`, 594 clips in one 5 MB file that is fetched in
the background and cached). A cue is a few fragments joined at its commas. Text the pack does not hold (sign text,
answers to questions) is spoken whole by the device voice, and so is everything until the pack has loaded. Settings,
Voice chooses between Natural, Device voice and ElevenLabs.

The device voice is chosen by `src/devicevoice.ts`: the most natural English female voice the browser lists, with
downloaded Premium and Enhanced system voices and Edge's online Natural voices ahead of the standard ones, and never
a novelty voice. Settings names the voice in use. On an iPhone a better system voice is added under Settings,
Accessibility, Read & Speak (Spoken Content on older iOS), Voices, English; whether Safari then offers it to web pages
differs between iOS versions and has not been checked on a device here.

| file | what |
|---|---|
| `src/detector.worker.ts`, `src/yolo.ts` | ONNX Runtime Web session, letterbox, decode + NMS |
| `src/tracker.ts` | IoU tracker + `updateDynamics` (distance, speeds, TTC, path probability) |
| `src/pcacs.ts` | event taxonomy, the 29 event features (Python order), utility, budget, actions; `public/config/pcacs.json` holds the tuned weights (exported by `scripts/export_web_config.py`); `DEFAULT_CFG` repeats them in case that file does not load |
| `src/learned.ts` | learned GRU + RL scheduler: per-track windows, ONNX Runtime session, legality masks, RL argmax, model slots; `public/config/scheduler.json` + `public/models/scheduler_rl.onnx` (exported by `scripts/export_web_scheduler.py`) |
| `parity/run.mjs` | Python ↔ TypeScript parity runner (fixtures from `scripts/web_parity.py`) |
| `src/version.ts` | visible app version (bump with every release) |
| `src/compose.ts` | slot-based minimal messages |
| `src/speech.ts` | the voice: recorded pack first, then the device voice or ElevenLabs; interrupt, vibration + earcon |
| `src/devicevoice.ts` | which device voice speaks what the pack does not hold, and the note Settings shows about it |
| `src/voicepack.ts` | what the recorded voice can say: every cue fragment (twice: before a comma, before a full stop) and the fixed app sentences; plans a sentence as clips, or hands it to another voice whole |
| `src/clipvoice.ts` | plays the pack through Web Audio from one sprite file (`public/voice/af_heart/`), back to back on the audio clock, with an exactly known length |
| `voice/build.mjs` | records the pack with Kokoro-82M (Apache-2.0); `cd web/voice && npm install && node build.mjs`; not part of the web build |
| `src/sheet.ts` | the Settings sheet on a phone: spring by damping ratio and response, one-to-one drag, momentum projection, rubber-banding |
| `src/tokens.css`, `src/base.css`, `src/styles.css` | the design system (Hallmark, theme Ribbon) and the one screen built on it |
| `src/playback.ts` | measured playback: each cue's start, end and how it ended, settled once against the scheduler's speech budget (unplayed seconds refunded, overruns charged); the channel stays busy while a cue is still sounding |
| `src/camera.ts` | `CameraSource`: facingMode, device picker, flip, video file; every open is cancellable (a newer open, a stop or a video file wins, and no stream is left running) |
| `src/camcontrol.ts` | `CameraControl`: the one path that opens and stops the camera for Start / Space, Flip, the picker, a video file and the voice commands start, pause, goodbye, flip / front / back camera; says only what the browser reports |
| `src/server.ts` | depth / OCR / TTS client |
| `src/gate.ts` | repeat filter (announcement gate) in front of either scheduler |
| `src/activity.ts` | camera motion: still / walking / unclear, desk webcam |
| `src/voice.ts` | command parser (spoken and typed) and the Web Speech listener (Talk, hands-free) |
| `src/cane.ts` | class-agnostic obstacle cues from the server's depth summary |
| `test/*.test.ts` | unit tests: `npm test` type-checks them (`test/tsconfig.json`) and runs them with node's test runner |
| `src/main.ts` | loop, modes, UI, ASK rules |

Known limits: WebGPU support differs by browser (iOS Safari falls back to WASM, ~10 FPS on older
devices); iOS Safari has no vibration API, so on iPhone a haptic cue is only the earcon; OCR and metric depth
need the server; ASK is rule-based (no LLM).

## Dev server vs production

ONNX Runtime is loaded as a raw bundle from `/ort/` (never bundled by Vite): in dev a small plugin in
`vite.config.ts` serves those files straight from `node_modules`; in production they come from
`public/ort/`. Under `npm run dev` the detector runs single-threaded (threaded WASM init hangs there);
the production server on :8000 is cross-origin-isolated and uses up to 4 threads, ~3× faster. Start-up is a
ladder: YOLO11s on WebGPU with one WASM thread (WebGPU session creation hangs in a worker with threaded WASM),
then YOLO11n on threaded WASM in a fresh worker, then single-threaded WASM, then the failure is said aloud. A
detection request that fails, or gets no answer within 5 s, returns no detections instead of stalling the loop.

## Beta readiness: accessibility and safety
This is an **experimental prototype**, not a mobility safety aid. It cannot establish that a path is clear.
Never use it instead of a cane, guide dog, or orientation-and-mobility training, especially at roads or stairs.

- Browser camera requires a secure context (HTTPS) or localhost.
- Start with **Find** or **Ask** in a familiar indoor setting and confirm spoken information independently.
- Camera frames feed the local browser detector; connecting the optional SeeForMe server additionally sends frames for OCR/depth processing to that server. ElevenLabs voice uses an optional server-backed provider.
- Do not deploy a public service with optional OCR/depth/TTS endpoints without authentication, rate limiting, and a clear privacy policy.
- Screen-reader labels, status announcements, and keyboard controls require manual testing with VoiceOver and TalkBack before any public beta.
- Do not claim the system is validated, safe for independent navigation, or production-ready on Meta glasses without actual user testing.
