# SeeForMe web app

Audio-first camera assistant that runs **in the browser**: YOLO11n (ONNX Runtime Web, WebGPU → WASM
fallback) in a Web Worker, an IoU tracker with metric dynamics (distance, relative speed, time-to-contact,
path-intersection probability), and a scheduler that decides speak / interrupt / haptic / silence under a
speech budget. The default scheduler is the **learned GRU + RL scheduler**, the same model and the same decision
rule as the Python pipeline (`pcacs_learned:rl`, `models/scheduler/scheduler_rl.pt`); the linear **P-CACS** is the
fallback. Works on phones, tablets and laptops; front and rear cameras; video files. Version 0.3.1.

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
   (Continuity Camera and external webcams appear here on a Mac). Front cameras are mirrored, so left/right
   in the cues are swapped automatically.
4. Add to Home Screen: it is a PWA (manifest + service worker). The app shell, configs, benchmark JSON, both
   ONNX models and the ORT loaders are precached with content revisions, so a release replaces exactly the
   files it changed; the ORT wasm binaries are cached on first use under a per-ORT-version cache. A new release
   activates at once (autoUpdate, skipWaiting + clientsClaim); the page reloads when it takes over, but while
   the camera runs the reload waits until Stop. After the reload the status says "Updated to version X" once.

## Modes

| mode | what it does |
|---|---|
| **Navigate** | push: P-CACS speaks only what changes your next action ("Stairs down, about 3 meters. Handrail left."), buzzes for imminent obstacles, cuts in for critical hazards |
| **Find** | pull: "exit", "room 204", "a seat" become goal tags (articles and words under 3 letters are dropped; "seat" also matches chair/bench); matching objects/signs are announced even off the walking path. Without the server only detector objects (COCO classes) can match, so for signs and room numbers the app says they need the server |
| **Read** | reads the text in view (server OCR) |
| **Ask** | rule-based answers over the live event graph: "what's around me", "any stairs?", "what's on my left", "where is the door" (sets a Find goal), "more detail"; voice input where Web Speech recognition exists |

Settings: **Scheduler** Learned (default) / Linear P-CACS (remembered in this browser); verbosity QUIET /
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

Keyboard: **Space** start/stop, **F** flip camera, **M** mute, **A** ask by voice. Shortcuts are ignored
while focus is on a button, link, form field or the Settings summary, and whenever Cmd, Ctrl or Alt is held.

The **Algorithm** panel (right column, always visible) shows the live P-CACS decision and, per candidate
track, distance, TTC, path probability, hazard, urgency, novelty, speech cost and utility, plus the
speech-budget bar, channel state and the last cues; **This session** and **Benchmark** sit below it. This
is the live view for the thesis demo.

Accessibility:

- Status changes (detector ready, camera on/off, errors) go to a single polite `role="status"` region;
  detector loading stages and per-frame diagnostics are visual only.
- Cues and answers are spoken by the app. While the app voice is muted (the **Mute** button, exposed with
  `aria-pressed`), speak cues and answers go to a polite `aria-live` region and interrupt cues go to an
  assertive one, so a screen reader reads each one once and urgent hazards are not queued.
- A skip link jumps to the camera controls, focus is always visible, and Start, Ask, Flip, Mute and the
  mode buttons are at least 44 px tall.
- The camera overlay and cue card are for low-vision users, developers and demos (hidden from screen readers).

## Where things are

| file | what |
|---|---|
| `src/detector.worker.ts`, `src/yolo.ts` | ONNX Runtime Web session, letterbox, decode + NMS |
| `src/tracker.ts` | IoU tracker + `updateDynamics` (distance, speeds, TTC, path probability) |
| `src/pcacs.ts` | event taxonomy, the 29 event features (Python order), utility, budget, actions; `public/config/pcacs.json` holds the tuned weights (exported by `scripts/export_web_config.py`); `DEFAULT_CFG` repeats them in case that file does not load |
| `src/learned.ts` | learned GRU + RL scheduler: per-track windows, ONNX Runtime session, legality masks, RL argmax, model slots; `public/config/scheduler.json` + `public/models/scheduler_rl.onnx` (exported by `scripts/export_web_scheduler.py`) |
| `parity/run.mjs` | Python ↔ TypeScript parity runner (fixtures from `scripts/web_parity.py`) |
| `src/version.ts` | visible app version (bump with every release) |
| `src/compose.ts` | slot-based minimal messages |
| `src/speech.ts` | Web Speech / ElevenLabs playback, interrupt, vibration + earcon |
| `src/camera.ts` | `CameraSource`: facingMode, device picker, flip, video file |
| `src/server.ts` | depth / OCR / TTS client |
| `src/main.ts` | loop, modes, UI, ASK rules |

Known limits: WebGPU support differs by browser (iOS Safari falls back to WASM, ~10 FPS on older
devices); OCR and metric depth need the server; ASK is rule-based (no LLM).

## Dev server vs production

ONNX Runtime is loaded as a raw bundle from `/ort/` (never bundled by Vite): in dev a small plugin in
`vite.config.ts` serves those files straight from `node_modules`; in production they come from
`public/ort/`. Under `npm run dev` the detector runs single-threaded (threaded WASM init hangs there);
the production server on :8000 is cross-origin-isolated and uses up to 4 threads, ~3× faster. If a
worker fails or does not become ready within 25 s the app restarts it single-threaded by itself. A
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
