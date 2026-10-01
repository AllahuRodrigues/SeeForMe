# SeeForMe web app

Audio-first camera assistant that runs **in the browser**: YOLO11n (ONNX Runtime Web, WebGPU → WASM
fallback) in a Web Worker, an IoU tracker with metric dynamics (distance, relative speed, time-to-contact,
path-intersection probability), and the **P-CACS** scheduler (speak / interrupt / haptic / silence under a
speech budget). Works on phones, tablets and laptops; front and rear cameras; video files.

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
4. Add to Home Screen: it is a PWA (manifest + service worker; the model and runtime are cached).

## Modes

| mode | what it does |
|---|---|
| **Navigate** | push: P-CACS speaks only what changes your next action ("Stairs down, about 3 meters. Handrail left."), buzzes for imminent obstacles, cuts in for critical hazards |
| **Find** | pull: "exit", "room 204", "a seat" become goal tags; matching objects/signs are announced even off the walking path |
| **Read** | reads the text in view (server OCR) |
| **Ask** | rule-based answers over the live event graph: "what's around me", "any stairs?", "what's on my left", "where is the door" (sets a Find goal), "more detail"; voice input where Web Speech recognition exists |

Verbosity QUIET / BALANCED / DETAILED changes the thresholds and which message slots are spoken; units
metres / feet; voice Browser / ElevenLabs. Keyboard: Space start/stop, F flip, M mute, D debug.
The **Debug** panel shows, per track, distance, TTC, path probability, hazard, urgency and utility, the
speech-budget bar and the last cues, which is the live view for the thesis demo.

Accessibility: every cue is also written to an `aria-live="assertive"` region, controls are 44 px+
targets with visible focus, and the visual overlay is for low-vision users, developers and demos.

## Where things are

| file | what |
|---|---|
| `src/detector.worker.ts`, `src/yolo.ts` | ONNX Runtime Web session, letterbox, decode + NMS |
| `src/tracker.ts` | IoU tracker + `updateDynamics` (distance, speeds, TTC, path probability) |
| `src/pcacs.ts` | event taxonomy, utility, budget, actions; `public/config/pcacs.json` holds the tuned weights (exported by `scripts/export_web_config.py`) |
| `src/compose.ts` | slot-based minimal messages |
| `src/speech.ts` | Web Speech / ElevenLabs playback, interrupt, vibration + earcon |
| `src/camera.ts` | `CameraSource`: facingMode, device picker, flip, video file |
| `src/server.ts` | depth / OCR / TTS client |
| `src/main.ts` | loop, modes, UI, ASK rules |

Known limits: WebGPU support differs by browser (iOS Safari falls back to WASM, ~10 FPS on older
devices); OCR and metric depth need the server; ASK is rule-based (no LLM).
