# M7 plan: free, natural, real-time audio for the web app

Owner request (2026-11-12): "good audio... like ElevenLabs but free... listens and provides output in real time".
Hand-off: the web session gives us `web/src/speech.ts` and a new `web/src/voice/` **after the live-demo PR merges to main** (it will message the commit). Until then: no edits under `web/`, `server/`, `seeforme/live/`.

## Design

- **Speak:** Kokoro-82M (Apache-2.0) via kokoro-js, WebGPU with WASM fallback, sentence-streamed. Lazy-loaded on first use, not in the PWA precache (q8 is about 90 MB).
- **Listen:** Whisper tiny.en (MIT) via transformers.js (Apache-2.0), with a voice-activity detector, fully on device. Lazy-loaded (about 40 MB).
- **Barge-in:** the user's voice cuts the current non-safety utterance.

## Constraints that must survive (from the web session, 2026-11-12)

| # | Constraint | Test |
|---|---|---|
| C1 | Command parsing stays in `web/src/voice.ts` (port of `seeforme/live/voice.py`); Whisper text goes through `parseCommand` unchanged; hands-free still needs the wake phrase | both suites pass `tests/data/voice_commands.json` (wake, direct, unicode lists) |
| C2 | Hazard, interrupt and haptic cues never wait on Kokoro warm-up or synthesis: use speechSynthesis or a pre-rendered clip | unit test: safety cue dispatch path never awaits the Kokoro engine |
| C3 | `Speech.busy` / `remaining()` stay honest under Kokoro (duration from rendered audio length, not word estimate) | unit test with a fake engine of known clip length |
| C4 | Echo suppression: mic ignores input while we speak and 0.4 s after (`markSound` / `soundUntilMs`); our own TTS can never trigger a command or barge-in | unit test; manual check with speakers, not headphones |
| C5 | App fully works before the models arrive; models are not precached | build test: precache manifest excludes model files |
| C6 | Licences: rows in `THIRD_PARTY_NOTICES.md` and entries in `scripts/build_manifests.py` for Kokoro-82M, kokoro-js, transformers.js, Whisper; update the Web Speech recognition row (on-device Whisper means audio no longer goes to Google) | manifest check |
| C7 | Version `web/src/version.ts` 0.4.1 -> 0.5.0 | |
| C8 | `npm test`, `npm run build`, `scripts/web_parity.py --check`, `node web/parity/run.mjs` stay green | CI-equivalent run before PR |
| C9 | iOS Safari: no `navigator.vibrate` (earcon covers it); WebGPU is version-dependent, so measure the WASM real-time factor on an iPhone before Kokoro becomes the default | real-time factor < 0.5 on device, else default stays speechSynthesis |

## Slices

| Slice | Deliverable |
|---|---|
| S1 | `SpeechEngine` interface; speechSynthesis engine extracted with no behaviour change (C2, C3 tests first) |
| S2 | Kokoro engine, lazy load, sentence streaming, honest duration; settings toggle; falls back on load failure |
| S3 | Whisper listener with VAD, echo suppression, wake phrase, `parseCommand` routing (C1, C4) |
| S4 | Barge-in; latency measurement (speech end to first audio, utterance end to parsed command) |
| S5 | Licences, version, parity, build (C5 to C8); iPhone real-time-factor check (C9) with the user's phone |
