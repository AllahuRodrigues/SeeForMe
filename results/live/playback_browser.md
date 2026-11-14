# Measured playback against the app's estimate (browser voice)

One run on 2026-11-13: Chromium 156 on macOS (the desktop app's built-in browser), voice Samantha, rate 1.05 (what
`web/src/speech.ts` picks), ten typical cues spoken one after another through `speechSynthesis`, timed from the
utterance's `start` event to its `end` event. The estimate is `speechSeconds` in `web/src/compose.ts`
(0.35 s per word, at least 1.2 s), which is what the scheduler charges to the speech budget when it decides.

| cue | estimate (s) | played (s) | played - estimate (s) |
|---|---|---|---|
| Car, ahead. | 1.20 | 1.23 | +0.03 |
| Person, slightly left. | 1.20 | 1.79 | +0.59 |
| Motorcycle, slightly right. | 1.20 | 2.19 | +0.99 |
| Bicycle, close ahead. | 1.20 | 1.79 | +0.59 |
| Stairs going down ahead. | 1.40 | 1.66 | +0.26 |
| Door on the right, 2 meters. | 2.10 | 2.03 | -0.07 |
| Car approaching from the left, 3 meters. | 2.45 | 2.72 | +0.27 |
| Person ahead, 4 meters, moving left. | 2.10 | 2.96 | +0.86 |
| Obstacle very close, stop. | 1.40 | 2.32 | +0.92 |
| Bench, slightly left, 5 meters. | 1.75 | 2.83 | +1.08 |
| **total** | **16.00** | **21.52** | **+5.52** |

The voice ran 34% longer than the estimate in total; the largest single overrun was 1.08 s. The gap from `speak()` to
the `start` event was under 0.01 s once the voice was warm. In a fresh page, before any utterance, the first cue took
about 5 s from `start` to `end` for the same two words (voice warm-up); the app's Start gesture speaks a silent
utterance for that reason.

What this means for the budget: charged with estimates alone, the app spoke about a third more seconds than its budget
counted on this voice. `web/src/playback.ts` now settles each cue against the measured seconds, so that difference
becomes debt and the budget follows what was heard.

Not measured here: Safari, any phone, the ElevenLabs voice, a second run. A phone's voice and rate differ, so these
numbers do not transfer.
