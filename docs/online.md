# Online learning from one-sided feedback (M1): development results

Run on 2026-11-12. Code: `seeforme/online/`, `scripts/run_online.py`. Raw outputs: `results/online/`.

## Setup

- **Stream:** EgoBlind `train` then `finetune`, 558 videos, 290 minutes, 160 events the detector can see, 111 proxy users (one per source recording). Videos are ordered by recording, then by clip.
- **Feedback:** after each cue the user taps with probability 0.5. Questions are always asked.
- **Budget:** the controller targets 2 cues a minute; every learner stays at about 1.7.
- **Evaluation:** progressive validation (predict, then update). Holdout and test are untouched.

## Coverable recall

| Policy | seed 0 | seed 1 | seed 2 | mean | quiet violations |
|---|---|---|---|---|---|
| P-CACS (fixed) | 0.087 | | | | 0.02 |
| October IL (frozen; trained on `train`, so in-sample there) | 0.331 overall; **0.148** on `finetune` | | | | 0.04 |
| October RL (frozen) | 0.050 | | | | 0.00 |
| Online greedy | 0.188 | 0.212 | 0.212 | **0.204** | 0.11 |
| Online apple tasting (25% of budget) | 0.169 | 0.181 | 0.200 | **0.183** | 0.10 |
| Online Thompson | 0.225 | 0.175 | 0.163 | **0.188** | 0.10 |
| Full-information upper bound | 0.525 | | | | 0.09 |

## What this says

1. **One-sided feedback costs about 60% of the learnable recall.** With the same features and the same budget, the learner that sees every label reaches 0.53; the learners that see only taps and questions reach about 0.20. This gap is the thesis problem, now measured.
2. **Naive exploration doesn't close the gap.** Budgeted apple tasting and Thompson sampling sit within seed noise of greedy. The seed-0 Thompson result on `finetune` (7/27) didn't replicate (4/27 and 4/27). Exploration isn't the bottleneck at this budget.
3. **Online learners beat the fixed rule and the frozen RL model** (0.20 vs 0.09 and 0.05). Against the frozen IL model on its out-of-sample part (`finetune`), they're level: 4/27 for IL against 2–7/27 for the online learners. With n = 27, nothing there is separable.
4. **Quiet-window violations go up** from 0.02–0.04 for the frozen models to about 0.10. Silence is never labelled, so the learner speaks in some windows where users had confirmed nothing was there.

## Next experiments (selected on `select`, never on holdout)

- **Warm start (online-to-batch):** initialise the learner from CampusBench full-information labels (layer 1 of the data plan), then learn online from one-sided feedback. This tests whether simulation pretraining recovers part of the 0.33 gap.
- **Credit assignment for question labels:** a missed question currently labels every candidate of the event in [earliest, t_q] equally. Weighting the moment closest to the ideal cue time may sharpen the signal.
- **Re-identification memory (M2) on and off:** measure repeat cues, once the GPU embedding run has finished.
- **Bootstrap 95% intervals** (`scripts/bootstrap_ci.py`) on the final comparison.
