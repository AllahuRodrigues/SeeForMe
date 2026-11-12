# Online learning from one-sided feedback (M1): development results

Run on 2026-11-12 (warm start added the same day). Code: `seeforme/online/`, `scripts/run_online.py`. Raw outputs: `results/online/`.

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

## Warm start from CampusBench (layer 1 of the data plan)

One full-information pass over the 84 CampusBench v2 training scenarios (about 17,400 labelled candidates, 1,850 positive), then the same one-sided stream. Coverable recall, three seeds each. The last column is the paired bootstrap of pooled recall against the cold start, resampling videos across all three seeds.

| Learner | seed 0 | seed 1 | seed 2 | mean | vs cold start, 95% CI |
|---|---|---|---|---|---|
| Greedy, carry everything | 0.169 | 0.138 | 0.131 | 0.146 | **-0.058** [-0.107, -0.013] |
| Thompson, carry everything | 0.138 | 0.163 | 0.150 | 0.150 | -0.038 [-0.089, +0.007] |
| Greedy, carry weights only | 0.194 | 0.150 | 0.188 | 0.177 | -0.027 [-0.067, +0.014] |
| Thompson, carry weights only | 0.206 | 0.194 | 0.219 | **0.206** | +0.019 [-0.016, +0.054] |

"Carry everything" keeps the simulator's AdaGrad accumulator, posterior precision and feature normaliser. "Weights only" keeps the weights as a prior mean and starts the rest fresh on real video.

5. **Naive simulation pretraining hurts.** Carrying the full optimiser state lowers greedy recall by 0.06, and the interval excludes zero. After 17,000 simulated labels the AdaGrad steps are small and the posterior is confident, so the learner cannot move far from a simulator that is wrong about real video. Quiet violations also rise, to 0.10–0.14.
6. **A weights-only prior is safe but doesn't close the gap.** It removes the harm. Thompson rises to 0.206, and on `finetune` it catches 6–8 of 27 events against 4 for the frozen IL model. Every interval still includes zero. The gap to full information (0.53) is not a cold-start problem, so the next lever is the label signal itself.

## Implicit negatives (a candidate nobody asked about)

When a remembered candidate leaves the 15 s lookback without any feedback covering it, it becomes a negative of weight w. Stream: `train`, `finetune`, then `select`. Pooled over three seeds; `select` holds 33 coverable events per seed, 99 in all.

| w | greedy dev recall | greedy `select` | Thompson dev recall | Thompson `select` |
|---|---|---|---|---|
| 0 (off) | **0.204** | **9/33** | **0.188** | **12/33** |
| 0.02 | 0.173 | 9/33 | 0.183 | 8/33 |
| 0.05 | 0.169 | 9/33 | 0.175 | 7/33 |
| 0.1 | 0.169 | 7/33 | 0.175 | 7/33 |
| 0.3 | 0.154 | 5/33 | 0.160 | 5/33 |

7. **Silence-as-negative makes things worse, and more weight makes them worse still.** Quiet-window violations don't improve consistently either. Selected on `select`: w = 0, so the feature stays off. Most candidates nobody asks about sit next to the ones that matter (same label, same moment), so the negatives push down exactly the features the rare positives rely on.

## Next experiments (selected on `select`, never on holdout)

- **Credit assignment for question labels:** a missed question currently labels every candidate of the event in [earliest, t_q] equally. Weighting the moment closest to the ideal cue time may sharpen the signal.
- **Re-identification memory (M2) on and off:** measure repeat cues, once the GPU embedding run has finished.
- **Bootstrap 95% intervals** (`scripts/bootstrap_ci.py`) on the final comparison.
