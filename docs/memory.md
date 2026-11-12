# Long-term appearance memory (M2): development results

Run on 2026-11-12. Code: `seeforme/tracking/reid.py`, `scripts/eval_reid.py`. Raw outputs: `results/memory/`.

Every detection in the 592 development-pool videos has a CLIP ViT-B/32 embedding computed on the GPU (414,017 detections, about 3 hours on MPS). A confirmed track that dies goes into a gallery for 120 s. A new track of the same label with cosine ≥ θ inherits its announced state.

## Results on `train` + `finetune` (558 videos, 290 minutes)

| Policy | Re-ID | Coverable recall | Label repeats / min | Re-IDs of announced objects | Δ recall (95% CI) | Δ label repeats / min (95% CI) |
|---|---|---|---|---|---|---|
| P-CACS | off | 0.087 | 0.214 | | | |
| P-CACS | θ 0.80 | 0.087 | 0.183 | 168 | +0.000 | **-0.031** [-0.064, -0.004] |
| P-CACS | θ 0.85 | 0.087 | 0.196 | 127 | +0.000 | -0.017 [-0.047, +0.000] |
| October IL | off | 0.331 | 0.338 | | | |
| October IL | θ 0.85 | 0.331 | 0.338 | 201 | +0.000 | +0.000 |
| Online greedy | off | 0.188 | 0.496 | | | |
| Online greedy | θ 0.80 | 0.125 | 0.545 | 611 | **-0.062** [-0.117, -0.012] | +0.048 [-0.071, +0.171] |
| Online greedy | θ 0.85 | 0.188 | 0.472 | 401 | +0.000 [-0.048, +0.044] | -0.024 [-0.104, +0.055] |

Match precision against ground-truth object ids, where both sides have one: **0.90** at θ 0.80 (n = 2,041) and 0.89 at θ 0.85 (n = 1,863).

## What this says

1. **Matching works; on EgoBlind there's almost nothing for it to do.** About 90% of matches are the same object. But the clips average 31 seconds, so an object rarely leaves and comes back after being announced: 127–611 such cases in 290 minutes. Most of the 85–92 re-identifications a minute are tracks that flicker out for two seconds and were never announced, and those change nothing.
2. **For the fixed rule it helps a little and costs nothing.** P-CACS loses 14% of its label repeats at θ 0.80 with no change in recall.
3. **For the online learner, a loose threshold hurts.** At θ 0.80 the inherited state changes which candidates look new, the learner drifts onto a different trajectory, and recall falls by 0.06. θ 0.85 is neutral.
4. **The October IL model's repeats aren't a memory problem.** 28.5% of its cues name an object it already announced, and re-ID doesn't change that. It re-announces the same live track. That needs fixing in the scheduler (M3).
5. **Long-horizon memory needs long recordings.** The case it's built for (a bench passed on the way out, then again on the way back) only appears in continuous walks. That makes it a measurement for the New Haven recordings (M5), not for EgoBlind.

## Still to do

- Hand-check 50 sampled matches (`results/memory/matches_dev.csv`), with crops shown side by side.
- Choose θ and the horizon on `select`; current lean: θ 0.85 for learners.
- Re-run on own long recordings once they exist.
