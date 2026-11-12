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
4. **The October IL model's repeats aren't a memory problem, and most aren't the same object.** 28.5% of its cues (65/228) name a weak event it already announced, and re-ID doesn't change that. But EgoBlind links every detection of the event's label inside its window to the event, so that count includes cues about a different person or car. Only 31/228 cues (13.6%) re-announce the same tracker track; 18 of those are the same event. See "Issue 6" below.

5. **Long-horizon memory needs long recordings.** The case it's built for (a bench passed on the way out, then again on the way back) only appears in continuous walks. That makes it a measurement for the New Haven recordings (M5), not for EgoBlind.

## Issue 6: same-track re-announcements by the IL scheduler (2026-11-12)

**Cause.** `SchedulerPolicy` (`seeforme/ml/scheduler_model.py`) blocks a track only for `recent_block_s` (about 2.8 s). P-CACS also multiplies its score by a novelty term that decays over 10 s, but the learned model does not reproduce it, so a track it liked becomes legal again after 2.8 s and is spoken at the first decision after the block (11 of the 18 same-event, same-track repeats come exactly 3.0 s later).

**Fix (opt-in, default unchanged).** `SchedulerPolicy(reannounce_guard_s=...)`, or the policy name `egoblind_learned:il+guard10`, makes speaking about a track illegal within that many seconds of its last announcement unless it has come at least `reannounce_move_m` (3 m) closer, the same exception P-CACS makes. `eval_reid.py` now reports `repeat_track` next to `repeat_gt`.

`eval_reid.py --policies egoblind_learned:il,egoblind_learned:il+guard10 --horizons 0 --tag il_fix` (dev, re-ID off); paired video bootstrap, guard minus off:

| | IL | IL + 10 s guard | difference [95% CI] |
|---|---|---|---|
| Coverable recall | 0.331 | 0.331 | +0.000 [+0.000, +0.000] |
| `finetune` (out of sample) | 4/27 | 4/27 | |
| Cues / min | 0.79 | 0.74 | -0.048 [-0.084, -0.018] |
| Same-track repeat share | 0.136 | 0.079 | -0.057 [-0.093, -0.023] |
| Same-event repeat share (`repeat_gt`) | 0.285 | 0.262 | -0.023 [-0.042, -0.006] |

The guard removes 14 cues, all of them same-track repeats, at no cost in recall. The 17 that remain pass the approach exception, mostly on far-range depth estimates (for example 29 m to 23 m), where a fixed 3 m rule is weak. The guard length (10 s, P-CACS's `reannounce_tau_s`) was not tuned.

## Still to do

- Hand-check 50 sampled matches (`results/memory/matches_dev.csv`), with crops shown side by side.
- Choose θ and the horizon on `select`; current lean: θ 0.85 for learners.
- Re-run on own long recordings once they exist.
