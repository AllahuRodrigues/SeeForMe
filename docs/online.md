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
| Full information, zero delay (not causal) | 0.525 | 0.500 | 0.513 | 0.512 | 0.10 |
| Full information, labels held to video end (causal) | 0.200 | 0.206 | 0.194 | **0.200** | 0.12 |
| Full information, labels held 2 s | 0.362 | 0.388 | 0.388 | 0.379 | 0.09 |
| No learning (lr = 0, weights stay 0) | 0.138 | 0.138 | 0.138 | 0.138 | 0.13 |

The full-information rows still also receive the one-sided taps and questions. Without a delay its labels for time t are
applied before the decision at t + 0.5 s, which scores nearly the same candidate in the same event window, so 0.51 is a
zero-delay reference and not a causal upper bound (`seeforme/online/policy.py`, `full_info_delay`).

## What this says

1. **With a causal reference there is no measurable cost of one-sided feedback on this benchmark.** Holding the full-information labels until each video ends gives 0.200 pooled over three seeds against 0.204 for one-sided greedy (paired video bootstrap, causal minus greedy: -0.004 [-0.054, +0.048]). The earlier 0.53 came from zero-delay labels. Delay matters a great deal within a video (2 s: 0.379), so fast feedback inside a clip is worth something, but labels that arrive after the clip are worth no more than taps and questions. Learning does help: greedy beats the no-learning floor by +0.067 [+0.010, +0.121]. Because every question in the data is asked (p_ask = 1), every scored outcome is eventually revealed, so on the scored region the problem is close to delayed full information.
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
6. **A weights-only prior is safe but doesn't close the gap.** It removes the harm. Thompson rises to 0.206, and on `finetune` it catches 6–8 of 27 events against 4 for the frozen IL model. Every interval still includes zero. Against the causal full-information reference (0.200) there is no gap left for a warm start to close.

## Implicit negatives (a candidate nobody asked about)

When a remembered candidate leaves the 15 s lookback without any feedback covering it, it becomes a negative of weight w. Stream: `train`, `finetune`, then `select`. `select` holds 11 coverable events per seed, so each `select` cell below is pooled over three seeds on the same 11 events (33 seed-events). Treat `select` differences as indicative only.

| w | greedy dev recall | greedy `select` | Thompson dev recall | Thompson `select` |
|---|---|---|---|---|
| 0 (off) | **0.204** | **9/33** | **0.188** | **12/33** |
| 0.02 | 0.173 | 9/33 | 0.183 | 8/33 |
| 0.05 | 0.169 | 9/33 | 0.175 | 7/33 |
| 0.1 | 0.169 | 7/33 | 0.175 | 7/33 |
| 0.3 | 0.154 | 5/33 | 0.160 | 5/33 |

7. **Silence-as-negative makes things worse, and more weight makes them worse still.** Quiet-window violations don't improve consistently either. Selected on `select`: w = 0, so the feature stays off. Most candidates nobody asks about sit next to the ones that matter (same label, same moment), so the negatives push down exactly the features the rare positives rely on.

## Causal full-information reference and the p_ask sweep (added 2026-11-12, evening)

Commands (from `scripts/`, `OMP_NUM_THREADS=2`):
`run_online.py --stream train,finetune,select --policies online_greedy,online_greedy+lr0,online_full_info,online_full_info+causal,online_full_info+d2 --seed S --tag causal_sS`
and `run_online.py --stream train,finetune --policies online_greedy,online_full_info+causal --p-ask P --seed S --tag paskP_sS`.
Pooled over seeds 0, 1, 2. Dev = `train` + `finetune` (160 coverable events per seed); `select` = 11 per seed.

| Policy | dev recall | dev quiet viol. | `select` recall | `select` quiet viol. | cues/min |
|---|---|---|---|---|---|
| Greedy (one-sided) | 98/480 = 0.204 | 0.115 | 9/33 = 0.273 | 0.148 | 1.67 |
| No learning (lr = 0) | 66/480 = 0.138 | 0.129 | 3/33 = 0.091 | 0.056 | 1.68 |
| Full information, zero delay | 246/480 = 0.512 | 0.096 | 17/33 = 0.515 | 0.056 | 1.66 |
| Full information, held to video end | 96/480 = 0.200 | 0.116 | 6/33 = 0.182 | 0.037 | 1.67 |
| Full information, held 2 s | 182/480 = 0.379 | 0.091 | 15/33 = 0.455 | 0.111 | 1.67 |

Paired video bootstrap (2,000 resamples, seeds pooled), causal full information minus greedy: dev -0.004 [-0.054, +0.048];
`select` -0.091 [-0.267, +0.067]. The pre-registered falsifier for "a real one-sided gap exists" (causal full information
ahead by more than 0.08 on `select` with the interval above 0) is not met.

Users who do not always ask (questions dropped with probability 1 - p_ask, dev only, for diagnosis):

| p_ask | greedy recall (s0 / s1 / s2, mean) | causal full information (mean) | greedy missed-question rows (s0) |
|---|---|---|---|
| 1 | 0.188 / 0.212 / 0.212, 0.204 | 0.200 | 3,055 |
| 0.5 | 0.194 / 0.138 / 0.169, 0.167 | 0.186 | 1,779 |
| 0.25 | 0.194 / 0.156 / 0.188, 0.179 | 0.179 | 688 |

Halving the questions lowers greedy's mean by 0.037 and quartering them by 0.025, both inside the seed spread (0.138 to 0.212).
The falsifier for item 2 of the algorithm plan (greedy at p_ask 0.25 within 0.03 of p_ask 1) is met, so on this
benchmark the feedback channel is not the bottleneck; the features and the amount of labelled data are.

## Label-delay curve (added 2026-11-12, night)

Full-information labels delivered d seconds after each decision; everything else as above. Dev stream (`train` + `finetune`), 160 coverable events per seed, three seeds. Outputs: `per_video_delay_s*.csv` (1, 4, 8 s) and `per_video_causal_s*.csv` (0 s, 2 s, clip end, greedy, floor).

| Label delay | seed 0 / 1 / 2 | pooled recall | quiet violations |
|---|---|---|---|
| 0 s | 0.525 / 0.500 / 0.512 | 0.512 | 0.096 |
| 1 s | 0.469 / 0.456 / 0.463 | 0.463 | 0.093 |
| 2 s | 0.362 / 0.388 / 0.388 | 0.379 | 0.091 |
| 4 s | 0.338 / 0.325 / 0.338 | 0.333 | 0.109 |
| 8 s | 0.244 / 0.225 / 0.244 | 0.237 | 0.105 |
| clip end | 0.200 / 0.206 / 0.194 | 0.200 | 0.116 |
| one-sided greedy (taps and questions) | 0.188 / 0.212 / 0.212 | 0.204 | 0.115 |
| no learning (floor) | 0.138 / 0.138 / 0.138 | 0.138 | 0.129 |

8. **Recall falls steadily with label delay, and by the end of the clip full labels are worth no more than taps and questions.** Events last a few seconds, so a label that arrives within a second or two mostly teaches the learner about the *same* event while it is still happening; that is adaptation within an event, which no real user can supply before the event. What transfers across videos is worth about 0.06 over the floor (0.20 vs 0.14). The open problem is therefore cross-event generalisation: features that recognise a should-speak moment in advance. That, not the feedback channel, is what the next experiments target.

**Recall over all events.** Coverable recall counts only the 160 of 293 dev events (55%) the detector can see. Counted over all weakly labelled events, with uncoverable events as misses, greedy catches 98 of 879 seed-events = **0.111**, the no-learning floor 0.075 and causal full information 0.109. That is far from what a dependable navigation aid needs; perception and anticipation matter more than the feedback channel.

**All feedback is simulated or replayed.** Taps are simulated (p_tap 0.5, oracle usefulness) and questions are replayed from EgoBlind's annotation, most of which was written after recording. These are simulated-feedback replay experiments; no BLV user has interacted with the learner.

## Anticipation feature gate: looming time to contact (added 2026-11-12, night)

Box-height growth gives time to contact without depth: h = fH/Z, so 1 / d(log h)/dt = Z / (-dZ/dt) (CPSC 480 lecture 3, magnification inversely proportional to depth). `Track.looming_ttc()`, gated with `scripts/feature_auc.py` (logistic on `train` candidates, AUC on `finetune`, paired bootstrap over videos; pass = +0.02 with the interval above 0).

| Features | AUC train | AUC finetune | Δ vs base [95% CI] | Gate |
|---|---|---|---|---|
| base 29 | 0.728 | 0.675 | | |
| + per-track looming | 0.729 | 0.675 | +0.001 [-0.003, +0.004] | fail |
| + decision-level looming | 0.732 | 0.683 | +0.008 [-0.004, +0.021] | fail |
| + both | 0.733 | 0.684 | +0.009 [-0.004, +0.023] | fail |

9. **Looming alone does not pass.** Decision-level looming separates somewhat on its own (single-feature AUC 0.60) but adds little beyond the existing box-growth and depth features. Next in line: camera ego-motion from corner matching and RANSAC (`docs/research/lectures.md`, idea 2), which also works when the detector sees nothing.

## Human label audit (added 2026-11-12, night)

Blind mixed audit (`scripts/review_events.py audit`): 30 random should-speak events and 20 user-confirmed quiet moments from `train` + `finetune`, shuffled, with the derived label and the annotators' answers hidden. Reviewer ar, median 2.7 s per item.

| Hidden label | speak | silent | unclear |
|---|---|---|---|
| should-speak (30) | 29 | 0 | 1 |
| user-confirmed quiet (20) | 20 | 0 | 0 |

**Second audit pass, rubric v2** (reviewer ar2, same 50 blind items, median 2.6 s per item): should-speak events marked warn 27 of 28 decided (0.96); user-confirmed quiet moments marked confirm-clear 6 of 19 decided (0.32), warn 13; overall agreement with the derived labels 33 of 47 (0.70) against 0.60 for always answering warn. Separating "warn" from "confirm the way is clear" helped (quiet moments recognised 0 of 20 -> 6 of 19), but most quiet moments are still marked as warnings. Some may genuinely contain something worth mentioning (the derived quiet label only records that the answerer said no), which is why an independent second rater, not either pass, decides between the readings.

10. **The first human verdicts did not discriminate**, so they cannot be cited as validating the weak labels (train 237/237 and finetune 55/56 speak were entered in bulk or fast passes). Two readings remain open: the passes defaulted to "speak", or the reviewer judges that even a "clear" moment deserves an unprompted cue ("clear to cross"), which would challenge treating those windows as quiet. Settling it needs a sharper rubric (separate "warn about an object" from "confirm all clear") and an independent second rater. Until then the weak labels stay weak.

## Wider perception (added 2026-11-12, night)

Why events are uncoverable (train + finetune, 293 events, 133 uncoverable): 64 name an object the COCO detector has no class for (door 25, pole 9, stairs 9, fence 9, wall 5, ...), 34 ask about the scene with no single object ("is the path clear?", "which direction?"), 31 are filtered taxonomy (decor), 4 truncated caches.

**Open-vocabulary cache (YOLO-World, 41-term navigation vocabulary), 3 seeds, dev:** coverable events rise from 160 to 232 per seed (79% of 293), but all-event recall does not: greedy 77/879 = 0.088 (COCO 0.111), Thompson 99/879 = 0.113 (COCO 0.102), floor 0.075. More detections bring as much noise as signal.

**OWLv2 probe** (`scripts/probe_openvocab.py`, google/owlv2-base-patch16-ensemble, Apache-2.0) on the 64 not-detectable events, review frames inside each window: fires on 46/64 (doors 14/24, stairs 7/9, poles 9/9, fences 6/6). Hand check of 11 hits: 7 correct object, 2 marginal (full-frame elevator, distant door), 2 wrong object (a distant pole instead of the one beside the user; a meaningless "steps" box). About 1.7 s per frame on MPS, so a full 5 fps cache is impractical; it is a teacher, not a runtime detector.

11. **Seeing more is not enough; the scheduler must know which detections matter.** Next: OWLv2 as a teacher for a fast student detector trained on human-verified boxes, and dense scene segmentation (doors, stairs, walls, railings, poles, free floor and sidewalk ahead) for both the architectural classes and the 34 scene-level questions.

**Scene segmentation probe** (`scripts/probe_segmentation.py`, SegFormer-B2 on ADE20K, research-only weights). A concept is "seen" if its classes cover at least 2% of the path region in a window frame; the base rate is the same test on events not about that concept.

| Concept | Seen on its not-detectable events | Base rate elsewhere |
|---|---|---|
| door | 7/24 (29%) | 9% |
| stairs | 5/9 (56%) | 10% |
| fence / railing | 5/6 (83%) | 22% |
| pole | 1/9 (11%) | 14% |
| wall | 4/4 | 72% |

Walkable share of the path region (floor, sidewalk, road, path): median 0.26 for the 34 "is the path clear?" events, 0.46 for other speak events, 0.22 for the 20 user-confirmed quiet moments.

12. **Segmentation is informative for stairs (5.6x base rate), fences and doors (3x), useless for thin poles and for walls**, and the walkable share does not separate "is the path clear?" moments from confirmed-clear ones, so free space alone will not answer scene-level questions. Next: per-second stairs / door / railing shares as decision features through the feature gate.

## Failure funnel and speech-budget curve (added 2026-11-13)

`scripts/failure_funnel.py` attributes each of the 293 dev events (seed 0) to the first stage that lost it (`results/funnel/dev.md`).

| Policy | Events | Detected | Tracked | Candidate | Spoken about | On time |
|---|---|---|---|---|---|---|
| P-CACS | 293 | 169 | 160 | 160 | 23 | 15 |
| October IL (in-sample on train) | 293 | 169 | 160 | 154 | 66 | 57 |
| Online greedy | 293 | 169 | 160 | 154 | 56 | 33 |

Perception loses 124 events for every policy (no object named 34, door 22, pole 9, stairs 8). The scheduler loses 88-137 events it was offered, none because the channel was busy, and the event was the top-ranked legal candidate in 70-103 of them (outranked only 5-8 times).

Speech-budget sweep, seed 0, dev (`per_video_budget*_s0.csv`), all-event recall and quiet-window violations:

| Cues/min | Greedy recall | Thompson recall | Quiet violations |
|---|---|---|---|
| 0.9 | 0.078 | 0.072 | 0.05 |
| 1.7 | 0.102 | 0.123 | 0.09-0.11 |
| 2.5 | 0.140 | 0.160 | 0.13-0.16 |
| 3.4 | 0.143 | 0.167 | 0.19-0.22 |
| 5.0 | 0.177 | 0.208 | 0.33 |

13. **The scheduler's misses are a scoring problem, not a ranking or budget problem.** Raising the speech rate buys recall roughly in proportion to quiet-window violations, with hits per 100 cues falling from about 9 to 4. The learner's scores do not separate should-speak moments from other top-ranked moments, so recall has to come from better decision features and calibration, compared at equal speech rate, not from talking more.

**Segmentation feature gate** (`feature_auc.py --seg`, 558 dev videos segmented at 1 fps, 20,991 frames, causal features: last sample at or before t and the 3 s maximum):

| Features | AUC train | AUC finetune | Δ [95% CI] | Gate |
|---|---|---|---|---|
| base 29 | 0.728 | 0.675 | | |
| + all segmentation shares | 0.757 | 0.655 | -0.019 [-0.059, +0.018] | fail |
| + stairs, door, railing, walkable | 0.737 | 0.654 | -0.020 [-0.050, +0.000] | fail |

14. **As a decision feature among existing candidates, segmentation overfits and fails** (no single share above 0.59 AUC on finetune). The gate only scores detector candidates, so the 124 events lost to perception never enter it: it cannot measure segmentation's actual strength, adding undetected doors and stairs. That needs a separate test: segmentation pseudo-candidates, measured through the failure funnel (detected events and on-time events) at equal speech rate.

## Exact silence reasons and segmentation as an event generator (added 2026-11-13)

**Silence reasons.** Each policy now logs, per candidate and decision, the rule that decided it (`last_reasons`). Over the in-window opportunities of every scheduler-missed event, the main reason was `below_threshold` for 133 of 137 P-CACS misses (4 low confidence), 88 of 88 IL misses and 94-98 of 96-98 online greedy misses (2 channel busy). Budget, guard and recent-announcement rules never decided a missed event.

**Segmentation candidates** (`seeforme/perception/seg_candidates.py`; stairs / door / railing share >= 2% of the path region for 2 consecutive seconds, causal, linked by label and concepts). Funnel `results/funnel/dev_segcand.md` vs `dev_reasons.md`:

| Policy | Detected | On time | Δ on time |
|---|---|---|---|
| P-CACS | 169 -> 177 | 15 -> 21 | +6 |
| October IL | 169 -> 177 | 57 -> 57 | 0 |
| Online greedy | 169 -> 177 | 33 -> 34 | +1 |

Cost over the 17,602 dev video-seconds: 4.5 candidate-seconds per minute (fence 2.0, door 1.5, stairs 1.0), of which 6%, 3% and 13% fall on a labelled event.

15. **The rule that silences missed events is the score threshold, and nothing else.** Combined with the budget curve, scheduler misses are a scoring problem.
16. **Segmentation as a generator recovers 8 of 124 perception losses and is not promoted.** It adds 4.5 candidate-seconds a minute at 3-13% event precision and does not help the learned schedulers. The 34 scene-level events and most doors (19 of 22 still lost) remain out of reach; a stairs-only generator (best precision) or OWLv2-taught door boxes are the remaining perception options.

## Speak-or-silence model ladder (added 2026-11-13)

`scripts/train_decision_models.py`: models fitted on `train` candidates (full-information labels), causal K-step history windows, standardised with train statistics; every hyperparameter, epoch, K and calibrator chosen on `finetune`, so finetune numbers are a development comparison. Frozen scorers replayed at equal speech rate (BudgetController, 2 cues/min) on the 104 finetune videos (`results/decision_models/`).

| Policy | Finetune AUC | All-event recall | vs logistic [95% CI] | Quiet violations | Cues/min |
|---|---|---|---|---|---|
| logistic (last step) | 0.682 | 27/168 = 0.161 | reference | 0.158 | 1.72 |
| logistic, class-weighted + Platt | 0.680 | 0.161 | +0.000 | 0.132 | 1.72 |
| MLP | 0.674 | 0.161 | +0.000 | 0.105 | 1.70 |
| **GRU, 4 s history** | **0.697** | **33/168 = 0.196** | **+0.036 [-0.031, +0.094]** | **0.053** | 1.70 |
| TCN, 1 s | 0.669 | 0.161 | +0.000 | 0.184 | 1.69 |
| online greedy (learning online) | | 13/168 = 0.077 | **-0.083 [-0.172, -0.011]** | 0.132 | 1.72 |

17. **A scorer trained offline on full labels beats learning online from one-sided feedback** at equal speech rate (frozen logistic vs online greedy, CI excludes 0). On this benchmark, supervised training on the train split is worth more than adapting online.
18. **The GRU with 4 s of history is the lead candidate, not yet a result:** highest recall (+0.036, CI includes 0) and fewer quiet-window violations (-0.105 [-0.229, -0.023], on 38 quiet windows), but K was chosen on the same videos. The networks overfit (train AUC 0.73-0.77, finetune flat), cross-fitted calibration is poor (ECE about 0.21), and the TCN loses at every K. Next: confirm GRU vs logistic once on `select`.

## One-shot confirmation on `select` (added 2026-11-13)

`scripts/confirm_decision_model.py`: frozen models from the finetune comparison, no refitting, scored once on the 34 `select` videos (27 events, 11 coverable; three feedback seeds pooled, frozen models identical across seeds).

| Policy | All-event recall | vs logistic [95% CI] | Coverable recall | Quiet violations | vs logistic [95% CI] | Cues/min |
|---|---|---|---|---|---|---|
| logistic | 9/81 = 0.111 | reference | 0.273 | 0.056 | reference | 1.68 |
| GRU, 4 s history | 6/81 = 0.074 | -0.037 [-0.130, +0.000] | 0.182 | 0.000 | -0.056 [-0.214, +0.000] | 1.70 |
| online greedy | 9/81 = 0.111 | +0.000 [-0.042, +0.032] | 0.273 | 0.148 | +0.093 [+0.000, +0.182] | 1.68 |

19. **The GRU lead does not confirm, so nothing is promoted.** On `select` the GRU catches fewer events (its quiet-window advantage remains but is not significant), and online greedy ties the frozen logistic on recall with more quiet-window violations. With 27 events, `select` can only rule out large effects; what is supported across both sets is that no candidate scorer beats a frozen logistic, and that the remaining lever is the information the scorer sees, not its architecture.

## Permissive detectors for the product track (added 2026-11-13, issue 10)

`scripts/compare_detectors.py`: the same review frames inside each `train` / `finetune` event window; an event is seen if a detector outputs one of its labels at confidence >= 0.35. Hugging Face COCO names normalised (motorbike -> motorcycle, diningtable -> dining table, ...); all detectors scored on the same 195 events.

| Detector | Licence | Events seen | Detections / frame | ms / frame (M1 Max) |
|---|---|---|---|---|
| YOLO11 (current) | AGPL-3.0 | 163 / 195 = 0.836 | 3.9 | 27 |
| RT-DETR R18 | Apache-2.0 | 176 / 195 = 0.903 | 7.7 | 270 |
| **D-FINE S** | **Apache-2.0** | **181 / 195 = 0.928** | 14.6 | 285 |

Gains come from tables (3 -> 13 of 16), cars, trucks, scissors. 20. **The product track can drop the AGPL detector without losing coverage**: D-FINE S sees more events on identical frames. Costs to resolve before switching: about 10x the latency in unoptimised PyTorch (YOLO11 runs through Ultralytics' optimised path; an ONNX / Core ML export is the fair speed test) and 2-4x more detections per frame, which the scheduler must absorb (replay check at equal speech rate). Doors, stairs and poles remain outside every COCO detector (OWLv2 teacher and segmentation results above).

## CARE scheduler, experiment E2: decision rule only (added 2026-11-13)

`scripts/compare_care.py`. Same candidates and the same frozen logistic scores; the threshold policy (adaptive tau, 2 cues/min) against CARE (utility = p_helpful - cost * duration, silence = 0) under a strict speech ledger refilled at the threshold policy's measured train rate (3.83 s/min), on one continuous stream clock. Speech cost chosen on `train`; reported on `finetune` (development).

| Policy | All-event recall | Quiet violations | Speech s/min | Recall vs threshold [95% CI] |
|---|---|---|---|---|
| threshold | 9/56 = 0.161 | 0.158 | 3.80 | reference |
| CARE, cost 0 (chosen on train) | 7/56 = 0.125 | 0.237 | 3.80 | -0.036 [-0.123, +0.050] |
| CARE, cost 0.1 | 11/56 = 0.196 | 0.211 | 3.79 | +0.036 [-0.036, +0.098] |
| CARE, cost 0.2 | 10/56 = 0.179 | 0.158 | 2.29 | +0.018 [-0.083, +0.128] |

A first run gave each clip a fresh ledger burst and was not budget-matched (CARE spoke 9-12 s/min); fixed and re-run with a regression test.

Audio lifecycle check (CARE, finetune replay, 104 videos): reserved 233.3 s, played 228.9 s; playback never exceeded the reservation in any video. The replay channel uses the same duration estimate as the ledger, so this is partly circular: in the live app the ledger must reconcile against measured TTS length and interruptions, and reset only at a real session boundary. Rubric v3 has no human verdicts yet; migrated v1/v2 verdicts are not treated as validated v3 labels.

With constant value and duration, CARE's rule reduces to p_helpful > cost * duration / value, another probability threshold over the same information, so no recall gain is expected until the usefulness estimate improves.

21. **On identical scores, the decision rule alone does not change recall at a matched speech budget.** CARE's current value is the guarantee (spoken seconds bounded over every interval, every silence has a logged reason), not recall. Consistent with items 13-19: the scores do not separate should-speak moments, so gains must come from better supervision (rubric v3) and decision inputs, then CARE's extra heads (claim reliability, redundancy).

## E3, first rows: scene context and a nonlinear learner on the current labels (added 2026-11-13)

`feature_auc.py --seg --ego` with logistic and with histogram gradient boosting (`--learner hgb`, early stopping on an internal train split). Same candidates, same weak labels, train -> finetune AUC, paired video bootstrap.

| Learner / features | AUC train | AUC finetune | Δ vs same learner's base [95% CI] |
|---|---|---|---|
| logistic, base 29 | 0.728 | **0.675** | |
| logistic + ego-motion | 0.758 | 0.694 | +0.020 [-0.024, +0.066] |
| logistic + all context (looming, segmentation, ego-motion) | 0.774 | 0.689 | +0.015 [-0.026, +0.057] |
| boosting, base 29 | 0.934 | 0.632 | |
| boosting + all context | 0.982 | 0.632 | -0.000 [-0.108, +0.097] |

Ego-motion (`seeforme/perception/egomotion.py`; Shi-Tomasi, LK with forward-backward check, RANSAC homography, 2 fps, 41,148 frame pairs, 18.5 ms per pair plus decode): 44.1% of pairs unreliable (coverage < 0.4 in 40.7%), the largest single-group gain so far but not significant.

22. **On the current weak labels, neither scene context nor a nonlinear learner separates should-speak moments better than the base logistic.** Boosting memorises train (AUC up to 0.98) and does worse on finetune than logistic; context adds at most +0.02 with intervals including 0. With a few hundred independent events, the binding constraint is now the supervision: E3's decisive row (corrected rubric v3 labels with independent agreement) needs human annotation before any further model or RL work.

## Coverage-first diagnostic: what can the current candidates deliver at all? (added 2026-11-13)

`scripts/compare_coverage.py`; full tables in `results/coverage/dev.md`. The coverage-first policy (`seeforme/scheduling/coverage_policy.py`) has no learned score: every candidate above the confidence floor is confirmed after 2 decisions, queued once and announced first-in first-out when the channel is free and the strict ledger can pay. It is a measuring instrument and must never be a product default. Budget x1 is the threshold rule's own speech rate on `train` (3.83 spoken s/min). `finetune`: 104 videos, 20 source recordings, 56 events, 27 of them with a candidate inside the window.

| Policy on `finetune` | events on time | of the 27 with a candidate | quiet moments spoken over | speech s/min | Δ events on time vs threshold [95%, by recording] |
|---|---|---|---|---|---|
| threshold (frozen logistic, production) | 9/56 = 0.161 | 9/27 | 16% | 3.80 | reference |
| coverage-first, budget x1 | 9/56 = 0.161 | 9/27 | 16% | 3.82 | +0.000 [-0.102, +0.108] |
| coverage-first, budget x2 | 11/56 = 0.196 | 13/27 | 29% | 7.53 | +0.036 [-0.091, +0.146] |
| coverage-first, budget x4 | 16/56 = 0.286 | 21/27 | 50% | 14.60 | +0.125 [+0.067, +0.179] |
| coverage-first, no budget | 11/56 = 0.196 | 21/27 | 87% | 37.68 | +0.036 [-0.091, +0.146] |

On train + finetune (293 events, 154 with a candidate inside the window): threshold 42/293 (in-sample on train), coverage-first x1 39/293, x4 84/293, no budget 86/293 by the benchmark's recall and 117/154 by the funnel. Misses among events with a candidate: threshold `below_threshold` 107 of 111; coverage-first x1 `budget` 78, `unconfirmed` 32; no budget `unconfirmed` 23, `channel_busy` 8, announced before the window 6. The two recall columns differ because the benchmark credits only the first cue about an object, while the funnel credits any cue about it inside the window.

23. **At the production speech budget the learned score selects no better than first-in first-out.** Same events on time (9/56), same false alerts (16%), same seconds of speech; the interval is wide (20 recordings), so this is "no measurable benefit", not "proven equal". It agrees with items 13 to 22: the score does not separate should-speak moments.
24. **The candidates set a ceiling near 0.40 of all events, and reaching it by talking more is not usable.** With no budget at all the assistant delivers 117 of the 154 events that have a candidate (0.40 of all 293 by the funnel, 0.29 by the benchmark's recall) while speaking 38 s per minute and over 87% of confirmed-clear moments. Four times the budget buys +0.125 recall for 50% false alerts. So two gaps remain, in this order of size: 139 of 293 events never reach the scheduler in time (perception), and at a fixed budget only 9 of 27 reachable events are chosen where 21 of 27 are reachable (selection). Selection needs supervision that says which candidates matter; a higher budget is not a substitute.

## Next experiments (selected on `select`, never on holdout)

- **Credit assignment for question labels:** expected to be null. Causal full information already has perfect per-candidate credit and only reaches 0.200; a hindsight relabelling prototype gave 0.156 and 0.175 on seed 0.
- **Decision-level "when" features** (change in ego speed, time since last cue, new tracks): check offline AUC on `train` to `finetune` first (baseline 0.683).
- **Re-identification memory (M2) on and off:** measure repeat cues, once the GPU embedding run has finished.
- **Bootstrap 95% intervals** (`scripts/bootstrap_ci.py`) on the final comparison.
