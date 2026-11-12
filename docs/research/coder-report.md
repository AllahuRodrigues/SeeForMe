# Coder report: SeeForMe research/next, 2026-11-12

All work is in `<repo>` on branch `research/next`. I ran no git commands that change state. I did not touch web/, server/, seeforme/live/, scripts/run_live.py or seeforme/pipeline.py, and I never opened holdout or test.

Short version:
- **(A) Causal full-information reference:** once its labels are held to video end, it does no better than one-sided greedy. The 0.51 "upper bound" measured zero-delay labels.
- **(B) Issue 6:** the 28.5% figure mostly counts cues about different objects that share a label. True same-track repeats are 13.6% of cues. A 10 s guard, off by default, brings that down to 7.9% with no change in recall.
- **(C) p_ask sweep:** having users ask only half or a quarter of the questions barely moves greedy.

Full suite: **121 passed, 2 skipped**, 176 s (`.venv/bin/python -m pytest -q --timeout 600 tests/`).

## (A) Causal full-information reference and a no-learning floor (algorithm item 1)

**Code changes:**
- `seeforme/online/policy.py`
  - New `OnlinePolicy(full_info_delay=None | "video" | seconds)`. The default `None` behaves exactly as before (zero delay).
  - `_learn_full_info` now puts labels into a buffer, `_pending`, unless the delay is `None`.
  - New `_flush_pending(now)`, called at the start of `decide` and in `reset()` (video end).
  - New `n_pending` property. Bad delay values raise ValueError.
  - The docstring now says that zero delay is not a causal reference. In every variant the full-information learner also receives the one-sided taps and questions.
- `scripts/run_online.py`: policy names now take suffixes: `online_<mode>[+campus|+campusw][+causal|+d<sec>][+lr0]`.

**Tests added** (`tests/test_online_policy.py`), all pass:
- `test_full_info_default_learns_immediately`
- `test_causal_full_info_holds_labels_until_video_end`
- `test_full_info_fixed_delay_releases_after_d_seconds`
- `test_full_info_delay_validation`
- `test_causal_full_info_equals_greedy_within_the_first_video`: inside video 1, causal full info makes exactly the same cues as greedy.

**Command** (from `scripts/`, `OMP_NUM_THREADS=2`), for S = 0, 1, 2:
`../.venv/bin/python run_online.py --stream train,finetune,select --policies online_greedy,online_greedy+lr0,online_full_info,online_full_info+causal,online_full_info+d2 --seed S --tag causal_sS`

Greedy reproduces the existing `per_video_dev_s*` and `per_video_ineg0_s*` numbers exactly: dev 30, 34 and 34 out of 160; select 4, 2 and 3 out of 11.

Results are pooled over the 3 seeds. Dev = train + finetune, 160 coverable events per seed. Select = 11 coverable events per seed.

| Policy | dev s0 / s1 / s2 | dev pooled | dev quiet viol. | select pooled | select quiet viol. | cues/min dev |
|---|---|---|---|---|---|---|
| greedy | 0.188 / 0.212 / 0.212 | 98/480 = 0.204 | 0.115 | 9/33 = 0.273 | 0.148 | 1.67 |
| greedy+lr0 (floor) | 0.138 / 0.138 / 0.138 | 66/480 = 0.138 | 0.129 | 3/33 = 0.091 | 0.056 | 1.68 |
| full_info, zero delay | 0.525 / 0.500 / 0.512 | 246/480 = 0.512 | 0.096 | 17/33 = 0.515 | 0.056 | 1.66 |
| full_info+causal (labels at video end) | 0.200 / 0.206 / 0.194 | 96/480 = 0.200 | 0.116 | 6/33 = 0.182 | 0.037 | 1.67 |
| full_info+d2 (labels 2 s late) | 0.362 / 0.388 / 0.388 | 182/480 = 0.379 | 0.091 | 15/33 = 0.455 | 0.111 | 1.67 |

Cues/min on select is 1.68 to 1.71 for every policy.

Paired video bootstrap (2,000 resamples, seeds pooled per video):

| Comparison | dev | select |
|---|---|---|
| causal minus greedy | -0.004 [-0.054, +0.048] | -0.091 [-0.267, +0.067] |
| zero-delay minus greedy | +0.308 [+0.244, +0.371] | not computed |
| greedy minus lr0 | +0.067 [+0.010, +0.121] | +0.182 [-0.061, +0.429] |
| d2 minus greedy | +0.175 [+0.112, +0.239] | not computed |

**What this means:**
- The algorithm expert's numbers are confirmed. With labels arriving after the video, full information ties one-sided greedy.
- The falsifier for item 1 (causal ahead by more than 0.08 on select, with the interval above 0) is not met.
- Learning does beat no learning on dev.
- Within-video delay matters a lot: a 2 s delay gives 0.379.
- Select has only 11 events per seed, so select comparisons are indicative only.

## (B) Issue 6: the IL scheduler's repeat cues

**Diagnosis** (dev stream, re-ID off; scratch scripts in `<scratch>`):
- I reproduced the issue's number: `repeat_gt` = 65/228 = 0.285.
- The metric overcounts. `PerceptionCache.frames` (`seeforme/egoblind/cache.py`) links every detection whose label matches the event, inside the event's window, to that event. So a cue about a *different* person or car counts as a repeat.
- Of the 65 "repeats":
  - 18 are the same track.
  - 26 are another track that was still alive at the time. Its box mostly does not overlap the earlier announced box (median IoU 0.09), so it is probably a different object.
  - 21 are another track that had already ended.
- True same-track repeats are 31/228 = 13.6% of cues.
- **Cause:** `SchedulerPolicy.decide` (`seeforme/ml/scheduler_model.py`) blocks a track only for `recent_block_s` (about 2.84 s). P-CACS also has a 10 s novelty decay, which the learned model does not reproduce. As a result, 11 of the 18 same-event, same-track repeats fire exactly 3.0 s later, at the first decision after the block. Track ids and `last_announced` updates are correct.

**Fix (off by default, so the October IL numbers are unchanged):**
- New `SchedulerPolicy(reannounce_guard_s=0.0)` and `_guarded()`. A track can't be spoken about again within the guard time unless it has come at least `reannounce_move_m` (3 m) closer, the same exception P-CACS uses.
- The guard is turned on through the name `egoblind_learned:il+guard10` (parsed in `scripts/egoblind_common.py::make_egoblind_policy`).
- New `seeforme/egoblind/metrics.py::repeat_counts`, which returns `repeat_gt`, `repeat_track` and `repeat_label`. `scripts/eval_reid.py` now uses it and reports a `repeat_track` share column.

**Tests:** new file `tests/test_scheduler_reannounce.py`, 7 tests, all pass:
- One documents the cause.
- Four cover the guard: it blocks; it releases after the guard time; the approach exception works; it also applies in RL mode.
- One checks the name suffix.
- One checks `repeat_counts`.

**Command:** `../.venv/bin/python eval_reid.py --policies egoblind_learned:il,egoblind_learned:il+guard10 --horizons 0 --tag il_fix`. Output: `results/memory/reid_il_fix.{md,csv,json}`.

Paired bootstrap, guard minus off:

| | IL | IL + 10 s guard | difference [95% CI] |
|---|---|---|---|
| Coverable recall | 0.331 | 0.331 | +0.000 [+0.000, +0.000] |
| finetune (out of sample) | 4/27 | 4/27 | |
| Cues/min | 0.786 | 0.738 | -0.048 [-0.084, -0.018] |
| Same-track repeat share | 0.136 | 0.079 | -0.057 [-0.093, -0.023] |
| `repeat_gt` share | 0.285 | 0.262 | -0.023 [-0.042, -0.006] |
| Quiet violations | 0.036 | 0.036 | |

- The guard removes 14 cues, and all of them are same-track repeats. It does not improve recall.
- The 17 same-track repeats that remain pass the approach exception, mostly because of noisy depth at far range (for example 29 m to 23 m). A fixed 3 m rule is weak there.
- The 10 s guard length was not tuned.
- The issue's wording, "re-announces the same live track in 28.5%", is wrong: the same-track figure is 13.6%.

## (C) p_ask: users who do not always ask (algorithm item 2)

**Code changes:**
- `FeedbackModel(p_ask=1.0)` in `seeforme/online/feedback.py`. Each question, about an event or a quiet window, is asked with probability p_ask. Questions use their own seeded random generator, so the tap stream does not change. The default is unchanged: greedy at p_ask 1 reproduces 0.188 on seed 0.
- `run_stream(p_ask=)` in `seeforme/online/stream.py`.
- `--p-ask` option in `run_online.py`.
- `OnlinePolicy.label_counts` (rows fitted per label kind) and `n_pos` are now saved in `to_dict`.

**Tests:** 4 in `tests/test_online_feedback.py` (default, zero, seeded and taps undisturbed, validation) and 2 in `tests/test_online_policy.py` (label counts are logged; `run_stream` passes p_ask). All pass.

**Command** (dev only, for diagnosis): `run_online.py --stream train,finetune --policies online_greedy,online_full_info+causal --p-ask P --seed S --tag paskP_sS`

| p_ask | greedy s0 / s1 / s2 (mean) | causal full info (mean) | quiet viol. (greedy) | greedy missed-question rows (s0) |
|---|---|---|---|---|
| 1 | 0.188 / 0.212 / 0.212 (0.204) | 0.200 | 0.115 | 3,055 (quiet_window 6,101; taps 22 useful, 75 not useful) |
| 0.5 | 0.194 / 0.138 / 0.169 (0.167) | 0.186 | 0.10 to 0.13 | 1,779 |
| 0.25 | 0.194 / 0.156 / 0.188 (0.179) | 0.179 | 0.10 to 0.11 | 688 |

Cues/min is 1.66 to 1.69 in every run.

- Greedy at p_ask 0.25 is 0.025 below p_ask 1. That meets item 2's falsifier (within 0.03): on this benchmark the feedback channel is not the bottleneck.
- The p_ask 0.5 mean is lower than the 0.25 mean, which shows how large the seed noise is.
- I did not implement the PU correction. Given this result it has little to recover.

## Audit follow-ups and anything left over

- **docs/online.md:**
  - Rewrote point 1 (no measurable one-sided cost against a causal reference) and point 6.
  - Replaced the "upper bound" table row with the zero-delay, causal, 2 s and lr0 rows.
  - Fixed the select count: 11 per seed, 33 pooled.
  - Added a section with the causal and p_ask tables and their commands, and updated "Next experiments".
- **docs/memory.md:** corrected the 28.5% statement and added an "Issue 6" section.
- **Stale summary file:** I regenerated `results/online/summary_dev.{md,json}` and the related per_video, curve and model files with the current defaults, seed 0. The apple row now reads 0.169 at 1.68 cues/min instead of the stale 0.281 at 3.96.
- **Not done:**
  - Removing the 0.525 / "60%" claim from README.md and any paper drafts. Those belong to the writer.
  - Telling the web-owning session about the audit's items: the web default is the CampusBench RL scheduler (`models/scheduler/scheduler_rl.pt`); OCR text goes into innerHTML unescaped at main.ts:312-315; styles.css:2 forces a white background in dark mode. I have no channel to that session.
  - The two items the audit says to raise with the user, not settle by guessing: `docs/ecc/ECC-OPERATING-MANUAL.md` has a home-directory path in a public repo, and the advisor naming differs between README:3 and update.md:4.
- **Nothing failed.** The new files are uncommitted for the user to review. `tests/test_scheduler_reannounce.py` is untracked, as are the new `results/online/*causal*` and `*pask*` files and `results/memory/reid_il_fix.*`.
