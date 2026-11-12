HALLUCINATION CHECK, FINAL PASS (2026-11-12)

Repo: <repo> (branch research/next). I ran only read-only git commands and wrote nothing to the repo. My scratch scripts are <scratch> and <scratch>

== (2) CODER REPORT AGAINST THE WORKING TREE ==

The working tree matches the report. `git diff --stat` shows 21 modified files: docs/online.md, docs/memory.md, results/online/summary_dev.*, per_video_dev.csv, curve_dev.csv, the model_*_dev.json files, scripts/{run_online,eval_reid,egoblind_common}.py, seeforme/online/{policy,feedback,stream}.py, seeforme/ml/scheduler_model.py, seeforme/egoblind/metrics.py and two test files. Untracked: tests/test_scheduler_reannounce.py, results/online/*causal*, *pask*, and results/memory/reid_il_fix.*. Nothing under web/, server/, seeforme/live/, scripts/run_live.py or seeforme/pipeline.py was touched. No data or video files are among the changes.

Claim by claim:
- CONFIRMED: `full_info_delay` (None / "video" / seconds), the `_pending` buffer, `_flush_pending` called at the start of decide() and in reset(), `n_pending`, and a ValueError on bad values (policy.py). The default None still fits immediately.
- CONFIRMED: run_online.py parses the suffixes +causal, +d<sec>, +lr0 and +campus/+campusw.
- CONFIRMED: `FeedbackModel(p_ask)` uses its own seeded RNG (seed*7919+17), so the tap stream does not change. It drops speak and quiet questions alike. `run_stream(p_ask=)` and `--p-ask` work. `label_counts` and `n_pos` are in to_dict().
- CONFIRMED: `SchedulerPolicy(reannounce_guard_s)` with `_guarded()` and the 3 m approach exception. The `il+guard10` name is parsed. `repeat_counts` is in metrics.py and eval_reid.py prints a repeat_track column.
- CONFIRMED: tests. The three named files give 39 passed (44 s). The full suite gives 121 passed, 2 skipped (125 s; the coder said 176 s).
- CONFIRMED, recomputed from per_video_causal_s{0,1,2}.csv:
  - Dev pooled: greedy 98/480 = 0.204; lr0 66/480 = 0.138; zero-delay 246/480 = 0.512 (per seed 84, 80, 82 of 160); causal 96/480 = 0.200 (32, 33, 31); d2 182/480 = 0.379 (58, 62, 62).
  - Select pooled: 9/33, 3/33, 17/33, 6/33, 15/33.
  - Quiet violations: 0.115, 0.129, 0.096, 0.116, 0.091.
- CONFIRMED: bootstrap. My own 2,000-resample paired video bootstrap gives:
  - causal minus greedy: -0.004 [-0.058, +0.047]
  - greedy minus lr0: +0.067 [+0.013, +0.123]
  - zero-delay minus greedy: +0.308 [+0.244, +0.375]
  - d2 minus greedy: +0.175 [+0.110, +0.239]
  These match the reported intervals within Monte Carlo noise. The select point estimates (-0.091, +0.182) also match.
- CONFIRMED: p_ask, recomputed from per_video_pask*.csv.
  - p=0.5: greedy 31, 22, 27 of 160 (mean 0.167); causal 89/480 = 0.185 (the report says 0.186, a rounding slip).
  - p=0.25: greedy 86/480 = 0.179; causal 86/480 = 0.179.
  - p=1, seed 0: 0.188.
- CONFIRMED: Issue 6, recomputed from results/memory/reid_il_fix.csv.
  - IL: repeat_gt 65/228 = 0.285, repeat_track 31/228 = 0.136, 0.786 cues/min, finetune 4/27.
  - With the guard: 56/214 = 0.262, 17/214 = 0.079, 0.738 cues/min, 4/27.
  - Exactly 14 cues are removed, all of them same-track repeats.
- NOT CONFIRMED FROM REPO OUTPUTS: the guard-minus-off intervals (-0.048 [-0.084, -0.018], -0.057 [...], -0.023 [...]). The paired-difference table in reid_il_fix.md is empty, because eval_reid only compares re-ID on and off. The intervals came from scratchpad/coder/boot_causal.py or diag_*.py, which live outside the repo. The point estimates are right.
- NOT CONFIRMED: "the 18 same-track / 26 other live track / 21 ended track breakdown, median IoU 0.09" and "11 of 18 fire exactly 3.0 s later". These exist only in the coder's scratch scripts. I did not rerun them.
- WRONG IN docs/online.md: "Halving or quartering the questions moves greedy by 0.03 or less on the mean". At p_ask 0.5 the drop is 0.204 - 0.167 = 0.037. Only p_ask 0.25 (0.025) is within 0.03. The coder's report itself words this correctly.
- CAVEAT ON THE lr0 FLOOR: with lr=0 the weights stay 0 (learners.py:26), so every score is 0.5 and `max(legal, key=p)` (policy.py:165) picks the first legal candidate. The floor is therefore "a rate-controlled, first-listed candidate". It is not a content prior. That is why it is identical on all 3 seeds.
- MINOR:
  - summary_dev.md (regenerated) is seed 0 only, and its header does not say so. Its apple row now reads 0.169 at 1.68 cues/min, so the stale-file fix is CONFIRMED.
  - In docs/memory.md, the inserted "Issue 6" section sits between list items 4 and 5, which orphans item 5.
  - .claude/prds/seeforme.prd.md:5 still says "All of them wait to be asked".

== (1) PAPER DRAFT PROBLEMS AND FIXES ==

Numbers checked and correct (they match docs/online.md, docs/memory.md or the CSVs):
- Section 6.1 table: P-CACS 0.087, RL 0.050, IL 0.331 and 4/27, every online per-seed value and mean, quiet violations.
- Section 6.3 warm-start table and its intervals.
- Section 6.4 implicit-negative table.
- Section 6.6 memory table, 414,017 detections, 592 videos, match precision 0.90 (n=2,041) and 0.89 (n=1,863).
- Split table; 160 of 293 = 55%; 111 proxy users.
- Seed-0 algo-log values: d12 0.206, video-end 0.200/0.206/0.194, lr 0.1 gives 0.181 and lr 0.02 gives 0.194.
- Logistic AUC 0.675 (train 0.728) and top-1 0.835 against random 0.839 over 249 decisions. I reproduced these by rerunning scratchpad/alg/sep.py on rows.pkl.
- Ego-speed single-feature AUC 0.691, reproduced the same way.

Problems:
1. Sections 6.5, the abstract and contribution 3 call the causal-reference result "(prelim.)" and source it from scratch logs. It is now in the repo with 3 seeds. Fix:
   - Use the repo values: zero delay 0.525/0.500/0.512, pooled 0.512; video end 0.200/0.206/0.194, pooled 0.200; 2 s delay 0.362/0.388/0.388, pooled 0.379; lr0 0.138.
   - Add the intervals: causal minus greedy -0.004 [-0.054, +0.048]; greedy minus lr0 +0.067 [+0.010, +0.121]; d2 minus greedy +0.175 [+0.112, +0.239].
   - The draft's "2 s delay 0.369" is the scratch seed-0 value. The repo has 0.362 for seed 0 and 0.379 pooled.
   - The 12 s row (0.206) is still scratch-only. Keep it marked prelim, or drop it.
   - Delete the [TO RUN] under section 6.5 and section 8 items 1 and 2. Both are done.
2. The 28.5% re-announcement claim is now contradicted. It appears in contribution 4 ("repeats come from re-announcing a live track, not from forgetting") and in section 6.6, bullet 3. Fix: "28.5% (65/228) of IL cues name an already-announced weak event. Only 13.6% (31/228) repeat the same tracker track, because EgoBlind links every same-label detection in an event window to that event. A 10 s re-announcement guard, not tuned, cuts same-track repeats to 7.9% (17/214) with recall unchanged (0.331; 4/27 on finetune)." The cause is the 2.8 s per-track block with no novelty decay in the learned scheduler. It is not a memory failure.
3. Discussion 3 ("a replay in which users always ask hides the one-sided structure"), section 11 (the ML variant leads with this) and 100/100 item 1 (p_ask sweep) are now contradicted by the run sweep. Greedy recall is 0.204 at p=1, 0.167 at p=0.5 and 0.179 at p=0.25 (3 seeds, dev, diagnosis only). Causal full information is 0.200, 0.185 and 0.179. The falsifier (within 0.03 at 0.25) is met. Fix: say that lowering the asking rate to 0.25 produced no change separable from seed noise (spread 0.138 to 0.212), and that no PU correction was run because there is no gap to recover. Rewrite section 8 item 4 as done, and drop the ML-variant framing that depends on it.
4. Section 6.5 says gradient boosting gets "train 0.966, finetune 0.613". This is not reproducible. Two reruns of sep.py gave train 0.965 and finetune 0.620 and 0.629, because HistGradientBoosting uses a random early-stopping split. Fix: "train about 0.97, finetune about 0.62 (varies by run)", or set random_state and rerun.
5. UNVERIFIED: decision-level AUC 0.683, which is also the gate base for section 8 item 5 (0.703). No log or script in scratchpad/alg computes it; sep.py only gives candidate-level AUCs. Mark it prelim/unverified or regenerate it in the repo.
6. UNVERIFIED: "Hindsight relabelling gave 0.156 and 0.175". probe.py implements it, but no log holds those numbers. docs/online.md repeats them. Mark them unverified or rerun.
7. Section 6.2 sources Sculley's observation "as reported in arXiv 2006.04858". The Jiang et al. abstract does not mention Sculley (arXiv API). Mark it [verify] until the body is checked, or cite Sculley KDD 2007 directly after opening it.
8. Limitations say quiet violations are "about 2.5 to 5 times higher than for the frozen models". The frozen RL model's rate is 0.00, so the ratio is undefined. Fix: "about 2.5 to 5 times P-CACS (0.02) and IL (0.04); RL speaks 0.03 cues/min (results/online/summary_dev.md)". Also report cues/min for the frozen baselines: P-CACS 0.57, IL 0.79, RL 0.03. The "about 1.7 cues/min" comparison is not rate-matched for them.
9. Section 6.5's conclusion "the one-sided channel is not the bottleneck" needs a qualifier. Labels inside the clip are worth a lot: the 2 s delay gives +0.175 with an interval that excludes 0. Say "labels arriving after the clip add nothing over taps and questions; within-clip label latency matters".
10. Describe the lr0 floor as a rate-controlled, first-listed-candidate policy (problem 2 caveat above). Do not call it a "no-learning prior".
11. Citation details checked now against the arXiv API (export.arxiv.org):
    - Correct as written: WalkVLM 2412.20903 (no venue); Vinci2 2607.11523 (ECCV 2026); PING 1804.04118 (CoRL 2018); Horvitz 1301.6707 (UAI 1999); Pike-Burke 1709.06853 (ICML 2018); Bechavod 1902.02242 (NeurIPS 2019); Badanidiyuru 1305.2545; Agrawal-Devanur 1507.06738; ORBIT 2104.03841 (ICCV 2021); EgoBlind 2503.08221 (NeurIPS'25 D&B); Misfitting 2408.06546 (ASSETS 2024); Chang 2508.03651 (ASSETS 2025); Say It My Way 2602.16930 (CHI 2026); Raman 2310.19064 (COLT 2024); Agarwal 2302.03784 (no venue).
    - ProactiveAgent 2410.12361: the reward model trained on accept and reject labels is confirmed. The ICLR 2025 venue is not in the arXiv metadata (unverified).
    - Jiang et al. 2006.04858: one-sided feedback with generalized linear models is confirmed. AISTATS 2021 is not in the metadata (unverified).
    - WalkVLM-LR 2508.16070: add "ICASSP 2026" from the arXiv comment.
    - GuideDog 2503.12844: add "ACL 2026" and "22K image-description pairs" from the arXiv metadata. The CC BY-NC licence is unverified here.
    - Still unopened: Zhang et al. EMNLP 2025, Helmbold et al. 2000, Cesa-Bianchi et al. 2005, Bartók et al., Joulani et al., Elkan-Noto, Kiryo et al., Sculley, Pielot, Okoshi, and the Seeing AI guidedogs/hadley URLs.
12. 100/100 item 7: the stale summary_dev.md is already fixed. Its apple row is now 0.169, and the draft's "against 0.183" mixes a seed-0 value with a 3-seed mean. The select count in online.md is already fixed, so item 7 reduces to the PRD line 5 wording.

== (3) MESSAGES TO THE STUDENT ==

Trustworthy now (reproduced from repo CSVs or code):
- When full-information labels are held to the end of each clip, they do no better than one-sided greedy: 0.200 against 0.204, difference -0.004 [-0.054, +0.048], 3 seeds. The old 0.525 "upper bound" measured zero-delay labels.
- Learning beats the no-learning floor on dev: +0.067 [+0.010, +0.121].
- Labels that arrive 2 s late inside a clip are worth a lot: 0.379.
- Asking only a quarter of the questions does not measurably hurt greedy on this benchmark.
- IL same-track repeats are 13.6%, not 28.5%. A 10 s guard cuts them to 7.9% with no recall change.
- The warm-start, implicit-negative and memory tables match the docs.
- The new code is small and correct as read, and the full suite passes (121 passed, 2 skipped).

Not trustworthy yet:
- The decision-level AUC 0.683 and the hindsight-relabelling numbers 0.156 and 0.175 have no log behind them.
- The gradient-boosting 0.613 changes from run to run.
- The guard intervals and the 18/26/21 repeat breakdown come only from scratch scripts.
- The claim that "always-asking hides the one-sided structure" is contradicted by your own sweep.
- docs/online.md overstates the p_ask result ("0.03 or less"; 0.5 is actually 0.037).
- Select comparisons rest on 11 events per seed.
- All feedback is simulated (p_tap 0.5, oracle usefulness), and no BLV person has used the learner.
- The paper's headline contribution is now negative and diagnostic. Sell it that way, or run the anticipation-feature and pilot work before claiming more.

Needs you, not an agent:
- docs/ecc/ECC-OPERATING-MANUAL.md, which contains a home-directory path, is in the public repo.
- The advisor naming differs between README:3 and update.md:4.
- IRB status.
- .claude/prds/seeforme.prd.md:5 still says all apps "wait to be asked".
- The web default scheduler is the CampusBench RL (models/scheduler/scheduler_rl.pt). That and the web bugs (unescaped OCR innerHTML at main.ts:312-315, and styles.css:2 forcing a white background in dark mode) belong to the web session, and no agent has told it yet.
- Commit and review all the uncommitted changes listed above.
