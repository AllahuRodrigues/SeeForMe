# SeeForMe 3.0: audit, verified state and staged plan (2026-10-09)

This page answers the "SeeForMe 3.0" roadmap with what the repository and this Mac actually contain, what was verified,
what the real-video failures come down to, and the next steps with exact commands and measured costs. It supersedes
`docs/plan_2026-10.md` where they differ. Machine: Apple M1 Max, 10 cores, 32 GB, 42 GB free.

## 1. What exists

Machine-readable versions: `data/manifests/datasets.json`, `data/manifests/models.json`, `data/splits/*.jsonl`, and a
`PROVENANCE.json` next to every trained model (`python scripts/build_manifests.py` regenerates them; nothing is moved).

| dataset | on disk | state | licence | commercial use |
|---|---|---|---|---|
| EgoBlind | 1,329 videos (673 official train, 656 official test), perception caches for all of them (COCO) and for the 673 train-pool videos (open vocabulary), weak-event scenarios for every split | complete | CC BY-NC 4.0 | no |
| GuideDog | 0.65 GB: gold 2,106, depth 383, object 435 | silver (19,978, 4.5 GB) not downloaded | CC BY-NC 4.0 plus gate terms | no |
| CampusBench v1 / v2 | 165 / 216 scenarios (the v2 OOD family is test-only) | complete | generated here | yes |
| BLV-Road-Nav-Accessibility | 256 MB: 90-object taxonomy, 671 labelled keyframes | complete (no videos are distributed) | no licence file | no (ask the authors) |
| TIMELI | 471 MB: all label JSONs and the generator code | labels only (frames 32 GB, videos 537 GB) | CC BY 4.0 | yes, with attribution |
| VizWiz-Classification | 0.7 GB: annotations + 1,500 of 8,900 photos | stratified sample | CC BY 4.0 | yes, with attribution |
| Mendeley navigation assistance | 87 MB | complete (features only) | CC BY 4.0 | yes, with attribution |
| DADA-2000 | annotation sheet only | videos need an external disk | no licence stated | no |
| Mobilio, WalkVLM-LR | code and study logs | reference only | MIT / none | Mobilio yes / WalkVLM no |
| ORBIT, JRDB, Ego4D, Aria | not downloaded | need your registration or an external disk | see manifest | no for JRDB, Ego4D, Aria |

**Splits (targets 70 / 15 / 5 / 10).** EgoBlind's official train clips are split by whole source recording into `train`
70% (imitation, P-CACS tuning), `finetune` 15% (reward fine-tuning), `select` 5% (every checkpoint, threshold and
silence-bias choice) and `holdout` 10% (unseen test, never used for training or selection); the realised shares are
67.5 / 15.5 / 5.1 / 11.4% of 673 clips because recordings are assigned whole. Overlapping source rows are merged into one
recording, clips without a source row go to `train`, and the 4 clips that share a recording with the official test are
excluded, so no recording has clips in two splits and none is shared with the official test; the manifests list every
clip with its recording. EgoBlind's official `test` (656 videos) is a second, external held-out set, likewise never used
for selection.
CampusBench keeps its generated split by scenario seed (6 train / 3 val / 6 test per family): its test scenarios come
from seeds never used for training or selection, which is the held-out principle; the proportions are kept because the
v1 results in the README are defined on that split, and new seeds can be generated at any proportion. GuideDog keeps the
authors' roles (gold, depth and object for evaluation, silver for training).

## 2. What was verified

* **Reproducibility.** `make repro-check` re-ran the CampusBench v2 test results of three policies and the EgoBlind
  holdout results of two policies from the committed code and the local perception caches (`data/egoblind/cache` is
  gitignored, not committed): all 163 metrics are identical
  (`results/repro_check.md`).
* **Tests and builds.** 67 tests pass (`make test`); the web app type-checks and builds; the browser and Python learned
  schedulers make identical decisions on 4,560 of 4,560 recorded steps (`tests/test_web_parity.py`).
* **Apps.** The live pipeline loads the learned scheduler with `--policy pcacs_learned:rl` (the live-demo update that
  makes it the default and adds a headless `make live-check` follows separately). In the browser (web app 0.3.1),
  the learned model loads, the scheduler switch works, and a new release replaces a cached one on the next load. The
  camera loop could not be exercised in the hidden preview pane here; check it on a phone or laptop.
* **Audit.** A read-only audit of the branch with adversarial verification of every finding; confirmed findings and their
  fixes are in section 7.

## 3. Why SeeForMe is silent on real video

On EgoBlind's official test (656 videos, 367 weak events), every scheduler that respects the user's quiet moments is at
0.00 to 0.20 coverable recall, while chatty baselines reach 0.57 only by speaking 16 times a minute and violating 83% of
the moments the user confirmed were clear (`docs/egoblind.md` has the tables with 95% intervals). The funnel for the
linear P-CACS (`results/egoblind/recall_diagnosis_test_official.md`):

| stage | events left | lost here |
|---|---|---|
| should-speak events derived from questions | 367 | |
| inside the detector vocabulary and the taxonomy (and before the cache ends) | 200 | 167: doors, stairs, fences, poles, unnamed things, COCO classes the taxonomy calls decor; 1 after a truncated cache |
| detected before the question | 174 | 26 |
| tracked and confident at a decision step | 160 | 14 |
| announced in time by the linear P-CACS | 14 | 141 never above the threshold, 5 early or late (one more, undetected, event was announced late) |
| announced in time by the EgoBlind-only imitation policy | 40 | |

The roadmap's hypothesis was that depth uncertainty suppresses useful cues. The depth-confidence penalty does not:
making it ideal pushes 1% [0.00, 0.05] of the silent events over the threshold. Depth also feeds distance, time to
contact and path probability (`seeforme/geometry/dynamics.py`), which this what-if leaves at their logged values, so
depth error is not ruled out.
`scripts/decompose_pcacs_utility.py` asks which single factor, made ideal, would push each silent coverable event over the
threshold, on the same events (`results/egoblind/utility_decomposition_test_official.md`; 95% Wilson intervals). These are
oracle what-ifs that set each factor to its most favourable value (upper bounds), not improvements anyone has achieved:

| official test: 150 silent coverable events | share that would cross the threshold |
|---|---|
| depth confidence 1 | 0.01 [0.00, 0.05] |
| no uncertainty penalty at all (detector and depth) | 0.11 [0.07, 0.17] |
| speech and queue costs removed | 0.21 [0.15, 0.28] |
| object on the walking path | 0.19 [0.14, 0.26] |
| contact imminent | 0.09 [0.06, 0.15] |
| on the path **and** imminent | 0.63 [0.55, 0.71] |

Three hypotheses, ordered by the what-if and the funnel above (not measured effect sizes; label semantics can only be
measured by the human review in stage 3):

1. **Label semantics.** The question-derived labels reward awareness ("is it safe?" when a person is near), P-CACS rewards
   collision avoidance; at their best moment the silent events are a median 4.8 m away and 3.1 s from contact. Neither a
   lower threshold nor a proximity term separates the two at matched airtime; only a human pass over the events can say
   which moments deserved a cue.
2. **Vocabulary.** 45% of the events are outside what a COCO detector plus the taxonomy can name. The open-vocabulary
   detector widens coverage on the holdout (30 to 46 of 63 events) without detecting more (23 to 23): YOLO-World does not
   see doors or stairs in this footage (`results/egoblind/openvocab_prompt_probe.md`).
3. **Training data and objective.** On the official test two of the three imitation policies have more coverable recall
   than the linear rule (EgoBlind-only imitation +0.13 [0.06, 0.20] over P-CACS at 1.9 cues a minute, CampusBench
   imitation +0.14 [0.07, 0.20] at 4.0), at more quiet-window violations (0.13 and 0.25 against 0.04); the mixed imitation
   policy does not separate (+0.04 [-0.01, 0.10]). Optimising the gated score on these labels learns silence: both
   REINFORCE runs and the P-CACS weights tuned on EgoBlind `train` end at (near) zero cues.

GuideDog adds the depth side (`results/guidedog/summary.json`, `make guidedog`). Depth Anything V2 Metric Small orders
GuideDog's 383 closer/farther object pairs correctly 88.5% of the time: 96% when the closer object is under 3 m, 90% at
3-8 m and 80% beyond 8 m; 75% when the two depths are within a factor 1.5 of each other and 97-100% beyond that. Its
absolute metres disagree with GuideDog's (median relative error 1.1 to 3.3 depending on the band), but GuideDog's depths
are themselves estimates, so only the ordering is a fair test. The 2,106 human-verified gold paragraphs describe 7,154
hazards by clock direction (and 2,394 directions as clear); 55% name a COCO class, and YOLO11n detects 86% of those in the
stated direction. The rest is the same vocabulary gap EgoBlind shows: unnamed structures, road edges, fences, signs,
trees, walls and poles.

## 4. Staged plan with commands and measured costs

Costs are measured on this M1 Max unless marked as an estimate. Run `make smoke` (about three minutes) before any long job.

| stage | what | command | cost |
|---|---|---|---|
| 0 | smoke test: tests, GuideDog synthetic rows, a 3-video EgoBlind replay | `make smoke` | ~3 min |
| 0 | exact re-run of committed results | `make repro-check` | ~15 min |
| 1a | retrain the EgoBlind-only imitation policy on open-vocabulary replays (all 673 train-pool videos are cached) | `SEEFORME_EGOBLIND_CACHE=data/egoblind/cache_ow python scripts/train_scheduler_egoblind.py --train egoblind --rl-epochs 0 --out models/egoblind_only_ow` | ~45 min |
| 1b | door / stairs perception: segmentation trial on the train videos that mention them | new script; candidate models must pass the licence check in section 6 | estimate: 1 day of work, < 1 GB of weights |
| 2 | simulator curriculum (timing, contention, occlusion) | `make archs`, `SEEFORME_CAMPUS=v2 python scripts/train_scheduler.py --memory-ttl 3` | ~5 to 10 min per encoder |
| 3 | human review of the weak events (needs you) | open `results/egoblind/review_events_{train,holdout,test_official}.csv`, fill `human_verdict` (keep / drop / quiet / awareness / collision) | ~650 rows, about 5 h at 30 s each |
| 3 | retrain on the reviewed labels | `make egoblind-scheduler` after the review is merged into the scenarios | ~45 to 65 min |
| 4 | DAgger rollouts | already closed-loop in `seeforme/ml/scheduler_model.py` (`--dagger-iters N`); relabels with the reference built from the (reviewed) events | ~10 min per iteration |
| 5 | constrained RL | REINFORCE with a quiet-window constraint (Lagrange multiplier) instead of a fixed penalty; only meaningful after stage 3 | ~10 min per epoch |
| 6 | evaluate on the holdout and the official test (the holdout has already informed stages 1a / 1b through the open-vocabulary and recall diagnostics, so report the official test as the clean held-out number) | `make egoblind-holdout`, `make egoblind-experiments` | ~10 min each |
| 7 | compress and export | `make export-models` (Core ML 5.2 MB, ONNX 10 MB); the scheduler is exported for the browser by `scripts/export_web_scheduler.py` (99 KB ONNX) | minutes |

## 5. Selective downloads (prepared, not run)

| dataset | command | disk | needs |
|---|---|---|---|
| GuideDog silver | `python scripts/download_guidedog.py --configs silver` | 4.5 GB | nothing more (you are logged in) |
| JRDB, a few crowded sequences | register at https://jrdb.erc.monash.edu.au/register/, then download the 2D tracking labels and images of 2-3 train sequences | ~5-10 GB (estimate) | your registration |
| Ego4D metadata only | `pip install ego4d && ego4d --output_directory data/ego4d --datasets annotations --metadata` | a few GB | your signed Ego4D licence and AWS keys |
| Aria Everyday Activities, a few sequences | accept the licence on projectaria.com, download the CDN JSON, then `aria_dataset_downloader -c <json> -o data/aria -l <sequence ids>` | ~5 GB per sequence (estimate) | your licence; a separate Python env if the tools do not support 3.13 |
| TIMELI frames | `gdown --folder` on the TIMELI data folder (images.tar.gz) | 32 GB | an external disk |
| ORBIT validation | `docs/orbit_notes.md` | 4.15 GB | nothing |

With 42 GB free, GuideDog silver fits; TIMELI frames, DADA-2000, Aria and anything larger need a 1-2 TB external SSD.
Do not run more than three caching or training jobs at once: the ten cores saturate.

## 6. Research-only and commercial artefacts

Every trained scheduler carries a `PROVENANCE.json`. Today:

* research only: everything trained on EgoBlind (`models/egoblind`, `models/egoblind_only`), the EgoBlind-tuned P-CACS
  configs (`configs/pcacs_egoblind.yaml`; the superseded pre-audit ones in `results/egoblind/exploratory_2026-10-08/`),
  and anything that will be trained on GuideDog, BLV-Road-Nav, DADA, JRDB, Ego4D or Aria;
* commercially clean on the data side: the CampusBench-trained schedulers (`models/scheduler`, which the live demo and the
  web app run, and `models/arch_*`), P-CACS with `configs/pcacs_tuned.yaml`, and models built only from CC BY 4.0 data
  (TIMELI, VizWiz, Mendeley) with attribution;
* conditional: YOLO11n and YOLO-World run through Ultralytics under AGPL-3.0, which requires offering the serving
  application's source or buying an Ultralytics Enterprise licence (the web app ships YOLO11n to every browser); Depth
  Anything V2 Metric Indoor Small is usable commercially, the Outdoor Small checkpoint was fine-tuned on non-commercial
  Virtual KITTI 2. `THIRD_PARTY_NOTICES.md` has the full list.

## 7. Audit findings and fixes

A read-only audit of the branch ran on 2026-10-09: six auditors (results against docs, library code, scripts and
leakage, Makefile and documented commands, licensing, web app), each finding re-checked by one to two independent
verifiers, then a completeness critic. 87 findings were confirmed and 2 more came from the critic. Everything below is
fixed in code and every affected result was regenerated; the numbers in this repository are the post-fix ones, and the
pre-audit EgoBlind results are kept as a labelled record in `results/egoblind/exploratory_2026-10-08/`.

| finding | what was wrong | fix and consequence |
|---|---|---|
| inverted weak labels | "is the path clear?" was read with yes = something there, "is it dangerous?" with yes = clear; clock directions ("at 2 o'clock") matched the COCO class "clock" | question kinds now decide what yes means; 120 of 1,398 labelled EgoBlind questions changed label; every EgoBlind result re-run |
| leaky validation split | `video_id % 7` put clips of the same source recording in train and val (86 of 97 val clips) | targets 70 / 15 / 5 / 10 by whole source recording: train, finetune + select, unseen holdout; plus the official test |
| selection on the reported split | EgoBlind schedulers picked their DAgger iteration, threshold and RL epoch on the reported videos; the memory setting was picked on CampusBench test | selection only on EgoBlind `select` and CampusBench `val`; reported splits never used for training or selection |
| annotation leak into perception | the indoor / outdoor depth model was chosen from each video's questions and answers | CLIP zero-shot on the first 2 s of frames chooses it; 144 of 673 videos changed depth model and were re-cached |
| truncated caches | decoding stopped at the first corrupt frame | corrupt frames are skipped; events after a real truncation are excluded |
| metric bug | external speech on the channel was charged to the policy, and cutting it off earned a bonus | only the policy's own airtime counts; CampusBench v2 occupancy and regret re-run |
| reward / score mismatch, attn positions | REINFORCE ignored repeat cues that the score charges; the attention encoder's positions depended on track age | reward mirrors the score; attn right-aligned and retrained |
| stress matrix not reproducible | per-process salted seeds; OOD rows mixed into every factor | stable seeds, OOD reported separately, re-run |
| recall diagnosis cut-offs | the RL scheduler was binned against threshold 0 instead of its learned silence bias | correct cut-offs; diagnosis re-run |
| no uncertainty | single numbers on 30 to 200 events | video-level bootstrap intervals and paired differences (`scripts/bootstrap_ci.py`), Wilson intervals on the what-ifs |
| GuideDog parser, sign study | the hazard parser read the whole paragraph and counted "o'clock" as a clock; the sign study scored a one-class subset | clause-per-direction parser; intent-grouped CV with balanced accuracy and baselines |
| web app | double screen-reader announcements, stale ElevenLabs audio after mute, no detector recovery, "room 204" matching "Room 310", shortcuts stealing keys | fixed; the browser now also runs the learned scheduler with Python parity |
| reproducibility and licensing hygiene | small checkpoints ignored by git, missing requirements, Makefile targets that crashed, no third-party notices, manifest marking a non-commercial depth model as commercial | tracked checkpoints, complete requirements, fixed targets, `THIRD_PARTY_NOTICES.md`, corrected manifests |
| documentation | 28 statements that overstated or misquoted results | rewritten from the regenerated results |

### Pre-merge review (2026-10-09)

A second review of the branch before merging (split leakage, results against docs, licensing hygiene, web app; every
finding adversarially verified) confirmed 47 findings. All are fixed; the split and label fixes changed the data, so
both EgoBlind schedulers were retrained, the linear P-CACS was re-tuned on EgoBlind `train`, and every EgoBlind result,
interval, diagnosis, decomposition, sweep and the reproducibility check were regenerated (`results/rerun_20261009b/`).

| finding | what was wrong | fix and consequence |
|---|---|---|
| overlapping source rows | two pairs of rows in `video_source.csv` claim the same clips; last-row-wins split two recordings across splits | union-find merges them into one recording |
| clips without a source row | 11 train-pool clips became one-clip recordings; holdout clip 87 and select clips 367 / 443 continue neighbouring train recordings | they always go to `train`; holdout 78 -> 77 videos |
| official test shares a recording | train clips 794-797 and official-test clips 1480-1495 come from one recording | 794-797 are excluded from every split |
| "not safe" read as clear | "No, it is not safe to step forward" matched the clear rule through "safe to" | an explicit unsafe rule wins; 2 official-test events and 1 finetune event changed |
| stale tuned P-CACS | `configs/pcacs_egoblind*.yaml` were tuned on the pre-audit split and labels yet still default policies | moved to the exploratory folder; re-tuned on `train` (`make egoblind-tune`); it converges to silence |
| stale defaults | the cacher still defaulted to the removed `val` split; an unknown split silently returned no videos | defaults fixed; an unknown split now fails |
| licensing hygiene | EgoBlind text shipped without the source-link attribution; GuideDog `scene_location` strings and parser test sentences copied from the gated data; a duplicate 10.7 MB YOLO export; a personal file in `docs/` | notices carry the EgoBlind citation and source link; GuideDog text removed or paraphrased; export restored; the personal file untracked |
| web app | Start said "I'll tell you what matters" after the detector had failed; a detector failing every frame went silent; ElevenLabs was the default voice and could speak a cue 8 s late; interrupts went to a polite live region; default config differed from the tuned one; "Repeat" gave false precision | detector state machine with restart and spoken failure; browser voice by default, ElevenLabs only when the server reports TTS, late cues dropped; assertive region for interrupts; tuned defaults; distance phrasing reused |
| documentation | 24 statements with wrong counts or overreaching wording (uploader overlap, ranking within noise, "perfect estimate", denominators, shares) | rewritten from the regenerated results with paired intervals |

## 8. Next experiments, in order

1. Human review of the ~650 weak events (needs you; stage 3). Everything downstream depends on it.
2. An explicit awareness event type (an object close in the frontal cone, announced once, low priority) evaluated against
   the reviewed labels: it is the hypothesis section 3 points to.
3. Door and stairs perception trial (stage 1b), then retrain on the open-vocabulary or segmentation replays (stage 1a).
4. A real motion estimator for "on the path and imminent", evaluated on the same events as the what-if in section 3, to
   see how much of the 63% upper bound survives when path and time to contact come from a real estimate.
5. Product track in parallel: test the web app and `make live` with VoiceOver and TalkBack users (cannot be automated
   here), then the Meta glasses mock-device loop (`docs/meta_glasses.md`).
