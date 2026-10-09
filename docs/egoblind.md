# EgoBlind as a real-world benchmark and training source

[EgoBlind](https://github.com/doc-doc/EgoBlind) (Xiao et al., NeurIPS 2025 Datasets & Benchmarks) is a
VideoQA dataset: 1,392 first-person videos recorded by blind users according to the paper (the released train / test
annotations used here cover 1,329 of them: 673 official train, 656 official test), 5,311 questions they asked at a known
moment of the video (`start-time/s`), up to four reference answers per question, and a question type
(safety warnings, navigation, information reading, tool use, communication, resource). The paper's own
task is to answer the question from the clip up to that moment; its finetuning code targets 7B-8B
vision-language models on 2 x 80 GB GPUs (`external/EgoBlind/finetune/`), which is out of scope here.
Licence: CC BY-NC 4.0 (research and education only); everything trained on it is research-only. Redistributed question
and answer text must carry the original video sources
(https://drive.google.com/file/d/1Wob42tJ95DL1VvKad7BPc5QIlHtiuttI/view; `THIRD_PARTY_NOTICES.md`).

SeeForMe uses EgoBlind for something it was not built for but records implicitly: **when a blind user
needed information, and about what**. A safety question at `t_q` answered "No, a car is coming" says a car
was action-relevant just before `t_q`; a system that only speaks after the user has had to ask is late.

All numbers on this page were produced on 2026-10-09, after the audit described in `docs/plan_v3.md` and a second
pre-merge review that tightened the split and one label rule (section 7 of the plan). Results from before the audit
(inverted labels, a leaky validation split, label-based depth selection) are kept, labelled as superseded, in
`results/egoblind/exploratory_2026-10-08/`.

## Pipeline

| step | command | output |
|---|---|---|
| download (67 GB of zips, one at a time, zip deleted after extraction) | `make egoblind-download` | `data/egoblind/videos/{train,test}/*.mp4`, `data/egoblind/meta/*.csv` |
| weak labels from the questions, per split | `make egoblind-scenarios` | `data/egoblind/scenarios/<split>/<id>.json`, `coverage.json` |
| perception once per video (YOLO11n + Depth Anything V2 metric, 5 fps; indoor / outdoor depth model chosen by CLIP from the first 2 s of frames) | `make egoblind-cache` | `data/egoblind/cache/<id>.npz` (local, gitignored) |
| train the learned schedulers (split below) | `make egoblind-scheduler` | `models/egoblind_only/`, `models/egoblind/` |
| tune the linear P-CACS on `train` by random search | `make egoblind-tune` | `configs/pcacs_egoblind.yaml` |
| evaluate on the unseen holdout, clean and degraded | `make egoblind-holdout` | `results/egoblind/summary_holdout_{final,degraded}.*` |
| evaluate on EgoBlind's official test, clean and degraded | `make egoblind-experiments` | `results/egoblind/summary_test_official*.*` |
| intervals | `python scripts/bootstrap_ci.py results/egoblind/per_scenario_<split>_<tag>.csv --vs pcacs [--pairs a:b,...]` | `*.ci.{json,md}` |

## Splits: targets 70 / 15 / 5 / 10 by source recording

EgoBlind's clips are cut from longer recordings (`data/egoblind/meta/video_source.csv` maps clip-id ranges to the
original video), so neighbouring clips show the same user, route and scene. The official train clips are therefore
split by **whole source recording**, never by clip, in a fixed sha256 order of the recording ids
(`scripts/egoblind_common.py`; `data/splits/egoblind_*.jsonl` lists every clip with its recording). Rows of
`video_source.csv` whose clip ranges overlap (665-667 / 667-669, 715-728 / 719-728) are merged into one recording.
Clips without a row (11 in the train pool) cannot be placed by recording, and several continue a neighbouring train
recording, so they always go to `train`. Train clips 794-797 come from the same recording as official-test clips
1480-1495, so they are excluded from every split.

| split | role | share of the 673 clips | videos | recordings | questions | should-speak events | quiet windows | coverable events |
|---|---|---|---|---|---|---|---|---|
| `train` | DAgger imitation; P-CACS random-search tuning | 67.5% | 454 | 101 | 1,751 | 237 | 241 | 137 (133 evaluated) |
| `finetune` | REINFORCE fine-tuning | 15.5% | 104 | 20 | 400 | 56 | 38 | 27 |
| `select` | every selection: DAgger iteration, IL threshold, RL silence bias and epoch | 5.1% | 34 | 15 | 217 | 27 | 18 | 11 |
| `holdout` | unseen test: never used for training or selection; scored clean, degraded and with the open-vocabulary cache, plus the diagnostics below | 11.4% | 77 | 22 | 369 | 63 | 26 | 30 |
| excluded | shares a recording with the official test | 0.6% | 4 | 1 | 9 | - | - | - |
| `test` (official) | external held-out benchmark, never used for training or selection | - | 656 | 148 | 2,565 | 367 | 297 | 201 (200 evaluated) |

Recordings count clips without a source row as their own recording. Coverable counts in brackets exclude events that
fall after a truncated perception cache (below). No recording has clips in two splits, and no evaluated train-pool
clip shares a recording with the official test. Shares are not exactly 70 / 15 / 5 / 10 because recordings are
assigned whole.

Uploaders are not disjoint. 57 of the 77 holdout clips come from uploaders who also appear in `train`, 10 from new
uploaders, and 10 have no uploader id in `video_source.csv`; one uploader recorded 213 of the 673 official train clips
(31.6%). EgoBlind's own train / test split is not uploader-disjoint either: 460 of the 656 official test clips come
from uploaders who also appear in `train` (92 from new uploaders, 104 with no uploader id; the largest test uploader,
206 clips, has 90 clips in `train`). Both the holdout and the official test therefore measure transfer to new
recordings of mostly known uploaders. The `select` split is small (27 events), so selection is noisy; it is kept
separate so that nothing reported was used to choose a model.

## Weak events (`seeforme/egoblind/data.py`)

Only **safety warnings** (severity 1.0) and **navigation** (0.7) questions are proactive. For each:

* the question's kind decides what "yes" means: "is it safe?" and "is the path clear?" mean yes = nothing in the way;
  "is it dangerous?", "any risk?" and "is there anything ahead?" mean yes = something is there (the first strong cue
  word decides; an earlier version read clear and danger questions backwards, 120 of 1,398 labelled questions);
* an answer that says "not safe" or "unsafe" is a no, never a clear (an earlier rule read "not safe to step forward"
  as clear through "safe to");
* the COCO classes named in the answers become the event's `labels`; non-COCO concepts (stairs, door, curb, pole, ...)
  are recorded for coverage; clock directions ("at 2 o'clock") are not the COCO class clock;
* "No" / "nothing" / "the path is clear" answers become **quiet windows** (`should_speak = False`,
  `[t_q - 4, t_q + 0.5]`): a cue there is a false alarm; a plain yes to a clear question that names a new object is
  ambiguous and skipped;
* windows: `earliest_useful = t_q - 6` (or the first cached detection of a matching label within the previous 10 s),
  `ideal = t_q - 2`, `latest_safe = t_q + 0.5`. The user had to ask: anything later is late;
* `coverable` marks events the perception stack can see at all (a detector class that is not decor in the P-CACS
  taxonomy); `detected` marks coverable events the detector actually saw before `t_q`. Recall is reported over all
  events (system), coverable events (what scheduling could fix) and detected events (scheduler alone). Each event keeps
  its cache-independent state, so refining it with a different perception cache always starts from the same labels.

Information, tool-use, communication and resource questions produce no event. Everything is a heuristic label
(`label_source = "egoblind_question"`); a human-verified annotation in the CampusBench format
(`docs/campus_annotation.md`) overrides it. Cue precision is a lower bound: a correct cue about something the user did
not ask about still counts against it.

About half of what blind users asked about cannot be covered by a COCO detector plus the taxonomy: 45% of the
official test events are not coverable (door 40, unnamed objects 26, stairs 13, fence 12, ground hazards 9 among the
leading concepts).

## Replay and training (`seeforme/egoblind/{cache,scenario,runner}.py`)

The cache stores every detection (label, confidence, box, metric distance, depth confidence) at 5 fps.
`EgoBlindScenario` duck-types `CampusScenario`, so the CampusBench runner semantics (one message at a time, interrupts,
haptics), the metrics, the reference scheduler (`CampusOracle`) and the DAgger / REINFORCE trainers run unchanged on
real video. `EgoNoise` perturbs a replay (missed detections, depth noise, blackouts) for the degraded condition.
Corrupt frames are skipped while decoding; 6 videos are genuinely truncated after a corrupt stretch (31 unreadable
frames each: 94, 219 and 921 in `train`, 185 in `finetune`, 1051 and 1306 in the official test), and the 5 events that
fall after the cut, in 3 of them (4 in `train`, 1 in the official test), are excluded (`cache_truncated`). The indoor
or outdoor depth model is chosen from the frames alone (CLIP zero-shot on the first 2 s); an earlier version chose it
from the questions and answers, an annotation leak.

## The objective on weakly labelled video

The CampusBench communication score charges every spoken cue outside an event window as a false alarm. On EgoBlind
that is wrong away from the questions: nobody knows whether a cue at 40 s was useful if the user asked nothing near
40 s. The score and the REINFORCE reward therefore count an off-event cue only within 6 s of a *labelled* moment, a
safety or navigation question that became a should-speak event or a quiet window (`EgoBlindScenario.counts_fp`).
Even so, with labels this sparse the score is maximised by speaking as little as possible: both REINFORCE runs
fine-tuned on this reward (EgoBlind only, EgoBlind + CampusBench) converge to near silence, and so does random-search
tuning of the linear P-CACS on `train` (it chose a speak threshold of 1.66 and says nothing on train, holdout or test).

## Results on the unseen holdout and the official test

`make egoblind-holdout` (77 videos, 63 should-speak events, 30 coverable, 26 quiet windows, 55 min) and
`make egoblind-experiments` (656 videos, 367 events, 200 coverable, 297 quiet windows, 372 min). Cells: coverable
recall with its 95% bootstrap interval over videos / precision / quiet-window violations / cues per minute
(`results/egoblind/per_scenario_{holdout_final,test_official}.csv.ci.md`).

| policy (trained on) | holdout | official test |
|---|---|---|
| reference scheduler (question-derived windows) | 0.67 [0.47, 0.83] / 1.00 / 0.00 / 0.5 | 0.78 [0.72, 0.84] / 1.00 / 0.02 / 0.5 |
| confidence + cooldown | 0.53 [0.34, 0.71] / 0.02 / 1.00 / 16.3 | 0.57 [0.50, 0.64] / 0.02 / 0.83 / 16.0 |
| CACS v1 | 0.33 [0.18, 0.52] / 0.03 / 0.85 / 8.5 | 0.42 [0.35, 0.49] / 0.03 / 0.62 / 8.8 |
| P-CACS linear (CampusBench-tuned) | 0.00 [0.00, 0.00] / 0.00 / 0.00 / 0.1 | 0.07 [0.04, 0.11] / 0.06 / 0.04 / 0.6 |
| P-CACS linear (tuned on EgoBlind `train`) | 0.00 / - / 0.00 / 0.0 | 0.00 / - / 0.00 / 0.0 |
| GRU imitation (CampusBench) | 0.03 [0.00, 0.12] / 0.01 / 0.42 / 4.1 | 0.20 [0.15, 0.26] / 0.03 / 0.25 / 4.0 |
| GRU + RL (CampusBench) | 0.00 [0.00, 0.00] / 0.00 / 0.04 / 1.1 | 0.10 [0.06, 0.15] / 0.04 / 0.06 / 1.3 |
| GRU imitation (EgoBlind + CampusBench) | 0.13 [0.00, 0.25] / 0.09 / 0.12 / 0.9 | 0.12 [0.07, 0.16] / 0.07 / 0.05 / 0.9 |
| GRU imitation (EgoBlind only) | 0.17 [0.03, 0.33] / 0.06 / 0.12 / 1.6 | 0.20 [0.15, 0.25] / 0.06 / 0.13 / 1.9 |
| GRU + RL (EgoBlind + CampusBench) | 0.00 / - / 0.00 / 0.0 | 0.01 [0.00, 0.01] / 0.05 / 0.00 / 0.1 |
| GRU + RL (EgoBlind only) | 0.00 / - / 0.00 / 0.0 | 0.00 / 0.00 / 0.00 / 0.0 |

Paired against linear P-CACS on the same official-test videos, coverable recall differs by +0.13 [0.06, 0.20] for
EgoBlind-only imitation, +0.14 [0.07, 0.20] for CampusBench imitation, +0.04 [-0.02, 0.09] for CampusBench GRU + RL and
+0.04 [-0.01, 0.10] for the mixed imitation policy. With degraded perception (missed detections, depth noise, blackouts;
`results/egoblind/per_scenario_test_official_degraded.csv.ci.md`) the point estimates keep the same order except for
one pair: reference 0.74, confidence + cooldown 0.55, CACS v1 0.39, CampusBench imitation 0.22, EgoBlind-only imitation
0.15, CampusBench GRU + RL 0.10, mixed imitation 0.09, P-CACS 0.06. The mixed imitation policy drops from just above
CampusBench GRU + RL (0.12 vs 0.10) to just below it; that gap is within noise (paired, degraded: -0.02 [-0.08, 0.04]).

Reading:

* **No scheduler that respects quiet windows recovers much on real video.** The chatty baselines reach 0.53 to 0.57 of
  the coverable events only by speaking 16 times a minute and violating 83 to 100% of the moments the user confirmed
  were clear. Every policy that stays quiet enough is at 0.00 to 0.20.
* **The holdout is too small to rank the quiet policies**: their intervals overlap zero or each other. On the official
  test, two of the three imitation policies (EgoBlind only, CampusBench) have more coverable recall than the linear
  rule, at more quiet-window violations (0.13 and 0.25 against 0.04); the mixed imitation policy and CampusBench
  GRU + RL do not separate from it.
* **Neither leading imitation policy wins outright.** On the clean official test their coverable recall cannot be
  separated (CampusBench minus EgoBlind-only, paired on the same videos: +0.01 [-0.07, 0.09]); the CampusBench-trained
  policy uses about twice the airtime (4.0 against 1.9 cues a minute) and violates more quiet windows (paired +0.12
  [0.06, 0.19]). Under degraded perception the CampusBench-trained policy is ahead on recall (0.22 against 0.15, paired
  +0.07 [0.00, 0.14]) at +0.18 [0.12, 0.25] more violations. The pre-audit claim that EgoBlind-only imitation is the best
  non-reference policy is therefore not supported as stated.
* **Optimising the gated score on these labels learns silence.** Both EgoBlind-reward REINFORCE runs end at near zero
  cues (none on the holdout; 19 and 1 cues in the 372 min official test, one of the 20 correct), and random-search tuning
  of the linear rule on `train` ends at none: the gated score is still maximised by not speaking when labelled moments
  are this sparse.

## Where recall is lost (`make egoblind-diagnose`)

Every should-speak event is replayed with a decision hook and put in exactly one bin (`scripts/diagnose_egoblind_recall.py`;
the learned RL policy is cut at its learned silence bias, not at 0):

| bin | official test, P-CACS | official test, EgoBlind-only imitation | holdout, P-CACS | train, P-CACS |
|---|---|---|---|---|
| events | 367 | 367 | 63 | 237 |
| spoken in the window | 14 | 40 | 0 | 11 |
| spoken early / late | 1 / 5 | 4 / 13 | 0 / 1 | 2 / 6 |
| not coverable (outside the vocabulary or decor, or after a truncated cache) | 167 | 167 | 33 | 104 |
| not detected before the question | 25 | 25 | 7 | 13 |
| never tracked / below minimum confidence | 13 / 1 | 13 / 1 | 2 / 1 | 6 / 2 |
| never above the speak threshold | 141 | 102 | 19 | 92 |
| budget empty / channel busy | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| other | 0 | 2 | 0 | 1 |

Three findings hold on every split: (1) vocabulary first: 44 to 52% of the events are outside what the detector and the
taxonomy can name; (2) the channel is never the problem on real video: not one event was lost to a busy channel or an
empty budget; (3) of the events that are detected and tracked, most never reach the speak threshold.

## Which factor keeps detected events silent (`make decompose`)

`scripts/decompose_pcacs_utility.py` recomputes the linear P-CACS utility of every coverable event at every decision
step inside its window and asks which single factor, made ideal, would push it over the threshold. Each row changes one
factor on the *same* events, so the rows compare interventions on identical events. They are oracle what-ifs that set
the factor to its most favourable value for every event (depth confidence 1, path probability 1, urgency 1, penalty or
cost removed): upper bounds on how far the threshold binds, not what a perfect estimator of the true geometry would
give, and not improvements a real estimator has achieved. Share of the silent events, 95% Wilson interval:

| if this factor were ideal | official test (150 silent events) | train (101) | holdout (21) |
|---|---|---|---|
| depth confidence 1 | 0.01 [0.00, 0.05] | 0.00 [0.00, 0.04] | 0.00 [0.00, 0.15] |
| no uncertainty penalty (detector and depth) | 0.11 [0.07, 0.17] | 0.10 [0.06, 0.17] | 0.00 [0.00, 0.15] |
| speech and queue costs removed | 0.21 [0.15, 0.28] | 0.19 [0.12, 0.28] | 0.00 [0.00, 0.15] |
| object on the walking path (path probability 1) | 0.19 [0.14, 0.26] | 0.22 [0.15, 0.31] | 0.14 [0.05, 0.35] |
| contact imminent (urgency 1) | 0.09 [0.06, 0.15] | 0.16 [0.10, 0.24] | 0.05 [0.01, 0.23] |
| on the path **and** imminent | 0.63 [0.55, 0.71] | 0.72 [0.63, 0.80] | 0.71 [0.50, 0.86] |

The depth-confidence penalty is not what keeps these events silent. Depth also feeds distance, time to contact and
path probability, which these what-ifs leave at their logged values, so depth error itself is not ruled out. What the
question-derived labels reward and P-CACS does not is plausibly *awareness*: at their best moment the silent
official-test events are a median 4.8 m away, 3.1 s from contact, with path probability 0.66, people and cars the user
asked about while they were not clearly on a collision course. Whether a real motion estimator would lift any of the 63%
over the threshold is the open question; the what-if only says where the threshold binds.

## Threshold and awareness sweeps (train split)

`make egoblind-thr` and the `pcacs-aware<w>` variants (`PCACSWeights.w_aware`, a bonus for a severe object close in the
frontal cone regardless of path and time to contact), 454 train videos:

| variant | coverable recall | precision | quiet-window violations | cues / min |
|---|---|---|---|---|
| P-CACS (threshold 0.92) | 0.08 | 0.08 | 0.02 | 0.6 |
| threshold 0.6 | 0.14 | 0.06 | 0.06 | 1.4 |
| threshold 0.3 | 0.20 | 0.04 | 0.22 | 3.1 |
| threshold 0.0 | 0.37 | 0.04 | 0.34 | 5.3 |
| awareness weight 1 | 0.14 | 0.05 | 0.06 | 1.5 |
| awareness weight 2 | 0.20 | 0.03 | 0.27 | 3.3 |
| awareness weight 4 | 0.26 | 0.02 | 0.46 | 6.1 |

At matched airtime the awareness term buys nothing over a lower threshold: proximity alone does not separate the people
the user asked about from the people they did not. The separation has to come from the labels (a human pass) or from
signals the replay does not carry.

## Open-vocabulary detection on the holdout (`make egoblind-cache-ow`)

YOLO-World (`yolov8s-worldv2.pt`) with the navigation vocabulary `perception.NAV_VOCAB` re-cached all 673 train-pool
videos (`data/egoblind/cache_ow`, local). On the 77 holdout videos (`results/egoblind/cache_comparison_holdout.md`;
a holdout diagnostic, so the perception work it motivates uses train videos only):

| 63 holdout events | YOLO11n (COCO) | YOLO-World (NAV_VOCAB) |
|---|---|---|
| coverable | 30 | 46 |
| detected before the question | 23 | 23 |

The vocabulary widens on paper and the detections do not follow: YOLO-World does not fire on doors or stairs in this
footage at the caching threshold, and alternative prompts fire rarely (`results/egoblind/openvocab_prompt_probe.md`).
With the open-vocabulary cache, CampusBench GRU + RL announces 3 holdout events in time (0 with COCO) and P-CACS still
none. The EgoBlind-only imitation policy, trained on COCO replays, announces 5 events in time with the COCO cache and 6
with the open-vocabulary one (2 in common); its coverable recall goes from 0.17 to 0.13 only because the coverable set
grows from 30 to 46 (`summary_holdout_final.md`, `summary_holdout_ow.md`). Doors and stairs need a segmentation model
or a fine-tuned detector, not more prompts.

## Human-review export

`results/egoblind/review_events_{train,holdout,test_official}.csv`: one row per should-speak event (237 / 63 / 367 rows;
the 241 / 26 / 297 quiet windows are not exported) with the question, the reference answers, the matched concepts and
labels, the window, coverability, detection, each policy's bin and score, and empty `human_verdict` / `human_note`
columns (`keep`, `drop`, `quiet`, `awareness`, `collision`). A human pass over these rows is the step everything above
points to. The question and answer text is EgoBlind annotation (CC BY-NC 4.0): keep it non-commercial, cite the
dataset, and attribute the original video sources (`THIRD_PARTY_NOTICES.md`).
