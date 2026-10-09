# SeeForMe: Predictive, Action-Aware Audio Cues for Blind and Low-Vision Navigation

CPSC 4900 Senior Project, Fall 2026. Student: Alláh-u-Abhá Rodrigues. Advisor: to be confirmed (course instructor: Prof. Sohee Kim Park).

A camera can recognise twenty things at once; a person cannot listen to twenty things at
once. SeeForMe treats spoken feedback as a **scarce communication channel**. Instead of
narrating detected objects, it estimates *when* a visual event becomes actionable, predicts
its spatial and temporal relation to the user (distance, time-to-contact, path intersection),
and selects the **minimum information** the user needs for their next safe action, at the
earliest useful moment, or stays silent.

```
camera -> YOLO11 detection + EasyOCR + metric depth (Depth Anything V2)
       -> IoU tracker with dynamics (distance, speed, TTC, path probability)
       -> event features (hazard, urgency, actionability, goal, novelty, uncertainty, cost)
       -> P-CACS scheduler: SILENCE | SPEAK | INTERRUPT | HAPTIC  under a speech budget
       -> minimal slot message ("Stairs down, about 3 meters. Handrail left.")
       -> speech (ElevenLabs / macOS say / Web Speech) + haptics + debug overlay
```

Two generations of the cue layer live in this repository:

* **CACS** (Sept 2026): "which detected object is important right now?" — a linear rule
  over six signals, evaluated on a detection-level synthetic benchmark. Section 7.
* **P-CACS, Predictive CACS** (Oct 2026): "will this event change what the user should do
  in the next few seconds, what is the shortest useful message, and when should it be
  said?" — evaluated on **CampusBench**, a metric egocentric benchmark with communication
  windows. Sections 2–6. The thesis question is now:

> Can predictive, action-aware cue scheduling reduce communication load **and late warnings**
> while preserving or improving timely communication of action-relevant events?

## 1. Layout

| path | what it is |
|---|---|
| `seeforme/perception/` | YOLO11 detector, EasyOCR reader, **`depth.py`** (Depth Anything V2 metric-small on MPS/CUDA/CPU; bbox-height fallback), quality checks |
| `seeforme/geometry/` | pinhole `CameraModel`; `TrackDynamics`: distance, lateral offset, relative speed, **time-to-contact**, **path-intersection probability**, near-term crossing |
| `seeforme/tracking/tracker.py` | IoU tracker; keeps announcement memory and the metric state per track |
| `seeforme/cues/pcacs.py` | **P-CACS**: utility `U = w_h H f(TTC) + w_a A P + w_text T + w_goal G` scaled by novelty, minus uncertainty, speech cost and queue cost; four actions; token-bucket audio budget; verbosity; FIND goals |
| `seeforme/cues/compose.py` | slot-based minimal messages (object, distance, direction, motion, action, detail); distance hedged when depth is uncertain; metres or feet |
| `seeforme/cues/cacs.py`, `baselines.py`, `features.py` | the first-generation CACS and the two confidence baselines |
| `seeforme/campus/` | **CampusBench**: metric world (`world.py`), 11 scene generators (`scenes.py`), detector + depth noise (`simulator.py`), closed-loop runner with interrupt/haptic channel (`runner.py`), reference scheduler (`oracle.py`), metrics (`metrics.py`) |
| `seeforme/ml/scheduler_model.py` | **learned scheduler**: GRU over the last 8 decision steps (29 signals), action / urgency / slot heads; DAgger imitation then REINFORCE with the benchmark score as reward |
| `seeforme/ml/` (rest) | text-importance classifier, priority models, scene clustering, first-generation sequence model |
| `seeforme/feedback/` | macOS `say`, **`elevenlabs_tts.py`** (key from `.env`, cached, interruptible) |
| `seeforme/pipeline.py` | `LivePipeline`: depth + dynamics + scheduler for camera and video |
| `web/` | **browser app** (Vite + TypeScript PWA): YOLO via ONNX Runtime Web in a worker, tracker + P-CACS ported to TS, phone/tablet/laptop cameras, NAVIGATE / FIND / READ / ASK, Web Speech or ElevenLabs, haptics, debug panel. See `web/README.md` |
| `server/app.py` | FastAPI: `/api/depth`, `/api/ocr`, `/api/tts`, serves `web/dist`; `--ssl` for phones |
| `scripts/` | every experiment as a script (below) |
| `docs/campus_annotation.md` | the CampusBench event format (also the format for annotating real recordings) |
| `seeforme/egoblind/`, `docs/egoblind.md` | **EgoBlind**: weak events from blind users' questions, perception cache, replay runner, EgoBlind metrics; `scripts/{download,build,cache,run,train}_*egoblind*.py` |
| `docs/research_plan.md` | expanded plan with citation check and status tables |
| `tests/` | 67 pytest tests (geometry, CampusBench v1 and v2, P-CACS, CACS, tracker + memory ledger, metrics, ML, EgoBlind splits and labels, web parity) |
| `research/`, `docs/research_v2.md`, `docs/datasets.md` | the October 2026 research blueprint (catalog, sign taxonomy, stress matrix) and what became code; external dataset access |

## 2. Setup

```bash
cd Project
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
cp .env.example .env   # add ELEVENLABS_API_KEY=... for the ElevenLabs voice (optional)
```
YOLO weights download on first use; the Depth Anything V2 metric-small models (indoor / outdoor,
~100 MB each) download from the Hugging Face hub on first use.

## 3. Live demos

```bash
make live                                                    # webcam, linear P-CACS, depth, spoken cues (macOS say)
.venv/bin/python scripts/run_live.py --policy pcacs_learned:rl --speak                # the learned scheduler (GRU + RL)
.venv/bin/python scripts/run_live.py --policy pcacs --speak --elevenlabs --units ft --goal "exit"
.venv/bin/python scripts/run_video.py walk.mp4 --policy pcacs                     # annotated video + cue log (JSONL)
.venv/bin/python scripts/demo_campus.py --kind stairs --policy pcacs              # rendered CampusBench scene
```
Keys in the live window: `1`–`5` policies (threshold, cooldown, CACS, P-CACS, learned), `s` speech,
`v` verbosity, `d` depth on/off, `f` FIND goal, `q` quit. The overlay shows distance, TTC, path
probability and utility per track, the action taken and the speech-budget bar.

`--policy pcacs_learned:rl` loads `models/scheduler/scheduler_rl.pt`, the GRU + RL scheduler trained on CampusBench (data
generated by this repository, so no dataset licence restriction). The live demo that makes it the default (with a spoken
fallback to P-CACS, an announcement gate and a headless `make live-check`) is being finished in a follow-up change.

**Web app** (phone, iPad, laptop; front and rear cameras; video files). It runs the same learned scheduler in the
browser (ONNX Runtime Web, `web/public/models/scheduler_rl.onnx`), with linear P-CACS selectable in Settings and used
automatically, with a status message, if the model cannot load. `python scripts/web_parity.py --check` and
`node web/parity/run.mjs` check that the browser and Python make identical decisions (4,560 of 4,560 recorded steps).
The PWA precaches the model by revision, so a new release replaces a cached one on the next load and says
"Updated to version X" once. As in Python, COCO classes the taxonomy calls decor (dog, umbrella, couch, bed) are not
announced.

```bash
make web            # build web/dist
make serve          # http://localhost:8000  (depth, OCR, ElevenLabs served to the browser)
make serve-ssl      # https://<mac-ip>:8000 for phones (self-signed certificate)
cd web && npm run dev   # development server on :5173
```

## 4. CampusBench

Real annotated assistive video is scarce, so the decision layer is trained and evaluated on a
**metric** simulator: objects have positions and velocities in metres relative to a user walking
at ~1.1 m/s, bounding boxes come from a pinhole camera (70° FOV, chest height, pitched down, with
hand sway in the poor-capture class), and a noisy detector *and* a noisy metric depth sensor produce
the observations. Eleven scene classes: Harkness-style courtyard, hallway, stairs, doors, street
crossing, finding (goal-directed), crowd, temporary hazard, poor capture, negative (busy scenes where
nothing should be said), changing (a chair dragged into the path, a pedestrian turning in, a bike
shooting out of a side street, a door closing). 66 train / 33 val / 66 test scenarios.

Ground truth is a **communication window per event**, not a binary importance flag:
`should_speak`, severity, earliest useful / ideal / latest safe time, path intersection, the minimum
information slots, and the ideal cue (`docs/campus_annotation.md`). A *sudden* flag marks events that
become visible after their latest safe time, which no policy can handle on time.

Metrics (`seeforme/campus/metrics.py`): action-weighted recall, late critical rate, mean lead time,
cue precision, audio occupancy, cues and haptics per minute, silence precision, critical-cue
blocking, minimum-message ratio, distance / TTC error, and **communication regret** against an
annotation-driven reference scheduler (score = Σ severity × timeliness − off-window cues −
late/missed events − airtime).

```bash
make campus              # generate the splits (data/campus/)
make tune-pcacs          # random search of the linear P-CACS weights on train -> configs/pcacs_tuned.yaml
make scheduler           # DAgger + REINFORCE learned scheduler -> models/scheduler/
make campus-experiments  # baselines vs CACS vs P-CACS vs ablations vs learned schedulers on test
make campus-report       # figures + results/campus/CAMPUS_RESULTS.md
make campus-all          # all of the above (~30 min)
```

## 5. Results on CampusBench (held-out test split, 66 scenarios × 3 noise seeds)

| policy | action-weighted recall | late critical rate | lead before deadline (s) | cue precision | audio occupancy | cues/min | silence precision | min-message ratio | regret |
|---|---|---|---|---|---|---|---|---|---|
| confidence threshold (baseline 1) | 0.25 | 0.27 | 1.87 | 0.06 | 0.73 | 31.8 | 1.00 | 0.47 | 8.47 |
| confidence + cooldown (baseline 2) | 0.26 | 0.27 | 1.99 | 0.07 | 0.64 | 27.0 | 0.98 | 0.48 | 7.32 |
| CACS, linear (first generation) | 0.23 | 0.61 | 0.88 | 0.25 | 0.23 | 8.5 | 0.78 | 0.39 | 3.20 |
| **P-CACS, linear (tuned)** | 0.48 | 0.28 | 1.92 | 0.55 | 0.21 | 7.3 | 0.80 | 0.97 | 1.41 |
| P-CACS learned, GRU imitation (DAgger) | 0.68 | 0.18 | 2.84 | 0.73 | 0.25 | 9.0 | 0.96 | 0.89 | 0.55 |
| **P-CACS learned, GRU + REINFORCE** | 0.70 | 0.21 | 2.47 | 0.88 | 0.20 | 7.4 | 0.90 | 0.89 | 0.11 |
| reference scheduler (annotation-driven) | 0.73 | 0.24 | 1.14 | 1.00 | 0.21 | 7.1 | 0.81 | 1.00 | -0.00 |

**Reading.** The confidence baselines occupy the audio channel 64–73% of the time and still warn
on time for only a quarter of the severity-weighted events, because confidence is not actionability.
First-generation CACS is quiet (23% occupancy) but *late*: with proximity derived from box size, 61% of
critical events are announced after their safe deadline. Linear P-CACS, with metric depth, TTC and
slot messages, doubles timely recall at the same airtime, halves lateness, and speaks messages that
match the annotated minimum information 97% of the time. The **learned scheduler** closes most of the
remaining gap to the reference scheduler: timely recall 0.70 vs 0.73, 88% precision, 20% airtime,
silence correct 90% of the time, regret 0.11. Where it wins is exactly the predictive part: crowds
(0.63 vs 0.00 for the linear rule), changing scenes (0.45 vs 0.18), poor capture (0.48 vs 0.18) and
doors (0.82 vs 0.38), i.e. deciding from the *history* of a track instead of its current state.
Street crossings stay hard for everyone (reference 0.38): laterally crossing vehicles are visible for
well under a second, and the benchmark reports them as *sudden*.

**Ablations (linear P-CACS).** Depth and slot selection are load-bearing; the budget, interrupt and
haptic rules rarely bind at this operating point; removing the queue cost raises recall slightly but
costs precision and airtime.

| variant | action-weighted recall | late critical rate | cue precision | audio occupancy | regret |
|---|---|---|---|---|---|
| pcacs | 0.48 | 0.28 | 0.55 | 0.21 | 1.41 |
| pcacs-no_depth | 0.35 | 0.47 | 0.65 | 0.14 | 1.64 |
| pcacs-no_slots | 0.42 | 0.31 | 0.57 | 0.22 | 1.69 |
| pcacs-no_queue | 0.50 | 0.27 | 0.51 | 0.23 | 1.51 |
| pcacs-no_ttc | 0.49 | 0.29 | 0.57 | 0.20 | 1.36 |
| pcacs-no_budget | 0.48 | 0.28 | 0.55 | 0.21 | 1.42 |
| pcacs-no_interrupt | 0.48 | 0.28 | 0.55 | 0.21 | 1.42 |
| pcacs-no_haptic | 0.48 | 0.28 | 0.55 | 0.21 | 1.41 |
| pcacs-detailed | 0.46 | 0.32 | 0.57 | 0.22 | 1.57 |

Full tables (per scene, non-sudden late rates, tuning, DAgger / REINFORCE curves) are in
`results/campus/CAMPUS_RESULTS.md`; figures in `results/campus/figures/`; two rendered scenes in
`results/campus/demo_*.mp4`.

## 6. EgoBlind: real blind-user video (NeurIPS 2025 D&B)

[EgoBlind](https://github.com/doc-doc/EgoBlind) has 1,392 first-person videos recorded by blind users (1,329 in the
released train / test annotations: 673 train, 656 test) and 5.3K questions they asked at known moments, with reference
answers and a question type. SeeForMe turns
every safety-warning and navigation question into a weak communication event in the CampusBench format
("No, a car is coming" at 31 s means a car was action-relevant before 31 s; a cue after the user had to ask
is late), "no obstacle" answers into quiet windows where any cue is a false alarm, and replays the
schedulers over cached YOLO + metric-depth detections of the real video. Details, the label rules, the tables with
intervals and the caveats are in `docs/egoblind.md`; the code is `seeforme/egoblind/`.

The official train clips are split **by whole source recording** (no recording on two sides; overlapping source rows
merged), targeting train 70% for imitation, finetune 15% for reward fine-tuning, select 5% for every checkpoint and
threshold choice, and an unseen holdout 10% never used for training or selection (`SPLIT_SHARES` in
`scripts/egoblind_common.py`). Because recordings are assigned whole, the realised shares of the 673 clips are 67.5 /
15.5 / 5.1 / 11.4% (454 / 104 / 34 / 77); 4 clips that share a recording with the official test are excluded.
EgoBlind's official test (656 videos) is a second, external held-out set, likewise never used for selection.

```bash
make egoblind-download     # 67 GB of zips from the authors' Google Drive (data/egoblind/videos)
make egoblind-scenarios    # question -> weak events per split (data/egoblind/scenarios/<split>/*.json)
make egoblind-cache        # YOLO11n + Depth Anything V2 once per video, 5 fps; depth model chosen from the frames
make egoblind-scheduler    # DAgger on train, REINFORCE on finetune, selection on select -> models/egoblind_only, models/egoblind
make egoblind-tune         # linear P-CACS tuned on train by random search -> configs/pcacs_egoblind.yaml
make egoblind-holdout      # every policy on the unseen holdout, clean and degraded
make egoblind-experiments  # every policy on the official test, clean and degraded
make egoblind-diagnose     # where each event's recall is lost
```

On the official test (656 videos, 200 coverable events), no scheduler that respects the user's quiet moments recovers
much: the chatty baselines reach 0.57 of the coverable events at 16 cues a minute while violating 83% of the moments the
user confirmed were clear, and every policy quiet enough is at 0.00 to 0.20. Two of the three imitation policies have
more coverable recall than the linear P-CACS (0.07 [0.04, 0.11]), at more quiet-window violations: EgoBlind-only
imitation 0.20 [0.15, 0.25] at 1.9 cues a minute, +0.13 [0.06, 0.20] over P-CACS, violations 0.13 against 0.04;
CampusBench imitation 0.20 [0.15, 0.26] at 4.0 cues a minute, +0.14 [0.07, 0.20], violations 0.25. The two cannot be
separated on recall (+0.01 [-0.07, 0.09]); the mixed EgoBlind + CampusBench imitation policy does not separate from
P-CACS (+0.04 [-0.01, 0.10]). Optimising the gated score on these labels learns silence: both REINFORCE runs and the
P-CACS weights tuned on EgoBlind `train` end at (near) zero cues. The 77-video holdout is too small to rank the quiet
policies.

Where the recall goes: 45% of the official-test events are outside what a COCO detector plus the taxonomy can name
(doors, stairs, fences, poles, things the answer does not name), 7% are never detected, and most of the rest never reach
the speak threshold; not one event was lost to a busy channel or an empty budget. Setting depth confidence to 1 would
lift 1% of the silent events over the speak threshold; forcing every silent object onto the walking path with contact
imminent (path probability 1, urgency 1) would lift 63%, an upper bound on the collision-course term rather than what a
perfect estimator would give (an oracle what-if, not a result). Lowering the threshold or adding a proximity term does
not separate the moments users asked about from the rest, and an open-vocabulary detector widens the vocabulary
without detecting more (holdout: 30 -> 46 coverable, 23 -> 23 detected).
The human-review export `results/egoblind/review_events_{train,holdout,test_official}.csv` is the next step.
EgoBlind is CC BY-NC 4.0: research only; cite the paper and attribute the original video sources
(https://drive.google.com/file/d/1Wob42tJ95DL1VvKad7BPc5QIlHtiuttI/view; local copy `data/egoblind/meta/video_source.csv`, not in git) with any
redistribution.

## 7. Research build (October 2026): CampusBench v2, stress matrix, encoder comparison

The blueprint in `research/` (scenario catalog, sign taxonomy, stress matrix) became four new scene families
(`contention`, `occlusion`, `wayfinding_airport`, `ood_aftermath`), a tracker memory ledger, external speech on the
channel, parameter-matched history encoders (`--arch gru|mlp1|mlpk|attn`), a stress-matrix runner and a multilingual
sign study. `docs/research_v2.md` maps each faculty comment to what now runs and what is still a plan;
`docs/datasets.md` lists every external dataset and what it needs from you.

```bash
make campus-v2               # 216 scenarios (ood_aftermath test-only), the 165 v1 ones content-identical (same seeds; v2 only adds empty default fields) (data/campus_v2)
make campus-v2-experiments   # v1-trained policies zero-shot on the new families, with and without memory
make stress                  # 200 sampled stress specifications x policies (results/campus_v2/stress_s200.md)
make archs                   # mlp1 / mlpk / attn / gru / gru+memory at ~14k parameters each (results/campus_v2/arch_comparison_il.md)
make sign-taxonomy           # keyword vs v1 vs taxonomy-trained v2 on 199 sign strings in four languages (CV grouped by sign intent)
make campus-v2-perfect       # Experiment A: oracle perception (exact boxes and ranges) vs the noisy simulator
make memory-grid             # Experiment B: memory TTL x decay, selected on val (results/campus_v2/memory_grid_{val,test}.md)
```

Headline numbers (v2 test split, one seed; `docs/research_v2.md` has the tables and the caveats):

* Zero-shot, the v1-trained GRU schedulers never interrupt on the contention family (0.24 to 0.26 recall, 78% late critical) while the
  hand-written P-CACS does (0.51, 39%). Trained on contention scenes, only the attention encoder and the GRU trained with the memory ledger learn it (0.62, 0.57); mlp1, mlpk and the plain GRU stay at 0.24 to 0.28.
* At 14k parameters each, the four history encoders are within 0.012 recall of each other (0.61 to 0.62); the GRU keeps the best precision
  (0.77) and regret (0.53). Recurrence is not load-bearing at this scale.
* The memory ledger (TTL 6 s) takes the learned schedulers from 0.49 to 0.83 recall on hidden hazards and the reference to 1.00; in crowds the
  reference and GRU + RL lose recall (GRU + RL 0.72 to 0.48). Trained naively with the ledger, the GRU over-speaks (precision 0.44).
* Memory expiry chosen on val (3 s, fast decay) keeps +0.34 recall on test occlusions at unchanged precision; the crowd effect flips between
  val and test, so it is unresolved on these small splits.
* Oracle perception (Experiment A) is worth 0.06 to 0.09 recall overall, about half of it in the capture-degraded families for the reference and GRU + RL (30% for P-CACS); `contention`,
  `crowd` and `occlusion` barely move, so those gaps are scheduling and memory, not perception.
* Stress matrix (stable seeds, OOD rows separate): busy channels hurt the reference most, crowd density hurts the linear rule most (0.01 in crowded scenes) and a busy channel hurts GRU imitation most (0.20 vs 0.45 free) and depth noise hurts GRU + RL most (0.23 vs 0.49), with bursts of simultaneous events close behind for both (0.23, late critical 0.77); a 1.6x reaction time costs the reference, the linear rule and GRU + RL (GRU imitation is flat: 0.35 at both 0.6x and 1.6x).
* Sign text, Portuguese / Spanish / French strings: balanced accuracy 0.60 for the English-template importance model and 0.81 for a
  taxonomy-trained v2 (cross-validation grouped by sign intent; answering "important" every time scores 0.50).

What to do next, with status and measured costs, is `docs/plan_v3.md` (it supersedes `docs/plan_2026-10.md`); the glasses track (Meta Wearables Device Access Toolkit) is
`docs/meta_glasses.md`; external datasets examined this month (BLV-Road-Nav, Mendeley, TIMELI, ORBIT, Mobilio,
WalkVLM-LR, VizWiz) are in `docs/datasets.md` and `docs/*_notes.md`.

## 8. Latency (this machine, Apple MPS)

| stack | detect | depth | tracker + scheduler | end-to-end |
|---|---|---|---|---|
| YOLO11n + Depth Anything V2 metric-small (252 px) + P-CACS | 14 ms | 33 ms | 0.13 ms | **48 ms → 21 FPS** |
| YOLO11n + bbox-height distance + P-CACS | 17 ms | – | 0.13 ms | **17 ms → 57 FPS** |

The scheduler costs ~0.01 ms; the contribution of this thesis adds no latency. In the browser,
YOLO11n runs at ~70 ms per frame on WASM (software rendering) and faster with WebGPU; metric depth is
fetched from the server at ~4 Hz. `results/latency_pcacs.json`.

## 9. First-generation results (CACS, detection-level benchmark)

Held-out synthetic test split, 36 scenarios × 3 seeds (`results/RESULTS.md`): the confidence
baselines speak ~36 cues/min with important-cue recall 0.21–0.44; linear CACS speaks 8.6 cues/min
with recall 0.82 and precision 0.94; the reward-fine-tuned GRU reaches recall 0.85 with half the
critical delay. Every learned weighting ranked detector confidence last among the six signals.
Reproduce with `make all`.

## 10. Ethics and limits

The prototype is an experimental information aid, not a replacement for a cane or guide dog.
Low-confidence predictions are suppressed or hedged ("possibly a chair"); uncertain distances are
spoken coarsely ("close") rather than as numbers. Processing is local (the optional server runs on
the user's own machine); no frames are stored. CampusBench is a simulator: the numbers show the
*relative* behaviour of the policies under identical perception noise. Real recordings are the next
step (`scripts/annotate_video.py` drafts CampusBench-format annotations from a video;
`scripts/score_video.py --campus` scores a cue log against them). EgoBlind carries no cue-window labels; the weak events derived from its
questions (section 6) are heuristics and are reported as such. YouTube video is not used.
