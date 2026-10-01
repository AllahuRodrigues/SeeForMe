# SeeForMe: Predictive, Action-Aware Audio Cues for Blind and Low-Vision Navigation

CPSC 4900 Senior Project, Fall 2026. Student: Alláh-u-Abhá Rodrigues. Advisor: Sohee Kim Park.

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
| `docs/research_plan.md` | expanded plan with citation check and status tables |
| `tests/` | 37 pytest tests (geometry, CampusBench, P-CACS, CACS, tracker, metrics, ML) |

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
make live                                                    # webcam, P-CACS, depth, spoken cues (macOS say)
.venv/bin/python scripts/run_live.py --policy pcacs --speak --elevenlabs --units ft --goal "exit"
.venv/bin/python scripts/run_live.py --policy pcacs_learned:rl --speak --no-ocr   # learned scheduler, faster start
.venv/bin/python scripts/run_video.py walk.mp4 --policy pcacs                     # annotated video + cue log (JSONL)
.venv/bin/python scripts/demo_campus.py --kind stairs --policy pcacs              # rendered CampusBench scene
```
Keys in the live window: `1`–`5` policies (threshold, cooldown, CACS, P-CACS, learned), `s` speech,
`v` verbosity, `d` depth on/off, `f` FIND goal, `q` quit. The overlay shows distance, TTC, path
probability and utility per track, the action taken and the speech-budget bar.

**Web app** (phone, iPad, laptop; front and rear cameras; video files):

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

## 6. Latency (this machine, Apple MPS)

| stack | detect | depth | tracker + scheduler | end-to-end |
|---|---|---|---|---|
| YOLO11n + Depth Anything V2 metric-small (252 px) + P-CACS | 14 ms | 33 ms | 0.13 ms | **48 ms → 21 FPS** |
| YOLO11n + bbox-height distance + P-CACS | 17 ms | – | 0.13 ms | **17 ms → 57 FPS** |

The scheduler costs ~0.01 ms; the contribution of this thesis adds no latency. In the browser,
YOLO11n runs at ~70 ms per frame on WASM (software rendering) and faster with WebGPU; metric depth is
fetched from the server at ~4 Hz. `results/latency_pcacs.json`.

## 7. First-generation results (CACS, detection-level benchmark)

Held-out synthetic test split, 36 scenarios × 3 seeds (`results/RESULTS.md`): the confidence
baselines speak ~36 cues/min with important-cue recall 0.21–0.44; linear CACS speaks 8.6 cues/min
with recall 0.82 and precision 0.94; the reward-fine-tuned GRU reaches recall 0.85 with half the
critical delay. Every learned weighting ranked detector confidence last among the six signals.
Reproduce with `make all`.

## 8. Ethics and limits

The prototype is an experimental information aid, not a replacement for a cane or guide dog.
Low-confidence predictions are suppressed or hedged ("possibly a chair"); uncertain distances are
spoken coarsely ("close") rather than as numbers. Processing is local (the optional server runs on
the user's own machine); no frames are stored. CampusBench is a simulator: the numbers show the
*relative* behaviour of the policies under identical perception noise. Real recordings are the next
step (`scripts/annotate_video.py` drafts CampusBench-format annotations from a video;
`scripts/score_video.py --campus` scores a cue log against them). EgoBlind requires an access
request and does not carry cue-window labels; YouTube video is not used.
