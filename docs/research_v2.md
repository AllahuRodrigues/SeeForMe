# SeeForMe 2.0 research build: what the blueprint became in code

`research/RESEARCH_BLUEPRINT.md` (2026-10-08) and its catalog (`research/scenario_catalog.json`, 154 templates in
22 domains), sign taxonomy (89 strings) and stress matrix (1,232 specifications) are a *coverage plan*. This page
records which parts now exist as runnable experiments, what each measures, and what is still a plan. Nothing here
is human-annotated real data; the real-video side is EgoBlind (`docs/egoblind.md`).

## 1. Faculty feedback, item by item

| who said it | what it asked for | what exists now | still missing |
|---|---|---|---|
| Kazuki Irie: explicit X, Y, loss, training distribution, generalisation; GRU against equal-size alternatives | parameter-matched encoders `gru`, `mlp1` (current step only), `mlpk` (K steps concatenated), `attn` (temporal attention), all about 14k parameters (`seeforme/ml/scheduler_model.py`, `--arch`); held-out scene *families* (train on the 11 v1 kinds, test on the 4 new ones: `results/campus_v2/summary_test_zeroshot.md`); real-video transfer (EgoBlind) | multiple seeds per architecture; the human-labelled targets |
| Aspnes: a finite-channel scheduling problem with proper baselines; detect unfamiliar situations | the channel is now contested: external speech on the channel (`distractor_speech`), two hazards with one deadline, critical message playing when a cyclist crosses (`contention` kind); unknown objects the taxonomy has no class for are announced hedged (`unknown_obstacle`); capture loss is simulated (`blackout_windows`) | earliest-deadline-first and urgency-first reference schedulers as named baselines (the annotation-driven reference exists) |
| Koenig: a verifiable runtime property under stated assumptions | the channel semantics are deterministic and unit-tested (interrupt only cuts a less severe message; haptic only for imminent hazards; one message at a time) | a formal statement and model check of "no critical hazard inside its reaction horizon is blocked by a less severe message when perception reported it" |
| guide-dog discussion: ignore distractions, stop at elevation changes, remember obstacles, refuse unsafe instructions | the memory ledger (`TrackerConfig.memory_ttl_s`): a confirmed track that disappears stays available with decaying confidence and an extrapolated range, so a hazard seen before a camera pan can still be announced, hedged; the `occlusion` kind measures the benefit (recall) and the cost (false alarms when the object was moved while hidden) | intelligent disobedience proper: declining a user request (READ / FIND) when a time-critical hazard is due |
| Park: reproducible ML methodology, narrow scope | every experiment is a script with a fixed seed; v1 scenarios are reproduced inside v2 with identical content (same seeds; v2 adds only empty default fields such as yaw_keyframes [] and occluded_windows []) (`scripts/make_campus.py --kinds all` keeps the original seeds); 54 tests | the spring-scoped human study |

## 2. The four new scene families (CampusBench v2)

`python scripts/make_campus.py --out data/campus_v2 --kinds all` writes 216 scenarios (15 kinds x 6 test; the 14 non-OOD kinds x 6
train and x 3 val, since `ood_aftermath` is test-only by default and `load_campus_split` refuses it in train or val); the 165 v1 scenarios inside it are identical to `data/campus`. Set `SEEFORME_CAMPUS=v2` to point
every CampusBench script at it (`results/campus_v2/`).

| kind | catalog domains | what happens | what it measures |
|---|---|---|---|
| `contention` | crowd_contention | a 6-8 s sign read is playing when stairs enter their window; a chair and an approaching person share a deadline; a critical message is playing when a cyclist crosses 2.5 m ahead | interrupt and haptic decisions, critical blocking, lateness under a busy channel (in v1 these rules never bound) |
| `occlusion` | lost_observation, poor_capture | a hazard seen at 9 m is hidden for 2, 4, 6 or 10 s by an occluder or a deliberate 55-degree camera pan; in 30% of seeds it is gone when the view clears. Windows follow the user's need (`memory_windows`), not the camera | memory retention by occlusion length; false alarms from stale memory |
| `wayfinding_airport` | airport_terminal, transit_station, multilingual_and_symbols, text_and_signs | five translated wayfinding signs (en/pt/es/fr), advertising, one goal (gate, baggage, platform, restroom), luggage, a queue, a cart | goal-directed sign relevance across languages; airtime spent on signs |
| `ood_aftermath` | accident_aftermath, extreme_rare (OOD, evaluation only) | low light, blur, two capture losses, a stopped car across the path, debris, broken glass, a fallen sign, bystanders, an unnamed object | hedged warnings about unknown obstacles; silence when the view is gone; never a deployment claim |

Mechanics added for them: per-object `occluded_windows`, scenario `yaw_keyframes` (pans), `distractor_speech`
(external messages that occupy the channel and can be interrupted), `reaction_scale` (user reaction time), the
`unknown_obstacle` event type, and the tracker memory ledger. All are serialised and covered by `tests/test_campus_v2.py`.

## 3. Stress matrix

`python scripts/run_stress_matrix.py --sample 200` maps every modifier of a sampled specification onto overrides:
light -> low light / glare; camera -> sway, blur, tilt, partial occlusion; density -> scene family (empty, crowded,
burst); speech -> short or long external messages, an empty budget, TTS delay; perception -> depth noise, track loss,
an unknown object in the path, camera loss; user goal -> a requested sign read; reaction time -> 0.6x / 1x / 1.6x
reaction horizons; language -> translated nav signs. Output: `results/campus_v2/stress_s200.md` with per-factor
means per policy. OOD-scope rows are reported under their own scope and never pooled.

## 4. Sign taxonomy and multilingual signage

`python scripts/eval_sign_taxonomy.py` scores the keyword rules, the deployed English-template classifier (v1) and a
taxonomy-trained classifier (v2) on 199 strings: 83 English taxonomy strings (the bundle's 89 minus the 6 symbol-only
entries, which have no text), English decorative strings, and translated signs and decorative text in Portuguese,
Spanish and French (`seeforme/cues/sign_vocab.py`).
v2 is scored by 5-fold cross-validation grouped by sign intent, so a sign and its translations never sit on both sides of
a split (an earlier version split strings at random and reported accuracy on a translated subset whose labels are all
"important", where answering "important" every time scores 1.00). Accent-aware normalisation (`SAÍDA -> saida`) is in
`normalise_text`. Cells: accuracy / balanced accuracy (mean of the two class recalls), `results/sign_taxonomy_eval.md`:

| subset (strings, important) | always "important" | keyword rules | v1 (English templates) | v2 (grouped CV) |
|---|---|---|---|---|
| all (199, 160) | 0.80 / 0.50 | 0.32 / 0.57 | 0.63 / 0.71 | 0.86 / 0.84 |
| English taxonomy (83, 76) | 0.92 / 0.50 | 0.37 / 0.66 | 0.78 / 0.82 | 0.87 / 0.80 |
| Portuguese, Spanish, French (108, 84) | 0.78 / 0.50 | 0.22 / 0.50 | 0.49 / 0.60 | 0.85 / 0.81 |

On the non-English strings v2 recalls 0.88 of the important signs and 0.75 of the decorative ones, against 0.41 and 0.79
for v1. The gain is real but smaller than first reported, and the decorative class has only 24 non-English strings. The
deployed model stays v1 so that v1 results reproduce; `models/text_importance_v2.joblib` is the candidate replacement.

## 5. Results

Filled in by the runs listed below as they finish; each table names its command.

* `results/campus_v2/summary_test_zeroshot.md` and `summary_by_kind_test_zeroshot.json`: `SEEFORME_CAMPUS=v2 python scripts/run_campus.py --split test --repeats 1 --tag zeroshot` (v1-trained policies on all 15 kinds, zero-shot on the 4 new ones)
* `results/campus_v2/stress_s200.md`: `SEEFORME_CAMPUS=v2 python scripts/run_stress_matrix.py --sample 200 --tag s200`
* `results/campus_v2/arch_comparison_il.md`: `make archs` (`SEEFORME_CAMPUS=v2 python scripts/train_scheduler.py --arch <arch> --dagger-iters 1 --epochs 15 --rl-epochs 0 --out models/arch_<arch>` for mlp1, mlpk, attn, gru, and gru with `--memory-ttl 6`), then `SEEFORME_CAMPUS=v2 python scripts/compare_archs.py --which il`
* `results/campus_v2/summary_test_mem.md`: `SEEFORME_CAMPUS=v2 python scripts/run_campus.py --split test --repeats 1 --policies oracle,pcacs,pcacs_learned:il,pcacs_learned:rl --memory-ttl 6 --tag mem`, compared per kind with the zero-shot run (`summary_by_kind_test_{zeroshot,mem}.json`)
* `results/campus_v2/memory_grid_{val,test}.md`: `SEEFORME_CAMPUS=v2 python scripts/memory_grid.py --split val` (selection), then the chosen setting once on test

### 5.1 Zero-shot transfer of the v1-trained policies to the four new families

`SEEFORME_CAMPUS=v2 python scripts/run_campus.py --split test --repeats 1 --tag zeroshot` (6 test scenarios per kind, one noise seed).
Cells: action-weighted recall / late critical rate / cue precision / audio occupancy. Occupancy counts only the policy's own speech
(since 2026-10-09 external messages on the channel are no longer charged to the policy; recall and precision are unaffected).

| policy | contention | occlusion | wayfinding_airport | ood_aftermath (OOD) | all 15 kinds |
|---|---|---|---|---|---|
| reference scheduler | 0.62 / 0.39 / 1.00 / 0.18 | 0.49 / 0.50 / 1.00 / 0.05 | 0.78 / 0.28 / 1.00 / 0.45 | 0.41 / 0.17 / 1.00 / 0.12 | 0.68 / 0.25 / 0.99 / 0.21 |
| confidence + cooldown | 0.15 / 0.50 / 0.13 / 0.37 | 0.00 / 0.08 / 0.00 / 0.55 | 0.22 / 0.00 / 0.12 / 0.66 | 0.00 / 0.00 / 0.00 / 0.44 | 0.20 / 0.22 / 0.06 / 0.61 |
| CACS v1 | 0.02 / 1.00 / 0.06 / 0.13 | 0.00 / 1.00 / 0.00 / 0.05 | 0.07 / 0.61 / 0.10 / 0.32 | 0.21 / 0.17 / 0.39 / 0.14 | 0.18 / 0.63 / 0.21 / 0.21 |
| P-CACS linear | 0.51 / 0.39 / 0.67 / 0.16 | 0.50 / 0.50 / 0.75 / 0.07 | 0.46 / 0.00 / 0.71 / 0.31 | 0.21 / 0.17 / 0.64 / 0.08 | 0.46 / 0.27 / 0.60 / 0.19 |
| GRU imitation (v1-trained) | 0.26 / 0.78 / 0.94 / 0.08 | 0.49 / 0.50 / 0.75 / 0.07 | 0.59 / 0.00 / 0.90 / 0.29 | 0.37 / 0.17 / 0.52 / 0.21 | 0.62 / 0.23 / 0.74 / 0.23 |
| GRU + REINFORCE (v1-trained) | 0.24 / 0.78 / 0.94 / 0.07 | 0.49 / 0.50 / 0.92 / 0.06 | 0.61 / 0.00 / 1.00 / 0.28 | 0.32 / 0.17 / 0.74 / 0.14 | 0.63 / 0.24 / 0.89 / 0.19 |

**Contention is where the learned schedulers break.** Trained on v1, where the channel was almost never busy at a critical moment (one INTERRUPT label in 43k states), the GRU policies never interrupt on the contention family: interrupts per scene 0.00 (reference 1.00, linear P-CACS 1.00), haptics 2.0 to 2.3/min, critical blocking 0.39 (linear P-CACS 0.00), late critical rate 0.78 against 0.39. The hand-written rule generalises to a situation its designer anticipated; the imitation learner cannot act in a state its training distribution never contained. That is the distribution-shift point from the Irie meeting, and the reason `contention` scenes belong in the training split (the encoder comparison below trains on them).

**Occlusion** is identical for the reference, P-CACS and both GRUs (0.49 to 0.50; confidence + cooldown and CACS v1 announce none, and the plain confidence threshold catches one of the 12 events, recall 0.08 at cue precision 0.01): the hidden half of the hazards is unreachable without memory; see 5.2 for the ledger.

**Wayfinding** transfers well: the learned schedulers reach 0.59 to 0.61 recall at 0.28 to 0.29 occupancy with zero late critical events, the reference 0.78.

**OOD aftermath** is perception-limited (reference 0.41): the learned schedulers announce 32 to 37% of the debris, glass and stopped-car events, hedged, with silence precision about 0.9; nothing here is a safety claim.

### 5.2 The memory ledger (frame-only -> memory TTL 6 s, same v1-trained policies)

`SEEFORME_CAMPUS=v2 python scripts/run_campus.py --split test --repeats 1 --policies oracle,pcacs,pcacs_learned:il,pcacs_learned:rl --memory-ttl 6 --tag mem`.
Cells: action-weighted recall / late critical rate / cue precision, frame-only -> with the ledger.

| policy | occlusion | poor_capture (blackouts) | crowd | contention | all 15 kinds |
|---|---|---|---|---|---|
| reference | 0.49 -> 1.00 / 0.50 -> 0.00 / 1.00 -> 1.00 | 0.56 -> 0.80 / 0.56 -> 0.22 / 1.00 -> 1.00 | 0.68 -> 0.60 / 0.00 -> 0.00 / 1.00 -> 1.00 | 0.62 -> 0.66 / 0.39 -> 0.33 / 1.00 -> 1.00 | 0.68 -> 0.75 / 0.25 -> 0.19 / 0.99 -> 0.99 |
| P-CACS linear | 0.50 -> 0.58 / 0.50 -> 0.42 / 0.75 -> 0.92 | 0.22 -> 0.30 / 0.78 -> 0.61 / 0.36 -> 0.41 | 0.00 -> 0.04 / 0.00 -> 0.00 / 0.17 -> 0.08 | 0.51 -> 0.51 / 0.39 -> 0.39 / 0.67 -> 0.67 | 0.46 -> 0.47 / 0.27 -> 0.27 / 0.60 -> 0.58 |
| GRU imitation | 0.49 -> 0.83 / 0.50 -> 0.17 / 0.75 -> 0.89 | 0.51 -> 0.56 / 0.33 -> 0.11 / 0.51 -> 0.43 | 0.64 -> 0.58 / 0.00 -> 0.00 / 0.57 -> 0.72 | 0.26 -> 0.24 / 0.78 -> 0.78 / 0.94 -> 0.75 | 0.62 -> 0.66 / 0.23 -> 0.18 / 0.74 -> 0.72 |
| GRU + RL | 0.49 -> 0.83 / 0.50 -> 0.17 / 0.92 -> 0.94 | 0.61 -> 0.60 / 0.28 -> 0.17 / 0.77 -> 0.61 | 0.72 -> 0.48 / 0.00 -> 0.00 / 0.86 -> 0.81 | 0.24 -> 0.21 / 0.78 -> 0.72 / 0.94 -> 0.78 | 0.63 -> 0.62 / 0.24 -> 0.20 / 0.89 -> 0.83 |

A hazard seen at 9 m and then hidden for up to 10 s is unreachable for a frame-only policy (0.49 for everyone, the hidden half). With the ledger the reference recovers all of them and the learned schedulers announce 83% in time, hedged ("Possibly a chair, close"), with *higher* precision because the cue now lands inside the window. Blackouts (`poor_capture`) gain for the reference (0.56 -> 0.80), P-CACS and GRU imitation; GRU + RL only becomes less late (0.28 -> 0.17) and loses precision (0.77 -> 0.61). The cost shows in `crowd`: remembered people who have walked on produce stale cues; the reference loses recall (0.68 -> 0.60) and GRU + RL loses recall and precision (0.72 -> 0.48, 0.86 -> 0.81), while GRU imitation's crowd precision rises (0.57 -> 0.72). Memory therefore needs to be learned, not switched on: the `gru_mem` model in 5.4 is trained with the ledger on.

### 5.3 Stress matrix (200 sampled specifications, every modifier applied at once)

`SEEFORME_CAMPUS=v2 python scripts/run_stress_matrix.py --sample 200 --tag s200` (`results/campus_v2/stress_s200.md`, per-row CSV alongside).
Rows stack a light, camera, density, speech, perception, goal, reaction-time and language modifier, so absolute recall is lower
than on the clean benchmark (the reference scheduler averages 0.44 over the 146 in-scope rows that contain events; 167 rows are in scope, and the 21 with no events are left out of every mean below). Scenario seeds come from a stable
hash of the specification id, so the run reproduces exactly, and the 33 out-of-distribution rows are kept out of every factor
below and reported on their own (an earlier version of this table used per-process salted seeds and mixed the OOD rows into
every factor). What matters is the drop per factor, policy by policy. Action-weighted recall:

| factor: level | n | reference | P-CACS linear | GRU imitation | GRU + RL |
|---|---|---|---|---|---|
| speech: free | 31 | 0.53 | 0.22 | 0.45 | 0.48 |
| speech: busy_short | 33 | 0.41 | 0.18 | 0.40 | 0.36 |
| speech: busy_long | 30 | 0.27 | 0.15 | 0.20 | 0.24 |
| speech: budget_empty | 25 | 0.50 | 0.15 | 0.34 | 0.38 |
| speech: tts_delay | 27 | 0.54 | 0.20 | 0.44 | 0.41 |
| reaction_time: short | 60 | 0.52 | 0.23 | 0.35 | 0.39 |
| reaction_time: medium | 41 | 0.49 | 0.19 | 0.41 | 0.43 |
| reaction_time: long | 45 | 0.31 | 0.10 | 0.35 | 0.31 |
| camera: stable | 26 | 0.46 | 0.16 | 0.42 | 0.43 |
| camera: handheld_pan | 30 | 0.36 | 0.13 | 0.31 | 0.28 |
| camera: motion_blur | 27 | 0.42 | 0.17 | 0.37 | 0.39 |
| camera: partial_occlusion | 34 | 0.51 | 0.23 | 0.40 | 0.44 |
| camera: camera_tilt | 29 | 0.46 | 0.19 | 0.34 | 0.34 |
| perception: nominal | 24 | 0.44 | 0.20 | 0.37 | 0.37 |
| perception: depth_noisy | 17 | 0.50 | 0.18 | 0.25 | 0.23 |
| perception: track_loss | 36 | 0.38 | 0.16 | 0.35 | 0.35 |
| perception: unknown_object | 24 | 0.44 | 0.17 | 0.45 | 0.49 |
| perception: camera_unavailable | 18 | 0.52 | 0.22 | 0.42 | 0.42 |
| perception: ocr_error | 27 | 0.46 | 0.17 | 0.37 | 0.37 |
| density: sparse | 36 | 0.44 | 0.24 | 0.47 | 0.46 |
| density: moderate | 32 | 0.55 | 0.24 | 0.47 | 0.44 |
| density: crowded | 44 | 0.40 | 0.01 | 0.27 | 0.32 |
| density: burst | 30 | 0.39 | 0.31 | 0.23 | 0.23 |
| light: bright | 31 | 0.49 | 0.18 | 0.40 | 0.45 |
| light: dim | 23 | 0.37 | 0.11 | 0.30 | 0.22 |
| light: night | 29 | 0.38 | 0.20 | 0.33 | 0.35 |
| light: backlit | 38 | 0.48 | 0.21 | 0.39 | 0.38 |
| light: glare | 25 | 0.47 | 0.18 | 0.40 | 0.44 |

Readings (largest spread across a factor's levels, per policy):

* **Different policies break on different factors.** For the reference scheduler a busy channel matters most (long external
  messages 0.27 against 0.53 free), then reaction time (1.6x horizons 0.31 against 0.52 short). For the linear P-CACS it is
  crowd density: in crowded scenes it announces almost nothing (0.01; only 2 of 44 crowded rows recover any event), because a person who is not on a collision course rarely
  clears its threshold (the same mechanism as the real-video diagnosis in `docs/egoblind.md`). For GRU imitation it is
  a busy channel (long external messages 0.20 against 0.45 free), with density close behind (bursts 0.23 against 0.47
  moderate); for GRU + RL it is perception (depth noise 0.23 against 0.49 unknown objects), with speech, light and density
  within 0.03. In bursts of simultaneous events both GRU schedulers' late critical rate reaches 0.77 against 0.56 for the
  linear rule: the contention effect of 5.1 again.
* **User reaction time** costs three of the four policies: from 1x to 1.6x horizons, recall falls 0.49 -> 0.31 for the reference,
  0.19 -> 0.10 for the linear scheduler and 0.43 -> 0.31 for GRU + RL, while GRU imitation barely moves (0.41 -> 0.35, the
  same 0.35 it scores at 0.6x), so at 1.6x the learned ones keep 0.31 to 0.35 against the reference's 0.31. Timing tuned for one assumed reaction time does not transfer to a
  slower user, the argument for the D_i = T_action - R_u - L_system - d_i formulation in the plan.
* **Depth noise** hurts the learned schedulers more than the others (GRU + RL 0.23 against 0.37 nominal, late critical 0.44
  against 0.35): their features lean on range and time to contact.
* **Out-of-distribution rows** (crashes, street edges, extreme events; never training data; means over the 27 of the 33 OOD rows that contain events): reference 0.28, GRU imitation
  0.24, GRU + RL 0.21 at precision 0.63, linear P-CACS 0.13.
* The language factor is confounded in a 200-row sample (it only changes sign text in scenes that carry those signs); the
  `wayfinding_airport` family and the sign study (section 4) are the right place to read it.

### 5.4 Parameter-matched history encoders (the Irie comparison)

`make archs`: `SEEFORME_CAMPUS=v2 python scripts/train_scheduler.py --arch <arch> --dagger-iters 1 --epochs 15 --rl-epochs 0 --out models/arch_<arch>`
for mlp1, mlpk, attn and gru (and gru with `--memory-ttl 6`), on the v2 train split (84 scenarios: the 11 v1 families plus
contention, occlusion and wayfinding; the OOD family is test-only), threshold chosen on the v2 val split, then
`SEEFORME_CAMPUS=v2 python scripts/compare_archs.py --which il` on the 90 v2 test scenarios (imitation checkpoints;
`results/campus_v2/arch_comparison_il.md`). Re-run on 2026-10-09 after two fixes: the attention encoder now right-aligns
its window, so the last position always marks the current step, and the memory model is scored against a reference
scheduler that also has the memory ledger. One seed, one reduced training budget: differences under about 0.05 are not
resolved. All encoders have 13.9k to 14.3k parameters.

| model | encoder | params | action-weighted recall | late critical | precision | occupancy | silence precision | regret |
|---|---|---|---|---|---|---|---|---|
| mlp1 | current step only (MLP) | 14146 | 0.611 | 0.21 | 0.704 | 0.23 | 0.95 | 0.72 |
| mlpk | 8 steps concatenated (MLP) | 13872 | 0.609 | 0.22 | 0.724 | 0.23 | 0.94 | 0.67 |
| attn | temporal attention | 13992 | 0.608 | 0.19 | 0.707 | 0.23 | 0.95 | 0.71 |
| gru | GRU | 14268 | 0.620 | 0.23 | 0.766 | 0.22 | 0.94 | 0.53 |
| gru_mem | GRU trained and run with the memory ledger (TTL 6 s) | 14268 | 0.514 | 0.15 | 0.438 | 0.31 | 0.96 | 1.98 |

Recall per scene family:

| kind | mlp1 | mlpk | attn | gru | gru_mem |
|---|---|---|---|---|---|
| changing | 0.42 | 0.38 | 0.38 | 0.47 | 0.28 |
| contention | 0.28 | 0.24 | 0.62 | 0.27 | 0.57 |
| courtyard | 0.38 | 0.40 | 0.32 | 0.37 | 0.22 |
| crossing | 0.16 | 0.29 | 0.22 | 0.22 | 0.11 |
| crowd | 0.62 | 0.58 | 0.72 | 0.66 | 0.33 |
| doors | 0.92 | 1.00 | 0.92 | 0.96 | 0.75 |
| finding | 0.56 | 0.58 | 0.53 | 0.59 | 0.64 |
| hallway | 0.91 | 0.77 | 0.78 | 0.88 | 0.52 |
| negative | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| occlusion | 0.58 | 0.58 | 0.58 | 0.49 | 0.26 |
| ood_aftermath | 0.34 | 0.37 | 0.32 | 0.34 | 0.36 |
| poor_capture | 0.62 | 0.65 | 0.54 | 0.57 | 0.61 |
| stairs | 0.93 | 0.93 | 0.93 | 0.97 | 0.84 |
| temporary_hazard | 0.77 | 0.71 | 0.61 | 0.77 | 0.54 |
| wayfinding_airport | 0.66 | 0.66 | 0.66 | 0.74 | 0.68 |

Readings:

* **The recurrent history is not load-bearing at this scale.** The four encoders land within 0.012 recall of each other
  (0.608 to 0.620), inside the resolution of one seed; the GRU keeps the best precision (0.77) and the lowest regret
  (0.53). The 29 features already summarise the past (TTC and path probability from a 2 s fit, novelty, time since the
  last announcement, channel state), so an 8-step window adds little the features do not carry. This is the result Irie
  asked for, and it argues for spending the modelling effort on the state (uncertainty, deadlines, memory) rather than on
  sequence depth.
* **Training on contention teaches interruption to some models and not others.** On the contention family attn
  reaches 0.62 and gru_mem 0.57 (one interrupt per scene, no critical blocking), while mlp1, mlpk and gru stay at 0.24 to 0.28: with 11 to 12 INTERRUPT labels in 35k training states the
  behaviour is fragile. Multiple seeds and a class-balanced or RL objective are the next step. (Before the position fix, mlp1, attn and gru_mem learned it; which encoder does is not stable.)
* **Memory needs its own objective.** gru_mem, trained and evaluated with the ledger, has the lowest late critical rate
  (0.15) and the best `finding` recall, but precision 0.44 and occupancy 0.31: imitation labels say "speak" for every
  remembered object still inside its window, so the model announces stale memories. The reference-with-memory result in
  5.2 shows what is attainable; closing the gap needs REINFORCE with the ledger on or labels that charge wrong memories.

### 5.5 Experiment A: is the gap perception or scheduling? (oracle perception switch)

`SEEFORME_CAMPUS=v2 python scripts/run_campus.py --split test --repeats 1 --policies oracle,confidence_cooldown,pcacs,pcacs_learned:il,pcacs_learned:rl --perfect-perception --tag perfect`:
the simulator reports every visible object with its exact box and range, no confusions, no clutter
(`CampusDetectorSimulator(perfect=True)`), everything else unchanged. Action-weighted recall, noisy -> perfect, same scenarios and seed.

| policy | all 15 kinds | ood_aftermath | poor_capture | doors | contention | crowd | occlusion |
|---|---|---|---|---|---|---|---|
| reference scheduler | 0.68 -> 0.74 | 0.41 -> 0.68 | 0.56 -> 0.73 | 0.75 -> 0.87 | 0.62 -> 0.66 | 0.68 -> 0.70 | 0.49 -> 0.49 |
| P-CACS linear | 0.46 -> 0.55 | 0.21 -> 0.41 | 0.22 -> 0.40 | 0.31 -> 0.43 | 0.51 -> 0.53 | 0.00 -> 0.04 | 0.50 -> 0.49 |
| GRU imitation | 0.62 -> 0.58 | 0.37 -> 0.44 | 0.51 -> 0.79 | 1.00 -> 0.79 | 0.26 -> 0.22 | 0.64 -> 0.54 | 0.49 -> 0.18 |
| GRU + RL | 0.63 -> 0.70 | 0.32 -> 0.57 | 0.61 -> 0.81 | 0.91 -> 0.90 | 0.24 -> 0.29 | 0.72 -> 0.72 | 0.49 -> 0.49 |

Reading. Perfect perception is worth 0.06 to 0.09 recall overall for the reference, P-CACS and GRU + RL (GRU imitation
loses 0.04). For the reference and GRU + RL about half of the gain comes from the two capture-degraded families (`ood_aftermath`,
`poor_capture`: 48% and 49% of the summed per-kind gain); for P-CACS it is 30%, and 62% of P-CACS's remaining gain comes
from crossings, doors, hallways and temporary hazards. For the reference the rest is spread over doors, wayfinding,
finding and courtyard (hallway and temporary hazards do not move). The families where the learned scheduler stays
far below the reference barely move: `contention` (0.24 -> 0.29 against the reference's 0.66: it still never
interrupts), `crowd` for the linear rule (0.00 -> 0.04: a person who is not on a collision course never clears the
threshold) and `occlusion` (0.49 for the reference, P-CACS and GRU + RL without the memory ledger: a hazard that is not
reported cannot be scheduled). Those are scheduling and memory problems, not perception problems. The imitation policy
gets *worse* with perfect perception (occlusion 0.49 -> 0.18, doors 1.00 -> 0.79): it was trained on noisy confidence and
depth-confidence features and the constant 0.95 values are off its training distribution, a small concrete instance of
the generalisation question raised in the Irie feedback. Full per-kind tables:
`results/campus_v2/summary_by_kind_test_{zeroshot,perfect}.json`.

### 5.6 Experiment B: memory expiry (TTL x decay), selected on val, checked once on test

`SEEFORME_CAMPUS=v2 python scripts/memory_grid.py --split val` (3 val scenarios per kind, one seed;
`results/campus_v2/memory_grid_val.md` has every policy), then the chosen setting on test
(`scripts/memory_grid.py --split test --ttls 3 --decays 0.8`, `memory_grid_test.md`). An earlier version of this section
chose the setting on the test split; it was redone on val. Cells: action-weighted recall / cue precision of the GRU + RL
scheduler (trained without memory).

| memory config (val) | occlusion | crowd | poor_capture |
|---|---|---|---|
| no memory | 0.67 / 1.00 | 0.61 / 0.89 | 0.46 / 0.89 |
| TTL 3 s, decay 0.8 | **0.83 / 0.83** | **0.69 / 0.92** | **0.73 / 0.71** |
| TTL 3 s, decay 0.95 | 0.83 / 0.83 | 0.61 / 0.89 | 0.70 / 0.70 |
| TTL 6 s, decay 0.8 | 0.83 / 0.83 | 0.44 / 0.67 | 0.73 / 0.67 |
| TTL 6 s, decay 0.95 | 1.00 / 0.72 | 0.52 / 0.78 | 0.70 / 0.64 |
| TTL 10 s, decay 0.8 | 0.83 / 0.83 | 0.44 / 0.83 | 0.73 / 0.67 |
| TTL 10 s, decay 0.95 | 1.00 / 0.50 | 0.52 / 1.00 | 0.64 / 0.56 |

| chosen setting on test | occlusion | crowd | poor_capture |
|---|---|---|---|
| no memory | 0.49 / 0.92 | 0.72 / 0.86 | 0.61 / 0.77 |
| TTL 3 s, decay 0.8 | 0.83 / 0.94 | 0.56 / 0.82 | 0.62 / 0.70 |

Reading. On val, a short memory with fast decay is the only setting that helps all three families; longer TTLs cost
crowd recall, and slow decay buys occlusion recall with false alarms from objects that were moved while hidden
(precision 0.83 -> 0.50 at 10 s / 0.95). On test the occlusion gain holds (+0.34 at unchanged precision) but the crowd
effect reverses (0.72 -> 0.56), so with 3 val and 6 test scenarios per family the crowd effect is not resolved; remembered
pedestrians occupying the queue is the likely mechanism. The reference scheduler reaches 1.00 on occlusions from TTL 6 s
on because its windows (`memory_windows`) follow the user's need, which shows the ceiling of the mechanism. Next: train
with the 3 s / 0.8 ledger on, and a crowd-aware rule (remember static hazards, not pedestrians) as an ablation.

## 6. Datasets

See `docs/datasets.md`: EgoBlind is local and integrated; GuideDog needs the Hugging Face gate accepted with your
account (`scripts/download_guidedog.py`, `scripts/guidedog_eval.py` ready); DADA-2000 is a 117 GB split archive that
does not fit (annotation sheet kept; the `ood_aftermath` scenes stand in); JRDB, Ego4D and Aria need your registrations.
