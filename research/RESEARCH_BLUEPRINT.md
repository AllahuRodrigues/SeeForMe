# SeeForMe 2.0: broad-scenario research and machine-learning blueprint

Prepared 2026-10-08. This is a **proposal and coverage plan**. It does not imply that external datasets were downloaded, that human decisions have been labeled, or that rare-event assistance has been validated.

## Thesis-sized scientific claim

> Under a limited audio budget, does a goal-conditioned, memory-augmented, uncertainty-aware scheduler outperform strong reactive and temporal baselines on independently labeled **action-relevant communication** in held-out indoor pedestrian settings?

**Product roadmap:** The same *interface and representation* may expand to airports, train stations, urban walkways, emergency incidents, and multilingual settings. General availability or safe navigation in those domains is **not** established by indoor/campus experiments.

## Integrating faculty suggestions

- **Kazuki Irie:** Show a reproducible learning problem with explicit X, Y, loss, train distribution, and generalization. Compare GRU memory against (a) one-frame MLP, (b) 8-step MLP, (c) GRU, (d) explicit tracker/map memory, and optionally (e) equal-size temporal attention. Match parameter count and training budget. Hold out object-disappearance/occlusion lengths to measure information retention; an unseen location test cannot be replaced by splitting adjacent windows.
- **Guide-dog analogy from another faculty discussion:** Study formal guidework instruction: maintaining purposeful travel line, ignoring distractions, stopping at elevation changes, clearance, and rejecting unsafe directives. Treat these as *observable skills*, not a model of canine cognition or a guarantee. Respect the handler's responsibility for routes and crossings. Put route goal and preferences in the ML state.
- **James Aspnes:** Define a finite-channel scheduling baseline: first-in-first-out, urgency-first, earliest-deadline-first, optimal offline with access to event truth (an upper bound), and online learned policy. Detect out-of-distribution visual conditions and avoid confident guidance.
- **Jérémie Koenig:** Specify and unit/model-check runtime monitor properties conditioned on accurate hazard observations and bounded processing latency. A safety shield cannot prove perceptual correctness.
- **Sohee Kim Park:** Use core ML coursework, reproducible experiments and narrow project scope; agree on deliverables/advisors well before Spring 2027.

## The research ML problem (Professor Irie's formulation)

**Environment:** one episode is a continuous first-person walk or simulated walk, with camera/IMU/OCR observations, tracked candidate events and a serialized communication channel.

**Inputs observed at step t:**

- Per event: detected class/text and confidence, normalized bbox, depth and its quality, lateral position, relative motion, TTC, path-crossing probability, track age/missed frames, velocity uncertainty, existing 29 P-CACS features. Use a missing-data mask; do not replace unavailable depth by perfect truth.
- Whole scene: user goal, walk mode, available speech tokens, current message/remaining playback, number and relative importance of candidates, turn/stop information if reliably estimated, distance/calibration uncertainty, recent user requests and preferences.
- Temporal context: last K=8 or 16 steps **plus** explicit object memory (track last seen, last observation confidence, expiration TTL); encode camera rotation separately to distinguish leaving view from being removed from the world.

**Output:** A *joint* choice from `SILENCE` or `(candidate_event, SPEAK|INTERRUPT|HAPTIC)`; associated urgency, message slot selection and uncertainty level. Add `INSUFFICIENT_VIEW` as a distinct reliability state which can trigger a cautious message (not a hidden fifth physical action). No model is authorized to affirm a route is safe merely because its confidence is high.

**Targets:** independently judged `should_alert`, allowable start window, expected comprehension/action usefulness, minimum semantic slots, priority relations when cues compete, and known perception limits. Four-action labels are state-dependent: `INTERRUPT` is meaningful only if the channel is currently busy, and `HAPTIC` requires a suitable device and an agreed tactile code. These labels cannot be inferred directly from ordinary single-frame object datasets.

### Candidate encoder and global decision

Let `x[i,t]` be the 29-d current feature vector for candidate i, optionally augmented with text/goal embeddings and measured uncertainty. Keep a backwards-compatible 29-feature baseline while testing an expanded v2 representation. Encode a K-step window per track with a GRU, then jointly compare all available candidates plus global channel state with cross-candidate attention or a small MLP scorer and legality mask.

```
Camera/OCR/Depth/IMU (pretrained) -> Track graph with missing-data masks
                                     | -> world-memory ledger (persistent, confidence decays)
                                     v
                                K-step GRU per event
                                     v
                     attention over candidate events + speech channel + goal
                                     v
                         joint policy [SILENCE, (event, modality)]
                                     v
                   runtime constraints + message composer + latency monitoring
                         | SPEAK / INTERRUPT / HAPTIC / SILENCE
                         + insufficent-view report where appropriate
```

**Important:** an explicit memory ledger is needed. The GRU cannot be assumed to remember a person or obstacle indefinitely after camera pans away. Persist events with timestamps, decay belief, discard unsupported old evidence, and test re-appearance/re-identification. This is a representation of uncertain world state, not proof that unseen hazards remain at old positions.

## Supervised imitation / DAgger / reinforcement learning

1. **Imitation**: Start with CampusBench, but distinctly label oracle-derived data `synthetic_reference`. Freeze a subset of real human-verified clips. Fit supervised action classification, urgency regression, and message-slot BCE; log independent class counts and label uncertainty. Train against single-frame/explicit-memory/GRU baselines. Do not use human gold test labels for model selection.
2. **DAgger**: Generate full simulation episode rollouts using the *current learned policy*. Include channel occupancy, prior alerts, model-visited states and stochastic observation errors in state. Obtain reference labels with access to the same policy state, but no access to future observations at model inference. Human adjudication can inform/replace simulator scoring rules; never claim synthetic oracle equals human agreement. Aggregate repeatedly while logging action legality.
3. **REINFORCE**: Fine-tune in a closed-loop simulator, not by trial-and-error human navigation. Reward timely, independently defined usefulness and comprehension; penalize unsupported reassuring advice, late critical cues, inappropriate interruptions, channel occupation and redundant speech. Stress multiple events, speech latency, TTS queue, depth errors and occlusion. REINFORCE is high variance; compare against imitation-only and optionally a lower-variance actor-critic baseline when justified.
4. **Calibration**: Train a separate confidence/uncertainty predictor on measured perception residuals from real held-out training/validation sessions. Report calibration error and selective risk/coverage, plus missed important events at each abstention threshold. An abstention can also miss a critical hazard; compare both costs.

A toy reward for *simulation research only*:

`r_t = +1.0*timely_useful_event - 1.5*missed_critical - 0.4*unsupported_assertion - 0.15*seconds_spoken - 0.2*bad_interrupt + 0.1*justified_abstention`

These weights are **illustrative**, not empirical user preferences. Expert elicitation and validation must determine scales, and accident outcomes are not a suitable dataset for tuning safety guarantees.

## Research data versus deployment claims

| Data family | Source | What the labels actually support |
| --- | --- | --- |
| Real BLV requests | EgoBlind | User questions/answers, not timing/actions |
| Accessibility captions | GuideDog | Human gold and VLM silver guidance content; no deadlines |
| Social dynamics | JRDB | Boxes/tracks/people actions; sensor geometry not your phone |
| Spatial reliability | Aria Everyday Activities | Wearable sensor pose/geometry, not BLV policy labels |
| OCR/text | TextOCR, HierText | Text transcription/detection, not relevance |
| Outside/crowds | Ego4D, Mapillary Vistas | Scene diversity; separate domain alignment needed |
| Accident anomalies | DADA-2000 | Dashcam incidents, not pedestrian route guidance. **OOD-only** |
| Human communication labels | Approved campus/indoor recordings | Independent event IDs, useful times, content and preferences |
| Counterfactual rewards | Modified CampusBench | Reproducible simulated outcomes under controlled assumptions |

See `source_registry.csv` for URLs and conditions. Always check current licenses. Do not distribute third-party videos without permission; keep participant recordings private.

## An annotation must record

- `clip_id`, `session_id`, `site_id`, `source_dataset`, `source_video_id`, camera orientation, observer goal, annotator ID and method, approval/license, split.
- `event_id` across track identity changes, one-to-many automatic track matches (subject to manual verification), `first_seen_s`, `last_seen_s`, object/sign text, uncertain/conflicting observations.
- `should_alert`, `reason`, `earliest_useful_s`, `ideal_s` (optional), `latest_useful_s`, `reaction_time_basis`, `minimum_information`, `severity`, `risk_interpretation` and any `unknown` outcomes. These are human judgments, not automatic ground truth.
- `channel_busy_s`, `current_message`, `remaining_audio_s`, `audio_budget`, `user_goal`, `permitted_actions`, `action_label`, `cue_text`, `required_slots` when asking a communication-decision question.
- `verified=true` only after actual review. Never auto-promote a model annotation draft to a gold label. Mark ambiguous association as unresolved, exclude from supervised gold, and preserve for error analysis.

## A difficult example: crash visible then camera pans away

A camera captures a damaged car and possible glass fragments. Immediately afterward the phone pans to a doorway and the scene disappears. A pure current-frame policy may behave as if the crash has ceased to exist. The model should retain **uncertain observations**, not simply hallucinate a safe detour. Track memory must (a) keep evidence for a bounded time, (b) track viewpoint change, (c) predict possible ongoing relevance, (d) hedge or abstain when exact geometry is not verified, and (e) avoid giving unsupported route instructions. Evaluate using recorded/simulated examples only, with no hazardous user trials.

## Generalization protocol

- **Primary**: train on ordinary indoor hallways, stairs, doors, signage, and temporary obstacles; validate on distinct sessions; final test on new buildings, people, and capture conditions.
- **Transfer**: separately report airport/transit wayfinding, multilingual signs, sidewalk hazards. Where no genuine labels exist, report qualitative/zero-shot capability only, not accurate performance.
- **OOD challenge**: crashes, smoke, evacuations, complete sensor blindness, misleading signs; score appropriate uncertainty, false reassurance, and avoidance of unsupported navigation instructions. Do not advertise tested safety from simulator results.
- **Comparisons**: confidence threshold, cooldown, heuristic P-CACS, urgency-first, earliest-deadline-first, non-temporal MLP, equal-capacity GRU, GRU + explicit memory, GRU+global attention, RL-tuned scheduler, and (when compatible) TIMELI baseline.
- **Measurements**: timely usefulness at matched audio occupancy, late critical rate, false confident cues, interruption benefit, memory retention after occlusion by duration, OCR/sign comprehension, calibration and selective risk, end-to-end camera-to-audio latency (not only scheduler latency).
- **Statistics**: preregister splits, metrics and hypotheses; cluster bootstrap by original video/session/site; multiple model seeds; independent annotations and BLV/O&M expert consultation with appropriate ethics review.

## Implement in this order

1. Keep P-CACS v1 frozen. Add `scenario_catalog.json`, `sign_taxonomy.json`, and `source_registry.csv` for coverage mapping. `sample_coverage.py` generates test *specifications*; it does not create scenes, labels or videos.
2. Expand CampusBench's scenario `WorldObject` and `EventSpec` for new trainable indoor domains. Add explicit channel contention, camera rotations, memory decay and adversarial uncertainty. Verify the new simulator produces coherent events and action legality.
3. Acquire authorized EgoBlind/GuideDog/JRDB/Aria subsets and collect real indoor walking clips with permission. Build human event/cue annotation tasks. Maintain split and source provenance.
4. Modify video exporter: call `PCACSPolicy._features` with exactly the live channel semantics, store pre-decision vectors, channel state and timestamps; reconcile tracker fragments into gold event IDs; validate windows; no label leakage or automatic gold.
5. Train a simple MLP, original GRU, GRU+explicit memory, and GRU+global decision. Publish confusion matrices by action class, especially interruption and haptic on legal simulated states.
6. Run DAgger and closed-loop RL only with clearly stated simulator reward assumptions. Tune against validation only; evaluate on completely held-out sites and stress conditions.
7. Request an ML faculty/HCI/O&M review of the precise task and experiments, before expanding into street/airport deployments.

## Files in this starter bundle

- `scenario_catalog.json` — 154 non-annotated scenario templates in 22 domains.
- `sign_taxonomy.json` — 89 illustrative strings/pictograms in 13 categories (not language-complete).
- `source_registry.csv` — 12 dataset/source records, with URLs, use and limitations.
- `sample_coverage.py` — deterministic metadata stress-configuration generator.
- `stress_matrix.csv` — 1,232 sampled configuration descriptions, not episodes/training labels.
- `test_catalog.py` — 5 reproducibility and integrity checks.

**Bottom line:** The intellectual contribution is a demonstrated generalization of *what/when/whether to communicate*, not a claim that the model learned all known signs or how canine brains work.
