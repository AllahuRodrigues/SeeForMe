# Announcement gate on EgoBlind development replays

Generated 2026-10-09 14:35 by `python scripts/eval_live_gate.py` (106 s). Splits: finetune, select (development only; holdout and test were not loaded).

Variants: **bare** is the scheduler alone; **gated** wraps it in `AnnouncementGate` with the default `GateConfig` (operator_filter off: the replays are walking clips, not a desk webcam); **gated + adapter** adds `ActivityDetector` in track-based mode (frame=None) and `TaskAdapter`, both stepped at every 0.5 s decision, with read-mode muting applied to the scheduler's cues before the gate.

Repeats are measured two ways. **Dup rate** (the main measure) is the `campus.metrics` duplicate rate: spoken cues about the same ground-truth object as a cue less than 6 s earlier, over all spoken cues; it does not depend on the gate's rules (only detections whose label an EgoBlind event names, inside that event's window, carry an object id, so repeats about unlabelled objects are not counted). The **label+sector repeat rate** is secondary: the share of spoken cues whose (label, direction sector) was already spoken in the previous 20 s while the object was not at least 1 m closer (sectors at +-20 deg; unknown distance counts as not closer). It restates the gate's own rule, so the gated rows score well on it by construction. Recall, precision and quiet-window violations are pooled over events, cues and windows exactly as in `scripts/run_egoblind.py`; precision is a lower bound (only the questioned objects are labelled).

These replays are walking / hand-held clips: none is a desk-webcam session, so they neither show nor test the live "Person." loop at a desk (operator_filter is off here). That case is covered by `tests/test_live_gate.py` (real tracker and dynamics, the user leaning in with depth jitter).

## Reading

- **gated** vs bare, 6 cache x policy comparisons: no timely event lost; 0 gained; 0 events timely in both runs said later (max shift +0.0 s); quiet-window violations unchanged in 6/6 (more in 0). Cues/min -19% to -4%, airtime -18% to -3%, dups (same object within 6 s) 9 -> 4 (-56%), label+sector repeats (the gate's own rule) 115 -> 57 (-50%); precision not lower in 6/6. The remaining label+sector repeats passed the gate as: new 28, reminder 10, critical 7, near 7, closing 4, closer 1.
- **gated + adapter** vs bare, 6 cache x policy comparisons: no timely event lost; 1 gained (bare said it late x1); 0 events timely in both runs said later (max shift +0.0 s); quiet-window violations unchanged in 6/6 (more in 0). Cues/min -20% to -2%, airtime -20% to -1%, dups (same object within 6 s) 9 -> 5 (-44%), label+sector repeats (the gate's own rule) 115 -> 57 (-50%); precision not lower in 6/6. The remaining label+sector repeats passed the gate as: new 28, reminder 10, near 7, critical 6, closing 4, approach 1, closer 1.
- Gated cues/min change by policy: `pcacs` coco -4%, ow -4%; `pcacs_learned:rl` coco -15%, ow -19%; `egoblind_learned:il@models/egoblind_only` coco -9%, ow -6%. Label+sector repeats without the gate, summed over the caches: `pcacs` 6, `pcacs_learned:rl` 55, `egoblind_learned:il@models/egoblind_only` 54: the gate removes the most where the scheduler repeats itself the most.
- 99 of the 115 label+sector repeats without the gate (86%) are about a different track id than the cue they repeat (tracker id switches, or a second object with the same label in the same direction): the scheduler's per-track memory cannot see them, the gate's identity memory can.
- The adapter keeps more cues than the plain gate in 2/6 comparisons: on these replays it is in navigate mode (background rule off, 20 s hold) for 88% of the decision steps (track-based activity; the adapter lines give how often it changes mode).
- Without the gate the schedulers say the same object again within 6 s 0.00-0.04 times per minute on these replays (0.01-0.29 label+sector repeats per minute); they are hand-held / head-worn clips, mostly walking, where objects leave the view quickly. None of them is a desk-webcam session (the user's own body in view for minutes, operator_filter on), so they do not show the live "Person." loop and say nothing about how much the gate saves there: they measure what the gate costs (lost or delayed events) and removes on real walking footage.

## COCO detector cache (yolo11n; the live demo runs the same COCO classes with yolo11s)

138 videos, 102.8 min, 83 should-speak events (38 coverable by this detector), 56 quiet windows (scenario files sha256 c319d3b6bd2b8a5a, newest 2026-10-09 12:24:04).

### `pcacs`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.132 (5/38) | 0.060 (5/83) | 0.104 | 0.036 (2/56) | 0.47 | **0.021** | 1 | 0.021 | 1 | 0.017 | 1.70 | 0.01 | -0.371 |
| gated | 0.132 (5/38) | 0.060 (5/83) | 0.109 | 0.036 (2/56) | 0.45 | **0.022** | 1 | 0.000 | 0 | 0.016 | 1.65 | 0.01 | -0.363 |
| gated + adapter | 0.132 (5/38) | 0.060 (5/83) | 0.109 | 0.036 (2/56) | 0.45 | **0.022** | 1 | 0.000 | 0 | 0.016 | 1.65 | 0.01 | -0.363 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 47 | 2 | 0 | 0 | 0 | new 28, critical 18, repeat 2, haptic 1 |
| gated + adapter | 47 | 2 | 0 | 0 | 0 | new 28, critical 18, repeat 2, haptic 1 |

- bare: 1 dups (6 s); 1 label+sector repeats, 0 on the same track id
- gated: 1 dups (6 s); 0 label+sector repeats, 0 on the same track id
- gated + adapter: 1 dups (6 s); 0 label+sector repeats, 0 on the same track id

Adapter: decision steps per mode navigate 89%, scene 11%; activity walking 57%, unknown 30%, stationary 11%, turning 2%; 147 mode changes (1.43/min); 79 scene / read episodes (median 4.7 s, max 91.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `pcacs_learned:rl`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.079 (3/38) | 0.036 (3/83) | 0.029 | 0.036 (2/56) | 1.02 | **0.019** | 2 | 0.238 | 25 | 0.026 | 2.64 | 0.27 | -0.447 |
| gated | 0.079 (3/38) | 0.036 (3/83) | 0.034 | 0.036 (2/56) | 0.87 | **0.000** | 0 | 0.090 | 8 | 0.021 | 2.19 | 0.19 | -0.413 |
| gated + adapter | 0.079 (3/38) | 0.036 (3/83) | 0.034 | 0.036 (2/56) | 0.87 | **0.000** | 0 | 0.090 | 8 | 0.021 | 2.19 | 0.19 | -0.413 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 109 | 18 | 1 | 0 | 3 | new 80, repeat 17, haptic 13, critical 12, haptic_gap 3, warmup 1, new (deferred) 1, critical_gap 1, closer 1, critical (deferred) 1, closing 1 |
| gated + adapter | 109 | 18 | 0 | 0 | 3 | new 81, repeat 17, haptic 13, critical 12, haptic_gap 3, critical_gap 1, closer 1, critical (deferred) 1, closing 1 |

- bare: 2 dups (6 s); 25 label+sector repeats, 3 on the same track id
- gated: 0 dups (6 s); 8 label+sector repeats, 0 on the same track id; the gate passed them as: new 5, critical 2, closing 1; haptics held by the gate: haptic_gap 3, critical_gap 1
- gated + adapter: 0 dups (6 s); 8 label+sector repeats, 0 on the same track id; the gate passed them as: new 5, critical 2, closing 1; haptics held by the gate: haptic_gap 3, critical_gap 1

Adapter: decision steps per mode navigate 89%, scene 11%; activity walking 57%, unknown 30%, stationary 11%, turning 2%; 147 mode changes (1.43/min); 79 scene / read episodes (median 4.7 s, max 91.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 3 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 3 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `egoblind_learned:il@models/egoblind_only`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.237 (9/38) | 0.108 (9/83) | 0.050 | 0.071 (4/56) | 1.75 | **0.000** | 0 | 0.150 | 27 | 0.038 | 3.92 | 0.00 | -0.504 |
| gated | 0.237 (9/38) | 0.108 (9/83) | 0.055 | 0.071 (4/56) | 1.59 | **0.000** | 0 | 0.092 | 15 | 0.034 | 3.51 | 0.00 | -0.473 |
| gated + adapter | 0.263 (10/38) | 0.120 (10/83) | 0.059 | 0.071 (4/56) | 1.64 | **0.006** | 1 | 0.095 | 16 | 0.035 | 3.62 | 0.00 | -0.475 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 163 | 21 | 3 | 0 | 0 | new 144, repeat 21, reminder 6, closer 4, near (deferred) 4, critical 3, warmup 2, closing (deferred) 1, background 1, new (deferred) 1 |
| gated + adapter | 169 | 20 | 1 | 0 | 0 | new 147, repeat 20, reminder 7, critical 4, closer 4, near (deferred) 4, approach (deferred) 1, closing (deferred) 1, warmup 1, new (deferred) 1 |

- bare: 0 dups (6 s); 27 label+sector repeats, 2 on the same track id
- gated: 0 dups (6 s); 15 label+sector repeats, 3 on the same track id; the gate passed them as: new 8, near 3, reminder 2, closing 1, critical 1
- gated + adapter: 1 dups (6 s); 16 label+sector repeats, 4 on the same track id; the gate passed them as: new 8, near 3, reminder 2, approach 1, closing 1, critical 1

Adapter: decision steps per mode navigate 89%, scene 11%; activity walking 57%, unknown 30%, stationary 11%, turning 2%; 147 mode changes (1.43/min); 79 scene / read episodes (median 4.7 s, max 91.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 9 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 1 gained. Of the 9 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).
- gained: video 472, person (navigation, coverable), window 40.6-51.0 s: bare said 'Person.' at 55.2 s; gated + adapter said 'Person.' at 51.0 s. Why: the bare run said it late (t=55.2 s); adapter mode navigate (walking).

## open-vocabulary detector cache (yolov8s-worldv2)

138 videos, 102.8 min, 83 should-speak events (66 coverable by this detector), 56 quiet windows (scenario files sha256 c319d3b6bd2b8a5a, newest 2026-10-09 12:24:04).

### `pcacs`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.076 (5/66) | 0.060 (5/83) | 0.104 | 0.036 (2/56) | 0.47 | **0.083** | 4 | 0.104 | 5 | 0.016 | 1.68 | 0.02 | -0.365 |
| gated | 0.076 (5/66) | 0.060 (5/83) | 0.109 | 0.036 (2/56) | 0.45 | **0.065** | 3 | 0.065 | 3 | 0.016 | 1.64 | 0.02 | -0.360 |
| gated + adapter | 0.076 (5/66) | 0.060 (5/83) | 0.106 | 0.036 (2/56) | 0.46 | **0.064** | 3 | 0.064 | 3 | 0.016 | 1.66 | 0.02 | -0.361 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 48 | 2 | 0 | 0 | 0 | new 31, critical 16, repeat 2, reminder 1 |
| gated + adapter | 49 | 2 | 0 | 0 | 0 | new 31, critical 17, repeat 2, reminder 1 |

- bare: 4 dups (6 s); 5 label+sector repeats, 0 on the same track id
- gated: 3 dups (6 s); 3 label+sector repeats, 0 on the same track id; the gate passed them as: critical 2, reminder 1
- gated + adapter: 3 dups (6 s); 3 label+sector repeats, 0 on the same track id; the gate passed them as: critical 2, reminder 1

Adapter: decision steps per mode navigate 87%, scene 13%; activity walking 53%, unknown 31%, stationary 13%, turning 3%; 204 mode changes (1.98/min); 110 scene / read episodes (median 4.2 s, max 73.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `pcacs_learned:rl`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.076 (5/66) | 0.060 (5/83) | 0.045 | 0.018 (1/56) | 1.07 | **0.018** | 2 | 0.273 | 30 | 0.028 | 2.85 | 0.20 | -0.431 |
| gated | 0.076 (5/66) | 0.060 (5/83) | 0.056 | 0.018 (1/56) | 0.87 | **0.000** | 0 | 0.101 | 9 | 0.023 | 2.33 | 0.16 | -0.393 |
| gated + adapter | 0.076 (5/66) | 0.060 (5/83) | 0.057 | 0.018 (1/56) | 0.86 | **0.000** | 0 | 0.102 | 9 | 0.022 | 2.30 | 0.15 | -0.391 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 105 | 22 | 0 | 0 | 4 | new 76, repeat 22, haptic 12, critical 6, reminder 6, closer 4, haptic_gap 4, closing (deferred) 1 |
| gated + adapter | 103 | 22 | 1 | 0 | 4 | new 75, repeat 22, haptic 11, critical 6, reminder 6, closer 4, haptic_gap 4, background 1, closing (deferred) 1 |

- bare: 2 dups (6 s); 30 label+sector repeats, 10 on the same track id
- gated: 0 dups (6 s); 9 label+sector repeats, 3 on the same track id; the gate passed them as: reminder 4, new 2, closer 1, critical 1, closing 1; haptics held by the gate: haptic_gap 4
- gated + adapter: 0 dups (6 s); 9 label+sector repeats, 3 on the same track id; the gate passed them as: reminder 4, new 2, closer 1, critical 1, closing 1; haptics held by the gate: haptic_gap 4

Adapter: decision steps per mode navigate 87%, scene 13%; activity walking 53%, unknown 31%, stationary 13%, turning 3%; 204 mode changes (1.98/min); 110 scene / read episodes (median 4.2 s, max 73.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `egoblind_learned:il@models/egoblind_only`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.061 (4/66) | 0.048 (4/83) | 0.025 | 0.018 (1/56) | 1.58 | **0.000** | 0 | 0.167 | 27 | 0.035 | 3.59 | 0.00 | -0.513 |
| gated | 0.061 (4/66) | 0.048 (4/83) | 0.026 | 0.018 (1/56) | 1.48 | **0.000** | 0 | 0.145 | 22 | 0.033 | 3.39 | 0.00 | -0.500 |
| gated + adapter | 0.061 (4/66) | 0.048 (4/83) | 0.026 | 0.018 (1/56) | 1.48 | **0.000** | 0 | 0.138 | 21 | 0.033 | 3.39 | 0.00 | -0.500 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 152 | 10 | 3 | 0 | 0 | new 134, repeat 10, reminder 5, closer 4, warmup 2, critical 2, near 2, near (deferred) 2, critical (deferred) 1, closing 1, background 1, new (deferred) 1 |
| gated + adapter | 152 | 11 | 0 | 0 | 0 | new 136, repeat 11, reminder 5, closer 4, critical 2, near 2, near (deferred) 2, closing 1 |

- bare: 0 dups (6 s); 27 label+sector repeats, 1 on the same track id
- gated: 0 dups (6 s); 22 label+sector repeats, 3 on the same track id; the gate passed them as: new 13, near 4, reminder 3, critical 1, closing 1
- gated + adapter: 0 dups (6 s); 21 label+sector repeats, 2 on the same track id; the gate passed them as: new 13, near 4, reminder 3, closing 1

Adapter: decision steps per mode navigate 87%, scene 13%; activity walking 53%, unknown 31%, stationary 13%, turning 3%; 204 mode changes (1.98/min); 110 scene / read episodes (median 4.2 s, max 73.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 4 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 4 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

## Per comparison

- coco / `pcacs` / gated: cues/min 0.47 -> 0.45 (-4%), airtime 1.70 -> 1.65 min (-3%), dup rate 0.021 -> 0.022 (1 -> 1 dups), label+sector repeat rate 0.021 -> 0.000 (1 -> 0), coverable recall 0.132 -> 0.132, precision 0.104 -> 0.109, quiet-window violations 2 -> 2; no timely event lost.
- coco / `pcacs` / gated + adapter: cues/min 0.47 -> 0.45 (-4%), airtime 1.70 -> 1.65 min (-3%), dup rate 0.021 -> 0.022 (1 -> 1 dups), label+sector repeat rate 0.021 -> 0.000 (1 -> 0), coverable recall 0.132 -> 0.132, precision 0.104 -> 0.109, quiet-window violations 2 -> 2; no timely event lost.
- coco / `pcacs_learned:rl` / gated: cues/min 1.02 -> 0.87 (-15%), airtime 2.64 -> 2.19 min (-17%), dup rate 0.019 -> 0.000 (2 -> 0 dups), label+sector repeat rate 0.238 -> 0.090 (25 -> 8), coverable recall 0.079 -> 0.079, precision 0.029 -> 0.034, quiet-window violations 2 -> 2; no timely event lost.
- coco / `pcacs_learned:rl` / gated + adapter: cues/min 1.02 -> 0.87 (-15%), airtime 2.64 -> 2.19 min (-17%), dup rate 0.019 -> 0.000 (2 -> 0 dups), label+sector repeat rate 0.238 -> 0.090 (25 -> 8), coverable recall 0.079 -> 0.079, precision 0.029 -> 0.034, quiet-window violations 2 -> 2; no timely event lost.
- coco / `egoblind_learned:il@models/egoblind_only` / gated: cues/min 1.75 -> 1.59 (-9%), airtime 3.92 -> 3.51 min (-10%), dup rate 0.000 -> 0.000 (0 -> 0 dups), label+sector repeat rate 0.150 -> 0.092 (27 -> 15), coverable recall 0.237 -> 0.237, precision 0.050 -> 0.055, quiet-window violations 4 -> 4; no timely event lost.
- coco / `egoblind_learned:il@models/egoblind_only` / gated + adapter: cues/min 1.75 -> 1.64 (-6%), airtime 3.92 -> 3.62 min (-8%), dup rate 0.000 -> 0.006 (0 -> 1 dups), label+sector repeat rate 0.150 -> 0.095 (27 -> 16), coverable recall 0.237 -> 0.263, precision 0.050 -> 0.059, quiet-window violations 4 -> 4; no timely event lost, 1 gained.
- ow / `pcacs` / gated: cues/min 0.47 -> 0.45 (-4%), airtime 1.68 -> 1.64 min (-3%), dup rate 0.083 -> 0.065 (4 -> 3 dups), label+sector repeat rate 0.104 -> 0.065 (5 -> 3), coverable recall 0.076 -> 0.076, precision 0.104 -> 0.109, quiet-window violations 2 -> 2; no timely event lost.
- ow / `pcacs` / gated + adapter: cues/min 0.47 -> 0.46 (-2%), airtime 1.68 -> 1.66 min (-1%), dup rate 0.083 -> 0.064 (4 -> 3 dups), label+sector repeat rate 0.104 -> 0.064 (5 -> 3), coverable recall 0.076 -> 0.076, precision 0.104 -> 0.106, quiet-window violations 2 -> 2; no timely event lost.
- ow / `pcacs_learned:rl` / gated: cues/min 1.07 -> 0.87 (-19%), airtime 2.85 -> 2.33 min (-18%), dup rate 0.018 -> 0.000 (2 -> 0 dups), label+sector repeat rate 0.273 -> 0.101 (30 -> 9), coverable recall 0.076 -> 0.076, precision 0.045 -> 0.056, quiet-window violations 1 -> 1; no timely event lost.
- ow / `pcacs_learned:rl` / gated + adapter: cues/min 1.07 -> 0.86 (-20%), airtime 2.85 -> 2.30 min (-20%), dup rate 0.018 -> 0.000 (2 -> 0 dups), label+sector repeat rate 0.273 -> 0.102 (30 -> 9), coverable recall 0.076 -> 0.076, precision 0.045 -> 0.057, quiet-window violations 1 -> 1; no timely event lost.
- ow / `egoblind_learned:il@models/egoblind_only` / gated: cues/min 1.58 -> 1.48 (-6%), airtime 3.59 -> 3.39 min (-5%), dup rate 0.000 -> 0.000 (0 -> 0 dups), label+sector repeat rate 0.167 -> 0.145 (27 -> 22), coverable recall 0.061 -> 0.061, precision 0.025 -> 0.026, quiet-window violations 1 -> 1; no timely event lost.
- ow / `egoblind_learned:il@models/egoblind_only` / gated + adapter: cues/min 1.58 -> 1.48 (-6%), airtime 3.59 -> 3.39 min (-6%), dup rate 0.000 -> 0.000 (0 -> 0 dups), label+sector repeat rate 0.167 -> 0.138 (27 -> 21), coverable recall 0.061 -> 0.061, precision 0.025 -> 0.026, quiet-window violations 1 -> 1; no timely event lost.

