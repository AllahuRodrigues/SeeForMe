# Announcement gate on EgoBlind development replays

Generated 2026-11-12 16:39 by `python scripts/eval_live_gate.py` (1058 s). Splits: finetune, select (development only; holdout and test were not loaded).

Variants: **bare** is the scheduler alone; **gated** wraps it in `AnnouncementGate` with the default `GateConfig` (operator_filter off: the replays are walking clips, not a desk webcam); **gated + adapter** adds `ActivityDetector` in track-based mode (frame=None) and `TaskAdapter`, both stepped at every 0.5 s decision, with read-mode muting applied to the scheduler's cues before the gate.

Repeats are measured two ways. **Dup rate** (the main measure) is the `campus.metrics` duplicate rate: spoken cues about the same ground-truth object as a cue less than 6 s earlier, over all spoken cues; it does not depend on the gate's rules (only detections whose label an EgoBlind event names, inside that event's window, carry an object id, so repeats about unlabelled objects are not counted). The **label+sector repeat rate** is secondary: the share of spoken cues whose (label, direction sector) was already spoken in the previous 20 s while the object was not at least 1 m closer (sectors at +-20 deg; unknown distance counts as not closer). It restates the gate's own rule, so the gated rows score well on it by construction. Recall, precision and quiet-window violations are pooled over events, cues and windows exactly as in `scripts/run_egoblind.py`; precision is a lower bound (only the questioned objects are labelled).

These replays are walking / hand-held clips: none is a desk-webcam session, so they neither show nor test the live "Person." loop at a desk (operator_filter is off here). That case is covered by `tests/test_live_gate.py` (real tracker and dynamics, the user leaning in with depth jitter).

## Reading

- **gated** vs bare, 6 cache x policy comparisons: no timely event lost; 0 gained; 0 events timely in both runs said later (max shift +0.0 s); quiet-window violations unchanged in 6/6 (more in 0). Cues/min -2% to +2%, airtime -1% to +2%, dups (same object within 6 s) 9 -> 10 (+11%), label+sector repeats (the gate's own rule) 115 -> 115 (+0%); precision not lower in 4/6. The remaining label+sector repeats passed the gate as: new 99, near 5, critical 5, closing 4, reacquired 1, reminder 1.
- **gated + adapter** vs bare, 6 cache x policy comparisons: no timely event lost; 1 gained (bare said it late x1); 0 events timely in both runs said later (max shift +0.0 s); quiet-window violations unchanged in 6/6 (more in 0). Cues/min -1% to +2%, airtime -1% to +2%, dups (same object within 6 s) 9 -> 11 (+22%), label+sector repeats (the gate's own rule) 115 -> 116 (+1%); precision not lower in 3/6. The remaining label+sector repeats passed the gate as: new 99, near 5, critical 5, closing 4, approach 1, reacquired 1, reminder 1.
- Gated cues/min change by policy: `pcacs` coco +0%, ow +0%; `pcacs_learned:rl` coco +2%, ow -1%; `egoblind_learned:il@models/egoblind_only` coco -2%, ow +1%. Label+sector repeats without the gate, summed over the caches: `pcacs` 6, `pcacs_learned:rl` 55, `egoblind_learned:il@models/egoblind_only` 54.
- 99 of the 115 label+sector repeats without the gate (86%) are about a different track id than the cue they repeat (tracker id switches, or a second object with the same label in the same direction): the scheduler's per-track memory cannot see them, the gate's identity memory can.
- The adapter keeps more cues than the plain gate in 3/6 comparisons: on these replays it is in navigate mode (background rule off, 20 s hold) for 88% of the decision steps (track-based activity; the adapter lines give how often it changes mode).
- Without the gate the schedulers say the same object again within 6 s 0.00-0.04 times per minute on these replays (0.01-0.29 label+sector repeats per minute); they are hand-held / head-worn clips, mostly walking, where objects leave the view quickly. None of them is a desk-webcam session (the user's own body in view for minutes, operator_filter on), so they do not show the live "Person." loop and say nothing about how much the gate saves there: they measure what the gate costs (lost or delayed events) and removes on real walking footage.

## COCO detector cache (yolo11n; the live demo runs the same COCO classes with yolo11s)

138 videos, 102.8 min, 83 should-speak events (38 coverable by this detector), 56 quiet windows (scenario files sha256 c319d3b6bd2b8a5a, newest 2026-10-09 12:24:04).

### `pcacs`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.132 (5/38) | 0.060 (5/83) | 0.104 | 0.036 (2/56) | 0.47 | **0.021** | 1 | 0.021 | 1 | 0.017 | 1.70 | 0.01 | -0.371 |
| gated | 0.132 (5/38) | 0.060 (5/83) | 0.104 | 0.036 (2/56) | 0.47 | **0.021** | 1 | 0.021 | 1 | 0.017 | 1.70 | 0.01 | -0.371 |
| gated + adapter | 0.132 (5/38) | 0.060 (5/83) | 0.104 | 0.036 (2/56) | 0.47 | **0.021** | 1 | 0.021 | 1 | 0.017 | 1.70 | 0.01 | -0.371 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 49 | 0 | 0 | 0 | 0 | new 30, critical 18, haptic 1 |
| gated + adapter | 49 | 0 | 0 | 0 | 0 | new 30, critical 18, haptic 1 |

- bare: 1 dups (6 s); 1 label+sector repeats, 0 on the same track id
- gated: 1 dups (6 s); 1 label+sector repeats, 0 on the same track id; the gate passed them as: new 1
- gated + adapter: 1 dups (6 s); 1 label+sector repeats, 0 on the same track id; the gate passed them as: new 1

Adapter: decision steps per mode navigate 89%, scene 11%; activity walking 57%, unknown 30%, stationary 11%, turning 2%; 147 mode changes (1.43/min); 79 scene / read episodes (median 4.7 s, max 91.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `pcacs_learned:rl`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.079 (3/38) | 0.036 (3/83) | 0.029 | 0.036 (2/56) | 1.02 | **0.019** | 2 | 0.238 | 25 | 0.026 | 2.64 | 0.27 | -0.447 |
| gated | 0.079 (3/38) | 0.036 (3/83) | 0.028 | 0.036 (2/56) | 1.04 | **0.019** | 2 | 0.252 | 27 | 0.026 | 2.68 | 0.23 | -0.449 |
| gated + adapter | 0.079 (3/38) | 0.036 (3/83) | 0.028 | 0.036 (2/56) | 1.04 | **0.019** | 2 | 0.252 | 27 | 0.026 | 2.68 | 0.24 | -0.449 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 131 | 1 | 0 | 0 | 3 | new 100, haptic 16, critical 13, haptic_gap 3, near (deferred) 2, critical_gap 1 |
| gated + adapter | 132 | 1 | 0 | 0 | 3 | new 100, haptic 17, critical 13, haptic_gap 3, near (deferred) 2, critical_gap 1 |

- bare: 2 dups (6 s); 25 label+sector repeats, 3 on the same track id
- gated: 2 dups (6 s); 27 label+sector repeats, 3 on the same track id; the gate passed them as: new 25, near 1, critical 1; haptics held by the gate: haptic_gap 3, critical_gap 1
- gated + adapter: 2 dups (6 s); 27 label+sector repeats, 3 on the same track id; the gate passed them as: new 25, near 1, critical 1; haptics held by the gate: haptic_gap 3, critical_gap 1

Adapter: decision steps per mode navigate 89%, scene 11%; activity walking 57%, unknown 30%, stationary 11%, turning 2%; 147 mode changes (1.43/min); 79 scene / read episodes (median 4.7 s, max 91.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 3 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 3 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `egoblind_learned:il@models/egoblind_only`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.237 (9/38) | 0.108 (9/83) | 0.050 | 0.071 (4/56) | 1.75 | **0.000** | 0 | 0.150 | 27 | 0.038 | 3.92 | 0.00 | -0.504 |
| gated | 0.237 (9/38) | 0.108 (9/83) | 0.051 | 0.071 (4/56) | 1.72 | **0.000** | 0 | 0.153 | 27 | 0.038 | 3.88 | 0.00 | -0.501 |
| gated + adapter | 0.263 (10/38) | 0.120 (10/83) | 0.055 | 0.071 (4/56) | 1.77 | **0.005** | 1 | 0.154 | 28 | 0.039 | 3.96 | 0.00 | -0.501 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 177 | 4 | 3 | 0 | 0 | new 166, repeat 4, critical 3, warmup 2, new (deferred) 2, background 1, closer 1, closing (deferred) 1, near (deferred) 1, reacquired 1, reminder 1, closing 1 |
| gated + adapter | 182 | 5 | 1 | 0 | 0 | new 167, repeat 5, critical 4, new (deferred) 3, reminder 2, approach (deferred) 1, closer 1, closing (deferred) 1, warmup 1, near (deferred) 1, reacquired 1, closing 1 |

- bare: 0 dups (6 s); 27 label+sector repeats, 2 on the same track id
- gated: 0 dups (6 s); 27 label+sector repeats, 3 on the same track id; the gate passed them as: new 23, closing 1, near 1, critical 1, reacquired 1
- gated + adapter: 1 dups (6 s); 28 label+sector repeats, 4 on the same track id; the gate passed them as: new 23, approach 1, closing 1, near 1, critical 1, reacquired 1

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
| gated | 0.076 (5/66) | 0.060 (5/83) | 0.104 | 0.036 (2/56) | 0.47 | **0.083** | 4 | 0.104 | 5 | 0.016 | 1.68 | 0.02 | -0.365 |
| gated + adapter | 0.076 (5/66) | 0.060 (5/83) | 0.102 | 0.036 (2/56) | 0.48 | **0.082** | 4 | 0.102 | 5 | 0.017 | 1.71 | 0.02 | -0.367 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 50 | 0 | 0 | 0 | 0 | new 34, critical 16 |
| gated + adapter | 51 | 0 | 0 | 0 | 0 | new 34, critical 17 |

- bare: 4 dups (6 s); 5 label+sector repeats, 0 on the same track id
- gated: 4 dups (6 s); 5 label+sector repeats, 0 on the same track id; the gate passed them as: new 3, critical 2
- gated + adapter: 4 dups (6 s); 5 label+sector repeats, 0 on the same track id; the gate passed them as: new 3, critical 2

Adapter: decision steps per mode navigate 87%, scene 13%; activity walking 53%, unknown 31%, stationary 13%, turning 3%; 204 mode changes (1.98/min); 110 scene / read episodes (median 4.2 s, max 73.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `pcacs_learned:rl`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.076 (5/66) | 0.060 (5/83) | 0.045 | 0.018 (1/56) | 1.07 | **0.018** | 2 | 0.273 | 30 | 0.028 | 2.85 | 0.20 | -0.431 |
| gated | 0.076 (5/66) | 0.060 (5/83) | 0.046 | 0.018 (1/56) | 1.06 | **0.018** | 2 | 0.257 | 28 | 0.028 | 2.83 | 0.17 | -0.430 |
| gated + adapter | 0.076 (5/66) | 0.060 (5/83) | 0.046 | 0.018 (1/56) | 1.06 | **0.018** | 2 | 0.257 | 28 | 0.028 | 2.83 | 0.16 | -0.430 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 126 | 8 | 0 | 0 | 4 | new 100, haptic 13, repeat 8, critical 6, haptic_gap 4, closing (deferred) 3, new (deferred) 2, near (deferred) 2 |
| gated + adapter | 125 | 8 | 0 | 0 | 4 | new 100, haptic 12, repeat 8, critical 6, haptic_gap 4, closing (deferred) 3, new (deferred) 2, near (deferred) 2 |

- bare: 2 dups (6 s); 30 label+sector repeats, 10 on the same track id
- gated: 2 dups (6 s); 28 label+sector repeats, 7 on the same track id; the gate passed them as: new 23, closing 3, near 1, critical 1; haptics held by the gate: haptic_gap 4
- gated + adapter: 2 dups (6 s); 28 label+sector repeats, 7 on the same track id; the gate passed them as: new 23, closing 3, near 1, critical 1; haptics held by the gate: haptic_gap 4

Adapter: decision steps per mode navigate 87%, scene 13%; activity walking 53%, unknown 31%, stationary 13%, turning 3%; 204 mode changes (1.98/min); 110 scene / read episodes (median 4.2 s, max 73.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 5 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

### `egoblind_learned:il@models/egoblind_only`

| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | **dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | airtime (min) | haptics/min | score (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare | 0.061 (4/66) | 0.048 (4/83) | 0.025 | 0.018 (1/56) | 1.58 | **0.000** | 0 | 0.167 | 27 | 0.035 | 3.59 | 0.00 | -0.513 |
| gated | 0.061 (4/66) | 0.048 (4/83) | 0.025 | 0.018 (1/56) | 1.59 | **0.006** | 1 | 0.166 | 27 | 0.035 | 3.61 | 0.00 | -0.516 |
| gated + adapter | 0.061 (4/66) | 0.048 (4/83) | 0.024 | 0.018 (1/56) | 1.60 | **0.006** | 1 | 0.165 | 27 | 0.035 | 3.63 | 0.00 | -0.517 |

| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | held: haptic | pass / hold reasons |
|---|---|---|---|---|---|---|
| gated | 163 | 1 | 2 | 0 | 0 | new 154, near (deferred) 4, warmup 2, critical 2, repeat 1, sector 1, closer 1, reminder 1 |
| gated + adapter | 164 | 2 | 0 | 0 | 0 | new 155, near (deferred) 4, critical 2, repeat 2, sector 1, closer 1, reminder 1 |

- bare: 0 dups (6 s); 27 label+sector repeats, 1 on the same track id
- gated: 1 dups (6 s); 27 label+sector repeats, 1 on the same track id; the gate passed them as: new 24, near 2, reminder 1
- gated + adapter: 1 dups (6 s); 27 label+sector repeats, 1 on the same track id; the gate passed them as: new 24, near 2, reminder 1

Adapter: decision steps per mode navigate 87%, scene 13%; activity walking 53%, unknown 31%, stationary 13%, turning 3%; 204 mode changes (1.98/min); 110 scene / read episodes (median 4.2 s, max 73.2 s), each entry rebaselines the gate; 0 cues muted by read mode (read mode cannot trigger here: the cached replays have no OCR text tracks).

**gated vs bare:** 0 timely event(s) lost, 0 gained. Of the 4 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

**gated + adapter vs bare:** 0 timely event(s) lost, 0 gained. Of the 4 events timely in both, 0 were said later and 0 earlier than bare (mean shift +0.00 s, max +0.00 s).

## Per comparison

- coco / `pcacs` / gated: cues/min 0.47 -> 0.47 (+0%), airtime 1.70 -> 1.70 min (+0%), dup rate 0.021 -> 0.021 (1 -> 1 dups), label+sector repeat rate 0.021 -> 0.021 (1 -> 1), coverable recall 0.132 -> 0.132, precision 0.104 -> 0.104, quiet-window violations 2 -> 2; no timely event lost.
- coco / `pcacs` / gated + adapter: cues/min 0.47 -> 0.47 (+0%), airtime 1.70 -> 1.70 min (+0%), dup rate 0.021 -> 0.021 (1 -> 1 dups), label+sector repeat rate 0.021 -> 0.021 (1 -> 1), coverable recall 0.132 -> 0.132, precision 0.104 -> 0.104, quiet-window violations 2 -> 2; no timely event lost.
- coco / `pcacs_learned:rl` / gated: cues/min 1.02 -> 1.04 (+2%), airtime 2.64 -> 2.68 min (+2%), dup rate 0.019 -> 0.019 (2 -> 2 dups), label+sector repeat rate 0.238 -> 0.252 (25 -> 27), coverable recall 0.079 -> 0.079, precision 0.029 -> 0.028, quiet-window violations 2 -> 2; no timely event lost.
- coco / `pcacs_learned:rl` / gated + adapter: cues/min 1.02 -> 1.04 (+2%), airtime 2.64 -> 2.68 min (+2%), dup rate 0.019 -> 0.019 (2 -> 2 dups), label+sector repeat rate 0.238 -> 0.252 (25 -> 27), coverable recall 0.079 -> 0.079, precision 0.029 -> 0.028, quiet-window violations 2 -> 2; no timely event lost.
- coco / `egoblind_learned:il@models/egoblind_only` / gated: cues/min 1.75 -> 1.72 (-2%), airtime 3.92 -> 3.88 min (-1%), dup rate 0.000 -> 0.000 (0 -> 0 dups), label+sector repeat rate 0.150 -> 0.153 (27 -> 27), coverable recall 0.237 -> 0.237, precision 0.050 -> 0.051, quiet-window violations 4 -> 4; no timely event lost.
- coco / `egoblind_learned:il@models/egoblind_only` / gated + adapter: cues/min 1.75 -> 1.77 (+1%), airtime 3.92 -> 3.96 min (+1%), dup rate 0.000 -> 0.005 (0 -> 1 dups), label+sector repeat rate 0.150 -> 0.154 (27 -> 28), coverable recall 0.237 -> 0.263, precision 0.050 -> 0.055, quiet-window violations 4 -> 4; no timely event lost, 1 gained.
- ow / `pcacs` / gated: cues/min 0.47 -> 0.47 (+0%), airtime 1.68 -> 1.68 min (+0%), dup rate 0.083 -> 0.083 (4 -> 4 dups), label+sector repeat rate 0.104 -> 0.104 (5 -> 5), coverable recall 0.076 -> 0.076, precision 0.104 -> 0.104, quiet-window violations 2 -> 2; no timely event lost.
- ow / `pcacs` / gated + adapter: cues/min 0.47 -> 0.48 (+2%), airtime 1.68 -> 1.71 min (+1%), dup rate 0.083 -> 0.082 (4 -> 4 dups), label+sector repeat rate 0.104 -> 0.102 (5 -> 5), coverable recall 0.076 -> 0.076, precision 0.104 -> 0.102, quiet-window violations 2 -> 2; no timely event lost.
- ow / `pcacs_learned:rl` / gated: cues/min 1.07 -> 1.06 (-1%), airtime 2.85 -> 2.83 min (-1%), dup rate 0.018 -> 0.018 (2 -> 2 dups), label+sector repeat rate 0.273 -> 0.257 (30 -> 28), coverable recall 0.076 -> 0.076, precision 0.045 -> 0.046, quiet-window violations 1 -> 1; no timely event lost.
- ow / `pcacs_learned:rl` / gated + adapter: cues/min 1.07 -> 1.06 (-1%), airtime 2.85 -> 2.83 min (-1%), dup rate 0.018 -> 0.018 (2 -> 2 dups), label+sector repeat rate 0.273 -> 0.257 (30 -> 28), coverable recall 0.076 -> 0.076, precision 0.045 -> 0.046, quiet-window violations 1 -> 1; no timely event lost.
- ow / `egoblind_learned:il@models/egoblind_only` / gated: cues/min 1.58 -> 1.59 (+1%), airtime 3.59 -> 3.61 min (+1%), dup rate 0.000 -> 0.006 (0 -> 1 dups), label+sector repeat rate 0.167 -> 0.166 (27 -> 27), coverable recall 0.061 -> 0.061, precision 0.025 -> 0.025, quiet-window violations 1 -> 1; no timely event lost.
- ow / `egoblind_learned:il@models/egoblind_only` / gated + adapter: cues/min 1.58 -> 1.60 (+1%), airtime 3.59 -> 3.63 min (+1%), dup rate 0.000 -> 0.006 (0 -> 1 dups), label+sector repeat rate 0.167 -> 0.165 (27 -> 27), coverable recall 0.061 -> 0.061, precision 0.025 -> 0.024, quiet-window violations 1 -> 1; no timely event lost.

