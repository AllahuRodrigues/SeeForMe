# Mobilio (Liu and Slade, Nature Biomedical Engineering 2026): what the repository holds and what SeeForMe can use

Paper: "Improving navigation for people with blindness using an AI-driven smartphone application and personalized audio
guidance", Raymond Liu and Patrick Slade (Harvard SEAS, Slade Lab). Press release 2026-08-25. Headline result from the
abstract: with Mobilio plus a white cane, 14 blind and low-vision participants navigated a community path 13 +/- 3% faster
and made 41 +/- 5% fewer environmental contacts than with Google Maps plus a cane.

Repository: https://github.com/Harvard-Slade-Lab/Mobilio, cloned 2026-10-08 (single commit `9dd54b8 Init`) into
`external/Mobilio` (172 MB, gitignored). **Licence: MIT** (Copyright 2025 Harvard Slade Lab), so code and data can be
reused with attribution. The app itself (Unity 2023.2.13f1, ARFoundation 5.1.2, ARKit / ARCore XR plugins 5.1.2, Unity
Sentis 1.6.0-pre1, UniTask 2.5.5) is **not** in the repository: only four algorithm scripts and the experiment logs are.

## Repository layout

```
external/Mobilio/
  README.md, LICENSE (MIT)
  scripts/            4 C# MonoBehaviours, 2,298 lines; no Unity project, models, audio clips or shaders
    DepthImage.cs     obstacle detection and avoidance from ARFoundation depth (867 lines)
    Vision.cs         MobileSeg-Tiny semantic segmentation -> walkable-path guidance (645 lines)
    Navigation.cs     Google Routes polyline + steps, waypoint logic, spatial-audio heading cue, TTS (570 lines)
    GPSData.cs        GPS polling, magnetic declination, compass correction by AR-to-GPS alignment (216 lines)
  data/README.md
  data/VI01 .. VI15   14 participants (VI13 absent), 7.5-15 MB each, 4,788 JSON files in total
    data/CMA.json                 audio-personalisation optimiser trace (200 turns)
    data/data_0 .. data_339.json  340 trial files (one trial = one JSON)
    manual.txt                    hand-recorded condition order, corrections and contacts
    points.json                   6 runs x 5 waypoint positions in the AR world frame (metres)
```

The README mentions a `validation_dataset` folder with a box plot of the walking results; it is not in the repository.
Nothing references the MobileSeg model weights, the audio clips (`Resources/Audio/heading`, collision beep), the `TTS`,
`SensorData`, `WebClient`, `MagDec`, `TextureResizer` or `VisionUtils` classes the scripts call, so the scripts document
the algorithms but do not build.

## The four algorithms, with the constants that matter

### 1. Obstacle detection and avoidance (`DepthImage.cs`)

* **Input.** ARFoundation environment depth (`AROcclusionManager`, DepthFloat32 on iPhone 12 Pro at 256 x 192, DepthUInt16
  on a OnePlus 11 at 160 x 90) plus the depth-confidence image; pixels with confidence below 0.51 (normalised) are dropped.
  Intrinsics are rescaled to the depth-map size; every remaining pixel is back-projected to a camera-frame vertex and
  transformed to the AR world frame.
* **Occupancy grid.** A sparse 3-D voxel grid (`Dictionary<Vector3, float>`, voxel 0.10 m) accumulates hit counts for
  points within 4 m (`searchRadius`). Each depth frame, voxels that lie fully inside the camera frustum are decayed
  (`count *= 0.9`), voxels farther than 5 m from the user are deleted, and a voxel counts as blocked when its count
  exceeds 2 (`pointsToBlockCell`). Voxel height must fall between the estimated ground (initially `-(height - 0.3)`,
  re-estimated as the mean height of blocked voxels within 0.3 m horizontally below the phone, plus 0.4 m padding, capped
  at -0.5 m) and ground + person height (default 1.7 m, user-settable).
* **Obstacle test.** Blocked voxels in the corridor `0 < z < 1.5 m` ahead (`distanceToObstacle`) and `|x| < 0.2 m`
  (`personRadius`) are counted; **2 or more** (`cellsToBlockPerson`) means "obstacle ahead" and the nearest one sets the
  distance. The grid is cleared when the app pauses, because the AR pose may be invalid afterwards.
* **Avoidance direction.** When blocked, the voxels are rasterised into an 81 x 81 2-D search grid (0.1 m cells, +/-4 m)
  in the user's heading frame; the user's own footprint is unblocked; rays (Amanatides-Woo grid traversal) are cast to a
  point 2 m away (`paddedDistance`) at 0, +2, -2, +4, -4, ... degrees up to +/-90. The first obstacle-free ray wins;
  otherwise the longest free ray, pushed out to at least +/-30 degrees. The result is multiplied by `avoidScale` (1.0).
* **Audio.** Either a collision beep whose repetition rate rises from 2 Hz at 1.5 m to 1 / clip length (about 11.7 Hz) at
  0.2 m, spatialised by placing the audio source 1 m from the user in the avoidance direction, or (default,
  `useHeadingAudio`) the heading cue of `Navigation.PlayOrientationAudio` pointed at the avoidance direction. Multiple
  `AudioSource`s are pre-scheduled on the DSP clock so beeps do not stall at low frame rates.

### 2. Path guidance from segmentation (`Vision.cs`)

* MobileSeg-Tiny (ONNX, 341 layers) on a 640 x 480 crop, output 160 x 120, 9 classes: background, terrain, road, curb,
  curb cut, sidewalk, plain crosswalk, zebra crosswalk, covering (manhole / grating). Inference is spread over frames
  (at most 180 layers or 80 ms per frame) to keep the app above 10 FPS.
* Ground-plane cells (0.2 m, radius 15 m) around the user are projected into the segmentation to build a 2-D walkability
  grid (`LaxWalkable` = curb .. covering, plus road when `roadWalkable`). Rays at 1-degree steps within +/-90 degrees of
  the GPS waypoint bearing return their walkable length (up to 2 skipped non-walkable cells); candidates longer than 5 m
  are merged into branches and the branch closest to the waypoint bearing is accepted if within 30 degrees
  (`maxDisparity`), replacing the raw GPS bearing for the heading cue for 1 s (`validDuration`). Side branches longer
  than the median + 3.5 m along the accepted path are reported as upcoming turns.
* Surface-class changes under the user (road / sidewalk / crosswalk) are announced by audio clip with a 3 s minimum
  interval (`classChangeInterval`).

### 3. GPS waypoint navigation and the heading cue (`Navigation.cs`, `GPSData.cs`)

* Google Routes API polyline decoded into waypoints; step start points carry the spoken instruction ("Step k: turn
  right onto ..."). Waypoint selection: within 5 m of a waypoint it is "reached" (`closeRadius`), otherwise the nearest
  segment within 15 m (`farLineDist`) or the nearest waypoint within 15 m (`farRadius`); farther than that the route is
  recalculated. Spoken instructions are rate-limited to one per 2 s (`instructionUpdateInterval`); an orientation
  sentence ("Facing north, head east for 30 meters") is spoken at most every 12 s (`orientationUpdateInterval`) and only
  when no obstacle is ahead.
* **Spatial-audio heading cue** (`PlayOrientationAudio`): the relative heading (bearing to the next waypoint minus compass
  heading, in [-180, 180]) is scaled by `audioScale`, the audio source is placed 1 m away at that angle (clamped to
  +/-90 degrees) so stereo panning conveys left / right, and the clip is chosen from a pitch bank with
  `pitch = 1 - (1 - minPitch) * |heading| / 180` (so straight ahead is the highest pitch), repeated at `audioRate`.
  Defaults `audioRate = 3 Hz, minPitch = 0.5, audioScale = 1.0`; these three numbers are the per-user `params` stored in
  every trial file and the quantity the CMA-ES personalisation optimises. The obstacle cue pre-empts the heading cue
  (`if (!DepthImage.obstacleAhead) PlayOrientationAudio(...)`): a fixed priority rule obstacle > heading > spoken
  orientation.
* GPS at 0.1 m requested accuracy, magnetic declination from a WMM lookup, and a compass offset estimated by a Kabsch /
  SVD best-fit rotation between the last 20 AR-frame positions and their GPS fixes (first 5 ignored, samples at least
  0.3 m apart). The community-path files store the corrected fixes as `latSync / lngSync` (and a second variant
  `latSync2 / lngSync2`).

## Experiment data: exactly what is in the files

All arrays are per-sample time series (sample rate about 30 Hz, from the AR frame loop); `time` is Unity
`Time.realtimeSinceStartup` seconds, so only differences inside one file are meaningful.

| file | trials | columns (JSON keys) and meaning |
|---|---|---|
| `data/data_0 .. data_199.json` | 200 personalisation turns | `user` compass heading of the user (deg), `target` desired heading (deg), `localX`, `localZ` AR-frame position (m), `time` (s), `filename` = `HEADING_<k>`, `params` = `[audioRate, minPitch, audioScale]` used on that turn (optimiser samples) |
| `data/data_200 .. data_263.json` | 64 validation turns | same columns; `params` alternate between the default `[3.0, 0.5, 1.0]` and the personalised triple |
| `data/data_264 .. data_299.json` | 36 community-path legs (6 runs x 6 waypoint legs) | adds `lat`, `lng` raw GPS fixes, `compass` raw heading, `latSync`, `lngSync`, `latSync2`, `lngSync2` corrected fixes (empty lists except in the Mobilio-with-GPS runs). `lat/lng/compass` have one more sample than `time/user/target/localX/localZ` (the GPS arrays are appended before the first frame) |
| `data/data_300 .. data_339.json` | 40 obstacle-course runs (4 blocks x 10 runs) | `user`, `target`, `localX`, `localZ`, `time`, `filename`, `params` (the last file is named `HEADING_340`) |
| `data/CMA.json` | one per participant | `inputs` 200 x 3 normalised parameter samples in [0, 1], `outputs` 200 normalised heading-error costs, `means` 200 x 3 CMA-ES means, `sigmas` 200 step sizes, `seed`, `filename` |
| `manual.txt` | one per participant | line 1: order of the three community conditions as `['3 gmaps', '1 oracle', '2 opt']`; line 2: experimenter corrections for the 6 runs; then four obstacle blocks, each `"<cond>:"` (`1 cane` or `2 opt`) followed by the 10 obstacle placements (two-letter codes such as `CF`, `GB`); finally four rows of 10 contact counts, one row per block |
| `points.json` | one per participant | 6 lists of 5 `[x, z]` AR-world-frame waypoint positions (m), one list per community run (the sixth leg ends at the start) |

Facts checked on the files (14 participants):

* Every participant has exactly 341 JSON files (340 trials + CMA) and all four blocks of 10 obstacle runs.
* The six community runs follow the order `A B C C B A` of the three listed conditions (counterbalanced): under that
  reading the corrected-GPS arrays are non-empty in 26 of the 28 `2 opt` runs and in none of the `1 oracle` or `3 gmaps`
  runs, which is consistent with the README ("runs where participants used Mobilio additionally have data on the
  offsets"). `points.json` gives the waypoints in the AR frame, which is presumably what the `1 oracle` condition used
  instead of GPS. Condition semantics must be confirmed from the paper.
* VI03's community files have no GPS keys and last 35-53 s per run against 125-530 s for everyone else: treat that
  participant's community data as incomplete.
* Obstacle course, contacts per 10-run block: cane only 42.1 +/- 12.9, Mobilio (`2 opt`) 24.9 +/- 12.3; mean
  per-participant change -41% (n = 14), which reproduces the paper's "41 +/- 5% fewer environmental contacts". Run
  duration from `time`: cane 13.7 s, Mobilio 16.3 s (median 13.8 vs 14.8 s): the app trades a little speed for contacts.
* Community path, summed leg durations per run: `3 gmaps` 230 s, `1 oracle` 195 s (-12% vs Google Maps, matching the
  paper's -13 +/- 3%), `2 opt` 230 s (no gain under the A B C C B A reading). Experimenter corrections per run: gmaps 0.79,
  oracle 0.14, opt 0.21.

These numbers come from `external/Mobilio/data` as cloned; they are not in any script of this repository.

## What SeeForMe can reuse or compare against

| Mobilio element | use in SeeForMe |
|---|---|
| **Environmental contacts per run** (hand-counted) and **completion time** | the only published real-user outcome metrics for a smartphone obstacle-avoidance app with n = 14 blind users; SeeForMe's simulated `regret` and quiet-window violations have no field counterpart yet, so a CampusBench-to-field study should count the same two things. Mobilio's within-subject design (A B C C B A, 10-run obstacle blocks) is a ready protocol |
| **Experimenter corrections per run** | an analogue of SeeForMe's "missed event" (the user needed help the system did not give) |
| **Fixed priority obstacle > heading > spoken orientation, cooldowns 2 s / 3 s / 12 s** | the simplest possible scheduler: a baseline to put next to `confidence_cooldown` and P-CACS in the CampusBench tables (CampusBench's `contention` kind is exactly the case this rule cannot handle: two hazards, one channel) |
| **Continuous non-speech channel** (3 Hz pitched beeps, stereo-panned) | Mobilio never competes with itself because heading is carried by a tone, not words; SeeForMe's `haptics` channel plays the same role. Their personalisation (CMA-ES over rate, minimum pitch, gain on 200 one-turn trials, normalised heading error as cost) is a template for tuning haptic / earcon parameters per user |
| **Occupancy grid with hit counts, 0.9 decay and 2-hit threshold; obstacle = 2+ voxels in a 1.5 m x 0.4 m corridor** | a depth-only obstacle detector that is class-agnostic: a drop-in for the `unknown_obstacle` event type and for the half of EgoBlind events that are not COCO classes (doors, poles, curbs). SeeForMe's Depth Anything V2 metric depth could feed the same grid on a plain phone without LiDAR, with the 0.51 confidence gate replaced by `depth_conf` |
| **Avoidance direction by ray sweep, minimum +/-30 degrees** | a concrete "action" slot for `compose()` (`"veer right 30 degrees"`) instead of distance-only cues |
| **Segmentation walkability grid and 30-degree disparity gate against GPS** | not needed by the scheduler, but it is how Mobilio decides a cue is *consistent* before speaking; the equivalent in SeeForMe is the `path_prob` corridor test |
| **GPS / compass alignment by Kabsch fit of AR trajectory to GPS** | relevant only if SeeForMe adds wayfinding to outdoor waypoints |

What Mobilio does not have: no timing labels, no notion of a cue being early or late (the beep is continuous), no speech
budget (orientation speech is simply suppressed while an obstacle is ahead), no object classes, and no video (the data
are trajectories and headings only). It therefore cannot be replayed through SeeForMe's event windows; its value is the
field protocol and outcome metrics, the depth-grid obstacle logic, and the personalised non-speech channel.
