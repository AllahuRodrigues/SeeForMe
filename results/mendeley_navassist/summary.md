# Mendeley 'Dataset for Deep Learning Based Navigation Assistance for the Visually Impaired People'

DOI 10.17632/66bjx8h79p.1 (v1, 2025-03-21), East West University (Dhaka), licence CC BY 4.0. Files: `data/mendeley_navassist/extracted/dataset.csv` (10.4 MB), `data/mendeley_navassist/extracted/dataset.json` (72.0 MB). Features extracted from recorded street video with a custom YOLO plus a monocular depth model; no images or video are included, so nothing here can be replayed through SeeForMe's own perception. The authors' stated purpose is to train a sequence model (RNN / LSTM / GRU) that predicts the movement decision.

## Schema

* CSV: 35261 rows x 39 columns: `image_file`, 16 per-frame counts (`total_object`, `total_{left,center,right}_object`, `moving_away_*`, `approaching_*`, `*_in_4m`, `*_in_2m`), the three nearest objects as `{0,1,2}_index_objects_{class_name, estimated_distance, avg_depth_object, nearest_object_depth, angle, posistion, movement_status}`, and `decision`.
* JSON: 35261 records with the same keys plus `object_collection`, a list of up to 20 objects (`class_id`, `class_name`, `estimated_distance`, `avg_depth_object`, `nearest_object_depth`, `angle`, `object_posistion`, `movement_status`). Objects per record: 0: 1366, 1: 5021, 2: 8869, 3: 7882, 4: 5752, 5: 3163, 6: 1455, 7: 736, 8: 403, 9: 248, 10: 153, 11: 124, 12: 52, 13: 21, 14: 4, 15: 2, 16: 2, 17: 3, 18: 2, 19: 1, 20: 2.
* Consistency: same row count True, same decisions True; the CSV's three objects are the first three of the JSON list in 35261 / 35261 rows; the JSON list is sorted by estimated distance in 28874 of 28874 multi-object records, so the CSV's three objects are the three nearest.
* `image_file` is `frame_<n>.jpg` with no video id (7289 distinct names for 35261 records); sequences can only be inferred from the frame number resetting: 50 sequences, length min / median / max 11 / 206.5 / 6926, 2.3% of frame numbers missing (frames without detections were dropped). No timestamps, no frame rate, no object identity across frames.

## Label fields

| decision | n | share |
|---|---|---|
| Move Forward | 24936 | 0.707 |
| Move Right | 3901 | 0.111 |
| Stop | 3224 | 0.091 |
| Move Left | 3200 | 0.091 |

Per object: `movement_status` {'approaching': 50948, 'moving away': 46267, 'unknown': 11406}, `object_posistion` {'center': 42484, 'right': 34578, 'left': 31559}. Numeric ranges (min / p5 / median / p95 / max):

* `estimated_distance`: 0.136 / 0.7 / 7.287 / 28.713 / 398.593
* `avg_depth_object`: 0.452 / 5.868 / 10.295 / 21.241 / 47.532
* `nearest_object_depth`: -6.435 / 3.126 / 7.351 / 11.564 / 33.078
* `angle`: -34.028 / -28.259 / 0.454 / 29.491 / 34.546

`estimated_distance` behaves like metres (median 7.3 m, a long tail of far objects); `avg_depth_object` / `nearest_object_depth` are raw depth-map statistics in model units (the nearest value can be negative); `angle` is degrees from the frame centre, consistent with a roughly 70-degree horizontal field of view.

## Classes

| class (id) | objects | frames | in CSV nearest-3 | SeeForMe label | COCO-80 | event type | severity | how |
|---|---|---|---|---|---|---|---|---|
| Person (14) | 36046 | 19753 | 27072 | person | yes | person_static | 0.5 | same class |
| Rickshaw (16) | 16585 | 11598 | 12281 | bicycle | yes | bike | 0.9 | cycle rickshaw, a pedal tricycle |
| Car (4) | 14152 | 9442 | 8950 | car | yes | vehicle | 1.0 | same class |
| Footpath (9) | 8281 | 7896 | 7972 |  |  |  |  | walkable surface (sidewalk); not an object event |
| Motorcycle (11) | 4714 | 3980 | 3837 | motorcycle | yes | vehicle | 1.0 | same class |
| CNG (3) | 4421 | 3821 | 3181 | motorcycle | yes | vehicle | 1.0 | CNG auto-rickshaw, a motorised three-wheeler |
| Electric Pole (6) | 4400 | 4254 | 3287 | pole |  | hazard_static | 0.7 | simulator / open-vocab label |
| Building Pillar (1) | 3607 | 1822 | 2396 | pole |  | hazard_static | 0.7 | simulator / open-vocab label |
| Bus (2) | 2577 | 2286 | 1777 | bus | yes | vehicle | 1.0 | same class |
| Cycle (5) | 2468 | 2269 | 2288 | bicycle | yes | bike | 0.9 | same class |
| Van gari (21) | 2212 | 2065 | 2141 | bicycle | yes | bike | 0.9 | pedal cargo tricycle |
| Parking Cone (13) | 1458 | 675 | 1245 | cone |  | barrier | 0.75 | simulator / open-vocab label |
| Obstacle (12) | 1198 | 641 | 888 | obstruction |  | unknown_obstacle | 0.6 | UNKNOWN_OBSTACLE_LABELS -> unknown_obstacle |
| Pickup (15) | 1108 | 1035 | 961 | truck | yes | vehicle | 1.0 | pickup truck |
| Van (20) | 956 | 953 | 654 | truck | yes | vehicle | 1.0 | van |
| Leguna (10) | 909 | 897 | 867 | bus | yes | vehicle | 1.0 | Leguna minibus |
| Food cart (8) | 861 | 648 | 658 | cart |  | hazard_static | 0.7 | simulator / open-vocab label |
| Stairs (17) | 843 | 758 | 728 | stairs |  | stairs | 1.0 | simulator / open-vocab label |
| Food Van (7) | 798 | 761 | 753 | truck | yes | vehicle | 1.0 | food truck |
| Bin (0) | 563 | 489 | 454 | trash can |  | hazard_static | 0.7 | simulator / open-vocab label |
| Truck (19) | 351 | 349 | 271 | truck | yes | vehicle | 1.0 | same class |
| Tree (18) | 113 | 89 | 113 | tree |  | hazard_static | 0.7 | open-vocab label |

13 of the 22 classes are COCO-80 classes YOLO11n would output; 21 reach a non-decor P-CACS event through LABEL_EVENT (the vehicle family of Dhaka streets: rickshaw, CNG auto-rickshaw, Leguna minibus, van gari cargo tricycle, is mapped onto the nearest COCO vehicle class).

## Their per-object features against SeeForMe's EventFeatures

`EventFeatures` (29 fields, `seeforme/cues/pcacs.py`) is what P-CACS and the learned scheduler see per candidate. 9 of them have a counterpart in the Mendeley records:

| EventFeatures field | Mendeley counterpart |
|---|---|
| severity | class_name (through LABEL_EVENT / EVENT_SPECS) |
| path_prob | object_posistion left / center / right (coarse) |
| hazard | derivable: severity x centre position |
| urgency | - |
| actionability | class_name |
| goal | - |
| novelty | - |
| uncertainty | - |
| speech_cost | - |
| queue_cost | - |
| confidence | - |
| distance | estimated_distance (m, monocular) |
| depth_conf | - |
| angle | angle (deg from the frame centre) |
| rel_speed | movement_status approaching / moving away (sign only) |
| lateral_speed | - |
| ttc | - |
| growth | - |
| since_announced | - |
| announce_count | - |
| text_importance | - |
| is_text | - |
| moving | movement_status != unknown (approx.) |
| channel_busy | - |
| channel_remaining | - |
| budget | - |
| density | total_object and the *_in_4m / *_in_2m counts |
| own_speed | - |
| ego_speed | - |

What they have that SeeForMe does not: a per-frame object census by side and range (`*_in_2m`, `*_in_4m`, approaching / receding counts), which is a richer `density` than the single scalar. What SeeForMe has that they lack: detector and depth confidence, time to contact and box growth, lateral motion, everything about the speech channel (busy, remaining, budget, queue cost) and announcement history (novelty, since_announced), goal relevance and text. Their motion is a binary approaching / receding flag per object with no tracks, so TTC cannot be recovered from the records.

## Is `decision` a usable weak scheduler target?

| decision | n | share | mean objects | centre object within 2 m | approaching centre object | median nearest centre (m) | no object at all |
|---|---|---|---|---|---|---|---|
| Move Forward | 24936 | 0.707 | 3.04 | 0.15 | 0.42 | 6.660 | 0.06 |
| Move Left | 3200 | 0.091 | 3.41 | 0.18 | 0.37 | 4.820 | 0.00 |
| Move Right | 3901 | 0.111 | 3.58 | 0.21 | 0.38 | 5.370 | 0.00 |
| Stop | 3224 | 0.091 | 2.43 | 1.00 | 0.60 | 1.040 | 0.00 |

* P(not Move Forward | centre object within 2 m) = 0.546; P(not Move Forward | no centre object within 2 m) = 0.213; P(Stop | centre object within 2 m) = 0.381; P(Stop | no centre object within 2 m) = 0.
* A decision tree on the 16 frame-level counts plus the nearest distance per side recovers the label with 5-fold accuracy 0.825 (depth 3), 0.899 (depth 6), 0.894 (depth 12) against a majority-class baseline of 0.707. Depth-3 rules:

```
|--- nearest_left_m <= 2.00
|   |--- nearest_right_m <= 1.58
|   |   |--- nearest_left_m <= 1.08
|   |   |   |--- class: Move Right
|   |   |--- nearest_left_m >  1.08
|   |   |   |--- class: Move Left
|   |--- nearest_right_m >  1.58
|   |   |--- nearest_center_m <= 1.58
|   |   |   |--- class: Move Right
|   |   |--- nearest_center_m >  1.58
|   |   |   |--- class: Move Right
|--- nearest_left_m >  2.00
|   |--- nearest_right_m <= 2.00
|   |   |--- nearest_right_m <= 0.73
|   |   |   |--- class: Move Forward
|   |   |--- nearest_right_m >  0.73
|   |   |   |--- class: Move Left
|   |--- nearest_right_m >  2.00
|   |   |--- nearest_center_m <= 2.00
|   |   |   |--- class: Move Forward
|   |   |--- nearest_center_m >  2.00
|   |   |   |--- class: Move Forward
```

**Verdict.** The decision is a locomotion command (where to move), not a communication decision (whether to speak); it lives in a different action space from P-CACS's silence / speak / interrupt / haptic. It is still usable as a *weak, frame-level* 'something action-relevant is ahead' target: `Stop` and `Move Left` / `Move Right` frames (together 29.3%) mark moments when an instruction was warranted, `Move Forward` frames mark quiet windows, in the same spirit as EgoBlind's present / clear windows. Limits: no timestamps or frame rate, so earliest / ideal / latest windows cannot be placed in seconds; no object identity, so the responsible object must be inferred (the nearest approaching centre object); no images, so the label cannot be attached to SeeForMe's own detections; the labelling procedure is not documented and much of the label is recoverable from the stored features (see the tree above: `Stop` never occurs without a centre object within 2 m), so it adds little information beyond those features; the scene is Dhaka street traffic, and the record does not say who carried the camera or whether blind users were involved. Nothing in it says *when* to speak or what to say.

