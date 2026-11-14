# Live detector choice: `yolo11n.pt` vs `yolo11s.pt`

Host: Apple M1 Max, macOS 26.5.1, torch 2.14.0, ultralytics 8.4.155, device `mps`, imgsz 640, generated 2026-10-09 11:54 by `scripts/eval_live_detector.py`.

## Recommendation

**Live default: `yolo11s.pt`** (detector confidence floor `perception.yolo_conf = 0.25`, unchanged). yolo11s.pt has the higher direct-class F1 at every threshold and its full pipeline runs at 54 ms per frame (median) <= 120 ms.

Rule: the live default is the more accurate model (higher direct-class micro F1 on the BLV-Road-Nav keyframes at every threshold scored) if its full per-frame pipeline time (median) stays under 120 ms, otherwise `yolo11n.pt`.

* conf 0.25: direct-class F1 0.863 vs 0.895 (yolo11s - yolo11n = +0.032, 95% CI [+0.013, +0.052] excludes 0); recall 0.801 vs 0.853, precision 0.937 vs 0.942
* conf 0.35: direct-class F1 0.855 vs 0.888 (yolo11s - yolo11n = +0.033, 95% CI [+0.014, +0.056] excludes 0); recall 0.775 vs 0.828, precision 0.952 vs 0.956
* `yolo11n.pt`: detector median 12.3 ms (p95 17.2); full pipeline median 51.1 ms incl. resize (end-to-end p95 61.8); conservative bound 75.0 ms
* `yolo11s.pt`: detector median 17.6 ms (p95 26.2); full pipeline median 54.0 ms incl. resize (end-to-end p95 65.8); conservative bound 80.3 ms
* person boxes on consecutive EgoBlind frames 0.1 s apart (conf 0.25): persistence 0.885 (116/131) vs 0.920 (127/138), presence flip rate 0.078 (7/90) vs 0.033 (3/90) (`yolo11n.pt`, `yolo11s.pt`; 10 clips, so indicative only); a person who drops out and comes back as a new track can be announced again
* raising the detector floor from 0.25 to 0.35 does not help that: person flip rate `yolo11n.pt` 0.078 -> 0.144, `yolo11s.pt` 0.033 -> 0.144; keep perception.yolo_conf = 0.25 (the scheduler already ignores tracks under its own min_confidence)

## 1. Latency on MPS

Detector only (`YoloDetector.detect`, wall time including NMS and the copy of the boxes to the CPU) on 100 EgoBlind development frames (10 clips of the finetune + select splits, resized to 640 px wide like a camera frame), 10 warm-up frames per model, 3 rounds interleaved frame by frame with alternating order.

| model | frames x rounds | median ms | p95 ms | mean ms | per-round medians | BLV keyframes median / p95 ms |
|---|---|---|---|---|---|---|
| `yolo11n.pt` | 300 | 12.28 | 17.23 | 13.07 | 12.14, 12.04, 12.89 | 13.12 / 18.54 |
| `yolo11s.pt` | 300 | 17.63 | 26.24 | 18.74 | 17.37, 17.63, 18.24 | 19.41 / 27.19 |

Resizing a camera-size frame to 640 px: median 2.61 ms (p95 25.85).

Full `LivePipeline.process` per frame on the same frames: quality check, detector, depth (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, input 308 px), object distances, IoU tracker with dynamics, scheduler `pcacs_learned:rl`, no OCR, no speech; 3 rounds interleaved. The budget column adds the resize.

| model | detect med / p95 | depth med / p95 | track med | scheduler med / p95 | end-to-end med / p95 | + resize (median) | fits 120 ms |
|---|---|---|---|---|---|---|---|
| `yolo11n.pt` | 12.83 / 16.56 | 34.90 / 44.80 | 0.09 | 0.00 / 0.70 | 48.50 / 61.77 | 51.11 | yes |
| `yolo11s.pt` | 16.70 / 21.55 | 33.93 / 44.76 | 0.09 | 0.00 / 0.68 | 51.38 / 65.77 | 53.99 | yes |

Conservative bound without the measured depth (detector median + Depth Anything at the top of its 30-60 ms range + tracker and scheduler + resize):

* `yolo11n.pt`: 74.97 ms
* `yolo11s.pt`: 80.33 ms

## 2. Accuracy: BLV-Road-Nav-Accessibility keyframes

671 labelled keyframes from 31 segments of blind pedestrians' street videos; presence labels (0 / 1 per object and keyframe, no boxes). A frame counts as a detection of an object when one of its COCO classes has a box at or above the threshold. Direct objects (12): Bench, Bicycle, Bus, Car, Chair, Dog, Fire hydrant, Motorcycle, Person, Stop Sign, Table, Traffic Signals. Scoring is `blvnav_taxonomy.score` (same code as `results/blvnav/detector_eval.*`); the frames are resized to 640 px wide first, like a camera frame. The difference column is a paired bootstrap over segments (95% CI).

### confidence >= 0.25

| model | TP | FP | FN | precision | recall | F1 | macro F1 (objects with >= 10 frames) | paper-style recall (N = 12) |
|---|---|---|---|---|---|---|---|---|
| `yolo11n.pt` | 919 | 62 | 229 | 0.937 | 0.801 | 0.863 | 0.624 | 0.127 |
| `yolo11s.pt` | 979 | 60 | 169 | 0.942 | 0.853 | 0.895 | 0.721 | 0.136 |
| yolo11s - yolo11n | | | | +0.005 [-0.023, +0.041] | +0.052 [+0.032, +0.075] | +0.032 [+0.013, +0.052] | | |

| object | frames present | `yolo11n` P / R / F1 | `yolo11s` P / R / F1 |
|---|---|---|---|
| Person | 615 | 0.99 / 0.98 / 0.98 | 0.98 / 0.98 / 0.98 |
| Car | 274 | 0.97 / 0.84 / 0.90 | 0.98 / 0.93 / 0.96 |
| Bench | 76 | 0.36 / 0.18 / 0.24 | 0.59 / 0.22 / 0.32 |
| Traffic Signals | 69 | 0.93 / 0.41 / 0.57 | 0.91 / 0.56 / 0.70 |
| Bicycle | 45 | 0.65 / 0.24 / 0.35 | 0.72 / 0.47 / 0.57 |
| Bus | 38 | 0.86 / 0.79 / 0.82 | 0.90 / 0.95 / 0.92 |
| Motorcycle | 12 | 0.62 / 0.42 / 0.50 | 0.75 / 0.50 / 0.60 |
| Table | 9 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| Stop Sign | 6 |  / 0.00 /  | 0.00 / 0.00 / 0.00 |
| Chair | 4 | 0.00 / 0.00 / 0.00 | 0.14 / 0.50 / 0.22 |
| Dog | 0 | 0.00 /  /  |  /  /  |
| Fire hydrant | 0 |  /  /  | 0.00 /  /  |

### confidence >= 0.35

| model | TP | FP | FN | precision | recall | F1 | macro F1 (objects with >= 10 frames) | paper-style recall (N = 12) |
|---|---|---|---|---|---|---|---|---|
| `yolo11n.pt` | 890 | 45 | 258 | 0.952 | 0.775 | 0.855 | 0.580 | 0.123 |
| `yolo11s.pt` | 951 | 44 | 197 | 0.956 | 0.828 | 0.888 | 0.686 | 0.132 |
| yolo11s - yolo11n | | | | +0.004 [-0.021, +0.037] | +0.053 [+0.030, +0.082] | +0.033 [+0.014, +0.056] | | |

| object | frames present | `yolo11n` P / R / F1 | `yolo11s` P / R / F1 |
|---|---|---|---|
| Person | 615 | 0.99 / 0.97 / 0.98 | 0.99 / 0.98 / 0.98 |
| Car | 274 | 0.98 / 0.80 / 0.88 | 0.98 / 0.89 / 0.94 |
| Bench | 76 | 0.35 / 0.16 / 0.22 | 0.56 / 0.12 / 0.20 |
| Traffic Signals | 69 | 1.00 / 0.35 / 0.52 | 0.94 / 0.49 / 0.65 |
| Bicycle | 45 | 0.60 / 0.13 / 0.22 | 0.78 / 0.40 / 0.53 |
| Bus | 38 | 0.84 / 0.71 / 0.77 | 0.90 / 0.92 / 0.91 |
| Motorcycle | 12 | 0.80 / 0.33 / 0.47 | 0.75 / 0.50 / 0.60 |
| Table | 9 |  / 0.00 /  | 0.00 / 0.00 / 0.00 |
| Stop Sign | 6 |  / 0.00 /  |  / 0.00 /  |
| Chair | 4 | 0.00 / 0.00 / 0.00 | 0.18 / 0.50 / 0.27 |
| Dog | 0 | 0.00 /  /  |  /  /  |
| Fire hydrant | 0 |  /  /  | 0.00 /  /  |

## 3. Agreement on the EgoBlind development frames

First the 100 timing frames (consecutive frames of 10 clips), then a broad sample of every development clip. Matched = same class, IoU >= 0.5, one-to-one. Persistence = share of a model's boxes that still have a same-class box with IoU >= 0.3 (the tracker's association threshold) in the next sampled frame of the clip, 0.1 s later; flip rate = share of consecutive frame pairs where the class appears or disappears. Head-mounted walking footage, so some misses are real motion.

### The 100 timing frames, confidence >= 0.25

| class | boxes `yolo11n` | boxes `yolo11s` | matched | only `yolo11n` | only `yolo11s` | frames `yolo11n` | frames `yolo11s` | frame agreement | persistence `yolo11n` / `yolo11s` | flip rate `yolo11n` / `yolo11s` |
|---|---|---|---|---|---|---|---|---|---|---|
| person | 148 | 154 | 109 | 39 | 45 | 94 | 96 | 0.960 | 0.885 / 0.920 | 0.078 / 0.033 |
| car | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |  /  | 0.000 / 0.000 |
| chair | 20 | 33 | 18 | 2 | 15 | 16 | 21 | 0.950 | 0.778 / 0.821 | 0.022 / 0.033 |
| bicycle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |  /  | 0.000 / 0.000 |

### The 100 timing frames, confidence >= 0.35

| class | boxes `yolo11n` | boxes `yolo11s` | matched | only `yolo11n` | only `yolo11s` | frames `yolo11n` | frames `yolo11s` | frame agreement | persistence `yolo11n` / `yolo11s` | flip rate `yolo11n` / `yolo11s` |
|---|---|---|---|---|---|---|---|---|---|---|
| person | 122 | 120 | 94 | 28 | 26 | 87 | 85 | 0.840 | 0.872 / 0.898 | 0.144 / 0.144 |
| car | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |  /  | 0.000 / 0.000 |
| chair | 15 | 27 | 14 | 1 | 13 | 14 | 20 | 0.940 | 0.846 / 0.833 | 0.022 / 0.033 |
| bicycle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |  /  | 0.000 / 0.000 |

### Broad sample: 417 frames (0.25, 0.5, 0.75 of the duration) of all 139 development clips

| conf | class | boxes `yolo11n` | boxes `yolo11s` | matched (IoU >= 0.5) | only `yolo11n` | only `yolo11s` | frames `yolo11n` | frames `yolo11s` | frame agreement |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | person | 1131 | 1245 | 916 | 215 | 329 | 348 | 353 | 0.945 |
| 0.25 | car | 164 | 180 | 121 | 43 | 59 | 66 | 64 | 0.947 |
| 0.25 | chair | 56 | 71 | 32 | 24 | 39 | 30 | 34 | 0.918 |
| 0.25 | bicycle | 4 | 11 | 3 | 1 | 8 | 3 | 10 | 0.983 |
| 0.35 | person | 885 | 1017 | 753 | 132 | 264 | 329 | 338 | 0.921 |
| 0.35 | car | 117 | 136 | 96 | 21 | 40 | 52 | 60 | 0.962 |
| 0.35 | chair | 39 | 48 | 19 | 20 | 29 | 21 | 25 | 0.942 |
| 0.35 | bicycle | 2 | 7 | 2 | 0 | 5 | 2 | 6 | 0.990 |

## Frames used

EgoBlind development clips only (finetune, select); the holdout and official test splits are never opened. finetune:434, finetune:715, select:184, select:471, select:472, finetune:665, finetune:121, select:348, finetune:440, select:469.

## Caveats

* BLV-Road-Nav keyframes are outdoor street scenes; the live complaint is a webcam on a desk indoors. Indoor classes (chair, dining table) have only a handful of labelled frames there, so the per-object rows for them are not informative; the EgoBlind agreement and persistence tables are the indoor-relevant evidence.
* Presence labels have no boxes, so a detection anywhere in the frame counts; precision is therefore an upper bound for box-level precision.
* The full-pipeline timing excludes OCR (EasyOCR every 8th frame adds roughly 85-130 ms on those frames, results/latency.json) and speech; both cost the same for either detector.
* Latency was measured on frames already in memory; camera capture runs in parallel with inference in the live loop.
* The EgoBlind perception cache (data/egoblind/cache, the replays behind run_on_egoblind and the holdout numbers) was built with yolo11n.pt; changing the live detector does not change any replay result, but the live demo then runs a different detector from the replays (yolo11s finds somewhat more boxes, mainly people, chairs and cars).
* All timings are one machine (arm64, 26.5.1) on one day; MPS timings vary by a few ms between runs, so the per-round medians are reported to show the spread.

Reproduce: `OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_detector.py`.

## Review notes (2026-10-09, independent review of this script)

* The "persistence" and "flip rate" numbers compare single detections 0.1 s apart. The tracker bridges gaps of up to
  8 missed frames (`TrackerConfig.max_age`), so they do not show that a flickering person becomes a new track and is
  announced again; they only show that yolo11s finds people more consistently frame to frame. Keeping
  `yolo_conf = 0.25` stands for a simpler reason: a lower floor feeds the tracker more detections.
* The pipeline benchmark's 100 frames produced no spoken cue with either model, so it times the silent path; the
  cue path costs well under a millisecond (`decision step latency` in `make live-check`).
* On OCR frames (every 8th frame, +85-130 ms) a frame takes 140-185 ms with either detector; the median above is
  for the other frames. The live demo runs depth every 2 frames by default (`--depth-every 2`), which brings the
  median frame to about 40-45 ms (measured 20-26 fps on the desk replays).
* `scripts/run_live.py` resolves the weights to an absolute path in the repository, so starting the demo from another
  directory cannot trigger a download.
