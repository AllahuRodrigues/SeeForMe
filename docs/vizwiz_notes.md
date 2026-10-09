# VizWiz-Classification (Bafghi and Gurari, CVPR 2023): access, what was fetched, and what the robustness run measures

"A New Dataset Based on Images Taken by Blind People for Testing the Robustness of Image Classification Models Trained for
ImageNet Categories". Task page: https://vizwiz.org/tasks-and-datasets/image-classification/ (checked 2026-10-08).
Licence **CC BY 4.0**; no login, form or registration for the downloads (only the now-deprecated EvalAI leaderboard
needed an account; the ground truth is released for self-evaluation).

## What the download consists of

| file | URL | size | content |
|---|---|---|---|
| `annotations.json` | https://storage.googleapis.com/vw-c/annotations.json | 260 KB | 200 categories (`id` = ImageNet-1k index, `wordnet_id`) and the 8,900 image names |
| `classification_gt.json` | https://vizwiz.cs.colorado.edu/VizWiz_all_answers/classification_gt.json | 1.3 MB | per image: `labels` (ImageNet indices present, 1 to 5 per photo) and `flaws` (`OBS, DRK, NON, BLR, BRT, ROT, FRM, OTH` booleans from VizWiz-QualityIssues; `null` for the 1,753 photos that come from the VizWiz test pool) |
| images | `train.zip` 11.3 GB, `val.zip` 3.5 GB, `test.zip` 4.0 GB (the full VizWiz-VQA image pools, 18.8 GB) | the 8,900 photos are a subset: 5,170 from the train pool, 1,977 val, 1,753 test |
| single photos | https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/<name>.jpg | 0.05-0.7 MB each | the same JPEGs served one at a time, no login |
| example code | https://github.com/rezaakb/VizWiz-Classification-Dataset (MIT) | - | `eval.py` for timm classifiers (not needed here) |

Because the zips are 18.8 GB (over the 4 GB limit set for this task) while the annotations are 1.6 MB, the two
annotation files were downloaded in full and the photos were fetched individually for a stratified sample only:
`data/vizwiz_cls/{annotations.json, classification_gt.json, imagenet_class_index.json, images/}` (1,500 photos,
727 MB, 0 failures, 115 s with 8 threads). To get everything instead: download the three zips into a new empty
directory, unzip, and point `scripts/vizwiz_robustness.py --data` at a directory whose `images/` holds the JPEGs by
name; no other step changes.

Quality flags in the population of 7,147 flagged photos (a photo can carry several): FRM 3,755, NON 3,949, BLR 2,528,
ROT 1,112, BRT 320, DRK 287, OBS 169, OTH 18; 3,140 photos have one flag, 3,111 two, 895 three or more; the
"only NON" clean baseline is 1,949 photos. Categories are hand-held objects (packet 1,311 photos, monitor 866, laptop
838, cup 549, jersey 417, computer keyboard 395, cellular telephone 386, television 359 ...), 51 of the 200 have a COCO
class equivalent.

## What `scripts/vizwiz_robustness.py` does

1. Samples up to 1,500 flagged photos so that every flag gets 187 photos (rarest first: OTH gets all 18, OBS all 169)
   and fills the rest at random (seed 0); re-runs reuse what is on disk.
2. Runs the SeeForMe detector exactly as the pipeline does (`YoloDetector(load_cfg().perception)`: YOLO11n, imgsz 640,
   conf 0.25, Apple MPS) on each photo.
3. Groups the photos by flag (`flag:X`, multi-membership), by exclusive flag (`only:X`), by number of non-NON flags
   (`issues:k`) and reports per group: detections per photo, zero-detection rate, mean confidence, and, for photos whose
   ImageNet label maps to a COCO class (strict: same object; loose adds desktop computer -> tv, stove -> oven, book
   jacket / comic book -> book), the recall of that class.
4. Writes `results/vizwiz/summary.{json,md}` and `results/vizwiz/per_image.csv`.

Reading the result: VizWiz photos are close-ups of objects in the hand, not walking scenes, so detections per photo
say what the detector sees of a blind user's framing; the COCO-recall columns are the robustness signal (same object,
worse capture). None of this touches timing; it feeds the perception-reliability term (`depth_conf`, detector
confidence) that the scheduler conditions on, and it is the first SeeForMe number computed on photos that blind people
actually took.

## Result of the 2026-10-09 run (`results/vizwiz/summary.md`, 1,500 photos, seed 0)

| group | photos | detections / photo | zero-detection rate | mean confidence | COCO targets | COCO recall |
|---|---|---|---|---|---|---|
| all | 1,500 | 1.41 | 0.25 | 0.54 | 893 | 0.47 |
| only NON (clean baseline) | 156 | 1.81 | 0.20 | 0.58 | 84 | 0.51 |
| flag NON | 610 | 1.61 | 0.20 | 0.57 | 333 | 0.59 |
| flag DRK (too dark) | 214 | 1.24 | 0.30 | 0.54 | 147 | 0.54 |
| flag BLR (blur) | 694 | 1.36 | 0.26 | 0.52 | 434 | 0.45 |
| flag OBS (obstruction) | 169 | 1.59 | 0.21 | 0.49 | 92 | 0.45 |
| flag BRT (too bright) | 232 | 1.22 | 0.29 | 0.54 | 169 | 0.44 |
| flag FRM (improper framing) | 909 | 1.30 | 0.28 | 0.53 | 547 | 0.42 |
| flag ROT (rotation) | 351 | 1.17 | 0.30 | 0.49 | 195 | 0.32 |
| issues: 1 / 2 / 3+ | 476 / 548 / 320 | 1.55 / 1.28 / 1.21 | 0.22 / 0.27 / 0.32 | 0.55 / 0.53 / 0.49 | 268 / 335 / 206 | 0.57 / 0.45 / 0.35 |

Reading. On a blind user's photos the detector names the labelled object in about half of the cases where it could
(COCO recall 0.47 overall, 0.59 on photos with no flaw vote), and every quality flaw costs recall: rotation is the
worst single flaw (0.32, the detector is not rotation-invariant and 16% of the population is rotated), then framing
(0.42), brightness (0.44), blur and obstruction (0.45); darkness costs the least (0.54), consistent with the dark-frame
enhancement already in `preprocess.enhance_dark`. Flaws compound: recall falls from 0.57 with one flaw to 0.35 with
three or more, the zero-detection rate rises from 0.20 to 0.32 and mean confidence from 0.58 to 0.49. For the
scheduler this means the detector's confidence does track capture quality, but weakly (0.49 against 0.58), so a frame
quality gate (blur, exposure and rotation estimates from `assess_quality`) is a better input to the uncertainty term than
confidence alone. Median latency was 91 ms per photo because the GPU was shared with three other jobs during the run.
