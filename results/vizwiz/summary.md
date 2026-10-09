# YOLO11n on VizWiz-Classification photos, by image-quality flag

1500 of the 7147 flagged photos (8,900 total; the 1,753 test-pool photos have no flags), sampled with seed 0 so that every flag has up to 187 photos. Detector: yolo11n.pt, imgsz 640, conf 0.25, mps, median latency 91.1 ms. `COCO recall` = share of photos whose ImageNet label has a COCO equivalent (51 of the 200 categories; strict = same object, loose adds desktop computer->tv, stove->oven, book jacket / comic book->book) in which the detector emitted that class at any confidence above the threshold.

| group | photos | detections / photo | zero-detection rate | mean confidence | COCO targets | COCO recall | strict targets | strict recall | top labels |
|---|---|---|---|---|---|---|---|---|---|
| all | 1500 | 1.41 | 0.25 | 0.54 | 893 | 0.47 | 860 | 0.48 | person 285, tv 180, bottle 125, cup 107, cell phone 106 |
| flag:BLR (blur) | 694 | 1.36 | 0.26 | 0.52 | 434 | 0.45 | 415 | 0.46 | person 131, tv 102, laptop 62, bottle 53, cup 52 |
| only:BLR (blur) | 45 | 1.33 | 0.22 | 0.53 | 28 | 0.39 | 27 | 0.41 | person 11, bottle 8, tv 4, cup 4, dining table 4 |
| flag:BRT (too bright) | 232 | 1.22 | 0.29 | 0.54 | 169 | 0.44 | 164 | 0.45 | tv 42, person 32, laptop 31, cell phone 23, keyboard 14 |
| only:BRT (too bright) | 8 | 0.75 | 0.62 | 0.52 | 7 | 0.29 | 7 | 0.29 | tv 2, cup 1, dining table 1, microwave 1, person 1 |
| flag:DRK (too dark) | 214 | 1.24 | 0.30 | 0.54 | 147 | 0.54 | 144 | 0.54 | person 39, tv 38, laptop 25, bottle 18, cup 12 |
| only:DRK (too dark) | 12 | 1.58 | 0.17 | 0.49 | 9 | 0.33 | 9 | 0.33 | person 6, bottle 2, teddy bear 1, tv 1, laptop 1 |
| flag:FRM (improper framing) | 909 | 1.30 | 0.28 | 0.53 | 547 | 0.42 | 529 | 0.44 | person 172, tv 112, laptop 65, keyboard 65, book 61 |
| only:FRM (improper framing) | 47 | 1.57 | 0.26 | 0.54 | 29 | 0.48 | 28 | 0.50 | person 11, cup 6, laptop 4, dining table 4, cell phone 3 |
| flag:OBS (obstruction) | 169 | 1.59 | 0.21 | 0.49 | 92 | 0.45 | 86 | 0.47 | person 53, tv 18, bottle 18, laptop 11, chair 11 |
| only:OBS (obstruction) | 12 | 1.67 | 0.33 | 0.51 | 5 | 0.40 | 5 | 0.40 | person 4, bottle 3, refrigerator 2, handbag 1, tv 1 |
| flag:ROT (rotation) | 351 | 1.17 | 0.30 | 0.49 | 195 | 0.32 | 185 | 0.32 | person 53, book 39, tv 31, bottle 20, cell phone 20 |
| only:ROT (rotation) | 34 | 1.18 | 0.29 | 0.48 | 18 | 0.33 | 18 | 0.33 | tv 4, person 4, book 4, bed 2, clock 2 |
| flag:OTH (other) | 18 | 1.44 | 0.28 | 0.52 | 9 | 0.56 | 8 | 0.62 | person 4, toothbrush 2, tv 2, cell phone 2, cup 2 |
| only:OTH (other) | 0 | - | - | - | 0 | - | 0 | - |  |
| flag:NON (no flaw) | 610 | 1.61 | 0.20 | 0.57 | 333 | 0.59 | 322 | 0.60 | person 118, bottle 70, tv 64, cup 59, cell phone 52 |
| only:NON (no flaw) | 156 | 1.81 | 0.20 | 0.58 | 84 | 0.51 | 81 | 0.53 | bottle 23, person 23, dining table 22, cup 19, cell phone 13 |
| issues:0 | 156 | 1.81 | 0.20 | 0.58 | 84 | 0.51 | 81 | 0.53 | bottle 23, person 23, dining table 22, cup 19, cell phone 13 |
| issues:1 | 476 | 1.55 | 0.22 | 0.55 | 268 | 0.57 | 260 | 0.59 | person 106, bottle 55, tv 49, cup 46, cell phone 41 |
| issues:2 | 548 | 1.28 | 0.27 | 0.53 | 335 | 0.45 | 322 | 0.46 | person 100, tv 81, book 46, cell phone 39, laptop 38 |
| issues:3+ | 320 | 1.21 | 0.32 | 0.49 | 206 | 0.35 | 197 | 0.36 | person 56, tv 42, laptop 33, bottle 20, book 19 |

Reading: `flag:X` counts a photo in every flag it carries (most photos carry two: 3,111 of 7,147 in the population), `only:X` keeps the photos with exactly that flag, `issues:k` groups by the number of non-NON flags. VizWiz photos are hand-held close-ups of objects, not walking scenes, so the detections-per-photo column measures what the detector sees of a blind user's framing, not hazards; the recall columns are the robustness signal (same object, worse capture).

Sources: images https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/{name}, annotations.json https://storage.googleapis.com/vw-c/annotations.json, classification_gt.json https://vizwiz.cs.colorado.edu/VizWiz_all_answers/classification_gt.json; licence CC BY 4.0; cite Bafghi & Gurari, CVPR 2023.
