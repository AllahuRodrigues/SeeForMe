# YOLO11n presence detection on the BLV-Road-Nav-Accessibility keyframes

671 labelled keyframes from 31 segments (presence labels 0 / 1 per object and keyframe, 79 annotated objects of the paper's earlier list); detector `yolo11n.pt` on `mps` at imgsz 640, cached in `results/blvnav/detections.jsonl`. A frame counts as a detection of an object when any of its COCO classes is present at or above the confidence threshold (boxes are not scored: the labels have no boxes). Only the objects that map to a COCO class can score; every other object is a miss. Proxy rows share a class with a direct object (a wheelchair user, a disabled person and a street vendor are all 'person'; a maintenance or paratransit vehicle is any truck / car / bus), so their false positives are every frame where the shared class is present for another reason.

## confidence >= 0.25

| object | group | COCO class(es) | match | frames present | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bench | **3** | bench | direct | 76 | 14 | 25 | 62 | 0.36 | 0.18 | 0.24 |
| Bicycle | 8 | bicycle | direct | 45 | 11 | 6 | 34 | 0.65 | 0.24 | 0.35 |
| Bus | 11 | bus | direct | 38 | 30 | 3 | 8 | 0.91 | 0.79 | 0.84 |
| Car | 11 | car | direct | 274 | 228 | 7 | 46 | 0.97 | 0.83 | 0.90 |
| Chair | 14 | chair | direct | 4 | 0 | 3 | 4 | 0.00 | 0.00 | 0.00 |
| Counter | 14 | dining table | proxy | 0 | 0 | 1 | 0 | 0.00 |  |  |
| Dog | 8 | dog | direct | 0 | 0 | 5 | 0 | 0.00 |  |  |
| Fire hydrant | 2 | fire hydrant | direct | 0 | 0 | 0 | 0 |  |  |  |
| Guide Dog | 8 | dog | proxy | 0 | 0 | 5 | 0 | 0.00 |  |  |
| Maintenance Vehicle | **7** | truck, car | proxy | 8 | 4 | 245 | 4 | 0.02 | 0.50 | 0.03 |
| Motorcycle | 11 | motorcycle | direct | 12 | 5 | 3 | 7 | 0.62 | 0.42 | 0.50 |
| Paratransit Vehicle | 10 | bus, truck, car | proxy | 29 | 27 | 236 | 2 | 0.10 | 0.93 | 0.18 |
| Person | 8 | person | direct | 615 | 601 | 6 | 14 | 0.99 | 0.98 | 0.98 |
| Person with a Disability | 8 | person | proxy | 590 | 576 | 31 | 14 | 0.95 | 0.98 | 0.96 |
| Roadside Parking | 10 | car | proxy | 158 | 141 | 94 | 17 | 0.60 | 0.89 | 0.72 |
| Stop Sign | 12 | stop sign | direct | 6 | 0 | 0 | 6 |  | 0.00 |  |
| Street Vendor | 8 | person | proxy | 9 | 9 | 598 | 0 | 0.01 | 1.00 | 0.03 |
| Table | 14 | dining table | direct | 9 | 0 | 1 | 9 | 0.00 | 0.00 | 0.00 |
| Traffic Signals | 12 | traffic light | direct | 69 | 29 | 3 | 40 | 0.91 | 0.42 | 0.57 |
| Wheelchair | 8 | person | proxy | 0 | 0 | 607 | 0 | 0.00 |  |  |

| micro average over | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| direct COCO objects | 918 | 62 | 230 | 0.94 | 0.80 | 0.86 |
| direct + proxy objects | 1675 | 1879 | 267 | 0.47 | 0.86 | 0.61 |
| all annotated objects, direct classes only (paper-style, N = 12) | 918 | 62 | 6292 | 0.94 | 0.13 | 0.22 |
| all annotated objects, direct + proxy classes | 1675 | 1879 | 5535 | 0.47 | 0.23 | 0.31 |

High-priority groups 3, 5, 7: 205 labelled frames contain at least one such object; the detector fires a mapped class for one of them in 18 (0.09 frame recall). Only Bench (direct) and Maintenance Vehicle (proxy truck / car) can score there; the rest have no COCO class.

## confidence >= 0.5

| object | group | COCO class(es) | match | frames present | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bench | **3** | bench | direct | 76 | 9 | 11 | 67 | 0.45 | 0.12 | 0.19 |
| Bicycle | 8 | bicycle | direct | 45 | 4 | 3 | 41 | 0.57 | 0.09 | 0.15 |
| Bus | 11 | bus | direct | 38 | 22 | 0 | 16 | 1.00 | 0.58 | 0.73 |
| Car | 11 | car | direct | 274 | 184 | 1 | 90 | 0.99 | 0.67 | 0.80 |
| Chair | 14 | chair | direct | 4 | 0 | 1 | 4 | 0.00 | 0.00 | 0.00 |
| Counter | 14 | dining table | proxy | 0 | 0 | 0 | 0 |  |  |  |
| Dog | 8 | dog | direct | 0 | 0 | 3 | 0 | 0.00 |  |  |
| Fire hydrant | 2 | fire hydrant | direct | 0 | 0 | 0 | 0 |  |  |  |
| Guide Dog | 8 | dog | proxy | 0 | 0 | 3 | 0 | 0.00 |  |  |
| Maintenance Vehicle | **7** | truck, car | proxy | 8 | 2 | 192 | 6 | 0.01 | 0.25 | 0.02 |
| Motorcycle | 11 | motorcycle | direct | 12 | 4 | 0 | 8 | 1.00 | 0.33 | 0.50 |
| Paratransit Vehicle | 10 | bus, truck, car | proxy | 29 | 20 | 186 | 9 | 0.10 | 0.69 | 0.17 |
| Person | 8 | person | direct | 615 | 580 | 3 | 35 | 0.99 | 0.94 | 0.97 |
| Person with a Disability | 8 | person | proxy | 590 | 555 | 28 | 35 | 0.95 | 0.94 | 0.95 |
| Roadside Parking | 10 | car | proxy | 158 | 119 | 66 | 39 | 0.64 | 0.75 | 0.69 |
| Stop Sign | 12 | stop sign | direct | 6 | 0 | 0 | 6 |  | 0.00 |  |
| Street Vendor | 8 | person | proxy | 9 | 9 | 574 | 0 | 0.01 | 1.00 | 0.03 |
| Table | 14 | dining table | direct | 9 | 0 | 0 | 9 |  | 0.00 |  |
| Traffic Signals | 12 | traffic light | direct | 69 | 12 | 0 | 57 | 1.00 | 0.17 | 0.30 |
| Wheelchair | 8 | person | proxy | 0 | 0 | 583 | 0 | 0.00 |  |  |

| micro average over | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| direct COCO objects | 815 | 22 | 333 | 0.97 | 0.71 | 0.82 |
| direct + proxy objects | 1520 | 1654 | 422 | 0.48 | 0.78 | 0.59 |
| all annotated objects, direct classes only (paper-style, N = 12) | 815 | 22 | 6395 | 0.97 | 0.11 | 0.20 |
| all annotated objects, direct + proxy classes | 1520 | 1654 | 5690 | 0.48 | 0.21 | 0.29 |

High-priority groups 3, 5, 7: 205 labelled frames contain at least one such object; the detector fires a mapped class for one of them in 11 (0.05 frame recall). Only Bench (direct) and Maintenance Vehicle (proxy truck / car) can score there; the rest have no COCO class.

For comparison, Table 5 of the paper reports YOLOv7 on the same keyframes at N = 12 predictable objects, micro precision 0.94, recall 0.16, F1 0.27; the 'direct classes only' row is the matching protocol (all annotated objects, non-predictable ones always missed).

