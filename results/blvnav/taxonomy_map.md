# BLV-Road-Nav-Accessibility: the 90 objects against YOLO11n (COCO-80) and the P-CACS event taxonomy

Source: Islam, Kabir, Pearce, Reza, Billah: A Dataset for Crucial Object Recognition in Blind and Low-Vision Individuals' Navigation, ASSETS 2024 (arXiv 2407.16777). Groups are Table 3 of the paper; groups 3, 5 and 7 are the focus group's most significant objects.
YOLO11n = the detector in `seeforme/perception/detector.py` (COCO-80 classes); events = `LABEL_EVENT` / `EVENT_SPECS` in
`seeforme/campus/world.py`. *direct* = the COCO class is the object; *proxy* = a COCO class fires on the object without naming it.

## Headline

* detectable by YOLO11n at all: **20 / 90** (12 direct, 8 proxy); of these 16 get a non-decor event and 4 are decor in LABEL_EVENT
* high-priority groups 3, 5, 7: **2 / 13** detectable (2 scheduled); 4 reachable only through an open-vocabulary prompt, 0 have an event for a label no detector produces, 6 by analogy, 1 nothing
* reachable through a NAV_VOCAB prompt of the open-vocabulary detector (YOLO-World; 41 prompts): **49 / 90** (6 / 13 high priority), of which 31 have no COCO class at all
* taxonomy-only (event exists for a simulator label no detector produces): 0; analogue only: 12; OCR only: 0; no class and no event: 27
* reachable through the OCR text path (signs): 8

## Per group

| group | parent concept | n | YOLO direct | YOLO proxy | scheduled | detected, decor | open-vocab only | taxonomy only | analogue | OCR only | nothing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Attributes of a sidewalk and driveway | 13 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 10 |
| 2 | Obstructions likely to be detected by a white cane | 10 | 1 | 0 | 1 | 0 | 5 | 0 | 1 | 0 | 3 |
| **3** | Obstructions less likely to be detected by a white cane | 5 | 1 | 0 | 1 | 0 | 4 | 0 | 0 | 0 | 0 |
| 4 | Objects that are too late to be detected by a white cane | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| **5** | Objects that pick you before you pick them | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 6 | Objects that provide navigational guidance | 4 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| **7** | Objects not supposed to be on the sidewalk | 7 | 0 | 1 | 1 | 0 | 0 | 0 | 5 | 0 | 1 |
| 8 | Moving objects sharing the sidewalk | 8 | 3 | 4 | 5 | 2 | 0 | 0 | 0 | 0 | 1 |
| 9 | Intersection | 7 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 |
| 10 | Objects on the road shoulder | 4 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| 11 | Objects on the road | 6 | 3 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | 2 |
| 12 | Traffic signals and street signs | 7 | 2 | 0 | 0 | 2 | 4 | 0 | 0 | 0 | 1 |
| 13 | Objects related to building exits and entrances | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| 14 | Indoor objects | 11 | 2 | 1 | 3 | 0 | 5 | 0 | 2 | 0 | 1 |
| 15 | Objects related to public transit | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| 3+5+7 | high priority | 13 | 1 | 1 | 2 | 0 | 4 | 0 | 6 | 0 | 1 |

Status meanings: *detected_and_scheduled* = YOLO11n outputs a class for it and LABEL_EVENT gives that class a non-decor event; *detected_but_decor* = YOLO11n outputs a class for it, but LABEL_EVENT treats that class as decor (never spoken); *open_vocab* = no COCO class, but a NAV_VOCAB prompt of the open-vocabulary detector (YOLO-World) produces a LABEL_EVENT label for it; *taxonomy_only* = an event exists for a simulator label (wet floor sign, scooter, ...) that no detector produces; *analogue* = no label at all; an event type is assigned here by analogy; *ocr_only* = a text sign: reachable through the OCR path, invisible to the detector; *no_class_no_event* = neither a detector class nor an event type

## All 90 objects

| # | object | group | COCO class(es) | match | open-vocab prompt(s) | event type | via | severity | status | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | "Use the Other Door" Sign | 12 |  | none | sign | landmark | LABEL_EVENT['sign'] (not a COCO class) | 0.4 | open_vocab |  |
| 2 | Accent Paving | 1 |  | none |  |  |  |  | no_class_no_event | surface attribute |
| 3 | Barrier Post | **3** |  | none | bollard, pole | hazard_static | LABEL_EVENT['pole'] (not a COCO class) | 0.7 | open_vocab | open-vocab prompt 'bollard' |
| 4 | Barrier Stump | **3** |  | none | bollard, pole | hazard_static | LABEL_EVENT['pole'] (not a COCO class) | 0.7 | open_vocab | short bollard; open-vocab prompt 'bollard' |
| 5 | Bench | **3** | bench | direct | bench | hazard_static | LABEL_EVENT['bench'] | 0.7 | detected_and_scheduled |  |
| 6 | Bicycle | 8 | bicycle | direct | bicycle | bike | LABEL_EVENT['bicycle'] | 0.9 | detected_and_scheduled |  |
| 7 | Brick Paving | 1 |  | none |  |  |  |  | no_class_no_event |  |
| 8 | Brick Wall | 2 |  | none |  |  |  |  | no_class_no_event |  |
| 9 | Bridge | 9 |  | none |  |  |  |  | no_class_no_event |  |
| 10 | Building | 14 |  | none |  |  |  |  | no_class_no_event |  |
| 11 | Bus | 11 | bus | direct | bus | vehicle | LABEL_EVENT['bus'] | 1.0 | detected_and_scheduled |  |
| 12 | Bus Stop | 15 |  | none | sign | landmark | LABEL_EVENT['sign'] (not a COCO class) | 0.4 | open_vocab | the sign is readable by OCR; a FIND-mode goal |
| 13 | Car | 11 | car | direct | car | vehicle | LABEL_EVENT['car'] | 1.0 | detected_and_scheduled |  |
| 14 | Chair | 14 | chair | direct | chair | hazard_static | LABEL_EVENT['chair'] | 0.7 | detected_and_scheduled |  |
| 15 | Clear Ice | 1 |  | none |  |  |  |  | no_class_no_event | not in Table 3 as published; surface condition |
| 16 | Closed Sidewalk | **3** |  | none | barrier | barrier | LABEL_EVENT['barrier'] (not a COCO class) | 0.75 | open_vocab | barricade plus sign: the text is readable by OCR |
| 17 | Cobblestone Paving | 1 |  | none |  |  |  |  | no_class_no_event |  |
| 18 | Counter | 14 | dining table | proxy | table | hazard_static | LABEL_EVENT['dining table'] | 0.7 | detected_and_scheduled | 'dining table' fires on some counters |
| 19 | Crosswalk | 9 |  | none | crosswalk | landmark | LABEL_EVENT['crossing'] (not a COCO class) | 0.4 | open_vocab |  |
| 20 | Curb | 9 |  | none | curb | curb | LABEL_EVENT['curb'] (not a COCO class) | 0.8 | open_vocab |  |
| 21 | Curved Railing | 6 |  | none | fence, railing | barrier | LABEL_EVENT['fence'] (not a COCO class) | 0.75 | open_vocab | open-vocab prompt 'railing' |
| 22 | Dog | 8 | dog | direct |  | decor | LABEL_EVENT default for 'dog' | 0.0 | detected_but_decor | the COCO label 'dog' is decor; the open-vocab route maps dog -> 'animal' -> person_moving |
| 23 | Doorway | 13 |  | none | door | door | LABEL_EVENT['door'] (not a COCO class) | 0.45 | open_vocab |  |
| 24 | Driveway (flat) | 1 |  | none |  |  |  |  | no_class_no_event | participants: the driveway ahead matters, its slope does not |
| 25 | Elevator | 14 |  | none | elevator | landmark | LABEL_EVENT['elevator'] (not a COCO class) | 0.4 | open_vocab |  |
| 26 | Escalator | 14 |  | none | escalator, stairs | stairs | LABEL_EVENT['stairs'] (not a COCO class) | 1.0 | open_vocab | open-vocab prompt 'escalator' |
| 27 | Fence | 2 |  | none | fence, railing | barrier | LABEL_EVENT['fence'] (not a COCO class) | 0.75 | open_vocab |  |
| 28 | Fire hydrant | 2 | fire hydrant | direct | fire hydrant | hazard_static | LABEL_EVENT['fire hydrant'] | 0.7 | detected_and_scheduled |  |
| 29 | Flush Door | 13 |  | none | door | door | LABEL_EVENT['door'] (not a COCO class) | 0.45 | open_vocab |  |
| 30 | Foldout Sign | **3** |  | none | sign | landmark | LABEL_EVENT['sign'] (not a COCO class) | 0.4 | open_vocab | A-frame sign: the 'sign' prompt reaches it only as a landmark (0.4); the simulator's 'wet floor sign' label (barrier, 0.75) is the better spec |
| 31 | Fountain | 14 |  | none |  | hazard_static | analogue (assigned here) | 0.7 | analogue |  |
| 32 | Gate | 13 |  | none |  | door | analogue (assigned here) | 0.45 | analogue | the 'door' prompt may or may not fire on a gate |
| 33 | Guide Dog | 8 | dog | proxy |  | decor | LABEL_EVENT default for 'dog' | 0.0 | detected_but_decor | detected as a dog (decor on the COCO route) |
| 34 | Gutter | 2 |  | none |  |  |  |  | no_class_no_event |  |
| 35 | Hose | **7** |  | none |  | unknown_obstacle | analogue (assigned here) | 0.6 | analogue | an open-vocabulary 'obstruction' label would reach unknown_obstacle; no prompt today |
| 36 | Intersection | 9 |  | none |  |  |  |  | no_class_no_event |  |
| 37 | Lamp Post | 2 |  | none | bollard, pole | hazard_static | LABEL_EVENT['pole'] (not a COCO class) | 0.7 | open_vocab |  |
| 38 | Mailbox | 2 |  | none |  | hazard_static | analogue (assigned here) | 0.7 | analogue | sidewalk obstruction like a pole; no prompt |
| 39 | Maintenance Vehicle | **7** | truck, car | proxy | car, truck | vehicle | LABEL_EVENT['truck'] | 1.0 | detected_and_scheduled | detected as a truck or car |
| 40 | Motorcycle | 11 | motorcycle | direct | motorcycle | vehicle | LABEL_EVENT['motorcycle'] | 1.0 | detected_and_scheduled |  |
| 41 | Moving Walk | 14 |  | none |  | stairs | analogue (assigned here) | 1.0 | analogue | moving surface; closest spec |
| 42 | Overhanging Tree Branches | **5** |  | none |  | hazard_static | analogue (assigned here) | 0.7 | analogue | head-height obstruction; 'tree trunk' does not fire on branches; no event class of its own |
| 43 | Parallel Parking Spot | 10 |  | none |  |  |  |  | no_class_no_event |  |
| 44 | Paratransit Vehicle | 10 | bus, truck, car | proxy | bus, car, truck | vehicle | LABEL_EVENT['bus'] | 1.0 | detected_and_scheduled | detected as a bus, truck or car |
| 45 | Pedestrian Crossing | 9 |  | none | crosswalk | landmark | LABEL_EVENT['crossing'] (not a COCO class) | 0.4 | open_vocab | open-vocab prompt 'crosswalk'; the signal is 'Traffic Signals' |
| 46 | Person | 8 | person | direct | person | person_static | LABEL_EVENT['person'] | 0.5 | detected_and_scheduled | person_moving once the track moves |
| 47 | Person with a Disability | 8 | person | proxy | person | person_static | LABEL_EVENT['person'] | 0.5 | detected_and_scheduled |  |
| 48 | Pillar | 14 |  | none | bollard, pole | hazard_static | LABEL_EVENT['pole'] (not a COCO class) | 0.7 | open_vocab |  |
| 49 | Pole | 2 |  | none | bollard, pole | hazard_static | LABEL_EVENT['pole'] (not a COCO class) | 0.7 | open_vocab |  |
| 50 | Puddle | 1 |  | none | pothole, puddle | hazard_static | LABEL_EVENT['ground hazard'] (not a COCO class) | 0.7 | open_vocab | open-vocab prompt 'puddle' |
| 51 | Push Button | 12 |  | none |  |  |  |  | no_class_no_event | pedestrian signal button; a FIND-mode goal rather than a hazard |
| 52 | Railing | 6 |  | none | fence, railing | barrier | LABEL_EVENT['fence'] (not a COCO class) | 0.75 | open_vocab | open-vocab prompt 'railing' |
| 53 | Raised Entryway | 1 |  | none |  | curb | analogue (assigned here) | 0.8 | analogue | a step up at an entrance: closest spec is curb |
| 54 | Retaining Wall | 6 |  | none |  |  |  |  | no_class_no_event | guidance surface |
| 55 | Road | 11 |  | none |  |  |  |  | no_class_no_event |  |
| 56 | Road Divider | 11 |  | none | barrier | barrier | LABEL_EVENT['barrier'] (not a COCO class) | 0.75 | open_vocab | median / jersey barrier |
| 57 | Road Shoulder | 10 |  | none |  |  |  |  | no_class_no_event |  |
| 58 | Roadside Parking | 10 | car | proxy | car | vehicle | LABEL_EVENT['car'] | 1.0 | detected_and_scheduled | parked cars are detected as cars; the parking zone is not |
| 59 | Sidewalk | 1 |  | none |  |  |  |  | no_class_no_event | walkable surface, not an event |
| 60 | Sidewalk Pits | 1 |  | none | pothole, puddle | hazard_static | LABEL_EVENT['ground hazard'] (not a COCO class) | 0.7 | open_vocab | open-vocab prompt 'pothole' |
| 61 | Sign | 12 |  | none | sign | landmark | LABEL_EVENT['sign'] (not a COCO class) | 0.4 | open_vocab |  |
| 62 | Sign Post | 12 |  | none | bollard, pole | hazard_static | LABEL_EVENT['pole'] (not a COCO class) | 0.7 | open_vocab | the post is a pole-type obstruction; the face is read by OCR |
| 63 | Sloped Driveway | 1 |  | none |  |  |  |  | no_class_no_event |  |
| 64 | Sloped Curb | 9 |  | none | curb | curb | LABEL_EVENT['curb'] (not a COCO class) | 0.8 | open_vocab | curb cut |
| 65 | Snow | **7** |  | none |  |  |  |  | no_class_no_event | surface condition |
| 66 | Stairs | 14 |  | none | escalator, stairs | stairs | LABEL_EVENT['stairs'] (not a COCO class) | 1.0 | open_vocab | severity 1.0 in EVENT_SPECS, no COCO class |
| 67 | Stop Sign | 12 | stop sign | direct | stop sign | sign_nav | OCR text path | 0.45 | detected_but_decor | the YOLO label maps to decor; the text STOP goes through the OCR path |
| 68 | Street Vendor | 8 | person | proxy | person | person_static | LABEL_EVENT['person'] | 0.5 | detected_and_scheduled | the vendor is a person; the stall has no class |
| 69 | Table | 14 | dining table | direct | table | hazard_static | LABEL_EVENT['dining table'] | 0.7 | detected_and_scheduled |  |
| 70 | Tactile Paving | 1 |  | none |  |  |  |  | no_class_no_event | guidance surface |
| 71 | Toilet Sign | 12 |  | none | sign | landmark | LABEL_EVENT['sign'] (not a COCO class) | 0.4 | open_vocab | pictogram plus text; a FIND-mode goal |
| 72 | Traffic Signals | 12 | traffic light | direct | traffic light | decor | LABEL_EVENT default for 'traffic light' | 0.0 | detected_but_decor | COCO class with no LABEL_EVENT entry: decor; SeeForMe has no crossing-signal event |
| 73 | Train Platform | 4 |  | none |  | curb | analogue (assigned here) | 0.8 | analogue | platform edge: closest spec is curb, which understates the drop |
| 74 | Train Tracks | 4 |  | none |  |  |  |  | no_class_no_event | 'train' fires on a train, never on the track |
| 75 | Trash Bins | 2 |  | none | trash can | hazard_static | LABEL_EVENT['trash can'] (not a COCO class) | 0.7 | open_vocab | COCO-80 has no trash can |
| 76 | Trash on Roads | **7** |  | none |  | unknown_obstacle | analogue (assigned here) | 0.6 | analogue | UNKNOWN_OBSTACLE_LABELS has 'debris'; no prompt today |
| 77 | Tree | 2 |  | none | tree trunk | hazard_static | LABEL_EVENT['tree'] (not a COCO class) | 0.7 | open_vocab | open-vocab prompt 'tree trunk' |
| 78 | Turnstile | 15 |  | none | kiosk, turnstile | hazard_static | LABEL_EVENT['machine'] (not a COCO class) | 0.7 | open_vocab | open-vocab prompt 'turnstile' |
| 79 | Uncontrolled Crossing | 9 |  | none |  |  |  |  | no_class_no_event |  |
| 80 | Uneven Stairs | 14 |  | none | escalator, stairs | stairs | LABEL_EVENT['stairs'] (not a COCO class) | 1.0 | open_vocab |  |
| 81 | Unpaved Road | 11 |  | none |  |  |  |  | no_class_no_event |  |
| 82 | Unpaved Sidewalk | 1 |  | none |  |  |  |  | no_class_no_event |  |
| 83 | Vegetation | 2 |  | none |  |  |  |  | no_class_no_event | 'potted plant' does not fire on street vegetation |
| 84 | Wall | 6 |  | none |  |  |  |  | no_class_no_event |  |
| 85 | Water Leakage | **7** |  | none |  | unknown_obstacle | analogue (assigned here) | 0.6 | analogue | UNKNOWN_OBSTACLE_LABELS has 'spill'; no prompt today |
| 86 | Water Pipes | **7** |  | none |  | unknown_obstacle | analogue (assigned here) | 0.6 | analogue | no prompt today |
| 87 | Wet Surface | 1 |  | none |  |  |  |  | no_class_no_event | surface condition; only the warning sign ('wet floor sign' -> barrier) is in LABEL_EVENT |
| 88 | Wheelchair | 8 | person | proxy | person | person_static | LABEL_EVENT['person'] | 0.5 | detected_and_scheduled | the user is detected as a person; the wheelchair has no class |
| 89 | White Cane | 8 |  | none |  |  |  |  | no_class_no_event | identifies a blind pedestrian; not an obstacle |
| 90 | Yard Waste | **7** |  | none |  | unknown_obstacle | analogue (assigned here) | 0.6 | analogue | UNKNOWN_OBSTACLE_LABELS has 'debris'; no prompt today |
