# TIMELI (Time-Aware Assistive Navigation, ICRA 2026): what is public, what is on disk, and how it maps to SeeForMe

Masaki Kuribayashi (Waseda), Zhongkai Shangguan and Eshed Ohn-Bar (Boston University). arXiv 2609.05596 (4 Sep 2026),
project page https://timeli-icra.github.io/. Checked 2026-10-08. The question TIMELI asks is SeeForMe's question: can an
agent learn not only what to say to a blind pedestrian but when, and when to stay silent.

## What exists and where

There is no GitHub repository and no Hugging Face release; the project page links two public Google Drive folders (no
login, no form). `dataset.croissant.json` declares the data **CC BY 4.0**; the code archives carry no licence file.

| Drive folder | file | size | content | local |
|---|---|---|---|---|
| Data `1unHMorAjR4jWkzqaDxxCXe2okDfET-Np` | `train_labels.json` | 360 MB | 56,281 CARLA frames, 269 session videos, Town05 + Town10HD | `data/timeli/train_labels.json` |
| | `test_labels.json` | 70 MB | 10,820 CARLA frames, 20 session videos, Town02 / 03 / 04 | `data/timeli/test_labels.json` |
| | `real_test.json` | 4.9 MB | 1,059 YouTube frames, 11 clips from 5 US walking videos, human-edited instructions | `data/timeli/real_test.json` |
| | `dataset.croissant.json` | 4 KB | Croissant metadata (schema, licence) | `data/timeli/dataset.croissant.json` |
| | `images.tar.gz` | 32 GB | one JPEG per CARLA frame | not downloaded |
| | `real_test.tar.gz` | 38 GB | real-test frames and clips | not downloaded |
| | `videos.tar.gz` | 537 GB | 16-frame clips (one per CARLA record) | not downloaded |
| Code `1VlNo86ne1Ilah83vQ0YQGIrj4M49qbEE` | `Dataset_Generation.tar.gz` | 17 KB | `make_dataset.py` (the ground-truth rule), `merge_and_split.py`, prompt templates | `data/timeli/code/extracted/Dataset_Generation` |
| | `Closed_Loop_Eval.tar.gz` | 271 KB | `CarlaWalker`: CARLA 0.9.15 pedestrian simulator, rule-based instruction generator, closed-loop scorer, 10 route files | `data/timeli/code/extracted/CarlaWalker` |
| | `TIMELI_Train_Eval.tar.gz` | 14 MB | model fine-tuning / open-loop evaluation code (LLaVA, VILA, TimeChat, TinyLLaVA) | not downloaded |

The paper says 67K navigation videos and a 2,000-video TIMELI-Real set; the released label files hold 67,101 CARLA
frames and 1,059 real frames. The page says the original YouTube links "will be listed in our GitHub repository", which
does not exist yet. To get the rest: `gdown --folder <url>` works without an account (the venv has gdown 6.4.1);
`videos.tar.gz` and `images.tar.gz` exceed this machine's disk budget, so a perception replay needs an external drive.

Scripts: `python scripts/timeli_inspect.py` writes `results/timeli/inspect.{json,md}` (fields, counts, silence
statistics); `python scripts/timeli_to_events.py` writes `data/timeli/events.jsonl` and
`results/timeli/events_summary.{json,md}`.

## Format: how instructions, times and silence are represented

One record per frame at **1 FPS**; `id` is the frame index and therefore the time in seconds since the start of the
session video `video_id`; frames are contiguous (ids 16..N in CARLA, 15..N in the real set; the first frames only feed the
history). Each record carries the full model prompt (`conversations[0]`, `QA[0].q`: task specification, goal
coordinate, high-level plan `action`, the 15 previous seconds' instructions) and the answer `QA[0].a`: a JSON string
`{"reason": ..., "instruction": ...}` in CARLA, plain text in the real set. **Silence is the instruction `"None"`** with
the reason `remain_silent` or `remain_silent_in_junction`. Other fields: `history` (15 strings, oldest first, index 14 =
1 s ago; verified equal to the previous frame's instruction in 100% of frames), `goal_coord` (next waypoint relative to
the user, metres), `is_junction`, `action` (one of walk straight, turn slightly left / right, turn left / right, turn
around, stop), `conveyed_obstacles` (`{class: [front|left|right]}` announced in this instruction), `description_cluster`
(semantic-map objects within 3 m: POLE, BUILDING, PEDESTRIAN, OBJECT, FENCE, VEGETATION, WALL, VEHICLE, TRAFFIC_SIGN,
GUARD_RAIL with a side), `object_detection` (a COCO detector's boxes on the frame; the label "banana" is 11% of them, a
CARLA artefact), `meta` (weather, town, traffic density), `QA[0].a_original` (the raw template, e.g. "go forward 1.5
meters"). The real set has `spoken_reason`, `prediction` (a model output, "None" in 76%), `seen`, no obstacle fields.

The ground-truth rule (`Dataset_Generation/make_dataset.py`, `determine_to_talk`), in priority order: first frame ->
`constant_instruction`; an instruction in the previous frame -> `remain_silent` (never two in a row); junction entered /
left -> `enter_junction` / `exit_junction` ("You are entering the junction"); a not-yet-conveyed obstacle class in front
-> `obstacle_in_front`; 6 to 9 (random) silent frames outside a junction -> `constant_instruction` ("Keep walking
straight", or `stop` at the goal); the plan changing from "walk straight" to a turn, or a left / right ambiguity ->
`direction_change`; otherwise `remain_silent`, `remain_silent_in_junction` inside a junction. "wall" is never conveyed,
"pedestrian" may be repeated after 3 frames, other classes once per history window. Turn phrasing comes from the yaw
change over one second (`CarlaWalker/src/instruction_generation.py`): under 15 degrees straight, 15-45 "slightly",
45-105 a clock-face direction ("turn to your two"), 105-145 a 120-degree turn, over 145 "turn around".

Counts (`results/timeli/inspect.md`):

| split | frames | videos | spoken share | instructions / min | words / instruction | silence runs (mean, longest) | junction share |
|---|---|---|---|---|---|---|---|
| train | 56,281 | 269 | 0.207 | 12.4 | 11.6 +/- 4.7 | 11,776 (3.8 s, 24 s) | 0.233 |
| test | 10,820 | 20 | 0.220 | 13.2 | 11.2 +/- 4.2 | 2,397 (3.5 s, 28 s) | 0.128 |
| real | 1,059 | 11 | 0.291 | 17.5 | 9.5 +/- 4.3 | 264 (2.8 s, 8 s) | 0.0 |

Reasons, train / test / real: remain_silent 37,536 / 7,699 / 784; remain_silent_in_junction 7,114 / 736 / 25;
direction_change 3,814 / 811 / 0; obstacle_in_front 3,588 / 898 / 153; constant_instruction 2,446 / 419 / 84;
enter_junction 890 / 129 / 6; exit_junction 884 / 128 / 6; stop 9 / 0 / 0. In the real set the reason and the text
disagree in 17% of frames (122 instructions labelled remain_silent, 45 silences labelled obstacle_in_front); the
converter lets the text win.

## Metrics, as defined in the paper and the released scorer, next to SeeForMe's

| TIMELI metric | definition | SeeForMe counterpart |
|---|---|---|
| **Timing F1 / AUC** | per second, did the model speak when the ground truth spoke, with a one-second tolerance (binary issuance) | event recall inside `[earliest_useful, latest_safe]` and the off-window false-alarm count; the converter sets the window to `t +/- 1 s`, so recall / precision over `events.jsonl` reproduce Timing F1 exactly. SeeForMe's `regret` (severity-weighted lateness) has no TIMELI equivalent because TIMELI has no "ideal" inside the window |
| **BLEU-4, ROUGE-L** (only when timing is within tolerance) | text overlap with the ground-truth instruction | `ideal_cue` carries the instruction text; SeeForMe scores slot coverage (`minimum_information`), not n-grams |
| **Conciseness** | overlapping non-stopword tokens / the prediction's non-stopword count | the slot-based cue length in `compose()`; a shorter cue with the same slots scores the same in SeeForMe |
| **Instruction Reason, Silence in Junction, Object Precision, Instruction Direction** (radar plot on the project page) | not defined in the paper text; by name: accuracy of the predicted reason label, share of junction frames kept silent, precision of the objects named, correctness of the left / right / straight direction | reason = `detail.timeli_reason` (SeeForMe has `event_type` instead); silence in junction = quiet-window violations restricted to `remain_silent_in_junction` windows (`detail.junction_frames`); object precision = label match between a cue and the event's `label`; direction = the `direction` slot against `direction_at_ideal_deg` (coarse, 0 / +/-45) |
| **Closed loop: SR, RC, NS, collisions / min, instructions / min, NQS** | `NQS = (SR . RC . NS . S_C . S_I)^(1/5)`, `S_C = exp(-C/4)`, `S_I = exp(-(I - 20)^2 / 200)`: 20 instructions per minute is the *optimum*; `NS = RC . 0.5^(collisions per metre) . (0.8 if mean path error >= 1.5 m)` in `score_calculation.py` | CampusBench has no closed loop; its channel cost is `airtime` / `cues per min` with a budget, and the best policies speak 1-5 times per minute. TIMELI's optimum (one instruction every 3 s, 13 words each) is the chatty baseline SeeForMe penalises; EgoBlind users asked 4 questions per video |
| **Sim-to-real** (Table III) | the same open-loop metrics on the 1,059 real frames | the same split exists here (`split = real`), with human-edited text and `label_source = timeli_human` |

Numbers to compare against: fine-tuned LLaVA-v1.6++ reaches Timing F1 0.79 / AUC 0.88 in CARLA and 0.67 / 0.77 on the
real set; off-the-shelf GPT-4o 0.45; the closed-loop best is NQS 0.82 with 50% success, 0.60 collisions / min and 21
instructions / min.

## The conversion (`data/timeli/events.jsonl`, 24,937 events, 33 MB)

| split | videos | seconds | speak events | per min | quiet windows (>= 2 s) | quiet seconds | coverable by COCO + taxonomy |
|---|---|---|---|---|---|---|---|
| train | 269 | 56,281 | 11,631 | 12.4 | 8,651 | 41,525 | 3,121 |
| test | 20 | 10,820 | 2,385 | 13.2 | 1,757 | 7,795 | 765 |
| real | 11 | 1,059 | 308 | 17.5 | 205 | 692 | 154 |

Mapping, per TIMELI reason: `obstacle_in_front` -> the obstacle's SeeForMe event type (pedestrian -> `person_moving`
0.5, pole -> `hazard_static` 0.7, vehicle -> `vehicle` 1.0, object -> `unknown_obstacle` 0.6; real-set objects parsed
from the text with `seeforme.egoblind.data.match_concepts`); `direction_change` -> `goal` 0.7; `enter_junction` ->
`landmark` 0.8 (urgency high); `exit_junction` -> `landmark` 0.5; `constant_instruction` -> `goal` 0.3 with
`detail.reassurance = True`; `stop` -> `goal` 1.0. Windows `[t - 1, t, t + 1]`, `ideal_cue` = the instruction,
`minimum_information` = the slots the text carries (object, direction, action, distance). Quiet windows: maximal runs
of silent frames of at least 2 s (`[t_first - 0.5, t_last + 0.5]`); the mandatory 1 s gap after every instruction is
dropped (`--min-quiet-s 1` keeps it). Event types over all splits: goal 7,615, person_moving 2,595, landmark 2,040,
hazard_static 1,302, unknown_obstacle 629, vehicle 134, barrier 6, bike 3.

What cannot be mapped: object identity and tracks (class and a coarse side only, so `obj_id` is a running index and
`direction_at_ideal_deg` is 0 / +/-45); distance and time to contact (obstacles are "within cane range", 1.5 m, by
construction; `distance_at_ideal_m` is None); an earliest-useful time (TIMELI has one timestamp with a +/-1 s tolerance,
not an interval of usefulness, so lateness cannot be scored); `sudden` events and interruption; user-confirmed clear
paths (a TIMELI quiet window means the rule fired no trigger, `detail.quiet_semantics = "no_trigger"`, so a cue inside
it is redundant rather than wrong); periodic reassurance (`constant_instruction`, 27% of the instructions) which SeeForMe
never emits; and any perception replay, because the frames are not on disk.

## What TIMELI gives SeeForMe

1. **A second labelled timing benchmark**, synthetic but dense: 14,000 speak events and 10,000 quiet windows against
   CampusBench's simulated ground truth and EgoBlind's 800 weak events. Its labels are rule-generated, so a scheduler
   that scores well on them has learned the rule, not the user; the 1,059 human-edited real frames are the honest part.
2. **The reason label** is the piece SeeForMe lacks: a categorical "why now" that is supervised directly. The paper's main
   finding, that predicting the reason before the instruction is what makes fine-tuned MLLMs time their output, is the
   same structure as P-CACS's per-event urgency terms; `detail.timeli_reason` lets the GRU scheduler be trained with
   an auxiliary reason head on these events.
3. **Silence as a first-class output** ("None" with a reason), including `remain_silent_in_junction` because the user
   is listening to traffic: this is SeeForMe's contention kind in CampusBench and its quiet windows in EgoBlind.
4. **A different operating point**: TIMELI rewards 20 instructions per minute, SeeForMe's tuned policies speak 1-5 times
   per minute. Running SeeForMe's schedulers on `events.jsonl` will show whether its budget is too tight for a
   turn-by-turn task, and running TIMELI-trained models on CampusBench whether a 13-word instruction every 3 s is
   tolerable when hazards carry a reaction-time deadline.
