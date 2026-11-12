# Cane on EgoBlind development videos

Generated 2026-10-09 19:00 by `python scripts/eval_live_cane.py` (15.7 min). Splits: finetune, select (development only; holdout and test were not loaded).

Processed 84 of 138 development videos with a local file (69.9 min of video, up to 3 s after each video's last question): all 34 of the 34 videos with an obstacle-type event first, then videos without one in a seeded random order until 84 videos (--max-videos, a fixed set) or the 40-minute budget ran out (54 videos not processed). Depth: Depth Anything V2 metric small at input 308 (as `scripts/run_live.py`), 73 indoor / 11 outdoor: the cached label-free CLIP choice for 16 videos, the indoor model (the live demo's default) for the 68 whose cache predates that record (their choice may have come from the labels, so it is not reused). 1 letterboxed video(s) (a portrait phone picture inside a landscape file) were cropped to the picture before depth, as a camera would deliver it: the depth model invents surfaces in black bars. The cane sees 2 depth maps per second and decides at each; the activity detector sees 4 frames per second. Selective attention uses the cached yolo11n detections (84 videos have them) as tracks.

## Reading

- Cane cues per minute: **6.09** if the user were always walking (upper bound), **5.57** with the activity detector (quiet while still unless something approaches), 6.82 walking without selective attention. Stops (haptic) per minute: walking 1.2, activity 1.09.
- Per video (walking stream): median 6.4 cues/min (quartiles 2.1-11.1); 19% of the videos get none, 31% more than 10 per minute. The busy ones are hand-held walks through shops, stations and crowds with counters, companions and escalators within a metre or two (several of the busiest looked at frame by frame); EgoBlind has no labels to tell real obstacles from noise there.
- What it said (walking stream): Obstacle ahead 244, Low obstacle ahead 88, Stop 84, Something at head height ahead 10.
- Obstacles the detector cannot name (29 events): lift 1.53 (1.11-2.0), the cane speaks in their windows more often than at random times (the interval excludes 1).
- Obstacles the detector covers (9 events): lift 0.89 (0.23-1.44), the cane speaks in their windows about as often as at random times (the interval includes 1): no evidence either way. A tracked chair or table is left to the scheduler (selective attention), so the cane is not expected to speak more here than elsewhere.
- Other should-speak events (control) (29 events): lift 0.94 (0.54-1.44), the cane speaks in their windows about as often as at random times (the interval includes 1): no evidence either way.
- Latency: cane update (analysis + persistence) 3.57 ms mean, 5.13 ms p95 per depth map; the depth model itself 46.3 ms mean, 57.4 ms p95.

## Obstacle questions (walking stream, the upper bound)

An event counts as a hit when the cane said something (a cue or a stop) inside its useful window [t_q - 6 s, t_q + 0.5 s]; **chance** is the same for windows of the same length at random times of the same video (not overlapping the event's), **lift** = hit rate / chance (bootstrap 95% interval over events). **Sees**: a confirmed obstacle within 2 m ahead at any decision in the window, whether or not the identity memory let it speak.

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 18 | 0.621 (0.44-0.773) | 0.406 | 1.53 (1.11-2.0) | 0.724 | 0.637 | 1.14 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 3 | 0.333 (0.121-0.646) | 0.373 | 0.89 (0.23-1.44) | 0.667 | 0.724 | 0.92 |
| other should-speak events (people, vehicles, unnamed) | 29 | 11 | 0.379 (0.227-0.56) | 0.403 | 0.94 (0.54-1.44) | 0.483 | 0.526 | 0.92 |

Same with the activity detector (what the live demo would say):

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO | 29 | 18 | 0.621 (0.44-0.773) | 0.395 | 1.57 (1.12-2.12) | 0.724 | 0.671 | 1.08 |
| obstacle, COCO-covered | 9 | 1 | 0.111 (0.02-0.435) | 0.316 | 0.35 (0.0-1.25) | 0.667 | 0.702 | 0.95 |
| other should-speak events | 29 | 10 | 0.345 (0.199-0.527) | 0.338 | 1.02 (0.56-1.61) | 0.483 | 0.55 | 0.88 |

## What these labels are

EgoBlind events are questions blind users asked while filming, weakly labelled from the question and the answers (`seeforme/egoblind/data.py`). An obstacle-type event names a door, stairs, a pole, a fence ... somewhere in the question or answer, which is often a destination ("how can I find the entrance?") or something to the side, not an obstacle in the walking corridor; its window is set by when the question was asked, not by when the object was close. The samples are small. Read the hit rates as weak evidence and the lift against random windows as the useful part; neither is a recall. Cues per minute count every cane cue as a potential false alarm, which overstates them (some are real walls and doors).

## Obstacle events

| video | t_q | names | question | hit | sees |
|---|---|---|---|---|---|
| 0 | 47.0 | door | Which direction will I need to move to exit the vehicle? | no | yes |
| 89 | 83.0 | door | Is the path clear to enter the store? | yes | yes |
| 124 | 15.0 | door, horse | In which direction should I guide the horse to leave the arena? | yes | yes |
| 127 | 4.0 | fence, horse | Which direction is the horse heading? | no | yes |
| 183 | 2.0 | door, sign | Where's the nearest stairway? | yes | yes |
| 347 | 11.0 | wall | Are there any obstacles on the road? | yes | yes |
| 347 | 33.3 | fence | How can I negotiate this ramp safely? | yes | yes |
| 347 | 37.3 | fence | How can I get to the exit from here? | yes | yes |
| 362 | 52.3 | fence, stairs, person | How to use this escalator safely? | yes | yes |
| 434 | 19.0 | door, person | How do I navigate this cafe? | yes | yes |
| 438 | 15.0 | door | Are there steps or a ramp at the entrance to this place? | yes | yes |
| 469 | 76.6 | door | Where is the cashier? | no | yes |
| 665 | 41.2 | door, machine | Is there an obstacle ahead? | yes | yes |
| 668 | 39.0 | tree, car | Is there an obstacle ahead? | yes | yes |
| 668 | 41.0 | tree | Where have I been? | yes | yes |
| 677 | 34.5 | stairs | Where to go next? | no | no |
| 708 | 34.6 | car | Is there an obstacle ahead? | no | no |
| 710 | 1.6 | pole | Is there an obstacle ahead? | yes | yes |
| 781 | 35.0 | door | How can I reach the exit from this table? | yes | yes |
| 805 | 3.0 | fence | Can I walk on the road ahead smoothly? | no | no |
| 807 | 35.0 | fence | Which direction should I go to reach the end of the fence? | yes | yes |
| 817 | 23.0 | fence, train | Is it safe to proceed? | yes | yes |
| 870 | 0.0 | stairs | Which direction does the moving walkway go? | no | no |
| 872 | 4.0 | door, person | How can I find the entrance? | no | no |
| 878 | 86.8 | door | Am I almost at the hospital? | yes | yes |
| 884 | 5.9 | pole, person | What is ahead? | no | no |
| 884 | 152.1 | stairs | Where is the subway station? | no | no |
| 910 | 14.8 | door | Are there any obstacles ahead? | yes | yes |
| 918 | 92.7 | door | Has the place for making keys arrived? | no | no |
| 184 | 5.0 | dining table | How do I navigate to the table? | no | yes |
| 405 | 27.0 | dining table, chair | What obstacles should I be aware of when moving in the area? | yes | yes |
| 406 | 153.0 | dining table | What obstructions on the counter should I be aware of? | no | no |
| 407 | 69.0 | dining table | Are there any obstacles around the counter? | yes | yes |
| 439 | 4.0 | dining table | Is there a counter nearby where I can get help? | no | yes |
| 440 | 13.0 | shelf, dining table, chair | Are there any obstacles or objects on the floor that could cause me to | no | yes |
| 472 | 50.5 | person, dining table | Can you tell me where to get to the nearest assistance point? | no | no |
| 910 | 118.6 | tree, potted plant | Where is this? | no | no |
| 919 | 92.8 | dining table | Where is the checkout page? | yes | yes |

Cane counters (walking stream, summed): said 342, stop 84, path 50, head 10, held 1066, explained 243, still 0, close_up 896, noted 0.
