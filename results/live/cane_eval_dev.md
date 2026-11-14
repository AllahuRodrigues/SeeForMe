# Cane on EgoBlind development videos

Generated 2026-11-12 11:41 by `python scripts/eval_live_cane.py` (0.3 min; the depth analysis it replays took 102 min). Splits: finetune, select (development only; holdout and test were not loaded).

Processed 84 of 138 development videos with a local file (69.9 min of video, up to 3 s after each video's last question): all 34 of the 34 videos with an obstacle-type event first, then videos without one in a seeded random order until 84 videos (--max-videos, a fixed set) or the 120-minute budget ran out (54 videos not processed). Depth: Depth Anything V2 metric small at input 308 (as `scripts/run_live.py`), 73 indoor / 11 outdoor: the cached label-free CLIP choice for 16 videos, the indoor model (the live demo's default) for the 68 whose cache predates that record (their choice may have come from the labels, so it is not reused). 1 letterboxed video(s) (a portrait phone picture inside a landscape file) were cropped to the picture before depth, as a camera would deliver it: the depth model invents surfaces in black bars. The cane sees 2 depth maps per second and decides at each; the activity detector sees 4 frames per second. Selective attention uses the cached yolo11n detections (84 videos have them) as tracks. Each depth map is analysed once and every travel aid runs on that analysis, so the aids differ only in what they say.

## Travel aid

What the cane layer says depends on what the user travels with (`--aid`, default cane; the web app's "I travel with" setting; the server's `aid` field). It follows the BLV-Road-Nav focus group (Islam et al., arXiv 2407.16777): stay quiet about what the user's cane can already touch, warn about "objects that pick you before you pick them" (head height) and about what a cane finds too late. The scheduler's cues about detected objects (people, vehicles, chairs, a FIND goal) are not affected and are not counted here.

- **none**: every cue, as before the aid existed: obstacles of any level within 2 m, the stop for any level.
- **cane** (default): no low obstacle within the cane's reach (1.2 m), beyond it only in a steady approach; body level from 1.2 to 2.5 m, said again only once 40% nearer; head level within 2.5 m always; the stop only for body or head level in a steady approach; path choice ("... and on your left") only for a body obstacle.
- **guide-dog**: head level within 2.5 m and the stop (body or head level, steady approach) only: the dog leads around ground and body obstacles.

| aid | cues/min walking | cues/min activity | stops/min walking | stops/min activity | lift, not-COCO obstacles | lift, COCO obstacles | lift, other events (control) | what it said (walking) |
|---|---|---|---|---|---|---|---|---|
| none | 6.09 | 5.57 | 1.2 | 1.09 | 1.56 (1.14-2.04) | 0.84 (0.22-1.32) | 0.89 (0.5-1.33) | Obstacle ahead 244, Low obstacle ahead 88, Stop 84, Something at head height ahead 10 |
| cane (default) | 3.18 | 2.85 | 0.41 | 0.39 | 1.53 (0.93-2.38) | 0.8 (0.0-1.92) | 0.92 (0.47-1.5) | Obstacle ahead 153, Stop 29, Low obstacle ahead 29, Something at head height ahead 11 |
| guide-dog | 0.57 | 0.52 | 0.41 | 0.39 | no hits (0/29) | no hits (0/9) | 0.79 (0.0-3.75) | Stop 29, Something at head height ahead 11 |

Before the aid existed (the report of 2026-10-09 19:00, the same 84 videos): 6.09 cues/min walking, 5.57 with the activity detector, 1.2 stops/min, not-COCO lift 1.53 (1.11-2.0); messages Obstacle ahead 244, Low obstacle ahead 88, Stop 84, Something at head height ahead 10. The row "none" is that behaviour re-run (its chance windows are drawn per stream now, so its lift can differ a little).

**Reading.** With a cane (the default) the layer says 3.18 cues/min instead of 6.09 (48% fewer) on the same videos; the lift for obstacles the detector cannot name falls from 1.56 (1.14-2.04) to 1.53 (0.93-2.38): still above 1, but the interval now includes 1, so with a cane there is no longer evidence that it speaks near those questions more than elsewhere. The proxy labels are weak and the sample small (see below), so this is a drop in evidence, not proof of harm.

Sensitivity (aid cane, walking stream; each row changes the speaking thresholds named, on the same analysis):

| thresholds | cues/min | stops/min | not-COCO hits | lift, not-COCO obstacles |
|---|---|---|---|---|
| as shipped (reach 1.2 m, detour 2.5 m, again 40% nearer) | 3.18 | 0.41 | 14/29 | 1.53 (0.93-2.38) |
| reach_m=1.0 | 3.66 | 0.41 | 16/29 | 1.6 (1.04-2.37) |
| detour_m=2.0 | 2.47 | 0.41 | 9/29 | 1.23 (0.56-2.24) |
| detour_m=3.0 | 3.66 | 0.41 | 15/29 | 1.47 (0.93-2.22) |
| aid_closer_frac=0.15 | 3.39 | 0.41 | 14/29 | 1.53 (0.93-2.38) |
| steady_resid_m=0.3 | 3.48 | 0.63 | 14/29 | 1.51 (0.92-2.36) |
| reach_m=1.0, detour_m=3.0 | 4.16 | 0.41 | 17/29 | 1.54 (1.03-2.22) |

The shipped thresholds come from the focus group's account of what a cane covers, not from this table; it shows how much the counts move with them on these videos (development splits only).

## Reading (default aid: cane)

- Cane cues per minute: **3.18** if the user were always walking (upper bound), **2.85** with the activity detector (quiet while still unless something approaches), 3.39 walking without selective attention. Stops (haptic) per minute: walking 0.41, activity 0.39.
- Per video (walking stream): median 2.7 cues/min (quartiles 0.0-5.5); 30% of the videos get none, 6% more than 10 per minute (without an aid: median 6.4, 31% above 10). The busy ones are hand-held walks through shops, stations and crowds with counters, companions and escalators within a metre or two; EgoBlind has no labels to tell real obstacles from noise there.
- What it said (walking stream): Obstacle ahead 153, Stop 29, Low obstacle ahead 29, Something at head height ahead 11.
- Obstacles the detector cannot name (29 events): lift 1.53 (0.93-2.38), the cane speaks in their windows about as often as at random times (the interval includes 1): no evidence either way.
- Obstacles the detector covers (9 events): lift 0.8 (0.0-1.92), the cane speaks in their windows about as often as at random times (the interval includes 1): no evidence either way. A tracked chair or table is left to the scheduler (selective attention), so the cane is not expected to speak more here than elsewhere.
- Other should-speak events (control) (29 events): lift 0.92 (0.47-1.5), the cane speaks in their windows about as often as at random times (the interval includes 1): no evidence either way.
- Latency: cane update (analysis + persistence) 24.63 ms mean, 78.14 ms p95 per depth map; the depth model itself 267.97 ms mean, 958.46 ms p95 (a shared machine: load average 304.8 at the start, 170.6 at the end, on 10 cores; the earlier report measured 3.6 ms and 46 ms on a quieter one). The analysis was replayed from --maps-cache, so these are the latencies of the run that made it.

## Obstacle questions

An event counts as a hit when the cane said something (a cue or a stop) inside its useful window [t_q - 6 s, t_q + 0.5 s]; **chance** is the same for windows of the same length at random times of the same video (not overlapping the event's), **lift** = hit rate / chance (bootstrap 95% interval over events). **Sees**: a confirmed obstacle within 2 m ahead at any decision in the window, whether or not the identity memory or the aid let it speak (the same for every aid).

Aid none, walking stream, the upper bound:

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 18 | 0.621 (0.44-0.773) | 0.399 | 1.56 (1.14-2.04) | 0.724 | 0.656 | 1.1 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 3 | 0.333 (0.121-0.646) | 0.396 | 0.84 (0.22-1.32) | 0.667 | 0.731 | 0.91 |
| other should-speak events (people, vehicles, unnamed) | 29 | 11 | 0.379 (0.227-0.56) | 0.428 | 0.89 (0.5-1.33) | 0.483 | 0.55 | 0.88 |

Aid none, with the activity detector, what the live demo would say:

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 18 | 0.621 (0.44-0.773) | 0.377 | 1.65 (1.18-2.21) | 0.724 | 0.648 | 1.12 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 1 | 0.111 (0.02-0.435) | 0.344 | 0.32 (0.0-0.88) | 0.667 | 0.729 | 0.91 |
| other should-speak events (people, vehicles, unnamed) | 29 | 10 | 0.345 (0.199-0.527) | 0.336 | 1.03 (0.56-1.63) | 0.483 | 0.541 | 0.89 |

Aid cane, walking stream, the upper bound:

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 14 | 0.483 (0.314-0.656) | 0.316 | 1.53 (0.93-2.38) | 0.724 | 0.657 | 1.1 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 2 | 0.222 (0.063-0.547) | 0.276 | 0.8 (0.0-1.92) | 0.667 | 0.711 | 0.94 |
| other should-speak events (people, vehicles, unnamed) | 29 | 10 | 0.345 (0.199-0.527) | 0.377 | 0.92 (0.47-1.5) | 0.483 | 0.538 | 0.9 |

Aid cane, with the activity detector, what the live demo would say:

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 14 | 0.483 (0.314-0.656) | 0.296 | 1.63 (0.99-2.53) | 0.724 | 0.67 | 1.08 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 1 | 0.111 (0.02-0.435) | 0.216 | 0.51 (0.0-2.63) | 0.667 | 0.696 | 0.96 |
| other should-speak events (people, vehicles, unnamed) | 29 | 10 | 0.345 (0.199-0.527) | 0.324 | 1.06 (0.53-1.83) | 0.483 | 0.554 | 0.87 |

Aid guide-dog, walking stream, the upper bound:

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 0 | 0.0 (0.0-0.117) | 0.088 | no hits (0/29) | 0.724 | 0.654 | 1.11 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 0 | 0.0 (0.0-0.299) | 0.011 | no hits (0/9) | 0.667 | 0.709 | 0.94 |
| other should-speak events (people, vehicles, unnamed) | 29 | 1 | 0.034 (0.006-0.172) | 0.043 | 0.79 (0.0-3.75) | 0.483 | 0.536 | 0.9 |

Aid guide-dog, with the activity detector, what the live demo would say:

| events | n | hits | hit rate (95% CI) | chance | lift | sees | sees chance | sees lift |
|---|---|---|---|---|---|---|---|---|
| obstacle, not COCO (door, stairs, pole, wall, fence, curb, tree, sign, ...) | 29 | 0 | 0.0 (0.0-0.117) | 0.072 | no hits (0/29) | 0.724 | 0.646 | 1.12 |
| obstacle, COCO-covered (bench, chair, table, ...) | 9 | 0 | 0.0 (0.0-0.299) | 0.002 | no hits (0/9) | 0.667 | 0.709 | 0.94 |
| other should-speak events (people, vehicles, unnamed) | 29 | 1 | 0.034 (0.006-0.172) | 0.025 | 1.36 (0.0-7.69) | 0.483 | 0.56 | 0.86 |

## What these labels are

EgoBlind events are questions blind users asked while filming, weakly labelled from the question and the answers (`seeforme/egoblind/data.py`). An obstacle-type event names a door, stairs, a pole, a fence ... somewhere in the question or answer, which is often a destination ("how can I find the entrance?") or something to the side, not an obstacle in the walking corridor; its window is set by when the question was asked, not by when the object was close. The samples are small. Read the hit rates as weak evidence and the lift against random windows as the useful part; neither is a recall. Cues per minute count every cane cue as a potential false alarm, which overstates them (some are real walls and doors). None of this says whether a cue was useful to a cane or guide-dog user: that needs a walking study.

## Obstacle events (walking stream)

| video | t_q | names | question | hit, none | hit, cane | hit, guide-dog | sees |
|---|---|---|---|---|---|---|---|
| 0 | 47.0 | door | Which direction will I need to move to exit the vehicle? | no | no | no | yes |
| 89 | 83.0 | door | Is the path clear to enter the store? | yes | no | no | yes |
| 124 | 15.0 | door, horse | In which direction should I guide the horse to leave the arena? | yes | yes | no | yes |
| 127 | 4.0 | fence, horse | Which direction is the horse heading? | no | no | no | yes |
| 183 | 2.0 | door, sign | Where's the nearest stairway? | yes | yes | no | yes |
| 347 | 11.0 | wall | Are there any obstacles on the road? | yes | no | no | yes |
| 347 | 33.3 | fence | How can I negotiate this ramp safely? | yes | yes | no | yes |
| 347 | 37.3 | fence | How can I get to the exit from here? | yes | no | no | yes |
| 362 | 52.3 | fence, stairs, person | How to use this escalator safely? | yes | yes | no | yes |
| 434 | 19.0 | door, person | How do I navigate this cafe? | yes | yes | no | yes |
| 438 | 15.0 | door | Are there steps or a ramp at the entrance to this place? | yes | yes | no | yes |
| 469 | 76.6 | door | Where is the cashier? | no | yes | no | yes |
| 665 | 41.2 | door, machine | Is there an obstacle ahead? | yes | yes | no | yes |
| 668 | 39.0 | tree, car | Is there an obstacle ahead? | yes | yes | no | yes |
| 668 | 41.0 | tree | Where have I been? | yes | yes | no | yes |
| 677 | 34.5 | stairs | Where to go next? | no | no | no | no |
| 708 | 34.6 | car | Is there an obstacle ahead? | no | no | no | no |
| 710 | 1.6 | pole | Is there an obstacle ahead? | yes | yes | no | yes |
| 781 | 35.0 | door | How can I reach the exit from this table? | yes | no | no | yes |
| 805 | 3.0 | fence | Can I walk on the road ahead smoothly? | no | no | no | no |
| 807 | 35.0 | fence | Which direction should I go to reach the end of the fence? | yes | no | no | yes |
| 817 | 23.0 | fence, train | Is it safe to proceed? | yes | yes | no | yes |
| 870 | 0.0 | stairs | Which direction does the moving walkway go? | no | no | no | no |
| 872 | 4.0 | door, person | How can I find the entrance? | no | no | no | no |
| 878 | 86.8 | door | Am I almost at the hospital? | yes | no | no | yes |
| 884 | 5.9 | pole, person | What is ahead? | no | yes | no | no |
| 884 | 152.1 | stairs | Where is the subway station? | no | no | no | no |
| 910 | 14.8 | door | Are there any obstacles ahead? | yes | yes | no | yes |
| 918 | 92.7 | door | Has the place for making keys arrived? | no | no | no | no |
| 184 | 5.0 | dining table | How do I navigate to the table? | no | no | no | yes |
| 405 | 27.0 | dining table, chair | What obstacles should I be aware of when moving in the area? | yes | yes | no | yes |
| 406 | 153.0 | dining table | What obstructions on the counter should I be aware of? | no | no | no | no |
| 407 | 69.0 | dining table | Are there any obstacles around the counter? | yes | no | no | yes |
| 439 | 4.0 | dining table | Is there a counter nearby where I can get help? | no | no | no | yes |
| 440 | 13.0 | shelf, dining table, chair | Are there any obstacles or objects on the floor that could cause me to | no | no | no | yes |
| 472 | 50.5 | person, dining table | Can you tell me where to get to the nearest assistance point? | no | no | no | no |
| 910 | 118.6 | tree, potted plant | Where is this? | no | no | no | no |
| 919 | 92.8 | dining table | Where is the checkout page? | yes | yes | no | yes |

Cane counters, aid none (walking stream, summed): said 342, stop 84, path 50, head 10, held 1066, explained 243, still 0, close_up 896, noted 0, aid 0.
Cane counters, aid cane (walking stream, summed): said 193, stop 29, path 15, head 11, held 500, explained 316, still 0, close_up 902, noted 0, aid 1172.
Cane counters, aid guide-dog (walking stream, summed): said 11, stop 29, path 0, head 11, held 42, explained 316, still 0, close_up 902, noted 0, aid 1812.
