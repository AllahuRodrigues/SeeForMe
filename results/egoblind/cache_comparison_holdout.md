# holdout: weak-event coverage under two perception caches (77 videos in both; 77 / 77 cached)

| | YOLO11n (COCO) | YOLO-World (NAV_VOCAB) |
|---|---|---|
| events | 63 | 63 |
| coverable | 30 | 46 |
| detected | 23 | 23 |
| uncoverable:no_object_named | 7 | 14 |
| uncoverable:not_detectable | 11 | 0 |
| uncoverable:taxonomy_decor | 15 | 3 |

Events whose coverability or detection changed (18):

| video | concepts | labels | coverable a -> b | detected a -> b | question |
|---|---|---|---|---|---|
| 85 | wall | couch,tv,dining table | True -> True | True -> False | please describe the layout of the room I'm in? |
| 85 | door |  | False -> True | None -> False | Which direction should I go to leave the room? |
| 99 |  | dog | False -> True | None -> True | Is there anything obstructing my path? |
| 101 |  | dog | False -> True | None -> False | Is it safe for me to walk forward without tripping? |
| 189 |  | cat | False -> True | None -> False | Is there anything on the ground that I might trip over? |
| 412 |  | cat | False -> True | None -> True | Is there anything on the floor that might trip me? |
| 421 | machine | bowl | False -> True | None -> False | Is there any obstruction or hot objects around the coffee machine? |
| 545 |  | bicycle | True -> True | True -> False | How do I get through this row of bikes? |
| 548 | door |  | False -> True | None -> False | Could you help me find the way to the nearest entrance? |
| 550 | door |  | False -> True | None -> False | How far is it from here to the door? |
| 551 | door |  | False -> True | None -> False | Can you tell me how to find the entrance to this room? |
| 551 | door |  | False -> True | None -> False | Where is the entrance to the store from here? |
| 552 | door |  | False -> True | None -> False | Where is the entrance? |
| 604 | sign |  | False -> True | None -> False | Which direction should I exit? |
| 605 | stairs |  | False -> True | None -> False | Am I at the top or bottom of the escalator? |
| 894 | fence |  | False -> True | None -> False | How to get here? |
| 897 | tree |  | False -> True | None -> False | What kind of obstacle is this? |
| 897 | wall,fence |  | False -> True | None -> False | Can I go this way? |

Detections by label in YOLO-World (NAV_VOCAB) caches: person 29894, car 4619, motorcycle 4061, shelf 2706, chair 1051, trash can 835, box 577, machine 522, potted plant 500, bench 475, bicycle 416, animal 413, dining table 307, cone 250, truck 247, backpack 225, pole 224, bus 216, traffic light 189, sign 115, suitcase 77, fire hydrant 72, stop sign 53, train 51, cart 42, curb 21, fence 4, tree 3, elevator 1, scooter 1
