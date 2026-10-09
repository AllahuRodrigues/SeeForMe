# val: weak-event coverage under two perception caches (97 videos in both; 97 / 97 cached)

| | YOLO11n (COCO) | YOLO-World (NAV_VOCAB) |
|---|---|---|
| events | 59 | 59 |
| coverable | 26 | 44 |
| detected | 22 | 27 |
| uncoverable:no_object_named | 4 | 4 |
| uncoverable:not_detectable | 18 | 1 |
| uncoverable:taxonomy_decor | 11 | 10 |

Events whose coverability or detection changed (20):

| video | concepts | labels | coverable a -> b | detected a -> b | question |
|---|---|---|---|---|---|
| 0 | door | door | False -> True | None -> False | Which direction will I need to move to exit the vehicle? |
| 21 | water | backpack,handbag,sink | True -> True | True -> False | Where's the lid? |
| 35 | tree | tree | False -> True | None -> False | Is there any tree or obstacle directly in front? |
| 91 | shelf | shelf | False -> True | None -> True | Is there anything on the floor? |
| 119 | fence | horse,fence | False -> True | None -> False | Where is the horse going? |
| 329 | pole | pole | False -> True | None -> True | What obstacles are ahead of me on the sidewalk? |
| 371 | fence | fence | False -> True | None -> True | Is there an obstruction in front of me on the sidewalk? |
| 378 | door | door | False -> True | None -> False | Can you tell me where the door is? |
| 392 | pole | pole | False -> True | None -> True | Are there any obstacles in my way as I walk now? |
| 441 | animal | animal | False -> True | None -> False | Are there any animals around me that I should be wary of? |
| 469 | door | door | False -> True | None -> False | Where is the cashier? |
| 497 | door | door | False -> True | None -> False | Where is the closest door to my current location? |
| 602 | pole | pole | False -> True | None -> False | How can I avoid the pole in front of me? |
| 602 | tree | tree | False -> True | None -> True | Is there an obstacle in front of me? |
| 665 | door,machine | door,machine | False -> True | None -> False | Is there an obstacle ahead? |
| 672 | stairs | stairs | False -> True | None -> False | Where are the stairs? |
| 714 | stairs | stairs | False -> True | None -> False | Is there an obstacle ahead? |
| 805 | fence | fence | False -> True | None -> False | Can I walk on the road ahead smoothly? |
| 910 | door | door | False -> True | None -> False | Are there any obstacles ahead? |
| 910 | tree | potted plant,tree | True -> True | False -> True | Where is this? |

Detections by label in YOLO-World (NAV_VOCAB) caches: person 36033, car 4000, motorcycle 3760, chair 3236, shelf 1529, trash can 1105, box 941, animal 850, dining table 786, potted plant 763, bicycle 625, truck 620, bench 421, backpack 389, machine 330, bus 205, cone 185, pole 119, stop sign 93, cart 79, traffic light 72, tree 67, sign 44, train 42, suitcase 35, curb 32, fence 28, fire hydrant 24, scooter 8, elevator 1
