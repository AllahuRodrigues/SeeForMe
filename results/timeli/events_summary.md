# TIMELI -> SeeForMe events

`/Users/allahurodrigues/Desktop/Desktop/My-Desk/Yale/Yale-Senior/cpsc-4900/Project/data/timeli/events.jsonl`: 24937 events (one JSON line each; speak events at the instruction second with a +/-1 s window, quiet windows for silent runs of at least 2 s).

| split | videos | seconds | speak events | per min | quiet windows | quiet s | mean severity | coverable (COCO + taxonomy) | reason inferred from text |
|---|---|---|---|---|---|---|---|---|---|
| train | 269 | 56281 | 11631 | 12.4 | 8651 | 41525 | 0.573 | 3121 | 0 |
| test | 20 | 10820 | 2385 | 13.23 | 1757 | 7795 | 0.579 | 765 | 0 |
| real | 11 | 1059 | 308 | 17.45 | 205 | 692 | 0.479 | 154 | 126 |

train: reasons {'constant_instruction': 2446, 'obstacle_in_front': 3588, 'direction_change': 3814, 'enter_junction': 890, 'exit_junction': 884, 'stop': 9}; event types {'goal': 6269, 'person_moving': 1991, 'landmark': 1774, 'hazard_static': 1019, 'unknown_obstacle': 467, 'vehicle': 111}; labels {'path': 6269, 'person': 1991, 'junction': 1774, 'pole': 1019, 'unknown object': 467, 'car': 111}

test: reasons {'constant_instruction': 419, 'direction_change': 811, 'obstacle_in_front': 898, 'enter_junction': 129, 'exit_junction': 128}; event types {'goal': 1230, 'person_moving': 500, 'landmark': 257, 'hazard_static': 248, 'unknown_obstacle': 133, 'vehicle': 17}; labels {'path': 1230, 'person': 500, 'junction': 257, 'pole': 248, 'unknown object': 133, 'car': 17}

real: reasons {'obstacle_in_front': 183, 'constant_instruction': 115, 'exit_junction': 5, 'enter_junction': 4, 'constantly_instruct': 1}; event types {'goal': 116, 'person_moving': 104, 'hazard_static': 35, 'unknown_obstacle': 29, 'landmark': 9, 'barrier': 6, 'vehicle': 6, 'bike': 3}; labels {'path': 116, 'person': 104, 'unknown object': 24, 'pole': 22, 'junction': 9, 'tree': 8, 'fence': 6, 'car': 6, 'obstacle': 4, 'bicycle': 3, 'trash can': 2, 'potted plant': 1}

Cannot be mapped: object identity / tracks (class + coarse side only); distance and time-to-contact (obstacles are 'within cane range' by construction); earliest-useful windows (one timestamp per instruction, +/-1 s tolerance); sudden events and interruption semantics; user-confirmed clear paths (quiet windows mean 'no trigger fired'); periodic reassurance (constant_instruction) has no SeeForMe event type; frames and videos (not downloaded: 32 GB / 38 GB / 537 GB), so no perception replay yet.
