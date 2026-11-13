E3-V: the same candidates and the base 29 features under two kinds of label (standardised logistic; intervals: bootstrap over held-out source videos)

| | EgoBlind, question proxies | VIABench, human proactive reminders |
|---|---|---|
| fit on | `train`: 265 videos, 20774 rows, 3454 positive | `via_train`: 149 videos, 22599 rows, 6326 positive |
| scored on | `finetune`: 48 videos, 4479 rows, 959 positive | `via_dev`: 47 videos, 7136 rows, 2140 positive |
| AUC, base 29 features | **0.674 [0.585, 0.768]** | **0.718 [0.633, 0.780]** |
| AUC, object class only | 0.612 [0.532, 0.712] | 0.653 [0.588, 0.711] |
| AUC inside `person` rows | 0.655 [0.550, 0.765] (627 positive of 2722) | 0.726 [0.633, 0.790] (1456 positive of 3646) |
| AUC with the other dataset's model | 0.622 [0.493, 0.731] | 0.599 [0.513, 0.669] |
| AUC inside one video (row-weighted mean) | 0.627 [0.540, 0.740] (19 videos with both classes, 29 skipped) | 0.646 [0.589, 0.715] (34 videos with both classes, 13 skipped) |
| AUC after permuting the training labels (200 draws) | 0.504 [0.366, 0.623], p = 0.005 | 0.493 [0.388, 0.601], p = 0.005 |

VIABench learning curve (AUC on `via_dev`, mean over seeded subsets of training videos): 10 videos 0.601 (0.526 to 0.689); 20 videos 0.644 (0.612 to 0.682); 40 videos 0.687 (0.669 to 0.720); 80 videos 0.699 (0.667 to 0.716); 120 videos 0.711 (0.696 to 0.717); all 149 videos 0.718.

Perception coverage of the human reminders (cached first part of 200 videos): 1644 reminders, 1185 name an object the label table knows, 712 have that object detected inside the reminder's window (43% of all).

| VIABench task | reminders | name an object | detector saw it in the window |
|---|---|---|---|
| Obstacle Alert | 839 | 695 | 535 |
| Active Avoidance | 235 | 187 | 138 |
| Stairs Up/Down | 76 | 76 | 1 |
| Service Facility Detection | 71 | 29 | 15 |
| Exit/Entry Recognition | 63 | 63 | 1 |
| Camera Content Anomaly | 54 | 0 | 0 |
| Road Branch | 41 | 1 | 0 |
| Road Surface Condition | 37 | 8 | 0 |
| Camera Obstruction | 29 | 2 | 1 |
| Direction Guidance | 27 | 10 | 1 |
| Blind Path Recognition | 25 | 25 | 0 |
| Pedestrian Traffic Light | 22 | 22 | 13 |
| Crossing to Other Side | 22 | 0 | 0 |
| Staircase Up/Down | 22 | 22 | 0 |
