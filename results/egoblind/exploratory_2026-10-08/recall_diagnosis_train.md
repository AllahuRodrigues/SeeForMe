# EgoBlind recall diagnosis (train: 562 videos, 338 should-speak events)

| bin | pcacs | pcacs_learned:rl | egoblind_learned:il |
|---|---|---|---|
| spoken_timely | 11 | 12 | 66 |
| spoken_early | 1 | 2 | 7 |
| spoken_late | 9 | 6 | 11 |
| not_coverable | 154 | 153 | 153 |
| not_detected | 30 | 30 | 30 |
| never_tracked | 8 | 8 | 7 |
| low_confidence | 3 | 2 | 3 |
| below_threshold | 120 | 104 | 61 |
| budget_blocked | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 |
| other | 2 | 21 | 0 |

Coverable events only:

| bin | pcacs | pcacs_learned:rl | egoblind_learned:il |
|---|---|---|---|
| spoken_timely | 11 | 11 | 65 |
| spoken_early | 1 | 2 | 7 |
| spoken_late | 9 | 6 | 11 |
| not_detected | 30 | 30 | 30 |
| never_tracked | 8 | 8 | 7 |
| low_confidence | 3 | 2 | 3 |
| below_threshold | 120 | 104 | 61 |
| budget_blocked | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 |
| other | 2 | 21 | 0 |
