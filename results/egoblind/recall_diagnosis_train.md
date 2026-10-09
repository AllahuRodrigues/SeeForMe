# EgoBlind recall diagnosis (train: 454 videos, 237 should-speak events)

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|---|
| spoken_timely | 11 | 0 | 17 | 70 |
| spoken_early | 2 | 0 | 1 | 15 |
| spoken_late | 6 | 0 | 3 | 2 |
| not_coverable | 104 | 104 | 103 | 100 |
| not_detected | 13 | 13 | 13 | 13 |
| never_tracked | 6 | 6 | 6 | 6 |
| low_confidence | 2 | 2 | 2 | 2 |
| below_threshold | 92 | 112 | 92 | 28 |
| budget_blocked | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 |
| other | 1 | 0 | 0 | 1 |

Coverable events only:

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|---|
| spoken_timely | 11 | 0 | 16 | 66 |
| spoken_early | 2 | 0 | 1 | 15 |
| spoken_late | 6 | 0 | 3 | 2 |
| not_detected | 13 | 13 | 13 | 13 |
| never_tracked | 6 | 6 | 6 | 6 |
| low_confidence | 2 | 2 | 2 | 2 |
| below_threshold | 92 | 112 | 92 | 28 |
| budget_blocked | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 |
| other | 1 | 0 | 0 | 1 |
