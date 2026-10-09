# EgoBlind recall diagnosis (test: 656 videos, 367 should-speak events)

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only | egoblind_learned:rl@models/egoblind_only |
|---|---|---|---|---|---|
| spoken_timely | 14 | 0 | 21 | 40 | 0 |
| spoken_early | 1 | 0 | 2 | 4 | 0 |
| spoken_late | 5 | 0 | 8 | 13 | 0 |
| not_coverable | 167 | 167 | 167 | 167 | 167 |
| not_detected | 25 | 26 | 25 | 25 | 26 |
| never_tracked | 13 | 13 | 13 | 13 | 13 |
| low_confidence | 1 | 1 | 1 | 1 | 1 |
| below_threshold | 141 | 160 | 130 | 102 | 160 |
| budget_blocked | 0 | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 2 | 0 |

Coverable events only:

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only | egoblind_learned:rl@models/egoblind_only |
|---|---|---|---|---|---|
| spoken_timely | 14 | 0 | 21 | 40 | 0 |
| spoken_early | 1 | 0 | 2 | 4 | 0 |
| spoken_late | 5 | 0 | 8 | 13 | 0 |
| not_detected | 25 | 26 | 25 | 25 | 26 |
| never_tracked | 13 | 13 | 13 | 13 | 13 |
| low_confidence | 1 | 1 | 1 | 1 | 1 |
| below_threshold | 141 | 160 | 130 | 102 | 160 |
| budget_blocked | 0 | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 2 | 0 |
