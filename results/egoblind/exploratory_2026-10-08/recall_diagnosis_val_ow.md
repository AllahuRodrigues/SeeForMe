# EgoBlind recall diagnosis (val: 97 videos, 59 should-speak events)

| bin | pcacs | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|
| spoken_timely | 3 | 3 | 1 |
| spoken_early | 0 | 0 | 0 |
| spoken_late | 1 | 0 | 3 |
| not_coverable | 15 | 15 | 15 |
| not_detected | 17 | 17 | 17 |
| never_tracked | 2 | 2 | 2 |
| low_confidence | 2 | 2 | 2 |
| below_threshold | 19 | 17 | 19 |
| budget_blocked | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 |
| other | 0 | 3 | 0 |

Coverable events only:

| bin | pcacs | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|
| spoken_timely | 3 | 3 | 1 |
| spoken_early | 0 | 0 | 0 |
| spoken_late | 1 | 0 | 3 |
| not_detected | 17 | 17 | 17 |
| never_tracked | 2 | 2 | 2 |
| low_confidence | 2 | 2 | 2 |
| below_threshold | 19 | 17 | 19 |
| budget_blocked | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 |
| other | 0 | 3 | 0 |
