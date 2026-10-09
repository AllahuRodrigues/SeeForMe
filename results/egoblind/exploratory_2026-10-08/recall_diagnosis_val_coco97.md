# EgoBlind recall diagnosis (val: 97 videos, 59 should-speak events)

| bin | pcacs | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|
| spoken_timely | 2 | 2 | 6 |
| spoken_early | 0 | 0 | 0 |
| spoken_late | 1 | 1 | 3 |
| not_coverable | 33 | 33 | 33 |
| not_detected | 4 | 4 | 4 |
| never_tracked | 1 | 1 | 1 |
| low_confidence | 0 | 0 | 0 |
| below_threshold | 18 | 15 | 12 |
| budget_blocked | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 |
| other | 0 | 3 | 0 |

Coverable events only:

| bin | pcacs | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|
| spoken_timely | 2 | 2 | 6 |
| spoken_early | 0 | 0 | 0 |
| spoken_late | 1 | 1 | 3 |
| not_detected | 4 | 4 | 4 |
| never_tracked | 1 | 1 | 1 |
| low_confidence | 0 | 0 | 0 |
| below_threshold | 18 | 15 | 12 |
| budget_blocked | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 |
| other | 0 | 3 | 0 |
