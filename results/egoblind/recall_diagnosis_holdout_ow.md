# EgoBlind recall diagnosis (holdout: 77 videos, 63 should-speak events)

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|---|
| spoken_timely | 0 | 0 | 3 | 6 |
| spoken_early | 0 | 0 | 0 | 0 |
| spoken_late | 0 | 0 | 1 | 2 |
| not_coverable | 17 | 17 | 17 | 17 |
| not_detected | 23 | 23 | 23 | 23 |
| never_tracked | 1 | 1 | 1 | 1 |
| low_confidence | 1 | 1 | 1 | 1 |
| below_threshold | 21 | 21 | 17 | 13 |
| budget_blocked | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 |

Coverable events only:

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only |
|---|---|---|---|---|
| spoken_timely | 0 | 0 | 3 | 6 |
| spoken_early | 0 | 0 | 0 | 0 |
| spoken_late | 0 | 0 | 1 | 2 |
| not_detected | 23 | 23 | 23 | 23 |
| never_tracked | 1 | 1 | 1 | 1 |
| low_confidence | 1 | 1 | 1 | 1 |
| below_threshold | 21 | 21 | 17 | 13 |
| budget_blocked | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 |
