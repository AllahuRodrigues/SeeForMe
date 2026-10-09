# EgoBlind recall diagnosis (holdout: 77 videos, 63 should-speak events)

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only | egoblind_learned:rl@models/egoblind_only |
|---|---|---|---|---|---|
| spoken_timely | 0 | 0 | 0 | 5 | 0 |
| spoken_early | 0 | 0 | 0 | 1 | 0 |
| spoken_late | 1 | 0 | 1 | 2 | 0 |
| not_coverable | 33 | 33 | 33 | 33 | 33 |
| not_detected | 7 | 7 | 7 | 7 | 7 |
| never_tracked | 2 | 2 | 2 | 2 | 2 |
| low_confidence | 1 | 1 | 0 | 1 | 1 |
| below_threshold | 19 | 20 | 20 | 12 | 20 |
| budget_blocked | 0 | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 | 0 |

Coverable events only:

| bin | pcacs | pcacs_egoblind | pcacs_learned:rl | egoblind_learned:il@models/egoblind_only | egoblind_learned:rl@models/egoblind_only |
|---|---|---|---|---|---|
| spoken_timely | 0 | 0 | 0 | 5 | 0 |
| spoken_early | 0 | 0 | 0 | 1 | 0 |
| spoken_late | 1 | 0 | 1 | 2 | 0 |
| not_detected | 7 | 7 | 7 | 7 | 7 |
| never_tracked | 2 | 2 | 2 | 2 | 2 |
| low_confidence | 1 | 1 | 0 | 1 | 1 |
| below_threshold | 19 | 20 | 20 | 12 | 20 |
| budget_blocked | 0 | 0 | 0 | 0 | 0 |
| channel_busy | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 | 0 |
