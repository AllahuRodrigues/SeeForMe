EgoBlind val: 97 videos, 59 should-speak events, 38 quiet windows, 59.8 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.37 | 0.50 | 0.81 | 1.00 | 0.03 | 0.01 | 0.4 | 1.49 | 0.28 | 0.00 |
| confidence_cooldown | 0.24 | 0.32 | 0.52 | 0.02 | 0.74 | 0.34 | 13.7 | 4.90 | 0.29 | 2.49 |
| cacs | 0.19 | 0.25 | 0.41 | 0.03 | 0.71 | 0.26 | 7.3 | 4.19 | 0.30 | 1.84 |
| pcacs | 0.05 | 0.07 | 0.11 | 0.10 | 0.00 | 0.02 | 0.5 | 1.13 | 0.36 | 0.35 |
| pcacs_learned:il | 0.07 | 0.09 | 0.15 | 0.02 | 0.34 | 0.08 | 3.3 | 4.15 | 0.35 | 0.79 |
| pcacs_learned:rl | 0.05 | 0.07 | 0.11 | 0.05 | 0.08 | 0.02 | 1.0 | 2.84 | 0.36 | 0.39 |
| egoblind_learned:il@models/egoblind_only | 0.02 | 0.02 | 0.04 | 0.02 | 0.08 | 0.02 | 0.9 | -0.49 | 0.38 | 0.40 |
| egoblind_learned:rl@models/egoblind_only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.1 | nan | 0.38 | 0.28 |
