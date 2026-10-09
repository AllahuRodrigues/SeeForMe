EgoBlind val: 97 videos, 59 should-speak events, 38 quiet windows, 59.8 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.42 | 0.77 | 0.86 | 1.00 | 0.03 | 0.01 | 0.4 | 1.35 | 0.24 | 0.00 |
| confidence_cooldown | 0.22 | 0.38 | 0.45 | 0.01 | 0.95 | 0.45 | 17.8 | 4.82 | 0.28 | 3.25 |
| cacs | 0.25 | 0.50 | 0.59 | 0.03 | 0.68 | 0.30 | 8.7 | 4.19 | 0.28 | 2.04 |
| pcacs | 0.03 | 0.08 | 0.09 | 0.06 | 0.00 | 0.02 | 0.5 | 2.03 | 0.37 | 0.39 |
| pcacs_learned:il | 0.08 | 0.15 | 0.18 | 0.02 | 0.24 | 0.09 | 3.8 | 2.50 | 0.36 | 0.83 |
| pcacs_learned:rl | 0.03 | 0.08 | 0.09 | 0.03 | 0.03 | 0.02 | 1.0 | 1.27 | 0.38 | 0.41 |
| egoblind_learned:il@models/egoblind_only | 0.10 | 0.23 | 0.27 | 0.10 | 0.11 | 0.02 | 1.0 | 5.69 | 0.35 | 0.35 |
| egoblind_learned:rl@models/egoblind_only | 0.02 | 0.04 | 0.05 | 0.14 | 0.00 | 0.00 | 0.1 | 0.00 | 0.38 | 0.29 |
