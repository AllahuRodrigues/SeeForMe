EgoBlind val: 47 videos, 21 should-speak events, 23 quiet windows, 20.4 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.48 | 1.00 | 1.00 | 1.00 | 0.00 | 0.01 | 0.5 | 1.50 | 0.16 | 0.00 |
| pcacs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.2 | nan | 0.30 | 0.29 |
| pcacs_learned:il | 0.14 | 0.33 | 0.33 | 0.03 | 0.22 | 0.11 | 4.7 | 0.77 | 0.26 | 0.68 |
| pcacs_learned:rl | 0.05 | 0.17 | 0.17 | 0.05 | 0.04 | 0.02 | 1.1 | -0.05 | 0.29 | 0.33 |
| egoblind_learned:il | 0.05 | 0.17 | 0.17 | 0.03 | 0.04 | 0.03 | 1.7 | 1.80 | 0.29 | 0.38 |
| egoblind_learned:rl | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.30 | 0.24 |
| egoblind_learned:il@models/egoblind_only | 0.14 | 0.50 | 0.50 | 0.12 | 0.04 | 0.02 | 1.2 | 3.79 | 0.26 | 0.28 |
| egoblind_learned:rl@models/egoblind_only | 0.05 | 0.17 | 0.17 | 0.33 | 0.00 | 0.00 | 0.1 | 0.00 | 0.29 | 0.23 |
