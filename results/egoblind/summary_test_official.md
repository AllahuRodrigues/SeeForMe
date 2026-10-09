EgoBlind test: 656 videos, 367 should-speak events, 297 quiet windows, 371.8 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.47 | 0.78 | 0.89 | 1.00 | 0.02 | 0.01 | 0.5 | 1.74 | 0.19 | 0.00 |
| confidence_threshold | 0.26 | 0.45 | 0.52 | 0.01 | 0.89 | 0.68 | 26.5 | 4.90 | 0.23 | 4.68 |
| confidence_cooldown | 0.34 | 0.57 | 0.66 | 0.02 | 0.83 | 0.41 | 16.0 | 5.03 | 0.22 | 2.75 |
| cacs | 0.25 | 0.42 | 0.48 | 0.03 | 0.62 | 0.30 | 8.8 | 4.98 | 0.26 | 1.97 |
| pcacs | 0.04 | 0.07 | 0.08 | 0.06 | 0.04 | 0.02 | 0.6 | 3.58 | 0.36 | 0.38 |
| pcacs_egoblind | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.37 | 0.29 |
| pcacs_learned:il | 0.11 | 0.20 | 0.24 | 0.03 | 0.25 | 0.10 | 4.0 | 4.81 | 0.32 | 0.87 |
| pcacs_learned:rl | 0.06 | 0.10 | 0.12 | 0.04 | 0.06 | 0.03 | 1.3 | 4.01 | 0.34 | 0.46 |
| egoblind_learned:il | 0.06 | 0.12 | 0.13 | 0.07 | 0.05 | 0.02 | 0.9 | 4.01 | 0.34 | 0.38 |
| egoblind_learned:rl | 0.00 | 0.01 | 0.01 | 0.05 | 0.00 | 0.00 | 0.1 | 1.38 | 0.37 | 0.30 |
| egoblind_learned:il@models/egoblind_only | 0.11 | 0.20 | 0.22 | 0.06 | 0.13 | 0.04 | 1.9 | 3.54 | 0.33 | 0.49 |
| egoblind_learned:rl@models/egoblind_only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | nan | 0.37 | 0.29 |
