EgoBlind holdout: 77 videos, 63 should-speak events, 26 quiet windows, 54.7 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.32 | 0.43 | 0.87 | 1.00 | 0.00 | 0.01 | 0.4 | 1.83 | 0.36 | 0.00 |
| confidence_cooldown | 0.21 | 0.28 | 0.57 | 0.02 | 0.77 | 0.33 | 12.8 | 4.59 | 0.37 | 2.90 |
| cacs | 0.11 | 0.15 | 0.30 | 0.02 | 0.81 | 0.25 | 6.9 | 3.21 | 0.42 | 2.13 |
| pcacs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.3 | nan | 0.49 | 0.38 |
| pcacs_egoblind | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.29 |
| pcacs_learned:il | 0.08 | 0.11 | 0.22 | 0.03 | 0.23 | 0.09 | 3.6 | 4.86 | 0.44 | 0.97 |
| pcacs_learned:rl | 0.05 | 0.07 | 0.13 | 0.04 | 0.04 | 0.03 | 1.2 | 5.85 | 0.46 | 0.49 |
| egoblind_learned:il@models/egoblind_only | 0.10 | 0.13 | 0.26 | 0.08 | 0.00 | 0.03 | 1.4 | 3.33 | 0.45 | 0.48 |
