EgoBlind holdout: 77 videos, 63 should-speak events, 26 quiet windows, 54.7 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.48 | 0.67 | 0.87 | 1.00 | 0.00 | 0.01 | 0.5 | 1.73 | 0.27 | 0.00 |
| confidence_threshold | 0.19 | 0.27 | 0.35 | 0.01 | 1.00 | 0.69 | 26.8 | 3.60 | 0.36 | 6.21 |
| confidence_cooldown | 0.33 | 0.53 | 0.70 | 0.02 | 1.00 | 0.42 | 16.3 | 3.84 | 0.31 | 3.60 |
| cacs | 0.19 | 0.33 | 0.43 | 0.03 | 0.85 | 0.30 | 8.5 | 3.55 | 0.39 | 2.56 |
| pcacs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.1 | nan | 0.49 | 0.48 |
| pcacs_egoblind | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.43 |
| pcacs_learned:il | 0.03 | 0.03 | 0.04 | 0.01 | 0.42 | 0.09 | 4.1 | 2.19 | 0.48 | 1.19 |
| pcacs_learned:rl | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.03 | 1.1 | nan | 0.49 | 0.63 |
| egoblind_learned:il | 0.06 | 0.13 | 0.17 | 0.09 | 0.12 | 0.02 | 0.9 | 3.92 | 0.47 | 0.60 |
| egoblind_learned:rl | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.43 |
| egoblind_learned:il@models/egoblind_only | 0.08 | 0.17 | 0.22 | 0.06 | 0.12 | 0.04 | 1.6 | 3.53 | 0.46 | 0.66 |
| egoblind_learned:rl@models/egoblind_only | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.43 |
