EgoBlind test: 656 videos, 367 should-speak events, 297 quiet windows, 371.8 min, noise p_miss=0.2,depth_rel=0.15,p_blackout=0.05

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.45 | 0.74 | 0.84 | 1.00 | 0.02 | 0.01 | 0.4 | 1.65 | 0.20 | 0.00 |
| confidence_threshold | 0.27 | 0.47 | 0.53 | 0.01 | 0.87 | 0.63 | 24.6 | 4.84 | 0.23 | 4.32 |
| confidence_cooldown | 0.31 | 0.55 | 0.63 | 0.02 | 0.81 | 0.38 | 14.8 | 5.36 | 0.23 | 2.55 |
| cacs | 0.23 | 0.39 | 0.45 | 0.03 | 0.55 | 0.27 | 7.9 | 4.92 | 0.26 | 1.76 |
| pcacs | 0.03 | 0.06 | 0.06 | 0.06 | 0.03 | 0.02 | 0.5 | 3.68 | 0.36 | 0.36 |
| pcacs_egoblind | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.37 | 0.27 |
| pcacs_learned:il | 0.12 | 0.22 | 0.25 | 0.03 | 0.27 | 0.10 | 4.1 | 4.39 | 0.31 | 0.85 |
| pcacs_learned:rl | 0.06 | 0.10 | 0.11 | 0.04 | 0.08 | 0.04 | 1.4 | 3.82 | 0.34 | 0.46 |
| egoblind_learned:il | 0.05 | 0.09 | 0.10 | 0.07 | 0.03 | 0.01 | 0.7 | 3.97 | 0.35 | 0.34 |
| egoblind_learned:rl | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | nan | 0.37 | 0.28 |
| egoblind_learned:il@models/egoblind_only | 0.08 | 0.15 | 0.17 | 0.06 | 0.08 | 0.03 | 1.4 | 2.84 | 0.33 | 0.42 |
| egoblind_learned:rl@models/egoblind_only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | nan | 0.37 | 0.27 |
