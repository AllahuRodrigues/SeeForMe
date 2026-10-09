EgoBlind holdout: 77 videos, 63 should-speak events, 26 quiet windows, 54.7 min, noise p_miss=0.2,depth_rel=0.15,p_blackout=0.05

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.46 | 0.67 | 0.87 | 1.00 | 0.00 | 0.01 | 0.5 | 1.70 | 0.28 | 0.00 |
| confidence_threshold | 0.17 | 0.23 | 0.30 | 0.01 | 1.00 | 0.64 | 25.0 | 4.82 | 0.36 | 5.81 |
| confidence_cooldown | 0.32 | 0.47 | 0.61 | 0.02 | 0.88 | 0.38 | 14.8 | 4.35 | 0.33 | 3.26 |
| cacs | 0.14 | 0.23 | 0.30 | 0.02 | 0.73 | 0.27 | 7.7 | 3.09 | 0.40 | 2.39 |
| pcacs | 0.02 | 0.03 | 0.04 | 0.12 | 0.00 | 0.01 | 0.1 | 6.18 | 0.49 | 0.44 |
| pcacs_egoblind | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.41 |
| pcacs_learned:il | 0.06 | 0.07 | 0.09 | 0.02 | 0.50 | 0.09 | 4.0 | 2.86 | 0.47 | 1.14 |
| pcacs_learned:rl | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.03 | 1.2 | nan | 0.49 | 0.64 |
| egoblind_learned:il | 0.02 | 0.03 | 0.04 | 0.05 | 0.04 | 0.01 | 0.3 | 1.60 | 0.49 | 0.49 |
| egoblind_learned:rl | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.41 |
| egoblind_learned:il@models/egoblind_only | 0.02 | 0.03 | 0.04 | 0.01 | 0.08 | 0.03 | 1.3 | 9.78 | 0.48 | 0.70 |
| egoblind_learned:rl@models/egoblind_only | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.49 | 0.41 |
