EgoBlind val: 47 videos, 21 should-speak events, 23 quiet windows, 20.4 min, noise p_miss=0.2,depth_rel=0.15,p_blackout=0.05

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.43 | 1.00 | 1.00 | 1.00 | 0.00 | 0.01 | 0.4 | 1.78 | 0.17 | 0.00 |
| confidence_cooldown | 0.29 | 0.67 | 0.67 | 0.02 | 0.87 | 0.40 | 15.9 | 4.41 | 0.20 | 2.19 |
| cacs | 0.29 | 0.83 | 0.83 | 0.03 | 0.70 | 0.32 | 9.7 | 5.65 | 0.21 | 1.62 |
| pcacs | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.01 | 0.4 | nan | 0.30 | 0.29 |
| pcacs_egoblind | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.30 | 0.23 |
| pcacs_learned:il | 0.05 | 0.17 | 0.17 | 0.01 | 0.26 | 0.10 | 4.6 | 3.39 | 0.28 | 0.74 |
| pcacs_learned:rl | 0.05 | 0.17 | 0.17 | 0.04 | 0.00 | 0.03 | 1.1 | 3.39 | 0.28 | 0.33 |
| egoblind_learned:il | 0.10 | 0.33 | 0.33 | 0.06 | 0.04 | 0.03 | 1.7 | 2.57 | 0.28 | 0.35 |
| egoblind_learned:rl | 0.00 | 0.00 | 0.00 | nan | 0.00 | 0.00 | 0.0 | nan | 0.30 | 0.23 |
