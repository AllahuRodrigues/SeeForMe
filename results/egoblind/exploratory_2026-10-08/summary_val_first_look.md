EgoBlind val: 47 videos, 21 should-speak events, 23 quiet windows, 20.4 min

| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.48 | 1.00 | 1.00 | 1.00 | 0.00 | 0.01 | 0.5 | 1.50 | 0.16 | 0.00 |
| confidence_threshold | 0.19 | 0.67 | 0.67 | 0.01 | 0.96 | 0.72 | 28.6 | 5.05 | 0.21 | 6.60 |
| confidence_cooldown | 0.29 | 0.67 | 0.67 | 0.02 | 0.96 | 0.42 | 16.9 | 4.59 | 0.20 | 3.86 |
| cacs | 0.33 | 0.83 | 0.83 | 0.03 | 0.83 | 0.35 | 10.6 | 5.40 | 0.20 | 2.76 |
| pcacs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.2 | nan | 0.30 | 0.32 |
| pcacs_learned:il | 0.14 | 0.33 | 0.33 | 0.03 | 0.22 | 0.11 | 4.7 | 0.77 | 0.26 | 1.17 |
| pcacs_learned:rl | 0.05 | 0.17 | 0.17 | 0.05 | 0.04 | 0.02 | 1.1 | -0.05 | 0.29 | 0.44 |
