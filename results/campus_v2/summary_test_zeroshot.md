| policy | action-weighted recall | late critical rate | mean lead (s) | cue precision | audio occupancy | cues/min | haptics/min | silence precision | critical blocking | min-message ratio | regret |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.683 ± 0.22 | 0.25 | 1.06 | 0.994 | 0.21 | 6.9 | 0.0 | 0.82 | 0.05 | 1.00 | 0.00 |
| confidence_threshold | 0.204 ± 0.26 | 0.25 | 1.94 | 0.052 | 0.70 | 30.1 | 0.0 | 1.00 | 0.15 | 0.47 | 8.17 |
| confidence_cooldown | 0.203 ± 0.25 | 0.22 | 2.07 | 0.063 | 0.61 | 25.4 | 0.0 | 0.98 | 0.12 | 0.47 | 7.02 |
| cacs | 0.180 ± 0.26 | 0.63 | 0.95 | 0.206 | 0.21 | 7.9 | 0.0 | 0.78 | 0.15 | 0.38 | 3.20 |
| pcacs | 0.465 ± 0.30 | 0.27 | 1.92 | 0.598 | 0.19 | 6.8 | 0.3 | 0.80 | 0.06 | 0.97 | 1.26 |
| pcacs_learned:il | 0.617 ± 0.30 | 0.23 | 2.69 | 0.736 | 0.23 | 8.3 | 0.7 | 0.95 | 0.08 | 0.87 | 0.62 |
| pcacs_learned:rl | 0.632 ± 0.27 | 0.24 | 2.39 | 0.888 | 0.19 | 6.8 | 0.5 | 0.89 | 0.06 | 0.87 | 0.21 |
