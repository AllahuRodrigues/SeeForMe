95% bootstrap intervals over 77 videos (2000 resamples, seed 0) for `per_scenario_holdout_final.csv`

| policy | recall (all) | recall (coverable) | precision | quiet-window violations | coverable recall minus pcacs |
|---|---|---|---|---|---|
| oracle | 0.48 [0.32, 0.62] | 0.67 [0.47, 0.83] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.67 [0.47, 0.83] |
| confidence_threshold | 0.19 [0.09, 0.30] | 0.27 [0.11, 0.43] | 0.01 [0.00, 0.01] | 1.00 [1.00, 1.00] | 0.27 [0.11, 0.43] |
| confidence_cooldown | 0.33 [0.21, 0.46] | 0.53 [0.34, 0.71] | 0.02 [0.01, 0.04] | 1.00 [1.00, 1.00] | 0.53 [0.34, 0.71] |
| cacs | 0.19 [0.09, 0.29] | 0.33 [0.18, 0.52] | 0.03 [0.01, 0.04] | 0.85 [0.70, 0.96] | 0.33 [0.18, 0.52] |
| pcacs | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| pcacs_egoblind | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| pcacs_learned:il | 0.03 [0.00, 0.08] | 0.03 [0.00, 0.12] | 0.01 [0.00, 0.03] | 0.42 [0.20, 0.64] | 0.03 [0.00, 0.12] |
| pcacs_learned:rl | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.04 [0.00, 0.14] | 0.00 [0.00, 0.00] |
| egoblind_learned:il | 0.06 [0.00, 0.13] | 0.13 [0.00, 0.25] | 0.09 [0.00, 0.14] | 0.12 [0.00, 0.26] | 0.13 [0.00, 0.25] |
| egoblind_learned:rl | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| egoblind_learned:il@models/egoblind_only | 0.08 [0.01, 0.16] | 0.17 [0.03, 0.33] | 0.06 [0.01, 0.13] | 0.12 [0.00, 0.25] | 0.17 [0.03, 0.33] |
| egoblind_learned:rl@models/egoblind_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

Paired differences (a minus b, same resampled videos)

| a minus b | recall (all) | recall (coverable) | quiet-window violations |
|---|---|---|---|
| pcacs_learned:il - egoblind_learned:il@models/egoblind_only | -0.05 [-0.15, 0.04] | -0.13 [-0.31, 0.04] | 0.31 [0.00, 0.58] |
| egoblind_learned:il@models/egoblind_only - pcacs_learned:rl | 0.08 [0.01, 0.16] | 0.17 [0.03, 0.33] | 0.08 [-0.07, 0.23] |
| egoblind_learned:il - pcacs_learned:rl | 0.06 [0.00, 0.13] | 0.13 [0.00, 0.25] | 0.08 [-0.07, 0.25] |
| egoblind_learned:il@models/egoblind_only - egoblind_learned:il | 0.02 [-0.05, 0.12] | 0.03 [-0.11, 0.24] | 0.00 [-0.12, 0.10] |
| pcacs_egoblind - pcacs | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
