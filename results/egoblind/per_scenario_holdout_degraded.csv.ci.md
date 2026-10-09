95% bootstrap intervals over 77 videos (2000 resamples, seed 0) for `per_scenario_holdout_degraded.csv`

| policy | recall (all) | recall (coverable) | precision | quiet-window violations | coverable recall minus pcacs |
|---|---|---|---|---|---|
| oracle | 0.46 [0.31, 0.60] | 0.67 [0.47, 0.83] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.63 [0.43, 0.83] |
| confidence_threshold | 0.17 [0.08, 0.30] | 0.23 [0.08, 0.43] | 0.01 [0.00, 0.01] | 1.00 [1.00, 1.00] | 0.20 [0.05, 0.41] |
| confidence_cooldown | 0.32 [0.20, 0.44] | 0.47 [0.28, 0.62] | 0.03 [0.01, 0.04] | 0.89 [0.71, 1.00] | 0.43 [0.26, 0.61] |
| cacs | 0.14 [0.06, 0.26] | 0.23 [0.08, 0.47] | 0.02 [0.01, 0.04] | 0.73 [0.54, 0.90] | 0.20 [0.03, 0.47] |
| pcacs | 0.02 [0.00, 0.04] | 0.03 [0.00, 0.09] | 0.12 [0.00, 0.50] | 0.00 [0.00, 0.00] | - |
| pcacs_egoblind | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | -0.03 [-0.09, 0.00] |
| pcacs_learned:il | 0.06 [0.00, 0.15] | 0.07 [0.00, 0.18] | 0.02 [0.00, 0.04] | 0.50 [0.28, 0.71] | 0.03 [-0.07, 0.17] |
| pcacs_learned:rl | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.21] | -0.03 [-0.09, 0.00] |
| egoblind_learned:il | 0.02 [0.00, 0.05] | 0.03 [0.00, 0.10] | 0.05 [0.00, 0.10] | 0.04 [0.00, 0.14] | 0.00 [-0.08, 0.10] |
| egoblind_learned:rl | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | -0.03 [-0.09, 0.00] |
| egoblind_learned:il@models/egoblind_only | 0.02 [0.00, 0.04] | 0.03 [0.00, 0.09] | 0.01 [0.00, 0.03] | 0.08 [0.00, 0.20] | 0.00 [0.00, 0.00] |
| egoblind_learned:rl@models/egoblind_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | -0.03 [-0.09, 0.00] |

Paired differences (a minus b, same resampled videos)

| a minus b | recall (all) | recall (coverable) | quiet-window violations |
|---|---|---|---|
| pcacs_learned:il - egoblind_learned:il@models/egoblind_only | 0.05 [-0.03, 0.14] | 0.03 [-0.07, 0.17] | 0.42 [0.14, 0.67] |
| egoblind_learned:il@models/egoblind_only - pcacs_learned:rl | 0.02 [0.00, 0.04] | 0.03 [0.00, 0.09] | 0.00 [-0.15, 0.15] |
| egoblind_learned:il - pcacs_learned:rl | 0.02 [0.00, 0.05] | 0.03 [0.00, 0.10] | -0.04 [-0.18, 0.09] |
| egoblind_learned:il@models/egoblind_only - egoblind_learned:il | 0.00 [-0.05, 0.04] | 0.00 [-0.10, 0.08] | 0.04 [-0.10, 0.18] |
| pcacs_egoblind - pcacs | -0.02 [-0.04, 0.00] | -0.03 [-0.09, 0.00] | 0.00 [0.00, 0.00] |
