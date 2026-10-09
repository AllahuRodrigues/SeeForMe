95% bootstrap intervals over 77 videos (2000 resamples, seed 0) for `per_scenario_holdout_ow.csv`

| policy | recall (all) | recall (coverable) | precision | quiet-window violations | coverable recall minus pcacs |
|---|---|---|---|---|---|
| oracle | 0.32 [0.17, 0.45] | 0.43 [0.26, 0.59] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.43 [0.26, 0.59] |
| confidence_cooldown | 0.21 [0.11, 0.30] | 0.28 [0.16, 0.40] | 0.02 [0.01, 0.03] | 0.77 [0.55, 0.97] | 0.28 [0.16, 0.40] |
| cacs | 0.11 [0.04, 0.18] | 0.15 [0.06, 0.25] | 0.02 [0.01, 0.03] | 0.81 [0.64, 0.94] | 0.15 [0.06, 0.25] |
| pcacs | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - |
| pcacs_egoblind | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| pcacs_learned:il | 0.08 [0.01, 0.17] | 0.11 [0.02, 0.24] | 0.03 [0.01, 0.05] | 0.23 [0.08, 0.42] | 0.11 [0.02, 0.24] |
| pcacs_learned:rl | 0.05 [0.00, 0.11] | 0.07 [0.00, 0.15] | 0.04 [0.00, 0.10] | 0.04 [0.00, 0.14] | 0.07 [0.00, 0.15] |
| egoblind_learned:il@models/egoblind_only | 0.10 [0.02, 0.18] | 0.13 [0.03, 0.24] | 0.08 [0.02, 0.14] | 0.00 [0.00, 0.00] | 0.13 [0.03, 0.24] |

Paired differences (a minus b, same resampled videos)

| a minus b | recall (all) | recall (coverable) | quiet-window violations |
|---|---|---|---|
| pcacs_learned:il - egoblind_learned:il@models/egoblind_only | -0.02 [-0.12, 0.10] | -0.02 [-0.17, 0.14] | 0.23 [0.08, 0.42] |
| pcacs_egoblind - pcacs | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
