95% bootstrap intervals over 656 videos (2000 resamples, seed 0) for `per_scenario_test_official.csv`

| policy | recall (all) | recall (coverable) | precision | quiet-window violations | coverable recall minus pcacs |
|---|---|---|---|---|---|
| oracle | 0.47 [0.42, 0.53] | 0.78 [0.72, 0.84] | 1.00 [1.00, 1.00] | 0.02 [0.01, 0.04] | 0.71 [0.64, 0.78] |
| confidence_threshold | 0.26 [0.21, 0.31] | 0.45 [0.38, 0.52] | 0.01 [0.01, 0.01] | 0.89 [0.85, 0.93] | 0.38 [0.31, 0.45] |
| confidence_cooldown | 0.34 [0.29, 0.39] | 0.57 [0.50, 0.64] | 0.02 [0.02, 0.03] | 0.83 [0.78, 0.88] | 0.51 [0.43, 0.58] |
| cacs | 0.25 [0.20, 0.30] | 0.42 [0.35, 0.49] | 0.03 [0.02, 0.04] | 0.62 [0.56, 0.67] | 0.35 [0.28, 0.43] |
| pcacs | 0.04 [0.02, 0.06] | 0.07 [0.04, 0.11] | 0.06 [0.03, 0.10] | 0.04 [0.02, 0.06] | - |
| pcacs_egoblind | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | -0.07 [-0.11, -0.04] |
| pcacs_learned:il | 0.11 [0.08, 0.15] | 0.20 [0.15, 0.26] | 0.03 [0.02, 0.04] | 0.25 [0.20, 0.31] | 0.14 [0.07, 0.20] |
| pcacs_learned:rl | 0.06 [0.03, 0.08] | 0.10 [0.06, 0.15] | 0.04 [0.03, 0.07] | 0.06 [0.04, 0.09] | 0.04 [-0.02, 0.09] |
| egoblind_learned:il | 0.06 [0.04, 0.09] | 0.12 [0.07, 0.16] | 0.07 [0.05, 0.10] | 0.05 [0.03, 0.08] | 0.04 [-0.01, 0.10] |
| egoblind_learned:rl | 0.00 [0.00, 0.01] | 0.01 [0.00, 0.01] | 0.05 [0.00, 0.14] | 0.00 [0.00, 0.00] | -0.07 [-0.11, -0.03] |
| egoblind_learned:il@models/egoblind_only | 0.11 [0.08, 0.14] | 0.20 [0.15, 0.25] | 0.06 [0.04, 0.08] | 0.13 [0.09, 0.17] | 0.13 [0.06, 0.20] |
| egoblind_learned:rl@models/egoblind_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | -0.07 [-0.11, -0.04] |

Paired differences (a minus b, same resampled videos)

| a minus b | recall (all) | recall (coverable) | quiet-window violations |
|---|---|---|---|
| pcacs_learned:il - egoblind_learned:il@models/egoblind_only | 0.00 [-0.04, 0.05] | 0.01 [-0.07, 0.09] | 0.12 [0.06, 0.19] |
| egoblind_learned:il@models/egoblind_only - pcacs_learned:rl | 0.05 [0.01, 0.09] | 0.10 [0.03, 0.16] | 0.07 [0.02, 0.11] |
| egoblind_learned:il - pcacs_learned:rl | 0.01 [-0.03, 0.04] | 0.01 [-0.05, 0.07] | -0.01 [-0.04, 0.03] |
| egoblind_learned:il@models/egoblind_only - egoblind_learned:il | 0.05 [0.01, 0.08] | 0.09 [0.03, 0.14] | 0.07 [0.03, 0.11] |
| pcacs_egoblind - pcacs | -0.04 [-0.06, -0.02] | -0.07 [-0.11, -0.04] | -0.04 [-0.06, -0.02] |
