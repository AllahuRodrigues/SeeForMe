95% bootstrap intervals over 656 videos (2000 resamples, seed 0) for `per_scenario_test_official_degraded.csv`

| policy | recall (all) | recall (coverable) | precision | quiet-window violations | coverable recall minus pcacs |
|---|---|---|---|---|---|
| oracle | 0.45 [0.39, 0.51] | 0.74 [0.68, 0.81] | 1.00 [1.00, 1.00] | 0.02 [0.01, 0.04] | 0.69 [0.62, 0.76] |
| confidence_threshold | 0.27 [0.22, 0.32] | 0.47 [0.40, 0.53] | 0.01 [0.01, 0.01] | 0.87 [0.82, 0.91] | 0.41 [0.34, 0.47] |
| confidence_cooldown | 0.31 [0.27, 0.36] | 0.55 [0.49, 0.61] | 0.02 [0.02, 0.03] | 0.81 [0.77, 0.86] | 0.49 [0.42, 0.56] |
| cacs | 0.23 [0.18, 0.28] | 0.39 [0.32, 0.46] | 0.03 [0.02, 0.04] | 0.55 [0.48, 0.61] | 0.34 [0.26, 0.41] |
| pcacs | 0.03 [0.01, 0.05] | 0.06 [0.02, 0.09] | 0.06 [0.02, 0.09] | 0.03 [0.01, 0.06] | - |
| pcacs_egoblind | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | 0.00 [0.00, 0.00] | -0.06 [-0.09, -0.02] |
| pcacs_learned:il | 0.12 [0.08, 0.16] | 0.22 [0.16, 0.28] | 0.03 [0.02, 0.04] | 0.27 [0.21, 0.33] | 0.17 [0.11, 0.23] |
| pcacs_learned:rl | 0.06 [0.03, 0.08] | 0.10 [0.06, 0.15] | 0.04 [0.02, 0.06] | 0.08 [0.05, 0.12] | 0.05 [0.00, 0.10] |
| egoblind_learned:il | 0.05 [0.02, 0.07] | 0.09 [0.05, 0.13] | 0.07 [0.03, 0.10] | 0.03 [0.01, 0.06] | 0.03 [-0.01, 0.08] |
| egoblind_learned:rl | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | -0.06 [-0.09, -0.02] |
| egoblind_learned:il@models/egoblind_only | 0.08 [0.05, 0.11] | 0.15 [0.10, 0.20] | 0.06 [0.04, 0.08] | 0.08 [0.05, 0.11] | 0.10 [0.04, 0.15] |
| egoblind_learned:rl@models/egoblind_only | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | -0.06 [-0.09, -0.02] |

Paired differences (a minus b, same resampled videos)

| a minus b | recall (all) | recall (coverable) | quiet-window violations |
|---|---|---|---|
| pcacs_learned:il - egoblind_learned:il@models/egoblind_only | 0.04 [0.00, 0.08] | 0.07 [0.00, 0.14] | 0.18 [0.12, 0.25] |
| egoblind_learned:il@models/egoblind_only - pcacs_learned:rl | 0.03 [-0.01, 0.06] | 0.04 [-0.02, 0.12] | -0.00 [-0.05, 0.04] |
| egoblind_learned:il - pcacs_learned:rl | -0.01 [-0.04, 0.02] | -0.02 [-0.08, 0.04] | -0.05 [-0.09, -0.01] |
| egoblind_learned:il@models/egoblind_only - egoblind_learned:il | 0.04 [0.01, 0.07] | 0.07 [0.01, 0.12] | 0.05 [0.01, 0.08] |
| pcacs_egoblind - pcacs | -0.03 [-0.05, -0.01] | -0.06 [-0.09, -0.02] | -0.03 [-0.06, -0.01] |
