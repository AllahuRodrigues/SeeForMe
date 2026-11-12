Feature gate (looming): logistic fit on train (20774 rows, 3454 pos), scored on finetune (4479 rows, 959 pos)

| features | AUC train | AUC finetune | delta vs base [95% CI] | gate |
|---|---|---|---|---|
| base 29 | 0.728 | 0.675 | | |
| base + looming_track | 0.729 | 0.675 | +0.001 [-0.003, +0.004] | fail |
| base + looming_decision | 0.732 | 0.683 | +0.008 [-0.004, +0.021] | fail |
| base + looming_all | 0.733 | 0.684 | +0.009 [-0.004, +0.023] | fail |

Single-feature AUC on finetune: has_loom 0.526, inv_loom 0.550, loom_lt3 0.530, dec_has_loom 0.524, dec_max_inv_loom 0.603, dec_n_loom_lt3 0.603
