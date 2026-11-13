Feature gate (egomotion): logistic fit on train (20774 rows, 3454 pos), scored on finetune (4479 rows, 959 pos)

| features | AUC train | AUC finetune | delta vs base [95% CI] | gate |
|---|---|---|---|---|
| base 29 | 0.728 | 0.675 | | |
| base + looming_track | 0.729 | 0.675 | +0.001 [-0.003, +0.004] | fail |
| base + looming_decision | 0.732 | 0.683 | +0.008 [-0.004, +0.021] | fail |
| base + egomotion | 0.758 | 0.694 | +0.020 [-0.024, +0.066] | fail |
| base + looming_all | 0.733 | 0.684 | +0.009 [-0.004, +0.023] | fail |

Single-feature AUC on finetune: has_loom 0.526, inv_loom 0.550, loom_lt3 0.530, dec_has_loom 0.524, dec_max_inv_loom 0.603, dec_n_loom_lt3 0.603, ego_yaw_abs 0.527, ego_still 0.421, ego_parallax 0.496, ego_inlier 0.429, ego_reliable 0.465, ego_yaw_abs_max3s 0.598, ego_still_max3s 0.370, ego_parallax_max3s 0.523, ego_inlier_max3s 0.388, ego_yaw_abs_mean3s 0.598, ego_still_mean3s 0.361, ego_parallax_mean3s 0.504, ego_inlier_mean3s 0.414, ego_reliable_frac3s 0.433
