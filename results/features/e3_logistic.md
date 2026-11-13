Feature gate (e3_logistic): logistic fit on train (20774 rows, 3454 pos), scored on finetune (4479 rows, 959 pos)

| features | AUC train | AUC finetune | delta vs base [95% CI] | gate |
|---|---|---|---|---|
| base 29 | 0.728 | 0.675 | | |
| base + looming_track | 0.729 | 0.675 | +0.001 [-0.003, +0.004] | fail |
| base + looming_decision | 0.732 | 0.683 | +0.008 [-0.004, +0.021] | fail |
| base + segmentation | 0.757 | 0.655 | -0.019 [-0.059, +0.018] | fail |
| base + segmentation_architecture | 0.737 | 0.654 | -0.020 [-0.050, +0.000] | fail |
| base + egomotion | 0.758 | 0.694 | +0.020 [-0.024, +0.066] | fail |
| base + looming_all | 0.733 | 0.684 | +0.009 [-0.004, +0.023] | fail |
| base + context_all | 0.774 | 0.689 | +0.015 [-0.026, +0.057] | fail |

Single-feature AUC on finetune: has_loom 0.526, inv_loom 0.550, loom_lt3 0.530, dec_has_loom 0.524, dec_max_inv_loom 0.603, dec_n_loom_lt3 0.603, seg_stairs 0.529, seg_door 0.540, seg_railing 0.486, seg_walkable 0.550, seg_vehicle 0.586, seg_person 0.491, seg_obstacle 0.580, seg_stairs_max3s 0.564, seg_door_max3s 0.556, seg_railing_max3s 0.463, seg_walkable_max3s 0.550, seg_vehicle_max3s 0.578, seg_person_max3s 0.474, seg_obstacle_max3s 0.563, ego_yaw_abs 0.527, ego_still 0.421, ego_parallax 0.496, ego_inlier 0.429, ego_reliable 0.465, ego_yaw_abs_max3s 0.598, ego_still_max3s 0.370, ego_parallax_max3s 0.523, ego_inlier_max3s 0.388, ego_yaw_abs_mean3s 0.598, ego_still_mean3s 0.361, ego_parallax_mean3s 0.504, ego_inlier_mean3s 0.414, ego_reliable_frac3s 0.433
