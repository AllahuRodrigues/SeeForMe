# Offline comparison: fitted on `train`, selected and scored on `finetune`

Rows: train 20774 (3454 positive), finetune 4479 (959 positive, base rate 0.214). Finetune chose every hyperparameter (logistic C, network weight decay, early-stopping epoch, K, Platt calibration), so these finetune numbers are a development comparison, not a held-out estimate. One training seed per network.

| model | K (s) | AUC finetune | AUC delta vs logistic [95% CI] | AP | Brier | ECE | AUC train | params | ms / decision |
|---|---|---|---|---|---|---|---|---|---|
| logistic | 1 (0.5) | 0.682 |  | 0.349 | 0.1576 | 0.0435 | 0.713 | 30 | 0.021 |
| logistic_cw_raw | 1 (0.5) | 0.680 |  | 0.347 | 0.2507 | 0.2962 | 0.714 | 30 | 0.045 |
| logistic_cw_platt | 1 (0.5) | 0.680 | -0.002 [-0.006, +0.004] | 0.347 | 0.1575 (0.2033 cf) | 0.0344 (0.2082 cf) | 0.714 | 32 | 0.021 |
| mlp | 1 (0.5) | 0.674 | -0.007 [-0.047, +0.032] | 0.313 | 0.1589 | 0.0325 | 0.745 | 6145 | 0.083 |
| gru_k2 | 2 (1) | 0.685 | +0.003 [-0.018, +0.023] | 0.327 | 0.1587 | 0.0368 | 0.730 | 6177 | 0.517 |
| gru_k4 | 4 (2) | 0.694 | +0.012 [-0.014, +0.039] | 0.335 | 0.1583 | 0.0468 | 0.727 | 6177 | 1.072 |
| gru_k8 | 8 (4) | 0.697 | +0.015 [-0.015, +0.046] | 0.339 | 0.1577 | 0.0455 | 0.732 | 6177 | 0.512 |
| gru_k16 | 16 (8) | 0.697 | +0.015 [-0.014, +0.043] | 0.341 | 0.1578 | 0.0448 | 0.732 | 6177 | 1.507 |
| tcn_k2 | 2 (1) | 0.669 | -0.012 [-0.054, +0.026] | 0.313 | 0.1591 | 0.0404 | 0.752 | 9153 | 1.772 |
| tcn_k4 | 4 (2) | 0.666 | -0.016 [-0.066, +0.032] | 0.308 | 0.1595 | 0.0413 | 0.758 | 9153 | 1.669 |
| tcn_k8 | 8 (4) | 0.665 | -0.017 [-0.065, +0.029] | 0.306 | 0.1599 | 0.0339 | 0.766 | 9153 | 1.071 |
| tcn_k16 | 16 (8) | 0.668 | -0.013 [-0.062, +0.034] | 0.314 | 0.1591 | 0.0319 | 0.766 | 9153 | 2.644 |

Best K by finetune AUC: GRU gru_k8, TCN tcn_k2. ECE: 15 equal-width bins. 'cf' = Platt cross-fitted over two halves of the finetune videos. ms / decision: all confirmed candidates of one real finetune decision scored in one batch, mean over 500 decisions, 2 CPU threads.
logistic_cw_raw is the class-weighted fit before calibration (its probabilities are inflated by the weighting).
