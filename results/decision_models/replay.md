# Replay at equal speech rate on `finetune` (104 videos, seeds 0, 1, 2 pooled)

Frozen scorers speak the top legal candidate when its score is at least the BudgetController's tau (target 2 cues/min), exactly like online greedy. Every policy streams `train` then `finetune` with the same seeds; only the `finetune` videos are reported (the `train` part lets the controller settle and is in-sample for the frozen models). online_greedy learns online from simulated taps and replayed questions, as in docs/online.md. Seeds only change the simulated feedback, so frozen scorers give the same cues on every seed. Model choice (K, weight decay, epochs, Platt) was made on these same finetune videos.

| policy | coverable recall | all-event recall | all-event recall vs logistic [95% CI] | quiet violations | cues/min | speech s/min |
|---|---|---|---|---|---|---|
| logistic | 27/81 = 0.333 | 27/168 = 0.161 | reference | 0.158 | 1.72 | 3.8 |
| logistic_cw_platt | 27/81 = 0.333 | 27/168 = 0.161 | +0.000 [-0.053, +0.046] | 0.132 | 1.72 | 3.9 |
| mlp | 27/81 = 0.333 | 27/168 = 0.161 | +0.000 [-0.079, +0.067] | 0.105 | 1.70 | 3.4 |
| gru_k8 | 33/81 = 0.407 | 33/168 = 0.196 | +0.036 [-0.031, +0.094] | 0.053 | 1.70 | 3.6 |
| tcn_k2 | 27/81 = 0.333 | 27/168 = 0.161 | +0.000 [-0.079, +0.067] | 0.184 | 1.69 | 3.5 |
| online_greedy | 13/81 = 0.160 | 13/168 = 0.077 | -0.083 [-0.172, -0.011] | 0.132 | 1.72 | 4.6 |

Paired bootstrap: 2,000 resamples of finetune videos, each resample used for all three seeds of both policies.
Per-seed all-event recall: logistic 0.161 / 0.161 / 0.161; logistic_cw_platt 0.161 / 0.161 / 0.161; mlp 0.161 / 0.161 / 0.161; gru_k8 0.196 / 0.196 / 0.196; tcn_k2 0.161 / 0.161 / 0.161; online_greedy 0.054 / 0.071 / 0.107
