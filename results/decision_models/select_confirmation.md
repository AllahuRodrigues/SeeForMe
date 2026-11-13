One-shot confirmation on `select` (34 videos; seeds 0,1,2 pooled; frozen models give identical cues on every seed)

| policy | all-event recall | vs logistic [95% CI] | coverable recall | quiet violations | vs logistic [95% CI] | cues/min | speech s/min |
|---|---|---|---|---|---|---|---|
| logistic | 9/81 = 0.111 | reference | 0.273 | 0.056 | reference | 1.68 | 4.0 |
| gru_k8 | 6/81 = 0.074 | -0.037 [-0.130, +0.000] | 0.182 | 0.000 | -0.056 [-0.214, +0.000] | 1.70 | 3.8 |
| online_greedy | 9/81 = 0.111 | +0.000 [-0.042, +0.032] | 0.273 | 0.148 | +0.093 [+0.000, +0.182] | 1.68 | 3.3 |
