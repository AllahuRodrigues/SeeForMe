Re-ID memory on ['train', 'finetune']: 558 videos (558 with embeddings)

| policy | coverable recall | quiet viol. | cues/min | repeat (gt share) | repeat (same track share) | label repeats/min | re-IDs/min | re-IDs of announced objects | match precision (gt) |
|---|---|---|---|---|---|---|---|---|---|
| pcacs@reid_off | 0.087 | 0.022 | 0.57 | 0.048 | 0.018 | 0.214 | 0.00 | 0 |  |
| pcacs@reid_h120_t0.85 | 0.087 | 0.022 | 0.55 | 0.044 | 0.013 | 0.196 | 84.74 | 127 | 0.89 (n=1863) |

Paired differences, on minus off (95% bootstrap over videos):

| variant | Δ recall | Δ label repeats/min | Δ cues/min | Δ quiet viol. |
|---|---|---|---|---|
| pcacs@reid_h120_t0.85 | +0.000 [+0.000, +0.000] | -0.017 [-0.047, +0.000] | -0.017 [-0.047, +0.000] | +0.000 [+0.000, +0.000] |
