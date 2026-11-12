Re-ID memory on ['train', 'finetune']: 558 videos (558 with embeddings)

| policy | coverable recall | quiet viol. | cues/min | repeat (gt share) | repeat (same track share) | label repeats/min | re-IDs/min | re-IDs of announced objects | match precision (gt) |
|---|---|---|---|---|---|---|---|---|---|
| egoblind_learned:il@reid_off | 0.331 | 0.036 | 0.79 | 0.285 | 0.136 | 0.338 | 0.00 | 0 |  |
| egoblind_learned:il+guard10@reid_off | 0.331 | 0.036 | 0.74 | 0.262 | 0.079 | 0.290 | 0.00 | 0 |  |

Paired differences, on minus off (95% bootstrap over videos):

| variant | Δ recall | Δ label repeats/min | Δ cues/min | Δ quiet viol. |
|---|---|---|---|---|
