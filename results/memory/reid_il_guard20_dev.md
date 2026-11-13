Re-ID memory on ['train', 'finetune']: 558 videos (558 with embeddings)

| policy | coverable recall | quiet viol. | cues/min | repeat (gt share) | repeat (same track share) | label repeats/min | re-IDs/min | re-IDs of announced objects | match precision (gt) |
|---|---|---|---|---|---|---|---|---|---|
| egoblind_learned:il+guard20@reid_off | 0.331 | 0.036 | 0.73 | 0.258 | 0.075 | 0.286 | 0.00 | 0 |  |

Paired differences, on minus off (95% bootstrap over videos):

| variant | Δ recall | Δ label repeats/min | Δ cues/min | Δ quiet viol. |
|---|---|---|---|---|
