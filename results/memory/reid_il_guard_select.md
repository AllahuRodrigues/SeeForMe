Re-ID memory on ['select']: 34 videos (34 with embeddings)

| policy | coverable recall | quiet viol. | cues/min | repeat (gt share) | repeat (same track share) | label repeats/min | re-IDs/min | re-IDs of announced objects | match precision (gt) |
|---|---|---|---|---|---|---|---|---|---|
| egoblind_learned:il@reid_off | 0.091 | 0.000 | 0.71 | 0.032 | 0.194 | 0.345 | 0.00 | 0 |  |
| egoblind_learned:il+guard5@reid_off | 0.091 | 0.000 | 0.67 | 0.000 | 0.138 | 0.299 | 0.00 | 0 |  |
| egoblind_learned:il+guard10@reid_off | 0.091 | 0.000 | 0.67 | 0.000 | 0.138 | 0.299 | 0.00 | 0 |  |
| egoblind_learned:il+guard20@reid_off | 0.091 | 0.000 | 0.64 | 0.000 | 0.107 | 0.276 | 0.00 | 0 |  |

Paired differences, on minus off (95% bootstrap over videos):

| variant | Δ recall | Δ label repeats/min | Δ cues/min | Δ quiet viol. |
|---|---|---|---|---|
