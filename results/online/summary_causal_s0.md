Online stream ['train', 'finetune', 'select']: 592 videos, 333.6 min, p_tap=0.5, target 2.0/min

| policy | coverable recall | quiet violations | cues/min | coverable events |
|---|---|---|---|---|
| online_greedy | 0.199 | 0.118 | 1.68 | 171 |
| online_greedy+lr0 | 0.135 | 0.125 | 1.68 | 171 |
| online_full_info | 0.520 | 0.091 | 1.66 | 171 |
| online_full_info+causal | 0.193 | 0.111 | 1.68 | 171 |
| online_full_info+d2 | 0.368 | 0.081 | 1.67 | 171 |
