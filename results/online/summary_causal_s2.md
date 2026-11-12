Online stream ['train', 'finetune', 'select']: 592 videos, 333.6 min, p_tap=0.5, target 2.0/min

| policy | coverable recall | quiet violations | cues/min | coverable events |
|---|---|---|---|---|
| online_greedy | 0.216 | 0.104 | 1.67 | 171 |
| online_greedy+lr0 | 0.135 | 0.125 | 1.68 | 171 |
| online_full_info | 0.515 | 0.091 | 1.67 | 171 |
| online_full_info+causal | 0.187 | 0.114 | 1.68 | 171 |
| online_full_info+d2 | 0.392 | 0.101 | 1.67 | 171 |
