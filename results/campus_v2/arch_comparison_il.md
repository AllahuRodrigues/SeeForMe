Checkpoint: il, split: test

| model | encoder | params | action-weighted recall | late critical | precision | occupancy | silence precision | regret |
|---|---|---|---|---|---|---|---|---|
| mlp1 | mlp1 | 14146 | 0.611 | 0.21 | 0.704 | 0.23 | 0.95 | 0.72 |
| mlpk | mlpk | 13872 | 0.609 | 0.22 | 0.724 | 0.23 | 0.94 | 0.67 |
| attn | attn | 13992 | 0.608 | 0.19 | 0.707 | 0.23 | 0.95 | 0.71 |
| gru | gru | 14268 | 0.620 | 0.23 | 0.766 | 0.22 | 0.94 | 0.53 |
| gru_mem | gru + memory ledger | 14268 | 0.514 | 0.15 | 0.438 | 0.31 | 0.96 | 1.98 |

Action-weighted recall per scene kind:

| kind | mlp1 | mlpk | attn | gru | gru_mem |
|---|---|---|---|---|---|
| changing | 0.42 | 0.38 | 0.38 | 0.47 | 0.28 |
| contention | 0.28 | 0.24 | 0.62 | 0.27 | 0.57 |
| courtyard | 0.38 | 0.40 | 0.32 | 0.37 | 0.22 |
| crossing | 0.16 | 0.29 | 0.22 | 0.22 | 0.11 |
| crowd | 0.62 | 0.58 | 0.72 | 0.66 | 0.33 |
| doors | 0.92 | 1.00 | 0.92 | 0.96 | 0.75 |
| finding | 0.56 | 0.58 | 0.53 | 0.59 | 0.64 |
| hallway | 0.91 | 0.77 | 0.78 | 0.88 | 0.52 |
| negative | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| occlusion | 0.58 | 0.58 | 0.58 | 0.49 | 0.26 |
| ood_aftermath | 0.34 | 0.37 | 0.32 | 0.34 | 0.36 |
| poor_capture | 0.62 | 0.65 | 0.54 | 0.57 | 0.61 |
| stairs | 0.93 | 0.93 | 0.93 | 0.97 | 0.84 |
| temporary_hazard | 0.77 | 0.71 | 0.61 | 0.77 | 0.54 |
| wayfinding_airport | 0.66 | 0.66 | 0.66 | 0.74 | 0.68 |
