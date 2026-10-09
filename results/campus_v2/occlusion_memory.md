### 5.2 The memory ledger (frame-only -> memory TTL 6 s, same v1-trained policies)

`SEEFORME_CAMPUS=v2 python scripts/run_campus.py --split test --repeats 1 --policies oracle,pcacs,pcacs_learned:il,pcacs_learned:rl --memory-ttl 6 --tag mem`.
Cells: action-weighted recall / late critical rate / cue precision, frame-only -> with the ledger.

| policy | occlusion | poor_capture (blackouts) | crowd | contention | all 15 kinds |
|---|---|---|---|---|---|
| reference | 0.49 -> 1.00 / 0.50 -> 0.00 / 1.00 -> 1.00 | 0.56 -> 0.80 / 0.56 -> 0.22 / 1.00 -> 1.00 | 0.68 -> 0.60 / 0.00 -> 0.00 / 1.00 -> 1.00 | 0.62 -> 0.66 / 0.39 -> 0.33 / 1.00 -> 1.00 | 0.68 -> 0.75 / 0.25 -> 0.19 / 0.99 -> 0.99 |
| P-CACS linear | 0.50 -> 0.58 / 0.50 -> 0.42 / 0.75 -> 0.92 | 0.22 -> 0.30 / 0.78 -> 0.61 / 0.36 -> 0.41 | 0.00 -> 0.04 / 0.00 -> 0.00 / 0.17 -> 0.08 | 0.51 -> 0.51 / 0.39 -> 0.39 / 0.67 -> 0.67 | 0.46 -> 0.47 / 0.27 -> 0.27 / 0.60 -> 0.58 |
| GRU imitation | 0.49 -> 0.83 / 0.50 -> 0.17 / 0.75 -> 0.89 | 0.51 -> 0.56 / 0.33 -> 0.11 / 0.51 -> 0.43 | 0.64 -> 0.58 / 0.00 -> 0.00 / 0.57 -> 0.72 | 0.26 -> 0.24 / 0.78 -> 0.78 / 0.94 -> 0.75 | 0.62 -> 0.66 / 0.23 -> 0.18 / 0.74 -> 0.72 |
| GRU + RL | 0.49 -> 0.83 / 0.50 -> 0.17 / 0.92 -> 0.94 | 0.61 -> 0.60 / 0.28 -> 0.17 / 0.77 -> 0.61 | 0.72 -> 0.48 / 0.00 -> 0.00 / 0.86 -> 0.81 | 0.24 -> 0.21 / 0.78 -> 0.72 / 0.94 -> 0.78 | 0.63 -> 0.62 / 0.24 -> 0.20 / 0.89 -> 0.83 |

