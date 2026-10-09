# P-CACS utility decomposition on EgoBlind train (454 videos)

117 coverable events were scored at least once inside their window; 101 never reached the speak threshold (0.92). Recomputed utility matched the policy's score at every step: yes.

Each counterfactual changes one factor on the *same* events, so the rows compare interventions on identical events. These are oracle what-ifs that set the factor to its most favourable value for every event (depth confidence 1, path probability 1, urgency 1, penalty or cost removed): upper bounds on how far the threshold binds, not what a perfect estimator of the true geometry would give, and not results a real estimator has achieved.

| if this factor were ideal | events that would cross the threshold | share of the silent events [95% Wilson] |
|---|---|---|
| no_uncertainty | 10 | 0.10 [0.06, 0.17] |
| depth_certain | 0 | 0.00 [0.00, 0.04] |
| on_path | 22 | 0.22 [0.15, 0.31] |
| imminent | 16 | 0.16 [0.10, 0.24] |
| on_path_imminent | 73 | 0.72 [0.63, 0.80] |
| free_speech | 19 | 0.19 [0.12, 0.28] |

Median feature values of the silent events at their best-scoring step:

| feature | median |
|---|---|
| value | 0.69 |
| novelty_factor | 1.00 |
| uncertainty_penalty | 0.43 |
| costs | 0.34 |
| path_prob | 0.64 |
| urgency | 0.68 |
| depth_conf | 0.68 |
| confidence | 0.68 |
| distance | 6.68 |
| ttc | 6.08 |
| severity | 0.50 |

Labels of the silent events: person 51, car 21, motorcycle 14, dining table 5, bicycle 4, chair 4, backpack 2
