# P-CACS utility decomposition on EgoBlind test (656 videos)

164 coverable events were scored at least once inside their window; 150 never reached the speak threshold (0.92). Recomputed utility matched the policy's score at every step: yes.

Each counterfactual changes one factor on the *same* events, so the rows compare interventions on identical events. These are oracle what-ifs that set the factor to its most favourable value for every event (depth confidence 1, path probability 1, urgency 1, penalty or cost removed): upper bounds on how far the threshold binds, not what a perfect estimator of the true geometry would give, and not results a real estimator has achieved.

| if this factor were ideal | events that would cross the threshold | share of the silent events [95% Wilson] |
|---|---|---|
| no_uncertainty | 17 | 0.11 [0.07, 0.17] |
| depth_certain | 2 | 0.01 [0.00, 0.05] |
| on_path | 29 | 0.19 [0.14, 0.26] |
| imminent | 14 | 0.09 [0.06, 0.15] |
| on_path_imminent | 95 | 0.63 [0.55, 0.71] |
| free_speech | 31 | 0.21 [0.15, 0.28] |

Median feature values of the silent events at their best-scoring step:

| feature | median |
|---|---|
| value | 0.84 |
| novelty_factor | 1.00 |
| uncertainty_penalty | 0.40 |
| costs | 0.34 |
| path_prob | 0.66 |
| urgency | 0.81 |
| depth_conf | 0.68 |
| confidence | 0.72 |
| distance | 4.77 |
| ttc | 3.14 |
| severity | 0.50 |

Labels of the silent events: person 94, car 26, motorcycle 12, bicycle 8, dining table 3, chair 3, suitcase 1, backpack 1, potted plant 1, fire hydrant 1
