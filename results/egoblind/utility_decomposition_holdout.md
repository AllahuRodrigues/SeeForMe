# P-CACS utility decomposition on EgoBlind holdout (77 videos)

21 coverable events were scored at least once inside their window; 21 never reached the speak threshold (0.92). Recomputed utility matched the policy's score at every step: yes.

Each counterfactual changes one factor on the *same* events, so the rows compare interventions on identical events. These are oracle what-ifs that set the factor to its most favourable value for every event (depth confidence 1, path probability 1, urgency 1, penalty or cost removed): upper bounds on how far the threshold binds, not what a perfect estimator of the true geometry would give, and not results a real estimator has achieved.

| if this factor were ideal | events that would cross the threshold | share of the silent events [95% Wilson] |
|---|---|---|
| no_uncertainty | 0 | 0.00 [0.00, 0.15] |
| depth_certain | 0 | 0.00 [0.00, 0.15] |
| on_path | 3 | 0.14 [0.05, 0.35] |
| imminent | 1 | 0.05 [0.01, 0.23] |
| on_path_imminent | 15 | 0.71 [0.50, 0.86] |
| free_speech | 0 | 0.00 [0.00, 0.15] |

Median feature values of the silent events at their best-scoring step:

| feature | median |
|---|---|
| value | 0.19 |
| novelty_factor | 1.00 |
| uncertainty_penalty | 0.47 |
| costs | 0.25 |
| path_prob | 0.44 |
| urgency | 0.00 |
| depth_conf | 0.55 |
| confidence | 0.63 |
| distance | 10.32 |
| ttc | 10.00 |
| severity | 0.50 |

Labels of the silent events: person 11, car 4, motorcycle 3, bicycle 2, dining table 1
