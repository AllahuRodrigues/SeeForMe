E3-V at policy level, on `via_dev` (48 held-out videos, first 100 s each; scorer fitted on 152 `via_train` videos). A reminder is answered by a cue about an object it names inside [start - 1 s, end + 0.5 s].

| policy | reminders answered | of those the detector could see | benchmark's strict rule | cues with no reminder within 3 s, per min | cues/min | speech s/min |
|---|---|---|---|---|---|---|
| threshold, VIABench scorer | 40/455 = 0.088 | 40/179 = 0.223 | 0.068 | 0.27 (16% of cues) | 1.66 | 3.95 |
| threshold, EgoBlind scorer | 16/455 = 0.035 | 16/179 = 0.089 | 0.020 | 0.51 (31% of cues) | 1.67 | 3.41 |
| coverage-first, equal speech | 20/455 = 0.044 | 20/179 = 0.112 | 0.026 | 1.23 (43% of cues) | 2.88 | 4.00 |
| control: object class only | 15/455 = 0.033 | 15/179 = 0.084 | 0.020 | 0.57 (34% of cues) | 1.67 | 3.48 |
| control: shuffled labels, draw 1 | 11/455 = 0.024 | 11/179 = 0.061 | 0.018 | 0.70 (42% of cues) | 1.67 | 3.48 |
| control: shuffled labels, draw 2 | 14/455 = 0.031 | 14/179 = 0.078 | 0.020 | 0.61 (37% of cues) | 1.67 | 4.00 |
| control: shuffled labels, draw 3 | 11/455 = 0.024 | 11/179 = 0.061 | 0.020 | 0.66 (39% of cues) | 1.68 | 2.87 |

Difference from first-in first-out at equal speech (95% interval, bootstrap over videos):

- threshold, VIABench scorer: reminders answered +0.044 [+0.021, +0.066]; unneeded cues per min -0.96 [-1.21, -0.70]; speech s/min -0.05 [-0.66, +0.57]
- threshold, EgoBlind scorer: reminders answered -0.009 [-0.039, +0.020]; unneeded cues per min -0.71 [-0.95, -0.48]; speech s/min -0.58 [-1.16, -0.00]
- control: object class only: reminders answered -0.011 [-0.032, +0.007]; unneeded cues per min -0.66 [-0.91, -0.40]; speech s/min -0.52 [-0.98, +0.01]
- control: shuffled labels, draw 1: reminders answered -0.020 [-0.036, -0.005]; unneeded cues per min -0.53 [-0.76, -0.29]; speech s/min -0.51 [-0.97, -0.07]
- control: shuffled labels, draw 2: reminders answered -0.013 [-0.035, +0.007]; unneeded cues per min -0.61 [-0.81, -0.41]; speech s/min -0.00 [-0.44, +0.40]
- control: shuffled labels, draw 3: reminders answered -0.020 [-0.044, +0.005]; unneeded cues per min -0.57 [-0.77, -0.36]; speech s/min -1.12 [-1.49, -0.74]

VIABench scorer against the object-class-only control: reminders answered +0.055 [+0.034, +0.080]; unneeded cues per min -0.30 [-0.49, -0.12]; speech s/min +0.47 [-0.21, +1.18]
