Coverage-first diagnostic on the EgoBlind dev stream (confirm after 2 decisions, ledger capacity 6 s; budget x1 = 3.83 spoken s/min, the threshold rule's rate on train)

**`finetune` (104 videos, 20 source recordings; out of sample for the threshold's logistic)**

| policy | events on time | of events with a candidate | quiet moments spoken over | cues/min | speech s/min | repeats, same object | repeats, same label in 20 s |
|---|---|---|---|---|---|---|---|
| threshold | 9/56 = 0.161 | 9/27 = 0.333 | 16% | 1.72 | 3.80 | 25% | 22% |
| coverage-first, budget x1 | 9/56 = 0.161 | 9/27 = 0.333 | 16% | 2.68 | 3.82 | 0% | 22% |
| coverage-first, budget x2 | 11/56 = 0.196 | 13/27 = 0.481 | 29% | 4.96 | 7.53 | 32% | 45% |
| coverage-first, budget x4 | 16/56 = 0.286 | 21/27 = 0.778 | 50% | 8.62 | 14.60 | 59% | 59% |
| coverage-first, no budget | 11/56 = 0.196 | 21/27 = 0.778 | 87% | 16.07 | 37.68 | 76% | 72% |

**train + finetune (558 videos; the threshold is in-sample on the train part)**

| policy | events on time | of events with a candidate | quiet moments spoken over | cues/min | speech s/min | repeats, same object | repeats, same label in 20 s |
|---|---|---|---|---|---|---|---|
| threshold | 42/293 = 0.143 | 43/154 = 0.279 | 11% | 1.68 | 3.82 | 27% | 24% |
| coverage-first, budget x1 | 39/293 = 0.133 | 40/154 = 0.260 | 24% | 2.71 | 3.80 | 9% | 18% |
| coverage-first, budget x2 | 69/293 = 0.235 | 81/154 = 0.526 | 38% | 5.00 | 7.48 | 34% | 40% |
| coverage-first, budget x4 | 84/293 = 0.287 | 101/154 = 0.656 | 59% | 8.84 | 14.57 | 52% | 57% |
| coverage-first, no budget | 86/293 = 0.294 | 117/154 = 0.760 | 88% | 16.94 | 38.72 | 70% | 69% |

**Difference from the threshold rule on `finetune`** (bootstrap over source recordings, 95%)

| policy | events on time | quiet moments spoken over |
|---|---|---|
| coverage-first, budget x1 | +0.000 [-0.102, +0.108] | +0.000 [-0.208, +0.148] |
| coverage-first, budget x2 | +0.036 [-0.091, +0.146] | +0.132 [-0.147, +0.317] |
| coverage-first, budget x4 | +0.125 [+0.067, +0.179] | +0.342 [+0.167, +0.476] |
| coverage-first, no budget | +0.036 [-0.091, +0.146] | +0.711 [+0.450, +0.862] |

**Why events with a candidate were still missed** (train + finetune; main reason logged inside the event's window)

- threshold: below_threshold 107, channel_busy 4
- coverage-first, budget x1: budget 78, unconfirmed 32, already_announced 3, channel_busy 1
- coverage-first, budget x2: budget 40, unconfirmed 30, already_announced 2, channel_busy 1
- coverage-first, budget x4: unconfirmed 28, budget 21, already_announced 3, channel_busy 1
- coverage-first, no budget: unconfirmed 23, channel_busy 8, already_announced 6

`events on time` uses the benchmark's recall (as in E2); `of events with a candidate` uses the failure funnel (a cue about the event's own object inside its window, over events whose object reached the scheduler in the window).
