Event-failure funnel on train,finetune (558 videos), seed 0, with segmentation candidates (share >= 0.02, 2 consecutive s)

| policy | N events | P detected | T tracked | C candidate | S spoken | D on time | lost: perception / tracking / candidate / scheduler (blocked) / timing |
|---|---|---|---|---|---|---|---|
| pcacs | 293 | 177 | 166 | 166 | 30 | 21 | 116 / 11 / 0 / 136 (0) / 9 |
| egoblind_learned:il | 293 | 177 | 166 | 161 | 66 | 57 | 116 / 11 / 5 / 95 (0) / 9 |
| online_greedy | 293 | 177 | 166 | 161 | 65 | 34 | 116 / 11 / 5 / 96 (0) / 31 |

Onset timing (D1). native: cue within [t_q - 10 s, t_q + 0.5 s]; EgoVoice-style: within [-2.5 s, +1.5 s] of ideal_s.

| policy | variant | recall | Start@0.5 / 1 / 2 (of matched) | Start@1 of all events | mean \|Δ\| (s) | mean signed Δ (s) |
|---|---|---|---|---|---|---|
| pcacs | native | 0.075 | 0.227 / 0.364 / 0.455 | 0.027 | 3.395 | -3.285 |
| pcacs | egovoice | 0.041 | 0.250 / 0.500 / 0.917 | 0.020 | 0.908 | 0.155 |
| egoblind_learned:il | native | 0.195 | 0.211 / 0.298 / 0.456 | 0.058 | 2.937 | -2.898 |
| egoblind_learned:il | egovoice | 0.113 | 0.273 / 0.636 / 1.000 | 0.072 | 0.838 | -0.249 |
| online_greedy | native | 0.116 | 0.235 / 0.353 / 0.500 | 0.041 | 2.819 | -2.743 |
| online_greedy | egovoice | 0.061 | 0.389 / 0.722 / 1.000 | 0.044 | 0.720 | 0.061 |

Why scheduler misses happened (inside the window): outranked = another object was spoken about while it was a legal candidate; top but below threshold = it ranked first and the policy stayed silent.

- pcacs: by rank: top_but_below_threshold 99, never_top_never_outranked 25, outranked 12
  - main logged reason per missed event: below_threshold 131, low_confidence 4, outranked 1
  - all in-window opportunities: below_threshold 4432, low_confidence 62, outranked 42
- egoblind_learned:il: by rank: top_but_below_threshold 74, never_top_never_outranked 15, outranked 6
  - main logged reason per missed event: below_threshold 95
  - all in-window opportunities: below_threshold 1607
- online_greedy: by rank: top_but_below_threshold 77, never_top_never_outranked 15, outranked 4
  - main logged reason per missed event: below_threshold 93, channel_busy 3
  - all in-window opportunities: below_threshold 2117, channel_busy 67, outranked 2, recently_announced 1

**pcacs**, most common labels lost at each stage:
- perception: unknown 34, door 19, pole 9, stairs 7, dining table 6, wall 4, car 4, sign 3
- tracking: dining table 2, car 2, knife 1, couch 1, person 1, stairs 1, motorcycle 1, horse 1
- scheduler: person 63, car 23, motorcycle 11, dining table 8, dog 4, chair 4, bicycle 4, knife 3
- timing: motorcycle 4, car 3, person 1, fence 1

**egoblind_learned:il**, most common labels lost at each stage:
- perception: unknown 34, door 19, pole 9, stairs 7, dining table 6, wall 4, car 4, sign 3
- tracking: dining table 2, car 2, knife 1, couch 1, person 1, stairs 1, motorcycle 1, horse 1
- candidate: dining table 2, car 2, bed 1
- scheduler: person 37, car 11, motorcycle 7, dining table 6, dog 4, chair 4, bicycle 4, knife 3
- timing: car 4, person 4, motorcycle 1

**online_greedy**, most common labels lost at each stage:
- perception: unknown 34, door 19, pole 9, stairs 7, dining table 6, wall 4, car 4, sign 3
- tracking: dining table 2, car 2, knife 1, couch 1, person 1, stairs 1, motorcycle 1, horse 1
- candidate: dining table 2, car 2, bed 1
- scheduler: person 40, motorcycle 10, car 10, dining table 6, dog 4, chair 4, bicycle 4, knife 3
- timing: person 14, car 9, motorcycle 5, bicycle 1, fence 1, horse 1
