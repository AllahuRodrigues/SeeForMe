Event-failure funnel on train,finetune (558 videos), seed 0

| policy | N events | P detected | T tracked | C candidate | S spoken | D on time | lost: perception / tracking / candidate / scheduler (blocked) / timing |
|---|---|---|---|---|---|---|---|
| pcacs | 293 | 169 | 160 | 160 | 23 | 15 | 124 / 9 / 0 / 137 (0) / 8 |
| egoblind_learned:il | 293 | 169 | 160 | 154 | 66 | 57 | 124 / 9 / 6 / 88 (0) / 9 |
| online_greedy | 293 | 169 | 160 | 154 | 56 | 33 | 124 / 9 / 6 / 98 (0) / 23 |

Onset timing (D1). native: cue within [t_q - 10 s, t_q + 0.5 s]; EgoVoice-style: within [-2.5 s, +1.5 s] of ideal_s.

| policy | variant | recall | Start@0.5 / 1 / 2 (of matched) | Start@1 of all events | mean \|Δ\| (s) | mean signed Δ (s) |
|---|---|---|---|---|---|---|
| pcacs | native | 0.051 | 0.200 / 0.400 / 0.467 | 0.020 | 3.539 | -3.455 |
| pcacs | egovoice | 0.027 | 0.250 / 0.375 / 0.875 | 0.010 | 1.032 | 0.237 |
| egoblind_learned:il | native | 0.195 | 0.193 / 0.263 / 0.421 | 0.051 | 3.053 | -3.014 |
| egoblind_learned:il | egovoice | 0.113 | 0.273 / 0.667 / 1.000 | 0.075 | 0.825 | -0.310 |
| online_greedy | native | 0.113 | 0.121 / 0.152 / 0.242 | 0.017 | 4.259 | -4.231 |
| online_greedy | egovoice | 0.061 | 0.278 / 0.444 / 0.889 | 0.027 | 1.097 | -0.654 |

Why scheduler misses happened (inside the window): outranked = another object was spoken about while it was a legal candidate; top but below threshold = it ranked first and the policy stayed silent.

- pcacs: top_but_below_threshold 103, never_top_never_outranked 26, outranked 8
- egoblind_learned:il: top_but_below_threshold 70, never_top_never_outranked 12, outranked 6
- online_greedy: top_but_below_threshold 80, never_top_never_outranked 13, outranked 5

**pcacs**, most common labels lost at each stage:
- perception: unknown 34, door 22, pole 9, stairs 8, dining table 6, fence 6, wall 4, car 4
- tracking: dining table 2, car 2, knife 1, couch 1, person 1, motorcycle 1, horse 1
- scheduler: person 65, car 23, motorcycle 12, dining table 8, dog 4, chair 4, bicycle 4, knife 3
- timing: motorcycle 4, car 3, person 1

**egoblind_learned:il**, most common labels lost at each stage:
- perception: unknown 34, door 22, pole 9, stairs 8, dining table 6, fence 6, wall 4, car 4
- tracking: dining table 2, car 2, knife 1, couch 1, person 1, motorcycle 1, horse 1
- candidate: dining table 2, car 2, bed 1, horse 1
- scheduler: person 37, car 11, motorcycle 7, dining table 6, dog 4, chair 4, bicycle 4, knife 3
- timing: car 4, person 4, motorcycle 1

**online_greedy**, most common labels lost at each stage:
- perception: unknown 34, door 22, pole 9, stairs 8, dining table 6, fence 6, wall 4, car 4
- tracking: dining table 2, car 2, knife 1, couch 1, person 1, motorcycle 1, horse 1
- candidate: dining table 2, car 2, bed 1, horse 1
- scheduler: person 42, car 16, motorcycle 9, dining table 5, dog 4, bicycle 4, knife 3, chair 3
- timing: person 9, car 7, motorcycle 6, bicycle 1
