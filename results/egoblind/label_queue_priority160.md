Label queue `priority160` (seed 7; 160 decision moments from 91 source recordings; train 132, finetune 28; at most 2 per recording)

| stratum | wanted | available | picked for the stratum | filled in from it | final |
|---|---|---|---|---|---|
| door | 24 | 34 | 24 | 1 | 25 |
| stairs | 24 | 15 | 12 | 0 | 12 |
| pole | 16 | 23 | 16 | 1 | 17 |
| scene | 28 | 34 | 17 | 0 | 17 |
| covered_other | 28 | 187 | 28 | 7 | 35 |
| quiet | 24 | 279 | 24 | 7 | 31 |
| unasked | 16 | 213 | 16 | 7 | 23 |

Proposed sentence: system 42, label 64, system_top 43, none 11 (system = live cue text for the event's own track; label = cue from the weak label because the detector missed it; system_top = cue for the most confident detection at that moment; none = no sentence, usefulness only).
No human labels are included. `select`, `holdout`, `test` and own recordings are never sampled.
