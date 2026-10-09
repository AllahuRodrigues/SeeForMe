# Exploratory EgoBlind results (2026-10-08), superseded

Everything in this folder was produced before the 2026-10-09 audit and is kept only as a record. Do not cite these
numbers. They were computed with three defects that the audit found and fixed (item 4 lists the configs tuned under them):

1. **Inverted weak labels.** "Is the path clear?" was read with yes = something there and "is it dangerous?" with
   yes = clear; clock directions ("at 2 o'clock") matched the COCO class "clock". 120 of 1,398 labelled questions had
   the wrong label.
2. **A leaky validation split.** `val` was EgoBlind train videos with `video_id % 7 == 0`; 86 of its 97 clips shared a
   source recording with training clips. The EgoBlind-trained schedulers also chose their checkpoints, thresholds and
   RL epoch on these same videos, so their `val` numbers are optimistic.
3. **An annotation leak into perception.** The indoor / outdoor depth model was chosen from each video's questions and
   answers.

4. **Pre-audit P-CACS configs.** `pcacs_egoblind.yaml`, `pcacs_egoblind_allq.yaml` and `pcacs_egoblind_ungated.yaml`
   (moved here from `configs/`) were tuned (`tune_stdout.log`) on 251 cached videos of the old `train` pool
   (`video_id % 7 != 0`) with the inverted labels; 65 of the 77 current holdout clips belong to that pool. No current
   result uses them; `configs/pcacs_egoblind.yaml` was re-tuned on 2026-10-09 on the new 70% `train` split with the
   corrected labels (`make egoblind-tune`).

The replacement results use the 70 / 15 / 5 / 10 source-recording split (train / finetune / select / holdout, see
`scripts/egoblind_common.py`), the corrected labels and label-free depth selection; they live one folder up
(`results/egoblind/*_holdout_*`, `*_train_*`) and are described in `docs/egoblind.md`.
