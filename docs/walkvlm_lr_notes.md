# WalkVLM-LR (arXiv 2508.16070): what is public, the rewards, the environment awareness discriminator, and how it relates to the SeeForMe scheduler

"Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants", Chongyang Li, Zhiqiang Yuan,
Hanbo Bi, Zexi Jia, Jinchao Zhang (WeChat AI, Tencent; UCAS). arXiv v1 2025-08-22, v4 2026-05-12; ICASSP 2026 Best
Industry Paper. Project page https://walkvlm-lr.github.io. Checked 2026-10-08.

## What is public

| item | where | size | status |
|---|---|---|---|
| code | https://github.com/xiaoyuan1996/walkvlm, directory `WalkVLM-LR/` | repo 4.6 MB, shallow clone in `external/walkvlm` (10 MB, gitignored) | **archived research snapshot**: `EAD.py` (model definition), `train_EAD.py` (legacy trainer with an empty `data_dir`), `inference.py` and `test.py` replaced by archive notices, `vlm_grpo_template/` (Open-R1 GRPO template, Apache 2.0; its `grpo_query_gene.py` is a 678-byte stub). The maintained successor is `WalkStream/` (January 2026, "IJCV submitted"): `src/otp/` (Online Triggered Prediction = the discriminator, same network), `src/reminder_generation/` (GRPO with the four rewards implemented in `src/open_r1/grpo_query_gene.py`), `src/infer/GPTScore.py` |
| WalkVLM-LR weights | none | - | `WalkVLM-LR/checkpoint/download_checkpoint.sh` only pulls the public bases `Qwen/Qwen2-VL-2B-Instruct`, `openai/clip-vit-base-patch32` and `gpt2` from Hugging Face; the trained LR model and the trained EAD head are not released |
| WalkStream checkpoint | Tencent COS, `WalkStream/src/checkpoint/download_checkpoint.sh` | `model.safetensors` 4.88 GB + tokenizer / config files | a Qwen2-VL-2B fine-tune; not downloaded (over the 1 GB limit) |
| WAD dataset (Walking Awareness Dataset, Yuan et al. 2024) | `wad_dataset/download_wad.sh` -> `wad_dataset.tar`; project page links `wad_dataset_v2.tar` | **129.3 GB** each | 12k video-annotation pairs, ~120k images, 10 locations in Europe and Asia, ~13 h; ~20% recorded by annotators, the rest from YouTube. Not downloaded. **No licence or terms are stated** on the project page, in the repository, or in the download scripts |
| WRD dataset (WalkStream) | `WalkStream/src/wrd_dataset/download_wad.sh` | `WRD.zip` 374.9 GB, `WRD_test.zip` 11.6 GB | 98k clips; not downloaded; no licence stated |
| Hugging Face | `RR32444/walk_vlm` (third-party, 2025-12-19) | 53 KB parquet, 200 rows | a sample of WAD-style annotations with columns `video, frame_path, weather_condition, area_type, danger_level, traffic_flow_rating, summary, alter`; no official HF release of models or data |

Repository licence: the repository root has no LICENSE file; only the Open-R1-derived `vlm_grpo_template` carries
Apache 2.0. Treat the rest as "all rights reserved, research reference".

## Method, as published and as implemented

**Base model and training.** Qwen2-VL-2B-Instruct fine-tuned with GRPO (Open-R1 framework) for one epoch on WAD; max input
1,024 tokens, max generation 700 tokens, batch 1 x grad-accumulation 2, bf16, FlashAttention-2, image area capped at
401,408 px.

**Four human-preference reward functions** (paper Eqs. 1-8; code names in `grpo_query_gene.py`, default
`reward_funcs = ['fluency', 'simplicity', 'sematic', 'keywords']`):

| reward | paper definition | implementation detail in the maintained code |
|---|---|---|
| Simplicity (conciseness) | `R = r_max - ((L_out - L0) / L0)^2`: quadratic penalty on the distance from an ideal length L0 (values of `r_max`, `L0` not given) | length-based; no separate function in the excerpt read, the length term is folded into the trainer |
| Fluency | `R = D_n / (D_n + PPL)` with `PPL` the GPT-2 perplexity and `D_n` the distinct-n-gram ratio | `fluency_reward`: GPT-2 perplexity scaled by 1e-4 through a sigmoid into [0.1, 1], combined with a 0.7 / 0.3 weighted 1-gram / 2-gram diversity as `D / (D + 0.5 * gpt_reward)` |
| Accuracy ("sematic") | `R = CosSim(text-encoder(output), text-encoder(annotation)) + mta`, `mta` = token-level exact-match rate | `semantic_similarity_reward`: `0.75 * mean token accuracy + 0.25 * CLIP (ViT-B/16) text cosine`, clipped to 1 |
| Keywords (keyword density) | `R = (1/n) sum_i sum_{s in S(k_i)} f_gen(s)`: frequency of the annotation's keywords and their synonyms in the output | `keywords_reward`: share of the output's unique unigrams + bigrams whose CLIP text embedding has cosine >= 0.9 with any annotation keyword; a `calculate_redundancy_penalty` (0.1 per repeated keyword) exists but is not wired in |

The weights that combine the four rewards are not stated in the paper.

**Environment awareness discriminator (EAD; `OTP` in WalkStream).** Sits between the VLM's visual encoder and its LLM and
reuses the encoder output (Qwen2-VL visual tokens, 1,536-d) for the current frame and the previous n = 3 frames. Network
(`EAD.py` / `otp_model.py`, identical): `MultiScaleConv` = three parallel 1-D convolutions (kernels 3, 5, 7, 1536 -> 512
each, concatenated to 1,536), an 8-head self-attention layer over the token sequence (the paper's "Scene Awareness
Inference Module"), mean pooling, then MLP 1536 -> 1024 -> 512 -> 3. Output: a danger grade per frame, **A (low), B
(medium), C (high)**. Loss: cross-entropy + focal loss (alpha 0.25, gamma 2) + label smoothing 0.1; Adam 1e-4; the
paper trains it for 4 epochs on 20 manually labelled real walking videos with the vision encoder frozen (the
`train_EAD.py` default is 60 epochs; the released trainer reads `<video>_features.json` files with `visual_features`
and a `label` string whose last token is the grade).

**Gating rule.** The LLM is only invoked when the discriminator says the scene is dangerous; the maintained inference
(`WalkStream/src/otp/inference.py`) implements it literally: predict a grade for each of the 1-3 input frames and
generate a reminder only if `"C" in predicted_labels` (`--always_generate` bypasses it). There is no hysteresis,
cooldown or memory beyond the three-frame window, and no notion of what was said before.

**Metrics.** ROUGE-1/2/L, keyword density (reference keywords plus CLIP synonyms at cosine > 0.9), a GPT score
(fluency and conciseness judged by GPT-4o relative to the reference), and **TRF, Temporal Redundancy F1**: the F1
between the predicted and ground-truth danger grades over consecutive frames, i.e. redundancy is measured as
*speaking when the frame was not graded dangerous*, not as repeating content. The public evaluation script
(`WalkStream/src/otp/test.py`) computes ROUGE and keyword density only; the TRF code is not released. Headline numbers:
ROUGE-L 0.406, keyword density 0.576, GPT score 0.776, TRF 0.670 (next best InternVL3 0.475, GPT-4o 0.430); EAD
ablation TRF 0.670 full, 0.653 without the multi-scale conv, 0.637 without attention, 0.437 without the MLP head.
Deployment appendix: 4-bit Qwen2-VL-2B needs about 2.9 GB of GPU memory, 5.5 tokens/s on a Mac mini M4 Pro, 1-2 s per
reminder.

## Relation to the SeeForMe scheduler

Both systems separate *whether to speak now* from *what to say*, and both put the decision in front of the expensive
generator. The differences are in what the decision is made from and how it is scored.

| aspect | WalkVLM-LR | SeeForMe |
|---|---|---|
| decision input | a danger grade A/B/C predicted from the last 4 frames' visual tokens (scene-level, class-agnostic) | per-object state from detection + tracking + metric depth: distance, TTC, corridor probability, severity by event type, plus channel state (what is playing, how long, haptic budget) |
| temporal redundancy | avoided by not speaking in frames graded A or B; the only memory is the 3-frame window, so a hazard that stays in view stays "C" and the generator is invoked again every cycle (the failure cases in the appendix are exactly repeated reminders about a distant object) | avoided by per-track `last_announced`, the memory ledger and the event-window rule: an object is announced once inside `[earliest_useful, latest_safe]`, re-announced only on a material change (closer, moving into the corridor) |
| silence | an outcome of the grade: there is no positive label for "the user is fine", so TRF only penalises speaking in A/B frames | a first-class label: **quiet windows** (`should_speak = False` with a window) where a cue is charged as a violation, and **audio occupancy** (fraction of time the channel is busy) is a reported cost alongside recall and regret |
| timing | frame-level: a reminder is right if the frame is graded C; no earliest / ideal / latest, no reaction time, no interruption | event windows derived from reaction time and TTC per event type; late cues score lower (`regret`), urgent cues may interrupt |
| verbosity | the Simplicity and Keywords rewards shorten the text at training time | `compose()` fills only the `minimum_information` slots of the event; length is not learned but prescribed per event type |
| evaluation data | WAD / WRD frames with reminder text and danger grade, 2 s sampling in the user study | CampusBench (simulated ground truth with windows) and EgoBlind weak events from users' questions |

Where WalkVLM-LR helps SeeForMe:

1. **A class-agnostic danger grade** is the closest published analogue to SeeForMe's `unknown_obstacle` severity and to
   the "should the detector's silence be trusted" problem on EgoBlind (half the asked-about hazards are not COCO
   classes). A three-grade scene classifier on the visual tokens, trained on a few hundred labelled frames, could feed
   the P-CACS urgency term when no track explains the risk.
2. **TRF** is a cheap metric to add to the EgoBlind and CampusBench tables: F1 between "spoke in this second" and
   "some event window is open in this second". It is weaker than the window-based recall / violation pair but makes
   the comparison with VLM assistants direct.
3. The **reward design** (length penalty, keyword density, GPT-judged conciseness) is a recipe if the cue text is ever
   generated by a VLM instead of `compose()`; the REINFORCE scheduler trainer already has the GRPO-style group baseline.

What does not transfer: the A/B/C labels are per frame and per video, so a WalkVLM-LR-style discriminator cannot be
scored with SeeForMe's event-window metrics without re-annotating when each hazard became relevant; and WAD / WRD
(129 GB / 375 GB, no licence) are out of reach for this project's disk and licensing constraints, so the comparison
stays at the level of the method, not the data.
