# Paper draft (v1.1, 2026-11-12)

Target: ASSETS 2027 poster or short paper, or CHI 2027 Late-Breaking Work. An ML-venue variant is sketched in section 11. Every number below comes from `docs/online.md`, `docs/memory.md`, `docs/research/coder-report.md` or the `results/online/` and `results/memory/` files, unless it is marked otherwise:
- **(prelim.)** marks a number that comes from the algorithm agent's scratch scripts outside the repo (reported in `docs/research/algorithms.md`). The final audit reproduced these, but they have to be regenerated with code in the repo before submission.
- **(unverified)** marks a number that no log or script in the repo or the scratch directory reproduces. It must be regenerated or removed before submission.

Corrections from the audits are applied throughout:
- The old 0.525 full-information result used zero-delay labels inside the same clip. It is called a **zero-delay reference**, not an upper bound. With labels held to video end (a causal reference), full information ties one-sided greedy. The "60% of learnable recall" claim is gone.
- The causal reference, the 2 s delay, the no-learning floor and the p_ask sweep are now in the repo with 3 seeds and paired bootstrap intervals.
- The no-learning floor is described as a rate-controlled, first-listed-candidate policy, not a content prior.
- The frozen IL scheduler's same-track repeat rate is 13.6%, not 28.5%.
- EgoBlind is described as 1,329 clips from 478 source videos.
- `select` has 11 coverable events per seed, so ratios like 9/33 are pooled over 3 seeds.
- Deployed apps are described as "wait to be asked or speak by fixed, content-gated rules".
- Timing and one-sided learning are presented as prior art. The contribution is a diagnostic benchmark and an empirical characterisation, not a new setting.

---

## Title

**When Should a Camera Assistant Speak? A Diagnostic Benchmark for Online Cue Timing from One-Sided Feedback on Blind Users' First-Person Video**

## Abstract

Phone-camera assistants for blind and low-vision (BLV) people now describe scenes well. Most of them either wait to be asked or narrate by fixed, content-gated rules, and none learns *when* to speak from the person using it. We study proactive cueing for BLV pedestrians as online learning from one-sided feedback under a speech budget:
- after a cue, the user may tap "useful" or "not useful";
- after silence, the only signal is a later question that reveals a miss;
- silence itself is never labelled.

We build a replayable, simulated-feedback stream from 558 EgoBlind videos recorded by blind people (290 minutes, 160 events the detector can see, 111 proxy users) and use it as a diagnostic benchmark. We compare online logistic learners (greedy, budgeted apple tasting, Thompson sampling) with a fixed utility rule, frozen offline schedulers, a no-learning floor and full-information references with controlled label delay.

The online learners reach 0.183 to 0.204 coverable recall (3 seeds). Coverable recall counts only the 160 of 293 events (55%) the detector can see; over all weakly labelled events greedy catches 0.111. That is far below what a dependable navigation aid needs, which is why perception and anticipation, not the feedback channel, are the priority. The fixed rule gets 0.087 and a frozen RL scheduler 0.050. Greedy beats a no-learning, rate-controlled floor (0.138) by +0.067 (95% CI [+0.010, +0.121]). Out of sample the learners cannot be separated from a frozen imitation-learned scheduler (4 of 27 events, against 2 to 7 of 27).

The central finding is about where the bottleneck is. A full-information learner whose labels arrive with zero delay reaches 0.512, but that reference is not causal. When its labels are held to the end of each clip it reaches 0.200, against 0.204 for one-sided greedy (difference −0.004 [−0.054, +0.048]). Labels that arrive 2 s late inside the clip reach 0.379 (+0.175 [+0.112, +0.239]). Asking only half or a quarter of the questions gives greedy 0.167 and 0.179, a change we cannot separate from seed noise. So on real BLV video the binding constraints are within-clip label latency and weak features, not the one-sidedness of the feedback. All feedback here is simulated (taps) or replayed from annotation (questions); no BLV user has yet interacted with the learner, so every result is a simulated-feedback replay experiment.

Three common remedies also fail:
- Exploration does not help.
- Warm-starting from a simulator with full optimiser state hurts (−0.058, 95% CI [−0.107, −0.013]).
- Treating silence as a weak negative hurts at every weight we tried.

We release the protocol and report what a real-user study must measure, above all how quickly real users give feedback inside an event.

## 1. Introduction

A camera can detect twenty things at once; a person cannot listen to twenty things at once. For a BLV pedestrian with a chest-mounted phone, every spoken cue uses up attention and masks environmental sound. Every missed cue may be an obstacle, a door or a turn the person wanted to know about.

Deployed assistants handle this tension in two ways:
- **On request.** Be My Eyes and Be My AI answer when asked.
- **By fixed, content-gated rules.** Seeing AI's Short Text channel reads text as soon as it is in view (https://www.guidedogs.org.uk/services-we-provide/technology/Seeing-AI, https://hadleyhelps.org/workshops/seeing-ai-apple-voiceover-series/reading-short-text) [verify]. Lookout's Explore mode (beta) speaks as the user points the camera around (https://support.google.com/accessibility/android/answer/9031274).

None of these learns when to speak. The nearest commercial systems are Google's Project Astra, described as able to "intuitively start conversations" (https://deepmind.google/models/project-astra/), and Gemini Live Guided Vision, launched on 2026-10-01 and not intended for navigation (https://www.androidauthority.com/gemini-live-guided-vision-launch-3718122/). Neither has a published method.

Research has started to treat timing as a learning target:
- TIMELI learns when to talk from simulator-rule labels (ICRA 2026, arXiv 2609.05596).
- VIABench scores proactive reminders on real BLV video (arXiv 2607.14660).
- WorldScribe gives unprompted live descriptions with hand-set triggers (UIST 2024, arXiv 2408.06627).

All of these learn offline from fixed labels or rules. In use, an assistant sees only what its user tells it, and that signal is one-sided. Chang et al. found that a live-video assistant falls short in dynamic scenes for BLV users, and they call for intervention timing beyond turn-taking (ASSETS 2025, n=8, arXiv 2508.03651).

We ask: **if an assistant learns when to speak from its user's feedback during use, what limits it: the one-sidedness of that feedback, how late it arrives, or what the features can see?**

**Contributions.**
1. **Problem instance.** We formalise proactive BLV cueing as budgeted apple tasting in which positive labels after silence are user-triggered, delayed by up to about 10.5 s, and attached to a time window rather than to a single candidate. This is a new instance of known settings (apple tasting; user-triggered supervision, Agarwal et al., arXiv 2302.03784). It is not a new setting.
2. **A replayable diagnostic benchmark on real BLV video.** We provide a feedback simulator over EgoBlind with progressive validation, coverable recall, quiet-window violations, a cue-rate budget, a controllable asking rate (p_ask) and full-information references with controllable label delay. It keeps strict split hygiene: holdout and test are never used.
3. **A diagnosis of the bottleneck.**
   - A zero-delay full-information reference overstates the cost of one-sided feedback. Held to clip end, full information ties one-sided greedy (0.200 against 0.204, difference −0.004 [−0.054, +0.048]).
   - Within-clip label latency matters a great deal: labels 2 s late reach 0.379 (+0.175 [+0.112, +0.239] over greedy).
   - Lowering the asking rate to 0.5 or 0.25 produces no change separable from seed noise.
   - Learning helps a little: greedy beats a no-learning floor by +0.067 [+0.010, +0.121].
   - Feature signal is weak (prelim.: offline AUC about 0.68 on finetune), led by the user's own walking speed.
4. **Negative results on common remedies.** Exploration, simulator warm start and silence-as-negative do not help, and the last two hurt.
5. **A memory and repetition result.** Appearance re-identification matches the same object about 90% of the time but has little effect on 31-second clips. The frozen imitation scheduler's repeat cues come mostly from a short per-track block, not from forgetting: 13.6% of its cues repeat the same track, and a 10 s re-announcement guard cuts that to 7.9% with recall unchanged.

## 2. Related work

All citations below were checked against the source at abstract level or deeper, except those marked [verify] or (unverified), which must be opened before submission.

**When to speak for BLV users.**
- **TIMELI** (Kuribayashi, Shangguan, Ohn-Bar; ICRA 2026, arXiv 2609.05596). Treats silence as an explicit label. Trains on about 67K CARLA clips with a rule-based target and has a small real test set.
- **VIABench** (arXiv 2607.14660). 761 videos, 46.9 h, 530 of them real videos from visually impaired creators. Its Proactive Reminder task scores alerts within event intervals and proposes TPAD, a training-free warning probability. LiveCC reaches 75.8% Stage-1 recall, which the authors attribute to dense output. There is no budget and no feedback.
- **WorldScribe** (UIST 2024, arXiv 2408.06627). Gives unprompted descriptions using hand-set triggers: orientation change, frame similarity and stability. It does not learn from feedback.
- **WalkVLM** (arXiv 2412.20903) and **WalkVLM-LR** (ICASSP 2026, arXiv 2508.16070). Use a discriminator to gate when to speak and use offline reward shaping to cut redundancy.
- **ForeSightGuide** (arXiv 2608.18993). Filters out objects that pose no threat. Offline.
- **VIA-Agent** (UIST 2026, arXiv 2511.00945). Names a Continuity principle that its abstract does not define.
- **Ohn-Bar's group** previously adapted to individual blind users online with weighted experts (PING, CoRL 2018, arXiv 1804.04118).

We differ from all of these in learning online from one-sided user feedback under an explicit budget, and in measuring how much that feedback structure, as opposed to its latency, costs.

**Proactive egocentric assistants (not BLV).**
- EgoVoice (EMNLP 2026, arXiv 2610.12248): when and what to say, trained by fine-tuning and offline DPO.
- Vinci2 (ECCV 2026, arXiv 2607.11523): a training-free decision on whether to intervene.
- Zhang et al. (EMNLP 2025, https://aclanthology.org/2025.emnlp-main.605/) [verify]: speak-or-stay-silent dialogue generation from streaming egocentric video.
- ProactiveAgent (arXiv 2410.12361; ICLR 2025 venue unverified): a reward model trained on accept and reject labels.

All of them are offline, and most use two-sided labels.

**Interruption management.** Horvitz et al., attention-sensitive alerting (UAI 1999, arXiv 1301.6707). Interruptibility prediction from behaviour [verify: Pielot et al., IMWUT 2017; Okoshi et al., KDD 2019]. In our setting the cost of silence is a missed event the user cannot see, not lost engagement.

**One-sided and partial feedback.**
- Apple tasting (Helmbold, Littlestone, Long, Inf. Comput. 2000, https://dblp1.uni-trier.de/rec/journals/iandc/HelmboldLL00.html) [verify]. Its agnostic characterisation, with Θ(1), Θ(√T) or Θ(T) realizable rates, is Raman et al., COLT 2024 (arXiv 2310.19064).
- Label-efficient prediction (Cesa-Bianchi, Lugosi, Stoltz, IEEE TIT 2005) [verify].
- Partial monitoring (Bartók et al. 2014, https://www.mit.edu/~rakhlin/papers/partial_monitoring.pdf) [verify].
- One-sided feedback in lending (Jiang, Jiang, Pacchiano, arXiv 2006.04858, AISTATS 2021 venue unverified; Bechavod et al., NeurIPS 2019, arXiv 1902.02242).
- Sculley (KDD 2007) [verify]: mistake-driven learners suffer under one-sided feedback.
- User-triggered supervision in contextual bandits (Agarwal, Gentile, Marinov, arXiv 2302.03784, no venue listed). This is our closest formal ancestor.
- Delay (Joulani et al., ICML 2013) [verify] and delayed, aggregated, anonymous feedback (Pike-Burke et al., ICML 2018, arXiv 1709.06853).
- Bandits with knapsacks (Badanidiyuru et al., arXiv 1305.2545; Agrawal and Devanur, arXiv 1507.06738).
- Positive-unlabelled learning (Elkan and Noto, KDD 2008; Kiryo et al., NeurIPS 2017) [verify].

**BLV egocentric data.**
- EgoBlind (NeurIPS 2025 D&B, arXiv 2503.08221): 1,329 clips from 478 source videos, 5,311 questions.
- GuideDog (ACL 2026, arXiv 2503.12844): 22K image-description pairs from still images, gated; the CC BY-NC licence is unverified.
- ORBIT (ICCV 2021, arXiv 2104.03841): teachable recognition.

None of them carries timing or feedback labels.

**What BLV users ask for.**
- Control over verbosity and preferences that persist across sessions (Say It My Way, CHI 2026, 11 blind participants, 418 interactions, arXiv 2602.16930).
- The ability to contest AI output and have it improve (Misfitting With AI, ASSETS 2024, 26 BLV participants, arXiv 2408.06546).

Both motivate a learner that adapts to feedback.

## 3. Problem formulation

**Rounds.** Decisions happen every Δ = 0.5 s. At round t the system sees a candidate set C_t of tracked detections. Each candidate has a 29-dimensional P-CACS feature vector x (hazard, urgency, actionability, goal relevance, novelty, uncertainty, cost, ego speed and others).

**Actions.** The system either stays silent or speaks about one legal candidate. A candidate is legal when the speech channel is free and it was not announced in the last 2.8 s.

**Budget.** The long-run cue rate must stay at or below B = 2 cues/min.

**Events and loss.** Each annotated event e has a window [s_e, f_e]:
- s_e is the first matching detection, at most 10 s before the question time t_q;
- f_e = t_q + 0.5 s.

The event is a hit if a cue about e falls inside its window; otherwise it is a miss. A quiet event (the user asked and the answer was "nothing there") is violated by any cue in [t_q − 4, t_q + 0.5]. A cue within 6 s of a labelled moment that is not the first timely mention of an event is a false alarm.

**Feedback (one-sided).**
- **Tap after a cue.** It arrives with probability p_tap = 0.5. Its label (+1 useful, −1 not useful) comes from the event windows and applies to the spoken candidate. This is the apple-tasting channel.
- **Missed question.** At t_q, a question about an unannounced event gives y = +1 to every remembered candidate of that event's track in [s_e, t_q], with total weight 1 split evenly. The label is therefore delayed by up to about 10.5 s and attached to a window.
- **Quiet question.** Gives y = −1 to all remembered candidates in the quiet window.
- **Anything else.** Silence and cues far from labelled moments get no label.

By default every annotated question is asked (p_ask = 1). For diagnosis, each question (about an event or a quiet window) can instead be asked with probability p_ask, using its own seeded random generator so the tap stream does not change.

**Relation to known settings.** Without questions, this is apple tasting with a knapsack constraint. With questions, the positives after silence resemble user-triggered supervision (Agarwal et al.), with added delay and ambiguity about which candidate gets the credit. We do not claim a regret bound. Which partial-monitoring class this game falls into is open (unverified).

## 4. Method

**Policy.** P(should speak | x) is an online logistic model. Each step the system picks the legal candidate with the highest score and speaks if that score is at least τ.
- **Features:** Welford running z-score, clipped to ±5.
- **Update:** AdaGrad with learning rate 0.5 and L2 1e-3.
- **Thompson sampling:** a diagonal Laplace posterior.

The model persists across videos.

**Learners.**
- **Greedy.**
- **Budgeted apple tasting.** 25% of the cue budget goes to exploration below τ. Importance weights are capped at 10.
- **Thompson sampling.**
- **No-learning floor (lr = 0).** Greedy with learning rate 0. The weights stay 0, so every candidate scores 0.5 and the policy names the first legal candidate whenever the budget controller lets it speak. It is a rate-controlled, first-listed-candidate policy, not a content prior, and it is identical on every seed.
- **Full-information references (not deployable).** Every candidate at every round is labelled from the event windows. The learner also receives the one-sided signals. The labels are buffered and released after a delay d (`full_info_delay` in `seeforme/online/policy.py`):
  - **zero delay:** labels for round t are applied before the decision at t + 0.5 s, which scores nearly the same candidate in the same event window. This reference is not causal;
  - **d = 2 s:** labels released 2 s later, still inside the clip;
  - **video end (causal):** labels released when the clip ends, so they can only help later clips.

**Budget controller.** Dual ascent on logit(τ) with step η = 0.3, τ_0 = 0.6, and τ clipped to [0.01, 0.999].

**Variants.**
- **Simulator warm start.** One full-information pass over 84 CampusBench v2 training scenarios (about 17,400 labelled candidates, 1,850 positive). Either the whole optimiser state is carried over (AdaGrad accumulator, posterior precision, normaliser) or only the weights, as a prior mean.
- **Implicit negatives.** A remembered candidate that no feedback covered by the time it leaves the 15 s lookback becomes a negative of weight w.

**Memory.** CLIP ViT-B/32 appearance re-identification keeps dead confirmed tracks for 120 s. A new track of the same label with cosine similarity ≥ θ inherits the old track's announced state.

**Re-announcement guard (frozen IL scheduler, opt-in).** A track cannot be spoken about again within g seconds of its last announcement unless it has come at least 3 m closer, the same exception P-CACS uses. We use g = 10 s, not tuned.

## 5. Experimental setup

**Data.** EgoBlind (arXiv 2503.08221), used under its research licence. We split the development pool by source recording:

| Split | Videos | Minutes | Events | Coverable events |
|---|---|---|---|---|
| train | 454 | 230.9 | 237 | 133 |
| finetune | 104 | 59.3 | 56 | 27 |
| select | 34 | 43.5 | 27 | 11 per seed |

The 133 coverable events on train is derived as 160 − 27.

Holdout and the official test split are never opened. The development stream is train then finetune: 558 videos, 290 minutes, 160 coverable events, 111 proxy users (one per source recording), ordered by recording and then by clip. "Coverable" means the detector produced a matching track. About 55% of events (160 of 293) are coverable.

**Validity caveat on questions.** EgoBlind's questions were mostly generated and then verified, or written afterwards. At most about 541 of the 5,311 come from the wearer in situ (pre-filter count, arXiv 2503.08221v4). Our miss signal is therefore a replay of an annotated question process, not of users' own asking. Taps are simulated.

**Baselines.**
- P-CACS: a fixed utility rule.
- October IL: a frozen imitation-learned scheduler trained on train, so it is in-sample there.
- October RL: a frozen RL scheduler.
- The no-learning floor (section 4).

**Metrics.**
- Coverable recall (hits / coverable events).
- Quiet-window violation rate.
- Cues per minute.
- Label repeats per minute, and the share of cues that repeat the same tracker track.
- Recall on the out-of-sample finetune part.

Evaluation uses progressive validation (predict, then update). Each learner was run with 3 seeds. Paired bootstrap 95% CIs (2,000 resamples) resample videos with seeds pooled per video (`scripts/bootstrap_ci.py`).

**Selection.** The implicit-negative weight was chosen on select (pooled over 3 seeds, 33 event-seeds). The memory threshold θ is still to be chosen on select. The p_ask sweep was run on dev for diagnosis only, and nothing was selected from it.

## 6. Results

### 6.1 Main comparison (development stream)

| Policy | s0 | s1 | s2 | Mean | Quiet violations | Cues/min (s0) |
|---|---|---|---|---|---|---|
| P-CACS (fixed) | 0.087 | | | | 0.02 | 0.57 |
| October RL (frozen) | 0.050 | | | | 0.00 | 0.03 |
| October IL (frozen, in-sample on train) | 0.331 overall; 0.148 (4/27) on finetune | | | | 0.04 | 0.79 |
| No learning (lr = 0) | 0.138 | 0.138 | 0.138 | 0.138 | 0.13 | 1.68 |
| Online greedy | 0.188 | 0.212 | 0.212 | **0.204** | 0.11 | 1.67 |
| Online apple tasting | 0.169 | 0.181 | 0.200 | 0.183 | 0.10 | 1.68 |
| Online Thompson | 0.225 | 0.175 | 0.163 | 0.188 | 0.10 | 1.70 |

The online learners and the floor run at about 1.7 cues/min against a target of 2. The frozen baselines speak far less (P-CACS 0.57, IL 0.79, RL 0.03 cues/min, `results/online/summary_dev.md`), so the comparisons with them are not rate-matched.

- Online learners beat the fixed rule and the frozen RL scheduler: 0.204 against 0.087 and 0.050, at a higher cue rate. CIs and rate-matched comparisons are [TO RUN] (section 8, item 1).
- Greedy beats the no-learning floor at the same cue rate: +0.067 [+0.010, +0.121]. Learning from the feedback does something, but not much.
- On finetune the learners are level with frozen IL: 2 to 7 of 27 against 4 of 27. The seed-0 Thompson result of 7/27 did not replicate (4/27 and 4/27). With n = 27, nothing there is separable.
- Quiet-window violations rise from 0.00 to 0.04 for the frozen models to about 0.10 for the online learners, because silence is never labelled.

### 6.2 Exploration does not help

Apple tasting (0.183) and Thompson sampling (0.188) are within seed noise of greedy (0.204). This agrees with Sculley's observation that explicit apple-tasting exploration can underperform greedy in practice [verify: cited second-hand; open Sculley, KDD 2007, before keeping this sentence].

### 6.3 Simulator warm start

Each comparison is the change in pooled recall against the cold start, with a paired bootstrap 95% CI.

| Learner | Mean recall | Δ vs cold start [95% CI] |
|---|---|---|
| Greedy, full state carried | 0.146 | **−0.058 [−0.107, −0.013]** |
| Thompson, full state carried | 0.150 | −0.038 [−0.089, +0.007] |
| Greedy, weights only | 0.177 | −0.027 [−0.067, +0.014] |
| Thompson, weights only | 0.206 | +0.019 [−0.016, +0.054] |

Carrying the optimiser state from about 17,000 simulated labels makes AdaGrad steps small and the posterior overconfident. The learner then cannot move away from a simulator that is wrong about real video, and quiet violations rise to between 0.10 and 0.14. A weights-only prior is neutral: every CI includes 0. Against the causal full-information reference (0.200) there is no feedback gap left for a warm start to close. This is a direct caution for simulator-first pipelines such as TIMELI's.

### 6.4 Silence as a weak negative

Select counts are hits pooled over 3 seeds on 11 distinct events.

| w | Greedy dev | Greedy select | Thompson dev | Thompson select |
|---|---|---|---|---|
| 0 | **0.204** | **9/33** | **0.188** | **12/33** |
| 0.02 | 0.173 | 9/33 | 0.183 | 8/33 |
| 0.05 | 0.169 | 9/33 | 0.175 | 7/33 |
| 0.1 | 0.169 | 7/33 | 0.175 | 7/33 |
| 0.3 | 0.154 | 5/33 | 0.160 | 5/33 |

Recall falls steadily as w grows on the dev stream. Unasked candidates share label and moment with the rare positives, so negatives push down exactly the features those positives rely on. This is consistent with the bias analysis for one-sided feedback in Jiang et al. The select differences rest on 11 events and carry no statistical weight.

### 6.5 Where is the bottleneck? Label delay against one-sidedness

Dev stream, 3 seeds, pooled over 160 coverable events per seed.

| Learner | s0 | s1 | s2 | Pooled | Quiet violations | Select (pooled) |
|---|---|---|---|---|---|---|
| Full information, zero delay (not causal) | 0.525 | 0.500 | 0.512 | 246/480 = 0.512 | 0.096 | 17/33 |
| Full information, labels held 2 s | 0.362 | 0.388 | 0.388 | 182/480 = 0.379 | 0.091 | 15/33 |
| Full information, labels held to video end (causal) | 0.200 | 0.206 | 0.194 | 96/480 = 0.200 | 0.116 | 6/33 |
| One-sided greedy | 0.188 | 0.212 | 0.212 | 98/480 = 0.204 | 0.115 | 9/33 |
| No learning (lr = 0) | 0.138 | 0.138 | 0.138 | 66/480 = 0.138 | 0.129 | 3/33 |

Every policy runs at 1.66 to 1.68 cues/min on dev. Paired video bootstrap, dev:

| Comparison | Δ recall [95% CI] |
|---|---|
| Causal full information minus greedy | −0.004 [−0.054, +0.048] |
| Zero-delay full information minus greedy | +0.308 [+0.244, +0.371] |
| Full information held 2 s minus greedy | **+0.175 [+0.112, +0.239]** |
| Greedy minus no learning | **+0.067 [+0.010, +0.121]** |

On select, causal minus greedy is −0.091 [−0.267, +0.067]. The pre-registered falsifier for "a real one-sided gap exists" (causal full information ahead by more than 0.08 on select, with the interval above 0) is not met. Select has 11 events per seed, so this is indicative only.

What this says:
- **The zero-delay reference is not an upper bound.** Its labels for round t are applied before the decision at t + 0.5 s, about nearly the same track in the same event window. Real labels cannot arrive that fast.
- **Labels arriving after the clip add nothing over taps and questions.** Perfect per-candidate labels, released at clip end, tie one-sided greedy. Because every question is asked (p_ask = 1) and scoring happens only near questions, every scored outcome is eventually revealed, so on the scored region the problem is close to delayed full information.
- **Within-clip label latency matters.** The same labels released 2 s late, still inside the clip, add +0.175 with an interval that excludes 0. What is worth having is feedback fast enough to act on within the same event.

Further diagnostics. These come from scratch scripts outside the repo:
- Learning rate does not matter on seed 0: lr 0.1 gives 0.181 and lr 0.02 gives 0.194 (prelim.).
- Relabelling missed questions in hindsight gave 0.156 and 0.175 on seed 0 (unverified: no log holds these numbers).
- Choosing which candidate to name is easy: at rounds that contain a positive, picking at random is right 83.9% of the time, and the model is right 83.5% (prelim.).
- Deciding when to speak is hard. A logistic model trained on train reaches AUC 0.675 on finetune (train 0.728) (prelim.). A decision-level "is now a speaking moment" model was reported at 0.683 (unverified: no script computes it). Gradient boosting overfits, at train about 0.97 and finetune about 0.62, varying by run because of its random early-stopping split (unverified until rerun with a fixed seed). The single strongest feature is the user's own walking speed (finetune AUC 0.691, prelim.).

### 6.6 Users who do not always ask

Questions dropped with probability 1 − p_ask, dev only, 3 seeds, for diagnosis.

| p_ask | Greedy s0 / s1 / s2 | Greedy mean | Causal full information (mean) | Greedy missed-question label rows (s0) |
|---|---|---|---|---|
| 1 | 0.188 / 0.212 / 0.212 | 0.204 | 0.200 | 3,055 |
| 0.5 | 0.194 / 0.138 / 0.169 | 0.167 | 0.185 | 1,779 |
| 0.25 | 0.194 / 0.156 / 0.188 | 0.179 | 0.179 | 688 |

Cues/min is 1.66 to 1.69 in every run. Lowering the asking rate to 0.5 or 0.25 lowers greedy by 0.037 and 0.025 on the mean, a change we cannot separate from seed noise: individual seeds range from 0.138 to 0.212, and the mean at 0.5 is below the mean at 0.25. The pre-registered falsifier (greedy at p_ask 0.25 within 0.03 of p_ask 1) is met. We did not fit a positive-unlabelled correction, because there is no gap for it to recover. Under this simulator, removing three quarters of the miss signal does not measurably hurt, which again points away from the feedback channel and toward the features.

### 6.7 Appearance memory and repeat cues

| Policy | Re-ID | Coverable recall | Label repeats/min | Δ recall [95% CI] | Δ repeats/min [95% CI] |
|---|---|---|---|---|---|
| P-CACS | off / θ 0.80 | 0.087 / 0.087 | 0.214 / 0.183 | +0.000 | **−0.031 [−0.064, −0.004]** |
| October IL | off / θ 0.85 | 0.331 / 0.331 | 0.338 / 0.338 | +0.000 | +0.000 |
| Online greedy | θ 0.80 | 0.125 | 0.545 | **−0.062 [−0.117, −0.012]** | +0.048 [−0.071, +0.171] |
| Online greedy | θ 0.85 | 0.188 | 0.472 | +0.000 [−0.048, +0.044] | −0.024 [−0.104, +0.055] |

The online greedy baseline without re-ID is 0.188 recall and 0.496 repeats/min. CLIP embeddings were computed for 414,017 detections across 592 videos.

- Match precision against ground-truth object ids is 0.90 at θ 0.80 (n = 2,041) and 0.89 at θ 0.85 (n = 1,863).
- Clips average 31 s, so announced objects rarely leave and return. The memory changes little.
- 28.5% (65/228) of IL cues name an already-announced weak event. Only 13.6% (31/228) repeat the same tracker track, because EgoBlind links every same-label detection in an event window to that event, so the larger count includes cues about a different person or car. The cause of the same-track repeats is the learned scheduler's 2.8 s per-track block with no novelty decay, not a memory failure. A 10 s re-announcement guard, not tuned, cuts same-track repeats to 7.9% (17/214) with recall unchanged (0.331; 4/27 on finetune) and cues/min falling from 0.79 to 0.74 (`results/memory/reid_il_fix.md`).

## 7. Discussion

1. **The one-sidedness of feedback is not what limits learning on this benchmark.** A causal full-information reference ties one-sided greedy, and dropping three quarters of the questions changes nothing we can separate from noise. The zero-delay reference that once suggested a large gap measured something no deployed system can have.
2. **Within-clip label latency is.** Labels 2 s late inside the clip are worth +0.175. Feedback that arrives while the event is still happening, and a model that can use it, is where the measurable value lies. How fast real users tap or ask is therefore the first thing a deployment study must measure.
3. **Features are the other limit.** Choosing which candidate to name is easy; choosing when to speak is hard, with offline AUC about 0.68 (prelim.). Features that predict *when* a person needs information (walking speed, stopping, scene change) are the next lever.
4. **Learning during use beats fixed and offline-RL timing, and a no-learning floor, but not a good offline imitation model out of sample.** It is not yet a better scheduler; it is a way to adapt without offline labels.
5. **The obvious fixes (exploration, simulator priors, silence-as-negative) do not help, and two of them hurt.** These are actionable negative results for anyone building proactive BLV assistants.
6. **Benchmarks for proactive assistance need a causal reference and a label-delay curve.** Without them, a zero-delay full-information learner can be mistaken for an upper bound on what feedback can buy.

## 8. Experiments not yet run [TO RUN], with exact protocols

Done since v0, and reported above: the causal full-information reference and the 2 s delay (section 6.5), the no-learning floor (sections 6.1 and 6.5), and the p_ask sweep (section 6.6).

All remaining items run on the dev stream (train then finetune) with 3 seeds, and selection happens on select only. Holdout and test are opened exactly once, after every choice is frozen.

1. **CIs for section 6.1.** Paired bootstraps (online against P-CACS, RL and IL), on dev overall and on the finetune slice. Compare at matched rates (±0.1 cues/min), since the frozen baselines speak at 0.03 to 0.79 cues/min.
2. **Label-delay curve.** Run `full_info_delay ∈ {0.5, 1, 2, 5, 12}` s and video end, 3 seeds, to locate where within-clip labels stop helping. The earlier seed-0 12 s value is scratch-only and is not reported.
3. **Move the scratch diagnostics into the repo.** Regenerate the decision-level AUC, the hindsight-relabelling runs, the gradient-boosting AUC with a fixed `random_state`, the learning-rate runs, the top-1 candidate accuracy and the ego-speed AUC. Also put the guard bootstrap and the 18 / 26 / 21 repeat breakdown into `eval_reid.py` output.
4. **Decision-level "when" features.** Ego-speed change over 1 s and 3 s, a stopped flag, time since the last cue, time since clip start, new tracks in the last 2 s, and maximum growth and TTC across tracks. Gate on offline decision-level AUC (train → finetune) of at least +0.02 over a baseline regenerated in the repo (item 3). Then run online and choose on select at equal cues/min.
5. **In-situ question slice.** Only if the EgoBlind annotation file marks question source (unverified). Report recall on wearer-asked questions separately.
6. **Raw recall.** Including events the detector cannot see, for every policy.
7. **Learning curves.** Recall against the cumulative number of feedback events, within each proxy user and over the whole stream. Report empirical regret against the best fixed threshold in hindsight on train, with no bound claimed.
8. **Formal baselines.**
   - An Agarwal-style user-triggered-supervision learner.
   - A Jiang-style adaptive UCB.
   - A WorldScribe-style trigger rule.
   - A weights-only TIMELI-pretrained warm start, if weights are available (unverified).
9. **Label quality (M3).** A blind human relabel of the weak events, with reviewer CSVs stripped of model columns (`bin:*`, `max_score:*`). Report Cohen's kappa.
10. **Re-ID.** Hand-check 50 sampled matches, choose θ and the horizon on select, and repeat on long continuous recordings (M5). Tune the re-announcement guard length on select, and replace the fixed 3 m approach exception, which is weak at far range.
11. **Power analysis.** Before the one-shot holdout run: the minimum detectable recall difference at the holdout's coverable-event count.
12. **One-shot final.** Holdout once, then the official test once, for the frozen configuration.
13. **External test.** VIABench, if its data and licence are released.
14. **BLV pilot (IRB).** 6 to 10 BLV participants walk a fixed route with a chest-mounted phone and real Useful / Not useful / Quiet controls. Log taps, questions and quiet requests with timestamps. Measure the delay between an event and the user's feedback (the quantity section 6.5 identifies as decisive), the real asking rate (which calibrates p_ask), tap rate, cue latency tolerance, and preferred speech rate and direction format.

## 9. Limitations

- **Simulated feedback.** Taps use p_tap = 0.5 with oracle usefulness. Questions are replayed from annotations, most of which were not asked in situ. No BLV person has given feedback to the system. The finding that one-sidedness does not bind holds for this simulator; real users may tap less, later or inconsistently.
- **Small n.** There are 160 coverable dev events. Finetune has 27 and select has 11 per seed, so select-based choices are not statistically supported.
- **Short clips.** At 31 s on average, within-user learning and long-horizon memory cannot really be tested. Proxy users are source recordings, not people.
- **Metric generosity.** Coverable recall drops the roughly 45% of events the detector cannot see. Ground truth is what users asked about, which is broader than what should be announced.
- **Quiet-window violations** for the online learners (about 0.10) are about 2.5 to 5 times those of P-CACS (0.02) and IL (0.04). RL has none, but it speaks only 0.03 cues/min (`results/online/summary_dev.md`). None of these comparisons is rate-matched.
- **Preliminary and unverified numbers.** The diagnostics in section 6.5 come from scratch scripts outside the repo; three of them have no log at all.
- **No theory.** We prove no regret bound for the delayed, interval-labelled setting.

## 10. Ethics and data

- **Participants.** No human-subjects data was collected for this paper. The pilot in section 8 requires Yale IRB approval, which has not yet been sought; its status should be stated by the author before submission. The protocol will include informed consent in an accessible format, a sighted spotter and cane or guide dog use throughout, the right to stop at any time, face blurring of bystanders, and no raw video in version control or cloud storage.
- **Safety framing.** SeeForMe supplements the cane or guide dog and does not replace them. A missed cue must never be the only safeguard.
- **Data licences.**
  - EgoBlind is used under its non-commercial research licence. Its videos are public uploads by blind creators. We redistribute no video or derived features, and models trained on it stay in a research-only track.
  - GuideDog (gated; CC BY-NC unverified) is not used in these results.
  - BLV-Road-Nav has no licence file and is not used until permission is granted.
  - A product track will be trained only on the authors' own recordings, CampusBench and permissively licensed detectors.
- **Representation.** EgoBlind creators are not a representative sample of BLV pedestrians (country, mobility aid, phone placement). Results must not be generalised to all BLV users.

## 11. Future work and ML-venue variant

Future work: run the BLV pilot and measure real feedback latency, collect long continuous walks, add anticipation features, and build a native iOS client with real haptics and a fast in-event feedback control.

**ML variant** (e.g., a workshop on online learning or human feedback). Lead with the methodological finding: a zero-delay full-information reference overstates the cost of one-sided feedback, and a causal reference plus a label-delay curve shows that, on this benchmark, the value of labels lies in their latency within an event, not in seeing the outcomes of silence. Add the full delay curve (section 8, item 2), a formal statement of the game, and an open-problem section on its partial-monitoring class.

---

## Reviewer-style self-assessment (out of 100)

Scored against the corrected framing: a diagnostic benchmark whose main result is negative.

| Criterion | Score /20 | Reason |
|---|---|---|
| Novelty | 9 | Timing (TIMELI, VIABench, WorldScribe, EgoVoice) and one-sided learning (apple tasting, Agarwal et al.) are prior art. The new part is a diagnosis on real BLV video: the one-sided cost vanishes against a causal reference, and within-clip latency carries the value. That is useful but negative, and it does not yet come with a method that exploits it. |
| Soundness | 12 | The causal reference, delay, floor and p_ask results are in the repo with 3 seeds, bootstrap CIs and pre-registered falsifiers. Split hygiene is good. Feedback is still fully simulated with oracle taps. Several diagnostics are scratch-only and three are unverified. The quiet-violation regression is unexplained. No theory. |
| Evaluation | 8 | n = 27 out of sample and 11 per seed on select. No CIs or rate matching for the frozen-baseline comparison. Only one within-clip delay point. No holdout or test result, no raw recall, no in-situ slice, no real users. |
| Clarity | 15 | The formulation, protocols and corrections are explicit, and the negative results are stated plainly. A system figure, a feedback-timeline figure and a delay curve are still needed. |
| Accessibility relevance | 12 | The problem is well motivated by BLV studies, and the finding points to a concrete design question (how fast users can give feedback). No BLV participant has been involved, and ASSETS reviewers weigh that heavily. |
| **Total** | **56 / 100** | Workshop or late-breaking level today. |

### What must be done to reach 100/100

**Novelty (→ 20)**
1. Turn the diagnosis into a method: show that a fast in-event feedback signal, or a when-feature model, beats both one-sided greedy and frozen IL out of sample with a CI excluding 0.
2. Measure the full label-delay curve and show where within-clip labels stop helping, with CIs.
3. State the game formally and prove or bound regret for the delayed, interval-labelled, budgeted instance, or cite a verified result for its partial-monitoring class.

**Soundness (→ 20)**
4. Regenerate every scratch-only and unverified diagnostic in the repo (section 8, item 3), or remove it.
5. Explain and reduce the quiet-violation regression, for example with a quiet-request control in the feedback model.
6. Replace oracle taps with measured tap behaviour from the pilot, and calibrate p_ask and feedback delay from real users. Re-test the "one-sidedness does not bind" finding under those values.
7. Fix the remaining repo wording: `docs/online.md` says halving or quartering the questions moves greedy "by 0.03 or less", but p_ask 0.5 moves it by 0.037; `.claude/prds/seeforme.prd.md` line 5 on what deployed apps do; the seed-0-only header of `results/online/summary_dev.md`; and the orphaned list item 5 in `docs/memory.md`.

**Evaluation (→ 20)**
8. Add bootstrap CIs to every comparison and compare with the frozen baselines at matched cue rates.
9. Run a power analysis, then the one-shot holdout and official test with frozen configurations.
10. Run the in-situ question slice and raw recall.
11. Do the human relabel (M3) with kappa.
12. Run the formal baselines (Agarwal-style, Jiang adaptive UCB, WorldScribe-style rule, TIMELI weights-only).
13. Add long continuous recordings (M5) for per-user learning curves and long-horizon memory.
14. Run an external test on VIABench, if released.

**Clarity (→ 20)**
15. Add a system figure, a feedback-timeline figure (cue, tap, question, window), the label-delay curve, learning curves, and a recall against cues/min frontier.
16. Open and verify every [verify] and (unverified) citation and complete the bibliography with DOIs.

**Accessibility relevance (→ 20)**
17. Run an IRB-approved BLV pilot (6 to 10 participants) with real feedback controls in the app. Report feedback latency, learning within a session, SUS or NASA-TLX, interviews, and preferences for latency, rate and direction format.
18. Ship the feedback channel in the deployed client, and replace the web client's current default scheduler (the CampusBench RL model, `models/scheduler/scheduler_rl.pt`, near zero on earlier EgoBlind evaluations) with the best-measured policy. This is owned by the web session; this track does not edit `web/`.
19. Co-design or review with O&M instructors, and report participant demographics and mobility aids.

---

### Items for the user
- `docs/ecc/ECC-OPERATING-MANUAL.md` sits in the public repo and contains a local home-directory path.
- The advisor naming differs between README line 3 and `update.md` line 4. Confirm it; do not guess.
- IRB status must come from the author.
- `.claude/prds/seeforme.prd.md` line 5: check that its description of deployed apps matches "wait to be asked or speak by fixed, content-gated rules".
- The web session has not been told that its default scheduler is the CampusBench RL model, that OCR text goes into innerHTML unescaped (`main.ts` lines 312 to 315), or that `styles.css` line 2 forces a white background in dark mode.
- The research/next changes (code, tests, `results/online/*causal*`, `*pask*`, `results/memory/reid_il_fix.*`) are uncommitted and need review.
