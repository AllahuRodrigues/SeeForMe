# SeeForMe scientific gap: prior work, gap, novelty, risks (research lead report, 2026-11-12)

Method: I read the repo docs you listed (online.md, memory.md, PRD, M1/M2 plans, README, timeli_notes.md, walkvlm_lr_notes.md) and checked papers with web search and arXiv abstract or HTML pages. I mark each claim with how far I checked it. "Abstract" means I only read the abstract page. "Unverified" means I could not confirm it. I wrote no files and changed nothing in git.

## Bottom line

1. **"When to speak unprompted for BLV users" is no longer an open problem as stated.** Since mid-2026, three papers address it directly: TIMELI (Sep 2026), VIABench's Proactive Reminder task (Jul 2026) and WorldScribe (UIST 2024). EgoVoice (Oct 2026, EMNLP 2026) and Vinci2 (ECCV 2026) do the same for general wearable assistants. A reviewer will reject a claim of novelty for the timing problem itself.
2. **The PRD sentence that Be My Eyes, Seeing AI and Lookout "answer only when asked" is factually wrong for two of them.** Seeing AI's Short Text channel reads text as soon as it appears. Lookout's Explore mode describes objects continuously as the camera moves. The defensible version: deployed apps either wait for a request or speak by fixed rules, and none of them learns when to speak.
3. **What still looks novel** (from what I could find; I did not run an exhaustive SLR):
   - learning when to speak for BLV users online, during use;
   - from one-sided feedback: taps after a cue, a later user question exposing a miss, silence never labelled;
   - under a speech budget;
   - evaluated on real first-person BLV video.

   Every BLV timing paper I found learns offline:
   - from simulator rules (TIMELI);
   - from fixed annotations (VIABench, GuideDog);
   - from offline preference data (EgoVoice uses DPO);
   - from hand-set thresholds (WorldScribe).

   None updates from the user's own reactions while in use.
4. **The theory half has close precedents.** These are apple tasting, label-efficient prediction, one-sided feedback in lending, and especially "user-triggered supervision" in contextual bandits (Agarwal, Gentile, Marinov 2023). The novelty has to be the problem instance and its measured structure, not a new learning setting. The specific structure is: the miss signal arrives late, credit assignment is ambiguous, there is a budget, and the negatives are biased. Claim it as a new instance; do not claim a new setting.
5. **The current numbers do not support the PRD thesis claim.** The claim needs online learners to beat frozen IL with a bootstrap CI excluding 0. Today:
   - IL scores 0.331 overall (in-sample on train), against 0.183 to 0.204 for the online learners.
   - On the out-of-sample finetune split, IL scores 4/27 and the online learners 2 to 7/27, which is not separable.

   What is currently strong and publishable is the measured cost of one-sided feedback: 0.525 with full information against about 0.20 one-sided. That is a gap of about 0.32, about 60% of the learnable recall. Negative results also hold: warm-start harm and implicit negatives hurt.

## 1. What is done (with citations)

### A. When to speak or stay silent, BLV specific (closest competitors)

- **TIMELI, "Time-Aware Assistive Navigation"**, Kuribayashi, Shangguan, Ohn-Bar. Source: https://arxiv.org/abs/2609.05596 (abstract) and repo `docs/timeli_notes.md`.
  - Asks "can agents learn when to speak, not just what to say". Silence is an explicit target label ("None").
  - Training data is CARLA simulation, about 67K clips, with a rule-based ground truth (`determine_to_talk`). There is a small real YouTube test set.
  - Supervised MLLM fine-tuning plus reason prediction. No user feedback, no online learning, and no BLV user study in the abstract.
  - Venue inconsistency: arXiv says "ICRA 2026" but it was submitted 4 Sep 2026. Unverified which ICRA.
- **VIABench**, Liu et al. (Nanjing, Limin Wang group). Source: https://arxiv.org/abs/2607.14660, HTML read.
  - 761 videos, 46.9 h; 530 real videos from visually impaired creators on YouTube, Bilibili and DouYin.
  - Its **Proactive Reminder** task annotates [ts, te] event intervals and scores a detection rate (an alert anywhere in the interval) plus GPT-5 similarity of the content.
  - It proposes TPAD, a training-free per-frame "should the user be warned? A/B" probability.
  - Offline models stay under 45% recall. LiveCC gets 75.8% by talking densely, which shows the benchmark does not penalise verbosity well.
  - No user feedback, no budget. Data release is promised ("will be released"); dataset licence not stated.
- **WorldScribe**, Chang, Liu, Guo, UIST 2024 Best Paper. Source: https://arxiv.org/abs/2408.06627, HTML read.
  - Gives continuous, unprompted live descriptions to blind users.
  - Hand-set triggers: 30 degree orientation change, frame similarity below 0.6, stability over 5 frames.
  - Adjusts verbosity by scene dynamics, ranks by stated intent, drops near-duplicates, and pauses when it hears speech.
  - No learning from feedback. Formative study n=5, evaluation n=6 BVI participants. Not built for navigation.
- **WalkVLM** (https://arxiv.org/abs/2412.20903) and **WalkVLM-LR, "Less Redundancy"** (https://arxiv.org/abs/2508.16070, ICASSP 2026 per repo notes).
  - Streaming walking reminders. An "environment awareness discriminator" gates when to speak, and GRPO with four reward functions targets less redundant output.
  - Offline. The WAD data has no stated licence (repo `docs/walkvlm_lr_notes.md`).
- **ForeSightGuide** (https://arxiv.org/abs/2608.18993, abstract): uses a VLM to anticipate hazards and filter out objects that pose no threat. Reports 0.299 redundant alerts per output and a 0.112 missed-hazard rate. Offline, no feedback.
- **Sighted by Default / VIA-Agent** (https://arxiv.org/abs/2511.00945, UIST 2026, abstract): a "Continuity" principle that guidance should arrive when actionable. Within-subjects n=9; BLV status of participants not stated in the abstract. Timing mechanism not described in the abstract.
- **ChatGPT live video for BVI users**, Chang et al., ASSETS 2025 (https://arxiv.org/abs/2508.03651, abstract): n=8. The system falls short on the live descriptions dynamic scenes need. The authors call for intervention timing beyond turn-taking. This is direct motivating evidence for SeeForMe.
- **Ohn-Bar line on personalisation**:
  - PING, CoRL 2018 (https://arxiv.org/abs/1804.04118): online, weighted-experts adaptation of a model of each blind user's dynamics.
  - BU final report (https://www.bu.edu/cise/learning-from-interactions-with-blind-users-for-customized-and-scalable-navigation-assistance-systems-final-report/): remote-guidance study with 13 people, 10 with visual impairments. Guides adjusted timing and content per walker. Meta-imitation personalisation "can also worsen performance for a subset of the participants". A participant quote: "ask each person how much information they would like."

  This lab (BU) is the most likely group to publish online, feedback-driven timing next.

### B. Proactive egocentric assistants, not BLV

- **EgoVoice** (https://arxiv.org/abs/2610.12248, EMNLP 2026, abstract): decides whether, when and what to say from first-person audio and video (HoloAssist). Fine-tuning plus offline DPO.
- **Vinci2 / EgoServe / EgoMemo** (https://arxiv.org/abs/2607.11523, ECCV 2026, abstract): training-free, memory-augmented decision on whether to intervene; more than 3,000 service instances. Its framing: existing benchmarks assume an explicit instruction upfront.
- **Proactive Assistant Dialogue Generation from Streaming Egocentric Videos** (Zhang et al., https://arxiv.org/abs/2506.05904; listed in ACL Anthology as EMNLP 2025 at https://aclanthology.org/2025.emnlp-main.605/): synthetic dialogues, a speak or stay-silent decision, class imbalance named as a difficulty.
- **EgoPro-Bench** (https://arxiv.org/abs/2605.07299, abstract): personalised proactive interaction from simulated user profiles. The abstract does not mention EgoBlind. A search snippet claimed it uses 420 EgoBlind videos and a navigation domain for visually impaired users; unverified.
- **YETI** (https://arxiv.org/abs/2501.09355): proactive intervention in AR tasks. Title only checked.
- **ProactiveAgent**, Lu et al., ICLR 2025 (https://arxiv.org/abs/2410.12361): a reward model trained on human accept or reject labels of proposed help. Offline and two-sided labels. Its F1 numbers came from a third-party summary; unverified.
- **Project Astra with Aira "Visual Interpreter"** (https://aira.io/projectastra, https://deepmind.google/models/project-astra/): Google markets Astra as able to start conversations. A pilot for BLV users is underway, with human interpreters monitoring calls. No public method. This is a commercial threat to "nobody does this".

### C. Interruption management (HCI)

From a curated list (https://github.com/tao-hpu/awesome-proactive-agents); I did not open each paper:

- Horvitz, Jacobs and Hovel, Attention-Sensitive Alerting, UAI 1999: https://arxiv.org/abs/1301.6707
- Horvitz, Principles of Mixed-Initiative UI, CHI 1999: https://doi.org/10.1145/302979.303030
- McFarlane 2002: https://doi.org/10.1207/S15327051HCI1701_2
- Fogarty et al., TOCHI 2005: https://dl.acm.org/doi/10.1145/1057237.1057243
- Pielot et al., Beyond Interruptibility, IMWUT 2017: https://dl.acm.org/doi/10.1145/3130956
- Okoshi et al., KDD 2019, adaptive notification scheduling in production: https://doi.org/10.1145/3292500.3330732

This literature already learns interruptibility from behaviour. The SeeForMe difference: the cost of silence is a missed safety or awareness event, not lost engagement, and the user cannot see what was missed.

### D. Learning from one-sided or implicit feedback (theory)

- **Apple tasting**: Helmbold, Littlestone, Long, FOCS 1992 and Information and Computation 161 (2000). Indexed in bibliographic catalogues found by search; I did not read the paper. Agnostic characterisation by Raman, Subedi, Raman, Tewari (https://arxiv.org/abs/2310.19064, COLT 2024 per search).
- **Label-efficient prediction**, Cesa-Bianchi, Lugosi, Stoltz, IEEE Transactions on Information Theory 2005 (https://hal.archives-ouvertes.fr/hal-00007537). **Partial monitoring**, same authors, Mathematics of Operations Research 2006 (search result only).
- **One-sided feedback in lending**:
  - Jiang, Jiang, Pacchiano, AISTATS 2021 (https://arxiv.org/abs/2006.04858). Without bias correction a GLM can converge suboptimally or not at all; they propose an adaptive UCB fix.
  - Bechavod et al., NeurIPS 2019 (https://arxiv.org/abs/1902.02242).
- **User-triggered supervision in contextual bandits**, Agarwal, Gentile, Marinov 2023 (https://arxiv.org/abs/2302.03784, abstract). The user sometimes volunteers the best action, as in autocomplete, and the method stays robust to the bias of that signal. **This is the closest formal analogue to "a question reveals a miss".**
- **Online positive-unlabeled (PU) learning**, Zhang et al., IJCAI 2020 (https://www.ijcai.org/proceedings/2020/311). Silence-as-unlabelled is a PU problem.
- **Budget**: Resourceful Contextual Bandits / bandits with knapsacks, Badanidiyuru, Langford, Slivkins, COLT 2014 (https://arxiv.org/abs/1402.6779).

### E. Egocentric BLV datasets (no timing or feedback labels)

- **EgoBlind**, NeurIPS 2025 D&B (https://arxiv.org/abs/2503.08221, HTML v4 read).
  - 1,392 videos (1,329 clips of about 40 s), downloaded from blind creators on Bilibili and TikTok.
  - 5,311 questions; 14.1% navigation and 23.7% safety (my arithmetic from Table 1).
  - Best MLLM about 60% against 87.4% for humans.
- **GuideDog** (https://arxiv.org/abs/2503.12844, ACL 2026 per Anthology): about 22K image-description pairs, about 2K gold. CC BY-NC 4.0, gated on Hugging Face. Still images, no timing.
- **ORBIT**, ICCV 2021 (https://arxiv.org/abs/2104.03841): 3,822 videos of 486 objects filmed by BLV users. Teachable recognition, not timing.
- **VizWiz**, Gurari et al., CVPR 2018: venue not re-verified in this pass; the ORBIT paper describes it as 31,173 images from 11,045 BLV users. Single photos with questions, entirely user-initiated.
- **Ego4D**: general egocentric data; I found no Ego4D-based BLV timing work.

## 2. What is NOT done (as far as I found)

1. No BLV system or paper learns when to speak online from the user's own reactions while in use. All of them are offline-supervised, preference-optimised, rule-based or training-free.
2. No work measures how much proactive-cue recall is lost to one-sided feedback against full labels, on real BLV video. SeeForMe's 0.525 against about 0.20 appears to be the first such measurement.
3. No BLV timing benchmark imposes an explicit speech budget. VIABench rewards dense output (LiveCC gets 75.8% recall by talking a lot), and TIMELI's rule hard-codes "never two in a row".
4. No benchmark uses later user questions as delayed "miss" labels for the timing policy. EgoBlind questions are used only as VideoQA targets.
5. No work tests whether simulator pretraining helps or hurts online adaptation on real BLV video. SeeForMe's warm-start result (-0.058, CI excludes 0, with optimiser state carried over) is a useful negative result against TIMELI-style sim-first pipelines.
6. No real-deployment learning curve for each user exists anywhere. SeeForMe does not have one either; it uses proxy users.

## 3. The precise gap and whether the claim is novel

**Reframed gap, defensible:** proactive BLV assistants now decide when to speak, but they learn that decision offline from simulator rules or annotators. In real use, the only signal is one-sided:
- a tap after a cue;
- a later question after a miss;
- nothing after silence.

SeeForMe is, to my knowledge, the first to:
- (a) formalise BLV proactive cueing as apple tasting with user-triggered miss labels under a speech budget;
- (b) measure the cost of that feedback regime on real first-person BLV video;
- (c) show that the obvious fixes fail: exploration (apple tasting, Thompson sampling), simulator warm start, and treating silence as a negative.

**Novelty verdict:**
- "When to speak" alone: **not novel** (TIMELI, VIABench, WorldScribe, EgoVoice).
- "One-sided online learning" alone: **not novel as a learning setting** (apple tasting, Jiang et al., Agarwal et al.).
- **The combination, plus the empirical characterisation on real BLV data:** novel as far as I can tell. It is a workshop or short-paper contribution today. It becomes a full-paper contribution if a method closes a meaningful part of the 0.32 gap with a CI excluding 0, or if a real BLV deployment shows within-user learning.

**Suggested thesis wording:**

> Existing proactive assistants for blind users learn when to speak from offline labels. In use, an assistant only sees one-sided feedback. We measure how much that costs on real blind users' video and develop learners that recover part of it under a 2 cue/min budget.

Drop "Be My Eyes, Seeing AI, Lookout answer only when asked". Replace it with "they either wait to be asked or speak by fixed rules (Seeing AI Short Text, Lookout Explore); none learns when".

## 4. Closest competing papers and how to differentiate

| Paper | Overlap | Differentiator to state |
|---|---|---|
| TIMELI (2609.05596) | Same question ("when to speak"), BLV navigation, explicit silence labels | Their labels come from a simulator rule, offline, two-sided. Ours: real BLV video, one-sided online feedback, budget. Our warm-start result directly tests and challenges their sim-first route. Use TIMELI as a baseline or pretraining source. |
| VIABench Proactive Reminder (2607.14660) | Real BLV video, timed events, alert-in-interval recall (essentially "coverable recall") | No budget, no feedback, rewards verbosity. Report our metric with cues/min and show that dense talkers win their metric. If their data is released, evaluate on it as an external test. |
| WorldScribe (2408.06627) | Unprompted live descriptions for blind users, deduplication, pausing | Hand-set thresholds, n=6, not navigation. Use it as the strongest rule-based baseline in the related-work argument. |
| WalkVLM-LR (2508.16070) and ForeSightGuide (2608.18993) | Reduce redundant alerts | Offline reward shaping or filtering. No user signal. |
| EgoVoice (2610.12248), Vinci2 (2607.11523), ProAssist-style 2506.05904 | Learned speak or silence decision from egocentric streams | Not BLV, offline (DPO or training-free), task-guidance domains. |
| Agarwal et al. (2302.03784) | User-triggered positive labels in contextual bandits | Ours adds delay and credit ambiguity (a question at t_q labels a window), a budget, and an empirical BLV instance. Cite it as the formal ancestor and do not claim the setting. |
| Jiang et al. (2006.04858), Bechavod et al. (1902.02242) | One-sided feedback, bias without correction | Our negative implicit-negative result is consistent with their bias analysis. Test their adaptive UCB as a learner. |
| Ohn-Bar lab (PING, BU report) | Personalising BLV guidance from interaction | Same lab as TIMELI, so the most likely to scoop. Move fast. Their meta-imitation hurt some users, which motivates our shrinkage toward global weights (S6). |

## 5. Risks to the claim (skeptical)

1. **The "questions as miss signal" assumption mostly does not hold on EgoBlind.**
   - Only 541 of 5,311 questions were asked in situ by the camera wearer.
   - The rest are GPT-4o-generated and then verified, or written afterwards by blind annotators "as if" in the scene (https://arxiv.org/html/2503.08221v4). Their timestamps are simulated.
   - So most "later question shows a miss" labels are synthetic, not user behaviour. This is the biggest threat to validity.
   - Mitigation: report results restricted to the in-situ questions (small n), and state plainly that the stream is a simulation of the feedback process.
2. **Taps are simulated.** p_tap = 0.5, and usefulness comes from oracle event windows. The "one-sided feedback" is therefore a model of users, not observed users. Without a real BLV pilot, a reviewer will call it simulation.
3. **The PRD success criterion is currently unmet.**
   - The online learners do not beat frozen IL: 0.20 against 0.331 in-sample, and 2 to 7/27 against 4/27 out of sample.
   - n=27 events on finetune and 33 on select cannot give a CI excluding 0 for realistic effects. Holdout and test may not either.
   - Do a power check before the one-shot final run.
4. **Quiet-window violations rise to about 0.10, against 0.02 to 0.04 for frozen models.** The PRD requires "not worse than frozen IL", which currently fails.
5. **Proxy users** are creators of source videos. Clips average about 31 to 40 s, and 111 proxy users cover 290 min, so "improves per user within one session" cannot really be tested on EgoBlind.
6. **Coverable recall excludes the roughly 45% of events the detector cannot see.** Reviewers may call the metric generous. Report raw recall as well.
7. **Weak ground truth.** Events are what users asked about, which is broader than what should have been announced. The M3 human relabel is needed before final claims.
8. **Scoop risk is high.** Six closely related papers appeared between May and Oct 2026 (EgoPro-Bench, Vinci2, VIABench, ForeSightGuide, TIMELI, EgoVoice). TIMELI's authors have the BLV-interaction data and motivation for an online-feedback follow-up.
9. **Theory overclaim.** "Sublinear regret against the best fixed threshold" needs a stated assumption, since one-sided feedback can break convergence (Jiang et al.). Either prove a bound for the specific delayed user-triggered setting or report regret only empirically.
10. **Licensing.** EgoBlind's source videos are creators' Bilibili and TikTok uploads. Even with CC BY-NC annotations, redistribution of derived video features may be restricted. Keep everything research-track only, as the PRD already says.

## Unverified items to check before citing

- VizWiz CVPR 2018 author and venue details.
- Apple tasting and partial-monitoring full texts.
- The DOIs on the interruption-management list (from a GitHub list, not opened).
- EgoPro-Bench's use of EgoBlind (snippet only).
- TIMELI's venue year.
- Whether VIABench's data and licence are released.
- ProactiveAgent's F1 figures.
- Whether VIA-Agent participants were BLV.
- Whether Lookout Explore still behaves as described: sources are 2019 to 2022, and Google's help page lists it as beta (https://support.google.com/accessibility/android/answer/9031274).
- Seeing AI Short Text evidence is from third-party guides (https://www.guidedogs.org.uk/services-we-provide/technology/Seeing-AI, https://hadleyhelps.org/workshops/seeing-ai-apple-voiceover-series/reading-short-text), not Microsoft documentation.

## Repo files consulted

- <repo>/docs/online.md
- <repo>/docs/memory.md
- <repo>/.claude/prds/seeforme.prd.md
- <repo>/.claude/plans/m1-online-feedback.plan.md
- <repo>/.claude/plans/m2-reid-memory.plan.md
- <repo>/docs/timeli_notes.md
- <repo>/docs/walkvlm_lr_notes.md
- <repo>/docs/research_plan.md

## Recommended immediate actions for other agents

- **Code agent:**
  - Add a "questions in situ only" evaluation slice using EgoBlind's manually extracted questions, if the annotation file distinguishes question source (unverified).
  - Add a Jiang-style adaptive UCB learner and an Agarwal-style user-triggered-supervision learner as baselines.
  - Add a TIMELI-pretrained warm start (weights only).
- **Paper agent:** rewrite the novelty claim as in section 3. Add TIMELI, VIABench, WorldScribe, EgoVoice and Agarwal et al. to related work as primary comparisons.
- **Data agent:**
  - A small real BLV pilot with real taps and real questions is the single biggest validity upgrade.
  - Long continuous walks (M5) are needed for the per-user learning-curve claim.
  - Track VIABench's data release as a possible external test set.
