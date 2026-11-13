# SeeForMe against proactive-assistance benchmarks and products (2026-11-12)

Sources were read from the papers, project pages, dataset cards and GitHub. Numbers marked **(fetch)** were extracted by a fetch tool and must be checked against the PDF before quoting.

## Headline

- No benchmark below can be "beaten" by SeeForMe today without changing domain. EgoVoice (assembly guidance) and ProactEgoMem (household and lab memory) are other domains with LLM-judged semantics. VIABench is the only one where SeeForMe could produce a comparable number, and its metric is recall-only, so a constantly talking system scores well.
- The honest path: adopt their timing metrics on EgoBlind (D1), run VIABench's in-scope reminder categories with a rate-aware report (D2), and add a zero-shot VLM baseline under SeeForMe's own budget (D3).
- Anchor convention: EgoBlind events are anchored at the question time t_q, the *latest* useful moment; ProactEgoMem scores against the event *start*. Scoring |Δstart| against t_q would penalise anticipation, so a stated convention is required (D1).
- Possible leakage: VIABench and EgoBlind both draw on public videos by blind creators. De-duplicate against EgoBlind train/finetune before any VIABench number.

## A. Benchmarks

| Benchmark | Task | Data and licence | Timing metric | Best published | SeeForMe on it? |
|---|---|---|---|---|---|
| **EgoVoice** (Kim, EMNLP 2026, pp. 29112-29136) | Speak or listen every 1 s chunk during assembly tasks, then generate speech | HoloAssist (CDLAv2); 1,249 train / 170 benchmark sessions | Hungarian onset matching, predicted onset within -2.5 s / +1.5 s of the instructor utterance; P, R, F1 per session, macro-averaged; end-vs-action (user's next action start minus utterance end) | EgoVoice-DPO P .482, R .285, F1 .324; ProAssist F1 .195; MiniCPM-o zero-shot .113; Gemini Live .022; gpt-realtime .030 (Table 1-2) | Not meaningfully (domain); use the protocol only |
| **ProactEgoMem** (arXiv 2610.30089) | Proactive memory: Information Capture, State Tracking, Location Tracking; Notification and Intervention levels | Ego4D (own licence) and lab clips; annotations CC BY-NC 4.0; test answers hidden (Codabench); 50 public dev clips | Start@τ (share of matched actions within τ s, τ = 0.2 headline), mean \|Δs\| and σ, P, R; LLM-judged Hungarian matching | (fetch) Notification best F1 19.0; best Start@0.2 28.8; QA with oracle memory 76.5 / 49.3 / 57.6 vs 47.6 / 23.6 / 24.2 self-kept | Partially (Location Tracking adapter on dev clips); low value; borrow the metrics |
| **VIABench** (arXiv 2607.14660) | Proactive Reminder (21-22 sub-tasks), VQA, guided interaction, on video from visually impaired people | 761 videos, 46.9 h; HF tag MIT, GitHub has no licence file; video licence unverified | Detected if a matching prompt falls inside the annotated [ts, te]; Proactive Detection Rate = detected / annotated; **no false-alarm term** | (fetch) best offline Stage-1 recall 39.6; LiveCC 71.7 or 75.8 via dense output | **Yes, offline, on in-scope categories** (obstacles, stairs, crosswalk, escalator) |
| **EgoBlind** (NeurIPS 2025 D&B) | Egocentric VideoQA | 1,392 videos per abstract (draft says 1,329 clips from 478 sources: check); research licence | None (QA accuracy) | best MLLM about 60%, human 87.4% | Already used, but SeeForMe's task is a *derived* timing benchmark; never call it EgoBlind accuracy |

What carries over from EgoVoice: user behaviour is fixed in replay (their Limitations section, same as ours); recall is the binding constraint there too (R .285); every system starts speaking at a similar point and **duration** decides whether a cue finishes in time, so report cue end, not only onset; live commercial assistants almost never speak unprompted (F1 at most .030). Their labels are two-sided and offline; ours are one-sided and online.

## B. Products and a head-to-head task battery

| Product | Measurable claims |
|---|---|
| Google Guided Vision | Framing cues, Read, Find, Explore; explicitly "not intended for navigation, safe-travel guidance, or obstacle detection" |
| Seeing AI | Short Text auto-read latency and character error rate; Ask |
| Lookout | Explore, Find (7 fixed categories), Read |
| Be My Eyes / Be My AI | Ask accuracy and latency (volunteer calls out of scope) |
| Envision Ally | Ask, Read, claimed memory |

| Task | Protocol | Metrics |
|---|---|---|
| Ambient | Fixed route or clip set with annotated event intervals and quiet stretches; user asks nothing | Detection Rate, Start@1 s, signed onset offset, false alerts/min, cues/min, speech seconds/min, quiet-window violations, end-before-need |
| Ask | Fixed questions with about 3 reference answers, including "is it clear?" with a known answer | Judged accuracy, latency, false-clear rate |
| Find | Lookout's 7 categories, 3 staged rooms | Success within 60 s, time to first correct direction, false "found" |
| Read | 40 printed items at 0.5 / 1 / 2 m, normal and dim light | Character and word error rate, time to first word |
| Remember | Place an object, walk, ask after 30 s / 2 min / 10 min | Location accuracy by delay, same-object repeat rate |
| Explore | Unfamiliar room with an annotated inventory | Item recall and precision, left/right/ahead accuracy, words per description |

Rule: claim a win only on a task where the competitor makes the matching claim, at an equal or lower speech rate, with the CI excluding 0. Third-party apps are operated by a human tester; score blind to which system spoke; mark "not designed for" separately from "failed".

## D. The three evaluation changes

- **D1. Onset-timing metrics on EgoBlind.** Reference time r_e = `ideal_s` (clamp(t_q - 2 s) in `data.py`), also reported against t_q. Hungarian matching of events to cues about the same source with cost \|Δ\|, allowed in [-L, +U]: native (L = 10 s, U = 0.5 s vs t_q) and EgoVoice-style (L = 2.5, U = 1.5 vs r_e). Report Start@τ for τ in {0.5, 1, 2} (0.2 for parity only; decisions are every 0.5 s), recall-weighted Start@τ over all events, mean / median / signed \|Δs\|, P (denominator: cues within 6 s of a labelled moment), R over all and coverable events, F1 macro over videos and pooled, paired Wilcoxon and bootstrap.
- **D2. Rate-aware detection rate and a VIABench run.** Detection rate always reported with cues/min, false alerts/min and a speak-every-step dense baseline, plus a detection-rate vs cues/min curve. On EgoBlind this is all-event recall (0.111 greedy). Before VIABench: de-duplicate sources, resolve the licence, confirm the interval fields.
- **D3. Zero-shot VLM baseline under the same budget.** A TPAD-style forced choice ("Should the user be warned?") on the last 10 frames at 1 fps; speak when p >= θ_t with θ_t from the same `BudgetController`; prompt chosen on `select`. This is the only head-to-head today's data supports.

## Lecture 10 (Statistical Learning Theory: online learning, agnostic)

Regret against the best fixed expert (p7-8); deterministic learners suffer linear regret (p9); randomised learners (p10); Multiplicative Weights with regret <= sqrt((T/2) ln\|H\|) (p11-15); Littlestone classes, regret Θ̃(sqrt(lit · T)) (p16-19); boosting from Multiplicative Weights (p20-22). Not covered: bandits, online convex optimisation, delayed feedback, apple tasting.

Mapping: (1) report empirical regret per event against a finite class of rate-matched replayable policies (P-CACS, IL + guard, RL, floor, a hindsight logistic fit); (2) a Hedge selector over existing schedulers per user, exact when p_ask = 1 reveals every miss but needing a bandit extension where a cue gets no tap; at T = 160-293 events the bound (0.05-0.10 per event) is the same size as measured effects, so it is a framing and evaluation tool, not an expected win; (3) randomisation as formal grounds for Thompson sampling (empirically no gain).

Lectures 13-14 additions: a linear probe on the 414,017 cached CLIP crop embeddings (R-CNN idea, L14 p39-41) gated through `feature_auc.py`; a detector confidence sweep reporting coverable share against false tracks per minute (L14 p46, p48); augmentation for the teacher-student detector (L13 p32), flipping left/right labels with horizontal flips.
