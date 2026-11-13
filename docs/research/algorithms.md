# SeeForMe M1: what the online-learning results show and what to change next

I read `seeforme/online/*`, `scripts/run_online.py`, the EgoBlind window and labelling code, and `docs/online.md` with its logs. I then ran some diagnostics myself, using the `train,finetune` dev stream only. I did not touch `select`, holdout or test. I edited no repo files and ran no git commands. My scripts and logs are in `<scratch>` (`probe.py`, `dump.py`, `sep.py`, `*.log`). My seed-0 greedy run reproduces the logged 0.188 exactly (30/160).

## Headline: the 0.53 "upper bound" comes from label leakage, so the 0.20 vs 0.53 gap is not real

In `OnlinePolicy.decide`, `_learn_full_info` labels the current candidates at time t and updates the weights. The decision at t+0.5 s then scores almost the same feature vector, for the same track, in the same event window. The full-information learner is therefore told the answer for the event it is about to announce. No deployable learner gets that.

I gave the same full-information labels to the learner but held them back until the video ended (`full_info+dv`). I also tried a fixed delay d in seconds (`full_info+d<d>`):

| Policy (dev stream) | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| Full info, as now (labels arrive immediately) | 84/160 = 0.525 | 0.500 | 0.513 | 0.513 |
| Full info, labels 2 s late | 0.369 | | | |
| Full info, labels 12 s late (longer than any event window) | 0.206 | | | |
| **Full info, labels at video end** | **0.200** | **0.206** | **0.194** | **0.200** |
| One-sided greedy (reproduced) | 0.188 | 0.212 (logged) | 0.212 (logged) | 0.204 |
| No learning (lr = 0, weights stay 0) | 0.138 | | | |

When full information arrives only after the decision it informs, it does no better than one-sided feedback (0.200 vs 0.204). So the claim "one-sided feedback costs about 60% of the learnable recall" is not supported. The thesis problem statement and `docs/online.md` points 1 and 6 need rewriting.

Two more checks point the same way:
- **Learning rate makes no difference.** At lr 0.1 greedy gets 0.181 and delayed full info 0.200. At lr 0.02 both get 0.194.
- **Better credit for question labels did not help.** I relabelled each missed question in hindsight, one update per decision, with the other candidates at that moment as negatives (`greedy+h1`, `greedy+h2`). Recall was 0.156 and 0.175 on seed 0.

## 1. The problem, stated formally

- **Rounds:** decisions every 0.5 s (`PCACSConfig.decision_interval`). The context is a set of candidates, each with 29 P-CACS features.
- **Actions:** silence, or speak about one legal candidate. A cue occupies the speech channel.
- **Budget:** roughly 2 cues per minute over the whole stream. That makes it a contextual bandit with a knapsack (speech) constraint, which `BudgetController` handles by dual ascent on the threshold.
- **Loss:**
  - A miss for event e if nothing about e is said inside its window, which runs from earliest_useful (the first matching detection, at most 10 s before t_q) to t_q + 0.5 s.
  - A false alarm for a cue within 6 s of a labelled moment that is not the first timely mention of an event (`counts_fp`).
  - A quiet violation for a cue in [t_q − 4, t_q + 0.5] when the user confirmed nothing was there.
- **Feedback:**
  - Speaking gives a tap with probability p_tap = 0.5. This is the apple-tasting channel.
  - Silence gives nothing, except that a later question at t_q reveals y = +1 for event e over its whole window. That feedback is delayed (up to about 10.5 s), aggregated over an interval, and arrives with the candidate identity hidden.
  - A quiet question reveals y = −1 over [t_q − 4, t_q + 0.5].

**Why the measured gap collapsed:** in the simulator every question is asked (`advance` never drops one), and the metric scores only decisions near questions. So every scored outcome is eventually revealed, with a short delay. On the scored region the problem is really delayed full information. Only the unscored region is truly one-sided, and there it affects nothing except the budget. Within a single video the stream is strongly autocorrelated, so a delay matters a great deal (0.513 → 0.369 at 2 s → 0.200), even though theory says delay adds only a small term in i.i.d. settings (Joulani et al.).

## 2. Existing theory and algorithms (citations checked)

- **Apple tasting** (Helmbold, Littlestone, Long 2000): the learner sees the label only when it accepts an instance. They give reductions to the standard mistake-bound model. Information and Computation 161(2):85-139, https://doi.org/10.1006/inco.2000.2870
  - Raman et al. (COLT 2024) show the Littlestone dimension still characterises the agnostic case, and the realizable rates are Θ(1), Θ(√T) or Θ(T). https://arxiv.org/abs/2310.19064
- **One-sided feedback in practice** (Sculley 2007): mistake-driven learners such as the Perceptron suffer under one-sided feedback. He tested on spam filtering. Jiang et al. (AISTATS 2021, section 2) report that he found the Helmbold et al. method less effective than a simple greedy strategy, which matches our finding that exploration does not help (second-hand: Sculley's full text was not opened). https://doi.org/10.1145/1281192.1281258 and https://arxiv.org/abs/2006.04858
- **Label-efficient prediction** (Cesa-Bianchi, Lugosi, Stoltz 2005): matching upper and lower bounds when the learner may query only a limited number of labels. IEEE Trans. Inf. Theory 51(6):2152-2162, https://doi.org/10.1109/TIT.2005.847729 (COLT 2004 version: https://doi.org/10.1007/978-3-540-27819-1_6)
- **Partial monitoring** (Bartók, Foster, Pál, Rakhlin, Szepesvári 2014): every finite game has minimax regret of order 0, √T, T^{2/3} or T, and local observability separates √T from T^{2/3}. Mathematics of Operations Research 39(4):967-997, https://doi.org/10.1287/moor.2014.0663 (preprint: https://www.mit.edu/~rakhlin/papers/partial_monitoring.pdf)
  - Unverified: which class two-action apple tasting falls into. My own reasoning is that it is locally observable (√T), because the speak action reveals the outcome. I found no source that says so.
- **Bandits with knapsacks** (Badanidiyuru, Kleinberg, Slivkins, FOCS 2013; J. ACM 65, 2018, https://doi.org/10.1145/3164539): https://arxiv.org/abs/1305.2545
  - Linear contextual version: Agrawal and Devanur (NeurIPS 2016, pp. 3450-3458), https://arxiv.org/abs/1507.06738
  - Resourceful contextual bandits: Badanidiyuru, Langford, Slivkins (COLT 2014), https://arxiv.org/abs/1402.6779
  - Primal-dual threshold updates of this kind are what `BudgetController` approximates.
- **Delayed feedback:** Joulani, György, Szepesvári (ICML 2013) show delay costs multiplicatively in adversarial settings and additively in stochastic ones. PMLR 28(3):1453-1461, https://proceedings.mlr.press/v28/joulani13.html
  - Delayed, aggregated, anonymous feedback, which is our question label over an interval: Pike-Burke, Agrawal, Szepesvári, Grünewälder (ICML 2018), https://ar5iv.labs.arxiv.org/html/1709.06853
- **Positive-unlabeled learning:** Elkan and Noto (KDD 2008) show that under "selected completely at random", a P-vs-U classifier differs from the true one by a constant factor. https://doi.org/10.1145/1401890.1401920
  - Kiryo, Niu, du Plessis, Sugiyama (NeurIPS 2017), non-negative PU risk estimator: https://neurips.cc/virtual/2017/poster/8958
- **KWIK** (Li, Littman, Walsh, ICML 2008): the learner may say "I don't know", with a bounded number of such answers. https://doi.org/10.1145/1390156.1390228

## 3. Why one-sided learners reach only about 0.20

1. **The upper bound leaks, as shown above.** Without leakage the real ceiling for these features and this model is about 0.20, and one-sided learners already reach it.
2. **Class balance is not the cause.** Seed-0 greedy fit calls by kind:
   - quiet_window: 265 calls, 6,101 rows, weight 265 (all negative)
   - tap_not_useful: 75, weight 75 (negative)
   - missed_question: 117 calls, 3,055 rows, weight 117 (positive)
   - tap_useful: 22 (positive)

   That is about 1 : 2.4 positive to negative gradient mass. Full info has 6,903 calls with 741 positive weight to 6,162 negative, about 1 : 8.3. The one-sided learner is actually the better balanced of the two.
3. **Credit assignment is not the cause.** Delayed full information already has perfect per-candidate credit and still gets 0.20. The hindsight relabelling gave 0.156 and 0.175. The "credit assignment" item in `online.md` (Next experiments) should be expected to be null.
4. **Which candidate to name is not the problem; when to speak is.** At decisions that contain a positive, picking a candidate at random is already right 83.9% of the time. The model gets 83.5%. Positives are matched by label, so most candidates at those moments are, for example, "person".
5. **The features carry little signal.** I trained on `train` and tested on `finetune`, using full-information rows near labelled moments (20,774 / 4,479 rows; 3,454 / 959 positive):

   | Model | AUC on train | AUC on finetune |
   |---|---|---|
   | Logistic regression | 0.728 | 0.675 |
   | Gradient boosting (overfits) | 0.966 | 0.613 |
   | Logistic + class one-hot | | 0.679 |
   | Decision-level "is now a speaking moment" | 0.758 | 0.683 |

   The single most predictive feature is `ego_speed`, the user's own walking speed (finetune AUC 0.691). With about 160 coverable events the data, not the learner, is the limit.
6. **A side issue: `results/online/summary_dev.md` is stale.** Its apple row shows eps = 0.05, tau stuck at 0.999 and 3.96 cues per minute. That contradicts the table in `online.md`, which uses the s0 to s2 runs. Delete or regenerate it before anyone cites it.

## 4. Ranked algorithmic changes

### 1. Make the full-information reference causal, and add a no-learning floor (under a day, already prototyped)

- **Change:** `OnlinePolicy._learn_full_info` stops calling `_fit` immediately. It appends (t, X, y, w) to a pending buffer. The buffer is flushed in `reset()` (video end) or once t passes the latest `latest_safe_s` among pending labels. Add a constructor argument `full_info_delay` ("video" or seconds) and a policy name `online_full_info+causal` in `run_online.py`. Also add an `online_greedy` run with lr = 0 as the no-learning floor.
- **Why:** it removes the leakage within a video, so the gap actually measures the cost of the feedback channel.
- **Expected effect:** the reference drops from about 0.51 to about 0.20, and the one-sided gap goes to about 0. Already measured: 0.200 / 0.206 / 0.194.
- **Test:** 3 seeds on dev, then the same on `select`. Use a paired bootstrap of causal full info against greedy, resampling videos (`scripts/bootstrap_ci.py`).
- **Falsified if:** on `select`, causal full info beats greedy by more than 0.08 in pooled recall with a CI that excludes 0. In that case a real one-sided gap exists.

### 2. Make the simulator actually one-sided: users do not always ask

- **Change:** add `p_ask` to `FeedbackModel`. In `advance()`, a "speak" question fires with probability p_ask, using a seeded RNG.
  - Also log `n_pos` and a per-kind label count in `OnlinePolicy.to_dict()`, so the logs show the label balance directly.
  - Then add a PU correction in `OnlinePolicy.learn`. Scale missed-question positives by 1/(p_ask·c), as in Elkan and Noto, where c is estimated from the tap and question rates. Or use the Kiryo non-negative risk for silent candidates in windows near questions.
- **Why:** with p_ask = 1 the question process reveals every scored outcome, which is why section 3 found no gap. The thesis claim is about real users, who often do not ask.
- **Expected effect:** as p_ask goes from 1 to 0.5 to 0.25, greedy recall should fall while causal full info stays near 0.20. That would reopen a gap of perhaps 0.03 to 0.08 (unverified guess), and the PU correction should recover part of it.
- **Test:** sweep p_ask on dev for diagnosis. Choose the correction weight on `select` only.
- **Falsified if:** greedy at p_ask = 0.25 stays within 0.03 of greedy at p_ask = 1 over 3 seeds. Then the feedback channel is not a bottleneck at any realistic asking rate, and the thesis should be reframed around anticipation and timing.

### 3. Add "when" context features at the policy level

- **Change:** in `OnlinePolicy.decide`, after building `raw`, append decision-level features. Do not change `EventFeatures`, because the frozen IL and RL models depend on it. Set `N_FEAT` accordingly. Candidates:
  - change in ego speed over the last 1 s and 3 s, and a "user stopped" flag
  - seconds since the last cue
  - seconds since clip start
  - number of new tracks in the last 2 s
  - maximum `growth` and `ttc` across all tracks
- **Why:** the bottleneck is anticipating the moment a user needs information, and `ego_speed` is the strongest feature we have.
- **Expected effect:** +0.02 to +0.06 recall (unverified).
- **Test:** first offline, with decision-level AUC trained on `train` and tested on `finetune`; the baseline is 0.683. Only if that rises by at least 0.02, run the 3-seed online comparison and choose the feature set on `select`.
- **Falsified if:** offline AUC gains less than 0.02, or online recall on `select` gains less than 0.02 at equal cues per minute.

### 4. Replace the threshold dual ascent with a score-quantile threshold

- **Change:** in `BudgetController`, keep a running quantile (for example P² or a recent-window quantile) of the best legal score per decision. Speak when the score is above the 1 − r quantile, where r = target · dt / 60, with dual ascent kept only as a slow correction.
- **Why:** today one cue moves logit(tau) by +0.29 (eta 0.3), and tau swings between 0.29 and 0.78 within a run. A quantile threshold is unaffected by drift in the weight scale.
- **Expected effect:** small, 0 to +0.03.
- **Test:** same as above, at matched cues per minute (±0.1).
- **Falsified if:** there is no gain on `select` at equal rate.

### 5. Leave exploration, warm start and model capacity alone until there is more labelled data

- **Change:** none to the algorithms. Spend the effort on more question-annotated events instead (own recordings, BLV-Road-Nav once permission arrives).
- **Why:** the existing evidence:
  - Exploration does not help (`online.md`).
  - Gradient boosting overfits badly: train AUC 0.966 against finetune 0.613.
  - Learning rate does not matter.
- **Falsified if:** a higher-capacity model (GRU or GBM) trained with causal full information beats logistic regression by more than 0.05 out of sample on `select`.

## Consequence for the thesis

As built, the benchmark cannot show a cost from one-sided feedback, because questions always come and only decisions near questions are scored. Two framings are defensible:

- **(a)** Anticipation is the bottleneck. With honest causal labels every learner reaches about 0.20 against a floor of about 0.14 with no learning, so the gain comes from features and data.
- **(b)** The one-sided structure matters only when users do not always ask (item 2), which the user study must measure.

Either way, the "0.525 upper bound" and "60% of learnable recall" numbers should be removed from the README, `online.md` and any paper drafts.
