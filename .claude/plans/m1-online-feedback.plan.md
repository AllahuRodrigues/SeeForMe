# M1 plan: one-sided feedback stream and online learners

PRD: `.claude/prds/seeforme.prd.md`. Branch: `research/online-learning` (worktree `../Project-research`), and later a PR to main.

## The idea in one paragraph

Replay EgoBlind videos in a fixed time order, as if one assistant were used day after day. At each decision point, the policy decides which tracked candidate to speak about, or stays silent. The user's feedback is one-sided:

| What happens | Label the learner receives |
|---|---|
| The assistant speaks and the cue was useful (inside an event window) | positive tap |
| The assistant speaks near a labelled moment but the cue wasn't useful | negative tap |
| The assistant speaks anywhere else | nothing (nobody knows) |
| The assistant stays silent and the user later asks (a question at t_q for an event it never announced in time) | the candidates for that event become **positives**: the missed-cue label |
| A quiet window (the user asked "is anything there?" and the answer was no) | candidates in that window become **negatives** |
| The assistant stays silent and nothing happens | nothing |

So silence is never labelled directly, which is apple tasting plus question-triggered labels. Learners use the 29 P-CACS features (`EventFeatures`), so they are comparable with P-CACS and the October GRU.

## Slices (each one: RED test, then GREEN, then review)

| Slice | Deliverable | Acceptance criteria |
|---|---|---|
| S1 | `seeforme/online/feedback.py`: `FeedbackModel(scenario)`, turning `(t, candidate, action)` into a feedback event or none, plus `question_labels(log)` | AC-1: a useful cue gives +1. AC-2: an off-event cue within 6 s of a labelled moment gives −1, and elsewhere gives none. AC-3: a missed event yields positive labels for that event's candidates in [earliest, t_q]. AC-4: a quiet window yields negatives. AC-5: silence alone gives no label. AC-6: tap probability `p_tap` < 1 drops taps reproducibly from a seed. |
| S2 | `seeforme/online/learners.py`: online logistic regression (AdaGrad), with an importance-weighted update | AC-7: it converges on a separable synthetic stream. AC-8: the update is a pure function that returns new weights (immutable). |
| S3 | `seeforme/online/policy.py`: `OnlinePolicy(PCACSPolicy)` with modes `greedy`, `apple` (ε-explore below the threshold, inverse-propensity weighted), `thompson` (Laplace posterior), `full_info` (an upper bound that sees every label); a budget controller (dual variable on the threshold, target 2 cues/min) | AC-9: the budget controller holds cues/min within ±0.5 of target on the train stream. AC-10: `full_info` ≥ `greedy` recall on a synthetic stream. AC-11: channel legality matches `SchedulerPolicy`. |
| S4 | `scripts/run_online.py`: progressive validation (predict, then update) over the train then finetune streams, ordered by source recording then time; hyperparameters chosen on `select` only; frozen baselines P-CACS, IL and RL on the same stream; learning curves; `results/online/` | AC-12: a run is reproducible from the seed. AC-13: the holdout and test splits can't be loaded unless `--final` is passed. AC-14: output includes per-video rows for `bootstrap_ci.py`. |
| S5 | Final evaluation, run once: freeze after train plus finetune, then (a) frozen and (b) still-learning on holdout, then the official test; `docs/online.md` with intervals | AC-15: every number has a 95% bootstrap CI against frozen IL. |
| S6 | Personalisation: global weights plus a per-user offset (one proxy user = one source recording), shrunk toward the global weights | AC-16: the learning curve within a user is reported against the global-only model. |

## Risks

- **Weak labels.** The positives are what users asked about, not ground truth. The M3 human review will relabel a subset, and results will be reported on both.
- **The 45% of events the detector can't see** cap recall whatever the learner does. Report coverable recall, with M4 as the fix.
- **Proxy users.** Source recordings aren't real users. Say so in the thesis.

## Out of scope for M1

GPU appearance memory (M2), new perception (M4), anything in `web/` or `server/`.
