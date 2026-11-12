# SeeForMe PRD (research track, from 2026-11-12)

## Problem

Blind and low-vision people already have good on-demand tools (Be My Eyes and Be My AI, Seeing AI, Lookout, Aira, Meta Live AI). They mostly wait to be asked; where they do speak unprompted (Seeing AI's Short Text channel, Lookout's Explore mode, OKO's crosswalk signals, Biped's obstacle alerts) they follow fixed rules. None learns from the user when to speak. Nobody has solved the general question: **when should an assistant speak about the surroundings without being asked, and when should it stay quiet?**

The October work showed why this is hard:
- Recall on EgoBlind is low: 0.20 for the learned policies, 0.07 for P-CACS.
- Offline learners trained on sparse labels collapse to silence.
- Weak labels capture what users wanted to know, which is broader than collision warnings.

## Thesis claim (every task must serve it)

> An assistant that learns from one-sided feedback (the user's questions and taps, never its own silence) catches more of what blind users want to know, at no more than about 2 cues a minute, than fixed or offline-trained policies. It also improves per user within one session.

Theory framing:
- **Apple tasting** (Helmbold, Littlestone and Long): the label is revealed only when the system acts, as in label-efficient prediction.
- **Partial monitoring** and contextual bandits.
- **KWIK abstention** for scenes the model can't read.
- **Online-to-batch conversion**, linking the online learner to the offline models.

Every citation must be verified before it goes into the thesis.

## Users

- **Primary:** blind and low-vision pedestrians with a chest-mounted phone.
- **Secondary:** orientation and mobility (O&M) instructors.
- **Research audience:** the CPSC 4900 advisor and readers.

## Success metrics

| Metric | Target | Where measured |
|---|---|---|
| Coverable recall at no more than about 2 cues/min | Online learner beats frozen IL, RL and P-CACS, with a bootstrap 95% CI that excludes 0 | EgoBlind holdout stream (once), then official test (once) |
| Quiet-window violation rate | Not worse than frozen IL | same |
| Learning curve | Recall rises with the number of feedback events within a proxy user (source recording) | finetune and select streams |
| Regret | Sublinear against the best fixed threshold in hindsight | train stream |

## Non-goals (for now)

- Reading text, describing photos, human volunteers.
- Pricing.
- Native iOS.
- Pre-training on new large datasets before the core experiment exists.

## Constraints

- **Licences.** Research track (a) may use EgoBlind (CC BY-NC) and GuideDog. Product track (b) may use only CampusBench, the owner's own recordings, and Mobilio (MIT, logs only). BLV-Road-Nav and WalkVLM have no licence, so they are research only, and permission must be asked. Every model carries a `licence_track` tag.
- **Evaluation hygiene.** Never train or select on the EgoBlind holdout or the official test set. Report bootstrap intervals (`scripts/bootstrap_ci.py`).
- **Privacy.** Raw video never goes to git or Supabase. Faces are blurred in the owner's clips.
- **Coordination.** The peer session owns `web/`, `server/`, `seeforme/live/`, `scripts/run_live.py` and `seeforme/pipeline.py`. This track stays out of them.
- **Design.** Apple Human Interface Guidelines, audio first: VoiceOver, haptics, earcons, interruption rules. Visual polish is secondary, applied to the annotation tool and the results site.

## Milestones

| ID | Name | Serves |
|---|---|---|
| M0 | Hygiene: ECC setup, the hanging test suite, project CLAUDE.md | all |
| M1 | One-sided feedback stream and online learners | the thesis claim (core) |
| M2 | Persistent object memory on GPU (appearance re-ID beyond the tracker TTL) | fewer repeat cues, plus a "seen before" feature for M1 |
| M3 | Label quality: review tool for about 650 weak events, an awareness event type, Supabase schema | ground truth for M1 |
| M4 | Vocabulary: doors and stairs (open vocabulary), evaluation of permissively licensed detectors | coverable fraction |
| M5 | Owner's New Haven recordings: capture protocol, face blur, dense labels, Cohen's kappa | external validity, product data |
| M6 | Thesis write-up and advisor package | deliverable |
