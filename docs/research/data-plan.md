# SeeForMe: status, gaps and data strategy (explainer report, 2026-11-12)

Written for the student and advisor. I read the repo docs, plans, results and the data folder directly. The data folder was only read, never copied. Every count below comes from `data/splits/*.jsonl`, `data/egoblind/scenarios/<split>/*.json` and `results/` unless a URL is given. Estimates are labelled as estimates. Anything I could not check is labelled "unverified".

---

## 0. Things the hallucination checker should look at first

1. **`docs/online.md` line 49 has a wrong count.** It says "`select` holds 33 coverable events per seed, 99 in all". In fact `select` has **11 coverable events per seed**, so 33 is the total over three seeds. I checked this against the scenarios and against `results/online/per_video_ineg0_s0.csv` (34 select videos, n_coverable sums to 11). The "9/33" and "12/33" entries in the implicit-negatives table are pooled over 3 seeds × 11 events. **So the choice of w = 0 rests on 11 distinct events.**
2. **The review CSVs show model results next to the label.** `results/egoblind/review_events_*.csv` have columns `bin:pcacs`, `max_score:*` and `bin:egoblind_learned:il...`. A reviewer who can see whether a policy caught an event is biased. Hide these columns before review, especially for holdout and test.
3. **`docs/ecc/ECC-OPERATING-MANUAL.md` is in git, but it is the manual for STAR, a different project.** It contains a local home-directory path, and the SeeForMe repo is public.
4. **The advisor is named inconsistently.** README: "Advisor: to be confirmed (course instructor: Prof. Sohee Kim Park)". `update.md`: "Advisor: Sohee Kim Park".
5. **TIMELI's `real_test.json` is built from YouTube walking videos** (`docs/timeli_notes.md`). The README says "YouTube video is not used". Only labels are on disk, but the thesis should state this.
6. **Proxy-user counts depend on how recordings are merged.** Union-find (`source_recordings()`) gives 111 proxy users for train+finetune, which matches `docs/online.md`. The `source_recording` field in `data/splits/egoblind_train.jsonl` gives 101 for train alone, against 91 by union-find, because 11 clips have no source row. This is not an error, but the thesis should say which definition it uses.

---

## 1. What is done and working

### Data actually on disk (EgoBlind, counted today)

| split | videos | proxy users (union-find) | minutes | questions | should-speak events | coverable | detected | quiet windows |
|---|---|---|---|---|---|---|---|---|
| train | 454 | 91 (+11 clips with no source row) | 230.9 | 1,751 | 237 | 133 | 120 | 241 |
| finetune | 104 | 20 | 59.3 | 400 | 56 | 27 | 24 | 38 |
| select | 34 | 15 | 43.5 | 217 | 27 | **11** | 9 | 18 |
| holdout (untouched) | 77 | 22 | 54.7 | 369 | 63 | 30 | 23 | 26 |
| official test (untouched) | 656 | 138 (+11 no row) | 371.8 | 2,565 | 367 | 200 | 174 | 297 |

- **Development stream** (train+finetune): 558 videos, 290 min, 160 coverable events, 111 proxy users. This matches `docs/online.md`.
- **Most questions produce no event.** 73 to 76% of questions are skipped (status `skip`), mostly "information" questions.
- **Why events are uncoverable** (train / finetune / select / holdout / test):
  - not_detectable: 54 / 10 / 14 / 11 / 114
  - taxonomy_decor: 25 / 6 / 0 / 15 / 25
  - no_object_named: 21 / 13 / 2 / 7 / 27
  - cache_truncated: 4 / 0 / 0 / 0 / 1
- **Should-speak event types in dev** (train+finetune+select): person 76, vehicle 69, static hazard 46, unknown 36, decor 31, door 29, stairs 9, barrier 8, bike 6, wall 5, landmark 5.
- **Other data on disk:**
  - Appearance caches: `data/egoblind_app/` holds 592 `.npz` files (train+finetune+select).
  - CampusBench v2 splits: 84 train / 42 val / 90 test.
  - GuideDog (gold, depth, object), TIMELI labels, VizWiz sample, Mendeley, and the BLV-Road-Nav keyframes (1,795 frames).

### Results that hold up (dev stream, 3 seeds, holdout and test untouched)

- **One-sided feedback costs about 60% of learnable recall.**
  - Full-information learner: 0.525 coverable recall.
  - One-sided learners: greedy 0.204, Thompson 0.188, apple tasting 0.183.
  - All learners run at about 1.7 cues/min against a 2/min target.
- **Online learners beat the fixed rule and the frozen RL model:** 0.20 against P-CACS 0.087 and frozen RL 0.050.
- **Against frozen IL they are not separable.** IL scores 0.331 overall, but it is in-sample on train. On finetune it is 4/27, against 2 to 7/27 for the learners.
- **Warm start from CampusBench:**
  - Carrying the full optimiser state hurts: greedy -0.058, CI [-0.107, -0.013].
  - Carrying weights only is neutral: Thompson 0.206, +0.019, CI [-0.016, +0.054].
- **Implicit negatives hurt at every weight**, and more weight hurts more.
- **Re-ID memory:**
  - Match precision 0.90 (θ 0.80, n = 2,041 matches with ground-truth ids).
  - P-CACS label repeats fall by 0.031/min, CI [-0.064, -0.004].
  - A loose θ hurts the online learner: -0.062 recall.
  - IL re-announces the same live track in 28.5% of its cues. That is a scheduler bug, not a memory problem.
- **Engineering that is in place:**
  - Leakage-free splits by whole recording, with hygiene that blocks holdout and test unless `--final` is passed.
  - A bootstrap CI tool, a reproducibility check (163/163 metrics identical), and 67+ tests.
  - Browser and Python parity (4,560/4,560 steps), and licence provenance on every model.
- **CampusBench (simulator) results:** learned GRU+RL reaches 0.70 recall against 0.73 for the reference. These are simulator numbers only.

---

## 2. What is difficult, and why

1. **The labels are weak and one-sided by design.**
   - A "should speak" event exists only where a blind user asked a navigation or safety question. The window comes from the question time, not from human judgement.
   - Silence is never labelled, which is the thesis setting. But it also means there is no negative ground truth outside the quiet windows (380 in dev).
2. **The evaluation sets are too small to separate the policies.** Normal-approximation half-width of a 95% CI for a recall near 0.25:

   | coverable events | ± (half-width) | split this applies to |
   |---|---|---|
   | 11 | 0.26 | select |
   | 30 | 0.155 | holdout |
   | 200 | 0.06 | official test |
   | 500 | 0.038 | |

   Selection decisions made on `select` are close to noise. Holdout cannot rank the quiet policies, as the README already says. Only the official test can resolve differences of about 0.10, and it may be used once.
3. **Nearly half the events are out of reach.** 45% of official-test events, and about 44% in dev, fall outside what COCO plus the taxonomy can name: doors, stairs, fences, poles, unnamed things. YOLO-World widens the vocabulary without detecting more (holdout: 23 → 23 detected).
4. **The labels and the objective mean different things.** Users ask awareness questions ("is it safe?"), while P-CACS scores collision risk. At their best moment, silent events are a median 4.8 m away and 3.1 s from contact. Only a human review can say which moments deserved a cue.
5. **The clips are short (mean 31 s).** Long-horizon memory and within-user learning curves cannot really be tested on EgoBlind. Proxy users (source recordings) are not real users.
6. **The licences split the work into two tracks.** The strongest real data (EgoBlind, GuideDog) is non-commercial. The product track has almost no real BLV video.
7. **Ultralytics YOLO is AGPL-3.0** (`THIRD_PARTY_NOTICES.md`). This constrains the product track even when the data is clean.

---

## 3. What is missing

- **Human review of the weak events: 0 of 730 rows done.**
  - `human_verdict` is empty in every review CSV: train 237, holdout 63 (×2 variants), test 367.
  - No review CSV exists yet for finetune (56) or select (27), and none for the quiet windows.
- **No ideal-timing annotations** (earliest useful, ideal, latest safe) on any real video.
- **No own recordings.** `real_recordings: present false` in `data/inventory_summary.json`.
  - No capture protocol in `docs/`, no face-blur script in `scripts/`, no Cohen's kappa workflow.
- **No door, stair or curb labels or detector.** This is milestone M4, a 1-day estimate in `plan_v3.md`.
- **No real user feedback, and no IRB protocol.** The only IRB mention is one line in `docs/research_plan.md`.
- **Not run yet:**
  - The final `--final` holdout and official-test runs of M1 (S5).
  - Per-user personalisation (S6).
  - Credit assignment for question labels.
  - Hand check of 50 re-ID matches (`results/memory/matches_dev.csv`, 154k rows).
  - Choice of θ on select.
- **Theory claims are not checked.** Regret sublinearity is not measured, and the apple-tasting, partial-monitoring and KWIK citations are still marked "must be verified".

---

## 4. Data needed to make the system genuinely accurate

### 4a. Verify the weak labels you already have (highest value, no new data)

- **What to label:**
  - Each dev should-speak event: train 237 + finetune 56 + select 27 = **320**.
  - Each dev quiet window: 241 + 38 + 18 = **297**.
  - Verdicts: keep / drop / quiet / awareness / collision. Also correct the window (earliest useful, ideal, latest safe) and record the object.
- **Cost:** about 617 rows, roughly 5 to 6 h at 30 s per row (the `plan_v3.md` estimate).
- **Inter-rater check:**
  - A second rater (advisor, labmate, or an O&M instructor if possible) labels a random 20% (about 120 rows).
  - Report Cohen's kappa.
  - Rewrite the rules if kappa is below about 0.6. That is a common rule of thumb, not a verified standard.
- **Evaluation splits** (holdout 63+26, test 367+297):
  - Review them only for label correction, with all model columns hidden.
  - Freeze the policies before opening these files.
  - Never use them to choose anything.
  - Report results on both the weak labels and the reviewed labels.
- **Fix the selection set.** Do not select on 11 events.
  - Use grouped cross-validation over dev recordings (train+finetune+select, 171 coverable events) for hyperparameters.
  - Or move whole finetune recordings into select.
  - Both options stay inside dev and leave holdout and test untouched.

### 4b. Own long recordings (external validity, memory, product track)

- **Why:** this is the only data that is both product-clean and continuous for minutes. Memory (M2) and within-user learning (S6) need it.
- **How much (estimate):** 6 to 10 h raw, as 20 to 30 routes of 15 to 30 min each, covering:
  - Yale campus paths, building entrances, stairs, curb cuts and crossings.
  - Indoor halls, night and rain.
  - Out-and-back routes, so that objects come back into view.
  - Dense-label at least 3 h, aiming for **at least 300 should-speak events**, which gives about ±0.05 on recall.
  - At EgoBlind's density of about 1 should-speak event per minute that would need 5 h, but dense annotation of every actionable object should give more events per minute (unverified).
- **Capture protocol** (write it as `docs/capture_protocol.md`):
  - Chest-mounted phone, fixed mount height and pitch, 1080p/30, phone IMU logged.
  - A route sheet with time, weather and lighting.
  - No recording inside private spaces.
  - Faces and plates blurred before anything leaves the phone or laptop. [EgoBlur](https://github.com/facebookresearch/EgoBlur) is Apache 2.0 according to its [Hugging Face card](https://huggingface.co/projectaria/EgoBlur). Raw video is deleted after blurring.
- **Labelling workflow:**
  1. Model pre-labels: run `scripts/annotate_video.py` for detection, depth and draft events. Every draft row carries `verification: unverified`.
  2. A human pass in [CVAT](https://docs.cvat.ai/) (MIT according to its docs) or a small custom tool sets `should_speak`, the time window, minimum information and the ideal cue, then flips `verification` to `human`.
  3. A second rater on 20% (kappa).
  4. Only `human` rows count as ground truth. Unverified pre-labels may be used for training only, and are reported as such.
  5. Split by route and session, never by frame.

### 4c. Door, stair and curb annotations (fixes the 45% vocabulary gap)

- **Targets:** door (open/closed), stairs (up/down) with a handrail side, curb and curb cut, pole/bollard, fence/barrier, wall/glass.
- **How much (estimate):** about 1,000 to 2,000 boxes or masks per class to fine-tune a small detector. This is a rule of thumb, unverified for this footage.
- **Sources by track:**
  - Research track: EgoBlind train frames, at about 1 fps around events of type door and stairs. Dev has 29 door and 9 stair events, so that alone is too few.
  - Product track: own recordings at 1 fps.
  - [Open Images V7](https://storage.googleapis.com/openimages/web/index.html): annotations CC BY 4.0, images listed as CC BY 2.0 according to secondary sources ([Google blog](https://research.google/blog/open-images-v7-now-featuring-point-labels/)). Whether it has Door and Stairs classes is **unverified**.
  - [Mapillary Vistas](https://www.mapillary.com/dataset/vistas): its licence is reported as both CC BY-SA ([ICCV 2017 paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf)) and CC BY-NC-SA 4.0 (third-party listing). Treat it as research only until you check the download's licence text. It is street-level and covers curbs and poles.
  - [CODa (UT Campus Object Dataset)](https://amrl.cs.utexas.edu/coda): campus pedestrian-scale robot data, 8.5 h, CC BY-NC-SA 4.0 per the project site ([arXiv 2309.13549](https://arxiv.org/abs/2309.13549)). Research only.
- **Detector licence check first.** YOLO-World and YOLO11 run through Ultralytics under AGPL.

### 4d. Real user feedback (the thesis claim needs it)

- **What:** taps (useful / not useful) and spontaneous questions from blind and low-vision users walking with the PWA. This is the real version of the simulated one-sided feedback (p_tap 0.5 is an assumption today).
- **How much (estimate):**
  - 6 to 10 participants, 2 sessions each of about 30 min, 1 to 2 weeks apart. This gives a real within-user learning curve and a real tap rate.
  - Plus an O&M instructor rating a sample of cues.
- **IRB:** required. Yale's HRPP, not the investigator, decides whether the study is exempt ([Yale HRPP exempt research slides](https://research-support.yale.edu/sites/default/files/2025-09/IRB%20Essentials_March_2025_Exempt_Research.pdf)). Human Subjects Protection Training is required even for exempt protocols ([Yale Law student guide](https://library.law.yale.edu/research-guides/student-guide-human-subjects-research)).
  - Video of participants plus a disability population means you should expect at least expedited-style scrutiny (unverified for Yale).
  - Contact: hrpp@yale.edu.
  - Recording yourself in public is probably not human-subjects research, but bystanders are filmed. Ask HRPP for a determination for 4b as well, and blur faces either way.
- **Recruitment channels:** unverified. Ask the advisor and Yale Student Accessibility Services. Blindfolded sighted pilots are fine for debugging only, never as evidence.

### 4e. Public datasets: licence and use

| dataset | licence (source) | use here | track |
|---|---|---|---|
| EgoBlind | CC BY-NC 4.0 (local `external/EgoBlind/LICENSE`; [repo](https://github.com/doc-doc/EgoBlind): "freely used for research purpose", cite video sources) | core dev and eval stream | research only |
| GuideDog | CC BY-NC 4.0 plus gate terms: no redistribution of data or derived data ([HF card](https://huggingface.co/datasets/kjunh/GuideDog), [arXiv 2503.12844](https://arxiv.org/abs/2503.12844)) | depth ordering check, hazard vocabulary; silver set (4.5 GB) not downloaded | research only |
| VizWiz | CC BY 4.0 per [arXiv 2102.06793](https://arxiv.org/pdf/2102.06793) citing vizwiz.org; one mirror says CC BY-SA ([Academic Torrents](https://academictorrents.com/details/b633e14aa084fab57f20ad0b4612e0932ae1f2dc)), so check vizwiz.org | capture-quality gate calibration | product OK with attribution (verify version) |
| ORBIT | CC BY 4.0 per paper appendix ([arXiv 2104.03841](https://arxiv.org/pdf/2104.03841)); data at [City figshare](https://city.figshare.com/articles/dataset/_/14294597) | blind-user capture quality, FIND mode | product OK with attribution |
| Ego4D | signed agreement via lab22.ai, about 48 h approval, credentials expire after 14 days ([start here](https://ego4d-data.org/docs/start-here/)); commercial terms **unverified** | negative and diverse scenes, metadata only | research only until verified |
| Ego-Exo4D | official docs say the agreement covers research and commercial use, with restrictions on redistribution ([getting started](https://docs.ego-exo4d-data.org/getting-started/)) | little direct value (skilled activities, not walking) | possibly product, read the agreement |
| BLV-Road-Nav-Accessibility | no LICENSE file (checked locally in `external/BLV-Road-Nav-Accessibility`; [repo](https://github.com/Shohan29531/BLV-Road-Nav-Accessibility)), so all rights reserved by default | 90-object BLV taxonomy, 671 labelled keyframes | research only; permission email drafted |
| Mapillary Vistas | conflicting reports (see 4c) | curbs, poles, street furniture | research only until verified |
| CODa | CC BY-NC-SA 4.0 per project site (see 4c) | campus pedestrian-scale scenes | research only |
| TIMELI | CC BY 4.0 per its local `dataset.croissant.json` ([project](https://timeli-icra.github.io/)); real split built from YouTube | the closest when-to-speak baseline; silence means "no rule fired" | labels usable with attribution |
| Open Images V7 | annotations CC BY 4.0, images CC BY 2.0 (see 4c) | door and stair boxes if the classes exist | product OK with attribution |

---

## 5. Week-by-week plan to Spring 2027

Spring term start and thesis due dates are unverified. Confirm them with the course. Weeks start on Mondays.

| week of | work | exit criterion |
|---|---|---|
| Nov 16 | Fix `docs/online.md` (11 select events); hide model columns in review CSVs; build dev review CSVs (finetune, select, quiet windows); email Yale HRPP for a determination on own recordings and the user study; start Human Subjects training | CSVs ready, HRPP email sent |
| Nov 23 (Thanksgiving) | Review the 320 dev should-speak events | verdicts filled |
| Nov 30 | Review the 297 dev quiet windows; second rater on 20%; kappa | kappa reported, rules revised |
| Dec 7 | Re-run M1 on reviewed dev labels; grouped CV over dev recordings replaces the 11-event select; fix IL same-track re-announce (M3 scheduler) | M1 table on weak and reviewed labels |
| Dec 14 | Write the capture protocol; set up the blur pipeline; first 3 own walks (pilot); draft the IRB protocol for the user study | 3 blurred walks, protocol written |
| Dec 21 to Jan 3 (break) | Record 10 to 15 more routes if on campus; pre-label with `annotate_video.py` (all rows unverified) | at least 4 h raw |
| Jan 4 | Credit-assignment experiment; re-ID θ chosen by grouped CV; hand-check 50 re-ID matches | θ fixed, M2 closed |
| Jan 11 | Door, stair and curb box labelling on own frames plus EgoBlind train frames; licence check of candidate detectors | about 1,000 boxes per class (estimate) |
| Jan 18 (term starts, unverified) | Fine-tune the door/stair detector; measure the change in coverable fraction on dev | coverable fraction reported |
| Jan 25 | Dense human labels on own recordings, first 1.5 h; second rater | kappa on own data |
| Feb 1 | Dense labels, second 1.5 h; personalisation S6 on dev | at least 300 own events |
| Feb 8 | **Freeze everything.** Review holdout and test labels blind to models; one `--final` run on holdout, then the official test, with bootstrap CIs against frozen IL | final M1 numbers |
| Feb 15 | Product-track evaluation on own recordings (memory, long horizon); IRB approval expected (unverified timing) | own-data table |
| Feb 22 to Mar 8 | User study sessions 1 and 2 (6 to 10 BLV participants), only if IRB is approved; otherwise an O&M expert rating of logged cues | real tap rate and learning curve |
| Mar 15 to 22 (spring break, unverified) | Analysis; verify every theory citation | analysis done |
| Mar 29 to Apr 12 | Write the thesis and paper draft; reviewer pass for claims against numbers | full draft |
| Apr 19 onward | Advisor revisions, final package, release notes per licence track | submitted |

**If something has to be cut,** keep 4a (review) and the frozen final run. They are cheap and they make the main claim honest. Next in priority come own recordings, then the door/stair detector, then the user study.

---

Files referenced:
- <repo>/docs/online.md (line 49 error)
- <repo>/results/egoblind/review_events_train.csv (and the holdout and test variants; model columns to hide)
- <repo>/docs/ecc/ECC-OPERATING-MANUAL.md (STAR manual in the public repo)
- <repo>/data/splits/
- <repo>/data/egoblind/scenarios/
- <repo>/results/online/per_video_ineg0_s0.csv

No files were written or changed and no git state was modified.
