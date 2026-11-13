# Citation check (issue #11), 2026-11-13

Scope: every external citation and product claim in `paper/draft.md`, `docs/research/scientific-gap.md`, `market.md`, `algorithms.md`, `benchmarks.md`, and the advisor emails in `thesis/advisor-emails-2026-11.md` (outside this repo; read only). Includes every item that was marked [verify], unverified or (fetch), and the list in `audit-final.md` section 11.

Method: arXiv API (export.arxiv.org) for every arXiv id; Crossref for DOIs, authors, volumes and pages; publisher, proceedings and ACL Anthology pages; arXiv HTML for numbers inside papers; GitHub and Hugging Face APIs for licences; direct page loads for product claims. DBLP blocks automated clients, so DBLP links were replaced with DOIs. Pages that return 403 to scripts (AppleVis, ACM DL, some news sites) were checked through search snippets and are marked as such.

Verdicts: **confirmed** (source says what we say), **corrected** (we fixed authors, year, venue, numbers or a dead link in place), **unverifiable** (could not open the source or the source does not settle it; marked in the doc), **removed** (claim dropped because the cited source does not support it).

Line numbers are after the edits of 2026-11-13. Abbreviations: D = paper/draft.md, SG = scientific-gap.md, AL = algorithms.md, BM = benchmarks.md, MK = market.md, EM = thesis/advisor-emails-2026-11.md.

## 1. Research citations

| # | Citation and claim | Used at | Verified source | Verdict |
|---|---|---|---|---|
| 1 | TIMELI, Kuribayashi, Shangguan, Ohn-Bar, ICRA 2026: when to speak, silence as an explicit label, ~67K CARLA clips | D:54,79; SG:33-37 | https://arxiv.org/abs/2609.05596 ; https://arxiv.org/html/2609.05596v1 ; https://timeli-icra.github.io/ | **corrected**: 67,101 sim videos (79.1% silent) confirmed; real set is TIMELI-Real, 2,000 videos (not just "small"); body reports a pilot with 10 blind participants; `determine_to_talk` is a repo name, not in the paper. Venue "ICRA 2026" is stated by arXiv and the project page, not checked against proceedings |
| 2 | VIABench, Liu et al.: 761 videos, 46.9 h, 530 real VI videos; detection-in-interval metric; TPAD; LiveCC 75.8% from dense output; best offline 39.6 | D:55,80; SG:38-43; BM:18 | https://arxiv.org/abs/2607.14660 ; https://arxiv.org/html/2607.14660v1 ; https://huggingface.co/datasets/MCG-NJU/VIABench | **corrected**: 21 sub-tasks (not 21-22); 71.7 is a different Table 3 column; data is released on HF (tag MIT, no licence file on GitHub). Other numbers confirmed |
| 3 | WorldScribe, Chang, Liu, Guo, UIST 2024 Best Paper; 30 deg / 0.6 / 5 frames triggers; n=5 formative, n=6 eval; pauses on speech | D:56,81; SG:44-48 | https://arxiv.org/abs/2408.06627 ; https://arxiv.org/html/2408.06627v1 ; https://cse.engin.umich.edu/stories/real-time-visual-description-program-for-people-with-low-vision-wins-best-paper-at-uist-2024 | confirmed |
| 4 | WalkVLM, Yuan et al., arXiv only | D:82; SG:49 | https://arxiv.org/abs/2412.20903 | confirmed (no venue) |
| 5 | WalkVLM-LR, Li et al., ICASSP 2026: discriminator gating, GRPO with four rewards | D:82; SG:49-51 | https://arxiv.org/abs/2508.16070 | confirmed (arXiv comment: ICASSP 2026 Best Industry Paper) |
| 6 | ForeSightGuide: filters non-threatening objects; 0.299 redundant alerts, 0.112 missed hazards | D:83; SG:52 | https://arxiv.org/abs/2608.18993 | confirmed |
| 7 | Sighted by Default / VIA-Agent, Zhao et al., UIST 2026: Continuity principle; within-subjects n=9 | D:84; SG:53 | https://arxiv.org/abs/2511.00945 | confirmed; BLV status of the n=9 participants is not in the abstract (unverifiable, already marked) |
| 8 | Chang et al., ASSETS 2025: ChatGPT live video, n=8 BVI, falls short in dynamic scenes | D:58; SG:54 | https://arxiv.org/abs/2508.03651 | confirmed |
| 9 | PING, Ohn-Bar, Kitani, Asakawa, CoRL 2018: weighted-experts personalisation | D:85; SG:56 | https://arxiv.org/abs/1804.04118 | confirmed |
| 10 | BU final report: 13 participants (10 VI + 3 O&M guides), personalisation can worsen some users, quote | SG:57 | https://www.bu.edu/cise/learning-from-interactions-with-blind-users-for-customized-and-scalable-navigation-assistance-systems-final-report/ | confirmed |
| 11 | EgoVoice, Kim, EMNLP 2026, pp. 29112-29136: HoloAssist (CDLAv2), 1,249/170 sessions, 1 s chunks, -2.5/+1.5 s Hungarian onset, Table 1 P/R/F1 | D:90; SG:63; BM:16,21 | https://arxiv.org/abs/2610.12248 ; https://aclanthology.org/2026.emnlp-main.1277/ | confirmed (arXiv id 2610.12248 is correct; all Table 1 numbers match) |
| 12 | Vinci2 / EgoServe / EgoMemo, ECCV 2026: training-free intervention decision, 3,000+ service instances | D:91; SG:64 | https://arxiv.org/abs/2607.11523 | confirmed |
| 13 | Zhang et al., Proactive Assistant Dialogue Generation from Streaming Egocentric Videos, EMNLP 2025 | D:92; SG:65 | https://aclanthology.org/2025.emnlp-main.605/ (pp. 12044-12068, doi 10.18653/v1/2025.emnlp-main.605) ; https://arxiv.org/abs/2506.05904 | confirmed |
| 14 | EgoPro-Bench, Ran et al.: simulated user profiles; uses 420 EgoBlind videos | SG:66 | https://arxiv.org/abs/2605.07299 | paper confirmed; the EgoBlind claim is **unverifiable** (snippet only, not in the abstract; already marked) |
| 15 | YETI, Bandyopadhyay et al. | SG:67 | https://arxiv.org/abs/2501.09355 | confirmed (title and authors) |
| 16 | ProactiveAgent, Lu et al., ICLR 2025: reward model on accept/reject labels; F1 66.47% | D:93; SG:68 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/75c37811e830bf029584b1c6fac17726-Abstract-Conference.html ; https://arxiv.org/abs/2410.12361 | confirmed (ICLR 2025 venue and F1 now verified) |
| 17 | Horvitz, Jacobs, Hovel, Attention-Sensitive Alerting, UAI 1999 | D:97; SG:75 | https://arxiv.org/abs/1301.6707 | confirmed |
| 18 | Horvitz, Principles of Mixed-Initiative UI, CHI 1999 | SG:76 | https://doi.org/10.1145/302979.303030 (Crossref) | confirmed |
| 19 | McFarlane 2002 | SG:77 | https://doi.org/10.1207/S15327051HCI1701_2 (Crossref) | **corrected**: full title and venue added (single author, HCI 17(1)) |
| 20 | Fogarty et al., Predicting human interruptibility with sensors, TOCHI 2005 | SG:78 | https://doi.org/10.1145/1057237.1057243 (Crossref) | confirmed |
| 21 | Pielot et al., Beyond Interruptibility, IMWUT 2017 | D:97; SG:79 | https://doi.org/10.1145/3130956 (Crossref) | confirmed |
| 22 | Okoshi et al., KDD 2019, adaptive notification scheduling | D:97; SG:80 | https://doi.org/10.1145/3292500.3330732 (Crossref) | **corrected**: exact title is "Real-World Product Deployment of Adaptive Push Notification Scheduling on Smartphones" (Okoshi, Tsubouchi, Tokuda) |
| 23 | Helmbold, Littlestone, Long, Apple tasting, Inf. Comput. 161(2):85-139, 2000 (and FOCS 1992) | D:100; SG:86; AL:44 | https://doi.org/10.1006/inco.2000.2870 ; https://doi.org/10.1109/SFCS.1992.267802 | confirmed (bibliographic; DBLP link replaced by DOI) |
| 24 | Raman, Subedi, Raman, Tewari, COLT 2024: Littlestone dim. characterises agnostic apple tasting; Θ(1)/Θ(√T)/Θ(T) realizable | D:100; SG:86; AL:45 | https://arxiv.org/abs/2310.19064 | **corrected** in AL: "2023" to COLT 2024; dead pdf.arxiv.org link (HTTP 500) replaced |
| 25 | Cesa-Bianchi, Lugosi, Stoltz, label-efficient prediction, IEEE TIT 2005 | D:101; SG:87; AL:47 | https://doi.org/10.1109/TIT.2005.847729 (51(6):2152-2162) | confirmed |
| 26 | Cesa-Bianchi, Lugosi, Stoltz, Regret Minimization Under Partial Monitoring, MOR 2006 | SG:87 | https://doi.org/10.1287/moor.1060.0206 | confirmed (was "search result only") |
| 27 | Bartók, Foster, Pál, Rakhlin, Szepesvári, partial monitoring classification, MOR 2014 | D:102; AL:48 | https://doi.org/10.1287/moor.2014.0663 (39(4):967-997) | confirmed; full author list and venue added to D |
| 28 | Jiang, Jiang, Pacchiano, Learning the Truth From Only One Side of the Story, AISTATS 2021 | D:103,259; SG:89; AL:46 | https://proceedings.mlr.press/v130/jiang21b.html ; https://arxiv.org/abs/2006.04858 | confirmed (AISTATS 2021 now verified, PMLR 130:2413-2421) |
| 29 | Bechavod et al., NeurIPS 2019 | D:103; SG:90 | https://arxiv.org/abs/1902.02242 | confirmed |
| 30 | Sculley, Practical learning from one-sided feedback, KDD 2007: one-sided feedback cripples mistake-driven learners | D:104; AL:46 | https://doi.org/10.1145/1281192.1281258 | confirmed (bibliographic and abstract) |
| 31 | Sculley found the Helmbold et al. method less effective than greedy | D:232; AL:46 | https://arxiv.org/abs/2006.04858 (Jiang et al. section 2 says so; Sculley's own text was not opened) | **unverifiable** first-hand: rewritten as an explicit second-hand citation via Jiang et al. The old pointer "arXiv 2006.04858 does not mention Sculley" was wrong: the body does |
| 32 | Agarwal, Gentile, Marinov, user-triggered supervision in contextual bandits | D:63,105; SG:91 | https://arxiv.org/abs/2302.03784 | confirmed (no venue in arXiv metadata) |
| 33 | Joulani, György, Szepesvári, ICML 2013: delay multiplicative (adversarial), additive (stochastic) | D:106; AL:40,54 | https://proceedings.mlr.press/v28/joulani13.html | confirmed |
| 34 | Pike-Burke et al., delayed aggregated anonymous feedback, ICML 2018 | D:106; AL:55 | https://arxiv.org/abs/1709.06853 | confirmed |
| 35 | Badanidiyuru, Kleinberg, Slivkins, Bandits with Knapsacks (FOCS 2013, JACM 2018) | D:107; AL:50 | https://arxiv.org/abs/1305.2545 ; https://doi.org/10.1145/3164539 | confirmed |
| 36 | Agrawal and Devanur, Linear Contextual Bandits with Knapsacks, NeurIPS 2016 | D:107; AL:51 | https://arxiv.org/abs/1507.06738 ; https://neurips.cc/virtual/2016/poster/7392 | confirmed |
| 37 | Badanidiyuru, Langford, Slivkins, Resourceful Contextual Bandits, COLT 2014 | SG:93; AL:52 | https://arxiv.org/abs/1402.6779 | confirmed |
| 38 | Elkan and Noto, KDD 2008: under random selection of labelled positives, P-vs-U classifier differs by a constant factor | D:108; AL:56,98 | https://doi.org/10.1145/1401890.1401920 | confirmed |
| 39 | Kiryo, Niu, du Plessis, Sugiyama, nnPU, NeurIPS 2017 | D:108; AL:57,98 | https://neurips.cc/virtual/2017/poster/8958 | confirmed |
| 40 | Zhang et al., Online Positive and Unlabeled Learning, IJCAI 2020 | SG:92 | https://www.ijcai.org/proceedings/2020/311 (doi 10.24963/ijcai.2020/311) | confirmed |
| 41 | Li, Littman, Walsh, KWIK, ICML 2008 | AL:58 | https://doi.org/10.1145/1390156.1390228 | confirmed |
| 42 | EgoBlind, Xiao et al., NeurIPS 2025 D&B: video count | D:12,111; SG:97-100; BM:19 | https://arxiv.org/abs/2503.08221 ; https://arxiv.org/html/2503.08221v4 | **corrected** in SG and BM: abstract says 1,392 videos, body (section 3.1, Table 2) gives 1,329 clips from 478 long-form videos. The draft's "1,329 clips from 478 source videos" is right |
| 43 | EgoBlind: 5,311 questions; 541 extracted in situ; navigation 14.1%, safety 23.7% (751 and 1,260 of 5,311); best MLLM ~60% vs human 87.4% | D:190; SG:99-100,154-155 | https://arxiv.org/html/2503.08221v4 | confirmed (also ~300 questions added directly by blind volunteers) |
| 44 | GuideDog, Kim et al., ACL 2026: 22K pairs, 2K verified, CC BY-NC 4.0, gated | D:112,372; SG:101 | https://arxiv.org/abs/2503.12844 ; https://huggingface.co/datasets/kjunh/GuideDog | confirmed (licence now verified: cc-by-nc-4.0, auto-approved gate; venue is from the arXiv comment, not the Anthology) |
| 45 | ORBIT, Massiceti et al., ICCV 2021: 3,822 videos, 486 objects | D:113; SG:102 | https://arxiv.org/abs/2104.03841 | confirmed |
| 46 | VizWiz, Gurari et al., CVPR 2018 | SG:103 | https://doi.org/10.1109/CVPR.2018.00380 | confirmed (venue); the 31,173 / 11,045 counts are as quoted by ORBIT, not re-checked |
| 47 | Say It My Way, Zamiri Zeraati et al., CHI 2026: 11 blind users, 418 interactions | D:118 | https://arxiv.org/abs/2602.16930 | confirmed |
| 48 | Misfitting With AI, Alharbi et al., ASSETS 2024: 26 blind participants | D:119 | https://arxiv.org/abs/2408.06546 | confirmed |
| 49 | ProactEgoMem, Liao et al.: data, licence, dev clips, metrics and numbers | BM:17 | https://arxiv.org/abs/2610.30089 ; https://arxiv.org/html/2610.30089v2 ; https://proactegomem.github.io/ | **corrected**: Sci track uses wet-lab data (FineBio, BioVL-QR) plus Ego4D lab clips; F1 19.0 and Start@0.2 28.8 confirmed (paper); QA triplets confirmed (project page, "memory-backed"); CC BY-NC 4.0 and "50 public dev clips" **unverifiable** (in neither source) |
| 50 | Seeing AI speaks text as soon as it is in view | D:49; SG:8,136,181; MK:20 | https://apps.apple.com/us/app/seeing-ai/id999062298 ; https://hadleyhelps.org/workshops/seeing-ai-apple-voiceover-series/reading-short-text | **corrected**: the channel is now "Read" (it replaced Short Text in v5.5.1); Microsoft's own listing says "hear text as soon as it appears in front of the camera". guidedogs.org.uk link is dead (404) and was dropped |
| 51 | Lookout Explore (beta) speaks about objects and text as you point the camera | D:49; SG:181; MK:21 | https://support.google.com/accessibility/android/answer/9031274 | confirmed ("still in beta and is less accurate") |
| 52 | Project Astra "can intuitively start conversations"; Aira Visual Interpreter prototype | D:51; SG:69; MK:9,23 | https://deepmind.google/models/project-astra/ | confirmed (quote exact) |
| 53 | Gemini Live Guided Vision launched 2026-10-01, built with Aira, not for navigation | D:51; MK:8,22; BM:27 | https://www.androidauthority.com/gemini-live-guided-vision-launch-3718122/ ; https://mixed-news.com/en/gemini-live-guided-vision-android-launch-aira/ | confirmed ("not intended for navigation, safe-travel guidance, or obstacle detection" is in the mixed-news article) |

## 2. Advisor emails (read only, not edited)

| # | Citation and claim | Used at | Verified source | Verdict |
|---|---|---|---|---|
| E1 | Vázquez, assisted photography for visually impaired users, ACM TOCHI 2014 | EM:34 | https://doi.org/10.1145/2651380 (Vázquez, Steinfeld, TOCHI 21) | confirmed |
| E2 | Vázquez, when a robot should verbally ask a person for feedback, HRI 2023 | EM:35 | https://doi.org/10.1145/3568162.3576980 (Candon, Zhou, Gillet, Vázquez, "Verbally Soliciting Human Feedback in Continuous Human-Robot Collaboration", HRI 2023) | confirmed (her group's paper; she is last author) |
| E3 | Scassellati lab, "To Help or Not to Help?" framework, THRI 2026 | EM:64 | https://doi.org/10.1145/3797264 (Ramnauth, Brščić, Scassellati, "To Help or Not to Help?: An Expanded Framework for Deciding Socially Appropriate Robot Assistance", THRI 15) | confirmed; the paper is about robot assistance, so "when a robot should offer help" is more exact than "an assistive system" |
| E4 | BLV-Road-Nav-Accessibility, Islam et al., ASSETS 2024, repo has no licence file | EM:116,121 | https://doi.org/10.1145/3663548.3688538 ; https://github.com/Shohan29531/BLV-Road-Nav-Accessibility (GitHub API: license null) | confirmed (authors Islam, Kabir, Pearce, Reza, Billah) |
| E5 | Product framing: Be My Eyes on request; Seeing AI and Lookout speak by fixed rules | EM:24,60 | rows 50, 51 and MK rows below | confirmed |
| E6 | EgoBlind "real first-person videos by blind users"; 558 videos, 0.20 / 0.14 / 0.20 / 0.38, 0.19 / 0.18 | EM:24-29,62,95-98 | arXiv 2503.08221; repo results (paper/draft.md section 6) | confirmed (numbers match draft sections 6.1 and 6.5) |

## 3. Market claims (market.md source tags)

| Tag | Claim | Verified source | Verdict |
|---|---|---|---|
| M1 | Meta May 2026 roundup: only user-started actions | about.fb.com/news/2026/05/... | confirmed |
| AV1 | Meta glasses describe only when prompted with "Hey Meta" | applevis thread (403; search snippet) | confirmed via search; OOrion "coast is clear", SDK beta: **unverifiable** (403) |
| AV2 | Early Be My AI blurred-face and refusal complaints | applevis 152935 (403) | **unverifiable** directly; other AppleVis comments and a Be My Eyes 2023 post agree |
| AV3 | Meta AI "the big weak point"; US/Canada | atlas-entry.applevis.com 180441, 206112 | confirmed; "30-60 min battery" and "grainy" **unverifiable**, marked in MK |
| G1 | Guided Vision: Android 9+, Aira, 1,000+ testers, framing prompts | androidauthority.com | confirmed; "Google eyewear" not in G1 or G2, marked **unverifiable** |
| G2 | "not intended for navigation, safe-travel guidance, or obstacle detection"; not a white-cane replacement | mixed-news.com | confirmed |
| G3 | Astra "can intuitively start conversations" | deepmind.google | confirmed |
| A1 | Aira offsets; Amazon, Google, Salesforce; VR in 5 states | aira.io/new-plans | confirmed (2023 plans; 2026 prices still unverified) |
| A2 | Aira FAQ: no memory across sessions | https://aira.io/introducing-project-astra-join-the-trusted-testers-waitlist-today | **corrected**: claim is on this page; the old link redirects to aira.io/ai/ |
| BME1 | ~1M BLV users, 10M volunteers, March 2026; free | bemyeyes.com news (2026-03-12) | confirmed |
| BME1 | Foundation aims at 50M users by 2035 | same | **removed** (page says "hundreds of millions") |
| BME2 | 361% revenue growth 2024; $6.1M Series A+ Jan 2025; $2.8M Series A 2020 | bemyeyes.com business post; coverager.com | confirmed |
| BME3 | Shares video with Meta for training, opt-out | bemyeyes.com (2026-02-27) | confirmed |
| S1 | Seeing AI free; channels | blogs.microsoft.com; windowscentral | confirmed; "Android version less polished" **unverifiable**, marked |
| S2 | Seeing AI free | digitaltrends (2017) | confirmed (old source) |
| S3 | Some users prefer Be My AI's speech | vi.ie (403; search snippet) | confirmed via search |
| LK1 | Lookout free | neilsquire.ca (403) | confirmed via other listings |
| LK2 | Android 6+; Explore beta; Q&A US/UK/CA | support.google.com 9031274 | confirmed |
| E1 | Solos $399 at launch with 1 year Ally Pro | letsenvision.com | confirmed |
| E2 | Envision Ally app free | letsenvision blog; App Store | confirmed, weak source (App Store id is now "Envision Companion") |
| E3 | Solos later $599-$699 | shop.letsenvision.com; getcoai.com | **corrected** to $699 (no source says $599) |
| E4 | Envision Glasses $1,899-$3,499; $199/yr | letsenvision.com | confirmed |
| MR1 | Ray-Ban Meta Gen 2 from $379 | telecomlead.com | confirmed; meta.com link dead (HTTP 400), **corrected** (dropped) |
| OR1 | OOrion continuous "coast is clear" | doubletaponair.com | partly confirmed (obstacle alerts, proactive guidance); exact phrase **unverifiable** |
| O1, O2 | OKO $4.99/mo subscription since Jan 2025; backlash | App Store id1583614988; applevis; doubletaponair | **corrected**: subscription dropped in July 2025 (v4.7.5: "All app access is now freely available") |
| AP1, AP2 | Magnifier Detection Mode; Apple says not for navigation | support.apple.com; abilitynet | confirmed |
| B1 | Biped NOA description | applevis (403); other sources | confirmed via other sources |
| B2 | 75 devices in launch year; 10 countries | ggba.swiss; bipedai.beehiiv | confirmed |
| GL1, GL2 | Glide first customers, waitlist; $1.5M pre-seed | glidance.io; therobotreport.com | confirmed |
| L1 | .lumen "replicates a guide dog" | dotlumen.com; ces.tech | confirmed |
| L2 | 1,500+ preorders in 10 days, Romanian programme | https://www.dotlumen.com/post/over-1-500-preorders-for-lumen-glasses-in-just-10-days | **corrected**: old medica-tradefair link is 404 |
| L3 | CE-certified; FDA unverified; ~$9,999 | dotlumen CE post; theresanaiforthat (403) | CE confirmed; $9,999 **unverifiable** (already marked) |
| W1 | WeWALK $850; assistant $4.99/mo; $1,150 bundle | androidheadlines (403; search); rnib | confirmed via search |
| N1 | NaviLens free for users; San Antonio $8,950 for 100 stops | navilens.com; sanantonioreport.org | confirmed |
| EV1 | EchoVision $599 ($449 preorder); refund; TestFlight | doubletaponair; applevis (403) | prices confirmed; refund and TestFlight from single AppleVis users |
| SS1 | Soundscape open source; ended by Microsoft | doubletaponair; applevis (403) | confirmed (sunset announced 2022, service to Aug 2023); backend-cost point **unverifiable** |
| LAN1 | 43.3M blind, 295M MSVI (2020); 61.0M, 474M (2050) | PubMed 33275950 | confirmed |
| NRTC1 | 52.3% employment, 8.1% unemployment (2023 ACS) | blind.msstate.edu PDF | confirmed |
| AFB1 | AFB cites 35% | afb.org (now archived) | **corrected** to "about 30-35% (2014-2017)" |
| VR1 | VR funds smart glasses tied to an employment plan | nelowvision.com PDF | confirmed |
| VR1 | Medicare treats smart glasses as non-covered | same | **removed** (the source lists Medicare as a possible funding path) |
| VR2 | VR funds devices | orcam.com (2016) | confirmed (old source) |

## 4. Counts

| Verdict | Research (sec. 1) | Emails (sec. 2) | Market (sec. 3) | Total |
|---|---|---|---|---|
| confirmed | 44 | 6 | 32 | 82 |
| corrected | 8 | 0 | 6 | 14 |
| unverifiable | 1 | 0 | 2 | 3 |
| removed | 0 | 0 | 2 | 2 |
| rows | 53 | 6 | 42 | 101 |

Rows with a confirmed main claim and an unverifiable side detail are counted as confirmed or corrected; their side details are listed in section 5.

## 5. Remaining unverifiable items, and why

Load-bearing for the paper: none. Each item below is a side detail, marked in its document.

1. **Sculley (KDD 2007) found apple tasting weaker than greedy** (D:232, AL:46). The ACM full text returns 403 to scripted access. The claim is reported in Jiang et al. (AISTATS 2021, section 2), and the draft now cites it as second-hand. Read the paper (DOI 10.1145/1281192.1281258) before submission or drop the sentence; section 6.2's result does not depend on it.
2. **EgoPro-Bench uses 420 EgoBlind videos** (SG:66). Only a search snippet says so; the abstract does not.
3. **TIMELI venue against the ICRA 2026 proceedings** (SG:37). arXiv and the project page both say ICRA 2026; the proceedings were not checked.
4. **VIA-Agent participants' BLV status** (SG:53). Not in the abstract.
5. **ProactEgoMem annotation licence and "50 public dev clips"** (BM:17). In neither the paper nor the project page.
6. **VIABench licence for the underlying creator videos** (SG:43, BM:18). HF tag is MIT; no licence file on GitHub.
7. **Market side details** (MK): OOrion "coast is clear" wording, Meta Live AI battery and camera, Seeing AI Android polish, Guided Vision on Google eyewear, .lumen $9,999, Soundscape backend cost, AppleVis-only user reports (403 to scripts).
