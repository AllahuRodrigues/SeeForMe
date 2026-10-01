# SeeForMe research plan (expanded, 2026-09-18)

This is the expanded plan written on 2026-09-18. The first section records a
citation check made the same day so the related-work paragraphs can be reused in
the thesis safely; the plan text itself follows unchanged, and the last section
maps every plan item to what already exists in this repository.

## Citation check (2026-09-18)

| claim in the plan | status | correct citation |
|---|---|---|
| AIDEN "(2023–24)", YOLO + OCR + LLaVA, Ionic/TypeScript, server offload, ~10 s latency, TAM questionnaires | **Real system, wrong year and stack detail.** The paper is from November 2025. The client is Ionic 6 + Capacitor 4 + Vue.js 3, with YOLOv8 and LLaVA on a central server; OCR is done by prompting LLaVA. TAM evaluation and the size-based object-centring guidance are in the paper. The ~10 s and 0.5 s latency figures were not verified. | *AIDEN: Design and Pilot Study of an AI Assistant for the Visually Impaired*, arXiv:2511.06080, 2025. https://arxiv.org/abs/2511.06080 |
| EgoBlind (2025), first-person VideoQA from blind users, best model ≈59 % vs ≈87 % human, multi-frame context helps, "act as a visual assistant" prompting helps | **Real, numbers consistent.** 1,392 videos, 5,311 questions, 16 MLLMs evaluated, best ≈60 % vs 87.4 % human; navigation is the hardest category. Verify the two prompting/multi-frame claims against Section 5 of the paper before quoting them. | Xiao et al., *EgoBlind: Towards Egocentric Visual Assistance for the Blind*, NeurIPS 2025 (Datasets & Benchmarks), arXiv:2503.08221. https://arxiv.org/abs/2503.08221 |
| "2025 arXiv paper" fine-tuning a VLM with a spatial-reasoning module and a camera attachment, evaluated on VizWiz | **Real work, but a Springer chapter, not arXiv.** | *A Light and Smart Wearable Platform with Multimodal Foundation Model for Enhanced Spatial Reasoning in People with Blindness and Low Vision*, Springer LNCS, 2025. https://link.springer.com/chapter/10.1007/978-3-031-92591-7_20 |
| (not in the plan, should be added) | Directly about the redundancy problem this thesis attacks: reducing redundant spoken output of VLM walking assistants on edge devices. | *Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants*, arXiv:2508.16070, 2025. https://arxiv.org/abs/2508.16070 |
| (not in the plan, optional) | LoRA fine-tuning of BLIP-2 to generate step-by-step navigation instructions. | *Fine-Tuning Vision-Language Models for Visual Navigation Assistance*, arXiv:2509.07488, 2025. https://arxiv.org/abs/2509.07488 |

Everything else in the plan (VizWiz, Seeing AI, YOLO, EasyOCR) was already
cited in the original proposal.

## Plan text

### Background & Related Work

Recent years have seen rapid progress in AI-powered assistive vision for blind and low-vision users. Smartphone apps like Microsoft's Seeing AI and others combine object detection and OCR to describe scenes, but typically report everything they see, leading to overwhelming audio. Research systems have begun to add context-awareness. For example, the AIDEN system integrates YOLO-based object detection and OCR with a large vision-language model (LLaVA) to answer visual questions. It runs on a hybrid mobile/web app (built with Ionic/Capacitor/Vue) that offloads heavy inference to a server. In user tests, AIDEN provided scene descriptions and object-finding guidance (prioritizing nearby objects by size) and evaluated user acceptance via standardized questionnaires. This demonstrates a trend toward combining on-device detectors with offloaded AI for richer assistance.

On the research front, new datasets and benchmarks highlight the need for context. The EgoBlind project (2025) introduced a first-person video QA dataset from real blind users. It categorizes everyday tasks (reading signs, safety warnings, navigation, etc.) and shows that even powerful multimodal LLMs struggle on assistive queries: the best model achieved only about 60 % accuracy versus 87.4 % for humans. The authors also found that providing multi-frame context improves performance on navigation/safety tasks, and that explicitly prompting models to "act as a visual assistant" (focusing on user intent) helps guide answers. These findings underscore two points: (1) purely reactive description is insufficient; models must understand which information is important to the user's current goal, and (2) temporal history (via tracking) can boost understanding for ongoing tasks.

Other recent work has begun to exploit foundation models for assistive reasoning. For example, a 2025 wearable platform fine-tunes a multimodal foundation model with a spatial-reasoning component to improve navigation and object-location advice, and reports gains on VizWiz. In summary, prior systems illustrate the capability of modern AI (YOLO detectors, OCR engines, large vision-language models) and motivate our focus on the complementary problem: deciding what to say and when to say it.

### System Architecture and Prototype

Our system captures live video (from a smartphone or wearable camera), runs on-device or edge ML for perception, and selects audio cues via the Context-Aware Cue Selection (CACS) layer. Key components:

* **Object & text perception.** A lightweight object detector (YOLO11n/s) recognises objects; an OCR module (EasyOCR) reads signage and text. These yield candidate detections with class labels, confidence scores and bounding boxes.
* **Multi-object tracking.** Object tracks are maintained over time (IoU-based association). Each track stores the object's label, confidence, age, location and last announcement time. This temporal memory avoids repeating the same cue every frame and provides motion context (an object approaching).
* **Contextual features.** For each tracked object: (1) confidence, (2) relevance, (3) novelty, (4) proximity, (5) text importance, (6) duplicate penalty. These are combined (initially by a weighted linear rule) into a cue score.
* **Cue selection.** At each decision step all candidates are ranked by score and the top one is spoken if it exceeds a threshold. Baselines speak every detection above a fixed confidence, or add a simple cooldown.

Implementation: Python/PyTorch for the research pipeline. For mobile/web deployment we will explore cross-platform solutions (React Native or Ionic for UI and camera; ONNX Runtime Web, TensorFlow Lite or Core ML for inference; native Swift/ARKit or Kotlin/ML Kit as alternatives). AIDEN offloaded heavy vision/LLM processing to a server; we will benchmark local and remote inference to decide the split.

### Data Collection (Supervised & Unsupervised)

* **Synthetic scenarios.** 100+ simulated scenarios across hallways, offices, streets, cluttered rooms, low light and text corridors, with ground-truth event labels, controllable detection noise and occlusion. A graphics engine (Unity, Unreal, AI Habitat) is a possible upgrade from the current detection-level simulator.
* **Real-world data.** Smartphone recordings in controlled environments, annotated post hoc for important cues; VizWiz text images to extend the text-importance classifier; Ego4D or EgoBlind for unsupervised pre-training.
* **Unsupervised context features.** Cluster scenes by their detection histograms so that objects rare for the current scene type receive higher relevance.

Train/val/test splits so every learned component is evaluated on held-out data.

### Machine Learning & Cue Models

1. **Heuristic CACS (manual weights).** S = αC + βR + γN + δP + ηT − λD, coefficients tuned on the training set.
2. **Supervised learning.** Logistic regression, gradient boosting and a small MLP predict whether a candidate cue should be spoken, learning the weighting from data.
3. **Deep context models.** A sequence model (GRU/Transformer) over the recent history of each track's features decides "speak now or not", trained by imitation of an oracle (with on-policy data aggregation) and fine-tuned with a reward that balances informative cues against brevity. Fine-tuning a vision-language model (LLaVA / InternVL) on cue-selection data is a speculative extension.

Efficiency: lightweight models run on-device at tens of FPS; heavier models could run on edge servers. Target per-frame processing < 0.2–0.3 s.

### Evaluation & Benchmarks

* **Synthetic benchmark.** Important-cue recall, cue precision, audio load, duplicate rate, critical delay on held-out scenarios; baselines vs CACS vs learned policies; ablations.
* **Real data testing.** Sighted annotators label important cues in real recordings; precision/recall of the spoken cues against those labels.
* **Latency & resources.** End-to-end delay for detection, OCR and cue selection on target hardware; target ≥ 5–10 FPS.
* **User study.** Blind or low-vision participants perform daily tasks with and without CACS; task performance plus SUS/TAM questionnaires; IRB approval and informed consent. (Outside the guaranteed one-semester deliverables, as stated in the proposal.)

### Deployment & Future Work

Mobile app (Core ML / TensorFlow Lite on-device, optional server assistance), browser demo (WebRTC camera + ONNX Runtime Web), wearable integration with spatial audio. Further ideas: RL-based dialogue where the assistant asks the user to prioritise, multimodal fusion with ultrasound/radar, crowdsourced active learning from user corrections.

## Where each plan item stands in this repository

| plan item | status | where |
|---|---|---|
| YOLO + OCR perception, IoU tracker, six CACS features, threshold cue selection, baselines | done | `seeforme/perception`, `seeforme/tracking`, `seeforme/cues` |
| Synthetic scenarios with ground truth, noise, splits | done (detection-level simulator, 90 scenarios) | `seeforme/sim/scenario.py`, `data/scenarios/` |
| Unsupervised scene clustering → rarity signal | done | `seeforme/ml/scene_context.py` |
| Manual weights tuned on train split | done | `scripts/tune_cacs.py`, `configs/cacs_tuned.yaml` |
| Supervised logreg / GBM / MLP | done | `seeforme/ml/priority_model.py` |
| Deep context model (GRU), imitation + on-policy aggregation, reward fine-tuning | done 2026-09-18 | `seeforme/ml/sequence_model.py`, `seeforme/ml/onpolicy.py`, `scripts/train_deep_context.py` |
| VLM fine-tuning (LLaVA / InternVL) | not started (needs GPU + real data) | — |
| Synthetic benchmark metrics, ablation | done | `seeforme/eval`, `results/RESULTS.md` |
| Real-video scoring pipeline | ready, no recordings yet | `scripts/run_video.py`, `scripts/score_video.py`, `docs/annotation_format.md` |
| Latency on this machine | done | `results/latency.json` |
| Mobile export (ONNX / Core ML) | done 2026-09-18 | `scripts/export_models.py`, `models/export/` |
| Browser demo, native mobile app, user study | web app done 2026-10-01 (`web/`, PWA; phone/tablet/laptop cameras; server for depth/OCR/TTS); native app and user study not started | `web/`, `server/app.py` |

## P-CACS extension (2026-10-01)

| item | status | where |
|---|---|---|
| Metric world simulator with cue windows (earliest / ideal / latest safe), severity, minimum information, ideal cue — **CampusBench**, 11 scene classes incl. Harkness-style courtyard, stairs, doors, crossings, finding, crowds, temporary hazards, poor capture, negatives, changing scenes | done | `seeforme/campus/`, `docs/campus_annotation.md`, `data/campus/` |
| Depth + dynamics: metric distance, relative speed, TTC, lateral position, path-intersection probability per track | done | `seeforme/geometry/`, `seeforme/perception/depth.py` (Depth Anything V2 metric small) |
| P-CACS scheduler: hazard × f(TTC) × actionability + goal − uncertainty − speech cost − queue cost; SILENCE / SPEAK / INTERRUPT / HAPTIC; token-bucket audio budget; slot-based minimal messages | done | `seeforme/cues/pcacs.py`, `seeforme/cues/compose.py` |
| New metrics: action-weighted recall, late critical rate, lead time, audio occupancy, critical blocking, silence precision, minimum-message ratio, distance / TTC error, communication regret vs an annotation-driven reference scheduler | done | `seeforme/campus/metrics.py` |
| Learned scheduler: GRU over the last 8 steps (29 signals), action + urgency + slot heads, DAgger imitation then REINFORCE with the benchmark score as reward | done | `seeforme/ml/scheduler_model.py`, `scripts/train_scheduler.py` |
| Live stack with depth and ElevenLabs voice | done | `scripts/run_live.py`, `seeforme/pipeline.py::LivePipeline`, `seeforme/feedback/elevenlabs_tts.py` |
| Real recordings annotated in the CampusBench format | drafting tool ready, no recordings yet | `scripts/annotate_video.py`, `scripts/score_video.py --campus` |
| EgoBlind / VizWiz for the decision layer | not started (EgoBlind needs an access request and cue-window labels) | — |
