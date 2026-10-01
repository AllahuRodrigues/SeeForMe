# SeeForMe — Progress Update 2

**CPSC 4900 Senior Project, Fall 2026**
Student: Alláh-u-Abhá Rodrigues · Advisor: Sohee Kim Park
Update written 2026-10-01. Supersedes the 2026-09-30 update (first-generation CACS results, now section 7 of the README).

---

## What changed since the last update

The thesis was reframed. "Reduce redundant audio" was no longer a defensible novelty on its own:
WalkVLM-LR (arXiv 2508.16070) attacks output redundancy directly, Google Lookout and Apple's Magnifier
already give distance to some objects, and Mobilio (2026) combines depth, sensor fusion and
personalised audio. The claim is now:

> SeeForMe introduces a **predictive, action-aware cue scheduler** that treats spoken feedback as a
> constrained communication channel. Rather than narrating detected objects, it estimates *when* a
> visual event becomes actionable, predicts its spatial and temporal relation to the user, and selects
> the minimum information required for timely action — or stays silent.

Everything below was built and measured in one day on top of the existing code base (not a restart).

## 1. CampusBench — a metric benchmark with communication windows

* Objects live in metres relative to a user walking at 1.1 m/s; boxes come from a pinhole camera;
  a noisy detector *and* a noisy metric depth sensor produce the observations.
* 11 scene classes: Harkness-style courtyard, hallway, stairs, doors, street crossing, finding
  (goal-directed: "find room 204 / the elevator / a seat"), crowd, temporary hazards, poor capture
  (pitch/yaw sway, blur, dark, camera covered), negative (busy scenes, nothing to say), changing
  (chair dragged into the path, pedestrian turning in, bike shooting out, door closing).
* Ground truth per event: should_speak, severity, **earliest useful / ideal / latest safe time**,
  path intersection, the minimum information slots and the ideal cue. Sudden events (visible only
  after their deadline) are flagged and reported separately. 165 scenarios, 831 should-speak events.
* New metrics: action-weighted recall, late critical rate, lead time before deadline, audio
  occupancy, critical-cue blocking, silence precision, minimum-message ratio, distance/TTC error,
  and communication **regret** against an annotation-driven reference scheduler.

## 2. P-CACS

Per tracked event: hazard = class severity × P(path intersection); urgency f(TTC); actionability;
goal relevance (FIND); novelty; uncertainty (detector × depth confidence); speech cost (seconds the
message occupies) and queue cost (what it would block). Four actions — SILENCE, SPEAK, INTERRUPT,
HAPTIC — under a token-bucket speech budget; a critical hazard inside its reaction horizon always cuts
a non-critical message. Messages are built from slots, so the scheduler decides *how much* to say:
"Stairs down, about 3 meters. Handrail left." / "Cyclist, crossing from the right." / silence for a
poster at 99% confidence. Distances are hedged ("close") when depth confidence is low.

Metric depth comes from Depth Anything V2 (metric, small): 33 ms per frame on this Mac, 48 ms
end-to-end with YOLO (21 FPS); the scheduler itself costs 0.01 ms.

## 3. Learned scheduler

A GRU over each track's last 8 decision steps (29 signals incl. channel state and budget) with action,
urgency and slot heads; trained by imitation of the reference scheduler with DAgger (3 iterations,
43k states), then REINFORCE in closed loop with the benchmark score as reward.

## 4. Results (held-out test split, 66 scenarios × 3 noise seeds)

| policy | action-weighted recall | late critical rate | cue precision | audio occupancy | regret |
|---|---|---|---|---|---|
| confidence threshold | 0.25 | 0.27 | 0.06 | 0.73 | 8.47 |
| confidence + cooldown | 0.26 | 0.27 | 0.07 | 0.64 | 7.32 |
| CACS (first generation) | 0.23 | 0.61 | 0.25 | 0.23 | 3.20 |
| **P-CACS linear (tuned)** | 0.48 | 0.28 | 0.55 | 0.21 | 1.41 |
| P-CACS GRU, imitation | 0.68 | 0.18 | 0.73 | 0.25 | 0.55 |
| **P-CACS GRU + REINFORCE** | **0.70** | 0.21 | **0.88** | 0.20 | **0.11** |
| reference scheduler | 0.73 | 0.24 | 1.00 | 0.21 | 0.00 |

* First-generation CACS is quiet but **late** (61% of critical events after the deadline): proximity
  from box size is not distance. Linear P-CACS doubles timely recall at the same airtime and halves
  lateness; its messages match the annotated minimum information 97% of the time.
* The learned scheduler nearly reaches the reference (0.70 vs 0.73) and wins where prediction from
  history matters: crowds 0.63 vs 0.00 (linear), changing scenes 0.45 vs 0.18, poor capture 0.48 vs
  0.18, doors 0.82 vs 0.38. Silence precision 0.90: when it says nothing, nothing was due.
* Ablations: depth and slot selection are load-bearing (−0.13 and −0.06 recall); budget, interrupt and
  haptic rules rarely bind at this operating point. TTC vs distance-only makes no difference yet —
  the moving-hazard classes are small and mostly *sudden*.
* Honest limits: it is a simulator; street crossings are hard for every policy (reference 0.38) because
  laterally crossing vehicles are visible for under a second; the reference scheduler is a strong
  heuristic, not an optimum (the learned policy beats it on validation by sending shorter messages).

## 5. Deployment

* **Web app** (`web/`): Vite + TypeScript PWA; YOLO11n via ONNX Runtime Web in a worker (WebGPU → WASM),
  tracker + dynamics + P-CACS ported to TypeScript; phone/tablet rear and front cameras, laptop and
  external webcams, video files; modes NAVIGATE / FIND / READ / ASK; QUIET–BALANCED–DETAILED; metres
  or feet; Web Speech or ElevenLabs; vibration + earcon for haptic cues; screen-reader live region;
  debug panel with distance / TTC / path / utility per track. Verified headless: browser detections
  match the Python detector box-for-box.
* **Server** (`server/app.py`): metric depth, OCR and ElevenLabs TTS for the browser; `--ssl` for phones.
* **ElevenLabs**: works with the free-tier default voice; key read from `.env` (gitignored).

## 6. Next steps

1. Record campus walks (Harkness, hallways, stairs, crossings) and annotate with
   `scripts/annotate_video.py` → score with `scripts/score_video.py --campus`.
2. Depth in the browser (quantised Depth Anything via transformers.js) to drop the server dependency.
3. Motion-aware handling of lateral crossings (the one scene type where everything is late).
4. Citation fixes from the 2026-09-18 check (AIDEN 2025, EgoBlind NeurIPS 2025, add WalkVLM-LR).

Reproduce: `make campus-all` (~30 min), `make web`, `make serve`. Full write-up:
`results/campus/CAMPUS_RESULTS.md`; figures `results/campus/figures/`; demo videos `results/campus/demo_*.mp4`.
