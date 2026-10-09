# SeeForMe on Meta glasses: integration design (prototype track, separate from the research baseline)

Status 2026-10-08: design only. Nothing here has run on glasses. The research baseline (CampusBench v2,
EgoBlind) does not depend on it, which is the point: the glasses are a camera and a speaker, the decision
layer stays the same.

## What the toolkit gives us (from Meta's developer pages, checked 2026-10-08)

* Meta Wearables Device Access Toolkit (DAT): a mobile SDK (iOS `facebook/meta-wearables-dat-ios`,
  Android `facebook/meta-wearables-dat-android`) through which a phone app opens a *session* with the
  glasses and receives the camera stream or single photos, receives microphone audio (8 kHz mono over the
  Bluetooth hands-free profile) and plays audio through the open-ear speakers via the normal phone audio
  path. Developer preview since 2025-12-04; publishing to the public is still limited to Meta's partners,
  builds for testers inside the developer's own organisation are allowed.
* Registration is a one-time deep link into the Meta AI app; the first camera use asks the wearer for
  permission (allow always / once / deny). One session per device; closing the hinges, taking the glasses
  off or tapping them pauses, resumes or ends the session, and the app is told about it.
* Apps cannot define their own tap/swipe gestures. Meta AI is not available inside a session: bring your
  own model. Meta Ray-Ban Display can show simple layouts (text, image, button, icon) on the right lens
  through `MWDATDisplay`; the Gen 1 / Gen 2 Ray-Ban Meta have no display.
* A Mock Device Kit simulates the glasses for development; supported firmware and Meta AI app versions
  are on the toolkit's Version Dependencies page and change between releases.

Sources: Meta's announcement (developers.meta.com/blog/introducing-meta-wearables-device-access-toolkit),
the integration overview (wearables.developer.meta.com/docs/develop/dat/build-overview) and the FAQ
(developers.meta.com/wearables/faq).

## Architecture

```
glasses camera  --(DAT video stream, phone)-->  iOS app
                                                 |  CoreML YOLO11n (or YOLO-World) @ 5-10 fps on the phone
                                                 |  bbox-range + optional metric depth (server) 
                                                 |  IoU tracker + TrackDynamics  (port of seeforme/tracking)
                                                 |  P-CACS scheduler              (port of seeforme/cues/pcacs.py, 29 features, 4 actions)
                                                 |  compose -> TTS (AVSpeech or ElevenLabs) --> glasses speakers
                                                 |  haptic: phone Taptic Engine (the glasses have none)
glasses mics    --(HFP audio)------------------>  wake word / ASK and FIND questions (on-device speech)
```

* **Perception on the phone**, not on the glasses: the toolkit streams video to the phone; the glasses
  run nothing of ours. `scripts/export_models.py` already produces the ONNX and Core ML artefacts
  (`models/export/`), and the web app's detector worker is the reference for the pre/post-processing.
* **Decision layer unchanged**: P-CACS is ~900 lines of Python with no weights beyond `configs/pcacs_tuned.yaml`
  (the learned scheduler is a 14k-parameter GRU, exportable to Core ML too). Porting it to Swift is a
  mechanical job and the CampusBench replay tests (`tests/test_campus*.py`) become the conformance suite:
  the Swift port must reproduce the Python decisions on the serialised scenarios.
* **Channel model**: the glasses' speakers are an ordinary Bluetooth audio route, so the `SpeechChannel`
  semantics (one message at a time, interrupt only a less severe message, minimum gap) are enforced by
  the app, exactly as in the browser demo (`web/src/speech.ts`).
* **Latency budget** (from the glasses' camera to speech start): stream + decode on the phone, detector
  (YOLO11n Core ML on an A17/M-class phone: tens of ms), tracker and scheduler (sub-millisecond), TTS
  onset (AVSpeech ~100-200 ms, ElevenLabs streaming ~300-600 ms). Measure with `scripts/benchmark_latency_pcacs.py` (`make latency-pcacs`)
  once a stream is available; the EgoBlind windows assume the cue is heard >= 2 s before the user would
  have asked, so a 0.5 s end-to-end latency is acceptable and 2 s is not.

## What to build first (one to two weeks of app work, after the research baseline is frozen)

1. **Mock Device Kit loop**: DAT session + video stream into the Swift tracker/scheduler port, replaying
   the EgoBlind demo clips through the mock device. Deliverable: the three demo videos in
   `results/egoblind/demo_105_*.mp4` reproduced on the phone with the same decisions.
2. **Real glasses, lab walk**: Ray-Ban Meta Gen 2, a 10-minute indoor walk, log every decision with the
   29-feature vector (same format as the Python `DecisionRecord`) so the walk can be replayed offline
   and scored against a human annotation in `docs/campus_annotation.md` format.
3. **ASK / FIND by voice**: wake word on the HFP audio, question to the rule-based answerer first (as in
   the web app), later a small VLM on the phone.
4. **Battery and heat**: continuous streaming is the known cost of the toolkit; measure minutes of session
   per charge at 5 fps and at 2 fps with the scheduler deciding when to raise the rate (a hazard in the
   window warrants 10 fps, an empty hallway does not).

## Risks and honest limits

* Publishing is partner-gated during the preview; a public beta on glasses is not possible until Meta
  opens distribution. Internal testers and the lab study are possible now.
* No haptics on the glasses: the `haptic` action goes to the phone or a wrist device, which changes the
  CampusBench assumption that a haptic cue is felt immediately.
* The toolkit's 8 kHz microphone audio is adequate for wake words and short questions, not for OCR-like
  tasks or music.
* Nothing in this document is a claim that the system is safe for independent navigation; the beta
  readiness notes in `web/README.md` apply unchanged.
