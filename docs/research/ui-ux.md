# SeeForMe UI/UX and accessibility research (2026-11-12)

Scope: published studies, Apple HIG, community reports, and a read-only audit of `web/` (no files modified). Nothing was committed or written to the repo. Evidence tags: **[Study]** peer-reviewed or arXiv, **[HIG]** Apple guidelines, **[WCAG]**, **[Community]** forums or press, **[Audit]** this repo's code, **[Own data]** numbers from `docs/online.md`.

---

## 0. Two corrections for the hallucination checker (they affect the thesis claim)

1. **"Lookout answers only when asked" is not accurate.** Google's own help page describes Lookout's Explore mode (beta) as "point your camera around you to hear what's in your environment", and AFB's 2019 review says it describes objects as the camera moves. https://support.google.com/accessibility/android/answer/9031274 , https://www.afb.org/aw/22/5/17557
2. **Seeing AI also speaks without being asked in some channels.** Short Text starts reading as soon as text is in view (AppleVis user reports: https://applevis.com/comment/177856). The World channel (LiDAR, Pro iPhones) announced objects with spatial audio from launch in December 2020 (https://www.windowscentral.com/microsofts-seeing-ai-app-now-supports-lidar-scanners-iphone-12). Whether it still exists after v5.5 is **unverified**; AppleVis users disagree (https://www.applevis.com/forum/ios-ipados/seeing-ai-where-world-channel).

**Suggested rewording:** existing products either answer on demand or narrate whatever the detector sees, gated only by content. None of them decides *when* to speak under a speech budget, or learns that timing from the user's own feedback. The literature backs this framing directly. The Michigan study of ChatGPT live video lists "misaligned intervention timing" as one of three core bottlenecks and names "determining appropriate intervention timing beyond turn-taking" as a design implication (Guo lab, arXiv 2508.03651, 8 BVI participants: https://arxiv.org/abs/2508.03651). VIABench (arXiv 2607.14660, July 2026) has a "Proactive Reminder" task that scores speaking up versus staying silent (https://huggingface.co/papers/2607.14660). I only read its abstract-level summary, so its details are **unverified**.

**Flag for the algorithm and code agents:** the PWA's default scheduler is "Learned (default)" = GRU + RL ONNX (`web/index.html:69`, `web/src/learned.ts:1-6`, `web/public/models/scheduler_rl.onnx`). If that is the frozen October RL model, it had the **lowest** measured coverable recall on EgoBlind (0.050 against 0.087 for P-CACS) [Own data]. That needs confirming with the peer session that owns `web/`. If it is the same model, users get the worst measured policy by default.

---

## 1. What BLV pedestrians want, what annoys them, what is missing

### Wants
- **Control over verbosity, adjustable on the fly.** CHI '26 "Say It My Way" (11 blind participants, 10-day diary with Be My AI, 418 interactions): participants tried to manage verbosity, steer attention to the details that mattered for the task, and build trust in the output. They asked for a "hurry mode", spatial references relative to their own body rather than the image frame, camera-framing guidance, and preferences that persist across sessions. https://arxiv.org/abs/2602.16930
- **Short answers.** The same UMD study found that 10 to 20 word questions drew answers more than 10 times longer, and the app always opened with a full image description. https://letsdatascience.com/news/umd-study-finds-be-my-ai-gives-blind-users-overlong-answers-36ef59a5
- **Verbosity tuned to the person.** Navigation instructions should adapt to expertise; the authors argue for personalised verbosity that keeps users in context with minimal distraction (Ahmetovic et al., W4A 2019). https://ri.cmu.edu/?p=121341
- **Control over when and how much information arrives (push versus pull), and earcons in place of speech to cut overload** (ASSETS 2013, DOI 10.1145/2513383.2513445). I saw this only through a third-party listing; verify against the ACM DOI. https://a11y-paradise.onrender.com/reviews/69d1c549dc41242cf4def027
- **Distance and direction, not just presence.** One AppleVis user (June 2026) said Meta glasses tell you something is nearby but not how far away it is. Another said vision is needed for unmarked crossings and roundabouts. https://mail.applevis.com/forum/ios-ipados/where-do-current-navigation-object-detection-tools-fall-short-you
- **Units people actually use.** Soundscape switched from yards to feet and miles after user feedback. https://mspoweruser.com/microsoft-updates-soundscape-visual-assistance-app-with-more-capabilities
- **A way to correct the AI and have it learn.** ASSETS '24 "Misfitting With AI" (26 BLV users): participants wanted to contest outputs directly, but the tools did not improve after correction. They verify through touch, sound and sighted people, and keep AI away from high-stakes tasks. https://arxiv.org/abs/2408.06546
- **Hands-free use with a chest-worn phone.** Mobilio (Nature Biomedical Engineering, 2026) used a chest-worn phone and needed about 72% fewer verbal corrections than Google Maps plus a cane (https://www.natureasia.com/en/info/press-releases/detail/9426). A chest-mounted camera with wrist haptics cut collisions by 37% (Mass Eye and Ear, JAMA Ophthalmology 2021, n=31: https://masseyeandear.org/news/press-releases/2021/07/wearable-devices-can-reduce-collision-risk-in-blind-and-visually-impaired-people).
- **To complement the cane or guide dog, not replace it.** Users in the GuideTouch study found replacement "highly undesirable" (https://arxiv.org/pdf/2601.13813v1). Soundscape's designers make the same point and stress that users must also hear the environment (https://news.microsoft.com/?p=404434).

### Annoyances
- **Verbosity and overlong answers.** See Say It My Way and the UMD coverage above.
- **Alert fatigue.** In a BuzzClip versus NOA comparison, frequent alerts demanded constant processing and caused mental fatigue; NOA's more selective AI filtering helped. https://www.medrxiv.org/content/10.1101/2025.02.20.25322389.full.pdf
- **Hallucinations, generic answers and overly agreeable replies in dynamic scenes, plus no visual memory** (ChatGPT live video, arXiv 2508.03651, link above).
- **Battery.** AppleVis users reported Seeing AI draining 10% in 5 minutes and 91% down to 46% in about 15 minutes. These are anecdotal 2017 reports. https://applevis.com/forum/ios-ipados/seeing-ai-battery-drain
- **Haptics need a free hand.** StreetNav participants valued haptics in noise, but phone haptics only work when the phone is in the hand, and one hand always holds the cane. They suggested a watch instead. https://arxiv.org/pdf/2310.00491
- **Latency.** I found no BLV-specific latency threshold study. **Unverified.** A local measure is needed (the code already drops any cue that has not started 2 s after issue, `web/src/speech.ts:7`).

### Haptic versus audio
- In an ICCHP 2014 study, blind participants clearly preferred haptic over audio for short-range obstacle information, because cane users already get short-range information through touch. https://cvhci.iar.kit.edu/~dkoester/publications/martinezICCHP2014tlx.pdf

### Not available in any product (the gap)
1. A budgeted decision about when to speak, in general pedestrian scenes (sources in section 0).
2. Learning from the user's own feedback, kept across sessions (Misfitting; Say It My Way).
3. Visual memory, so the same thing is not re-announced (ChatGPT live video gaps). This matches our own 28.5% re-announcement rate for the IL model [Own data].
4. Metric distance plus a body-relative direction, delivered hands-free (AppleVis 2026; Say It My Way).

### Urgent, in order
Proactive timing with feedback, then concise output, then hands-free use, then trust and verification cues. Policy context: NFB Resolution 2024-08 concerns regulating AI in visual interpretation; only its title was confirmed, its text is **unverified** (https://nfb.org/node/15858). AFB reports 79% of BLV respondents use AI visual descriptions daily or weekly; this comes from a search snippet and the PDF could not be parsed, so treat the number as **unverified** (https://afb.org/sites/default/files/2026-08/Innovation%20for%20Access%20-%20AI%20Assistive%20Technology%20Report%20-%20Accessible.pdf). An AFB survey found 69.2% prefer AI or human help "depending on the task", 28.2% prefer humans and 2.6% prefer AI (https://afb.org/aw/fall2024/ai-usage-research). I found no r/Blind threads in search, so community evidence here is AppleVis only.

---

## 2. Apple HIG and VoiceOver requirements that apply

Sources: https://developer.apple.com/design/human-interface-guidelines/accessibility , https://developer.apple.com/design/human-interface-guidelines/voiceover , https://developer.apple.com/design/human-interface-guidelines/playing-haptics

- **Controls:** iOS default size 44x44 pt, minimum 28x28 pt. Spacing matters: about 12 pt around controls with a bezel.
- **Text:** default 17 pt, minimum 11 pt. Support at least 200% enlargement or Dynamic Type. On the web this means `font: -apple-system-body` plus rem units; support is flaky by report, so test on device (https://dev.to/colingourlay/how-to-support-apple-s-dynamic-text-in-your-web-content-with-css-40c0).
- **Contrast:** 4.5:1 for text up to 17 pt; 3:1 at 18 pt or bold. Check both light and dark appearances, and offer higher contrast when Increase Contrast is on.
- **Audio and haptics together:** pair audio cues with matching haptics and visual cues. Use haptics consistently, keep them short for discrete events, don't overuse them, and make them optional. Make sure they don't disturb the camera.
- **Timers:** keep time-boxed UI to a minimum; dismiss on an explicit action instead of auto-dismissing.
- **Playback control:** let people start and stop audio playback.
- **Gestures:** use the simplest gestures and offer on-screen alternatives. Support Voice Control (labels must match visible text), Siri and Shortcuts, and the Action button.
- **Motion:** respect Reduce Motion.
- **VoiceOver:** label every key element, hide decorative elements, use headings and a clear hierarchy, group related elements, announce layout changes, and support the rotor.
- **Web platform limits:**
  - `navigator.vibrate` is not supported in iOS Safari (https://caniuse.com/vibration). WebKit's position is "oppose" (https://web-platform-dx.github.io/web-features-explorer/features/vibration).
  - Screen Wake Lock works in Safari from 16.4. Home Screen web apps were fixed in iOS 18.4 (https://bugs.webkit.org/show_bug.cgi?id=254545).
  - Media Session action handlers are supported from Safari 15 (https://web.dev/articles/media-session). Whether AirPods or headset button presses reach them on iOS is **unverified** and needs a device test.
  - VoiceOver's own sounds and haptics are system-controlled; apps cannot choose them (https://developer.apple.com/forums/thread/773182).
  - Magic Tap for web content is **unverified**.
- **WCAG 2.1.4 Character Key Shortcuts:** single-key shortcuts need a way to turn off or remap them. https://www.w3.org/WAI/WCAG21/Understanding/character-key-shortcuts.html

---

## 3. Audit of the PWA (`web/`, read-only)

All paths are relative to `<repo>/`. Severity runs **H** (high), **M** (medium), **L** (low).

| # | Sev | Where | Problem |
|---|---|---|---|
| A1 | H | `web/src/main.ts` (no feedback UI anywhere) | No "useful / not useful" control, no "repeat" or "quiet" button. The thesis feedback channel does not exist in the product, and the only miss signal is the Ask box. |
| A2 | H | `web/src/speech.ts:94-96`, `web/src/main.ts:266,268` | `navigator.vibrate` does nothing on iPhone. The "BUZZ obstacle imminent" cue falls back to one 660 Hz earcon (`speech.ts:99-107`). Interrupt cues get `haptic([40])` and no sound at all, so on iOS there is no non-speech signal for them. |
| A3 | H | `web/src/main.ts` (no `wakeLock` or `visibilitychange`) | A chest-mounted phone will auto-lock, which stops the camera without warning. Nothing announces "paused" or resumes. |
| A4 | H | `web/index.html:69` | The default scheduler is the RL model. See section 0; it may be the worst policy measured on real video. |
| A5 | M | `web/index.html:83-125`, `web/styles.css:66` | Below 900 px the research aside (formula, 9-column table, debug log, stats, benchmark) sits in the VoiceOver reading order after the controls. That is dozens of swipes of jargon. The table headers (`index.html:88`: "P", "H", "urg", "N", "U") have no `scope` or expansion. |
| A6 | M | `web/src/main.ts:312`, `main.ts:345` | Both run `innerHTML` rebuilds. The candidate table is rewritten every decision (every 0.5 s, `pcacs.ts:84`), so VoiceOver focus inside it is lost. Separately, OCR text from the scene is interpolated unescaped (`main.ts:315`): an HTML injection path from camera text, which is a security issue for the code agent. |
| A7 | M | `web/src/styles.css:2` | Dark mode is overridden to `--bg: #ffffff`, so there is no dark theme for light-sensitive users. There is no `prefers-contrast` handling, and `theme-color` is white (`index.html:6`, manifest). |
| A8 | M | `web/src/styles.css:4,8,19,30,39,55,63` | Fixed px type throughout: a 15 px base and many 11 px labels. No Dynamic Type (`-apple-system-body`) and no rem scaling, against HIG's 17 pt default. |
| A9 | M | `web/src/styles.css:35` | White text on amber in the "Listening…" state is 3.19:1 at 16 px semibold. Fails 4.5:1. |
| A10 | M | `web/src/styles.css:43` | Amber "haptic" status text on `#fafafa` is 3.05:1 at 15 px. Fails. |
| A11 | M | `web/src/styles.css:22,28` | Button and input borders `#e6e8eb` on white are 1.23:1. Fails the 3:1 non-text contrast for control boundaries (WCAG 1.4.11). The bench bars `#9ca3af` on `#e5e7eb` are 2.05:1 (`styles.css:60`). |
| A12 | M | `web/src/main.ts:494-502` | Single-key shortcuts (Space, F, M, A) fire whenever focus is on the body, with no way to turn them off. Violates WCAG 2.1.4. |
| A13 | M | `web/index.html:45,54,63`; `web/src/main.ts:483,487` | Four controls are named "Ask" or "Ask by voice". While listening, the visible text changes to "Listening…" but the accessible name stays "Ask by voice", so the state is not exposed (no `aria-pressed` or `aria-busy`). |
| A14 | M | `web/src/main.ts:457-462` | Switching modes reveals a form without moving focus or announcing the change. HIG says to inform VoiceOver of layout changes. |
| A15 | M | `web/src/speech.ts:62-65` | Speech rate is fixed at 1.05 and the voice is a hard-coded preference list. There is no rate setting; experienced screen-reader users' fast-rate preference is **unverified** and needs our own survey. |
| A16 | M | `web/src/main.ts:207-234`, `web/src/camera.ts:17` | Detection runs on every animation frame with no throttle. The camera is requested at 1280x720, 30 fps, though frames are downscaled to 640 wide. Depth is called at about 4 Hz and OCR every 2 s, with no stationary or low-power duty cycle and no pause when hidden. This is a battery cost (AppleVis drain reports above). |
| A17 | L | `web/src/main.ts:277` | The cue card auto-hides after 2.6 s (HIG: keep time-boxed UI to a minimum). This is mitigated by the persistent `#last-cue` line (`index.html:79`). |
| A18 | L | `web/src/main.ts:45`; `web/index.html:47` | With the app voice on and VoiceOver on, two voices can overlap; the web cannot detect VoiceOver. "Mute" really means "send cues to the screen reader", but the label doesn't say so. |
| A19 | L | `web/index.html:71`; `web/src/compose.ts:28-32` | Meters is the default for en-US users. Directions are only ahead, slightly left/right and on your left/right; there is no clock-face option. Say It My Way supports body-relative references; clock-face preference is **unverified**. |
| A20 | L | `web/src/styles.css:56` | The bench select's min-height of 34 px is under 44 pt. Buttons and inputs at 42 px (`styles.css:22,28`) are just under the 44 pt default. |
| A21 | L | `web/src/styles.css:65` | `.sr-only` lacks `margin:-1px; padding:0; border:0`. This is a minor robustness issue. |

Good already: `aria-live` split into polite and assertive regions, with hazards sent to assertive (`index.html:40-41`, `main.ts:42-45`). Per-frame diagnostics are `aria-hidden` (`main.ts:300`). There is a skip link, a Reduce Motion rule (`styles.css:75`), zoom is not blocked (`index.html:5`), and a clear safety notice (`index.html:22-26`). Late cues are dropped rather than spoken late (`speech.ts:5-7`).

---

## 4. ASAP top 10 UX changes (prioritised)

1. **Build the feedback channel into the live app.**
   - After each cue, offer two large targets, "Useful" and "Not useful", plus one full-width "Quiet / repeat" zone that works one-handed while the phone is chest-mounted.
   - Log taps and asked questions locally, as the online learner expects.
   - Try headset buttons through Media Session (needs a device test).
   - Evidence: [Audit A1], the thesis design, [Study] Misfitting (users want to contest outputs), [Study] Say It My Way (preferences don't persist).
2. **Keep the session alive and tell the user when it stops.**
   - Use the Screen Wake Lock API.
   - On `visibilitychange`, announce "Camera paused" and resume cleanly.
   - Add a one-screen onboarding step for chest mounting.
   - Evidence: [Audit A3], [Community/WebKit] bug 254545 (fixed in iOS 18.4), [Study] Mobilio and the Mass Eye and Ear chest-camera studies.
3. **Make the live default the best-measured policy, not the RL model,** pending confirmation that `scheduler_rl.onnx` is the October RL model. Evidence: [Audit A4], [Own data] 0.050 for RL against 0.087 for P-CACS; the online learner reached 0.20.
4. **Replace the dead iPhone vibration with an earcon language.**
   - Distinct short earcons per cue class.
   - Stereo-panned by direction.
   - Played for interrupts too.
   - Real haptics go on the native iOS roadmap.
   - Evidence: [Audit A2], [WCAG/caniuse] vibrate unsupported, [HIG] pair audio with haptics and keep them consistent, [Study] ICCHP 2014 (blind participants prefer haptics), [Study] ASSETS 2013 (earcons reduce overload).
5. **Put verbosity, speech rate, units and voice up front, and persist them.**
   - Add a speech-rate slider and use the system voice by default.
   - Default to feet for en-US.
   - Add a "hurry" toggle.
   - Evidence: [Audit A15, A19], [Study] Say It My Way, [Study] Ahmetovic W4A 2019, [Community] Soundscape units change, [Community] UMD finding of answers over 10 times longer.
6. **Add instant "stop talking" and "repeat last" controls** as big single targets, keeping Space and M on desktop. Evidence: [HIG] let people control audio playback, [Audit A1, A17], [Study] Say It My Way.
7. **Take research diagnostics out of the phone reading order.** Move them to a separate "Research" page, or a collapsed `<details>` that is closed by default on narrow screens. Stop rebuilding the table with `innerHTML` and escape OCR text. Evidence: [Audit A5, A6], [HIG] VoiceOver hierarchy and decorative content.
8. **Support Dynamic Type, dark mode and contrast.**
   - `-apple-system-body` with rem units.
   - A real dark theme and `prefers-contrast`.
   - Fix A9, A10 and A11.
   - Evidence: [HIG] 17 pt default, 200% enlargement, 4.5:1 contrast, [Audit A7-A11].
9. **Fix accessible names and state.**
   - Rename the duplicate "Ask" controls.
   - Expose the listening state.
   - Move focus or announce on mode change.
   - Explain "Mute" as "use VoiceOver's voice instead".
   - Add a toggle for single-key shortcuts.
   - Evidence: [Audit A12-A14, A18], [WCAG 2.1.4], [HIG] Voice Control labels.
10. **Add a battery-aware duty cycle.**
    - Request the camera at 640 px.
    - Lower detector fps when stationary.
    - Pause depth and OCR when idle or hidden.
    - Add an optional low-power mode.
    - Evidence: [Audit A16], [Community] AppleVis Seeing AI drain reports.

### Gaps needing our own data (cannot be sourced)
- An acceptable cue latency for BLV pedestrians.
- Preferred speech rate.
- Clock-face versus left/right directions.
- Whether headset buttons reach the Media Session handlers on iOS.
- How VoiceOver and Web Speech coexist on iOS.

All of these suit a short survey plus a device test with participants at M5. They are **unverified** until then.
