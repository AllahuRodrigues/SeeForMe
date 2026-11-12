# SeeForMe market map, 2026-11-12 (Market Researcher)

Scope: what exists in 2026 for blind and low-vision (BLV) visual assistance and mobility, which products speak unprompted, who pays, and where SeeForMe can defend a position. Every external claim has a URL. Claims I could not confirm from a primary or dated source are marked **unverified**. Nothing in the repo was changed.

## 1. Main findings

1. **General-purpose AI assistants still speak only when asked.** That covers Be My Eyes / Be My AI, Seeing AI, Lookout, Envision Ally, Meta AI and Live AI, and Aira. Meta's own May 2026 accessibility roundup describes only user-started actions [M1]. AppleVis users confirm that Meta glasses won't describe surroundings unless prompted with "Hey Meta"; Live AI only drops the need to repeat the wake word [AV1].
2. **Google Guided Vision in Gemini Live launched on 2026-10-01, built with Aira.** It is the most important new competitor. It is free on Android 9+ and speaks on its own only to coach camera framing ("pan, tilt, step back"). Google says it is "not intended for navigation, safe-travel guidance, or obstacle detection" and is "not a white cane replacement" [G1][G2]. Google, with the most resources in the field, chose to stay out of unprompted environmental speech for pedestrians, apparently over liability.
3. **Project Astra is the real threat to the thesis novelty.** DeepMind's page says Astra "can intuitively start conversations", and its Aira "Visual Interpreter" is a trusted-tester research prototype under human oversight [G3][A2]. I found no published evaluation of when or how often it speaks (**unverified** whether any metric exists). The thesis should cite it as the nearest prior system and say what is new: a learned, budgeted speak policy trained from one-sided feedback and measured on real BLV video.
4. **Today's proactive products are narrow or need dedicated hardware.** OKO speaks about one thing, pedestrian signals [O1]. Apple Magnifier Detection Mode speaks continuously about people, doors and furniture, needs a LiDAR "Pro" device, and Apple warns not to use it for navigation [AP1][AP2]. Biped NOA, Glidance Glide, .lumen and WeWALK are obstacle hardware [B1][GL1][L1][W1]. OOrion is a Meta SDK app that keeps saying "the coast is clear or not" while walking, but the SDK is beta and quirky [AV1][OR1]. None of them learns per user what is worth saying, and none publishes a speech budget.
5. **BLV users mostly don't pay for software; institutions do.** The leading apps are free: Be My Eyes, Seeing AI, Lookout, Guided Vision and Envision's app [BME1][S1][LK1][G1][E2]. When OKO moved to a $4.99/month subscription in January 2025, AppleVis users objected, partly because some features were broken when billing started [O2]. Money comes from employers, state vocational rehabilitation (VR), the VA, governments and enterprise partners (Section 4).

## 2. Comparison table

P = speaks unprompted about the environment. P-narrow = unprompted, but only for one cue type or only obstacles. OD = on demand only.

| Product | What it does | P / OD | Platform | Price (user) | Licence | Main user complaints |
|---|---|---|---|---|---|---|
| **Be My Eyes + Be My AI** | Volunteer video calls (10M volunteers, ~1M BLV users, March 2026); GPT-based image Q&A; Service Directory; trusted-contact calls on Meta glasses (May 2026) | OD | iOS, Android, Windows, Ray-Ban/Oakley Meta | Free; paid by enterprises [BME1][BME2] | Proprietary | Early Be My AI falsely said faces were blurred; refuses ordinary photos under content rules; inconsistent results [AV2]. Shares multimodal video with Meta for AI training (opt-out offered) [BME3] |
| **Microsoft Seeing AI** | Channels: short text (reads as soon as text enters view), documents, products, currency, scene, handwriting; GPT descriptions | OD (auto-read of text in view is the only exception) | iOS, iPad, Android | Free [S1][S2] | Proprietary | Some users find its speech hard to listen to and prefer Be My AI [S3]; Android version less polished [S1] |
| **Google Lookout** | Modes: text, documents, food labels, currency, explore, images, image Q&A | Mostly OD; Explore mode announces objects in view (beta, less accurate per one source) | Android 6+ only | Free [LK1][LK2] | Proprietary | Image Q&A limited to US/UK/CA in 2024 [LK2]; current state **unverified** |
| **Google Guided Vision (Gemini Live)** | Live camera description, follow-up questions, spoken reframing prompts; built with Aira; 1,000+ testers | OD, plus prompts about camera framing | Android 9+; Google eyewear with an Android phone | Free (no price stated) [G1][G2] | Proprietary | New (2026-10-01); Google states it can make mistakes and must not be used for navigation [G2] |
| **Project Astra Visual Interpreter (Aira)** | Prototype assistant that "can intuitively start conversations" | P claimed, unmeasured | Trusted testers only | n/a | Proprietary | Aira FAQ: no memory across sessions [A2] |
| **Aira** | Professional human visual interpreters; AI calls watched by a human | OD | iOS, Android, Meta glasses | Minute-based plans; current 2026 prices **unverified**. Offsets on request; free through Amazon, Google and Salesforce as accommodation, and through VR in AL, CA, CT, CO, TX [A1] | Proprietary | Cost (inferred from the offset programme) |
| **Envision Ally app / Ally Solos glasses** | Conversational assistant: read text, describe surroundings, recognise faces, reminders | OD | iOS, Android; Solos glasses | App free [E2]; Solos $399 at launch with 1 year of Ally Pro, later $599-$699 (sources conflict) [E1][E3] | Proprietary | Price above Ray-Ban Meta and AirGo Vision [E3] |
| **Envision Glasses (Google Glass based)** | OCR, scene description, call an ally | OD | Glass hardware | $1,899-$3,499 by edition, plus a ~$199/yr update pass (2023-24 figures) [E4] | Proprietary | Price |
| **Meta AI on Ray-Ban/Oakley Meta (incl. Live AI, detailed responses)** | Wake-word Q&A; Live AI keeps the camera on for continuous Q&A; detailed descriptions setting; Be My Eyes and Aira on glasses | OD (Live AI removes the wake word but stays user-driven) | Glasses + Meta AI app | Gen 2 from $379 US [MR1] | Proprietary; Device Access Toolkit (DAT) dev preview, public publishing limited to partners (repo `docs/meta_glasses.md`) | Live AI battery 30-60 min; "grainy" camera; AI is "the big weak point"; features often US/Canada only [AV3][AV1] |
| **OOrion (Meta glasses app)** | Find objects, read text, obstacle alerts, continuous "coast clear" updates | P (obstacles, objects) | Meta glasses via DAT | **unverified** | Proprietary | Takes over the glasses (no "Hey Meta" while it runs); SDK is beta [AV1][M1] |
| **OKO (AYES)** | Pedestrian signal state by audio and haptics; turn-by-turn maps | P-narrow (signals) | iOS | Since Jan 2025, $4.99/mo or about $29.99/yr after 14-day trial (user reports) [O1][O2] | Proprietary | Subscription backlash; pulled from Canada; maps unavailable in some regions [O2][O1] |
| **Apple Magnifier Detection Mode / Live Recognition** | People (with distance), doors (with attributes), furniture, scene text; VoiceOver rotor | P-narrow (continuous while active) | iPhone/iPad Pro (LiDAR) | Free with device | Proprietary | Apple: do not use for navigation [AP1][AP2] |
| **Biped NOA** | Shoulder-worn vest: obstacle detection with 3D sound, GPS, scene description | P (obstacles) | Dedicated hardware | **unverified**; 75 devices shipped in launch year; 10 countries [B1][B2] | Proprietary | **unverified** |
| **Glidance Glide** | Two-wheeled robotic guide: avoids obstacles, finds doors and crosswalks, haptic/audio cues | P (obstacles), physical guidance | Dedicated hardware | **unverified**; first customers, waitlist [GL1]; $1.5M pre-seed [GL2] | Proprietary | **unverified** |
| **.lumen glasses** | Headset that "replicates a guide dog": pedestrian autonomous-driving AI, haptic guidance | P (path, obstacles) | Dedicated hardware | ~$9,999 per one directory (**unverified**); 1,500+ Romanian preorders free through a government programme; CE-certified, FDA status **unverified** [L1][L2][L3] | Proprietary | **unverified** |
| **WeWALK Smart Cane 2** | Ultrasonic obstacle cane + GPS + ChatGPT voice assistant | P (obstacles) + OD (assistant) | Cane + phone | $850 base; voice assistant $4.99/mo or $1,150 Plus [W1] | Proprietary | Price |
| **NaviLens** | Long-range colour codes on signage, read by camera without aiming | P-narrow (codes in view) | iOS, Android | Free for users; cities and transit pay (e.g. San Antonio VIA $8,950 for 100 stops) [N1] | Proprietary | Works only where codes are installed |
| **AGIGA EchoVision** | Glasses: tap for scene description, OCR, Be My Eyes/Aira calls | OD | Glasses | $599 ($449 preorder) in 2024-25 [EV1] | Proprietary | Refund reported; TestFlight-only app [EV1] |
| **Soundscape (ex-Microsoft, now community)** | 3D audio beacons and callouts of points of interest while walking | P (map-based callouts, not camera) | iOS | Free | **Open source** since 2023 [SS1] | The precedent: Microsoft ended it in 2023; community forks revived it, but running the backend was the open question [SS1] |

The Seeing AI and Lookout rows use sources from 2023-24. I found no dated 2026 changelog for either (**unverified** whether they gained proactive modes in 2026).

## 3. Market size signals

- **Global:** 43.3M blind and 295M with moderate or severe vision impairment in 2020; projected 61.0M and 474M by 2050 [LAN1].
- **US employment:** 52.3% employment rate and 8.1% unemployment for working-age adults with visual impairment (2023 ACS, via the Mississippi State NRTC) [NRTC1]. AFB cites an older, lower 35% [AFB1]. This matters because VR money is tied to employment goals.
- **Scale of the leader:** Be My Eyes reached about 1M BLV users and 10M volunteers in March 2026. It reported 361% revenue growth in 2024 and raised $6.1M (Series A+) in January 2025, after a $2.8M Series A in 2020 [BME1][BME2]. Its foundation aims at 50M users by 2035 [BME1].
- **Hardware demand:** .lumen reported 1,500+ preorders in 10 days, funded by the Romanian government [L2]. Biped shipped 75 NOA units in its launch year [B2]. Glidance raised a $1.5M pre-seed [GL2]. Hardware volumes are small, and public payers decide them.
- **Big platforms entered in 2025-26:** Meta (glasses + Be My Eyes + Aira), Google (Guided Vision, built with Aira), Apple (Live Recognition). Free, built-in, on-demand description is now a commodity. A standalone "describe my scene" app has no market left.

## 4. Who pays

| Payer | Evidence | Fit for SeeForMe |
|---|---|---|
| BLV users directly | Tolerated for hardware ($299-$10k). Small subscriptions draw complaints: OKO at $4.99/mo [O2]; WeWALK charges $4.99/mo for its assistant [W1] | Poor. Don't charge users for the core |
| Employers (workplace accommodation) | Aira through Amazon, Google, Salesforce [A1]; Be My Eyes enterprise revenue [BME2] | Weak for a pedestrian tool; maybe campus or employer commute programmes |
| State vocational rehabilitation, VA | Aira through VR in 5 states [A1]; VR funds AI smart glasses tied to an employment plan [VR1][VR2]; Medicare treats smart glasses as non-covered [VR1] | Good for hardware bundles and O&M training, needs a vendor of record |
| Governments, cities, transit | NaviLens city deployments [N1]; Romania funded .lumen [L2] | Plausible later (city pilots, data collection) |
| Platforms and enterprises (B2B) | Be My Eyes is funded by enterprise partners and supplies data to Meta [BME2][BME3] | The realistic path: license or integrate the speak/silence policy into others' apps and glasses |
| Research funders and universities | Not researched in this pass (**unverified** which grants fit, e.g. NSF or NEI SBIR) | Best fit during and right after the thesis |

## 5. Open source or paid?

**Recommendation: open source the research artifact. If a product follows, fund it through B2B and institutions. Never charge BLV users for the core.**

- **Free is the market norm.** The four strongest on-demand tools are free [BME1][S1][LK1][G1]. A paid SeeForMe would compete against Google's free Guided Vision on the same phone.
- **Openness builds trust here.** Users already worry about wrong answers [AV2], data flows to Meta [BME3] and startup survival (EchoVision refunds) [EV1]. An auditable, open policy for *when* to speak is a credible stance, especially since every big vendor disclaims safety use [G2][AP1].
- **Lesson from Soundscape:** Microsoft's code survived as open source, but the communities had to fund servers [SS1]. Keep SeeForMe on-device (phone detector + scheduler) so no backend cost can kill it.
- **Licence caution:** AGPL on the research release is fine. For a product track meant to ship in the App Store or be embedded by partners, a permissive licence (Apache-2.0 or MIT) for the scheduler library removes friction. Copyleft-and-App-Store compatibility is a known dispute area; check it with legal before choosing (**unverified**, no source checked in this pass).
- **Sustainable model, if pursued:**
  - Free app for users.
  - Permissively licensed "speech-budget scheduler" SDK, to be embedded by Meta DAT apps (OOrion-style), Be My Eyes and Aira on glasses, and hardware makers (Biped, Glidance, .lumen, WeWALK) whose proactive alerts are rule-based today.
  - Paid integration and support contracts with those companies.
  - VR- and O&M-funded training deployments.
  - Grants for the research.
- **Ship constraint:** Meta's DAT public publishing is limited to partners (repo `docs/meta_glasses.md`), so the glasses path needs a partnership. The PWA and native iOS path does not.

## 6. The one defensible differentiator

**A learned, per-user, budget-rationed policy that decides when to speak about the surroundings unprompted. It is trained only from the one-sided feedback a real user gives (taps after cues, questions after missed events) and evaluated on real first-person BLV video with published airtime and quiet-window metrics.**

Why this survives the 2026 landscape:
- The platforms stop at on-demand description and hedge away from mobility and safety [G2][AP1][AV1].
- Proactive incumbents are either single-cue (OKO, NaviLens) or rule-based obstacle hardware (Biped, Glide, .lumen, WeWALK). None claims a speech budget or per-user learning. Their public pages don't describe how they decide what to say (**unverified** internally).
- Astra claims proactivity but publishes no policy or metric [G3]. SeeForMe can be the first to *measure* the problem: 160 coverable events, 0.525 full-information vs 0.204 one-sided recall, ~2 cues/min.

What it must not claim:
- **Not a navigation or obstacle-safety device.** Everyone else disclaims this too, and current recall (0.20) does not support it.
- **Not "the first proactive assistant".** Apple Detection Mode, OKO, OOrion and the obstacle devices already speak unprompted. The defensible word is *learned and budgeted* proactivity about what users actually ask about, beyond collision warnings.

Risk to watch: Google or Meta adds a "notify me" mode to Guided Vision or Live AI. Mitigations: publish first, keep the method open, and position SeeForMe as the policy layer others can adopt.

## Gaps and items I could not verify

- Current 2026 prices for Aira, Biped NOA, Glide, .lumen (only the $9,999 directory figure), OOrion and Ally Solos.
- Whether Seeing AI or Lookout added proactive modes in 2026.
- Any Reddit r/Blind threads: searches returned only AppleVis.
- The current state of Lookout Explore mode.
- Grant programmes.
- AGPL App Store compatibility.

## Sources

- [M1] https://about.fb.com/news/2026/05/meta-ai-wearables-changing-the-game-for-disabled-people/
- [AV1] https://applevis.com/forum/assistive-technology/meta-glasses-will-they-actually-give-me-real-independence-blind-person
- [AV2] https://applevis.com/comment/152935 (Be My AI thread, 2023)
- [AV3] https://atlas-entry.applevis.com/comment/180441 ; https://atlas-entry.applevis.com/comment/206112
- [G1] https://www.androidauthority.com/gemini-live-guided-vision-launch-3718122/
- [G2] https://mixed-news.com/en/gemini-live-guided-vision-android-launch-aira/
- [G3] https://deepmind.google/models/project-astra/
- [A1] https://aira.io/new-plans
- [A2] https://aira.io/projectastra ; https://nfb.org/images/nfb/publications/bm/bm25/bm2507/bm250712.htm
- [BME1] https://www.bemyeyes.com/news/be-my-eyes-reaches-1-million-blind-and-low-vision-users-and-10-million-volunteers/ ; https://apps.apple.com/app/905177575
- [BME2] https://www.bemyeyes.com/business/?p=2211 ; https://coverager.com/be-my-eyes-raises-2-8-million/
- [BME3] https://www.bemyeyes.com/?p=21193995
- [S1] https://blogs.microsoft.com/accessibility/seeing-ai-app-launches-on-android-including-new-and-updated-features-and-new-languages/ ; https://www.windowscentral.com/software-apps/microsoft-brings-its-free-seeing-ai-app-to-3-billion-people
- [S2] https://www.digitaltrends.com/phones/microsoft-releases-seeing-ai/
- [S3] https://vi.ie/?p=875237
- [LK1] https://www.neilsquire.ca/lookout-vision-app/
- [LK2] https://support.google.com/accessibility/android/answer/9031274 ; https://apkmirror.com/apk/google-inc/lookout/lookout-5-0_reveal_20240508-02_rc03-release
- [E1] https://www.letsenvision.com/blog/ally-solos-glasses
- [E2] https://www.letsenvision.com/blog/creating-a-more-accessible-envision ; https://apps.apple.com/ca/app/envision-ally/id1525230292
- [E3] https://shop.letsenvision.com/products/ally-solos-glasses ; https://getcoai.com/news/envision-and-solos-launch-399-ai-glasses-for-blind-users/
- [E4] https://letsenvision.com/blog/envision-glasses-editions-launch
- [MR1] https://telecomlead.com/smart-phone/ray-ban-meta-gen-2-key-new-features-every-consumer-should-know-122644 ; https://www.meta.com/blog/ray-ban-meta-gen-2-now-available-ai-glasses-extended-battery-life-3k-video/
- [OR1] https://doubletaponair.com/smart-glasses-get-smarter-steven-buys-a-thing-2/
- [O1] https://apps.apple.com/us/app/ayes-oko/id1583614988
- [O2] https://www.applevis.com/apps/ios/navigation/oko-ai-copilot-blind ; https://doubletaponair.com/oko-app-introduces-subscription-model-steven-wants-to-delete-facebook/
- [AP1] https://support.apple.com/en-mn/guide/iphone/iph32deb9296/ios
- [AP2] https://mcmw.abilitynet.org.uk/how-to-detect-people-doors-and-furniture-in-your-surroundings-using-the-magnifier-app-in-ios-26-on-your-iphone-pro-or-ipad-pro
- [B1] https://applevis.com/comment/171390
- [B2] https://ggba.swiss/en/?p=25887 ; https://bipedai.beehiiv.com/p/noa-update-is-here-faster-ai-follow
- [GL1] https://glidance.io/
- [GL2] https://www.therobotreport.com/glidance-raises-15m-develop-robots-for-visually-impaired/
- [L1] https://www.dotlumen.com/ ; https://www.ces.tech/ces-innovation-awards/2026/glasses-for-the-blind/
- [L2] https://www.medica-tradefair.com/en/Media_News/Spheres_of_MEDICA_-_Magazine/Over_1,500_preorders_for.lumen_glasses_in_just_10_days
- [L3] https://www.dotlumen.com/post/ce-certification-glasses-for-the-blind ; https://theresanaiforthat.com/device/lumen-glasses/
- [W1] https://www.androidheadlines.com/2025/01/wewalk-debuts-its-new-ai-powered-smart-cane-2.html ; https://shop.rnib.org.uk/products/wewalk-smart-cane-2-ai
- [N1] https://navilens.com/en/ ; https://sanantonioreport.org/via-technology-visually-impaired/
- [EV1] https://doubletaponair.com/echovision-glasses-from-agiga/ ; https://www.applevis.com/comment/185357
- [SS1] https://applevis.com/comment/142609 ; https://doubletaponair.com/soundscape-community-brings-popular-navigation-app-back/
- [LAN1] https://pubmed.ncbi.nlm.nih.gov/33275950/
- [NRTC1] https://www.blind.msstate.edu/sites/www.blind.msstate.edu/files/2024-12/Employment%20Statistics%20for%20Adults%20with%20Visual%20Impairments.pdf
- [AFB1] https://afb.org/research-and-initiatives/statistics/key-employment-statistics
- [VR1] https://nelowvision.com/wp-content/uploads/2026/06/Vocational-Rehabilitation-Funding-for-Assistive-Technology.pdf
- [VR2] https://orcam.com/en-us/blog/vocational-rehabilitation-can-help
