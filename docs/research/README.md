# Research reports, 2026-11-12

These reports were written by separate research agents (literature gap, market, algorithms, UI/UX, data), then checked by an independent audit pass that opened the cited sources and recomputed every project number from `results/`. Read the corrections below first: where a report disagrees with them, the corrections win.

| File | What it covers |
|---|---|
| `scientific-gap.md` | Prior work on proactive assistance for blind users, what is novel, risks |
| `market.md` | Products in 2026 (Be My Eyes, Seeing AI, Lookout, Aira, Meta, OKO, Glidance, ...), pricing, open source vs paid |
| `algorithms.md` | Formal problem, related theory, diagnosis of the learning results, ranked fixes |
| `ui-ux.md` | What blind pedestrians want, HIG and VoiceOver requirements, audit of `web/` |
| `data-plan.md` | What is done, hard and missing; which data to get, how, where, and under what licence |
| `coder-report.md` | Implementation and experiments that followed (causal reference, p_ask, issue #6) |
| `audit-1.json`, `audit-final.md` | The two audit passes, claim by claim |

## Corrections that override the reports

1. **The 0.525 "full-information upper bound" is not a fair reference.** It used labels at zero delay inside the same clip. With labels held to the end of each clip, full information reaches 0.200 against 0.204 for one-sided greedy (difference -0.004, 95% CI [-0.054, +0.048]). So there is no measured "60% cost of one-sided feedback"; within-clip label latency is what matters (2 s delay: 0.379). Any "0.53 vs 0.20 gap" sentence in `scientific-gap.md`, `market.md` or `data-plan.md` is superseded.
2. **Seeing AI and Lookout do speak unprompted** (Short Text channel, Explore mode), by fixed rules. The defensible claim is that no deployed app learns when to speak.
3. **IL repeats:** 13.6% of IL cues repeat the same track; the 28.5% figure counts different objects with the same label.
4. **`select` has 11 coverable events per seed** (33 pooled over three seeds).
5. **OKO** removed its subscription on 2025-07-17 per its App Store page; see `audit-1.json` for the other market fixes.
6. **Web findings in `ui-ux.md`** (unescaped OCR text, dark-mode background, scheduler provenance, iOS haptics) were fixed on `live-demo-20261009` (web 0.4.1) by the web session.
7. Items marked unverified in `audit-final.md` (decision-level AUC 0.683, hindsight relabelling 0.156 / 0.175) must not be quoted.
