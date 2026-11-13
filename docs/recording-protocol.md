# Own recordings: protocol (issue #9)

Goal: continuous, narrated, chest-mounted walks around campus that give timed, in-situ labels, long-horizon memory tests and the licence-clean product dataset. Ingest with `scripts/ingest_own.py`.

## How much

| | Pilot | Full set |
|---|---|---|
| Videos | 6 | 30-40 |
| Length | 8-10 min each, one continuous take | 8-10 min |
| Total | about 1 h | about 5 h |
| Storage (1080p30 HEVC, about 60 MB/min) | about 4 GB | about 18-20 GB |
| Lockbox hour (see below) | none | about 1 h on routes never used for development |

Five hours should give at least 300 should-speak moments (recall to about +-0.05) and enough boxes to fine-tune a detector (1,000-2,000 per class is a rule of thumb, unverified for this footage).

## Two batches that never mix (added 2026-11-13)

| | Development batch | Lockbox batch |
|---|---|---|
| Purpose | tuning thresholds, cue length and features; label and error analysis | one final evaluation of frozen models |
| Order | recorded first; the pilot walks belong here | recorded only after a development batch exists |
| Routes | any | other buildings and streets; no route segment shared with the development batch |
| Folder | `data/own/raw/dev/` | `data/own/raw/lockbox/` |
| Looking at it | freely | not at all before the plan below is frozen; ingest runs unattended and prints counts only |

Evaluation plan, written and committed before the first lockbox result is computed:

1. **Frozen artefacts.** Detector weights, scorer checkpoint, threshold or budget, cue templates: each with its commit and file hash.
2. **Primary measure.** Narrated should-speak moments announced on time, at a stated number of spoken seconds per minute. Secondary: speech during narrated "clear" moments, onset timing, seconds of speech.
3. **Comparison.** The production threshold rule against whatever was promoted on development data, with intervals from a bootstrap over walks (a walk is the independent unit, not a moment).
4. **One run.** The result is reported whatever it is. Nothing is re-tuned afterwards; if anything changes, the lockbox has become development data and a new one is needed.
5. **Labels.** The narration is the recorder's own judgement. A second person labels at least 20% of the lockbox moments blind with the review tool (rubric v4) before scoring, and agreement is reported.

`select`, which was used twice during development, is treated as development data from now on. EgoBlind `holdout` and the official test stay sealed.

## Content

Each at least 3 times, different places and light: stairs up and down (indoor, outdoor, with and without railings); doors (push, pull, automatic, heavy gates, revolving); curb ramps, signalised and unsignalised crossings; hallways, entering an empty classroom and finding a seat, elevators, dining-hall line, library aisles; bikes and scooters, construction, poles, benches, crowds; day, dusk, night, rain. Example routes: AKW to SSS, Cross Campus to Sterling, Old Campus to Chapel St, Commons to Bass. Keep development and lockbox routes separate.

## How

- iPhone 13 Pro: Settings > Camera > Record Video > 1080p at 30 fps; Formats > High Efficiency; 1x lens (not 0.5x, Cinematic or Slo-mo).
- Chest harness, phone at the sternum, landscape, tilted down about 15 degrees (the angle the camera model assumes). Not handheld.
- Narrate as you walk: say a keyword at the moment you would want a warning: "stairs down", "door", "curb", "crossing", "bike left", "pole", "elevator", "construction". Say "clear" where nothing is needed. The ingest turns these into timed labels (full vocabulary in `seeforme/own/ingest.py`).
- Walk naturally; stop where a blind pedestrian would stop (top of stairs, curb edge).

## Privacy and rules

- Raw video goes only to `data/own/raw/dev/` or `data/own/raw/lockbox/` (git-ignored) via AirDrop or cable; name files `YYYY-MM-DD_route_takeN.MOV` and keep a log of route, time, light and notes.
- The ingest blurs the head region of every detected person before writing `data/own/processed/`; raw video and audio never leave the Mac.
- Record empty classrooms or ask first. Bystanders are filmed in public without consent forms, so nothing unblurred is ever shared and raw files stay on this Mac.
- Ask the advisor whether recordings of yourself need IRB review, and get that answer in writing before any recording involves another person as a participant (a second walker, a blind tester).

## Ingest

```
python scripts/ingest_own.py                    # every new file in data/own/raw/dev/
python scripts/ingest_own.py --batch lockbox    # the sealed walks; prints counts only
```

Output per video: a blurred 640-px MP4 (no audio) and `<name>.labels.json` with Vosk word timestamps and the narration events.
