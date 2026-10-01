# Annotation format for real video scenarios

Real recordings are scored with `scripts/run_video.py` (produces a cue log) and
`scripts/score_video.py` (compares it with a hand-made annotation file).

```json
{
  "video": "hallway_walk_01.mp4",
  "duration": 42.0,
  "events": [
    {"label": "chair", "t_start": 3.0, "t_end": 8.0, "note": "chair in walking path"},
    {"label": "text", "t_start": 12.5, "t_end": 18.0, "note": "EXIT sign becomes legible"},
    {"label": "person", "t_start": 20.0, "t_end": 23.0, "note": "person crosses directly ahead"}
  ]
}
```

* `t_start` is the moment a reasonable sighted guide would first mention the event.
* `t_end` is when the information stops being useful (object passed, sign out of view).
* Objects that are visible but not worth mentioning (posters, far-away people,
  clutter) are simply not listed; any cue about them counts against cue precision.
* A cue counts as a hit when its label matches and its timestamp is inside the window.

Synthetic scenarios (`data/scenarios/*/*.json`) use the richer `Scenario` schema
from `seeforme/sim/scenario.py`, which additionally stores the trajectory of every
object so a noisy detector can be simulated.
