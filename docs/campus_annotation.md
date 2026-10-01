# CampusBench annotation format

Every CampusBench scenario (`data/campus/<split>/*.json`) carries two layers:

1. **World** — objects in metres relative to a walking user (`keyframes` of `(t, x, z)`, size,
   mounting height, detail such as handrail or handle side, goal tags) plus the user's own
   motion (`user_keyframes`), the camera (field of view, height, pitch, sway) and capture
   degradations (low light, motion blur, clutter false positives, depth noise, blackout windows).
2. **Events** — one record per object, produced by `CampusScenario.events()` from the world
   geometry and the event taxonomy in `seeforme/campus/world.py`:

```json
{
  "obj_id": 4, "label": "stairs", "kind": "object", "event_type": "stairs", "text": null,
  "should_speak": true, "reason": "",
  "severity": 1.0, "urgency": "high", "interrupt_current_speech": true,
  "earliest_useful_s": 9.4, "ideal_s": 12.3, "latest_safe_s": 13.4, "t_contact": 15.9,
  "path_intersection": 1.0, "distance_at_ideal_m": 4.6, "direction_at_ideal_deg": 0.0,
  "moving": false, "sudden": false,
  "minimum_information": ["object", "distance", "action"],
  "ideal_cue": "Stairs down, about 4.5 meters. Handrail right.",
  "detail": {"direction": "down", "handrail": "right"}
}
```

and for something that must *not* be spoken:

```json
{"obj_id": 2, "label": "poster", "event_type": "decor", "should_speak": false, "reason": "not_action_relevant"}
{"obj_id": 7, "label": "person", "event_type": "person_static", "should_speak": false, "reason": "off_path"}
```

* `earliest_useful_s` — first moment the information helps (the object is within the notice range
  of its type, or a moving object's time-to-contact is within the notice horizon, or a sign is legible).
* `ideal_s` — the moment a good sighted guide would speak (type-specific range / TTC).
* `latest_safe_s` — `t_contact` minus the reaction time of the event type. A cue after this is *late*.
* `sudden` — the object became visible after its latest safe time (a bike shooting out of a side
  street); such events are reported separately because no policy can be on time.
* `minimum_information` — the semantic slots a minimal message needs; the composer in
  `seeforme/cues/compose.py` renders them into `ideal_cue`, so the "minimum message ratio" compares
  the slots a policy spoke with the slots the event needed.
* `goal` events — when the scenario has a user goal (`"find the elevator"`), objects whose
  `goal_tags` or sign text match become `goal` events even if they are off the walking path.

## Annotating real recordings

`scripts/annotate_video.py` writes the same structure for a video: run detection + depth to get a
draft, then edit `events` by hand (timestamps, `should_speak`, `minimum_information`, `ideal_cue`).
`scripts/score_video.py --campus` scores a cue log against it with the CampusBench metrics.
