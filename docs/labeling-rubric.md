# Labeling rubric v4: should the assistant speak, and is its sentence true?

Used by every rater of a rubric v4 queue (`scripts/review_events.py`), human or model. A rater sees the frames up to one decision moment and, for most moments, the sentence the assistant would say. Nothing else: no question text, no earlier labels, no category.

## The setting

A blind person is walking or doing an everyday task with a phone camera on the chest or in the hand. The assistant can say a short cue without being asked. Speech is costly: the person is listening to traffic, footsteps and a cane, and the budget is about two cues a minute. At the decision moment (the last frame, orange border) nobody has asked anything.

## Question 1: should the assistant speak up on its own, now?

Decide this from the scene before reading the sentence.

| Answer | Key | Choose it when |
|---|---|---|
| Yes: announce something (`proactive_warn`) | 1 | Something affects the person's safety or path in the next few seconds and they would not know to ask: an obstacle in the walking line at close range, a vehicle, bike or person on a crossing course, steps, a kerb or a drop, a door or a narrowing, something at head height. Waiting to be asked would be too late. |
| Yes: say the way is clear (`proactive_clear`) | 2 | The person is visibly waiting or hesitating (stopped at a crossing or doorway) and the view shows they can go. Rare. |
| Only if the user asks (`on_request`) | 3 | There is something worth describing, but nothing urgent: items on a table or shelf, a sign, people standing nearby, an object off the path, far away or moving away. Saying it unprompted would be chatter. |
| No: stay silent (`silent`) | 4 | Nothing worth saying even if asked "anything I should know?": an open path, the same unchanged scene, floor, sky, a blur with no content. |
| Can't tell from the frames (`uncertain`) | 5 | Too dark, blurred or cropped to judge, or it is unclear whether the person is moving towards the thing. |

Rules:

1. **The test for answer 1:** would a sighted companion walking beside this person say something right now, without being asked? If they would wait, it is not answer 1.
2. **"Speak" is not the default.** Every cue costs attention. An object being visible is not a reason to announce it.
3. **Use the earlier frames** to see whether the person is moving and whether the object is getting closer. A parked scooter three steps ahead on the walking line is answer 1; the same scooter across the road is answer 3 or 4.
4. **Ignore burned-in subtitles and captions.** Some videos show what was said as text on the frame. Do not treat that as a request and do not use it to guess what the person wants. Judge the visual scene only. If subtitles are visible, start the note with `[sub]`.
5. **The sentence must not change answer 1 to 5.** A warning can be needed about something other than what the sentence names. Then answer 1 and mark the sentence false.
6. **Uncertain is a real answer.** It is left out of training and is never counted as "stay silent". Do not guess to avoid it.

## Question 2: is the sentence true of what the frames show?

Only when a sentence is shown. It names an object or a place, often with a direction ("Motorcycle, slightly right."). Directions are relative to the middle of the decision frame.

| Answer | Key | Choose it when |
|---|---|---|
| True (`correct`) | Q | The named thing is visible at the decision moment and the direction is roughly right. A near-synonym counts (an electric scooter called a motorcycle). |
| Partly true (`partially_correct`) | W | The thing is there but the direction is wrong, or the name is a loose fit. |
| False (`wrong`) | E | No such thing is visible, or it is clearly something else. |
| Can't check from these frames (`not_assessable`) | R | The frames are too poor to say either way. |

Optional: whether you can see the thing the sentence refers to (yes, no, not sure) and how sure you are overall (sure, fairly sure, guessing). A short note helps when something is wrong or missing.

## Raters

A person reviews under their own initials. A model's first pass is stored under its own id with `verification: "model"`; it is never reported as a human label, and agreement between a model and a person is reported as that, not as agreement between two people. The second rater must not see the first rater's answers before finishing.
