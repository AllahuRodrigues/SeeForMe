"""AnnouncementGate: a "what did I already tell the user?" memory in front of any scheduler (live demo only).

The schedulers decide per track id and forget quickly (P-CACS blocks a track for `recent_block_s` = 2.8 s and
its novelty recovers with tau = 10 s). On a desk webcam that turns the user's own body into "Person." /
"Person, about 1.5 meters." every few seconds, and every tracker id switch looks like a brand-new person.
The gate remembers announcements by *object identity* instead of track id:

    identity   label (with a few aliases: truck/bus -> car, ...) for objects; normalised text for signs, where two
               OCR readings are one sign only if their numbers are identical and the rest nearly so ("ROOM 204" /
               "R0OM 2O4" are one sign; "room 204" / "room 206", "gate 12" / "gate 13", "exit" / "exit 2" are not)
    sector     left (< -sector_deg) | ahead | right (> +sector_deg) of the box centre, with hysteresis
    memory     last announcement time, distance, angle, track id, box and cue of that identity in that sector

Safety principle: the gate only removes cues that carry no new safety information. Silence about the desk user
themself and about static repeats is the goal; silence about something approaching the user, in the walking path,
a vehicle closing in, or a FIND goal coming into view is a bug.

Identity across track ids. A track in view for the first time takes over from a track of the same identity that
vanished within switch_gap_s, in the same sector (or within 2 x sector_hyst_deg of its angle) as where it was last
seen, not farther than it was then by half of max(closer_m, closer_frac x range), not nearer than the object could
have come while out of view (its top speed by identity, a chair none, plus the user's walk; at least that step), and
whose last match came before the new track's first one (both matched in one frame: two objects, or one box twice):
its cue held for confirmation, its held cue, its approach release and its tracker matches (its memory entries are
adopted when the new track is next judged, so the track the scheduler talks about wins), unless the object had moved
away since it was said by max(closer_m, closer_frac x range) (one that walked off and vanished, and another coming
from there, are two objects: the web gate's successorOf). So a lighting dropout or an id switch keeps what the gate
owes the user about the object, and an object seen in two short sightings at the same place is confirmed. Memory
passes *only* along that chain of successors (or to a duplicate box of the same object, overlapping it at the same
range, or back to a track id the tracker re-acquires): a car that drives off to the side and another car coming from
6 m where it stood is a new car. The IoU tracker itself may keep one track id for two objects (another person steps
in where the first stood within its max_age, or in the very frame the first leaves): a track whose range jumps farther
by half the closer step, across a dropout or from one frame to the next by more than the object could move away in
that time, lets go of the memory said about it before the jump (and of its background record), as if its object had
left view.

A cue passes when one of these is true, checked in this order:

    critical      measured TTC < critical_ttc_s and path_prob >= critical_path (passes even unconfirmed). For a track
                  the gate still holds as the operator, background or warm-up object it also needs a measured closing
                  speed of at least approach_min_mps (the user leaning toward the laptop at 0.8 m is "TTC 0", not a
                  hazard), and the operator's own track nearer than its remembered leans (below). At most one critical cue
                  per object every critical_gap_s, unless its TTC dropped by critical_worse_ttc_s or its distance by
                  critical_worse_m since the last critical cue (worsening)
    approach      a background object or the operator that was released because it measurably closes in (below): its
                  first cue after the release passes (within approach_keep_s), whatever was said about it before
    new           no memory of this identity in this sector (a second object of the same label that was in view
                  together with the remembered one, as a separate box at the same decision, counts as new: an id
                  switch never has both tracks visible at once; a duplicate box of the same object overlaps it by
                  dup_iou at the same range)
    reacquired    back in view after more than lost_s out of view (a successor track, within its first release_span_s),
                  in the walking path or corridor and max(reacquired_m, reacquired_frac x range) nearer than when it was
                  last seen: it came closer unseen (a dropout as it starts to approach). Once per return, even within
                  min_gap_s (a static object back where it was says nothing new)
    sector        the same track was last announced in another sector (at most every min_gap_s)
    closer        distance dropped by max(closer_m, closer_frac x distance at the last announcement) (min_gap_s)
    near          no measured depth (depth off, or depth_conf < 0.4), so nothing is ever critical: an object in the
                  walking path or corridor that is near (range < nodepth_near_m, or its box bottom in the lowest
                  nodepth_bottom_frac of the frame) and closing at >= nodepth_approach_mps, by the range rate over >= 1 s
                  or by the box growing over 0.5-1 s (only the sides the frame edge does not cut count: a floor obstacle
                  a step away is cut off at the bottom, so its box-height range *grows* while the user walks into it),
                  at most every min_gap_s. A static object (the user at a desk) does not loom, so it is not repeated
    closing       an object in the walking path or corridor that is max(release_m, release_frac x range) nearer than
                  when it was last said (smoothed ranges: measured depth, else the box), at most every min_gap_s: finer
                  than `closer` (a person or a car that starts coming at a user who stands still is said again before
                  its TTC turns critical, and so is a chair the user walks at slowly, or a FIND goal sign the user walks
                  to: harness find:2913). The scheduler offering it is enough, at any range; the gate says it unasked
                  (rescue) only nearer than path_m and while it measurably closes too. Without measured depth (nothing is critical) a near one (range under
                  nodepth_near_m, or its box bottom in the lowest nodepth_bottom_frac) the scheduler offers passes within
                  min_gap_s too: a walk covers it faster than that; and so does one that closes at release_fast_mps or
                  more now (it was said a moment ago, standing, and walks at the user)
    reminder      hold time elapsed: hold_goal_s for a FIND goal, hold_moving_s if the object moves or is in the walking
                  path or corridor between operator_max_m and path_m and release_m nearer than when it was said (the user
                  walking at it: without depth, or in a flicker, no closing speed may show), else hold_static_s
    interrupt     the scheduler wants to cut the current message for it, the object moves or approaches (measured),
                  and it was said at least min_gap_s ago
and it is held back when:

    unconfirmed   the track has fewer than min_hits matches (or, frame-rate independent, 2 matches spanning
                  confirm_age_s), counting those of the tracks it took over from, and it is neither in the walking path
                  or corridor nor the FIND goal (those cannot wait a step: in lighting dropouts the follow-up
                  sightings may all fall between two decisions; during the warm-up with operator_filter, a person who
                  may be the desk user still waits); the cue is deferred, not dropped: it
                  is said (distance and direction refreshed) once the track is confirmed, within pending_s, or, if the
                  track flickered out, once it or a successor is back and confirmed, within flicker_keep_s. A flicker
                  that never comes back is never spoken; a real object waits one decision step
    warmup        during the warm-up (the first warmup_s after the camera first sees something): a track the gate has
                  seen static for warm_min_samples box samples spanning warm_hold_span_s (never on its first samples;
                  in the walking path or corridor bg_min_span_s) and that is not a vehicle, not closing in and not in
                  the walking path nearby (by the depth this warm-up measured, else by the box at its nearest:
                  BOX_SMALL), nor an object in the path that came into view after the warm-up began (an arrival is not
                  part of the scene the warm-up learns), nor one a rebaseline found owed a say-once cue; or that looks
                  like the operator, or is the user the gate saw before (_known_user), and is not closing in (a
                  stranger walking up to the webcam during a rebaseline's warm-up)
    background    static through the warm-up (warm_min_samples box samples spanning bg_min_span_s inside it, no
                  measurable closing), until it is released. Vehicles (car, bus, truck, bicycle, motorcycle, ...) and
                  tracks closing in never become background. An object in the walking path (path_prob >= critical_path,
                  or its box inside the walking corridor) nearer than bg_path_m is said once before it is silenced (by
                  the depth the warm-up measured, trusted over the box; without any, by the box range at its nearest:
                  an object half the label's typical size reads twice as far, BOX_SMALL), and so is a person or animal
                  the gate saw arrive (its box moved more than bg_release_disp since it came into view, before a
                  rebaseline's warm-up: someone who walked in and stopped is news, not furniture). A rebaseline keeps
                  what the gate owes: an in-path object the old warm-up found must be said once, not yet said, is not
                  held by the new warm-up and is said once before it is background again. A
                  new track overlapping the place of a background object is that object only at about its range (not
                  max(closer_m, closer_frac x range) farther: another person stepping in from farther away is not).
                  One in the walking path between operator_max_m and path_m that was not said for hold_moving_s (or
                  never) and is release_m nearer than where it was said (or rests) is judged like any object: its
                  approach release needs max(release_m, release_frac x range), which an approach that starts as it
                  drops out of view, or in a flicker, may never show while in view
    operator      operator_filter (desk webcam): the user, i.e. the largest centred person present and still through
                  the warm-up and at most operator_max_m away (measured depth; without depth the nearer of the box-height
                  and box-width estimates, since a desk user's box is cut off at the bottom), until released. A person
                  standing 2 m in front of the camera is an ordinary object (background if static), and so is a new
                  track (or a second box over the user) farther than the user's place by max(release_m, release_frac x
                  range): a stranger who walks up after the user left, or a visitor in front of the user. A second,
                  smaller person box attached to the user's while the user's track is in view (its bottom cut by the
                  frame where the user's is, from the first sighting of its object, overlapping the user's box, not
                  closing in, not farther than the user by measured depth) is a part of the user detected as a person
                  (an arm, a hand): held, with no release of its own. A visitor walking up has their feet in view (a
                  box bottom above the frame's) until about 1.6 m away, so never was attached. A warm-up that missed
                  the user (its box flickered out as it ended) finds them later: a person in view (by its chain of
                  tracks) since before the warm-up ended, large, centred and within operator_max_m from its first
                  sighting, whose box never moved, at rest now (_late_operator, _known_user)
    critical_gap  a critical cue for the same object less than critical_gap_s ago and the hazard did not worsen
    haptic        a haptic for the same identity and sector less than haptic_gap_s ago
    muted         objects_muted (read mode, set by the TaskAdapter): an object cue the adapter's read-mode filter would
                  drop (not critical, severity under muted_severity, TTC not under approach_ttc_s, not an interrupt), and
                  that does not pass as an approach (critical, approach, closing, near, reacquired), nor is about a
                  vehicle, the FIND goal or an object whose measured range closes at release_mps: read mode mutes the
                  static scene around a user who stands and reads, never an object coming at them. Held *before* it is remembered as said, so the warning the scheduler offers once
                  the object approaches is not a repeat of a cue nobody heard
    repeat        none of the pass rules hold
A FIND goal (label or sign text, multi-word goals match as a phrase) is never held as warm-up or background, nor as
the operator unless it is the user themself (FIND "person" at a desk must not announce the user every 8 s).

The scheduler's view. The gate must not change what the scheduler decides: P-CACS and the learned scheduler decide on
the channel, their speech budget and the 'announced' marks on the tracks (novelty, recent block, the learned model's
features), so a scheduler that sees another state than it would alone stops giving the warnings it gives alone (a
rescued bench stamped as announced kept P-CACS from warning about it 1.8 s before contact; a routine sign cue said 1 s
late, taken for the scheduler's own message, kept the learned scheduler from ever naming a person walking at the user).
So the gate never touches the scheduler's budget or the tracks' marks (a held cue was spent, as alone; the gate's own
messages are not the scheduler's), and builds the channel it shows the scheduler from the scheduler's own offers only:
each speech cue it offers (a "speak" on the channel it was shown free, or an interrupt) holds that channel from when it
was offered for its spoken length, whether the gate passed it, held it, or said it later; nothing else does (the gate's
own messages: rescues, lost, deferred, FIND swaps; a cue said late; another cue of the scheduler's list; a reply or a
cane cue). Only while the real channel plays exactly the offer, let through when it was offered, is the scheduler shown
the real channel (the same message; the live speaker may run longer than the estimate). What the scheduler is shown and
offers under the gate is then what it is shown and offers alone (scripts/live_safety.py --diverge checks it). A safety
cue it offers while the real channel plays another message (critical, approach, closing, near, reacquired, or about a
vehicle, the FIND goal, an object in the path or corridor, a measured TTC under approach_ttc_s) cuts that message; a
routine one waits for the channel (within DISPLACED_KEEP_S), because the live speaker drops a message it gets while it
speaks.

Release. A background object or the operator becomes an ordinary object (so the critical rule applies) as soon as it
measurably closes in: its distance is below its reference by max(release_m, release_frac x reference) (measured
depth: the median over the last release_smooth_s; without depth: the box's unclipped width, else height, else area
over the last 3 frames against its reference size; the operator by measured depth: plus max(lean_noise_m, twice the
spread of its own depth at rest, estimated at the warm-up and at every re-capture), so a shallow lean on a noisy depth
map is no approach), or its latest sample is that much nearer while it closes at
>= release_fast_mps over the last 0.5 s (a median lags a fast approach), or, without measured depth, it looms in the
walking path (the near rule), or it closes at >= release_mps (measured) at every decision for release_span_s, or its
measured TTC is under release_ttc_s while it closes at approach_min_mps. That is an "approach" release (its next cue
passes). A box that moved by bg_release_disp is released without one, and so is a background person or animal
re-acquired after more than lost_s out of view (it may have moved meanwhile: the scheduler's cue about it is judged
like any other). The reference starts as the warm-up median and follows the object while it rests (recapture_s at
rest: depth within rest_spread_m, box size within rest_size_frac, not closing), so a lean during the warm-up or the
user settling a little nearer is not an approach later; an object released as approaching (or lost from view) that
then rests for recapture_s is held again (it came up and stopped: nothing new until it moves), so a false release
costs one cue, never a "Person." loop. A background object whose range is now max(closer_m, closer_frac x reference)
*farther* than its reference is not at its place either (the tracker matched another object that stepped in where it
stood during a short dropout, or it moved away): released without one, held again once at rest. An operator that
comes back to its place after an approach release remembers the nearest point it came to: the same track (one the gate
saw rest at the user's place) is released (or critical) again only lean_margin_m nearer than that, so a desk user's
repeated leans cost one cue, not one per lean (leaning deeper than ever before costs one more). The lean memory belongs
to that track: a stranger or a visitor later taken for the user (a new track id, a second box) never inherits it.

Rescue. A release must not depend on the scheduler offering the object again: P-CACS blocks a track for 2.8 s after
offering it, and its novelty decays, so its one offer may have been held; a learned scheduler offers a parked car
every few seconds and then not while it creeps in. The gate keeps the last cue it held per track (held_keep_s, or
repeat_keep_s for a repeat: a known object takes a few seconds to come materially closer; it follows the object across
id switches) and the last cue it let through per object. At a decision where no offered or
pending cue passes, it says one of them itself (distance and direction refreshed, about the object's current track)
when that object now qualifies: a held cue (the scheduler wanted it said) for critical / approach / near / closing /
closer / reacquired, or for new once its hold no longer applies (a warm-up, background, operator or read-mode hold
lifted, or the memory it repeated let go of the track: another person stepped in where the said one stood; a muted
cue: only once it is no longer muted); a cue said earlier only for what the gate's own rules make new: approach (and
then critical), or near, the latter only when the user walks straight into the object (box centre inside the
corridor). It asks for the first rule *among these* that holds, so a sector change or a reminder never hides closing.
When the channel is busy only critical, approach and near cut in, never on a message about the same object, and only a
critical one on a message about safety the gate let through (a vehicle, the FIND goal, an object in the path or
corridor, a measured approach: harness find:511113, a near rescue cut the FIND goal's cue before it named it), and that
only if the user reaches its object sooner than the playing message's (measured TTCs: find:400183). Lost: a
cue about an object (not a sign) in the walking path between operator_max_m and path_m that the user did not hear about
for hold_moving_s (or never), held as a repeat or background, whose object then leaves view at once (no track of its
identity over its last box), is said as the scheduler chose it at the next decision (on a free channel, within 1 s):
the gate cannot watch it come closer, and an approach that starts as it goes out of view is otherwise first seen when it
is back, a step away; a static object costs at most one such reminder per hold_moving_s. FIND: a goal cue held as a
repeat (or only a reminder) while another object the goal names is in view and was never said is said about that one
instead (a scheduler may keep naming the person it already named, never the other one the user asked for too), unless
the goal object it names may carry news itself (a vehicle, a measured approach, or in the walking path and half the
`closing` step nearer than said, or closing now: a start is not told from depth noise any sooner): its cue is then
judged as any other. A swapped cue waits as held, and its object's warning (critical, approach, closing, near,
reacquired, closer) cuts the swap's message: the scheduler's cue would have been playing there instead.

Cut messages. A message the gate let through that an interrupt or a spoken reply cuts before the user heard what it is
about and how far (heard_enough: the words through its distance phrase) is taken back (cut): the memory is as before
it, so the scheduler's next cue about the object is not held as a repeat of words nobody heard. The gate notices its
own interrupts on the real channel; the LiveController, which shows it the scheduler's channel, reports the others.

Without measured depth, box growth (near, closing, release) compares the medians of the boxes within 0.25 s of each
end (3 frames at 10 fps), so a jittering box of a static sign or chair is not an object closing in.

"Measured" TTC / speed means the track has depth (depth_conf >= 0.4) and >= 4 range samples spanning >= 1 s, all of
them depth samples: the tracker's speed window (2 s) starts RATE_SETTLE_S after the decision that first saw the depth
(a range rate across a switch between the box and the depth model reads the step between them as an approach: the
desk user's cut-off box says 1.8 m, the depth 0.7 m). Without depth, geometry.TrackDynamics *assumes* the user walks at
1 m/s, so a static person 1.5 m in front of a laptop gets TTC 0.9 s and would look critical forever; a slope through
the first 4 samples (0.3 s) is noise.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..geometry.camera import DEFAULT_SIZE_M, TYPICAL_SIZE_M, CameraModel
from ..types import BBox, Cue, Track, bbox_area, bbox_center, bbox_iou

# labels that are the same physical object to the user when a tracker id switch also flips the class
LABEL_ALIAS = {"truck": "car", "bus": "car", "van": "car", "motorcycle": "bicycle", "motorbike": "bicycle",
               "scooter": "bicycle", "bike": "bicycle", "tram": "train", "pedestrian": "person", "people": "person"}
# identities (after LABEL_ALIAS) that are never background or held for the warm-up: they move under their own power
VEHICLES = frozenset({"car", "bicycle", "train"})
# background identities that may have moved while out of view (a person lost for a second is not known to be static)
SELF_MOVING = VEHICLES | {"person", "dog", "cat", "horse", "cow", "sheep", "bird"}
# how fast an object of the identity may come closer while out of view (m/s; anything else does not move itself), and the
# user's own walking: a successor track nearer than that could have come is another object (_same_range)
_MAX_SPEED = {"car": 20.0, "train": 20.0, "bicycle": 10.0, "person": 3.0, "dog": 5.0, "cat": 5.0, "horse": 8.0,
              "cow": 3.0, "sheep": 3.0, "bird": 5.0}
_WALK_MPS = 1.5
# how fast an object of the identity may move *away* between two samples (_range_jump): a person walks off at a brisk
# pace at most (a sprint away is rare; one stepping in where another stood as it leaves is the case that matters)
_AWAY_MPS = {"person": 2.0}
SPEECH_ACTIONS = ("speak", "interrupt")
DISPLACED_KEEP_S = 3.0           # a routine scheduler cue that waits for the gate's own message to end: said within this
# which stats counter a hold reason goes to (anything else: suppressed_background)
_HOLD_STAT = {"repeat": "suppressed_repeat", "critical_gap": "suppressed_repeat",
              "unconfirmed": "suppressed_unconfirmed", "haptic_gap": "suppressed_haptic", "muted": "suppressed_muted"}
_SILENCE = ("warmup", "background", "operator", "muted")
# rescue (module docstring): the reasons the gate says a cue itself for, most urgent first. From a cue it held (the
# scheduler wanted it said) all of them (`new` once its hold no longer applies: a warm-up / background / operator /
# read-mode hold lifted, or the memory it repeated let go of the track); from a cue it let through earlier only what
# the gate's own release or the missing depth makes new (approach, near)
_APPROACH_REASONS = frozenset({"critical", "approach", "closing", "near", "reacquired"})   # pass rules about an approach
_RESCUE_PRIO = {"critical": 5, "approach": 4, "near": 3, "closing": 3, "closer": 2, "reacquired": 2, "new": 1}
_RESCUE_HELD = frozenset(_RESCUE_PRIO)
_RESCUE_SAID = frozenset({"approach", "near"})
_EDGE = 0.01                     # a box side within this of the frame edge is cut off (its size says nothing)
_BOX_WIN_S = 0.25                # box growth: the median of the boxes this close to each end (3 frames at 10 fps)
# a track's range source changed (box <-> measured depth: the depth model switched on or off, its confidence crossed
# 0.4): its range rate counts again only once the tracker's speed window (geometry.DynamicsConfig.history_s) holds no
# sample from before the change plus this long (the tracker's depth EMA settles), so the step between the two sources
# is no approach (v4_desk_still:900231: the desk user, 1.8 m by the cut-off box, 0.7 m by depth when it came back,
# read as closing at 0.6 m/s with TTC 0.1 s)
RATE_SETTLE_S = 0.5
# without measured depth the box range assumes the label's typical size: an object half that size (a child, a small
# dog, a stool) reads twice as far. The say-once rule (bg_path_m) takes it at its nearest (v4_flip_size:970034: a child
# 2.6 m away in the path read 3.8-4.1 m, beyond bg_path_m, and a rebaseline's warm-up held the first cue about it)
BOX_SMALL = 0.5
_RANGE_WORDS = re.compile(r"\b(meters?|metres?|feet|foot|close|front of you)\b|\d\s*(m|ft)\b")   # (a distance phrase)


@dataclass
class GateConfig:
    hold_static_s: float = 30.0      # same object, same sector, not materially closer: do not repeat within this
    hold_moving_s: float = 8.0       # ... for an object that is moving
    closer_m: float = 1.0            # re-announce when the distance dropped by at least closer_m
    closer_frac: float = 0.25        # ... and by at least this fraction of the distance at the last announcement
    sector_deg: float = 20.0         # left < -sector_deg, ahead, right > +sector_deg (camera angle of the box centre)
    min_hits: int = 3                # tracker matches before a track's first spoken cue (held cues wait up to pending_s)
    confirm_age_s: float = 0.15      # ... or 2 matches at least this far apart: 3 hits at 10-15 fps, 2 at the 5 fps replays
    haptic_gap_s: float = 3.0        # at most one haptic per identity and sector in this window
    critical_ttc_s: float = 2.0      # measured TTC below this ...
    critical_path: float = 0.5       # ... with path_prob at or above this always passes
    warmup_s: float = 3.0            # objects present and static during the first warmup_s become background
    static_disp: float = 0.03        # max box-centre displacement (normalised) to count as static during the warm-up
    background: bool = True          # suppress background objects unless they start approaching
    operator_filter: bool = False    # desk webcam: the large, centred, static person present from the start is the user
    # refinements beyond the brief (module docstring; evidence from EgoBlind finetune+select replays and a desk simulation)
    hold_goal_s: float = 8.0         # a FIND goal may be repeated this often (the user asked for it)
    pending_s: float = 1.0           # a cue held for confirmation is said within this long once the track is confirmed
    min_gap_s: float = 2.5           # no non-critical repeat of the same object sooner than this (closer / sector / near /
                                     # closing / interrupt)
    switch_gap_s: float = 5.0        # a new track inherits a vanished track's memory only if it appeared within this long
                                     # after the old one was last seen (a different car 30 s later is new, not a repeat)
    sector_hyst_deg: float = 5.0     # an angle must cross a sector boundary by this much to count as a sector change
    moving_speed_mps: float = 0.3    # |range rate| or lateral speed (+0.05 m/s per metre of range) above this = moving
    approach_ttc_s: float = 6.0      # interrupt rule: an object with measured TTC below this ...
    approach_min_mps: float = 0.5    # ... closing at least this fast is approaching (leaning toward a laptop is ~0.2 m/s);
                                     # also the closing speed an operator / background track needs to be critical
    static_size_frac: float = 0.2    # max relative change of the box height to count as static during the warm-up
    bg_release_disp: float = 0.15    # a background object whose box centre moved this far is no longer background
    operator_min_height: float = 0.55   # operator: box height (normalised) above this ...
    operator_center_tol: float = 0.2    # ... box centre within 0.5 +- this horizontally ...
    operator_disp: float = 0.12         # ... and centre displacement during the warm-up at most this (people fidget)
    dup_iou: float = 0.6             # two boxes of one label in view together, overlapping this much at the same range,
                                     # are one object detected twice (else two objects: neither inherits the other's memory)
    operator_dup_iou: float = 0.6       # while the user's track is visible, another person box overlapping it this much
                                        # is a duplicate detection of the user (held, but the user box does not move)
    critical_gap_s: float = 2.0      # at most one critical cue per object in this window ...
    critical_worse_ttc_s: float = 0.5   # ... unless its TTC dropped by this much since the last critical cue ...
    critical_worse_m: float = 0.3       # ... or its distance by this much (a worsening hazard always passes)
    nodepth_near_m: float = 2.0      # without measured depth: an object in the path nearer than this ...
    nodepth_approach_mps: float = 0.3   # ... that the range rate or the box growth shows closing this fast: after min_gap_s
    bg_path_m: float = 3.0           # an object in the walking path nearer than this is said once before it is background
    path_m: float = 6.0              # an object in the walking path or corridor nearer than this (6 s at a walking pace) is
                                     # said again at the finer `closing` step, and reminded after hold_moving_s
    corridor_half_m: float = 0.45    # half-width of the walking corridor (geometry.DynamicsConfig.corridor_half_m)
    # safety: who may be held at all, when a held object is released, and the gate saying a held cue itself
    operator_max_m: float = 1.2      # operator: at most this far away (a laptop user sits 0.4-1.0 m from the webcam)
    release_m: float = 0.35          # background / operator released when its distance is below its reference (where it
                                     # rests) by max(release_m, release_frac x reference) (0.3 m + a margin for 5 cm depth
                                     # jitter: the desk user leans in by 0.25 m); also the step of the `closing` rule ...
    release_frac: float = 0.15
    release_smooth_s: float = 0.3    # ... (measured depth: the median over this long; no depth: the last 3 boxes) ...
    release_fast_mps: float = 0.6    # ... or by its latest sample while closing this fast over the last 0.5 s ...
    release_mps: float = 0.25        # ... or it closes at least this fast (measured) at every decision ...
    release_span_s: float = 1.0      # ... for this long ...
    release_ttc_s: float = 3.0       # ... or its measured TTC is under this while it closes at approach_min_mps
    recapture_s: float = 3.0         # background / operator at rest this long: its reference moves there, and one released
                                     # as approaching is held again (one false release costs one cue, not a "Person." loop)
    rest_spread_m: float = 0.15      # at rest: smoothed depth within this ...
    rest_size_frac: float = 0.08     # ... and the box width / height within this fraction
    warm_min_samples: int = 3        # box samples before the warm-up may hold a track (a first sample is never "static")
    warm_hold_span_s: float = 0.4    # ... spanning at least this long (10 fps: from the first decision after it appears)
    bg_min_span_s: float = 1.0       # background: samples spanning at least this long inside the warm-up, no closing
    held_keep_s: float = 3.0         # the gate says a cue it held itself when the object qualifies within this long ...
    repeat_keep_s: float = 8.0       # ... or, held as a repeat (the user knows the object) or muted, within this long: it
                                     # takes a few seconds to come materially closer, and a sparse scheduler offers a
                                     # parked car every 4-5 s, then not while it creeps in
    flicker_keep_s: float = 3.0      # a cue held for confirmation whose track flickered out waits this long for it to
                                     # come back (or for a successor track of the same identity)
    approach_keep_s: float = 5.0     # after an approach release the object's next cue passes, within this long
    lost_s: float = 0.8              # out of view longer than this (the tracker's max_age: 8 frames at 10 fps), it may
                                     # have moved meanwhile: a background person / animal re-acquired is not background
                                     # until it rests for recapture_s again; an object back in the path: `reacquired`
    lean_margin_m: float = 0.15      # operator released and re-captured at its place: released again only this much nearer
                                     # than the nearest point it came to before (a desk user's repeated leans)
    lean_noise_m: float = 0.05       # operator by measured depth: its release step plus max(this, 2 x its depth noise at
                                     # rest) (a 0.34 m lean on 4 cm depth noise and a 1 cm reference error: desk:900054)
    nodepth_grow: float = 0.10       # no measured depth: the box grew by at least this fraction (closing evidence) ...
    nodepth_bottom_frac: float = 0.15   # ... near: box bottom in the lowest nodepth_bottom_frac of the frame (in the corridor)
    jump_gap_s: float = 0.25         # background: its track's range jumped farther across a dropout this long (or a new
    jump_m: float = 0.3              # track over its place starts farther than it was last) by max(jump_m, jump_frac x
    jump_frac: float = 0.1           # range): another object stepped in where it stood, not the object at its place
    reacquired_m: float = 0.1        # reacquired: back in view at least max(reacquired_m, reacquired_frac x range) nearer
    reacquired_frac: float = 0.03    # than when last seen (a static object's depth noise rarely; one that started to
                                     # approach as it dropped out, at 0.7 m/s, is 0.2 m nearer 0.3 s after its return)
    # read mode (TaskAdapter: MODE_SETTINGS["read"].objects_muted). With this field the adapter leaves the muting to the
    # gate, which applies the adapter's own filter before it remembers a cue as said
    objects_muted: bool = False      # hold object cues that are not about safety ...
    muted_severity: float = 0.9      # ... (passes: severity >= this, TTC under approach_ttc_s, critical, interrupt)


@dataclass
class _Said:
    """One remembered announcement."""
    identity: Tuple[str, str]
    sector: str
    angle: float
    t: float
    dist: Optional[float]
    track_id: int
    bbox: Optional[BBox]
    goal: bool = False
    seen: float = 0.0                # last decision time its track was in view (id-switch inheritance window)
    apart: Set[int] = field(default_factory=set)   # same-identity tracks seen at the same decision as a separate box
    dups: Set[int] = field(default_factory=set)    # ... as a second box over it at the same range (one object twice)
    tids: Set[int] = field(default_factory=set)    # every track id this entry was about (a re-acquired older id is it)
    crit_t: Optional[float] = None   # last critical cue about this object: time, TTC and distance then
    crit_ttc: float = math.inf
    crit_dist: Optional[float] = None
    cue: Optional[Cue] = None        # the cue said (the gate may say it again itself: rescue)
    rng: Optional[float] = None      # range then by measured depth, else by the box (no-depth closing rule)


@dataclass
class _Warm:
    """What the gate saw of one track during the warm-up (box samples from the track's history, one per frame)."""
    identity: Tuple[str, str]
    kind: str
    label: str
    c0: Tuple[float, float]
    h0: float
    first_t: float
    last_t: float
    bbox: BBox
    n: int = 0                       # box samples seen (the track's history, one per frame)
    hits: int = 0                    # tracker matches (a scripted track without box history still counts its frames)
    max_disp: float = 0.0
    max_size: float = 0.0
    path: float = 0.0                # path_prob at the last decision
    closing: bool = False            # a measured closing speed >= release_mps was seen
    dists: List[float] = field(default_factory=list)    # distance at each decision (any source)
    depths: List[float] = field(default_factory=list)   # measured depth (depth_conf >= 0.4) at each decision
    widths: List[float] = field(default_factory=list)   # box widths not cut off at the left / right frame edge
    heights: List[float] = field(default_factory=list)  # box heights not cut off at the top / bottom frame edge
    areas: List[float] = field(default_factory=list)
    trk: Optional[Track] = None      # the tracker's object (its last match: lost-from-view after the warm-up)


@dataclass
class _Background:
    identity: Tuple[str, str]
    kind: str
    label: str
    bbox0: BBox                      # box at the end of the warm-up (release-on-displacement reference)
    bbox: BBox                       # last matched box (re-association after an id switch)
    dist_ref: Optional[float]        # median measured depth in the warm-up
    w_ref: Optional[float] = None    # median unclipped box width / height / area in the warm-up (no-depth release)
    h_ref: Optional[float] = None
    a_ref: Optional[float] = None
    track_ids: Set[int] = field(default_factory=set)
    released: bool = False
    say_once: bool = False           # in the walking path and near: said once before it is silenced
    said: bool = False
    approach: bool = False           # released because it closed in (its next cue passes)
    recapture: bool = False          # released because it closed in or was lost from view: held again once at rest
    rest: List[Tuple] = field(default_factory=list)   # (t, depth, width, height, area, centre) while not closing
    last_trk: Optional[Track] = None  # its track last seen at a decision (lost from view: lost_s, by its last match)
    nearest: Optional[float] = None  # operator: nearest range (m) since its approach release
    lean: Optional[float] = None     # operator: nearest range it came to and came back from (released again only nearer)
    rest_tids: Set[int] = field(default_factory=set)   # operator: tracks seen at rest at its place (the lean memory's)
    noise: float = 0.0               # operator: spread (sigma, m) of its measured range at rest (_depth_noise)


@dataclass
class _Seen:
    """A track as the gate last saw it at a decision, for tracker id switches (module docstring: Identity)."""
    identity: Tuple[str, str]
    angle: float
    t: float                         # last decision it was in view
    trk: Track                       # the tracker's object: its last_seen moves on while the tracker still matches it
    carried: int = 0                 # matches of the tracks it took over from (confirmation survives an id switch)
    first: float = 0.0               # first_seen of the first of them
    t0: float = 0.0                  # first decision this track was in view
    gap: float = 0.0                 # how long its object was out of view before this track took over (0: said since)
    back_rng: Optional[float] = None    # its range when it was last seen before that
    pred: Tuple[int, ...] = ()       # the track ids it took over from, the latest first
    c0: Tuple[float, float] = (0.5, 0.5)   # box centre where its object (first of the tracks it took over from) appeared
    moved: float = 0.0               # farthest its box centre has been from c0 (an arrival: not background)
    b0: Optional[BBox] = None        # its box there (_part_of: an arm attached to the desk user from its first sighting)
    r0: Optional[float] = None       # its range there (_known_user: near from its first sighting, no one who walked up)
    r_in: Optional[float] = None     # this track's own first range (a successor: where it came back; _closing_in_path)


def _speech_s(cue: Cue) -> float:
    """How long a cue holds the channel: its duration (the channel sets it when it speaks), else words x 0.35 s, at
    least 1.2 s (eval.runner.SpeechChannel)."""
    d = float(cue.duration_s or 0.0)
    return d if d > 0 else max(1.2, 0.35 * len((cue.message or "").split()))


def heard_enough(cue: Cue, heard_s: float) -> bool:
    """Did the user hear what a message is about and how far it is, when it was cut heard_s seconds in? The words through
    its distance phrase ("Person, about 3 meters, ...": 4; "Chair, very close.": 3), or through its first comma (the
    object or the sign's text) when it has none, at the pace the message was spoken (its duration over its words)."""
    words = (cue.message or "").split()
    if not words:
        return True
    parts, need = (cue.message or "").split(","), 0
    for part in parts:
        need += len(part.split())
        if _RANGE_WORDS.search(part.lower()):
            break
    else:
        need = len(parts[0].split())
    return len(words) * max(0.0, heard_s) / _speech_s(cue) >= need - 1e-6


def normalize_text(s: Optional[str]) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


# OCR confusions: letters read in place of digits inside a number, digits read in place of letters inside a word
_OCR_TO_DIGIT = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "z": "2", "b": "8"})
_OCR_TO_LETTER = str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b"})


def sign_key(txt: str) -> Tuple[str, str]:
    """(canonical text, its digit sequence) of a normalised OCR reading, for comparing two readings: a token that is
    at least half digits is a number and letters OCR confuses with digits are read as digits ("2o4" -> "204"); in a
    word such digits are read as letters ("r0om" -> "room")."""
    toks = []
    for tok in txt.split():
        nd = sum(ch.isdigit() for ch in tok)
        if nd and 2 * nd >= len(tok):
            tok = tok.translate(_OCR_TO_DIGIT)
        elif nd:
            tok = tok.translate(_OCR_TO_LETTER)
        toks.append(tok)
    canon = " ".join(toks)
    return canon, "".join(ch for ch in canon if ch.isdigit())


def same_sign(a: str, b: str, ratio: float = 0.8) -> bool:
    """Are two normalised OCR readings the same sign? Only if their numbers are identical (room 204 is not room 206,
    exit is not exit 2), their lengths are similar, and the rest differs by OCR noise (SequenceMatcher >= ratio)."""
    if a == b:
        return True
    ca, da = sign_key(a)
    cb, db = sign_key(b)
    if da != db or abs(len(ca) - len(cb)) > max(1, round(0.2 * max(len(ca), len(cb)))):
        return False
    return ca == cb or SequenceMatcher(None, ca, cb).ratio() >= ratio


def goal_words_match(goal: str, key: str) -> bool:
    """Does the normalised goal ("room 206", "chair") name this normalised label or sign text ("room 206 lab")? As a
    whole word / phrase, so "206" names "room 206" but "20" does not."""
    return bool(goal) and bool(key) and (goal == key or f" {goal} " in f" {key} ")


def sector_of(angle_deg: float, sector_deg: float) -> str:
    if angle_deg < -sector_deg:
        return "left"
    return "right" if angle_deg > sector_deg else "ahead"


def in_sector(angle_deg: float, sector: str, sector_deg: float, hyst_deg: float = 0.0) -> bool:
    """Is the angle inside `sector`, widened by `hyst_deg` on every boundary (hysteresis)?"""
    if sector == "left":
        return angle_deg < -sector_deg + hyst_deg
    if sector == "right":
        return angle_deg > sector_deg - hyst_deg
    return -sector_deg - hyst_deg <= angle_deg <= sector_deg + hyst_deg


def history_span_ok(trk: Track, min_span_s: float = 1.0) -> bool:
    """>= 4 range samples spanning >= min_span_s (whatever their source: depth model or box size)."""
    h = trk.dist_history
    return len(h) >= 4 and (h[-1][0] - h[0][0]) >= min_span_s


def measured(trk: Track, min_span_s: float = 1.0) -> bool:
    """Range rate / TTC come from real depth samples over at least `min_span_s`, not from the assumed 1 m/s walking
    speed (depth_conf < 0.4) or a slope through the first few noisy samples (4 samples over 0.3 s at 10 fps)."""
    return trk.depth_conf >= 0.4 and history_span_ok(trk, min_span_s)


def materially_closer(d_then: Optional[float], d_now: Optional[float], closer_m: float, closer_frac: float) -> bool:
    if d_then is None or d_now is None:
        return False
    return (d_then - d_now) >= max(closer_m, closer_frac * d_then)


def _cut_lr(b: BBox) -> bool:
    return b[0] <= _EDGE or b[2] >= 1.0 - _EDGE


def _cut_tb(b: BBox) -> bool:
    return b[1] <= _EDGE or b[3] >= 1.0 - _EDGE


def _med(xs: Sequence[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


class AnnouncementGate:
    """Wraps any policy (duck-typed: decide/select, reset, name, decision_interval, cfg, ...) and filters its cues.

    `gate.cfg` is the *inner* policy's config (verbosity, units); the gate's own settings are `gate.gate_cfg`
    and change at run time through `set_mode_params(**kw)`. Unknown attributes are read from (and, when the inner
    policy has them, written to) the inner policy, so the gate is a drop-in replacement in the live pipeline.
    """

    _OWN = frozenset({"gate_cfg", "cam", "stats", "reasons", "last_reason", "last_verdict", "releases"})

    def __init__(self, policy, cfg: Optional[GateConfig] = None, camera: Optional[CameraModel] = None,
                 name: Optional[str] = None):
        object.__setattr__(self, "_inner", policy)
        self.gate_cfg = cfg or GateConfig()
        self.cam = camera or getattr(policy, "cam", None) or CameraModel()
        self._name = name
        self._goal_tags: List[str] = [g.lower() for g in (getattr(policy, "goal_tags", None) or [])]
        # the gate sees the real speech channel in decide, so it notices when its own interrupt cuts the message it let
        # through before (cut). The LiveController shows it the scheduler's channel instead and reports cuts itself
        self._own_cuts = True
        self._clear()

    # ---- delegation -----------------------------------------------------------------------------------------
    def __getattr__(self, k: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None or k.startswith("__"):
            raise AttributeError(k)
        return getattr(inner, k)

    def __setattr__(self, k: str, v: Any) -> None:
        if k == "goal_tags":                     # keep the gate's own copy (used by _is_goal) in step
            object.__setattr__(self, "_goal_tags", [str(g).lower() for g in (v or [])])
        if k in self._OWN or k.startswith("_") or isinstance(getattr(type(self), k, None), property):
            object.__setattr__(self, k, v)
        elif hasattr(self._inner, k):
            setattr(self._inner, k, v)
        else:
            object.__setattr__(self, k, v)

    @property
    def inner(self):
        return self._inner

    @property
    def name(self) -> str:
        return self._name or getattr(self._inner, "name", "policy")

    @name.setter
    def name(self, v: str) -> None:
        self._name = v

    @property
    def decision_interval(self) -> float:
        return getattr(self._inner, "decision_interval", 0.5)

    @decision_interval.setter
    def decision_interval(self, v: float) -> None:
        self._inner.decision_interval = v

    @property
    def supports_actions(self) -> bool:
        return bool(getattr(self._inner, "supports_actions", False))

    @property
    def cfg(self):
        return getattr(self._inner, "cfg", None)

    @cfg.setter
    def cfg(self, v) -> None:
        self._inner.cfg = v

    def last_observations(self):
        f = getattr(self._inner, "last_observations", None)
        return f() if callable(f) else []

    def _current_goal_tags(self) -> List[str]:
        """The FIND goal in force: the inner policy's goal_tags when it has that attribute (so `gate.goal_tags = [...]`,
        written through to the inner policy, and a goal set on the inner policy directly both count), else the gate's
        own copy (set_goal / gate.goal_tags)."""
        tags = getattr(self._inner, "goal_tags", None)
        if tags is None:
            tags = self._goal_tags
        return [str(g).lower() for g in tags or []]

    def set_goal(self, tags: Optional[Sequence[str]]) -> None:
        """Forward a FIND request; memory of objects and signs the new goal names (as a word or phrase, normalised:
        "room 206" names the sign "ROOM 206 LAB") is dropped, so the next goal cue is said at once."""
        new = [g.lower() for g in (tags or [])]
        old = self._current_goal_tags()
        f = getattr(self._inner, "set_goal", None)
        if callable(f):
            f(tags)
        if new and new != old:
            goals = [k for k in (normalize_text(g) for g in new) if k]
            for ident in list(self._mem):
                if any(self._names(gk, ident[1]) for gk in goals):
                    del self._mem[ident]
        self._goal_tags = new

    @staticmethod
    def _names(goal: str, key: str) -> bool:
        """Does a normalised goal name a normalised label / text? ("seat" names a chair or a bench, as in P-CACS.)"""
        return goal_words_match(goal, key) or (goal == "seat" and key in ("chair", "bench"))

    # ---- state ----------------------------------------------------------------------------------------------
    def _clear(self) -> None:
        self._mem: Dict[Tuple[str, str], List[_Said]] = {}
        self._haptic: Dict[Tuple[Tuple[str, str], str], float] = {}
        self._crit_haptic: Dict[Tuple[Tuple[str, str], str], Tuple[float, float, Optional[float]]] = {}
        self.stats: Dict[str, int] = {"passed": 0, "suppressed_repeat": 0, "suppressed_background": 0,
                                      "suppressed_unconfirmed": 0, "suppressed_haptic": 0, "suppressed_muted": 0,
                                      "rescued": 0}
        self.reasons: Counter = Counter()
        # per track id; a successor track (id switch, module docstring: Identity) takes them over
        self._pending: Dict[int, Tuple[Cue, float, float, bool]] = {}   # -> (cue, time held, angle, lost from view since)
        self._held: Dict[int, Tuple[Cue, float, str, float]] = {}       # -> (cue, time, hold reason, angle)
        self._lost_due: Dict[int, float] = {}                           # -> time a stale in-path cue was held (_say_lost)
        self._held_ref: Dict[int, float] = {}                           # -> its smoothed range then (rescue: closing)
        self._closing_ref: Optional[float] = None      # while the rescue judges such a cue: that range (_closing_in_path)
        self._approach: Dict[int, float] = {}                           # -> time of its approach release
        self._closing_since: Dict[int, float] = {}                      # -> since when it closes >= release_mps
        self._seen: Dict[int, _Seen] = {}
        self._jumps: Dict[int, float] = {}                              # -> its latest range jump farther (_detach_jumps)
        self.releases: List[Tuple[float, int, str]] = []             # (time, track id, why) of every release / re-capture
        self._rescuing: Optional[Set[str]] = None      # while the rescue judges its cue: the reasons it may say it for
        self._said_last: Optional[Tuple[Cue, float, bool]] = None   # the last speech cue let through, when, and whether
                                                                     # it is the gate's own (not the scheduler's at its time)
        self._offer: Optional[Tuple[Cue, float, float]] = None   # the scheduler's last speech cue on its own channel
                                                                  # (passed or held): (cue, offered, end alone)
        self._waiting: Optional[Tuple[Cue, float, float]] = None   # a routine cue that passed while another message
                                                                    # played: (cue, time, angle) (decide)
        # per track id: [measured depth now, since when every new range sample is of that source (the last decision it
        # was seen with the other one), its Track, last decision seen] (RATE_SETTLE_S, _rate_ok)
        self._src: Dict[int, list] = {}
        # per track id: what its latest spoken cue changed in the memory, to take back if the message is cut before the
        # user heard it (cut): (time, identity, entry, entry before or None, approach release, background said flags,
        # reacquired gap, owed a say-once cue)
        self._undo: Dict[int, Tuple] = {}
        self._owed: Set[int] = set()     # tracks a rebaseline found owed a say-once cue (rebaseline)
        self._swapped: Optional[Tuple[Cue, int]] = None   # the last FIND swap: its message, and the track whose cue it
                                                          # replaced (that object's warning may cut it: _rescue)
        self.last_reason: str = ""
        self.last_verdict: Optional[Dict[str, Any]] = None
        self._rebaseline_state()

    def _rebaseline_state(self) -> None:
        self._t0: Optional[float] = None
        self._warm: Dict[int, _Warm] = {}
        self._warm_done = False
        self._bg: List[_Background] = []
        self._operator: Optional[_Background] = None

    def reset(self) -> None:
        self._clear()
        f = getattr(self._inner, "reset", None)
        if callable(f):
            f()

    def rebaseline(self) -> None:
        """Forget the background / operator and restart the warm-up at the next decision (e.g. the user sat down
        somewhere new). What was said, held cues and approach releases are kept, and so is what the gate still owes the
        user: an object in the walking path the old warm-up found must be said once and that was not said yet is not
        held by the new warm-up and is said once before it is background again (a rebaseline must not turn first news
        into silence: v4_flip_size:970034, auto setup's desk detection)."""
        for b in self._bg:
            if b.say_once and not b.said and not b.released:
                self._owed |= b.track_ids
        self._rebaseline_state()

    def set_mode_params(self, **kw) -> Dict[str, Any]:
        """Change GateConfig fields at run time (in place); returns the previous values, so
        `gate.set_mode_params(**old)` restores them. Unknown field names raise ValueError."""
        valid = {f.name for f in fields(GateConfig)}
        bad = sorted(set(kw) - valid)
        if bad:
            raise ValueError(f"unknown GateConfig field(s) {bad}; valid: {sorted(valid)}")
        old = {k: getattr(self.gate_cfg, k) for k in kw}
        for k, v in kw.items():
            setattr(self.gate_cfg, k, v)
        return old

    # ---- scheduler interface --------------------------------------------------------------------------------
    def decide(self, tracks: List[Track], t: float, ch=None, *args, **kw) -> List[Cue]:
        ch_in = self._scheduler_channel(ch, t)
        cues = self._inner.decide(tracks, t, ch_in, *args, **kw)
        c0 = cues[0] if cues else None           # (the pipeline plays the first cue only)
        if (c0 is not None and c0.action in SPEECH_ACTIONS
                and (c0.action == "interrupt" or not getattr(ch_in, "busy", False))):
            # alone, this cue holds the channel from now for its spoken length (a "speak" onto a busy channel is dropped
            # by the speaker, alone too: it changes nothing)
            self._offer = (c0, t, t + _speech_s(c0))
        said_last = self._said_last
        out = self.filter(cues, tracks, t, ch=ch)
        v = self.last_verdict
        if (getattr(ch, "busy", False) and out and out[0].action == "speak" and v is not None and not v["deferred"]
                and v["t"] == t):
            # the scheduler was shown its own channel, free, while the real one plays another message (the gate's own, a
            # cue said late, a reply or a cane cue): the speaker would drop a "speak"
            trk = next((x for x in tracks if x.track_id == out[0].track_id), None)
            if self._is_safety(out[0], trk, v["reason"]):
                out = [replace(out[0], action="interrupt", interrupted_track=getattr(ch, "current_track", None))]
            else:                                # routine: wait for the channel
                ang = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
                self._waiting, self._said_last, out = (out[0], t, ang), said_last, []
        out = out or self._say_waiting(tracks, t, ch)
        if (self._own_cuts and out and out[0].action == "interrupt" and said_last is not None
                and getattr(ch, "busy", False) and getattr(ch, "current_track", None) == said_last[0].track_id
                and said_last[0].track_id != out[0].track_id and t < said_last[1] + _speech_s(said_last[0]) - 1e-9):
            self.cut(said_last[0], said_last[1], t - said_last[1])    # (this interrupt cuts the message it let through)
        return out

    def _say_waiting(self, tracks: List[Track], t: float, ch) -> List[Cue]:
        """The routine cue that passed while another message played (decide), on a free channel within DISPLACED_KEEP_S
        (distance and direction refreshed): the live speaker drops a message it gets while it speaks, so the cue the gate
        recorded as said would never be heard. Dropped when its object is talked about now. Said late, it is the gate's
        own message for the scheduler's view: alone, the scheduler's channel held it from when it was offered."""
        w = self._waiting
        busy = bool(getattr(ch, "busy", False))
        if w is None or (busy and getattr(ch, "current_track", None) == w[0].track_id) or t - w[1] > DISPLACED_KEEP_S + 1e-9:
            self._waiting = None
            return []
        if busy or t <= w[1] + 1e-9:
            return []
        self._waiting = None
        trk = next((x for x in tracks if x.track_id == w[0].track_id and not x.remembered and x.misses == 0), None)
        out = replace(self._refreshed(w[0], w[2], trk, t) if trk is not None else w[0], t=t, action="speak")
        self._said_last = (out, t, True)
        return [out]

    def _scheduler_channel(self, ch, t: float):
        """The channel the scheduler is shown: the one it would have alone (module docstring: The scheduler's view),
        built from its own offers only. While the real channel plays the scheduler's last offer, let through when it was
        offered, the real channel (it is the same message; the live speaker may run longer than the estimate). Otherwise
        its last offer (passed, held, or said later) while that would still be playing alone, from when it was offered,
        else a free channel, whatever the real channel plays: the gate's own message (rescue, lost, deferred, FIND swap),
        the scheduler's cue said late (waiting), another cue of its list, or a message the scheduler never chose (a reply,
        a cane cue). A channel busier than alone holds back a warning it would give alone (v3_find_disp:620507: a routine
        sign cue said 1 s late kept the learned scheduler from ever naming the person walking at the user), a freer one
        lets it speak where alone it would wait: either way its next cues are not the ones it gives alone."""
        from ..cues.pcacs import ChannelState
        o, last = self._offer, self._said_last
        if (ch is not None and getattr(ch, "busy", False) and o is not None and last is not None and not last[2]
                and last[0] is o[0] and abs(last[1] - o[1]) < 1e-9 and getattr(ch, "current_track", None) == o[0].track_id):
            return ch
        if o is not None and o[2] > t + 1e-9:
            c = o[0]
            return ChannelState(busy=True, remaining_s=o[2] - t, current_utility=c.utility or 0.0,
                                current_severity=c.severity or 0.0, current_track=c.track_id)
        return ChannelState() if ch is not None else None

    def _is_safety(self, cue: Cue, trk: Optional[Track], reason: str) -> bool:
        """Is a passed cue about safety (it may cut a message the gate said on its own)? Passed as critical, approach,
        closing, near or reacquired, or about a vehicle, the FIND goal, an object in the walking path or corridor, or one
        with a measured TTC under approach_ttc_s."""
        g = self.gate_cfg
        if reason in _APPROACH_REASONS:
            return True
        if self._is_vehicle(self._identity_of(trk, cue)) or self._is_goal(cue, trk):
            return True
        if trk is None:
            return False
        if trk.path_prob >= g.critical_path:
            return True
        rng = self._range_now(trk)
        if rng is not None and self._in_corridor(trk.label, trk.bbox, rng):
            return True
        return self._measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.approach_ttc_s

    def select(self, tracks: List[Track], t: float, ctx=None, *args, **kw) -> List[Cue]:
        cues = self._inner.select(tracks, t, ctx, *args, **kw)
        return self.filter(cues, tracks, t)

    def observe(self, tracks: List[Track], t: float) -> None:
        """Warm-up bookkeeping. `filter` calls it; a caller may also call it every frame. Box samples come from the
        track's own history (every frame), so a track is judged static on what was seen of it, not on one sample."""
        g = self.gate_cfg
        if self._warm_done:
            return
        if self._t0 is None:
            # the warm-up clock starts when the camera first sees something (the first frames of a webcam are often
            # dark while auto-exposure settles), so the user is not missed because YOLO found them 1 s late
            if not any(not trk.remembered and trk.misses == 0 for trk in tracks):
                return
            self._t0 = t
        if t >= self._t0 + g.warmup_s:
            self._finish_warmup()
            return
        for trk in tracks:
            if trk.remembered or trk.misses > 0:
                continue
            samples = [(tt, b) for tt, b in trk.bbox_history if tt <= t + 1e-9] or [(t, trk.bbox)]
            w = self._warm.get(trk.track_id)
            if w is None:
                t0, b0 = samples[0]
                w = _Warm(self._identity_of(trk), trk.kind, trk.label, bbox_center(b0), max(1e-3, b0[3] - b0[1]),
                          t0, t0, b0)
                self._warm[trk.track_id] = w
                new = samples
            else:
                new = [(tt, b) for tt, b in samples if tt > w.last_t + 1e-9]
            for tt, b in new:
                self._warm_sample(w, tt, b)
            w.path, w.hits, w.trk = float(trk.path_prob), max(w.hits, int(trk.hits)), trk
            if trk.distance_m is not None:
                w.dists.append(float(trk.distance_m))
                if trk.depth_conf >= 0.4:
                    w.depths.append(float(trk.distance_m))
            if self._measured(trk) and -trk.rel_speed_mps >= g.release_mps:
                w.closing = True             # (a box-growth closing does not stop it becoming background: once it is,
                                             # an approach releases it at a finer step than `closer` would re-say it)

    @staticmethod
    def _warm_sample(w: _Warm, t: float, b: BBox) -> None:
        c = bbox_center(b)
        h = max(1e-3, b[3] - b[1])
        w.n += 1
        w.last_t, w.bbox = t, b
        w.max_disp = max(w.max_disp, math.hypot(c[0] - w.c0[0], c[1] - w.c0[1]))
        w.max_size = max(w.max_size, abs(h - w.h0) / w.h0)
        if not _cut_lr(b):
            w.widths.append(b[2] - b[0])
        if not _cut_tb(b):
            w.heights.append(h)
        w.areas.append(bbox_area(b))

    def filter(self, cues: Sequence[Cue], tracks: List[Track], t: float, ch=None) -> List[Cue]:
        """The first cue that passes the gate, as a one-element list, or [].

        A cue held only because its track is not confirmed yet is deferred, not dropped: it is said at a later
        decision once the track (or, after a flicker, its successor) is confirmed, if the scheduler has nothing better.
        The scheduler already marked it as announced, so it would not offer it again for recent_block_s. When nothing
        passes, the gate says a held or earlier cue itself if its object now qualifies for safety (rescue).
        `ch` (the ChannelState given to decide) only tells a deferred or rescued cue to wait while the channel is busy."""
        self._track_sources(tracks, t)
        self.observe(tracks, t)
        self._prune(t)
        by_id = {trk.track_id: trk for trk in tracks}
        visible = {trk.track_id: trk for trk in tracks if not trk.remembered and trk.misses == 0}
        self._succeed(visible, t)
        self._detach_jumps(visible, t)
        self._update_closing(visible, t)
        self._track_records(visible, t)
        self._late_operator(visible, t)
        idents: Optional[Dict[int, Tuple[str, str]]] = None
        for es in self._mem.values():
            for e in es:
                own = visible.get(e.track_id)
                if own is None:
                    continue
                e.seen = t
                # another track of the same identity in view now, as a separate box: a second object, so it must
                # never inherit this memory as an id switch later; a second box over it at the same range is the same
                # object detected twice (it may take the memory over when the first box goes)
                for o in visible.values():
                    if o is own or o.track_id in e.apart:
                        continue
                    if idents is None:
                        idents = {tid: self._identity_of(tr) for tid, tr in visible.items()}
                    if idents[o.track_id] != e.identity:
                        continue
                    if not self._dup_box(o, own):
                        e.apart.add(o.track_id)
                        e.dups.discard(o.track_id)
                    else:
                        e.dups.add(o.track_id)
        goal_tried = False
        for cue in cues or ():
            trk = by_id.get(cue.track_id)
            if not goal_tried and cue.action in SPEECH_ACTIONS and self._is_goal(cue, trk):
                goal_tried = True
                _, why, e, _ = self._check(cue, trk, visible, t)
                if why in ("repeat", "reminder") and not self._goal_news(trk, e):
                    out = self._goal_other(cue, visible, t, ch)
                    if out:
                        # (the scheduler's cue waits as held: the rescue says it if its object comes closer meanwhile,
                        # cutting this message: _rescue)
                        angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
                        self._held[cue.track_id] = (cue, t, "repeat", angle)
                        self._swapped = (out[0], cue.track_id)
                        return out
            out = self._judge(cue, trk, visible, t)
            if out is not None:
                return out
        out = self._flush_pending(visible, t, ch)
        if out:
            return out
        return self._rescue(visible, t, ch) or self._say_lost(visible, t, ch)

    def _goal_news(self, trk: Optional[Track], e: Optional[_Said]) -> bool:
        """May a FIND goal the scheduler names again carry news for the user's safety, so the gate must not say another
        goal object in its place (_goal_other)? A vehicle, a measured approach (TTC under approach_ttc_s), or an object
        in the walking path or corridor nearer than path_m that may have started to come at the user: half the `closing`
        step (max(release_m, release_frac x range) / 2) nearer than when it was last said, by its smoothed range, or
        closing now (_closing_now: measured, or its box growing). A start is not told from noise any sooner (half a
        second into a 1.5 m/s walk the tracker's closing speed reads 0.1-0.2 m/s), and the goal object the gate says in
        its place holds the channel while this one comes on: the approach, closing and near rescues wait for a message
        about the goal (v4_swap_fast:975020: a person walking at the user, offered as it started, was next named 1 s
        before contact; the scheduler's cue is held and the closing rescue says it a step later)."""
        g = self.gate_cfg
        if trk is None:
            return False
        if self._is_vehicle(self._identity_of(trk)) or self._approaching(trk, None):
            return True
        if e is None or e.rng is None or not self._in_path(trk, g.path_m):
            return False
        d = self._smoothed_range(trk)
        step = max(g.release_m, g.release_frac * e.rng)
        return (d is not None and e.rng - d >= 0.5 * step - 1e-9) or self._closing_now(trk)

    def _goal_other(self, cue: Cue, visible: Dict[int, Track], t: float, ch) -> List[Cue]:
        """FIND: the scheduler's cue about the goal is a repeat (held) or only a reminder while another object the goal
        names is in view and was never said since the goal was set (another person when the user asked for a person, the
        EXIT sign when the scheduler keeps naming EXIT 2): the gate says that one instead, on a free channel (the nearest;
        distance and direction refreshed). The user asked for the goal, so an unsaid one is news, and a scheduler may
        keep choosing the one it already said (harness find:2133: the learned scheduler named the same person every
        1.5 s, all held, and never the other one 6 m away on the left)."""
        if getattr(ch, "busy", False):
            return []
        trk0 = visible.get(cue.track_id)
        angle0 = self.cam.angle_deg_from_u(bbox_center(trk0.bbox)[0]) if trk0 is not None else 0.0
        cands = []
        for tid, trk in visible.items():
            if tid == cue.track_id or trk.kind != (trk0.kind if trk0 is not None else cue.kind):
                continue
            if not self._goal_names(trk) or not self._confirmed(trk):
                continue
            if self._own_entry(tid) is not None:
                continue                         # said before (under this track or one it took over from)
            rng = self._range_now(trk)
            cands.append((rng if rng is not None else math.inf, tid, trk))
        ident0 = self._identity_of(trk0, cue)
        for _, tid, trk in sorted(cands, key=lambda c: c[:2]):
            other = self._refreshed(cue, angle0, trk, t)
            if self._identity_of(trk) != ident0:     # (another sign or label: its own words)
                from ..cues.compose import compose
                rng = self._range_now(trk)
                msg = compose(("object", "distance", "direction"), trk.label, trk.kind, trk.text, distance_m=rng,
                              depth_conf=trk.depth_conf, angle_deg=self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]),
                              units=getattr(self.cfg, "units", "m") or "m")
                other = replace(other, message=msg, label=trk.label, kind=trk.kind, text=trk.text,
                                slots=("object", "distance", "direction"))
            self._rescuing = {"new"}
            try:
                out = self._judge(replace(other, action="speak"), trk, visible, t, deferred=True)
            finally:
                self._rescuing = None
            if out:
                self.stats["rescued"] += 1
                return out
        return []

    def _judge(self, cue: Cue, trk: Optional[Track], visible: Dict[int, Track], t: float, deferred: bool = False) -> Optional[List[Cue]]:
        # (a rescued cue is judged on the reasons the rescue may say it for: self._rescuing)
        ok, reason, entry, info = self._check(cue, trk, visible, t, allowed=self._rescuing or None)
        if ok and self._muted(cue, trk, reason):
            ok, reason, info = False, "muted", f"{info} (read mode: objects muted, passed {reason})"
        self.reasons[reason + (" (deferred)" if deferred else "")] += 1
        self.last_reason = f"{'pass' if ok else 'hold'} {reason}{' (deferred)' if deferred else ''}: {info}"
        self.last_verdict = {"t": t, "track_id": cue.track_id, "label": cue.label, "action": cue.action,
                             "passed": ok, "reason": reason, "info": info, "deferred": deferred}
        if ok:
            self.stats["passed"] += 1
            self._pending.pop(cue.track_id, None)     # (an older held cue for this track will never be said)
            if cue.action in SPEECH_ACTIONS:
                self._held.pop(cue.track_id, None)
                self._lost_due.pop(cue.track_id, None)
                self._held_ref.pop(cue.track_id, None)
                self._said_last = (cue, t, deferred)
            self._commit(cue, trk, entry, t, reason)
            return [cue]
        self.stats[_HOLD_STAT.get(reason, "suppressed_background")] += 1
        angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
        if reason == "unconfirmed":                # (superseding an older held cue for the same track)
            self._pending[cue.track_id] = (cue, t, angle, False)
        elif not self._rescuing and cue.action in SPEECH_ACTIONS:
            self._held[cue.track_id] = (cue, t, reason, angle)   # the gate may say it itself once it qualifies (rescue)
            if reason in ("repeat", "background") and self._stale_in_path(
                    trk, t, entry if entry is not None else self._own_entry(cue.track_id)):
                self._lost_due[cue.track_id] = t
                rng = self._smoothed_range(trk)
                if rng is not None:
                    self._held_ref[cue.track_id] = rng
            else:
                self._held_ref.pop(cue.track_id, None)
        return None

    def _flush_pending(self, visible: Dict[int, Track], t: float, ch) -> List[Cue]:
        """Cues held for confirmation: judged again (distance and direction refreshed) once the track is confirmed,
        within pending_s; a track that flickered out keeps its cue for flicker_keep_s, for its own return or a successor
        (id switch), whose matches count towards confirmation. A flicker that never comes back is never spoken."""
        g = self.gate_cfg
        busy = bool(getattr(ch, "busy", False))
        for tid, (cue, t_held, angle0, lost) in sorted(self._pending.items(), key=lambda kv: -kv[1][0].utility):
            trk = visible.get(tid)
            lost = lost or trk is None
            if t - t_held > (g.flicker_keep_s if lost else g.pending_s) + 1e-9:
                del self._pending[tid]           # a flicker that did not come back: exactly what min_hits is for
                self.reasons["unconfirmed (dropped)"] += 1
                continue
            self._pending[tid] = (cue, t_held, angle0, lost)
            if trk is None or not self._ready(trk, cue):
                continue
            action = cue.action
            if action == "interrupt" and not busy:
                action = "speak"
            elif action == "speak" and busy:
                continue                         # wait for the channel (until the cue expires)
            del self._pending[tid]
            out = self._judge(replace(self._refreshed(cue, angle0, trk, t), action=action), trk, visible, t, deferred=True)
            if out is not None:
                return out
        return []

    def _held_keep(self, reason: str) -> float:
        g = self.gate_cfg
        # (a muted cue too: in read mode the object may start to approach long after the scheduler last offered it)
        return max(g.held_keep_s, g.repeat_keep_s) if reason in ("repeat", "critical_gap", "muted") else g.held_keep_s

    def _rescue(self, visible: Dict[int, Track], t: float, ch) -> List[Cue]:
        """Say a cue the gate held (or one it said earlier) itself when its object now qualifies for safety and the
        scheduler did not offer it again (module docstring: Rescue). At most one, the most urgent."""
        busy = bool(getattr(ch, "busy", False))
        best = None
        for tid, trk in visible.items():
            if not self._confirmed(trk) or (busy and getattr(ch, "current_track", None) == tid):
                continue                         # (never cut in on a message about the same object)
            h = self._held.get(tid)
            if h is not None and 1e-9 < t - h[1] <= self._held_keep(h[2]) + 1e-9:
                src, angle0 = h[0], h[3]
                allowed = set(_RESCUE_HELD)
            else:
                e = self._own_entry(tid)
                if e is None or e.cue is None:
                    continue
                src, angle0 = e.cue, e.angle
                # a said object the scheduler does not offer again: what is new since only because the gate released it
                # (approach, then its critical cue) or because without depth nothing is ever critical (near)
                allowed = set(_RESCUE_SAID) | ({"critical"} if tid in self._approach else set())
                rng = self._range_now(trk)
                if rng is None or not self._in_corridor(trk.label, trk.bbox, rng, True):
                    allowed.discard("near")      # said by the gate unasked only when the user walks straight into it
            cue = self._refreshed(src, angle0, trk, t)
            # the first rule among those allowed that holds (a sector change or a reminder must not hide `closing`)
            ref = self._held_ref.get(tid) if h is not None and src is h[0] else None
            self._closing_ref = ref
            try:
                ok, reason, _, _ = self._check(cue, trk, visible, t, allowed=allowed)   # (may release it as approaching)
            finally:
                self._closing_ref = None
            if not ok or self._muted(cue, trk, reason):
                continue
            key = (_RESCUE_PRIO[reason], float(cue.utility or 0.0))
            if best is None or key > best[0]:
                best = (key, cue, trk, reason, allowed, ref)
        if best is None:
            return []
        _, cue, trk, reason, allowed, ref = best
        if busy:
            # the channel plays the FIND swap that took this object's place (_goal_other): its warning cuts it, as the
            # scheduler's cue it replaced would have been playing instead (v4_swap_fast:975019: a goal person that started
            # to walk at the user as the scheduler named them again, before any range step showed)
            sw, last = self._swapped, self._said_last
            swap_cut = (sw is not None and last is not None and last[0] is sw[0] and trk.track_id == sw[1]
                        and getattr(ch, "current_track", None) == sw[0].track_id
                        and (reason in _APPROACH_REASONS or reason == "closer"))
            if reason not in ("critical", "approach", "near") and not swap_cut:
                return []
            if reason != "critical" and not swap_cut and self._playing_safety(ch, visible):
                return []                        # (a warning playing is not cut for a lesser one: it waits)
            if reason == "critical" and not swap_cut and self._playing_sooner(ch, visible, trk):
                return []                        # (find:400183: nor for a critical one about something reached later)
            cue = replace(cue, action="interrupt")
        else:
            cue = replace(cue, action="speak")
        self._rescuing, self._closing_ref = allowed, ref
        try:
            out = self._judge(cue, trk, visible, t, deferred=True)
        finally:
            self._rescuing, self._closing_ref = None, None
        if out:
            self.stats["rescued"] += 1
        return out or []

    def _playing_sooner(self, ch, visible: Dict[int, Track], trk: Track) -> bool:
        """Does the real channel play a message the gate let through about safety (_playing_safety) whose object the user
        reaches no later than this one (both measured TTCs)? The gate's own critical cue then waits for it (harness
        find:400183: a critical rescue about a sign the user would reach in 1.4 s cut the cue about the FIND-goal sign
        beside it, reached in 0.7 s, before its words named it)."""
        if not self._playing_safety(ch, visible) or not (self._measured(trk) and math.isfinite(trk.ttc_s)):
            return False
        tid = self._said_last[0].track_id
        s = self._seen.get(tid)
        p = visible.get(tid) or (s.trk if s is not None else None)
        return (p is not None and self._measured(p) and math.isfinite(p.ttc_s) and p.ttc_s <= trk.ttc_s + 1e-9
                and p.path_prob >= self.gate_cfg.critical_path)

    def _playing_safety(self, ch, visible: Dict[int, Track]) -> bool:
        """Does the real channel play a message the gate let through that is about safety (_is_safety: a vehicle, the FIND
        goal, an object in the walking path or corridor, a measured approach)? The gate's own approach or near message
        then waits for it (harness find:511113: a near rescue about one sign cut the scheduler's cue about the FIND goal
        sign after 1 s, before its words named it); only a critical one cuts it."""
        last = self._said_last
        if not getattr(ch, "busy", False) or last is None or getattr(ch, "current_track", None) != last[0].track_id:
            return False
        tid = last[0].track_id
        s = self._seen.get(tid)
        trk = visible.get(tid) or (s.trk if s is not None else None)   # (out of view a moment: as last seen)
        return self._is_safety(last[0], trk, "")

    def _stale_in_path(self, trk: Optional[Track], t: float, e: Optional[_Said]) -> bool:
        """An object (not a sign) in the walking path or corridor between operator_max_m and path_m (a vehicle at any
        range), seen steadily (matched in half the frames since its track began, release_span_s ago or more), not
        said for hold_moving_s (or
        never: entry e): a cue about it held now is said by the gate if the object leaves view at once (_say_lost), and
        the rescue judges `closing` from where it was then (_held_ref)."""
        g = self.gate_cfg
        if trk is None or trk.kind != "object":
            return False
        if not self._in_path(trk, math.inf if self._is_vehicle(self._identity_of(trk)) else g.path_m):
            return False                         # (a bicycle covers path_m in two seconds)
        rng = self._range_now(trk)
        if rng is None or rng < g.operator_max_m:
            return False
        if trk.hits < 0.5 * (trk.age + 1) or trk.last_seen - trk.first_seen < g.release_span_s - 1e-9:
            return False                         # a flicker (a hand over the desk user): going out of view says nothing
        return e is None or t - e.t >= g.hold_moving_s - 1e-9

    def _say_lost(self, visible: Dict[int, Track], t: float, ch) -> List[Cue]:
        """An object in the walking path the user last heard of hold_moving_s ago (or never), whose cue the gate held as
        nothing new (a repeat, background) at the last decision, and that left view right after: the gate cannot watch
        it come closer, so it says the held cue (as the scheduler chose it) on a free channel, within 1 s. An approach
        that starts as the object goes out of view (an occlusion, a lighting dropout) is otherwise first seen when it is
        back, a step away; a static object in the path costs at most one reminder per hold_moving_s, and only when it
        vanishes right after the scheduler offered it."""
        for tid in [k for k, th in self._lost_due.items() if t - th > 1.0 + 1e-9 or k not in self._held]:
            del self._lost_due[tid]
        if getattr(ch, "busy", False):
            return []
        lost = []
        for tid, th in self._lost_due.items():
            s = self._seen.get(tid)
            if tid in visible or t - th < 1e-9 or s is None:
                continue
            if any(self._identity_of(o) == s.identity and bbox_iou(o.bbox, s.trk.bbox) >= 0.3 for o in visible.values()):
                continue                         # in view under another id (a switch the succession did not link)
            lost.append((self._held[tid][0], tid))
        if not lost:
            return []
        cue, tid = max(lost, key=lambda x: float(x[0].utility or 0.0))
        del self._lost_due[tid]
        self._held.pop(tid, None)
        s = self._seen.get(tid)
        trk = s.trk if s is not None else None
        out = replace(cue, t=t, action="speak")
        e = self._own_entry(tid)
        ago = f"said {t - e.t:.1f} s ago" if e is not None else "never said"
        info = f"{cue.label} out of view since the cue was held, {ago}"
        self.stats["passed"] += 1
        self.stats["rescued"] += 1
        self.reasons["lost (deferred)"] += 1
        self.last_reason = f"pass lost (deferred): {info}"
        self.last_verdict = {"t": t, "track_id": tid, "label": cue.label, "action": out.action, "passed": True,
                             "reason": "lost", "info": info, "deferred": True}
        self._said_last = (out, t, True)
        self._commit(out, trk, e, t, "lost")
        return [out]

    # ---- the decision ---------------------------------------------------------------------------------------
    def _check(self, cue: Cue, trk: Optional[Track], visible: Dict[int, Track], t: float,
               allowed: Optional[Set[str]] = None):
        """(passes, reason, memory entry, info): the first pass rule that holds (module docstring), or why it is held.
        `allowed` (rescue): the first rule *among these* that holds, so a reason the gate does not say a cue itself for
        (e.g. a reminder) never hides one it does (closing, near)."""
        g = self.gate_cfg
        ident = self._identity_of(trk, cue)
        angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
        sec = sector_of(angle, g.sector_deg)
        desc = f"{ident[1] or cue.label} {sec}"
        # background / operator bookkeeping first (it follows the user's box and releases approaching objects), so
        # a track the gate holds as the user or as background is critical only when it measurably closes in
        silenced = self._background_reason(trk, t, visible) if trk is not None else None
        if silenced in ("warmup", "background") and self._is_goal(cue, trk):
            silenced = None                      # a FIND goal is never background (the operator, the user, stays held)
        if (trk is not None and self._critical(trk)
                and (silenced is None or (-trk.rel_speed_mps >= g.approach_min_mps
                                          and (silenced != "operator" or self._past_lean(trk, self._operator))))):
            e = self._lookup(ident, angle, cue.track_id, trk, visible, t)
            if cue.action == "haptic":
                last = self._crit_haptic.get((ident, sec))
            else:
                last = (e.crit_t, e.crit_ttc, e.crit_dist) if e is not None and e.crit_t is not None else None
            held = self._critical_gap(last, trk, t)
            if held:
                return False, "critical_gap", e, f"{desc}, ttc {trk.ttc_s:.1f} s, {held}"
            if allowed is None or "critical" in allowed:
                return True, "critical", e, f"{desc}, ttc {trk.ttc_s:.1f} s"
        if trk is not None and not self._ready(trk, cue):
            return False, "unconfirmed", None, f"{desc}, {trk.hits} hits"
        if silenced:
            return False, silenced, None, desc
        if cue.action == "haptic":
            last = self._haptic.get((ident, sec))
            if last is not None and t - last < g.haptic_gap_s:
                return False, "haptic_gap", None, f"{desc}, buzzed {t - last:.1f} s ago"
            return True, "haptic", None, desc
        e = self._lookup(ident, angle, cue.track_id, trk, visible, t)
        for reason, info in self._passes(cue, trk, e, angle, desc, t, asked=allowed is None):
            if allowed is None or reason in allowed:
                return True, reason, e, info
        if e is None:
            return False, "repeat", None, desc
        return False, "repeat", e, f"{desc}, said {t - e.t:.1f} s ago (hold {self._hold_s(cue, trk, e):.0f} s)"

    def _passes(self, cue: Cue, trk: Optional[Track], e: Optional[_Said], angle: float, desc: str, t: float,
                asked: bool = True):
        """The pass rules after the holds, in order (lazily: the normal path stops at the first). `asked`: the scheduler
        offers the cue now (not the rescue)."""
        g = self.gate_cfg
        released = self._approach.get(cue.track_id)
        if released is not None and t - released <= g.approach_keep_s:
            yield "approach", f"{desc}, closing in (released {t - released:.1f} s ago)"
        if e is None:
            yield "new", desc
            return
        back = self._reacquired(cue, trk, t)
        if back:
            yield "reacquired", f"{desc}, {back}"
        ago = t - e.t
        gap = ago >= g.min_gap_s
        info = f"{desc}, said {ago:.1f} s ago"
        if e.track_id == cue.track_id and not in_sector(angle, e.sector, g.sector_deg, g.sector_hyst_deg):
            if not gap:
                return                           # (repeat)
            yield "sector", f"{desc}, was {e.sector}"
        d_now = trk.distance_m if trk is not None else cue.distance_m
        if gap and materially_closer(e.dist, d_now, g.closer_m, g.closer_frac):
            yield "closer", f"{desc}, {e.dist:.1f} -> {d_now:.1f} m"
        if gap and trk is not None and self._near_without_depth(trk):
            yield "near", f"{desc}, {self._range_now(trk):.1f} m, closing (no measured depth), said {ago:.1f} s ago"
        # (min_gap_s is longer than a walk takes to cover a near object: without measured depth, where nothing is ever
        # critical, a near one's closing step offered by the scheduler is news within it, harness find:500033; and so is
        # the closing step of one that comes at the user at release_fast_mps or more now, a moment after it was said
        # standing: bgref:400366, no depth, a person said at 5.3 m who then walked at the user at 1.7 m/s)
        quick = asked and trk is not None and (
            (not self._measured(trk) and self._nearby(trk)) or self._short_closing(trk) >= g.release_fast_mps)
        if (gap or quick) and trk is not None and self._closing_in_path(trk, e.rng, asked):
            yield "closing", f"{desc}, {self._range_now(trk):.1f} m and closing (said at {e.rng or 0.0:.1f} m)"
        moving = trk is not None and self._moving(trk)
        hold = self._hold_s(cue, trk, e)
        if ago >= hold:
            yield "reminder", f"{info} (hold {hold:.0f} s)"
        if gap and cue.action == "interrupt" and trk is not None and (moving or self._approaching(trk, e.dist)):
            yield "interrupt", info

    def _hold_s(self, cue: Cue, trk: Optional[Track], e: Optional[_Said]) -> float:
        """How long a said object is not repeated: hold_goal_s for a FIND goal, hold_moving_s for an object that moves
        or that is in the walking path and came release_m nearer since it was said (_path_nearer: the user may be walking
        at it, and without measured depth, or in a flicker, nothing else may tell the gate before it is a step or two
        away), else hold_static_s."""
        g = self.gate_cfg
        if self._is_goal(cue, trk):
            return g.hold_goal_s
        if trk is not None and (self._moving(trk) or (e is not None and self._path_nearer(trk, e.rng))):
            return g.hold_moving_s
        return g.hold_static_s

    def _path_nearer(self, trk: Track, ref: Optional[float]) -> bool:
        """In the walking path or corridor between operator_max_m and path_m, and its smoothed range at least release_m
        nearer than `ref` (where it was when last said). Nearer than operator_max_m it is the desk user, or a step away,
        where the critical and near rules speak; a static object's smoothed range does not drift by release_m."""
        g = self.gate_cfg
        d = self._smoothed_range(trk)
        if ref is None or d is None or d < g.operator_max_m or not self._in_path(trk, g.path_m):
            return False
        return ref - d >= g.release_m - 1e-9

    def _reacquired(self, cue: Cue, trk: Optional[Track], t: float) -> str:
        """reacquired: back in view (a successor track, within its first release_span_s) after more than lost_s out of
        view, in the walking path or corridor and max(reacquired_m, reacquired_frac x range) nearer than when it was last
        seen: it came closer unseen (a lighting dropout as it starts to approach). A static object back where it was is
        nothing new (every dropout of a person said at 2.5 m would say it again). Once per return; '' if not."""
        g = self.gate_cfg
        s = self._seen.get(cue.track_id)
        if trk is None or s is None or s.trk is not trk or s.gap <= g.lost_s or s.back_rng is None:
            return ""
        if t - s.t0 > g.release_span_s + 1e-9:
            return ""
        rng = self._range_now(trk)
        if rng is None or s.back_rng - rng < max(g.reacquired_m, g.reacquired_frac * s.back_rng) - 1e-9:
            return ""
        if not (trk.path_prob >= g.critical_path or self._in_corridor(trk.label, trk.bbox, rng)):
            return ""
        return f"back after {s.gap:.1f} s out of view, {s.back_rng:.1f} -> {rng:.1f} m"

    def _confirmed(self, trk: Track) -> bool:
        """min_hits tracker matches, or 2 spanning confirm_age_s, counting the matches of the tracks it took over from
        (module docstring: Identity): an object seen in two short sightings at the same place is no flicker."""
        g = self.gate_cfg
        s = self._seen.get(trk.track_id)
        carried, first = (s.carried, min(s.first, trk.first_seen)) if s is not None and s.trk is trk else (0, trk.first_seen)
        hits = trk.hits + carried
        return hits >= g.min_hits or (hits >= 2 and trk.last_seen - first >= g.confirm_age_s)

    def _ready(self, trk: Track, cue: Optional[Cue] = None) -> bool:
        """Confirmed, or in the walking path or corridor, or the FIND goal: that cannot wait a decision step (its TTC
        is not measured yet on a new track, so it is not critical either; in lighting dropouts the sightings that would
        confirm it may all fall between two decisions, which the gate never sees, and the next one may come a second
        later and 1.5 m nearer). The tracker's own n_init still holds. Except, during the warm-up with operator_filter,
        a person who may be the desk user (large, centred, within operator_max_m): the warm-up needs a few samples to
        tell the user (held) from someone who stepped up."""
        if self._confirmed(trk):
            return True
        ident = self._identity_of(trk)
        rng = self._range_now(trk)
        in_path = rng is not None and (trk.path_prob >= self.gate_cfg.critical_path
                                       or self._in_corridor(trk.label, trk.bbox, rng))
        if not (in_path or (cue is not None and self._is_goal(cue, trk))):
            return False
        g = self.gate_cfg
        if g.operator_filter and not self._warm_done and ident == ("object", "person"):
            h, cx = trk.bbox[3] - trk.bbox[1], bbox_center(trk.bbox)[0]
            if h >= 0.8 * g.operator_min_height and abs(cx - 0.5) <= g.operator_center_tol + 0.1:
                if trk.depth_conf >= 0.4 and trk.distance_m is not None:
                    dist = trk.distance_m
                else:                            # (as _operator_like: the desk user's box is cut off at the bottom)
                    rs = [r for r in self._box_ranges(trk.bbox, "person") if r is not None]
                    dist = min(rs) if rs else 0.0
                if dist <= g.operator_max_m:
                    return False
        return True

    def _critical(self, trk: Track) -> bool:
        g = self.gate_cfg
        return self._measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.critical_ttc_s and trk.path_prob >= g.critical_path

    def _muted(self, cue: Cue, trk: Optional[Track], reason: str) -> bool:
        """Read mode (objects_muted): would the TaskAdapter's read-mode filter drop this cue? It keeps text cues,
        haptics, interrupts, severity >= muted_severity and anything approaching (the cue's TTC, or the track's measured
        one, under approach_ttc_s); the gate also keeps what it passes as an approach (critical, approach, closing, near,
        reacquired), a vehicle, the FIND goal and an object whose measured range closes at release_mps or more (as the
        adapter's filter_cues): read mode mutes the static scene around a user who stands and reads, never an object
        coming at them. Judged before the cue is remembered as said."""
        g = self.gate_cfg
        if not g.objects_muted or cue.kind == "text" or cue.action != "speak" or reason in _APPROACH_REASONS:
            return False
        if (cue.severity or 0.0) >= g.muted_severity or self._is_goal(cue, trk):
            return False
        if self._is_vehicle(self._identity_of(trk, cue)):
            return False
        if trk is not None and self._measured(trk) and -trk.rel_speed_mps >= g.release_mps:
            return False
        ttcs = [cue.ttc_s] + ([trk.ttc_s] if trk is not None and self._measured(trk) else [])
        return not any(x is not None and math.isfinite(x) and x < g.approach_ttc_s for x in ttcs)

    def _critical_gap(self, last: Optional[Tuple[float, float, Optional[float]]], trk: Track, t: float) -> str:
        """'' when a critical cue may pass, else why it is held: one critical cue per object every critical_gap_s,
        unless the hazard worsened since the last one (TTC down by critical_worse_ttc_s or distance by critical_worse_m)."""
        g = self.gate_cfg
        if last is None or t - last[0] >= g.critical_gap_s:
            return ""
        t0, ttc0, d0 = last
        if math.isfinite(ttc0) and trk.ttc_s <= ttc0 - g.critical_worse_ttc_s:
            return ""
        if d0 is not None and trk.distance_m is not None and trk.distance_m <= d0 - g.critical_worse_m:
            return ""
        return f"critical {t - t0:.1f} s ago and not worse"

    # ---- range and closing evidence -------------------------------------------------------------------------
    def _track_sources(self, tracks: Sequence[Track], t: float) -> None:
        """Which range source each track in view has (measured depth or the box) and since when: the first decision it
        was seen with it (the change came at most one decision step before; RATE_SETTLE_S covers the tracker's depth EMA
        blending the two for a few samples after it). A track first seen now has one source all along. Kept by track id
        (a copy of the track, or a new tracker reusing the id, keeps it)."""
        for trk in tracks:
            if trk.remembered or trk.misses > 0:
                continue
            d = trk.depth_conf >= 0.4
            s = self._src.get(trk.track_id)
            if s is None:
                self._src[trk.track_id] = [d, -math.inf, trk, t]
            elif s[0] != d:
                self._src[trk.track_id] = [d, t, trk, t]
            else:
                s[2], s[3] = trk, t
        for tid in [k for k, s in self._src.items() if t - s[3] > 10.0]:
            del self._src[tid]

    def _since(self, trk: Track) -> float:
        """Since when the track has its range source now (the first decision seen with it; -inf: all along)."""
        s = self._src.get(trk.track_id)
        return s[1] if s is not None else -math.inf

    def _rate_ok(self, trk: Track) -> bool:
        """Is the tracker's range rate (and TTC) from one range source? Its speed window (geometry.DynamicsConfig
        history_s) starts RATE_SETTLE_S after the decision that first saw the source it has now."""
        since = self._since(trk)
        return since == -math.inf or (bool(trk.dist_history) and trk.dist_history[0][0] > since + RATE_SETTLE_S - 1e-9)

    def _measured(self, trk: Track, min_span_s: float = 1.0) -> bool:
        """measured (module level: depth_conf >= 0.4, 4 range samples over min_span_s), and every sample the tracker's
        range rate comes from is a depth sample (_rate_ok)."""
        return measured(trk, min_span_s) and self._rate_ok(trk)

    def _box_ranges(self, bbox: BBox, label: str) -> Tuple[Optional[float], Optional[float]]:
        """Range from the box height and from the box width (typical object size), None for a dimension the frame
        edge cuts off."""
        H, W = TYPICAL_SIZE_M.get(label, DEFAULT_SIZE_M)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rh = self.cam.fy * H / h if h > 1e-3 and not _cut_tb(bbox) else None
        rw = self.cam.fx * W / w if w > 1e-3 and not _cut_lr(bbox) else None
        return rh, rw

    def _box_range(self, bbox: BBox, label: str) -> Optional[float]:
        """Range from one box: its width when the frame edge does not cut it, else its height (_range_now)."""
        rh, rw = self._box_ranges(bbox, label)
        return rw if rw is not None else rh

    def _range_now(self, trk: Track) -> Optional[float]:
        """Measured depth when there is any (depth_conf >= 0.4), else the box: its width when the frame edge does not
        cut it (a floor obstacle a step away is cut off at the bottom; its box-height range grows as it comes closer),
        else its height, else the tracker's estimate."""
        if trk.depth_conf >= 0.4 and trk.distance_m is not None:
            return trk.distance_m
        rh, rw = self._box_ranges(trk.bbox, trk.label)
        r = rw if rw is not None else rh
        return r if r is not None else trk.distance_m

    def _in_corridor(self, label: str, bbox: BBox, dist: float, strict: bool = False) -> bool:
        """Does the object overlap the walking corridor (`strict`: is its box centre inside it, straight ahead)?"""
        _, width = TYPICAL_SIZE_M.get(label, DEFAULT_SIZE_M)
        half = self.gate_cfg.corridor_half_m + (0.0 if strict else width / 2)
        return abs(self.cam.lateral_from_u(bbox_center(bbox)[0], dist)) <= half

    def _box_closing(self, trk: Track, rng: float) -> float:
        """Closing speed (m/s) the box growth shows over the last ~0.5 s and ~1 s, from the sides the frame edge does
        not cut (box size ~ 1 / range); 0 unless the box grew by at least nodepth_grow. Each end is the median of the
        boxes within _BOX_WIN_S of it (3 frames at 10 fps; one box at a slower rate), so a jittering box of a static
        sign or chair does not look like it closes in."""
        g = self.gate_cfg
        hist = list(trk.bbox_history)
        if len(hist) < 2:
            return 0.0
        t1 = hist[-1][0]
        now = [b for tt, b in hist if tt >= t1 - _BOX_WIN_S - 1e-9]
        best = 0.0
        for lag in (0.5, 1.0):
            prev = [(tt, b) for tt, b in hist if tt <= t1 - lag + 0.05]
            if not prev:
                continue
            t0 = prev[-1][0]
            dt = t1 - t0
            if dt < 0.3:
                continue
            then = [b for tt, b in prev if tt >= t0 - _BOX_WIN_S - 1e-9]
            logs = []
            if not any(_cut_lr(b) for b in now + then):
                w0, w1 = _med([b[2] - b[0] for b in then]), _med([b[2] - b[0] for b in now])
                if w0 > 1e-3 and w1 > 1e-3:
                    logs.append(math.log(w1 / w0))
            if not any(_cut_tb(b) for b in now + then):
                h0, h1 = _med([b[3] - b[1] for b in then]), _med([b[3] - b[1] for b in now])
                if h0 > 1e-3 and h1 > 1e-3:
                    logs.append(math.log(h1 / h0))
            if not logs:
                continue
            grow = math.exp(sum(logs) / len(logs))
            if grow >= 1.0 + g.nodepth_grow:
                best = max(best, rng * (grow - 1.0) / dt)
        return best

    def _nearby(self, trk: Track) -> bool:
        """Near (the near rule's test, with or without depth): range under nodepth_near_m, or the box bottom in the lowest
        nodepth_bottom_frac of the frame (a floor object a step or two away; its box range may be off: cut by the frame)."""
        g = self.gate_cfg
        rng = self._range_now(trk)
        return (rng is not None and rng < g.nodepth_near_m) or trk.bbox[3] >= 1.0 - g.nodepth_bottom_frac

    def _near_without_depth(self, trk: Track) -> bool:
        """No measured depth (TTC is assumed, so nothing is critical): an object in the walking path or corridor that
        is near (range < nodepth_near_m, or its box bottom in the lowest nodepth_bottom_frac of the frame) and closing
        at >= nodepth_approach_mps, by the range rate over >= 1 s or by the box growth (sides the frame does not cut)."""
        g = self.gate_cfg
        if self._measured(trk):
            return False
        rng = self._range_now(trk)
        if rng is None:
            return False
        if not (trk.path_prob >= g.critical_path or self._in_corridor(trk.label, trk.bbox, rng)):
            return False
        if not (rng < g.nodepth_near_m or trk.bbox[3] >= 1.0 - g.nodepth_bottom_frac):
            return False
        if (trk.distance_m is not None and history_span_ok(trk) and self._rate_ok(trk) and trk.distance_m < g.nodepth_near_m
                and trk.path_prob >= g.critical_path and -trk.rel_speed_mps >= g.nodepth_approach_mps):
            return True                          # (not while the range rate spans a change of range source)
        return self._box_closing(trk, rng) >= g.nodepth_approach_mps

    def _smoothed_depth(self, trk: Track) -> Optional[float]:
        """Median of the range samples over the last release_smooth_s (one noisy frame does not count), of its range
        source now (settled: RATE_SETTLE_S after a change), else its latest sample."""
        h = trk.dist_history
        if not h:
            return trk.distance_m
        t1, lo = h[-1][0], self._since(trk) + RATE_SETTLE_S - 1e-9
        zs = [z for tt, z, _ in h if tt >= t1 - self.gate_cfg.release_smooth_s - 1e-9 and tt > lo]
        return _med(zs) if zs else h[-1][1]

    def _smoothed_range(self, trk: Track) -> Optional[float]:
        """The range over the last release_smooth_s: the median of the measured depth samples, else of the box ranges."""
        if measured(trk):
            return self._smoothed_depth(trk)
        hist = list(trk.bbox_history)
        t1 = hist[-1][0] if hist else 0.0
        rs = [r for r in (self._box_range(b, trk.label) for tt, b in hist if tt >= t1 - self.gate_cfg.release_smooth_s - 1e-9)
              if r is not None]
        return _med(rs) if rs else self._range_now(trk)

    def _closing_in_path(self, trk: Track, said_rng: Optional[float], asked: bool = False) -> bool:
        """closing: an object in the walking path or corridor, nearer than path_m (offered by the scheduler: at any
        range), at least max(release_m, release_frac x range) nearer than when it was last said (smoothed ranges) and,
        unless the scheduler offers it now (`asked`), closing now (>= release_mps; or, by its latest sample, nearer by that much while closing at
        >= release_fast_mps over the last 0.5 s): the gate says nothing unasked on a range step alone. Measured depth when
        there is any, else the box (sides the frame does not cut). Finer than `closer` (1 m) for what is a few steps
        away: a person or a car that starts coming at a user who stands still is said again before its TTC turns
        critical, and so is a chair the user walks at slowly (0.2 m/s: no closing speed to measure)."""
        g = self.gate_cfg
        if said_rng is None or trk.distance_m is None:
            return False
        rng = self._range_now(trk)
        if rng is None or (rng >= g.path_m and not asked):
            return False                         # (offered by the scheduler: at any range, harness find:2913)
        if not (trk.path_prob >= g.critical_path or self._in_corridor(trk.label, trk.bbox, rng)):
            return False
        d = self._smoothed_range(trk)
        d_last = trk.dist_history[-1][1] if measured(trk) else rng
        # (a cue held while the user last heard of the object hold_moving_s ago: from where it was when the scheduler
        # offered it, if that is farther; a stale mention says nothing about an approach that starts now. A track that
        # took over from another one farther than that was said: from where it came back, bgref:906, a dog stepping in
        # 0.6 m behind where the said one stood, which then walked at the user)
        ref = said_rng if self._closing_ref is None else max(said_rng, self._closing_ref)
        s = self._seen.get(trk.track_id)
        if s is not None and s.trk is trk and s.pred and s.r_in is not None:
            ref = max(ref, s.r_in)
        thr = max(g.release_m, g.release_frac * ref)
        if asked and d is not None and ref - d >= thr - 1e-9:
            return True
        fast = self._short_closing(trk)
        if ref - d >= thr - 1e-9 and (fast >= g.release_mps or (self._measured(trk) and -trk.rel_speed_mps >= g.release_mps)):
            return True
        return ref - d_last >= thr - 1e-9 and fast >= g.release_fast_mps

    def _moving(self, trk: Track) -> bool:
        """Lateral speed above moving_speed_mps (+0.05 m/s per metre: it is noisier at range), or a measured range
        rate above moving_speed_mps. A TTC under 6 s from a slower drift (0.12-0.3 m/s: depth jitter, the user
        leaning toward the laptop) does not count, or the user at a desk would be 'moving' and repeated every 8 s."""
        g = self.gate_cfg
        if len(trk.dist_history) < 4:
            return False
        z = trk.distance_m if trk.distance_m is not None else 3.0
        if abs(trk.lateral_speed_mps) > g.moving_speed_mps + 0.05 * z and self._rate_ok(trk):
            return True                          # (the lateral offset scales with the range: a source change moves it)
        return self._measured(trk) and abs(trk.rel_speed_mps) > g.moving_speed_mps

    def _approaching(self, trk: Track, dist_ref: Optional[float]) -> bool:
        g = self.gate_cfg
        if materially_closer(dist_ref, trk.distance_m, g.closer_m, g.closer_frac):
            return True
        return (self._measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.approach_ttc_s
                and -trk.rel_speed_mps >= g.approach_min_mps)

    def _goal_names(self, trk: Track) -> bool:
        """Does the FIND goal in force name this track (its label, alias or sign text)?"""
        tags = [k for k in (normalize_text(g) for g in self._current_goal_tags()) if k]
        if not tags:
            return False
        label = (trk.label or "").lower()
        keys = {label, LABEL_ALIAS.get(label, label)}
        txt = normalize_text(trk.text)
        if txt:
            keys.add(txt)
        return any(self._names(gk, k) for gk in tags for k in keys)

    def _is_goal(self, cue: Cue, trk: Optional[Track]) -> bool:
        if "goal" in (cue.slots or ()):
            return True
        lf = getattr(self._inner, "last_features", None)
        if isinstance(lf, dict):
            f = lf.get(cue.track_id)
            if f is not None and getattr(f, "goal", 0.0) > 0:
                return True
        tags = [k for k in (normalize_text(g) for g in self._current_goal_tags()) if k]
        if not tags:
            return False
        label = (trk.label if trk is not None else cue.label or "").lower()
        keys = {label, LABEL_ALIAS.get(label, label)}
        txt = normalize_text(trk.text if trk is not None and trk.text else cue.text)
        if txt:
            keys.add(txt)
        return any(self._names(gk, k) for gk in tags for k in keys)

    # ---- identity memory ------------------------------------------------------------------------------------
    def _identity_of(self, trk: Optional[Track], cue: Optional[Cue] = None) -> Tuple[str, str]:
        kind = trk.kind if trk is not None else (cue.kind if cue is not None else "object")
        if kind == "text":
            raw = (trk.text if trk is not None and trk.text else None) or (cue.text if cue is not None else None)
            txt = normalize_text(raw)
            if txt and ("text", txt) not in self._mem:
                for ident in self._mem:            # OCR variants of the same sign ("ROOM 204" / "R0OM 2O4")
                    if ident[0] == "text" and same_sign(ident[1], txt):
                        return ident
            return ("text", txt)
        label = (trk.label if trk is not None else cue.label if cue is not None else "object").lower()
        return ("object", LABEL_ALIAS.get(label, label))

    def _lookup(self, ident, angle: float, track_id: int, trk: Optional[Track], visible: Dict[int, Track], t: float) -> Optional[_Said]:
        """The memory entry this cue is about: the entry of its own track (or of a track id it was about before: the
        tracker re-acquired an older id), else (tracker id switch) the most recent entry of the same identity in the same
        sector whose track this one took over from (_succeed: where and at the range it vanished, shortly before) or
        that it duplicated (a second box over it at the same range). The matched entry adopts the new track id, so a
        chain of id switches stays one object. Nothing else inherits: another object of the label that turns up in
        the sector (a car from 6 m after the one said at 3 m drove off to the side) is new."""
        g = self.gate_cfg
        entries = self._mem.get(ident)
        if not entries:
            return None
        own = [e for e in entries if self._owns(e, track_id, trk)]
        if own:
            return max(own, key=lambda e: e.t)
        s = self._seen.get(track_id)
        pred = s.pred if s is not None and trk is not None and s.trk is trk else ()
        cands = []
        for e in entries:
            if not in_sector(angle, e.sector, g.sector_deg, g.sector_hyst_deg):
                continue
            if trk is not None and track_id in e.apart:
                continue                     # was in view together with the old track: a second object, not a switch
            other = visible.get(e.track_id)
            if other is not None and trk is not None and other is not trk:
                # the remembered object is still in view as a separate track: this cue is about a second object,
                # unless the two boxes are the same thing (a duplicate detection: overlapping, at the same range)
                if not self._dup_box(trk, other):
                    continue
            elif other is None and trk is not None:
                if e.track_id not in pred and not (track_id in e.dups and trk.first_seen - e.seen <= g.switch_gap_s):
                    continue                 # not its successor: a different object
                if (e.track_id in pred and e.rng is not None and s.back_rng is not None
                        and s.back_rng - e.rng >= max(g.closer_m, g.closer_frac * e.rng) - 1e-9):
                    continue                 # its object had moved away since it was said: one walked off and vanished,
                                             # another comes from there (new to the user)
            cands.append(e)
        if not cands:
            return None
        e = max(cands, key=lambda e: e.t)
        if visible.get(e.track_id) is None:
            e.track_id, e.seen = track_id, t
            e.tids.add(track_id)
        return e

    def _dup_box(self, a: Track, b: Track) -> bool:
        """Two tracks of one label in view together are one object detected twice only if their boxes overlap by
        dup_iou and they are at the same range; two people standing side by side overlap less (a second person walking
        up next to one already said is new). With measured depth on both the range must agree within max(release_m,
        release_frac x range) (one object's pixels have one depth: a car 0.8 m in front of another, its box over the
        first's, is a second car); box ranges are coarser (the `closer` step)."""
        g = self.gate_cfg
        if bbox_iou(a.bbox, b.bbox) < g.dup_iou:
            return False
        if a.depth_conf >= 0.4 and b.depth_conf >= 0.4 and a.distance_m is not None and b.distance_m is not None:
            return abs(a.distance_m - b.distance_m) < max(g.release_m, g.release_frac * min(a.distance_m, b.distance_m)) - 1e-9
        return self._same_range(self._range_now(a), self._range_now(b))

    def _same_range(self, r_new: Optional[float], r_old: Optional[float], nearer_ok: bool = False,
                    span_s: float = 0.0, ident: Optional[Tuple[str, str]] = None) -> bool:
        """Two range readings of one object: not materially nearer or farther (max(closer_m, closer_frac x range), as the
        `closer` rule); an unknown range does not tell them apart. `nearer_ok` (a track that takes over after a dropout,
        the readings span_s apart): nearer by as much as the object could have come closer unseen (its held cue then
        follows it: _MAX_SPEED of its identity plus the user's walk, at least that step); but half the step farther is
        another object (a person said at 3.8 m leaves; one 4.4 m away walking up where they stood is new: treating a
        farther object as new costs a repeat, inheriting a nearer one's memory a late warning), and so is one nearer than
        anything could have come (a chair 2 m away where one 6 m away vanished half a second before)."""
        g = self.gate_cfg
        if r_new is None or r_old is None:
            return True
        if nearer_ok:
            step = max(g.closer_m, g.closer_frac * r_old)
            if r_new - r_old >= 0.5 * step - 1e-9:
                return False
            v = _MAX_SPEED.get(ident[1] if ident is not None else "", 0.0) + _WALK_MPS
            return r_old - r_new <= max(step, v * max(0.0, span_s)) + 1e-9
        return abs(r_new - r_old) < max(g.closer_m, g.closer_frac * min(r_new, r_old)) - 1e-9

    @staticmethod
    def _owns(e: _Said, track_id: int, trk: Optional[Track]) -> bool:
        """Is the entry about this track: its current track, or one it was about before that the tracker re-acquired
        (first seen before the entry's object was last in view: not a reused id after a tracker reset)?"""
        return e.track_id == track_id or (track_id in e.tids and (trk is None or trk.first_seen <= e.seen + 1e-9))

    def _own_entry(self, track_id: int) -> Optional[_Said]:
        """The latest memory entry of this track, else of the tracks it took over from (module docstring: Identity)."""
        s = self._seen.get(track_id)
        trk = s.trk if s is not None else None
        for tid in (track_id,) + (s.pred if s is not None else ()):
            own = [e for es in self._mem.values() for e in es    # (its predecessors were this object's tracks)
                   if (self._owns(e, tid, trk) if tid == track_id else e.track_id == tid or tid in e.tids)]
            if own:
                return max(own, key=lambda e: e.t)
        return None

    def _succeed(self, visible: Dict[int, Track], t: float) -> None:
        """Tracker id switches (module docstring: Identity). A track in view for the first time takes over from a track
        of the same identity that vanished within switch_gap_s, in the same sector or within 2 x sector_hyst_deg, not
        materially farther than it was last seen (_same_range), and whose last match came before its first one (both
        matched in one frame: two objects, or one box twice): its cue held for confirmation, its held cue, its approach
        release and its matches (its memory entries: _own_entry, and _lookup adopts them when the new track is judged)."""
        g = self.gate_cfg
        for tid, trk in visible.items():
            if tid in self._seen and self._seen[tid].trk is not trk:
                del self._seen[tid]              # a new tracker reusing ids (the gate outlived a tracker reset)
        for trk in sorted((x for tid, x in visible.items() if tid not in self._seen), key=lambda x: (x.first_seen, x.track_id)):
            ident = self._identity_of(trk)
            angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0])
            best = None
            for tid, s in self._seen.items():
                if tid in visible or s.identity != ident or t - s.t > g.switch_gap_s + 1e-9:
                    continue
                if trk.first_seen <= s.trk.last_seen + 1e-9:
                    continue
                if not (in_sector(angle, sector_of(s.angle, g.sector_deg), g.sector_deg, g.sector_hyst_deg)
                        or abs(angle - s.angle) <= 2 * g.sector_hyst_deg):
                    continue
                if not self._same_range(self._range_now(trk), self._range_now(s.trk), nearer_ok=True,
                                        span_s=t - s.trk.last_seen, ident=ident):
                    continue                     # another object of the label: farther, or nearer than it could have come
                key = (s.t, -abs(angle - s.angle))
                if best is None or key > best[0]:
                    best = (key, tid, s)
            if best is None:
                hist = trk.bbox_history
                b0 = hist[0][1] if hist else trk.bbox
                self._seen[trk.track_id] = _Seen(ident, angle, t, trk, 0, trk.first_seen, t, c0=bbox_center(b0), b0=b0,
                                                 r0=self._range_now(trk), r_in=self._range_now(trk))
                continue
            _, old, s = best
            del self._seen[old]                  # one successor per track
            self._seen[trk.track_id] = _Seen(ident, angle, t, trk, s.carried + s.trk.hits, min(s.first, s.trk.first_seen), t,
                                             trk.first_seen - s.trk.last_seen, self._range_now(s.trk), ((old,) + s.pred)[:8],
                                             s.c0, s.moved, s.b0, s.r0, self._range_now(trk))
            new = trk.track_id
            if old in self._pending and new not in self._pending:
                cue, t_held, a0, _ = self._pending.pop(old)
                self._pending[new] = (cue, t_held, a0, True)
            if old in self._held and new not in self._held:
                self._held[new] = self._held.pop(old)
                if old in self._held_ref:
                    self._held_ref[new] = self._held_ref.pop(old)
            self._lost_due.pop(old, None)        # (back in view: the rescue judges the held cue on what it sees)
            if old in self._owed:
                self._owed.add(new)
            if old in self._approach and new not in self._approach:
                self._approach[new] = self._approach.pop(old)
            for es in self._mem.values():        # (its memory entries are adopted when it is next judged: _lookup)
                for e in es:
                    if old in e.apart:
                        e.apart.add(new)         # a second object stays a second object
        for tid, trk in visible.items():
            s = self._seen[tid]
            s.angle, s.t, s.trk = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]), t, trk
            for _, b in list(trk.bbox_history)[-6:] or [(t, trk.bbox)]:   # (the frames since the last decision)
                c = bbox_center(b)
                s.moved = max(s.moved, math.hypot(c[0] - s.c0[0], c[1] - s.c0[1]))
        for tid in [k for k, s in self._seen.items() if k not in visible and t - s.t > g.switch_gap_s + 1e-9]:
            del self._seen[tid]
        self._owed &= set(self._seen)

    def _detach_jumps(self, visible: Dict[int, Track], t: float) -> None:
        """A track whose range jumped farther across a dropout (_range_jump: the IoU tracker matched another object that
        stepped in where the first one stood, within its max_age) is not what was said about it before: the memory
        entries said about it before the jump let go of it, as if its object had left view (module docstring:
        Identity), so the newcomer's first cue is judged as new, not as a repeat of a cue about someone else."""
        for tid, trk in visible.items():
            j = self._range_jump(trk)
            if j is None or j <= self._jumps.get(tid, -math.inf) + 1e-9:
                continue
            self._jumps[tid] = j
            for es in self._mem.values():
                for e in es:
                    if e.t < j - 1e-9 and (e.track_id == tid or tid in e.tids):
                        e.tids.discard(tid)
                        if e.track_id == tid:
                            e.track_id = -1      # its object left view: no track continues it
        for tid in [k for k in self._jumps if k not in visible and k not in self._seen]:
            del self._jumps[tid]

    def _range_jump(self, trk: Track) -> Optional[float]:
        """When the track's range last jumped farther (within the last release_span_s) by half the `closer` step (as
        _same_range: a successor that much farther is another object), or None. Medians of up to 3 samples on each
        side, so box jitter is not a jump. Across a dropout (samples jump_gap_s or more apart), or from one frame to the
        next when all 3 samples after are farther than all 3 before by more than the object could move away in that
        time (_AWAY_MPS, else _MAX_SPEED: the user walking on only brings it nearer): another person stepping in where
        the first stood as they leave, with no gap the tracker could see (an occlusion handover), is a step, not a
        walk."""
        g = self.gate_cfg
        rs = self._sample_ranges(trk)
        ident = self._identity_of(trk)[1]       # (walking on, the user only brings it nearer)
        v = _AWAY_MPS.get(ident, _MAX_SPEED.get(ident, 0.0))
        for k in range(len(rs) - 1, 0, -1):
            t0, t1 = rs[k - 1][0], rs[k][0]
            if rs[-1][0] - t1 > g.release_span_s + 1e-9:
                break
            before, after = [r for _, r in rs[max(0, k - 3):k]], [r for _, r in rs[k:k + 3]]
            step = _med(after) - _med(before) >= 0.5 * max(g.closer_m, g.closer_frac * _med(before)) - 1e-9
            if t1 - t0 >= g.jump_gap_s - 1e-9:
                return t1 if step else None
            if step and len(before) == 3 and len(after) == 3 and min(after) - max(before) >= v * (t1 - t0) - 1e-9:
                return t1
        return None

    def _commit(self, cue: Cue, trk: Optional[Track], e: Optional[_Said], t: float, reason: str = "") -> None:
        g = self.gate_cfg
        ident = self._identity_of(trk, cue)
        angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0]) if trk is not None else 0.0
        sec = sector_of(angle, g.sector_deg)
        if cue.action == "haptic":
            self._haptic[(ident, sec)] = t
            if reason == "critical" and trk is not None:
                self._crit_haptic[(ident, sec)] = (t, trk.ttc_s, trk.distance_m)
            return
        if cue.action not in SPEECH_ACTIONS:
            return
        before = replace(e, apart=set(e.apart), dups=set(e.dups), tids=set(e.tids)) if e is not None else None
        appr = self._approach.pop(cue.track_id, None)   # the approach was said
        bg_said = [(b, b.said) for b in self._bg if cue.track_id in b.track_ids]
        for b, _ in bg_said:                     # an in-path object said once is background from now on
            b.said = True
        owed = cue.track_id in self._owed
        self._owed.discard(cue.track_id)
        s = self._seen.get(cue.track_id)
        gap = s.gap if s is not None else None
        dist = trk.distance_m if trk is not None and trk.distance_m is not None else cue.distance_m
        goal = self._is_goal(cue, trk)
        if e is None:
            e = _Said(ident, sec, angle, t, dist, cue.track_id, trk.bbox if trk is not None else None, goal, t)
            self._mem.setdefault(ident, []).append(e)
        else:
            if not in_sector(angle, e.sector, g.sector_deg, g.sector_hyst_deg):
                e.sector = sec
            e.angle, e.t, e.dist, e.track_id, e.goal, e.seen = angle, t, dist, cue.track_id, goal, t
            e.bbox = trk.bbox if trk is not None else e.bbox
        e.tids.add(cue.track_id)
        e.cue = cue
        e.rng = self._smoothed_range(trk) if trk is not None else dist
        if s is not None:
            s.gap = 0.0                          # said since it came back: `reacquired` is used up
        if reason == "critical" and trk is not None:
            e.crit_t, e.crit_ttc, e.crit_dist = t, trk.ttc_s, trk.distance_m
        self._undo[cue.track_id] = (t, ident, e, before, appr, bg_said, gap, owed)

    def cut(self, cue: Cue, t_said: float, heard_s: float) -> bool:
        """A message the gate let through (about cue.track_id, said at t_said: the scheduler's cue, the gate's own, or one
        said late) was cut heard_s seconds in, by an interrupt or a spoken reply (the LiveController reports those; the
        gate on its own notices its own interrupts). If the user did not hear what it is about and how far it is
        (heard_enough), the memory takes the message back: the object is as it was before it (new, or as last heard),
        and a background object owed a say-once cue owes it again. So the scheduler's next cue about it is not held as
        a repeat of words nobody heard (v4_flip_size:970079: a reply cut "Chair, about 2.5 meters." after "Chair,";
        "Chair, very close." was then held as background until 1.2 s before contact). True if it took it back."""
        u = self._undo.get(cue.track_id)
        if u is None or heard_enough(cue, heard_s):
            return False
        t0, ident, e, before, appr, bg_said, gap, owed = u
        if t0 > t_said + 1e-6 or t_said - t0 > DISPLACED_KEEP_S + 1.0 or abs(e.t - t0) > 1e-9:
            return False                         # (a later message about it, or its object was said again since)
        del self._undo[cue.track_id]
        es = self._mem.get(ident)
        if before is None:
            if es is not None:
                es[:] = [x for x in es if x is not e]
                if not es:
                    del self._mem[ident]
        else:
            for f in fields(_Said):
                setattr(e, f.name, getattr(before, f.name))
        if appr is not None:
            self._approach.setdefault(cue.track_id, appr)
        for b, said in bg_said:
            b.said = said
        if owed:
            self._owed.add(cue.track_id)
        s = self._seen.get(cue.track_id)
        if s is not None and gap is not None:
            s.gap = gap
        self.stats["cut"] = self.stats.get("cut", 0) + 1
        self.reasons["cut (taken back)"] += 1
        return True

    def _prune(self, t: float) -> None:
        g = self.gate_cfg
        keep = 2.0 * max(g.hold_static_s, g.hold_moving_s, g.hold_goal_s)   # older entries pass anyway (as reminders)
        for ident in list(self._mem):
            # the 8 entries said most recently (an id switch adopts an old entry, so list order is creation order)
            es = sorted((e for e in self._mem[ident] if t - e.t <= keep), key=lambda e: e.t)[-8:]
            if es:
                self._mem[ident] = es
            else:
                del self._mem[ident]
        for k in [k for k, v in self._haptic.items() if t - v > g.haptic_gap_s]:
            del self._haptic[k]
        for k in [k for k, v in self._crit_haptic.items() if t - v[0] > g.critical_gap_s]:
            del self._crit_haptic[k]
        for k in [k for k, v in self._held.items() if t - v[1] > self._held_keep(v[2])]:
            del self._held[k]
        for k in [k for k in self._held_ref if k not in self._held]:
            del self._held_ref[k]
        for k in [k for k, v in self._approach.items() if t - v > g.approach_keep_s]:
            del self._approach[k]
        for k in [k for k, v in self._undo.items() if t - v[0] > DISPLACED_KEEP_S + 10.0]:
            del self._undo[k]

    # ---- background and operator ----------------------------------------------------------------------------
    def _in_path_near(self, label: str, bbox: BBox, dist: Optional[float], path_prob: float,
                      measured_dist: bool = True) -> bool:
        """In the walking path (path_prob >= critical_path, or the box inside the walking corridor) and nearer than
        bg_path_m. The corridor test matters with depth: a static object straight ahead has path_prob ~0.3 (the
        dynamics halve it when nothing approaches), but the user will walk into it. A box range (`measured_dist` False:
        the label's typical size assumed) may read an object half that size twice as far (BOX_SMALL): nearer than
        bg_path_m / BOX_SMALL."""
        g = self.gate_cfg
        if dist is None or dist >= (g.bg_path_m if measured_dist else g.bg_path_m / BOX_SMALL):
            return False
        return path_prob >= g.critical_path or self._in_corridor(label, bbox, dist)

    @staticmethod
    def _is_vehicle(identity: Tuple[str, str]) -> bool:
        return identity[0] == "object" and identity[1] in VEHICLES

    def _warm_closing(self, w: _Warm) -> bool:
        """Did the track measurably close in during the warm-up? A measured closing speed of release_mps, its depth
        dropping by max(release_m, release_frac x depth), or its box (unclipped width or height) growing by nodepth_grow
        (first third of the samples against the last third)."""
        g = self.gate_cfg
        if w.closing:
            return True
        if len(w.depths) >= 3:               # measured depth: trust it over the box
            k = max(1, len(w.depths) // 3)
            first, last = _med(w.depths[:k]), _med(w.depths[-k:])
            return first - last >= max(g.release_m, g.release_frac * first)
        for xs in (w.widths, w.heights):
            if len(xs) >= 3:
                k = max(1, len(xs) // 3)
                if _med(xs[-k:]) >= (1.0 + g.nodepth_grow) * _med(xs[:k]):
                    return True
        return False

    def _seen_enough(self, w: _Warm, span_s: float) -> bool:
        """Seen long enough to be judged: never on its first sample, warm_min_samples frames (box samples, or tracker
        matches for a track without box history) spanning span_s."""
        g = self.gate_cfg
        return w.n >= 2 and max(w.n, w.hits) >= g.warm_min_samples and w.last_t - w.first_t >= span_s - 1e-9

    def _warm_static(self, w: _Warm, span_s: float) -> bool:
        g = self.gate_cfg
        return (self._seen_enough(w, span_s) and not self._is_vehicle(w.identity)
                and w.max_disp <= g.static_disp and w.max_size <= g.static_size_frac and not self._warm_closing(w))

    def _late_operator(self, visible: Dict[int, Track], t: float) -> None:
        """operator_filter, and the warm-up found no user (its box flickered out as the warm-up ended, or its track was
        cut into short ones: v4_desk_still:900399): the user is found later, as the warm-up would have found them. A
        person whose object (its chain of tracks: _succeed) was in view before the warm-up ended, large and centred
        from its first sighting (no one who walked up), whose box never moved more than operator_disp, at rest and
        operator-like now (large, centred, within operator_max_m) for warm_hold_span_s of samples, becomes the operator
        (no cue about a still desk user)."""
        g = self.gate_cfg
        if not g.operator_filter or not self._warm_done or self._operator is not None or self._t0 is None:
            return
        best = None
        for tid, trk in visible.items():
            if not self._known_user(trk, self._t0 + g.warmup_s) or self._closing_now(trk):
                continue
            samples = [(tt, b) for tt, b in trk.bbox_history if tt <= t + 1e-9] or [(t, trk.bbox)]
            w = _Warm(self._identity_of(trk), trk.kind, trk.label, bbox_center(samples[0][1]),
                      max(1e-3, samples[0][1][3] - samples[0][1][1]), samples[0][0], samples[0][0], samples[0][1])
            for tt, b in samples:
                self._warm_sample(w, tt, b)
            w.hits, w.trk = int(trk.hits), trk
            if trk.distance_m is not None and trk.depth_conf >= 0.4:
                w.depths.append(float(trk.distance_m))
            if w.max_disp > g.operator_disp or w.max_size > g.static_size_frac:
                continue
            h = trk.bbox[3] - trk.bbox[1]
            if best is None or h > best[0]:
                best = (h, tid, w)
        if best is None:
            return
        _, tid, w = best
        self._bg = [b for b in self._bg if tid not in b.track_ids]   # (the user, not background)
        self._operator = self._record(w, tid)
        self._operator.noise = self._depth_noise(w.trk)
        self._operator.rest_tids.add(tid)
        self.releases.append((t, tid, "operator found after the warm-up (it missed the user)"))

    def _known_user(self, trk: Track, before: Optional[float]) -> bool:
        """A person in view (its chain of tracks: _succeed) since before `before`, large, centred and within
        operator_max_m from its first sighting (no one who walked up), whose box never moved more than operator_disp,
        and large, centred and within operator_max_m now (measured depth, else the nearer box range, as _operator_like):
        the desk user, however short its current track (a dropout cut it)."""
        g = self.gate_cfg
        if before is None or trk.kind != "object" or self._identity_of(trk)[1] != "person":
            return False
        s = self._seen.get(trk.track_id)
        if s is None or s.trk is not trk or s.b0 is None or min(s.first, trk.first_seen) > before + 1e-9:
            return False
        b0, b = s.b0, trk.bbox
        if (b0[3] - b0[1] < 0.8 * g.operator_min_height or abs(bbox_center(b0)[0] - 0.5) > g.operator_center_tol
                or s.moved > g.operator_disp or s.r0 is None or s.r0 > g.operator_max_m):
            return False
        if b[3] - b[1] <= g.operator_min_height or abs(bbox_center(b)[0] - 0.5) > g.operator_center_tol:
            return False
        if trk.depth_conf >= 0.4 and trk.distance_m is not None:
            dist = trk.distance_m
        else:
            rs = [r for r in self._box_ranges(b, "person") if r is not None]
            dist = min(rs) if rs else 0.0
        return dist <= g.operator_max_m

    def _operator_like(self, w: _Warm, bbox: BBox) -> bool:
        """The user in front of a desk webcam: a large, centred, still person at most operator_max_m away (measured
        depth; without it the nearer of the box-height and box-width ranges: the desk user's box is cut off at the
        bottom, so its height says 'far'; a box cut off on every side fills the frame: near)."""
        g = self.gate_cfg
        if w.kind != "object" or w.identity[1] != "person" or not self._seen_enough(w, g.warm_hold_span_s):
            return False
        if (bbox[3] - bbox[1]) <= g.operator_min_height or abs(bbox_center(bbox)[0] - 0.5) > g.operator_center_tol:
            return False
        if w.max_disp > g.operator_disp:
            return False
        if w.depths:
            dist = _med(w.depths)
        else:
            rs = [r for r in self._box_ranges(bbox, "person") if r is not None]
            dist = min(rs) if rs else 0.0
        return dist <= g.operator_max_m

    def _record(self, w: _Warm, tid: int, say_once: bool = False, said: bool = False) -> _Background:
        half = max(1, w.n // 2)              # a reference size only from a side the frame rarely cut
        return _Background(w.identity, w.kind, w.label, w.bbox, w.bbox, _med(w.depths) if w.depths else None,
                           _med(w.widths) if len(w.widths) >= half else None,
                           _med(w.heights) if len(w.heights) >= half else None,
                           _med(w.areas) if w.areas else None, {tid}, say_once=say_once, said=said,
                           last_trk=w.trk)       # (out of view as the warm-up ends: its return is judged as lost)

    def _finish_warmup(self) -> None:
        g = self.gate_cfg
        self._warm_done = True
        t_end = self._t0 + g.warmup_s
        best_op: Optional[Tuple[float, _Warm, int]] = None
        bg: List[Tuple[int, _Warm, bool, bool]] = []
        for tid, w in self._warm.items():
            present = self._seen_enough(w, max(g.bg_min_span_s, 0.5 * g.warmup_s)) and w.last_t >= t_end - 1.0
            if not present:
                continue
            if g.operator_filter and self._operator_like(w, w.bbox):
                h = w.bbox[3] - w.bbox[1]
                if best_op is None or h > best_op[0]:
                    best_op = (h, w, tid)
            if self._warm_static(w, g.bg_min_span_s):
                # (measured depth seen in the warm-up is trusted over the box: a child's box reads it farther)
                dref = _med(w.depths) if w.depths else (_med(w.dists) if w.dists else None)
                # said once before it is background: in the walking path and near, or a person / animal that walked into
                # place before this warm-up (it arrived: a rebaseline must not silence an arrival nobody heard of), or
                # owed a say-once cue when the warm-up began (rebaseline)
                s = self._seen.get(tid)
                arrived = (w.identity[1] in SELF_MOVING and s is not None and s.trk is w.trk
                           and s.moved > g.bg_release_disp)
                once = self._in_path_near(w.label, w.bbox, dref, w.path, bool(w.depths)) or arrived or tid in self._owed
                said = once and any(e.track_id == tid or tid in e.tids for es in self._mem.values() for e in es)
                bg.append((tid, w, once, said))
        op_tid = best_op[2] if best_op is not None else None
        for tid, w, once, said in bg:
            if tid != op_tid:                    # the user is the operator, not background
                self._bg.append(self._record(w, tid, say_once=once, said=said))
        self._owed -= set(self._warm)            # (owed now through its record, or the user, or gone)
        if best_op is not None:
            self._operator = self._record(best_op[1], best_op[2])
            if best_op[1].trk is not None:
                self._operator.noise = self._depth_noise(best_op[1].trk)
            self._operator.rest_tids.add(best_op[2])   # still through the warm-up
        self._warm.clear()

    def _update_closing(self, visible: Dict[int, Track], t: float) -> None:
        """Since when each visible track closes in at >= release_mps (measured), at every decision."""
        g = self.gate_cfg
        for tid in [k for k in self._closing_since if k not in visible]:
            del self._closing_since[tid]
        for tid, trk in visible.items():
            if self._measured(trk) and -trk.rel_speed_mps >= g.release_mps:
                self._closing_since.setdefault(tid, t)
            else:
                self._closing_since.pop(tid, None)

    def _drop(self, trk: Track, rec: _Background, latest: bool = False,
              first: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """(how much nearer than its warm-up reference, the reference) in metres: measured depth (median over
        release_smooth_s) when there is depth now and was in the warm-up, else the box (median of the last 3 frames)
        against its warm-up median: unclipped width, else height, else area, scaled by the reference range.
        `latest`: the last sample only (for a fast approach, where a median lags); `first`: the track's first one."""
        if trk.depth_conf >= 0.4 and rec.dist_ref is not None and trk.dist_history:
            d = (trk.dist_history[0][1] if first else trk.dist_history[-1][1] if latest else self._smoothed_depth(trk))
            return rec.dist_ref - d, rec.dist_ref
        hist = list(trk.bbox_history)
        boxes = [b for _, b in (hist[:1] if first else hist[-(1 if latest else 3):])] or [trk.bbox]
        H, W = TYPICAL_SIZE_M.get(rec.label, DEFAULT_SIZE_M)
        if rec.w_ref and not any(_cut_lr(b) for b in boxes):
            ratio, ref = _med([b[2] - b[0] for b in boxes]) / rec.w_ref, self.cam.fx * W / rec.w_ref
        elif rec.h_ref and not any(_cut_tb(b) for b in boxes):
            ratio, ref = _med([b[3] - b[1] for b in boxes]) / rec.h_ref, self.cam.fy * H / rec.h_ref
        elif rec.a_ref:
            # every side cut off now: the area of a cut box grows slower than the object's, so this under-reads
            ratio = math.sqrt(_med([bbox_area(b) for b in boxes]) / rec.a_ref)
            rs = [r for r in self._box_ranges(rec.bbox0, rec.label) if r is not None]
            ref = min(rs) if rs else None
        else:
            return None, None
        if rec.dist_ref is not None:
            ref = rec.dist_ref
        if ref is None or ratio <= 1e-6:
            return None, None
        return ref * (1.0 - 1.0 / ratio), ref

    def _closing_now(self, trk: Track) -> bool:
        """Closing in now: measured (the tracker's range rate >= release_mps), or without measured depth the box
        growing (>= nodepth_grow over 0.5-1 s, at >= nodepth_approach_mps). The warm-up judges a track on all its
        samples (after a rebaseline, its whole history): one that started to approach a moment ago is not static."""
        g = self.gate_cfg
        if self._measured(trk):
            return -trk.rel_speed_mps >= g.release_mps
        rng = self._range_now(trk)
        return rng is not None and self._box_closing(trk, rng) >= g.nodepth_approach_mps

    def _short_closing(self, trk: Track) -> float:
        """Closing speed over the last ~0.5 s: measured depth (no EMA lag beyond the tracker's; samples of the range
        source it has now, settled), else the box growth."""
        if trk.depth_conf >= 0.4 and len(trk.dist_history) >= 2:
            t1, z1, _ = trk.dist_history[-1]
            since = self._since(trk)
            prev = [(tt, z) for tt, z, _ in trk.dist_history if tt <= t1 - 0.45]
            if prev and prev[-1][0] > since + RATE_SETTLE_S - 1e-9:
                t0, z0 = prev[-1]
                return (z0 - z1) / max(1e-3, t1 - t0)
            if since == -math.inf or not prev:
                return 0.0                       # (else depth came back a moment ago: the box growth)
        rng = self._range_now(trk)
        return self._box_closing(trk, rng) if rng is not None else 0.0

    def _track_records(self, visible: Dict[int, Track], t: float) -> None:
        """Rest tracking for the background objects and the operator (module docstring: Release). At every decision the
        record's track (if in view and not closing in) adds its smoothed depth and box size; once it has rested for
        recapture_s (depth within rest_spread_m, box size within rest_size_frac, box centre within static_disp x 2) the
        record's references move to where it rests, so a lean during the warm-up or the user settling a little nearer
        is not an approach later, and a record released as approaching (or lost from view) is held again: it came up
        and stopped, and says nothing new until it moves again (a false release costs one cue, not a "Person." loop).
        An operator that comes back to its place after an approach release remembers the nearest point it came to."""
        g = self.gate_cfg
        if not self._warm_done:
            return
        for rec in ([self._operator] if self._operator is not None else []) + self._bg:
            if rec.released and not rec.recapture:
                continue                         # moved away from its place: an ordinary object for good
            trk = next((visible[tid] for tid in rec.track_ids if tid in visible), None)
            if trk is None:
                rec.rest.clear()
                continue
            rec.last_trk = trk
            if rec is not self._operator and not rec.released:
                jump = self._jump_after_gap(trk) or ("a step farther with no gap (another object in its place)"
                                                     if self._range_jump(trk) is not None else "")
                if jump or self._farther(trk, rec):
                    # another object stepped in where it stood: let go now, not only when a cue about it is judged (the
                    # jump leaves the 1 s window before the scheduler may name the newcomer)
                    drop, ref = self._drop(trk, rec)
                    self._release(rec, trk, t, approach=False, recapture=True,
                                  why=jump or f"{-(drop or 0.0):.2f} m farther than its {ref or 0.0:.1f} m reference")
                    continue
            if rec is self._operator and rec.released and rec.approach:
                r = self._range_of(trk, rec)
                if r is not None:
                    rec.nearest = r if rec.nearest is None else min(rec.nearest, r)
            if trk.track_id in self._closing_since or self._short_closing(trk) >= g.release_mps:
                rec.rest.clear()
                continue
            b = trk.bbox
            rec.rest.append((t, self._smoothed_depth(trk) if trk.depth_conf >= 0.4 else None,
                             None if _cut_lr(b) else b[2] - b[0], None if _cut_tb(b) else b[3] - b[1], bbox_area(b),
                             bbox_center(b)))
            if t - rec.rest[0][0] < g.recapture_s - 1e-9:
                continue
            while len(rec.rest) > 1 and t - rec.rest[1][0] >= g.recapture_s - 1e-9:
                rec.rest.pop(0)                  # keep exactly one sample at or before t - recapture_s
            rest = self._at_rest(rec.rest)
            if rest is None:
                continue
            d, w, h, a = rest
            if rec is self._operator and rec.released and rec.approach:
                back, ref = self._drop(trk, rec)  # where it rests, against the reference it was released from
                if back is not None and ref is not None and back < max(g.release_m, g.release_frac * ref) - 1e-9:
                    if rec.nearest is not None:  # came back: the same lean again is nothing new
                        rec.lean = rec.nearest if rec.lean is None else min(rec.lean, rec.nearest)
                else:
                    rec.lean = None              # settled somewhere nearer: its leans from there start afresh
                rec.nearest = None
            if rec is self._operator:
                rec.rest_tids.add(trk.track_id)  # seen resting at the user's place: the lean memory is about this track
                rec.noise = self._depth_noise(trk)
            if d is not None:
                rec.dist_ref = d
            rec.w_ref, rec.h_ref, rec.a_ref = (w or rec.w_ref), (h or rec.h_ref), a
            rec.bbox0 = b
            if rec.released:
                rec.released, rec.approach, rec.recapture = False, False, False
                if rec is self._operator:
                    rec.bbox = b
                for tid in rec.track_ids:
                    self._approach.pop(tid, None)
                where = f"{d:.2f} m" if d is not None else "its box"
                self.releases.append((t, trk.track_id, ("operator" if rec is self._operator else "background")
                                      + f" re-captured: at rest for {g.recapture_s:.0f} s at {where}"))

    def _at_rest(self, samples: List[Tuple]) -> Optional[Tuple[Optional[float], Optional[float], Optional[float], float]]:
        """(depth, width, height, area) medians when the samples show the object at rest, else None."""
        g = self.gate_cfg
        n = len(samples)
        if n < 3:
            return None
        cs = [x[5] for x in samples]
        if max(math.hypot(c[0] - cs[0][0], c[1] - cs[0][1]) for c in cs) > 2 * g.static_disp:
            return None
        meds: List[Optional[float]] = []
        for k in (1, 2, 3):
            xs = [x[k] for x in samples if x[k] is not None]
            if len(xs) < n:                      # a value missing in part of the window (depth dropped, edge cut)
                meds.append(None)
                continue
            spread = max(xs) - min(xs) if k == 1 else max(xs) / max(1e-6, min(xs)) - 1.0
            if spread > (g.rest_spread_m if k == 1 else g.rest_size_frac):
                return None
            meds.append(_med(xs))
        if meds[0] is None and meds[1] is None and meds[2] is None:
            areas = [x[4] for x in samples]
            if max(areas) / max(1e-6, min(areas)) - 1.0 > 2 * g.rest_size_frac:
                return None
        return meds[0], meds[1], meds[2], _med([x[4] for x in samples])

    def _range_of(self, trk: Track, rec: _Background) -> Optional[float]:
        """Its range now (m) on the record's scale: smoothed depth, else the box against the reference size."""
        drop, ref = self._drop(trk, rec)
        return None if drop is None or ref is None else ref - drop

    def _past_lean(self, trk: Track, rec: Optional[_Background]) -> bool:
        """An operator that came back to its place after an approach release is released (or critical) again only once
        it is lean_margin_m nearer than the nearest point it came back from: the desk user's leans repeat themselves.
        Only for a track the gate saw resting at the user's place: a stranger who walks up to the camera after the user
        left, or a visitor in front of the user, taken for the user (a new id over its box, a second box), never leaned."""
        if rec is None or rec.lean is None or trk.track_id not in rec.rest_tids:
            return True
        r = self._range_of(trk, rec)
        return r is None or r <= rec.lean - self.gate_cfg.lean_margin_m + 1e-9

    def _release_why(self, trk: Track, rec: _Background, t: float) -> str:
        """Why a background / operator track is released as approaching, or '' (module docstring: Release)."""
        g = self.gate_cfg
        if rec is self._operator and not self._past_lean(trk, rec):
            return ""
        drop, ref = self._drop(trk, rec)
        # (the operator, by measured depth: a margin for its own depth noise and the error of its reference on top, so a
        # shallow lean on a noisy depth map is no approach; nobody else is ever taken for the operator: _match_operator)
        margin = (max(g.lean_noise_m, 2.0 * rec.noise)
                  if rec is self._operator and trk.depth_conf >= 0.4 and rec.dist_ref is not None else 0.0)
        if drop is not None and ref is not None and drop >= max(g.release_m, g.release_frac * ref) + margin - 1e-9:
            return f"{drop:.2f} m nearer than its {ref:.1f} m reference"
        if ref is not None:
            last, _ = self._drop(trk, rec, latest=True)
            fast = self._short_closing(trk)
            if (last is not None and last >= max(g.release_m, g.release_frac * ref) + margin - 1e-9
                    and fast >= g.release_fast_mps):
                return f"{last:.2f} m nearer than its {ref:.1f} m reference, closing {fast:.1f} m/s"
        if not self._measured(trk) and self._near_without_depth(trk):
            return "looming in the walking path (no measured depth)"
        since = self._closing_since.get(trk.track_id)
        if since is not None and t - since >= g.release_span_s - 1e-9:
            return f"closing {-trk.rel_speed_mps:.2f} m/s for {t - since:.1f} s"
        if (self._measured(trk) and math.isfinite(trk.ttc_s) and trk.ttc_s < g.release_ttc_s
                and -trk.rel_speed_mps >= g.approach_min_mps):
            return f"ttc {trk.ttc_s:.1f} s closing {-trk.rel_speed_mps:.2f} m/s"
        return ""

    def _release(self, rec: _Background, trk: Track, t: float, approach: bool, why: str = "",
                 recapture: Optional[bool] = None) -> None:
        """approach: its next cue passes (approach_keep_s); recapture (default: approach): held again once at rest."""
        rec.released, rec.approach = True, approach
        rec.recapture = approach if recapture is None else recapture
        rec.rest.clear()
        if approach and rec is self._operator:
            rec.nearest = self._range_of(trk, rec)
        self.releases.append((t, trk.track_id, ("operator " if rec is self._operator else "background ") + why))
        if approach:
            for tid in rec.track_ids | {trk.track_id}:
                self._approach[tid] = t

    def _in_path(self, trk: Track, max_m: float = math.inf) -> bool:
        """In the walking path (path_prob >= critical_path) or its box in the walking corridor, nearer than max_m."""
        rng = self._range_now(trk)
        if rng is not None and rng >= max_m:
            return False
        if trk.path_prob >= self.gate_cfg.critical_path:
            return True
        return rng is not None and self._in_corridor(trk.label, trk.bbox, rng)

    def _path_arrival(self, trk: Track) -> bool:
        """Did the track's object come into view after this warm-up began (by more than warm_hold_span_s; counting the
        tracks it took over from)? Then it is not part of the scene the warm-up learns: someone stepping into the path a
        second after the camera started (or after a rebaseline) is not held for the warm-up."""
        if self._t0 is None:
            return False
        s = self._seen.get(trk.track_id)
        first = min(s.first, trk.first_seen) if s is not None and s.trk is trk else trk.first_seen
        return first > self._t0 + self.gate_cfg.warm_hold_span_s + 1e-9

    def _match_background(self, trk: Track, t: float, visible: Optional[Dict[int, Track]] = None) -> Optional[_Background]:
        """The background record of this track (its own, else an unreleased one of the same identity its box overlaps
        at about its range, now and where the track was first seen, while its own track is out of view: an id switch;
        a person stepping in from farther away where it stood is another object, also once it has come that near).
        A person or animal re-acquired after more than lost_s out of view is released until it rests again
        (recapture_s): it may have moved meanwhile, so the scheduler's cue about it is not held as background."""
        g = self.gate_cfg
        b = next((b for b in self._bg if trk.track_id in b.track_ids), None)
        if b is None:
            ident = self._identity_of(trk)
            b = next((b for b in self._bg if not b.released and b.identity == ident and bbox_iou(b.bbox, trk.bbox) >= 0.3
                      and not any(tid in (visible or {}) for tid in b.track_ids)    # (its object is in view: a second one)
                      and not self._farther(trk, b) and not self._farther(trk, b, first=True)
                      and not self._jump_from(b.last_trk, trk)), None)
            if b is None:
                return None
            b.track_ids.add(trk.track_id)
        prev = b.last_trk
        if (not b.released and b.identity[1] in SELF_MOVING and prev is not None and prev is not trk
                and trk.first_seen - prev.last_seen > g.lost_s + 1e-9):
            self._release(b, trk, t, approach=False, recapture=True,
                          why=f"lost from view for {trk.first_seen - prev.last_seen:.1f} s")
        b.last_trk = trk
        return b

    def _match_operator(self, trk: Track, visible: Optional[Dict[int, Track]] = None):
        """Is the track the desk user: True (the user's track, a tracker id switch of it, or the user detected twice),
        "part" (a part of the user detected as a person: an arm, a hand; _part_of), or False (anyone else)."""
        g, op = self.gate_cfg, self._operator
        if op is None or op.released or trk.kind != "object" or self._identity_of(trk)[1] != "person":
            return False
        if trk.track_id not in op.track_ids:
            mine = [visible[tid] for tid in op.track_ids if tid in visible] if visible else []
            if mine:
                # the user's own track is in view: this is another person (a visitor leaning in beside, behind or in
                # front of the user), unless the two boxes are one detection twice (overlapping, and at the user's
                # range when both have depth) or a part of the user detected as a person (an arm, a hand: _part_of);
                # the user's box never moves onto it
                if any(bbox_iou(m.bbox, trk.bbox) >= g.operator_dup_iou and self._dup_range(trk, m) for m in mine):
                    return True
                return "part" if any(self._part_of(trk, m) for m in mine) else False
            h = trk.bbox[3] - trk.bbox[1]
            cx = bbox_center(trk.bbox)[0]
            if not (h >= 0.8 * g.operator_min_height and abs(cx - 0.5) <= g.operator_center_tol + 0.1
                    and bbox_iou(op.bbox, trk.bbox) >= 0.3):
                return False
            for first in (False, True):      # farther than the user's place, now or where it was first seen:
                drop, ref = self._drop(trk, op, first=first)   # someone else (a stranger walking up to the webcam)
                if drop is not None and ref is not None and -drop >= max(g.release_m, g.release_frac * ref) - 1e-9:
                    return False
            op.track_ids.add(trk.track_id)   # a tracker id switch of the user
        op.bbox = trk.bbox          # follow the user as they shift in the chair
        return True

    def _part_of(self, trk: Track, user: Track) -> bool:
        """A second, smaller person box on the desk user's (an arm or a hand the detector calls a person): its bottom is
        cut by the frame where the user's is (a body part attached to the user; anyone standing in front of a desk webcam
        has their feet in view until they are about 1.6 m away), it overlaps the user's box, it was so from the first
        sighting of its object (one who walked up never was), it does not close in, and with measured depth on both it
        is not farther than the user (a visitor in front of the user always is). Held as the user: no reminder, no
        `lost` cue about the user's own arm."""
        g = self.gate_cfg
        b, u = trk.bbox, user.bbox
        if not (u[3] >= 0.9 and abs(b[3] - u[3]) <= 0.03 and min(b[2], u[2]) - max(b[0], u[0]) > 0.0):
            return False
        s = self._seen.get(trk.track_id)
        b0 = s.b0 if s is not None and s.trk is trk and s.b0 is not None else (trk.bbox_history[0][1] if trk.bbox_history else b)
        if abs(b0[3] - u[3]) > 0.03:
            return False
        if (trk.depth_conf >= 0.4 and user.depth_conf >= 0.4 and trk.distance_m is not None and user.distance_m is not None
                and trk.distance_m - user.distance_m >= max(g.release_m, g.release_frac * user.distance_m) - 1e-9):
            return False
        return not self._closing_now(trk)

    @staticmethod
    def _depth_noise(trk: Track) -> float:
        """The spread (sigma, m; robust: 1.4826 x the median absolute deviation, at most 0.08) of a track's measured range
        samples, for one at rest; 0 without measured depth or with fewer than 6 samples."""
        if trk.depth_conf < 0.4 or len(trk.dist_history) < 6:
            return 0.0
        zs = [z for _, z, _ in trk.dist_history]
        m = _med(zs)
        return min(0.08, 1.4826 * _med([abs(z - m) for z in zs]))

    def _sample_ranges(self, trk: Track) -> List[Tuple[float, float]]:
        """(time, range) of the track's samples: measured depth when it has depth now, else its boxes (width when the
        frame edge does not cut it, else height)."""
        if trk.depth_conf >= 0.4 and trk.dist_history:
            lo = self._since(trk) + RATE_SETTLE_S - 1e-9
            zs = [(tt, z) for tt, z, _ in trk.dist_history if tt > lo]
            if len(zs) >= 2:                     # (depth since it came back, settled; until then the boxes)
                return zs
        out = []
        for tt, b in trk.bbox_history:
            rh, rw = self._box_ranges(b, trk.label)
            r = rw if rw is not None else rh
            if r is not None:
                out.append((tt, r))
        return out

    def _jump_after_gap(self, trk: Track) -> str:
        """The track's range jumped farther across a dropout (samples jump_gap_s or more apart, within the last
        release_span_s) by max(jump_m, jump_frac x range): the IoU tracker matched another object that stepped in where
        a background object stood (a person 3.5 m away where one stood at 3.0 m). '' if not."""
        g = self.gate_cfg
        rs = self._sample_ranges(trk)
        for (t0, r0), (t1, r1) in zip(reversed(rs[:-1]), reversed(rs[1:])):
            if rs[-1][0] - t1 > g.release_span_s + 1e-9:
                break
            if t1 - t0 >= g.jump_gap_s - 1e-9:
                if r1 - r0 >= max(g.jump_m, g.jump_frac * r0) - 1e-9:
                    return f"{r1 - r0:.2f} m farther after a {t1 - t0:.1f} s dropout"
                break
        return ""

    def _jump_from(self, old: Optional[Track], trk: Track) -> bool:
        """Is a new track first seen materially farther (max(jump_m, jump_frac x range)) than the old one was last?"""
        g = self.gate_cfg
        a, b = (self._sample_ranges(old) if old is not None else []), self._sample_ranges(trk)
        return bool(a and b) and b[0][1] - a[-1][1] >= max(g.jump_m, g.jump_frac * a[-1][1]) - 1e-9

    def _dup_range(self, trk: Track, user: Track) -> bool:
        """A second box over the user's is the user detected twice only at the user's range (both with depth: within
        max(release_m, release_frac x range)); without depth the boxes cannot tell, and the overlap decides."""
        g = self.gate_cfg
        if trk.depth_conf < 0.4 or user.depth_conf < 0.4 or trk.distance_m is None or user.distance_m is None:
            return True
        return abs(trk.distance_m - user.distance_m) < max(g.release_m, g.release_frac * user.distance_m) - 1e-9

    def _farther(self, trk: Track, rec: _Background, first: bool = False) -> bool:
        """Is the track materially farther than the record's reference (max(closer_m, closer_frac x reference), as the
        `closer` rule)? Then it is not the object resting at that place. `first`: where the track was first seen."""
        g = self.gate_cfg
        drop, ref = self._drop(trk, rec, first=first)
        return drop is not None and ref is not None and -drop >= max(g.closer_m, g.closer_frac * ref) - 1e-9

    def _background_due(self, trk: Track, rec: _Background, t: float) -> bool:
        """A background object in the walking path, not said for hold_moving_s (or never), that came release_m nearer
        than where it was last said (or than where it rests): the scheduler's cue about it is judged like any other's
        (_hold_s: a reminder; or new). Its approach release needs max(release_m, release_frac x range) (a desk user's
        lean is 0.25 m): an approach that started as it went out of view, or in a flicker, may never show that much
        while in view. (While the rescue judges a cue held about it as stale: from where it was when the scheduler
        offered it, if that is farther, as _closing_in_path.)"""
        g = self.gate_cfg
        e = self._own_entry(trk.track_id)
        if e is not None:
            if t - e.t < g.hold_moving_s - 1e-9:
                return False
            ref = e.rng
        else:
            ref = self._drop(trk, rec)[1]
        if ref is None:
            return False
        # (closing at the `closing` step from there, by its latest sample when it closes fast: a median lags)
        return (self._path_nearer(trk, ref if self._closing_ref is None else max(ref, self._closing_ref))
                or self._closing_in_path(trk, ref))

    def _background_reason(self, trk: Track, t: float, visible: Optional[Dict[int, Track]] = None) -> Optional[str]:
        """'warmup' / 'background' / 'operator' when the cue must be held for that reason, else None. Releases a
        background / operator track that closes in (an approach release: its next cue passes)."""
        g = self.gate_cfg
        if not (g.background or g.operator_filter):
            return None
        if not self._warm_done:
            w = self._warm.get(trk.track_id)
            if w is None:
                return None
            # (in the walking path: only once seen static for bg_min_span_s, never an arrival since the warm-up began)
            in_path = self._in_path(trk)
            # (in the path and near by measured depth seen in this warm-up, else by the box at its nearest: BOX_SMALL; or
            # owed a say-once cue: rebaseline)
            meas = bool(w.depths) or trk.depth_conf >= 0.4
            near_d = _med(w.depths) if w.depths else trk.distance_m
            static = (g.background and self._warm_static(w, g.bg_min_span_s if in_path else g.warm_hold_span_s)
                      and not self._closing_now(trk)
                      and not self._in_path_near(trk.label, trk.bbox, near_d, trk.path_prob, meas)
                      and not (in_path and self._path_arrival(trk)) and trk.track_id not in self._owed)
            # (someone who may be the desk user: not while closing in, as the background rule: a stranger walking up to
            # the webcam during a rebaseline's warm-up is not held, deskwalk:302868; or the user the gate saw before this
            # warm-up began, now under a new track after a dropout: _known_user)
            op_like = (g.operator_filter and (self._operator_like(w, trk.bbox) or self._known_user(trk, self._t0))
                       and not self._closing_now(trk))
            return "warmup" if (static or op_like) else None
        op = self._match_operator(trk, visible) if g.operator_filter else False
        if op == "part":
            return "operator"                    # (the user's own track speaks for the user: its release, its leans)
        if op:
            why = self._release_why(trk, self._operator, t)
            if why:
                self._release(self._operator, trk, t, approach=True, why=why)
                return None
            return "operator"
        if g.background:
            b = self._match_background(trk, t, visible)
            if b is not None and not b.released:
                why = self._release_why(trk, b, t)
                if why:
                    self._release(b, trk, t, approach=True, why=why)
                    return None
                c, c0 = bbox_center(trk.bbox), bbox_center(b.bbox0)
                if math.hypot(c[0] - c0[0], c[1] - c0[1]) > g.bg_release_disp:
                    self._release(b, trk, t, approach=False, why="moved")
                    return None
                jump = self._jump_after_gap(trk) or ("a step farther with no gap (another object in its place)"
                                                     if self._range_jump(trk) is not None else "")
                if self._farther(trk, b) or jump:
                    # another object the tracker matched to it after a short dropout, or as the first one left (it stepped
                    # in where the background object stood), or it moved away: not at its place; held again once it
                    # rests (recapture_s)
                    drop, ref = self._drop(trk, b)
                    self._release(b, trk, t, approach=False, recapture=True,
                                  why=jump or f"{-(drop or 0.0):.2f} m farther than its {ref or 0.0:.1f} m reference")
                    return None
                b.bbox = trk.bbox
                if b.say_once and not b.said:
                    return None              # in the walking path and near: said once before it is silenced
                if self._background_due(trk, b, t):
                    return None              # in the walking path, not said for hold_moving_s: judged as any object
                return "background"
        return None

    # ---- saying a cue again ----------------------------------------------------------------------------------
    def _refreshed(self, cue: Cue, angle0: float, trk: Track, t: float) -> Cue:
        """A copy of an earlier cue for saying now: its distance and direction phrases follow the object, and it is
        about the object's current track (after an id switch, the successor)."""
        from ..cues.compose import direction_phrase, distance_phrase
        msg = cue.message or ""
        units = getattr(self.cfg, "units", "m") or "m"
        d_now = self._range_now(trk)
        if msg and cue.distance_m is not None and d_now is not None:
            new = distance_phrase(d_now, trk.depth_conf, units)
            # the phrase as the cue said it: with the depth confidence it had then, which may differ from the track's
            # now (a stale "about 6 meters" said of an object 1.6 m away would be worse than none)
            for conf in (trk.depth_conf, 0.9, 0.0):
                old = distance_phrase(cue.distance_m, conf, units)
                if old and f", {old}" in msg:
                    if old != new:
                        msg = msg.replace(f", {old}", f", {new}" if new else "", 1)
                    break
        if msg:
            angle = self.cam.angle_deg_from_u(bbox_center(trk.bbox)[0])
            old, new = direction_phrase(angle0), direction_phrase(angle)
            if old != new and f", {old}" in msg:
                msg = msg.replace(f", {old}", f", {new}", 1)
        ttc = trk.ttc_s if self._measured(trk) and math.isfinite(trk.ttc_s) else cue.ttc_s
        return replace(cue, t=t, message=msg, distance_m=d_now if d_now is not None else cue.distance_m, ttc_s=ttc,
                       track_id=trk.track_id, source_id=getattr(trk, "source_id", cue.source_id))

    # ---- reporting ------------------------------------------------------------------------------------------
    def describe(self) -> str:
        s, g = self.stats, self.gate_cfg
        held = sum(v for k, v in s.items() if k.startswith("suppressed_"))
        bg = sum(1 for b in self._bg if not b.released)
        op = "on" if g.operator_filter else "off"
        if g.operator_filter:
            op += " (user found)" if self._operator is not None and not self._operator.released else " (no user)"
        return (f"gate[{self.name}]: {s['passed']} passed ({s.get('rescued', 0)} said by the gate itself), {held} held "
                f"(repeat {s['suppressed_repeat']}, background {s['suppressed_background']}, unconfirmed "
                f"{s['suppressed_unconfirmed']}, haptic {s['suppressed_haptic']}, "
                f"read-mode muted {s.get('suppressed_muted', 0)}); "
                f"hold {g.hold_static_s:.0f}/{g.hold_moving_s:.0f} s, sector +-{g.sector_deg:.0f} deg, "
                f"background {'on' if g.background else 'off'} ({bg} objects), operator {op}; "
                f"remembers {sum(len(v) for v in self._mem.values())}")
