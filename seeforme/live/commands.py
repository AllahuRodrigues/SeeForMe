"""What a voice (or typed) command does in the terminal demo, and the phrase that confirms it.

    goal_tags(text)          "the exit" -> [exit]; "room 204" -> [room 204, 204]; "a seat" -> [seat, chair, bench]
                             (a port of goalTags in web/src/main.ts, so both apps look for the same thing)
    answer(question, tracks) rule-based answers about the current view (a port of the web app's answer()); it never
                             says or implies that a path is clear: silence is not "all clear". Path questions also
                             report the cane's confirmed obstacle ahead (LiveController.cane_note), as the web app does
    is_safety_cue / is_safety_track
                             does a cue (a track) carry safety information: the pause filter passes such cues, and the
                             controller lets the scheduler cut a cane cue or a reply with them
    CommandRouter            executes a voice.Command against the LiveController, the Speaker and the camera, and
                             returns the short phrase to speak (scripts/run_live.py speaks and prints it)

Pause. "Pause" / "stop" / "wait" stops the scheduler's routine announcements while the camera, tracker, gate and
preview keep running: router.wrap(policy) is the policy the pipeline calls (outside the gate, the mode filter and the
cane layer), and while paused it passes only cues that carry safety information (is_safety_cue: an interrupt or
haptic, a cane / guide-dog cue, a critical severity, an approaching object, an object in the walking path, a vehicle,
or the FIND goal), unchanged and at the same decision, so a pause never delays a hazard warning. Cues it holds stay
"said" for the gate's repeat memory. The camera stays on: a pause that names it ("camera off", "stop the camera") gets a
reply that says so and that quitting the demo turns it off (CAMERA_STAYS_ON), never one that lets the user think the camera
is off. "Mute" is the full silence (the Speaker is disabled, as the space key does).
"Start" resumes both, with a one-word reply. A reply holds the audio channel (scripts/run_live.py registers it with
controller.reply_cue), but never ahead of a hazard: while it plays and something in view could need a safety cue
(is_safety_track), the controller shows the scheduler the channel it would have without the reply, and a safety cue
cuts the reply. Typed "p" / "s" toggle pause / sound (Command arg "toggle"), as the window keys do.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from ..campus.world import EVENT_SPECS, default_event_type
from ..geometry.camera import CameraModel
from .voice import Command, normalize

# what the COCO detector can name (web/src/yolo.ts COCO); every other goal needs OCR (signs) or an open vocabulary
COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
LABELS = frozenset(COCO)
STOPWORDS = frozenset({"a", "an", "the", "my", "your", "some", "any", "me", "to", "for", "of", "in", "on", "at", "and",
                       "or", "please", "nearest", "closest"})
ALIASES = {"seat": ["chair", "bench"], "people": ["person"], "bike": ["bicycle"], "table": ["dining table"],
           "sofa": ["couch"], "plant": ["potted plant"], "phone": ["cell phone"], "television": ["tv"]}
VEHICLES = frozenset({"car", "bus", "truck", "bicycle", "motorcycle", "train", "scooter", "van", "motorbike", "bike", "tram"})
# words a question can ask about besides detector labels: matched against labels (open-vocabulary detectors) and signs
NAV_WORDS = frozenset({"stairs", "staircase", "steps", "door", "doors", "exit", "entrance", "elevator", "lift", "escalator",
                       "restroom", "bathroom", "curb", "pole", "crosswalk", "crossing"})
SIGN_WORDS = frozenset({"sign", "signs", "text", "writing"})
VEHICLE_WORDS = {"vehicle": sorted(VEHICLES), "vehicles": sorted(VEHICLES), "traffic": sorted(VEHICLES)}
CRITICAL_SEVERITY = 0.9      # as TaskAdapter.critical_severity
APPROACH_TTC_S = 6.0         # as TaskAdapter.approach_ttc_s
PATH_PROB = 0.5              # as GateConfig.critical_path
MEASURED_DEPTH = 0.4         # depth_conf of a measured range (without it TrackDynamics assumes the user walks at 1 m/s)
CLOSING_MPS = 0.25           # closing speed that counts as approaching


def _has_digit(w: str) -> bool:
    return any(c.isdigit() for c in w)


def goal_tags(raw: str, labels: Iterable[str] = LABELS) -> List[str]:
    """"the exit" -> [exit]; "room 204" -> [room 204, 204]; "a seat" -> [seat, chair, bench]. Stopwords and tokens
    shorter than 3 characters are dropped (a lone "a" would match almost any sign); aliases and plurals map to detector
    labels. Same rules as goalTags in web/src/main.ts."""
    labels = labels if isinstance(labels, (set, frozenset)) else frozenset(labels)
    words = [w for w in re.sub(r"[^\w\s]", " ", (raw or "").lower()).replace("_", " ").split() if w not in STOPWORDS]
    # "room 204": the number identifies the goal, so a lone "room" (any Room 310 sign) is not a tag
    singles = [w for w in words if _has_digit(w)] if len(words) > 1 and any(_has_digit(w) for w in words) else words
    phrase = " ".join(words)
    # a lone one-digit number ("gate 7" -> "7") would match any sign with a 7, so it only counts inside the phrase
    tags = [w for w in [phrase, *singles]
            if w and (len(w) >= 3 or w in labels or (_has_digit(w) and (len(w) >= 2 or w == phrase)))]
    for w in words:
        tags += ALIASES.get(w, [])
        if w.endswith("s") and w[:-1] in labels:
            tags.append(w[:-1])
    return list(dict.fromkeys(tags))


def _has_word(hay: str, w: str) -> bool:
    """Whole-word match on normalised text, so a "204" goal does not fire on a "1204" sign."""
    return bool(w) and f" {w} " in f" {normalize(hay)} "


def names_goal(label: str, text: Optional[str], tags: Sequence[str]) -> bool:
    lab = (label or "").lower()
    for g in tags:
        g = str(g).lower()
        if g == lab or (g == "seat" and lab in ("chair", "bench")) or (text and _has_word(text, g)):
            return True
    return False


def _matches(trk, tags: Sequence[str]) -> bool:
    return names_goal(trk.label, trk.text if trk.kind == "text" else None, tags)


def _class_severity(label: str, kind: str) -> float:
    try:
        return EVENT_SPECS[default_event_type(label, kind)].severity
    except KeyError:
        return 0.0


def is_safety_track(track, goal: Sequence[str] = ()) -> bool:
    """Could a cue about this track carry safety information? An object in the walking path, an approaching one (finite
    TTC under 6 s, or a measured closing speed), a vehicle or another class severe enough to be critical (stairs), or
    the FIND goal. The controller asks this before it lets the scheduler decide as if a cane cue or a reply did not
    hold the channel, so it errs towards True."""
    if track is None:
        return False
    kind = getattr(track, "kind", "object") or "object"
    label = (getattr(track, "label", "") or "").lower()
    if kind == "object" and (getattr(track, "path_prob", 0.0) or 0.0) >= PATH_PROB:
        return True
    ttc = getattr(track, "ttc_s", math.inf)
    if ttc is not None and math.isfinite(ttc) and ttc < APPROACH_TTC_S:
        return True
    if (getattr(track, "depth_conf", 0.0) or 0.0) >= MEASURED_DEPTH and (getattr(track, "rel_speed_mps", 0.0) or 0.0) <= -CLOSING_MPS:
        return True                          # measurably closing in, however far
    if kind == "object" and (label in VEHICLES or _class_severity(label, kind) >= CRITICAL_SEVERITY):
        return True
    return bool(goal) and names_goal(label, getattr(track, "text", None) if kind == "text" else None, goal)


def is_safety_cue(cue, track=None, goal: Sequence[str] = ()) -> bool:
    """Does this cue carry safety information (it must be said even while paused, and it may cut a reply)? An interrupt
    or haptic, a cane / guide-dog cue (the corridor ahead), a critical severity, an approaching object (finite TTC
    under 6 s, measured or not, or a measured closing speed), an object in the walking path, a vehicle, or the FIND
    goal. Errs towards True."""
    if getattr(cue, "action", "speak") in ("interrupt", "haptic"):
        return True
    # the controller's own layers (the cane / guide dog: track id -7, policy "cane") speak only about the walking corridor
    if getattr(cue, "policy", "") == "cane" or (getattr(cue, "track_id", 0) or 0) < 0:
        return True
    if (getattr(cue, "severity", 0.0) or 0.0) >= CRITICAL_SEVERITY:
        return True
    ttc = getattr(cue, "ttc_s", None)
    if ttc is not None and math.isfinite(ttc) and ttc < APPROACH_TTC_S:
        return True
    label = (getattr(cue, "label", "") or "").lower()
    if label in VEHICLES:
        return True
    if is_safety_track(track, goal):
        return True
    return bool(goal) and names_goal(label, getattr(cue, "text", None), goal)


def read_phrase(tracks: Iterable, ocr: bool = True) -> str:
    """What the r key and "read" say: the text tracks in view now."""
    texts = [tr.text for tr in tracks if tr.kind == "text" and tr.text and tr.misses == 0]
    if texts:
        return "The text says: " + ". ".join(dict.fromkeys(texts))
    if ocr:
        return "I don't see readable text. Hold the sign steady, closer."
    return "Reading is off. Restart without the no-OCR option."


# ---- answers ------------------------------------------------------------------------------------------------------
def _visible(tracks: Iterable) -> list:
    return [t for t in tracks if getattr(t, "confirmed", True) and getattr(t, "misses", 0) == 0
            and not getattr(t, "remembered", False)]


def _dist_of(t) -> float:
    d = getattr(t, "distance_m", None)
    return d if d is not None and math.isfinite(d) else 99.0


def _angle(t, cam: CameraModel) -> float:
    return cam.angle_deg_from_u(0.5 * (t.bbox[0] + t.bbox[2]))


def _direction(t, cam: CameraModel) -> str:
    a = _angle(t, cam)
    if abs(a) < 10:
        return "straight ahead"
    if abs(a) < 25:
        return "slightly to your right" if a > 0 else "slightly to your left"
    return "on your right" if a > 0 else "on your left"


def _distance(t, units: str) -> str:
    d = getattr(t, "distance_m", None)
    if d is None or not math.isfinite(d):
        return ""
    if getattr(t, "depth_conf", 0.0) < 0.45:       # a box-size guess: no number
        return "very close" if d < 2.5 else "fairly close" if d < 4.5 else "further away"
    if units == "ft":
        return f"about {round(d * 3.28)} feet away"
    v = round(d * 2) / 2 if d < 3 else round(d)
    return f"about {v:g} meter{'' if v == 1 else 's'} away"


def _article(w: str) -> str:
    return "an" if w[:1].lower() in "aeiou" else "a"


def _desc(t, cam: CameraModel, units: str) -> str:
    if t.kind == "text" and t.text:
        return f"a sign that says {t.text}, {_direction(t, cam)}"
    d = _distance(t, units)
    return f"{_article(t.label)} {t.label}{' ' + d + ',' if d else ''} {_direction(t, cam)}"


def _list(ts: Sequence, cam: CameraModel, units: str) -> str:
    parts = [_desc(t, cam, units) for t in ts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]


def _side(q: str) -> Optional[str]:
    if re.search(r"\bleft\b", q):
        return "left"
    if re.search(r"\bright\b", q):
        return "right"
    if re.search(r"\b(?:ahead|in front|straight)\b", q):
        return "ahead"
    return None


def _on_side(t, side: Optional[str], cam: CameraModel) -> bool:
    a = _angle(t, cam)
    return side is None or (side == "left" and a < -10) or (side == "right" and a > 10) or (side == "ahead" and abs(a) < 12)


def _asked_object(q: str) -> Optional[Tuple[str, List[str]]]:
    """The first thing the question names (word as said, tags to match tracks), or None."""
    padded = f" {q} "
    for lab in COCO:
        if " " in lab and f" {lab} " in padded:
            return lab, [lab]
    for w in q.split():
        if w in VEHICLE_WORDS:
            return "vehicle", VEHICLE_WORDS[w]
        if w in SIGN_WORDS:
            return "sign", []                       # [] = any text track
        if w in LABELS or w in ALIASES or w in NAV_WORDS or (w.endswith("s") and w[:-1] in LABELS):
            return w, goal_tags(w) or [w]
    return None


_QUESTION = re.compile(r"^(?:what|what's|whats|where|where's|is|are|isn't|aren't|how|how's|can|could|do|does|did|who|which|"
                       r"any|anything|anyone|anybody|tell|should|am|will|would)\b")
_PATH_WORDS = re.compile(r"\b(?:clear|safe|path|way|walk|obstacles?|blocked)\b")


def is_question(text: str) -> bool:
    """Something answer() can answer: a question, or a phrase that names an object, a side or the path."""
    q = normalize(text)
    return bool(q) and bool(_QUESTION.match(q) or _asked_object(q) or _side(q) or _PATH_WORDS.search(q))


NOT_CLEAR = "that does not mean it is clear"
# how a path answer ends: the user's own travel aid (LiveController.aid, run_live --aid; web/src/cane.ts aidAdvice says
# the same), never "clear". Without a known aid, both.
AID_ADVICE = {"cane": "keep using your cane", "guide-dog": "keep working with your guide dog", "none": "go carefully"}


def aid_advice(aid: Optional[str]) -> str:
    return AID_ADVICE.get(aid or "", "keep using your cane or guide dog")


def says_clear(text: str) -> bool:
    """Does this text say or imply that something is clear? (The only "clear" allowed is in NOT_CLEAR.)"""
    t = normalize(text).replace(normalize(NOT_CLEAR), " ").replace("does not mean the way is clear", " ")
    return bool(re.search(r"\b(?:clear|clearer|safe to|nothing there|all good|no obstacles?)\b", t))


def answer(question: str, tracks: Iterable, camera: Optional[CameraModel] = None, units: str = "m",
           cane_note: str = "", aid: Optional[str] = None) -> str:
    """A spoken answer about the current camera view. Never says a path is clear: an empty answer says what could not
    be confirmed, that this does not mean it is clear, and that the camera can miss things. `cane_note` (from
    LiveController.cane_note, "" without one) is the cane's confirmed obstacle ahead: path and "ahead" questions report
    it, as the web app's answer() does, so an empty answer never ignores a wall the depth map shows. `aid`: what the
    user travels with; a path answer tells them to keep using it (a guide dog user is not told to use a cane)."""
    cam = camera or CameraModel()
    q = normalize(question)
    near = sorted(_visible(tracks), key=_dist_of)
    side = _side(q)
    where = {"left": " on your left", "right": " on your right", "ahead": " ahead"}.get(side or "", "")
    note = f" {cane_note}" if cane_note else ""
    asked = _asked_object(q)
    if asked is not None:
        word, tags = asked
        hits = [t for t in near if (t.kind == "text" if not tags else _matches(t, tags)) and _on_side(t, side, cam)]
        if not hits:
            return f"I could not confirm {_article(word)} {word}{where} in the current camera view. I can miss things."
        if len(hits) == 1:
            return f"Yes, there's {_desc(hits[0], cam, units)}."
        return f"Yes, I see {len(hits)}: {_list(hits[:3], cam, units)}."
    if _PATH_WORDS.search(q):
        hits = [t for t in near if t.kind == "object" and t.path_prob > PATH_PROB and _dist_of(t) < 6.0]
        if hits:
            return f"Careful, there's {_list(hits[:2], cam, units)} in your path.{note}"
        if cane_note:
            return f"{cane_note} I can miss things, so {aid_advice(aid)}."
        return f"I did not detect anything in your path, but {NOT_CLEAR}. I can miss obstacles, so {aid_advice(aid)}."
    if side is not None:
        hits = [t for t in near if _on_side(t, side, cam)]
        if hits:
            lead = {"left": "On your left", "right": "On your right", "ahead": "Ahead of you"}[side]
            return f"{lead} there's {_list(hits[:3], cam, units)}.{note if side == 'ahead' else ''}"
        if side == "ahead":
            return f"I could not confirm anything ahead in the current camera view, but {NOT_CLEAR}.{note}"
        return f"I could not confirm anything{where} in the current camera view. I can miss things."
    if not near:
        return f"I could not confirm any objects in the current camera view. I can miss things.{note}"
    if len(near) == 1:
        return f"I see {_desc(near[0], cam, units)}.{note}"
    return f"I see {len(near)} things. The closest are {_list(near[:3], cam, units)}.{note}"


# ---- pause --------------------------------------------------------------------------------------------------------
class _PauseFilter:
    """The policy the pipeline sees: the controller's policy, with routine cues dropped while the router is paused.
    Every other attribute is the wrapped policy's (name, cfg, decision_interval, set_goal, last_observations ...)."""

    def __init__(self, inner, router: "CommandRouter"):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_router", router)

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __setattr__(self, k, v):
        setattr(self._inner, k, v)

    def decide(self, tracks, t, ch=None, *a, **kw):
        return self._router.filter_cues(self._inner.decide(tracks, t, ch, *a, **kw), tracks, t)

    def select(self, tracks, t, ctx=None, *a, **kw):
        return self._router.filter_cues(self._inner.select(tracks, t, ctx, *a, **kw), tracks, t)

    def reset(self):
        self._inner.reset()


# ---- the router ---------------------------------------------------------------------------------------------------
HELP_VOICE = ("Say SeeForMe, then a command: find a door, what's around me, read, is there a chair, repeat, pause, "
              "start, mute, more detail, less detail, help, or quit.")
HELP_TYPED = ("Type a command and press return: find door, what's around me, read, is there a chair, repeat, pause, "
              "start, mute, more detail, less detail, help, or quit.")
# the reply to a pause that names the camera ("camera off"): the terminal demo never turns the camera off while it runs
CAMERA_STAYS_ON = "The camera stays on so I can warn you about hazards. To turn it off, quit the demo."
_MODE_PHRASE = {"navigate": "Navigation mode.", "scene": "Scene mode.", "read": "Reading mode."}
_VERBOSITY_PHRASE = {"detailed": "More detail.", "quiet": "Less detail.", "balanced": "Normal detail."}
_THE = frozenset({"exit", "entrance", "elevator", "lift", "stairs", "escalator", "bathroom", "restroom", "toilet", "lobby"})


def _target_phrase(target: str, labels: Iterable[str]) -> str:
    if target in _THE:
        return f"the {target}"
    if " " not in target and not _has_digit(target) and (target in labels or target in ALIASES or target in NAV_WORDS) \
            and not target.endswith("s"):
        return f"{_article(target)} {target}"
    return target


class CommandRouter:
    """Executes Commands against the LiveController (goal, mode, summary), the Speaker (mute, repeat) and the camera,
    and returns the phrase to speak. run_live calls handle(), speaks the phrase, then after_reply()."""

    def __init__(self, controller, speaker, ocr: bool = True, labels: Iterable[str] = LABELS, voice: bool = False,
                 flip: Optional[Callable[[], str]] = None, log=None):
        self.ctl = controller
        self.speaker = speaker
        self.ocr = ocr
        self.labels = frozenset(labels)
        self.voice = voice                  # hands-free voice is on (help text)
        self.flip = flip                    # switches the camera and returns the phrase; None: not possible here
        self.log = log if log is not None else getattr(controller, "log", None)
        self.paused = False
        self.quit_requested = False
        self.last_message: Optional[str] = None
        self.counts: dict = {}
        self.held_while_paused = 0
        self.passed_while_paused = 0
        self.held: list = []
        self._mute_after_reply = False

    # ---- the pipeline side ----------------------------------------------------------------------------------
    def wrap(self, policy) -> _PauseFilter:
        """The policy to give the pipeline (wrap again whenever the controller's policy object changes)."""
        return _PauseFilter(policy, self)

    def filter_cues(self, cues, tracks, t: float) -> list:
        cues = list(cues or [])
        if not self.paused or not cues:
            return cues
        by_id = {tr.track_id: tr for tr in (tracks or [])}
        goal = list(getattr(self.ctl, "goal_tags", []) or [])
        kept = [c for c in cues if is_safety_cue(c, by_id.get(c.track_id), goal)]
        held = [c for c in cues if not any(c is k for k in kept)]
        self.held_while_paused += len(held)
        self.passed_while_paused += len(kept)
        if held:
            self.held.extend((t, c) for c in held)
            if self.log is not None:
                self.log.write("paused", t, held=[c.message for c in held])
        return kept

    def take_held(self) -> list:
        """(t, cue) of the routine cues held by the pause since the last call (run_live prints them)."""
        out, self.held = self.held, []
        return out

    def note_spoken(self, cue) -> None:
        """A scheduler cue was spoken: it is what "repeat" says."""
        if getattr(cue, "action", "speak") != "haptic" and getattr(cue, "message", ""):
            self.last_message = cue.message

    # ---- commands -------------------------------------------------------------------------------------------
    def handle(self, cmd: Command, tracks: Sequence = (), t: float = 0.0) -> str:
        """Execute the command; returns the confirmation or answer to speak ('' when a sound already acknowledged it)."""
        self.counts[cmd.intent] = self.counts.get(cmd.intent, 0) + 1
        tracks = list(tracks or [])
        reply = getattr(self, f"_do_{cmd.intent}", self._do_ask)(cmd, tracks, t)
        if self.log is not None:
            self.log.write("command", t, intent=cmd.intent, arg=cmd.arg, source=cmd.source, heard=cmd.text, said=reply)
        return reply

    def after_reply(self) -> None:
        """Apply what had to wait until the reply was spoken (mute: "Sound off." is still heard)."""
        if self._mute_after_reply:
            self._mute_after_reply = False
            self.speaker.enabled = False

    def _cam(self) -> CameraModel:
        return getattr(self.ctl, "camera", None) or CameraModel()

    def _units(self) -> str:
        return getattr(self.ctl, "units", "m")

    def _do_wake(self, cmd, tracks, t) -> str:
        chime = getattr(self.speaker, "chime", None)
        return "" if callable(chime) and chime() else "Yes?"

    def _do_start(self, cmd, tracks, t) -> str:
        if not self.paused and getattr(self.speaker, "enabled", True):
            return "I'm already on."
        self.paused = False
        self.speaker.enabled = True
        # short on purpose: a reply holds the audio channel, and the cane layer waits for it to end
        return "Resumed."

    def _do_pause(self, cmd, tracks, t) -> str:
        """"Pause" holds routine cues; the camera keeps running so safety cues go on. A pause that names the camera ("camera
        off", "stop the camera": arg "camera") is told so, also when already paused: the user cannot check a camera light,
        and the web app, which does turn its camera off for the same words, says that it is off."""
        if self.paused:
            if cmd.arg == "toggle":
                return self._do_start(cmd, tracks, t)
            return f"Already paused. {CAMERA_STAYS_ON}" if cmd.arg == "camera" else "Already paused."
        self.paused = True
        return f"Paused. {CAMERA_STAYS_ON}" if cmd.arg == "camera" else "Paused. I'll still warn you about hazards."

    def _do_quit(self, cmd, tracks, t) -> str:
        self.quit_requested = True
        return "Goodbye."

    def _do_find(self, cmd, tracks, t) -> str:
        target = (cmd.arg or "").strip()
        tags = goal_tags(target, self.labels) if target else []
        if not tags:
            return "Tell me what to look for, for example a chair or the exit."     # keep the current goal
        self.ctl.set_goal(tags)
        what = _target_phrase(target, self.labels)
        hit = next((tr for tr in sorted(_visible(tracks), key=_dist_of) if _matches(tr, tags)), None)
        if hit is not None:
            return f"Looking for {what}. I can see it now: {_desc(hit, self._cam(), self._units())}."
        reply = f"Looking for {what}. I'll tell you when I see it."
        if not any(g in self.labels for g in tags) and not self.ocr:
            reply += " Reading is off, so I can only find objects such as a chair or a person."
        return reply

    def _do_stop_find(self, cmd, tracks, t) -> str:
        had = bool(getattr(self.ctl, "goal_tags", None))
        self.ctl.set_goal([])
        return "Okay, I stopped looking." if had else "I wasn't looking for anything."

    def _do_read(self, cmd, tracks, t) -> str:
        msg = read_phrase(tracks, self.ocr)
        self.last_message = msg
        return msg

    def _cane_note(self, t: float, said_in: Optional[str] = None) -> str:
        """The controller's cane note ("" without one). said_in: the reply about to be spoken; when it contains the
        note, the cane holds that obstacle as said, so it does not cut the reply to say it again."""
        note = getattr(self.ctl, "cane_note", None)
        if not callable(note):
            return ""
        text = note(t, self._units()) or ""
        if text and said_in is not None and text in said_in:
            note(t, self._units(), commit=True)
        return text

    def _do_describe(self, cmd, tracks, t) -> str:
        msg = self.ctl.summary(tracks)
        note = self._cane_note(t)
        if note and note not in msg:
            msg = f"{note} {msg}"                 # the obstacle ahead first: the reply may be cut by a warning
            self._cane_note(t, said_in=msg)
        self.last_message = msg
        return msg

    def _do_repeat(self, cmd, tracks, t) -> str:
        return f"I said: {self.last_message}" if self.last_message else "I haven't said anything yet."

    def _do_mute(self, cmd, tracks, t) -> str:
        if cmd.arg == "toggle" and not getattr(self.speaker, "enabled", True):
            return self._do_unmute(cmd, tracks, t)
        self._mute_after_reply = True
        return "Sound off."

    def _do_unmute(self, cmd, tracks, t) -> str:
        self.speaker.enabled = True
        return "Sound on."

    def _do_help(self, cmd, tracks, t) -> str:
        return HELP_VOICE if self.voice else HELP_TYPED

    def _do_mode(self, cmd, tracks, t) -> str:
        ad = self.ctl.adapter
        if cmd.arg in (None, "auto"):
            ad.force_mode(None)
            act = getattr(getattr(self.ctl, "activity", None), "activity", None)
            if act is not None:
                ad.step(act, t, has_goal=bool(getattr(self.ctl, "goal_tags", None)))
            return f"Automatic mode, now {ad.mode}."
        ad.force_mode(cmd.arg)
        return _MODE_PHRASE.get(cmd.arg, f"Mode {cmd.arg}.")

    def _do_verbosity(self, cmd, tracks, t) -> str:
        cfg = getattr(getattr(self.ctl, "policy", None), "cfg", None)
        if cfg is None or not hasattr(cfg, "verbosity"):
            return "This scheduler has no detail setting."
        cfg.verbosity = cmd.arg or "balanced"
        return _VERBOSITY_PHRASE.get(cfg.verbosity, f"Detail {cfg.verbosity}.")

    def _do_flip(self, cmd, tracks, t) -> str:
        """"Flip camera" switches to the next camera. "Front camera" / "back camera" (arg "front" / "back") do the same:
        the terminal demo cycles camera indices and cannot tell which way a camera faces, so the reply says so instead
        of naming a side."""
        if self.flip is None:
            return "Switching cameras is only in the web app."
        reply = self.flip()
        return f"This demo cannot tell which way a camera faces. {reply}" if cmd.arg else reply

    def _do_ask(self, cmd, tracks, t) -> str:
        q = cmd.arg or cmd.text
        if not is_question(q):        # a mis-heard command ("see for me newt"): say so instead of describing the scene
            return ("Sorry, I didn't catch that. Say SeeForMe, help, for the commands." if self.voice
                    else "Sorry, I didn't understand. Type help for the commands.")
        msg = answer(q, tracks, self._cam(), self._units(), cane_note=self._cane_note(t),
                     aid=getattr(self.ctl, "aid", None))
        self._cane_note(t, said_in=msg)
        self.last_message = msg
        return msg

    def report(self) -> List[str]:
        lines = []
        if self.counts:
            lines.append("commands: " + ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())))
        if self.held_while_paused or self.passed_while_paused:
            lines.append(f"paused: {self.held_while_paused} routine cue(s) held, {self.passed_while_paused} safety cue(s) said")
        return lines
