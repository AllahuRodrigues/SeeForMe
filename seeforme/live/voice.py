"""Voice commands for the terminal demo: "SeeForMe, find a door", "SeeForMe, pause", "SeeForMe, what's around me".

A blind user cannot find the keys of an OpenCV window, so the live demo takes spoken (or typed) commands instead.

    parse_command(text, require_wake=True) -> Command(intent, arg) | None
        Hands-free (require_wake=True): the wake phrase "see for me" is required, also as "seeforme", "hey see for me"
        and the recogniser's usual mis-hearings ("sea for me", "c for me", "see four me", "see 4 me"); the command is
        the text after the LAST wake phrase, so chatter before it is ignored, and the wake phrase alone is intent
        "wake". Push-to-talk and typed commands (require_wake=False): the wake phrase is optional. Text with no
        command words is intent "ask" (arg = the text), answered by the rule-based answer() in commands.py. Polite
        openings ("please", "can you", "tell me", "do you know", "find out") are dropped first. Then, in this order
        (_core_command): stop_find; a FIND verb ("find", "where is", "where can I find", "how do I get to", "take me
        to", "get me" ...); "where X is"; "show me ..." (a describe, read or help phrase after it keeps its meaning,
        so "show me what you see" is a describe and "show me what it says" a read; a question stays a question; any
        other rest is a FIND target, "show me the exit"); the mode and detail phrases, which must be the whole command
        ("navigation mode", "switch to reading mode", "give me more detail"), so "find the scene mode button" or "find
        the balanced rock" is a FIND; a named camera side ("front camera", "switch to the back camera": flip with arg
        "front" or "back"; "flip camera" has none); the fixed phrases (pause, start, read, describe ...); a bare
        "exit". A FIND target is lowercase, without a leading "me" ("find me a chair"), "the way to", articles or
        "nearest" / "closest", with spoken numbers as digits ("room two oh four" -> "room 204", "room oh four" ->
        "room 04"); "the way out" is the exit. tests/data/voice_commands.json holds the vectors the web parser shares.

    VoiceListener(speaker)
        Recognisers run in background threads and call hear(); the main loop calls poll() once per frame and gets the
        Commands. Hands-free state machine for microphone backends: a bare wake word opens a follow-up window
        (follow_up_s = 5 s of time the user can actually be heard) in which one command needs no wake word; anything
        else without the wake phrase is ignored. Echo suppression: while the Speaker is speaking, and echo_margin_s
        (0.4 s) after, microphone results are dropped (the Vosk backend does not even feed that audio to the
        recogniser), so the demo never obeys its own voice. Typed and scripted commands are never suppressed.

    Backends
        VoskBackend      offline recognition (vosk + sounddevice, imported lazily; model in models/vosk/ or
                         $SEEFORME_VOSK_MODEL; 16 kHz mono int16, streaming KaldiRecognizer, final results only).
                         recogniser "snap" (default): the open vocabulary, and an utterance that does not read as a
                         command is decoded again against a grammar of the command vocabulary ("see for me mood" ->
                         "mute"); "open", "grammar" and "hybrid" (grammar first) too (scripts/eval_voice.py compares
                         them, results/live/voice_eval.md). source= a WAV file or samples instead of the mic
        TypedBackend     lines typed in the terminal or piped into stdin (wake word optional); a single letter is the
                         window key of the same name (q quit, h help, w describe, r read, p pause / resume, s sound
                         off / on, f: the next line is a FIND target)
        FakeBackend      injected recognition results, for tests
        ScriptedCommands "1.0:find door;3.0:describe": fired at video / stream time, for headless runs
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

INTENTS = ("wake", "start", "pause", "quit", "find", "stop_find", "read", "describe", "repeat", "mute", "unmute",
           "help", "mode", "verbosity", "flip", "ask")
ROOT = Path(__file__).resolve().parents[2]
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
DEFAULT_VOSK_MODEL = ROOT / "models" / "vosk" / VOSK_MODEL_NAME


@dataclass(frozen=True)
class Command:
    """One recognised command. Equality is (intent, arg): what was heard and where it came from are context."""
    intent: str
    arg: Optional[str] = None
    text: str = field(default="", compare=False)       # the words as heard or typed
    source: str = field(default="", compare=False)     # vosk | typed | script | key | fake ...

    def describe(self) -> str:
        return self.intent + (f" {self.arg}" if self.arg else "")


# ---- parsing ------------------------------------------------------------------------------------------------------
# "see for me" and what small recognisers make of it; "hey" / "ok" in front belong to the wake phrase
_WAKE = re.compile(r"\b(?:(?:hey|ok|okay)\s+)?(?:see|sea|c)\s*(?:for|four|4)\s*me\b")
# polite and indirect openings, dropped before the rules: "can you tell me what this says" is "what this says"
_FILLER = re.compile(r"^(?:(?:please|ok|okay|um|uh|so|and|now|hey|can you|could you|would you|will you|tell me|do you know|"
                     r"find out)\s+)+")
_TRAILING = re.compile(r"(?:\s+(?:please|now|thanks|thank you))+$")

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ZERO = ("oh", "o")                      # zero only inside a number: "two oh four"
# how a recogniser spells digits it hears as words ("room two oh four" -> "room to oh for"); read as digits only in a
# run with an inner "oh", where they cannot be the preposition ("a table for two" stays "table for 2")
_HOMOPHONES = {"to": "two", "too": "two", "for": "four", "fore": "four", "won": "one", "ate": "eight"}


def normalize(text: str) -> str:
    """NFC, lowercase, punctuation to spaces, apostrophes kept inside words ("what's"), single spaces. NFC first, as the
    web parser does: a decomposed "café" (an e and a combining accent, as some keyboards and recognisers write it) is
    the same word as the composed one, and the accent is not dropped as punctuation."""
    s = unicodedata.normalize("NFC", text or "").lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s']", " ", s).replace("_", " ")
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)
    return " ".join(s.split())


def _is_num(w: str) -> bool:
    # isdecimal: digits of any script (Unicode Nd, the web parser's \p{Nd}); isdigit would also take "²" and "①"
    return w in _UNITS or w in _TEENS or w in _TENS or w == "hundred" or w.isdecimal()


def _number_run(ws: Sequence[str]) -> str:
    """Digits for a run of number words: digit by digit ("two oh four" -> 204), tens with a unit as one number
    ("twenty one" -> 21), teens ("twelve" -> 12), "X hundred [and] Y"."""
    out: List[str] = []
    i = 0

    def small(j: int) -> Tuple[Optional[int], int]:     # a value below 100 starting at j, and the next index
        w = ws[j] if j < len(ws) else ""
        if w in _TENS:
            if j + 1 < len(ws) and ws[j + 1] in _UNITS and _UNITS[ws[j + 1]]:
                return _TENS[w] + _UNITS[ws[j + 1]], j + 2
            return _TENS[w], j + 1
        if w in _TEENS:
            return _TEENS[w], j + 1
        if w in _UNITS:
            return _UNITS[w], j + 1
        return None, j

    while i < len(ws):
        w = ws[i]
        if w == "hundred" or (w in _UNITS and i + 1 < len(ws) and ws[i + 1] == "hundred"):
            base = 100 * (_UNITS[w] if w in _UNITS else 1)
            i += 1 if w == "hundred" else 2
            if i < len(ws) and ws[i] == "and":
                i += 1
            rest, i = small(i)
            out.append(str(base + (rest or 0)))
            continue
        if w in _ZERO:
            out.append("0"); i += 1; continue
        if w.isdecimal():
            out.append(w); i += 1; continue
        v, i2 = small(i)
        if v is None:                     # not a number word (cannot happen for a run, but never loop forever)
            out.append(w); i += 1; continue
        out.append(str(v)); i = i2
    return "".join(out)


def words_to_digits(text: str) -> str:
    """Spoken numbers as digits: "room two oh four" -> "room 204", "gate twenty one" -> "gate 21"."""
    ws = text.split()
    out: List[str] = []
    i = 0
    while i < len(ws):
        j = i
        while j < len(ws) and (_is_num(ws[j]) or ws[j] in _ZERO or ws[j] in _HOMOPHONES):
            j += 1
        run = ws[i:j]
        if len(run) >= 2 and any(w in _ZERO for w in run[1:]) and run[0] not in _ZERO:
            out.append(_number_run([_HOMOPHONES.get(w, w) for w in run])); i = j
            continue
        if ws[i] in _ZERO and i + 1 < len(ws) and _is_num(ws[i + 1]):
            j = i + 1                     # a leading "oh" before a number word: "room oh four" -> "room 04"
            while j < len(ws) and (_is_num(ws[j]) or ws[j] in _ZERO):
                j += 1
            out.append(_number_run(ws[i:j])); i = j
            continue
        if not _is_num(ws[i]):
            out.append(ws[i]); i += 1; continue
        j = i
        while j < len(ws) and (_is_num(ws[j]) or (ws[j] in _ZERO and j > i)
                               or (ws[j] == "and" and j > i and ws[j - 1] == "hundred" and j + 1 < len(ws) and _is_num(ws[j + 1]))):
            j += 1
        out.append(_number_run(ws[i:j])); i = j
    return " ".join(out)


_ARG_DROP = {"a", "an", "the", "nearest", "closest", "please"}


def clean_target(text: str) -> str:
    """A FIND target: no articles, "nearest" / "closest" or "please", spoken numbers as digits."""
    s = " ".join(w for w in normalize(text).split() if w not in _ARG_DROP)
    return words_to_digits(re.sub(r"(?:^|\s+)for me$", "", s).strip())


# "get me to" before "get me" ("get me to the exit" is the exit, "get me a chair" a chair); "where are we" is a describe
_FIND = re.compile(r"^(?:help me |i want to |i need to |i'd like to |i would like to |start |keep )?"
                   r"(?:find|look for|looking for|search for|locate|where is|where's|wheres|where are(?! we$)|take me to|"
                   r"guide me to|lead me to|bring me to|navigate to|get me to|get me|get to|i'm looking for|i am looking for|"
                   r"where (?:can|could|do|would|should|will|might) (?:i|we) find|where (?:i|we) (?:can|could) find|"
                   r"how (?:do|can|could|would|should) (?:i|we) (?:get to|find|reach)|how to (?:get to|find|reach)|the way to)"
                   r"(?:\s+(.*))?$")
# "where the exit is" (after "show me", "tell me", "do you know"): a FIND too
_FIND_WHERE = re.compile(r"^where (?!we are$)(.+) (?:is|are)$")
# "show me ...": a describe or read phrase after it keeps its meaning, a question stays a question, anything else is a
# FIND target ("show me the exit"); see _show
_SHOW = re.compile(r"^show me(?: (.+))?$")
_SHOW_DESCRIBE = re.compile(r"^(?:around(?: (?:here|me|us))?|(?:my |the )?surroundings|everything|the (?:scene|room|area|view))$")
_SUBJ = r"(?:it|this|that|(?:this|that|the) \w+(?: \w+)?)"      # what a sign question is about: "it", "the sign", "this menu"
_SHOW_READ = re.compile(rf"^(?:the |this |that |any )?(?:text|writing|words)(?: (?:here|there|on {_SUBJ}))?$")
_QUESTION_WORD = re.compile(r"^(?:what|what's|whats|where|where's|wheres|how|how's|which|who|who's|why|when|whether|if|is|"
                            r"are|was|were|can|could|do|does|did|will|would|should)(?: |$)")
_STOP_FIND = re.compile(r"^(?:(?:stop|quit) (?:looking|searching|finding)\b.*|cancel\b.*|never ?mind\b.*|forget it\b.*|"
                        r"clear (?:the )?(?:find|goal|search)|stop find)$")
# mode and detail phrases are the whole command (a FIND target may contain them: "find the scene mode button")
_MODE = re.compile(r"^(?:(?:switch|change|go|turn|set|put|use|enter|back)(?: it| me| back)?(?: (?:to|into|on|in))? )?(?:the )?"
                   r"(navigation|navigate|navigating|walking|walk|scene|read|reading|text|auto|automatic) mode(?: on)?$")
_MODE_ARG = {"navigation": "navigate", "navigate": "navigate", "navigating": "navigate", "walking": "navigate",
             "walk": "navigate", "scene": "scene", "read": "read", "reading": "read", "text": "read",
             "auto": "auto", "automatic": "auto"}
_VERB_PREFIX = r"^(?:(?:give me|i want|i'd like|i would like|use|switch to|go to|set|turn on|with) )?"
_VERBOSITY = tuple((level, re.compile(_VERB_PREFIX + f"(?:{pat})(?: (?:please|now|mode))?$")) for level, pat in (
    ("detailed", r"more (?:detail|details|detailed|information|info)|more verbose|be more detailed|detailed"),
    ("quiet", r"less (?:detail|details|detailed|information|info|talking)|fewer details|talk less|be brief|briefer"),
    ("balanced", r"(?:normal|default|medium|standard|usual) (?:detail|details|verbosity)|balanced"),
))
# a named camera side: flip with arg "front" or "back" ("flip camera", "other camera": flip without one)
_SIDE = r"(front|back|rear|selfie)(?: facing)?"
_CAMERA_SIDE = re.compile(
    rf"^(?:(?:(?:use|switch to|change to|go to|flip to|swap to|turn to|turn on|open|select|try)(?: the)? |the )?{_SIDE} (?:camera|cam)"
    rf"|(?:flip|switch|change|swap|turn|toggle) (?:the )?(?:camera|cam) (?:to|around to) (?:the )?{_SIDE}(?: (?:one|camera|cam|side))?"
    rf"|(?:flip|switch|swap) to (?:the )?{_SIDE}(?: one| side)?)$")
_SIDE_ARG = {"front": "front", "selfie": "front", "back": "back", "rear": "back"}
_EXACT = (   # (intent, pattern on the body without fillers); first match wins
    ("pause", r"(?:stop|pause|paws|wait|hold on|hang on)(?: (?:talking|announcing|announcements|it|that|for now|a (?:moment|second|sec|minute)))*|"
              r"(?:stop|pause|turn off|switch off|shut off|close)(?: the)? (?:camera|cam)|(?:turn|switch|shut) (?:the )?(?:camera|cam) off|"
              r"(?:the )?(?:camera|cam) off"),
    ("start", r"(?:start|resume|continue|unpause|restart|keep going|go on|carry on)(?: (?:the|camera|announcing|announcements|talking|again|it|up|listening))*|"
              r"(?:turn on|switch on|open)(?: the)? (?:camera|cam)|(?:turn|switch) (?:the )?(?:camera|cam) on|(?:the )?(?:camera|cam) on"),
    ("quit", r"(?:quit|goodbye|good bye|bye|bye bye|shut down|shutdown|(?:exit|quit|close) (?:the )?(?:app|program|demo))"),
    ("read", rf"(?:read(?: .+)?|what (?:does|do) {_SUBJ} say|what {_SUBJ} says|(?:what's|what is|whats) written(?: (?:here|there|on {_SUBJ}))?)"),
    ("describe", r"(?:(?:what's|what is|whats) (?:around(?: me| here| us)?|here|near me|nearby)|what (?:do|can) you see(?: now| here)?|"
                 r"what you (?:can )?see|describe(?: .+)?|look around|where am i|where are we|where i am|where we are)"),
    ("repeat", r"(?:repeat(?: (?:that|it|again))*|say (?:that|it) again|say again|again|come again|pardon(?: me)?|what did you say|what was that|one more time)"),
    ("unmute", r"(?:unmute|un mute|sound on|turn (?:the )?sound on|turn on (?:the )?sound|voice on|unmute (?:sound|voice|audio))"),
    ("mute", r"(?:mute|be quiet|quiet|silence|hush|shut up|sound off|turn (?:the )?sound off|turn off (?:the )?sound|voice off|mute (?:sound|voice|audio))"),
    ("help", r"(?:help(?: me| please| (?:for|with) (?:the )?commands)?|what can i say|what can you do|(?:the )?commands|"
             r"list (?:the )?commands|what are the commands)"),
    ("flip", r"(?:flip|(?:flip|switch|change|swap|toggle) (?:the )?(?:camera|cameras|cam)|(?:next|other) camera|use the other camera)"),
)
_EXACT_RE = tuple((intent, re.compile(f"^(?:{pat})$")) for intent, pat in _EXACT)
_BARE_EXIT = re.compile(r"^(?:exit|exits|way out)$")
_SAME_GOAL = {"way out": "exit", "out of here": "exit"}


def _find_target(raw: str) -> Optional[str]:
    """A FIND target from the words after the verb: clean_target without a leading "me" ("find me a chair") or "the way
    to" ("show me the way to the exit"); "the way out" is the exit. None when nothing is left ("find me")."""
    s = re.sub(r"^me(?: |$)", "", raw)
    s = re.sub(r"^(?:the |a )?way to ", "", s)
    t = clean_target(s)
    return _SAME_GOAL.get(t, t) or None


def _strip(text: str) -> str:
    return _TRAILING.sub("", _FILLER.sub("", text)).strip() or text


def _show(rest: str, body: str) -> Command:
    """"show me <rest>": describe ("show me around", "show me what you see"), read ("show me the text", "show me what it
    says") or help when <rest> reads as one; a FIND when <rest> is one ("show me where the exit is"); a question stays a
    question ("show me what's on my left"); anything else is a FIND target ("show me the exit", "show me a chair")."""
    if not rest:
        return Command("ask", body)
    if _SHOW_DESCRIBE.match(rest):
        return Command("describe")
    if _SHOW_READ.match(rest):
        return Command("read")
    cmd = _core_command(_strip(rest), rest, show=False)
    if cmd.intent in ("find", "describe", "read", "help"):
        return cmd
    if cmd.intent == "ask" and not _QUESTION_WORD.match(rest):
        return Command("find", _find_target(rest))
    return Command("ask", body)


def _core_command(core: str, body: str, show: bool = True) -> Command:
    """Rules in order: stop_find; a FIND verb (it may name mode or detail words: "find the scene mode button"); "where X
    is"; "show me ..." (_show); the whole-command mode and detail phrases; a named camera side; the fixed phrases; a
    bare "exit"; else a question (arg = the body as heard)."""
    if _STOP_FIND.match(core):
        return Command("stop_find")
    m = _FIND.match(core) or _FIND_WHERE.match(core)
    if m:
        return Command("find", _find_target(m.group(1) or ""))
    m = _SHOW.match(core) if show else None
    if m:
        return _show(m.group(1) or "", body)
    m = _MODE.match(core)
    if m:
        return Command("mode", _MODE_ARG[m.group(1)])
    for level, rx in _VERBOSITY:
        if rx.match(core):
            return Command("verbosity", level)
    m = _CAMERA_SIDE.match(core)
    if m:
        return Command("flip", _SIDE_ARG[next(g for g in m.groups() if g)])
    for intent, rx in _EXACT_RE:
        if rx.match(core):
            return Command(intent)
    if _BARE_EXIT.match(core):
        return Command("find", "exit")
    return Command("ask", body)


def _body_command(body: str) -> Optional[Command]:
    """The command in the words after the wake phrase (normalised); None only for empty text."""
    if not body:
        return None
    return _core_command(_strip(body), body)


def parse_command(text: str, require_wake: bool = True) -> Optional[Command]:
    """Parse one recognised utterance or typed line (see the module docstring). None: ignore it."""
    s = normalize(text)
    if not s:
        return None
    wakes = list(_WAKE.finditer(s))
    if wakes:
        body = s[wakes[-1].end():].strip()
        return _body_command(body) if body else Command("wake")
    if require_wake:
        return None
    return _body_command(s)


# ---- scripted commands --------------------------------------------------------------------------------------------
def parse_script(spec: str) -> List[Tuple[float, str]]:
    """"1.0:find door;3.0:describe" -> [(1.0, "find door"), (3.0, "describe")], sorted by time. Each command must
    parse (wake word optional), so a typo fails at start-up instead of silently doing nothing."""
    items: List[Tuple[float, str]] = []
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        t_str, sep, text = part.partition(":")
        try:
            t = float(t_str)
        except ValueError:
            raise ValueError(f"scripted command {part!r}: expected TIME:COMMAND, e.g. 2.5:find a door") from None
        if not sep or parse_command(text, require_wake=False) is None:
            raise ValueError(f"scripted command {part!r}: no command after the time")
        items.append((t, text.strip()))
    return sorted(items, key=lambda x: x[0])


class ScriptedCommands:
    """Commands that fire at a stream time (the video's own clock in a replay): headless runs exercise voice commands."""
    name = "script"

    def __init__(self, items: Iterable[Tuple[float, str]]):
        self.items = sorted(items, key=lambda x: x[0])
        self._i = 0

    def due(self, t: float) -> List[str]:
        out = []
        while self._i < len(self.items) and self.items[self._i][0] <= t + 1e-9:
            out.append(self.items[self._i][1]); self._i += 1
        return out

    @property
    def remaining(self) -> int:
        return len(self.items) - self._i


# ---- the listener -------------------------------------------------------------------------------------------------
class VoiceListener:
    """Turns recognised text from any backend into Commands on a queue the main loop drains with poll()."""

    def __init__(self, speaker=None, follow_up_s: float = 5.0, echo_margin_s: float = 0.4,
                 clock: Callable[[], float] = time.monotonic, max_follow_up_s: float = 15.0):
        self.speaker = speaker
        self.follow_up_s = follow_up_s
        self.echo_margin_s = echo_margin_s
        self.max_follow_up_s = max_follow_up_s     # a follow-up window never outlives the wake word by more than this
        self.clock = clock
        self.backends: list = []
        self.script: Optional[ScriptedCommands] = None
        self._q: "queue.Queue[Command]" = queue.Queue()
        self._notices: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self._follow_until = -math.inf
        self._wake_t = -math.inf
        self._seen_busy_at = -math.inf              # for speakers that only have busy()
        self.ignored_echo = 0                       # microphone results dropped because the demo was speaking
        self.ignored_no_wake = 0                    # microphone results without the wake phrase, outside a window

    # ---- wiring ---------------------------------------------------------------------------------------------
    def add(self, backend):
        self.backends.append(backend)
        return backend

    def set_script(self, items: Iterable[Tuple[float, str]]) -> None:
        self.script = ScriptedCommands(items)

    def start(self) -> None:
        for b in self.backends:
            b.start(self)

    def stop(self, timeout: float = 0.0) -> None:
        """Stop every backend; with a timeout, wait for their threads (the Vosk model is then freed while the
        interpreter is still whole, not by exit-time destructors)."""
        for b in self.backends:
            stop = getattr(b, "stop", None)
            if callable(stop):
                stop()
        if timeout > 0:
            deadline = time.monotonic() + timeout
            for b in self.backends:
                join = getattr(b, "join", None)
                if callable(join):
                    join(max(0.0, deadline - time.monotonic()))

    def notify(self, message: str) -> None:
        """A backend's one-line status for the user (e.g. the microphone could not be opened); run_live says it."""
        self._notices.put(message)

    def notices(self) -> List[str]:
        out = []
        while True:
            try:
                out.append(self._notices.get_nowait())
            except queue.Empty:
                return out

    @property
    def voice_running(self) -> bool:
        return any(getattr(b, "audio", False) and getattr(b, "running", False) for b in self.backends)

    @property
    def typed_running(self) -> bool:
        return any(not getattr(b, "audio", False) and getattr(b, "running", False) for b in self.backends)

    # ---- echo suppression -----------------------------------------------------------------------------------
    def _speaker_busy(self) -> bool:
        sp = self.speaker
        if sp is None:
            return False
        f = getattr(sp, "is_speaking", None) or getattr(sp, "busy", None)
        try:
            return bool(f()) if callable(f) else False
        except Exception:
            return False

    def echo_blocked(self, at: Optional[float] = None) -> bool:
        """True while the speaker is speaking or ended less than echo_margin_s before `at` (listener clock)."""
        now = self.clock()
        at = now if at is None else at
        if self._speaker_busy():
            self._seen_busy_at = now
            return True
        ended = getattr(self.speaker, "speech_ended_at", None) if self.speaker is not None else None
        ended = ended if isinstance(ended, (int, float)) else -math.inf
        return at < max(ended, self._seen_busy_at) + self.echo_margin_s

    # ---- input ----------------------------------------------------------------------------------------------
    def listen_once(self, at: Optional[float] = None) -> None:
        """Push-to-talk: the next utterance within follow_up_s is a command without the wake word."""
        at = self.clock() if at is None else at
        with self._lock:
            self._wake_t = at
            self._follow_until = at + self.follow_up_s

    def in_window(self, at: Optional[float] = None) -> bool:
        at = self.clock() if at is None else at
        return at <= self._follow_until

    def put(self, cmd: Command) -> None:
        self._q.put(cmd)

    def hear(self, text: str, source: str = "voice", audio: bool = True, at: Optional[float] = None) -> Optional[Command]:
        """A final recognition result (audio=True: microphone, hands-free rules and echo suppression) or a typed line
        (audio=False: wake word optional). Returns the Command queued, or None when the text is ignored."""
        at = self.clock() if at is None else at
        with self._lock:
            if audio:
                if self.echo_blocked(at):
                    self.ignored_echo += 1
                    return None
                cmd = parse_command(text, require_wake=True)
                if cmd is None and self.in_window(at):
                    cmd = parse_command(text, require_wake=False)
                if cmd is None:
                    self.ignored_no_wake += int(bool(normalize(text)))
                    return None
                if cmd.intent == "wake":
                    self._wake_t = at
                    self._follow_until = at + self.follow_up_s
                else:
                    self._follow_until = -math.inf
            else:
                cmd = parse_command(text, require_wake=False)
                if cmd is None:
                    return None
            cmd = replace(cmd, text=text.strip(), source=source)
        self._q.put(cmd)
        return cmd

    def poll(self, stream_t: Optional[float] = None) -> List[Command]:
        """Every frame: the commands heard or typed since the last call, plus scripted ones due at stream_t. While the
        demo speaks during a follow-up window the window is held open (the user cannot be heard then), at most
        max_follow_up_s after the wake word."""
        now = self.clock()
        with self._lock:
            if self._follow_until >= now and self.echo_blocked(now):
                self._follow_until = min(max(self._follow_until, now + self.echo_margin_s + self.follow_up_s),
                                         self._wake_t + self.max_follow_up_s)
        if self.script is not None and stream_t is not None:
            for text in self.script.due(stream_t):
                cmd = parse_command(text, require_wake=False)
                if cmd is not None:
                    self._q.put(replace(cmd, text=text, source="script"))
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out

    def state(self, at: Optional[float] = None) -> str:
        """listening (a follow-up window is open) | wake (waiting for "see for me") | typed (no microphone) | off."""
        if self.voice_running:
            return "listening" if self.in_window(at) else "wake"
        return "typed" if self.typed_running else "off"


# ---- backends -----------------------------------------------------------------------------------------------------
class FakeBackend:
    """Injectable recogniser for tests: feed(text) is a final result heard now (or at `at`)."""
    name = "fake"

    def __init__(self, audio: bool = True):
        self.audio = audio
        self.running = False
        self.listener: Optional[VoiceListener] = None

    def start(self, listener: VoiceListener) -> None:
        self.listener, self.running = listener, True

    def stop(self) -> None:
        self.running = False

    def feed(self, text: str, at: Optional[float] = None) -> Optional[Command]:
        assert self.listener is not None, "start the listener first"
        return self.listener.hear(text, source=self.name, audio=self.audio, at=at)


# a single typed letter is the window key of the same name (the terminal is where a typed "q" goes)
KEY_LINES = {"q": Command("quit"), "h": Command("help"), "w": Command("describe"), "r": Command("read"),
             "p": Command("pause", "toggle"), "s": Command("mute", "toggle")}


class TypedBackend:
    """Commands typed in the terminal (or piped into stdin), one per line, read by a daemon thread (the wake word is
    optional). A single letter is a window key (KEY_LINES). The f key, typed or in the window, calls expect_goal(): the
    next line is a FIND target (an empty line clears it)."""
    name = "typed"
    audio = False

    def __init__(self, stream=None):
        self.stream = stream
        self.running = False
        self._goal_next = False
        self.listener: Optional[VoiceListener] = None

    def expect_goal(self) -> None:
        self._goal_next = True

    def start(self, listener: VoiceListener) -> None:
        self.listener, self.running = listener, True
        threading.Thread(target=self._run, name="seeforme-typed", daemon=True).start()

    def stop(self) -> None:
        self.running = False

    def line(self, text: str) -> Optional[Command]:
        """Handle one typed line (the thread calls this; tests may too)."""
        assert self.listener is not None
        if self._goal_next:
            self._goal_next = False
            target = clean_target(text)
            cmd = Command("find", target, text=text.strip(), source=self.name) if target else Command("stop_find", text=text.strip(), source=self.name)
            self.listener.put(cmd)
            return cmd
        key = normalize(text)
        if key == "f":
            self.expect_goal()
            self.listener.notify("Type what to find and press return.")
            return None
        if key in KEY_LINES:
            cmd = replace(KEY_LINES[key], text=text.strip(), source=self.name)
            self.listener.put(cmd)
            return cmd
        return self.listener.hear(text, source=self.name, audio=False)

    def _run(self) -> None:
        stream = self.stream if self.stream is not None else sys.stdin
        try:
            tty = bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            tty = False
        try:
            while self.running:
                raw = stream.readline()
                if raw == "":
                    if not tty:
                        break                # a pipe or a file ended
                    time.sleep(0.2)          # ctrl-D in a terminal: an end of file that is not the end; keep listening
                    continue
                self.line(raw)
        except (OSError, ValueError):       # stdin closed under us
            pass
        self.running = False


def input_end_stops_run(headless: bool, video: bool, frames: int, stdin_tty: bool, sources: Sequence[str]) -> bool:
    """Should the end of piped stdin end the run? Only for a headless camera run (a camera never ends, and a headless
    run has no window key) without a --frames limit whose sole command source is stdin from a pipe or a file: then
    nothing else could ever stop it (before, it ran until Ctrl-C after the last command). A replayed video ends by
    itself, a terminal's ctrl-D is not the end, and a microphone or a script may still say "quit"."""
    return headless and not video and not frames and not stdin_tty and list(sources) == ["typed"]


def vosk_model_path(path: Optional[str] = None) -> Path:
    return Path(path or os.environ.get("SEEFORME_VOSK_MODEL") or DEFAULT_VOSK_MODEL)


def vosk_status(path: Optional[str] = None) -> Tuple[bool, str]:
    """(usable, why not): vosk and sounddevice importable and the model directory present. Imports nothing heavy."""
    import importlib.util
    missing = [m for m in ("vosk", "sounddevice") if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"{' and '.join(missing)} not installed"
    p = vosk_model_path(path)
    if not (p / "am").is_dir() and not (p / "conf").is_dir():
        return False, f"no Vosk model at {p}"
    return True, str(p)


# ---- the Vosk recogniser: vocabulary and decoding modes -----------------------------------------------------------
RECOGNISERS = ("open", "grammar", "hybrid", "snap")
DEFAULT_RECOGNISER = "snap"          # chosen by scripts/eval_voice.py (results/live/voice_eval.md): most commands right
                                     # without more false wakes than the open vocabulary
# FIND targets the grammar knows besides the detector's labels (commands.COCO, commands.NAV_WORDS): places and things
# blind travellers ask for (an open-vocabulary target outside this list is heard by the open recogniser in hybrid mode)
GRAMMAR_TARGETS = ("door", "exit", "entrance", "stairs", "steps", "staircase", "elevator", "lift", "escalator", "bathroom",
                   "restroom", "toilet", "room", "gate", "counter", "cashier", "checkout", "desk", "seat", "table",
                   "bench", "chair", "sign", "bus stop", "crosswalk", "crossing", "platform", "train", "bus", "taxi",
                   "car", "person", "dog", "cup", "bottle", "phone", "keys", "bag", "backpack", "laptop", "window",
                   "wall", "pole", "curb", "path", "hallway", "corridor", "lobby", "office", "kitchen", "store", "shop")
_NUMBER_WORDS = ("zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                 "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred")


def grammar_phrases() -> List[str]:
    """The phrase list of a Vosk grammar recogniser: the wake phrase and every command the parser knows, written out
    from its own patterns (not from the test vectors), FIND / question templates over GRAMMAR_TARGETS and the detector's
    labels, and "[unk]" so that other words are not forced onto these. Vosk turns the list into a bigram language
    model, so word orders beyond these phrases stay possible."""
    from .commands import COCO, NAV_WORDS          # (commands imports this module)
    wake = ["see for me", "hey see for me", "okay see for me"]
    cmds = ["stop", "pause", "wait", "hold on", "stop talking", "start", "resume", "continue", "keep going", "start again",
            "quit", "goodbye", "exit the app", "shut down", "read", "read the sign", "read this", "read it to me",
            "what does it say", "what does the sign say", "what's around me", "what is around me", "what's here",
            "what's nearby", "what do you see", "what can you see", "describe", "describe the scene", "look around",
            "where am i", "repeat", "repeat that", "say that again", "say again", "what did you say", "pardon",
            "one more time", "mute", "be quiet", "quiet", "silence", "sound off", "turn the sound off",
            "un mute", "sound on", "turn the sound on", "voice on",       # ("unmute" is not in the small model)
            "help", "help me", "what can i say", "what can you do", "commands",
            "flip camera", "switch camera", "other camera", "stop looking", "stop searching", "cancel", "never mind",
            "forget it", "more detail", "more details", "less detail", "fewer details", "talk less", "be brief",
            "normal detail", "balanced", "detailed",
            # everyday phrasings: show me, written, camera sides, camera off / on, where are we
            "show me around", "show me what's around me", "show me what you see", "tell me what you see",
            "show me the text", "show me what it says", "show me where the exit is", "show me the way out",
            "what's written", "what is written here", "what's written on the sign", "what does this sign say",
            "read me the sign", "describe what's around me", "where are we", "how do i get to the exit",
            "do you know where the exit is", "front camera", "back camera", "rear camera", "use the front camera",
            "use the back camera", "switch to the front camera", "switch to the back camera",   # ("selfie" is not in
            "stop the camera", "camera off", "turn off the camera", "turn the camera off",        # the small model)
            "camera on", "turn on the camera", "turn the camera on"]
    cmds += [f"{m} mode" for m in ("navigation", "walking", "scene", "reading", "read", "text", "auto", "automatic")]
    cmds += [f"switch to {m} mode" for m in ("navigation", "scene", "reading", "auto")]
    targets = sorted(set(GRAMMAR_TARGETS) | set(NAV_WORDS) | set(COCO))
    for tg in targets:
        cmds += [f"find the {tg}", f"find a {tg}", f"where is the {tg}", f"where's the nearest {tg}", f"look for a {tg}",
                 f"take me to the {tg}", f"is there a {tg}", f"is there a {tg} ahead", f"find me a {tg}",
                 f"show me the {tg}", f"where can i find a {tg}", f"where do i find the {tg}", f"get me a {tg}"]
    cmds += ["find room " + w for w in ("two oh four", "one oh one", "three twenty", "four hundred")]
    cmds += ["is the path clear", "is it safe to walk", "what's in front of me", "what is on my left",
             "what is on my right", "what's on my left", "what's on my right", "how far is it", "which way",
             "is anyone there", "is someone coming", "can i cross", _NUMBER_WORDS]
    return wake + [f"see for me {c}" for c in cmds] + cmds + ["[unk]"]


def audio_source_blocks(source, samplerate: int = 16000, block_s: float = 0.25, lead_s: float = 0.5,
                        tail_s: float = 1.0) -> List[bytes]:
    """int16 mono byte blocks from a WAV file (16-bit mono at `samplerate`) or an array of samples (int16, or float in
    -1..1), with silence before and after so the recogniser's end-pointer closes the utterance as after real speech."""
    import wave
    import numpy as np
    if isinstance(source, (str, os.PathLike)):
        with wave.open(str(source), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != samplerate:
                raise ValueError(f"{source}: need {samplerate} Hz 16-bit mono, got {w.getframerate()} Hz, "
                                 f"{8 * w.getsampwidth()}-bit, {w.getnchannels()} channel(s)")
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    else:
        a = np.asarray(source)
        pcm = (np.clip(a, -1.0, 1.0) * 32767).astype("<i2") if a.dtype.kind == "f" else a.astype("<i2")
    pcm = np.concatenate([np.zeros(int(lead_s * samplerate), "<i2"), pcm.ravel(), np.zeros(int(tail_s * samplerate), "<i2")])
    n = max(1, int(samplerate * block_s))
    data = pcm.tobytes()
    return [data[i:i + 2 * n] for i in range(0, len(data), 2 * n)]


def load_vosk_model(path: Optional[str] = None):
    """The Vosk model at `path` (vosk_model_path); never downloads. Loading takes seconds and a recogniser is cheap, so
    a caller that decodes many sources (scripts/eval_voice.py) loads it once and passes it to each VoskBackend."""
    import vosk
    p = vosk_model_path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"no Vosk model at {p}")
    try:
        vosk.SetLogLevel(-1)
    except Exception:
        pass
    return vosk.Model(model_path=str(p))


_MAX_UTTERANCE_BYTES = 2 * 16000 * 20     # hybrid re-decode: at most the last 20 s of 16 kHz audio


def _result_text(raw: str) -> str:
    r = json.loads(raw or "{}")
    if "alternatives" in r:
        return (r["alternatives"][0].get("text", "") if r["alternatives"] else "")
    return r.get("text", "")


def needs_open_pass(grammar_text: str) -> bool:
    """Hybrid mode: decode the utterance again with the open vocabulary when the grammar heard a wake phrase (the
    grammar, a language model of commands that all start with it, forces chatter such as "is this coffee for me" into
    "... see for me ...") or a word outside its vocabulary ("[unk]")."""
    return "[unk]" in grammar_text.split() or bool(_WAKE.search(normalize(grammar_text)))


def hybrid_text(grammar_text: str, open_text: Optional[str]) -> str:
    """The text a hybrid recogniser hands on. The open vocabulary decides whether the wake phrase was said: when the
    grammar heard one and the open recogniser did not, the open reading goes on (so chatter wakes it no more often than
    the open recogniser alone). Otherwise the grammar's reading, which hears short command words right ("mute", not
    "mood"), except where it heard a word outside its vocabulary ("[unk]") and the open reading is a command (a FIND
    target such as "pharmacy" is outside any fixed list). open_text None: no second pass was needed."""
    g = " ".join(w for w in grammar_text.split() if w != "[unk]")
    if open_text is None:
        return g
    if _WAKE.search(normalize(grammar_text)) and not _WAKE.search(normalize(open_text)):
        return open_text
    if "[unk]" in grammar_text.split():
        o = parse_command(open_text, require_wake=False)
        if o is not None and o.intent != "ask":
            return open_text
    return g


def snap_text(open_text: str, grammar_text: Optional[str]) -> str:
    """The text a snap recogniser hands on: the open recogniser's, unless it does not read as a command (nothing, or a
    question, as "see for me mood" is); then the command grammar's reading of the same audio when that is a command
    and adds no wake phrase the open reading lacked (chatter must not wake it through the grammar's bias)."""
    if grammar_text is None:
        return open_text
    oc = parse_command(open_text, require_wake=False) if open_text else None
    if oc is not None and oc.intent != "ask":
        return open_text
    g = " ".join(w for w in grammar_text.split() if w != "[unk]")
    gc = parse_command(g, require_wake=False)
    if gc is None or gc.intent == "ask":
        return open_text
    if _WAKE.search(normalize(g)) and not _WAKE.search(normalize(open_text)):
        return open_text
    return g


class VoskBackend:
    """Offline speech recognition with Vosk (https://alphacephei.com/vosk): a daemon thread reads 16 kHz mono int16
    blocks from the microphone (sounddevice.RawInputStream) into a streaming KaldiRecognizer and hands every final
    result to the listener. Audio captured while the demo speaks (plus the echo margin) is not fed to the recogniser,
    which is reset afterwards, so no partial echo leaks into the next result. Nothing is downloaded: the model must be
    on disk.

    recogniser: how words are decoded (scripts/eval_voice.py measured them on synthesised speech, see
    results/live/voice_eval.md):
        open     the model's full vocabulary: any FIND target, but short command words go astray ("mute" -> "mood")
        grammar  a KaldiRecognizer with grammar_phrases(): the command vocabulary plus "[unk]"; fast and robust on
                 commands, but a target outside the list is "[unk]"
        hybrid   the grammar recogniser; an utterance in which it heard the wake phrase or "[unk]" is decoded
                 again by the open recogniser, which decides whether the wake phrase was said (the grammar's bias forces
                 chatter into "see for me"), and hybrid_text() picks the reading
        snap     (default) the other way round: the open recogniser; an utterance that does not read as a command is
                 decoded again with the grammar, which snaps near-misses to the command words (snap_text()). On
                 synthesised speech: 90% of commands right against 82% for open, the same false wakes as open (grammar
                 alone: 91% but a quarter of the "see / for / me" chatter woke it), about open's CPU
    source: a WAV file or an array of samples instead of the microphone (audio_source_blocks; tests and
    scripts/eval_voice.py); audio_blocks: an iterable of int16 byte blocks. model: a loaded vosk.Model to share."""
    name = "vosk"
    audio = True

    def __init__(self, model_path: Optional[str] = None, samplerate: int = 16000, device=None, block_s: float = 0.25,
                 audio_blocks: Optional[Iterable[bytes]] = None, source=None, recogniser: str = DEFAULT_RECOGNISER,
                 model=None, second_pass: str = "auto"):
        if recogniser not in RECOGNISERS:
            raise ValueError(f"recogniser must be one of {', '.join(RECOGNISERS)}")
        self.model_path = vosk_model_path(model_path)
        self.samplerate = int(samplerate)
        self.device = device
        self.block_s = block_s
        if source is not None:
            audio_blocks = audio_source_blocks(source, self.samplerate, block_s)
        self.audio_blocks = audio_blocks
        self.recogniser = recogniser
        self.model = model
        # "always": run the second recogniser (hybrid, snap) on every utterance, though the choice uses it only when
        # needed: `raw` then holds both readings of the same audio (scripts/eval_voice.py scores the modes on them)
        self.second_pass = second_pass
        self.running = False
        self.error: Optional[str] = None
        self.results: List[str] = []         # every final result, before the listener's rules (for debugging)
        # per final result: primary text, second reading (or None), CPU s of each, whether the mode needed the second
        self.raw: List[dict] = []
        self.decode_s = 0.0                  # CPU time the mode spent decoding (scripts/eval_voice.py)
        self.listener: Optional[VoiceListener] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, listener: VoiceListener) -> None:
        self.listener = listener
        self.running = True                  # optimistic: the status says "voice" while the model loads
        self._thread = threading.Thread(target=self._run, name="seeforme-vosk", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _mic_blocks(self) -> Iterator[Tuple[float, bytes]]:
        import sounddevice as sd
        q: "queue.Queue[Tuple[float, bytes]]" = queue.Queue()

        def callback(indata, frames, time_info, status):     # runs in the PortAudio thread: copy and hand over
            q.put((time.monotonic(), bytes(indata)))

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=int(self.samplerate * self.block_s),
                               device=self.device, dtype="int16", channels=1, callback=callback):
            while not self._stop.is_set():
                try:
                    yield q.get(timeout=0.5)
                except queue.Empty:
                    continue

    def _recognisers(self, vosk, model):
        """(primary, re-decoder or None) for this backend's mode: hybrid re-decodes with the open vocabulary, snap with
        the grammar."""
        def grammar():
            return vosk.KaldiRecognizer(model, self.samplerate, json.dumps(grammar_phrases()))
        m = self.recogniser
        if m == "open":
            return vosk.KaldiRecognizer(model, self.samplerate), None
        if m == "snap":
            return vosk.KaldiRecognizer(model, self.samplerate), grammar()
        return grammar(), (vosk.KaldiRecognizer(model, self.samplerate) if m == "hybrid" else None)

    def _redecode(self, opener, audio: bytes) -> str:
        """The second recogniser's reading of one utterance's audio (hybrid and snap modes)."""
        if hasattr(opener, "Reset"):
            opener.Reset()
        opener.AcceptWaveform(audio)
        return _result_text(opener.FinalResult())

    def _run(self) -> None:
        lis = self.listener
        assert lis is not None
        rec = opener = model = blocks = None
        try:
            import vosk
            if self.model is None and not self.model_path.is_dir():
                raise FileNotFoundError(f"no Vosk model at {self.model_path}")
            model = self.model if self.model is not None else load_vosk_model(str(self.model_path))
            rec, opener = self._recognisers(vosk, model)
            self._cpu = 0.0                  # primary decoding since the last final result
            blocks = (((lis.clock(), b) for b in self.audio_blocks) if self.audio_blocks is not None else self._mic_blocks())
            dirty = False
            utt = bytearray()                # this utterance's audio, for the open re-decode
            for t_cap, data in blocks:
                if self._stop.is_set():
                    break
                if lis.echo_blocked(t_cap):
                    dirty = True             # drop the demo's own voice, and whatever the recogniser half-heard of it
                    continue
                if dirty:
                    if hasattr(rec, "Reset"):
                        rec.Reset()
                    else:
                        rec, opener = self._recognisers(vosk, model)
                    utt.clear()
                    dirty = False
                t0 = time.process_time()
                utt += data
                if len(utt) > _MAX_UTTERANCE_BYTES:
                    del utt[:len(utt) - _MAX_UTTERANCE_BYTES]
                done = rec.AcceptWaveform(data)
                self._cpu += time.process_time() - t0
                if done:
                    self._utterance(rec.Result(), opener, bytes(utt), t_cap)
                    utt.clear()
            if self.audio_blocks is not None and not self._stop.is_set():
                t0 = time.process_time()
                last = rec.FinalResult()
                self._cpu += time.process_time() - t0
                self._utterance(last, opener, bytes(utt), lis.clock())
        except Exception as e:               # no microphone, no permission, broken model: typed commands still work
            self.error = f"{type(e).__name__}: {e}"
            lis.notify(f"Voice commands are off ({self.error}). You can type commands in the terminal.")
        finally:
            if hasattr(blocks, "close"):
                blocks.close()               # closes the microphone stream
            rec = opener = model = blocks = None    # free the recognisers and our own model in this thread, now
            self.running = False

    def _utterance(self, raw: str, opener, audio: bytes, at: float) -> None:
        text = _result_text(raw).strip()
        cpu, self._cpu = self._cpu, 0.0
        second, cpu2, need = None, 0.0, False
        if opener is not None:
            if self.recogniser == "snap":
                c = parse_command(text, require_wake=False) if text else None
                need = bool(text) and (c is None or c.intent == "ask")
            else:
                need = needs_open_pass(text)
            if need or (self.second_pass == "always" and audio):
                t0 = time.process_time()
                second = self._redecode(opener, audio)
                cpu2 = time.process_time() - t0
        self.decode_s += cpu + (cpu2 if need else 0.0)
        if text or second:
            self.raw.append({"text": text, "second": second, "cpu": cpu, "cpu_second": cpu2, "needed": need})
        if opener is None:
            self._final(text, at)
        else:
            used = second if need else None
            self._final(snap_text(text, used) if self.recogniser == "snap" else hybrid_text(text, used), at)

    def _final(self, text: str, at: float) -> None:
        text = " ".join(w for w in (text or "").split() if w != "[unk]").strip()
        if text and self.listener is not None:
            self.results.append(text)
            self.listener.hear(text, source=self.name, audio=True, at=at)
