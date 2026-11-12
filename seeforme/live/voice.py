"""Voice commands for the terminal demo: "SeeForMe, find a door", "SeeForMe, pause", "SeeForMe, what's around me".

A blind user cannot find the keys of an OpenCV window, so the live demo takes spoken (or typed) commands instead.

    parse_command(text, require_wake=True) -> Command(intent, arg) | None
        Hands-free (require_wake=True): the wake phrase "see for me" is required, also as "seeforme", "hey see for me"
        and the recogniser's usual mis-hearings ("sea for me", "c for me", "see four me", "see 4 me"); the command is
        the text after the LAST wake phrase, so chatter before it is ignored, and the wake phrase alone is intent
        "wake". Push-to-talk and typed commands (require_wake=False): the wake phrase is optional. Text with no
        command words is intent "ask" (arg = the text), answered by the rule-based answer() in commands.py. A FIND
        verb is tried before the mode and detail phrases, which must be the whole command ("navigation mode", "switch
        to reading mode", "give me more detail"), so "find the scene mode button" or "find the balanced rock" is a
        FIND. A FIND target is lowercase, without articles or "nearest" / "closest", with spoken numbers as digits
        ("room two oh four" -> "room 204", "room oh four" -> "room 04"). tests/data/voice_commands.json holds the
        vectors the web parser shares.

    VoiceListener(speaker)
        Recognisers run in background threads and call hear(); the main loop calls poll() once per frame and gets the
        Commands. Hands-free state machine for microphone backends: a bare wake word opens a follow-up window
        (follow_up_s = 5 s of time the user can actually be heard) in which one command needs no wake word; anything
        else without the wake phrase is ignored. Echo suppression: while the Speaker is speaking, and echo_margin_s
        (0.4 s) after, microphone results are dropped (the Vosk backend does not even feed that audio to the
        recogniser), so the demo never obeys its own voice. Typed and scripted commands are never suppressed.

    Backends
        VoskBackend      offline recognition (vosk + sounddevice, imported lazily; model in models/vosk/ or
                         $SEEFORME_VOSK_MODEL; 16 kHz mono int16, streaming KaldiRecognizer, final results only)
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
_FILLER = re.compile(r"^(?:(?:please|ok|okay|um|uh|so|and|now|hey|can you|could you|would you|will you)\s+)+")
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
    """Lowercase, punctuation to spaces, apostrophes kept inside words ("what's"), single spaces."""
    s = (text or "").lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s']", " ", s).replace("_", " ")
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)
    return " ".join(s.split())


def _is_num(w: str) -> bool:
    return w in _UNITS or w in _TEENS or w in _TENS or w == "hundred" or w.isdigit()


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
        if w.isdigit():
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


_FIND = re.compile(r"^(?:help me |i want to |i need to |i'd like to |i would like to |start |keep )?"
                   r"(?:find|look for|looking for|search for|locate|where is|where's|wheres|where are|take me to|"
                   r"guide me to|lead me to|bring me to|navigate to|get me to|i'm looking for|i am looking for)(?:\s+(.*))?$")
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
_EXACT = (   # (intent, pattern on the body without fillers); first match wins
    ("pause", r"(?:stop|pause|paws|wait|hold on|hang on)(?: (?:talking|announcing|announcements|it|that|for now|a (?:moment|second|sec|minute)))*"),
    ("start", r"(?:start|resume|continue|unpause|restart|keep going|go on|carry on)(?: (?:the|camera|announcing|announcements|talking|again|it|up|listening))*"),
    ("quit", r"(?:quit|goodbye|good bye|bye|bye bye|shut down|shutdown|(?:exit|quit|close) (?:the )?(?:app|program|demo))"),
    ("read", r"(?:read(?: (?:the|this|that|it|text|sign|signs|label|out|aloud|to me|for me))*|what (?:does|do) (?:it|this|that|the \w+(?: \w+)?) say|what (?:it|this|that) says)"),
    ("describe", r"(?:(?:what's|what is|whats) (?:around(?: me| here| us)?|here|near me|nearby)|what (?:do|can) you see(?: now| here)?|"
                 r"describe(?: (?:the scene|the room|it|this|what you see|my surroundings|the surroundings|surroundings|everything|around me))?|look around|where am i)"),
    ("repeat", r"(?:repeat(?: (?:that|it|again))*|say (?:that|it) again|say again|again|come again|pardon(?: me)?|what did you say|what was that|one more time)"),
    ("unmute", r"(?:unmute|un mute|sound on|turn (?:the )?sound on|turn on (?:the )?sound|voice on|unmute (?:sound|voice|audio))"),
    ("mute", r"(?:mute|be quiet|quiet|silence|hush|shut up|sound off|turn (?:the )?sound off|turn off (?:the )?sound|voice off|mute (?:sound|voice|audio))"),
    ("help", r"(?:help(?: me| please| (?:for|with) (?:the )?commands)?|what can i say|what can you do|commands|list commands|"
             r"what are the commands)"),
    ("flip", r"(?:flip|(?:flip|switch|change|swap|toggle) (?:the )?(?:camera|cameras|cam)|(?:next|other) camera|use the other camera)"),
)
_EXACT_RE = tuple((intent, re.compile(f"^{pat}$")) for intent, pat in _EXACT)
_BARE_EXIT = re.compile(r"^(?:exit|exits|way out)$")


def _body_command(body: str) -> Optional[Command]:
    """The command in the words after the wake phrase (normalised); None only for empty text."""
    if not body:
        return None
    core = _TRAILING.sub("", _FILLER.sub("", body)).strip() or body
    if _STOP_FIND.match(core):
        return Command("stop_find")
    m = _FIND.match(core)
    if m:
        target = clean_target(m.group(1) or "")
        return Command("find", target or None)
    m = _MODE.match(core)
    if m:
        return Command("mode", _MODE_ARG[m.group(1)])
    for level, rx in _VERBOSITY:
        if rx.match(core):
            return Command("verbosity", level)
    for intent, rx in _EXACT_RE:
        if rx.match(core):
            return Command(intent)
    if _BARE_EXIT.match(core):
        return Command("find", "exit")
    return Command("ask", body)


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


class VoskBackend:
    """Offline speech recognition with Vosk (https://alphacephei.com/vosk): a daemon thread reads 16 kHz mono int16
    blocks from the microphone (sounddevice.RawInputStream) into a streaming KaldiRecognizer and hands every final
    result to the listener. Audio captured while the demo speaks (plus the echo margin) is not fed to the recogniser,
    which is reset afterwards, so no partial echo leaks into the next result. `audio_blocks` replaces the microphone
    with an iterable of int16 byte blocks (tests, files). Nothing is downloaded: the model must be on disk."""
    name = "vosk"
    audio = True

    def __init__(self, model_path: Optional[str] = None, samplerate: int = 16000, device=None, block_s: float = 0.25,
                 audio_blocks: Optional[Iterable[bytes]] = None):
        self.model_path = vosk_model_path(model_path)
        self.samplerate = int(samplerate)
        self.device = device
        self.block_s = block_s
        self.audio_blocks = audio_blocks
        self.running = False
        self.error: Optional[str] = None
        self.results: List[str] = []         # every final result, before the listener's rules (for debugging)
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

    def _run(self) -> None:
        lis = self.listener
        assert lis is not None
        rec = model = blocks = None
        try:
            import vosk
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"no Vosk model at {self.model_path}")
            try:
                vosk.SetLogLevel(-1)
            except Exception:
                pass
            model = vosk.Model(model_path=str(self.model_path))     # a path: never downloads
            rec = vosk.KaldiRecognizer(model, self.samplerate)
            blocks = (((lis.clock(), b) for b in self.audio_blocks) if self.audio_blocks is not None else self._mic_blocks())
            dirty = False
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
                        rec = vosk.KaldiRecognizer(model, self.samplerate)
                    dirty = False
                if rec.AcceptWaveform(data):
                    self._final(json.loads(rec.Result()).get("text", ""), t_cap)
            if self.audio_blocks is not None and not self._stop.is_set():
                self._final(json.loads(rec.FinalResult()).get("text", ""), lis.clock())
        except Exception as e:               # no microphone, no permission, broken model: typed commands still work
            self.error = f"{type(e).__name__}: {e}"
            lis.notify(f"Voice commands are off ({self.error}). You can type commands in the terminal.")
        finally:
            if hasattr(blocks, "close"):
                blocks.close()               # closes the microphone stream
            rec = model = blocks = None      # free the recogniser and the model in this thread, now
            self.running = False

    def _final(self, text: str, at: float) -> None:
        text = (text or "").strip()
        if text and self.listener is not None:
            self.results.append(text)
            self.listener.hear(text, source=self.name, audio=True, at=at)
