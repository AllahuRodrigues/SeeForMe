"""Voice commands for the terminal demo: the parser against the vectors shared with the web app, the hands-free
follow-up window, echo suppression, and the typed, scripted, fake and Vosk backends. No microphone is opened: the
Vosk tests feed speech synthesised by macOS `say` to the real recogniser in a child interpreter (skipped without vosk,
its model or `say`)."""
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from seeforme.live.voice import (Command, FakeBackend, ScriptedCommands, TypedBackend, VoiceListener, VoskBackend,
                                 clean_target, normalize, parse_command, parse_script, vosk_status, words_to_digits)

VECTORS = json.loads((Path(__file__).parent / "data" / "voice_commands.json").read_text())
WEB = Path(__file__).resolve().parents[1] / "web"
NODE = shutil.which("node")
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _cases(kind):
    return [pytest.param(c["text"], c["intent"], c["arg"], id=f"{kind}:{c['text'] or '<empty>'}") for c in VECTORS[kind]]


@pytest.mark.parametrize("text,intent,arg", _cases("wake"))
def test_hands_free_vectors(text, intent, arg):
    got = parse_command(text, require_wake=True)
    assert ((got.intent, got.arg) if got else (None, None)) == (intent, arg)


@pytest.mark.parametrize("text,intent,arg", _cases("direct"))
def test_direct_vectors(text, intent, arg):
    got = parse_command(text, require_wake=False)
    assert ((got.intent, got.arg) if got else (None, None)) == (intent, arg)


@pytest.mark.parametrize("text,mode,intent,arg", [pytest.param(c["text"], c["mode"], c["intent"], c["arg"],
                                                                id=f"{c['mode']}:{c['text'].encode('unicode_escape').decode()}")
                                                   for c in VECTORS["unicode"]])
def test_unicode_vectors(text, mode, intent, arg):
    got = parse_command(text, require_wake=mode == "wake")
    assert ((got.intent, got.arg) if got else (None, None)) == (intent, arg)


def test_every_vector_intent_is_known():
    from seeforme.live.voice import INTENTS
    assert set(VECTORS["intents"]) == set(INTENTS)
    for kind in ("wake", "direct", "unicode"):
        assert all(c["intent"] is None or c["intent"] in INTENTS for c in VECTORS[kind])
    assert all(c["mode"] in ("wake", "direct") for c in VECTORS["unicode"])


# pieces of a parser fuzz corpus: command words, the wake phrase and its mis-hearings, spoken numbers, and the non-ASCII
# text an all-ASCII corpus never tries (accented and decomposed letters, other scripts' letters and digits, superscripts,
# combining marks, zero-width, non-breaking and control spaces, curly apostrophes, punctuation glued to words)
_FUZZ_WORDS = (
    "see", "for", "me", "sea", "c", "four", "4", "seeforme", "hey", "ok", "find", "look", "where's", "the", "a", "nearest",
    "door", "exit", "room", "two", "oh", "o", "to", "too", "won", "twenty", "one", "hundred", "and", "teen", "stop",
    "looking", "cancel", "never", "mind", "forget", "it", "clear", "goal", "what", "does", "do", "say", "says", "read",
    "sign", "mode", "scene", "walking", "more", "detail", "less", "please", "now", "thanks", "pause", "start", "mute",
    "help", "flip", "camera", "describe", "around", "again", "exit", "way", "out", "it's", "don't",
    "show", "written", "front", "back", "rear", "selfie", "facing", "off", "on", "turn", "get", "tell", "know", "where",
    "can", "i", "is", "are", "we", "how", "what's", "you", "see",
    "café", "cafe\u0301", "menú", "naïve", "josé", "ño", "señal", "ÉLÉVATEUR", "straße", "İstanbul", "ǅ", "日本", "مرحبا",
    "٤", "٢٠٤", "１２", "²", "①", "Ⅻ", "½", "é", "\u0301", "_", "'", "’", "‘", "!", "?", ",", "-", ".")
_FUZZ_JOIN = (" ", " ", " ", "", ", ", "'", "\u00a0", "\u200b", "\u001c", "\u0085", "\ufeff", "\t", "\n", "_")


def _fuzz_corpus(n: int = 3000, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        words = [rng.choice(_FUZZ_WORDS) for _ in range(rng.randint(1, 7))]
        s = words[0] + "".join(rng.choice(_FUZZ_JOIN) + w for w in words[1:])
        r = rng.random()
        s = (("see for me" + rng.choice(_FUZZ_JOIN) + s) if r < 0.4 else (s + rng.choice(_FUZZ_JOIN) + "see for me find a door")
             if r < 0.5 else s)
        out.append(s.upper() if rng.random() < 0.1 else s)
    return out


# the everyday phrasings of the last round put together in every order (openings x heads x rests): random words seldom
# hit the rules that need several words in a row ("show me what it says", "switch the camera to the front")
_OPENINGS = ("", "see for me ", "SeeForMe, ", "please ", "can you ", "tell me ", "do you know ", "find out ")
_HEADS = ("show me", "find me", "get me", "get me to", "where can I find", "where do I find", "how do I get to", "where",
          "what's written", "what does", "turn", "switch to", "use", "flip", "stop", "find", "read", "describe")
_RESTS = ("", "the exit", "a chair", "the way out", "the way to the bathroom", "the exit is", "we are", "around", "the text",
          "what it says", "what you see", "what's on my left", "front camera", "the back camera", "rear cam", "camera off",
          "the camera on", "the camera to the front", "on the sign", "the sign say", "me", "room two oh four", "the café",
          "to the front", "commands", "here")


def _phrase_corpus():
    return [f"{o}{h} {r}".strip() for o in _OPENINGS for h in _HEADS for r in _RESTS]


@pytest.mark.skipif(NODE is None or not ESBUILD.exists(), reason="needs Node.js and web/node_modules (cd web && npm install)")
def test_web_parser_matches_python_on_unicode_and_fuzz(tmp_path):
    """web/src/voice.ts and parse_command give the same command, in both modes, and the same normalised text for every
    shared vector, 3,000 seeded random utterances with non-ASCII text and the everyday phrasings in every combination
    (_phrase_corpus). The ASCII-only cross-check of an earlier round missed 9 differences (JavaScript's \\w, \\b and \\d
    are ASCII; Python's are Unicode; NFC on one side only)."""
    corpus = ([c["text"] for k in ("wake", "direct", "unicode") for c in VECTORS[k]] + _fuzz_corpus() + _phrase_corpus())
    bundle = tmp_path / "voice.mjs"
    subprocess.run([str(ESBUILD), str(WEB / "src" / "voice.ts"), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=warning"], check=True, timeout=120)
    (tmp_path / "run.mjs").write_text(
        'import { normaliseSpeech, parseCommand } from "./voice.mjs";\n'
        'import { readFileSync, writeFileSync } from "node:fs";\n'
        'const c = JSON.parse(readFileSync(process.argv[2], "utf8"));\n'
        'writeFileSync(process.argv[3], JSON.stringify(c.map((s) => [normaliseSpeech(s), ...["wake", "direct"].map((m) => {\n'
        '  const r = parseCommand(s, m); return r ? [r.intent, r.arg] : null; })])));\n')
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))
    subprocess.run([NODE, str(tmp_path / "run.mjs"), str(tmp_path / "corpus.json"), str(tmp_path / "ts.json")],
                   check=True, timeout=120)
    ts = json.loads((tmp_path / "ts.json").read_text(encoding="utf-8"))

    def py(s):
        return [normalize(s)] + [None if c is None else [c.intent, c.arg]
                                 for c in (parse_command(s, require_wake=w) for w in (True, False))]
    diffs = [(s, p, t) for s, t in zip(corpus, ts) for p in [py(s)] if p != t]
    assert len(ts) == len(corpus) and diffs == [], diffs[:10]
    assert {r[1][0] for r in ts if r[1]} == set(VECTORS["intents"])      # every intent is reached, not only "ask"


@pytest.mark.parametrize("spoken,digits", [
    ("room two oh four", "room 204"), ("gate twenty one", "gate 21"), ("gate twelve", "gate 12"),
    ("room two hundred and four", "room 204"), ("room four oh one", "room 401"), ("flight one two three", "flight 123"),
    ("room 204", "room 204"), ("oh no", "oh no"), ("gate ninety", "gate 90"),
    ("room oh four", "room 04"), ("room oh four oh four", "room 0404"), ("oh find a door", "oh find a door")])
def test_spoken_numbers_become_digits(spoken, digits):
    assert words_to_digits(spoken) == digits


@pytest.mark.parametrize("text,intent,arg", [
    # reviewer repros: mode and detail words inside a FIND target hijacked it
    ("see for me find the balanced rock", "find", "balanced rock"), ("where is the read mode", "find", "read mode"),
    ("find the scene mode button", "find", "scene mode button"), ("take me to the auto mode store", "find", "auto mode store"),
    ("find room oh four", "find", "room 04"),
    # mode and detail phrases are still commands when they are the whole command
    ("switch to reading mode", "mode", "read"), ("go back to auto mode", "mode", "auto"), ("turn on scene mode", "mode", "scene"),
    ("could you give me more details please", "verbosity", "detailed"), ("talk less", "verbosity", "quiet"),
    ("is the table balanced", "ask", "is the table balanced"), ("is there a read mode", "ask", "is there a read mode"),
    ("see for me help me find the door", "find", "door"), ("help for commands", "help", None)])
def test_find_wins_over_mode_and_detail_words(text, intent, arg):
    got = parse_command(text, require_wake=False)
    assert (got.intent, got.arg) == (intent, arg)


@pytest.mark.parametrize("text,intent,arg", [
    # "show me": a describe or read phrase after it keeps its meaning, a question stays a question, the rest is a FIND
    ("show me what's around me", "describe", None), ("show me what you see", "describe", None),
    ("show me around", "describe", None), ("show me the text", "read", None), ("show me what it says", "read", None),
    ("show me where the exit is", "find", "exit"), ("show me the way out", "find", "exit"), ("show me a chair", "find", "chair"),
    ("show me what's on my left", "ask", "show me what's on my left"), ("show me the commands", "help", None),
    # "me" alone is no target; "get me to X" and "get me X" are both X
    ("find me", "find", None), ("find me a chair", "find", "chair"), ("get me to the exit", "find", "exit"),
    ("get me a chair", "find", "chair"), ("find the way out", "find", "exit"),
    # a named camera side is a flip with that side; without one, a flip to the next camera
    ("use the front camera", "flip", "front"), ("selfie camera", "flip", "front"), ("rear camera", "flip", "back"),
    ("switch the camera to the front", "flip", "front"), ("flip camera", "flip", None),
    # camera off / on; "stop talking" stays a pause, questions about the camera keep their meaning
    ("turn the camera off", "pause", None), ("stop talking", "pause", None), ("camera on", "start", None),
    ("is the camera on", "ask", "is the camera on"), ("where is the camera", "find", "camera"),
    ("is there a door to show me", "ask", "is there a door to show me"), ("where are we", "describe", None),
    # every fixed phrase must be the whole command: the app's own "Paused. The camera is off." once parsed as a pause,
    # because "^pause...|...|camera off$" anchored only its first and last alternatives
    ("Paused. The camera is off.", "ask", "paused the camera is off"), ("stopped the camera", "ask", "stopped the camera"),
    ("Already on the front camera.", "ask", "already on the front camera")])
def test_everyday_phrasings_and_rule_order(text, intent, arg):
    got = parse_command(text, require_wake=False)
    assert (got.intent, got.arg) == (intent, arg)


def test_grammar_holds_the_everyday_phrasings():
    """The Vosk grammar (and so the snap recogniser) can produce every new phrasing; words outside the small model's
    vocabulary stay out of it (Vosk drops them with a warning: "selfie camera" would become "camera")."""
    from seeforme.live.voice import grammar_phrases
    phrases = set(grammar_phrases())
    for ph in ("show me the exit", "show me what you see", "show me what it says", "show me where the exit is",
               "show me the way out", "what's written", "what is written here", "front camera", "back camera",
               "rear camera", "use the front camera", "switch to the back camera", "stop the camera", "camera off",
               "turn the camera off", "camera on", "turn on the camera", "find me a chair", "get me a chair",
               "where can i find a chair", "where do i find the exit", "how do i get to the exit", "where are we"):
        assert ph in phrases and f"see for me {ph}" in phrases, ph
        assert parse_command(ph, require_wake=False).intent != "ask", ph
    words = {w for ph in phrases for w in ph.split()}
    assert {"show", "written", "front", "back", "rear", "camera", "off", "on"} <= words
    assert not words & {"selfie", "unmute"}


def test_terminal_flip_to_a_side_claims_no_side():
    """The terminal demo cycles camera indices: "front camera" switches to the next one and says it cannot tell which
    way a camera faces, never "front camera"."""
    import re
    import types
    from seeforme.live.commands import CommandRouter
    r = CommandRouter(types.SimpleNamespace(), types.SimpleNamespace(enabled=True), flip=lambda: "Looking for another camera.")
    assert r.handle(parse_command("see for me flip camera")) == "Looking for another camera."
    for said in ("see for me front camera", "see for me switch to the back camera"):
        reply = r.handle(parse_command(said))
        assert reply == "This demo cannot tell which way a camera faces. Looking for another camera.", reply
        assert not re.search(r"\b(?:front|back|rear)\b", reply)


def test_the_apps_own_phrases_change_nothing():
    """Heard back hands-free if echo suppression ever fails (e.g. another app plays them): never a mode, detail, mute,
    pause or quit, and the start-up phrase is at most "help"."""
    from seeforme.live.commands import HELP_TYPED, HELP_VOICE
    assert parse_command("SeeForMe ready. Say 'SeeForMe, help' for commands.").intent == "help"
    for phrase in (HELP_VOICE, HELP_TYPED, "Paused. I'll still warn you about hazards.", "Looking for a chair. I'll tell you when I see it.",
                   "This demo cannot tell which way a camera faces. Looking for another camera."):
        got = parse_command(phrase, require_wake=True)
        assert got is None or got.intent in ("ask", "help"), (phrase, got)


def test_find_targets_drop_articles_and_superlatives():
    assert clean_target("the nearest bathroom") == "bathroom"
    assert clean_target("a chair for me please") == "chair"
    assert parse_command("see for me find room twenty one b").arg == "room 21 b"


def test_command_equality_ignores_context():
    assert Command("find", "door", text="see for me find a door", source="vosk") == Command("find", "door")


# ---- the listener -------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeSpeaker:
    """is_speaking() / speech_ended_at in the listener's clock."""

    def __init__(self, clock):
        self.clock = clock
        self.until = -math.inf          # speaking until this time

    def speak(self, start, end):
        self.clock.t = start
        self.until = end

    def is_speaking(self):
        return self.clock.t < self.until

    @property
    def speech_ended_at(self):
        return self.clock.t if self.is_speaking() else self.until


def make(clock=None, speaker=None, **kw):
    clock = clock or Clock()
    lis = VoiceListener(speaker, clock=clock, **kw)
    mic = lis.add(FakeBackend())
    lis.start()
    return lis, mic, clock


def heard(lis):
    return [(c.intent, c.arg) for c in lis.poll()]


def test_bare_wake_word_opens_a_five_second_window_for_one_command():
    lis, mic, clock = make()
    assert mic.feed("find a door", at=0.0) is None                    # no wake word: ignored
    assert mic.feed("see for me", at=1.0) == Command("wake")
    assert lis.state(at=2.0) == "listening"
    assert mic.feed("find a door", at=3.0) == Command("find", "door")  # within 5 s: no wake word needed
    assert mic.feed("describe", at=3.5) is None                       # one command per wake word
    assert lis.state(at=3.5) == "wake"
    mic.feed("see for me", at=10.0)
    assert mic.feed("what's around me", at=15.5) is None              # the window closed at 15.0
    assert heard(lis) == [("wake", None), ("find", "door"), ("wake", None)]
    assert lis.ignored_no_wake == 3


def test_full_command_after_the_wake_word_needs_no_window():
    lis, mic, _ = make()
    assert mic.feed("um see for me pause", at=0.0) == Command("pause")
    assert mic.feed("the dog barked see for me find a chair", at=1.0) == Command("find", "chair")
    assert heard(lis) == [("pause", None), ("find", "chair")]


def test_push_to_talk_opens_the_window_without_a_wake_word():
    lis, mic, clock = make()
    lis.listen_once(at=0.0)
    assert mic.feed("stop", at=2.0) == Command("pause")


def test_echo_suppression_while_speaking_and_for_0_4_s_after():
    clock = Clock()
    sp = FakeSpeaker(clock)
    lis, mic, _ = make(clock, sp)
    sp.speak(0.0, 3.0)                       # the demo says something for 3 s
    clock.t = 1.0
    assert mic.feed("see for me stop") is None
    clock.t = 3.2
    assert mic.feed("see for me stop") is None                       # 0.2 s after it ended
    clock.t = 3.5
    assert mic.feed("see for me stop") == Command("pause")           # 0.5 s after: heard
    assert lis.ignored_echo == 2
    assert heard(lis) == [("pause", None)]


def test_echo_suppression_with_a_speaker_that_only_has_busy():
    clock = Clock()

    class BusyOnly:
        speaking = True

        def busy(self):
            return self.speaking
    sp = BusyOnly()
    lis, mic, _ = make(clock, sp)
    clock.t = 1.0
    assert mic.feed("see for me find a door") is None
    sp.speaking = False
    clock.t = 1.3
    assert mic.feed("see for me find a door") is None                # last seen busy at 1.0
    clock.t = 1.5
    assert mic.feed("see for me find a door") == Command("find", "door")


def test_typed_and_scripted_commands_are_not_echo_suppressed():
    clock = Clock()
    sp = FakeSpeaker(clock)
    lis = VoiceListener(sp, clock=clock)
    kb = lis.add(FakeBackend(audio=False))
    lis.set_script([(1.0, "describe")])
    lis.start()
    sp.speak(0.0, 10.0)
    clock.t = 0.5
    assert kb.feed("pause") == Command("pause")                       # typed: wake word optional, never suppressed
    assert [(c.intent, c.source) for c in lis.poll(stream_t=1.0)] == [("pause", "fake"), ("describe", "script")]


def test_follow_up_window_is_held_open_while_the_demo_speaks():
    clock = Clock()
    sp = FakeSpeaker(clock)
    lis, mic, _ = make(clock, sp)
    mic.feed("see for me", at=0.0)
    sp.speak(1.0, 4.0)                       # e.g. "Yes?" or a cue: the user cannot be heard meanwhile
    for t in (1.0, 2.0, 3.0, 3.9):
        clock.t = t
        lis.poll()
    clock.t = 8.0                            # 5 s after the speech + margin, not 5 s after the wake word
    assert mic.feed("what's around me") == Command("describe")


def test_follow_up_window_has_a_hard_cap():
    clock = Clock()
    sp = FakeSpeaker(clock)
    lis, mic, _ = make(clock, sp, max_follow_up_s=15.0)
    mic.feed("see for me", at=0.0)
    sp.speak(0.5, 60.0)                      # speaks for a minute
    for t in range(1, 60):
        clock.t = float(t)
        lis.poll()
    sp.until = 59.0
    clock.t = 61.0
    assert not lis.in_window() and mic.feed("describe") is None


def test_listener_states():
    lis = VoiceListener(None, clock=Clock())
    assert lis.state() == "off"
    kb = lis.add(FakeBackend(audio=False)); lis.start()
    assert lis.state() == "typed"
    lis2, mic, clock = make()
    assert lis2.state() == "wake"
    mic.feed("see for me", at=0.0)
    assert lis2.state(at=1.0) == "listening"


def test_typed_backend_reads_lines_in_a_thread():
    lis = VoiceListener(None)
    typed = lis.add(TypedBackend(stream=io.StringIO("find a door\nsee for me pause\n\nhello there\n")))
    lis.start()
    deadline = time.time() + 5
    got = []
    while time.time() < deadline and len(got) < 3:
        got += lis.poll()
        time.sleep(0.01)
    assert [(c.intent, c.arg) for c in got] == [("find", "door"), ("pause", None), ("ask", "hello there")]
    assert all(c.source == "typed" for c in got)


def test_typed_goal_prompt_for_the_f_key():
    lis = VoiceListener(None)
    typed = TypedBackend(stream=io.StringIO(""))
    typed.listener = lis
    typed.expect_goal()
    assert typed.line("Room two oh four") == Command("find", "room 204")
    typed.expect_goal()
    assert typed.line("") == Command("stop_find")
    assert typed.line("help") == Command("help")                      # back to normal commands


class Terminal:
    """A terminal stream: readline() returns "" for ctrl-D, and more lines may follow."""

    def __init__(self, lines):
        self.lines = list(lines)

    def isatty(self):
        return True

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        time.sleep(0.02)
        return ""


def test_typed_backend_keeps_listening_after_ctrl_d_in_a_terminal():
    """macOS `script` (and a stray ctrl-D) sends an end of file to a terminal first: the reader stopped there."""
    lis = VoiceListener(None)
    typed = lis.add(TypedBackend(stream=Terminal(["", "describe\n"])))
    lis.start()
    got, deadline = [], time.time() + 5
    while time.time() < deadline and not got:
        got += lis.poll()
        time.sleep(0.01)
    assert [c.intent for c in got] == ["describe"] and typed.running
    typed.stop()


def test_typed_single_letters_are_the_window_keys():
    """Reviewer repro: typing q (or h) in the terminal was an "ask" ("Sorry, I didn't understand")."""
    lis = VoiceListener(None)
    typed = TypedBackend(stream=io.StringIO(""))
    typed.listener = lis
    assert typed.line("q\n") == Command("quit") and typed.line(" H ") == Command("help")
    assert typed.line("w") == Command("describe") and typed.line("r") == Command("read")
    assert typed.line("p") == Command("pause", "toggle") and typed.line("s") == Command("mute", "toggle")
    assert [c.source for c in lis.poll()] == ["typed"] * 6
    assert typed.line("f") is None and lis.notices() == ["Type what to find and press return."]
    assert typed.line("the exit") == Command("find", "exit")
    assert typed.line("quit") == Command("quit") and typed.line("x").intent == "ask"


def test_scripted_commands_fire_at_stream_time():
    items = parse_script("3.0:describe; 1:see for me find a door ;")
    assert items == [(1.0, "see for me find a door"), (3.0, "describe")]
    lis = VoiceListener(None, clock=Clock())
    lis.set_script(items)
    assert lis.poll(stream_t=0.5) == []
    assert [(c.intent, c.arg, c.source) for c in lis.poll(stream_t=1.0)] == [("find", "door", "script")]
    assert [c.intent for c in lis.poll(stream_t=9.0)] == ["describe"]
    assert lis.script.remaining == 0


@pytest.mark.parametrize("bad", ["find a door", "x:find", "2:", "2:   "])
def test_bad_scripts_fail_at_start_up(bad):
    with pytest.raises(ValueError):
        parse_script(bad)


def test_scripted_commands_class_alone():
    s = ScriptedCommands([(2.0, "b"), (1.0, "a")])
    assert s.due(1.5) == ["a"] and s.due(2.0) == ["b"] and s.remaining == 0


def test_vosk_status_without_a_model(tmp_path):
    ok, why = vosk_status(str(tmp_path / "missing"))
    assert not ok and why


# ---- Vosk, offline, on synthesised speech -------------------------------------------------------------------------
# All checks run in one child interpreter, so the recogniser's native library and its 68 MB model are loaded (once) and
# freed outside the pytest process. "mute" is the word the open vocabulary heard as "mood" or "newt"; it goes in as a
# WAV file (source=) through the default recogniser, as scripts/eval_voice.py feeds its phrases.
_VOSK_CHILD = r"""
import json, subprocess, sys, tempfile, time, wave, os
from seeforme.live.voice import DEFAULT_RECOGNISER, VoiceListener, VoskBackend, load_vosk_model

class Talking:
    def is_speaking(self):
        return True

out = {}
d = tempfile.mkdtemp()
t0 = time.perf_counter()
model = load_vosk_model()
for name, text, speaker in (("command", "See for me, find a chair.", None), ("echo", "See for me, stop.", Talking()),
                            ("wav", "See for me, mute.", None)):
    wav = os.path.join(d, name + ".wav")
    subprocess.run(["say", "-o", wav, "--data-format=LEI16@16000", text], check=True, timeout=30)
    lis = VoiceListener(speaker)
    if name == "wav":
        vb = lis.add(VoskBackend(source=wav, model=model))
    else:
        with wave.open(wav) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            data = w.readframes(w.getnframes())
        data = b"\0\0" * 8000 + data + b"\0\0" * 16000          # silence around it ends the utterance
        vb = lis.add(VoskBackend(audio_blocks=[data[i:i + 8000] for i in range(0, len(data), 8000)], model=model))
    lis.start()
    vb.join(timeout=60)
    out[name] = {"error": vb.error, "results": vb.results, "commands": [[c.intent, c.arg] for c in lis.poll()],
                 "ignored_echo": lis.ignored_echo, "recogniser": vb.recogniser}
out["seconds"] = time.perf_counter() - t0
out["default"] = DEFAULT_RECOGNISER
print(json.dumps(out))
"""


def _vosk_ready():
    ok, _ = vosk_status()
    return ok and shutil.which("say") is not None


@pytest.fixture(scope="module")
def vosk_run():
    env = dict(os.environ, OMP_NUM_THREADS="2", PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    p = subprocess.run([sys.executable, "-c", _VOSK_CHILD], capture_output=True, text=True, timeout=300, env=env,
                       cwd=Path(__file__).resolve().parents[1])
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not _vosk_ready(), reason="needs vosk, sounddevice, the Vosk model (make voice-setup) and macOS say")
def test_vosk_backend_turns_speech_into_a_command(vosk_run):
    r = vosk_run["command"]
    assert r["error"] is None and r["results"], r
    assert ["find", "chair"] in r["commands"], r


@pytest.mark.skipif(not _vosk_ready(), reason="needs vosk, sounddevice, the Vosk model (make voice-setup) and macOS say")
def test_vosk_backend_never_hears_the_demo_itself(vosk_run):
    r = vosk_run["echo"]
    assert r["error"] is None and r["results"] == [] and r["commands"] == [], r


@pytest.mark.skipif(not _vosk_ready(), reason="needs vosk, sounddevice, the Vosk model (make voice-setup) and macOS say")
def test_vosk_backend_hears_a_short_command_from_a_wav_file(vosk_run):
    """One synthesised command through VoskBackend(source=wav) and the default recogniser, in a few seconds."""
    r = vosk_run["wav"]
    assert r["error"] is None and r["recogniser"] == vosk_run["default"], r
    assert r["commands"] == [["mute", None]], r
    assert vosk_run["seconds"] < 60, vosk_run["seconds"]      # model load + 3 syntheses + decoding (about 6 s unloaded)


def test_audio_source_blocks_and_hybrid_choice(tmp_path):
    """The file / array source pads silence around the speech and checks the format; the hybrid recogniser keeps the
    grammar's reading unless it heard an unknown word and the open reading is a command."""
    import wave
    import numpy as np
    from seeforme.live.voice import audio_source_blocks, grammar_phrases, hybrid_text, snap_text
    x = np.linspace(-0.5, 0.5, 16000)
    blocks = audio_source_blocks(x, 16000, 0.25, lead_s=0.5, tail_s=1.0)
    assert sum(len(b) for b in blocks) == 2 * (16000 + 8000 + 16000) and all(len(b) <= 8000 for b in blocks)
    assert blocks[0] == b"\0\0" * 4000                         # leading silence
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    assert b"".join(audio_source_blocks(wav)) == b"".join(blocks)
    with wave.open(str(tmp_path / "b.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\0\0" * 800)
    with pytest.raises(ValueError):
        audio_source_blocks(tmp_path / "b.wav")
    from seeforme.live.voice import needs_open_pass
    assert hybrid_text("mute", None) == "mute" and not needs_open_pass("mute")        # a follow-up: grammar alone
    assert needs_open_pass("see for me mute") and needs_open_pass("[unk] door")
    assert hybrid_text("see for me mute", "see for me mood") == "see for me mute"     # the grammar hears the word
    assert hybrid_text("see for me find the [unk]", "see for me find the pharmacy") == "see for me find the pharmacy"
    assert hybrid_text("see for me [unk] mute", "see for me do mood") == "see for me mute"     # open is no command
    assert hybrid_text("[unk] [unk]", "the bus is late") == ""
    # chatter the grammar forced into the wake phrase: the open vocabulary decides, and it heard no wake phrase
    assert hybrid_text("is there car see for me where for you", "is this coffee for me or for you") == "is this coffee for me or for you"
    assert parse_command(hybrid_text("light see for me find", "let me see for men walked"), require_wake=True) is None
    # snap: the open reading unless it is no command; the grammar's then, but never a wake the open reading lacked
    assert snap_text("see for me find the pharmacy", "see for me find the [unk]") == "see for me find the pharmacy"
    assert snap_text("see for me mood", "see for me mute") == "see for me mute"
    assert snap_text("mood", "mute") == "mute"                                # in a follow-up window
    assert snap_text("let's see four me it's fine", "see for me it's fine") == "let's see four me it's fine"   # a question
    assert snap_text("i saw four meals", "see for me pause") == "i saw four meals"     # the grammar may not wake it
    assert snap_text("see for me cause", None) == "see for me cause"
    phrases = grammar_phrases()
    assert phrases[-1] == "[unk]" and "see for me" in phrases and "see for me find the door" in phrases
    from seeforme.live.voice import INTENTS
    heard = {parse_command(ph, require_wake=False).intent for ph in phrases if ph.startswith("see for me ")}
    assert heard >= set(INTENTS) - {"wake"}, set(INTENTS) - heard        # every command can be said in the grammar
    with pytest.raises(ValueError):
        VoskBackend(recogniser="loose")


def test_vosk_backend_reports_a_missing_model(tmp_path, monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "vosk", types.ModuleType("vosk"))     # never the real library in this process
    lis = VoiceListener(None)
    vb = lis.add(VoskBackend(model_path=str(tmp_path / "missing"), audio_blocks=[]))
    lis.start()
    vb.join(timeout=10)
    assert vb.error and not vb.running
    assert any("Voice commands are off" in n for n in lis.notices())
