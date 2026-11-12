"""Voice commands for the terminal demo: the parser against the vectors shared with the web app, the hands-free
follow-up window, echo suppression, and the typed, scripted, fake and Vosk backends. No microphone is opened: the
Vosk tests feed speech synthesised by macOS `say` to the real recogniser in a child interpreter (skipped without vosk,
its model or `say`)."""
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from seeforme.live.voice import (Command, FakeBackend, ScriptedCommands, TypedBackend, VoiceListener, VoskBackend,
                                 clean_target, parse_command, parse_script, vosk_status, words_to_digits)

VECTORS = json.loads((Path(__file__).parent / "data" / "voice_commands.json").read_text())


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


def test_every_vector_intent_is_known():
    from seeforme.live.voice import INTENTS
    assert set(VECTORS["intents"]) == set(INTENTS)
    for kind in ("wake", "direct"):
        assert all(c["intent"] is None or c["intent"] in INTENTS for c in VECTORS[kind])


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


def test_the_apps_own_phrases_change_nothing():
    """Heard back hands-free if echo suppression ever fails (e.g. another app plays them): never a mode, detail, mute,
    pause or quit, and the start-up phrase is at most "help"."""
    from seeforme.live.commands import HELP_TYPED, HELP_VOICE
    assert parse_command("SeeForMe ready. Say 'SeeForMe, help' for commands.").intent == "help"
    for phrase in (HELP_VOICE, HELP_TYPED, "Paused. I'll still warn you about hazards.", "Looking for a chair. I'll tell you when I see it."):
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
# Both checks run in one child interpreter, so the recogniser's native library and its 68 MB model are loaded and freed
# outside the pytest process.
_VOSK_CHILD = r"""
import json, subprocess, sys, tempfile, wave, os
from seeforme.live.voice import VoiceListener, VoskBackend

class Talking:
    def is_speaking(self):
        return True

out = {}
d = tempfile.mkdtemp()
for name, text, speaker in (("command", "See for me, find a chair.", None), ("echo", "See for me, stop.", Talking())):
    wav = os.path.join(d, name + ".wav")
    subprocess.run(["say", "-o", wav, "--data-format=LEI16@16000", text], check=True, timeout=30)
    with wave.open(wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        data = w.readframes(w.getnframes())
    data = b"\0\0" * 8000 + data + b"\0\0" * 16000          # silence around it ends the utterance
    blocks = [data[i:i + 8000] for i in range(0, len(data), 8000)]
    lis = VoiceListener(speaker)
    vb = lis.add(VoskBackend(audio_blocks=blocks))
    lis.start()
    vb.join(timeout=60)
    out[name] = {"error": vb.error, "results": vb.results, "commands": [[c.intent, c.arg] for c in lis.poll()],
                 "ignored_echo": lis.ignored_echo}
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


def test_vosk_backend_reports_a_missing_model(tmp_path, monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "vosk", types.ModuleType("vosk"))     # never the real library in this process
    lis = VoiceListener(None)
    vb = lis.add(VoskBackend(model_path=str(tmp_path / "missing"), audio_blocks=[]))
    lis.start()
    vb.join(timeout=10)
    assert vb.error and not vb.running
    assert any("Voice commands are off" in n for n in lis.notices())
