"""The macOS speaker: an interrupt must actually speak (stop waits for the old `say`), muting silences, haptic is a sound.
No audio is played: subprocess.Popen is replaced by a fake."""
import seeforme.feedback.speech as speech


class FakeProc:
    started = []

    def __init__(self, cmd, **kw):
        self.cmd, self.alive = cmd, True
        FakeProc.started.append(self)

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


def make(monkeypatch, enabled=True):
    FakeProc.started = []
    monkeypatch.setattr(speech.subprocess, "Popen", FakeProc)
    sp = speech.Speaker(enabled=enabled)
    sp.backend = "say"
    return sp


def test_interrupt_speaks_after_cutting_the_current_message(monkeypatch):
    sp = make(monkeypatch)
    sp.say("a long sign being read")
    sp.say("dropped while busy")                     # the channel is busy: dropped, as before
    sp.interrupt("Stop. Stairs down, one meter.")
    says = [p for p in FakeProc.started if p.cmd[0] == "say"]
    assert [p.cmd[-1] for p in says] == ["a long sign being read", "Stop. Stairs down, one meter."]
    assert not says[0].alive and says[1].alive and sp.busy()


def test_stop_then_say_also_works(monkeypatch):
    sp = make(monkeypatch)
    sp.say("first")
    sp.stop()
    sp.say("second")
    assert [p.cmd[-1] for p in FakeProc.started if p.cmd[0] == "say"] == ["first", "second"]


def test_muted_speaker_starts_no_process(monkeypatch):
    sp = make(monkeypatch)
    sp.enabled = False
    sp.say("person"); sp.interrupt("stairs"); sp.haptic()
    assert FakeProc.started == [] and not sp.busy()


def test_haptic_plays_a_sound_on_macos(monkeypatch):
    sp = make(monkeypatch)
    monkeypatch.setattr(speech.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/" + name)
    sp.haptic()
    assert FakeProc.started and FakeProc.started[0].cmd[0] == "afplay"


# ---- what voice commands need: is the demo speaking, and since when is it quiet? ----------------------------------
def test_is_speaking_and_when_speech_ended(monkeypatch):
    sp = make(monkeypatch)
    assert not sp.is_speaking() and sp.speech_ended_at == float("-inf")
    sp.say("Person, about 2 meters.")
    assert sp.is_speaking() and sp.busy()
    t0 = speech.time.monotonic()
    FakeProc.started[-1].alive = False                # `say` finished on its own
    assert not sp.is_speaking()
    assert t0 <= sp.speech_ended_at <= speech.time.monotonic()
    ended = sp.speech_ended_at
    assert sp.speech_ended_at == ended                # remembered, not re-stamped


def test_stop_and_interrupt_stamp_the_end(monkeypatch):
    sp = make(monkeypatch)
    sp.say("a long sign being read")
    t0 = speech.time.monotonic()
    sp.stop()
    assert not sp.is_speaking() and sp.speech_ended_at >= t0


def test_log_to_stdout_off_prints_nothing(monkeypatch, capsys):
    FakeProc.started = []
    monkeypatch.setattr(speech.subprocess, "Popen", FakeProc)
    sp = speech.Speaker(enabled=True, log_to_stdout=False)
    sp.backend = "say"
    sp.say("Person."); sp.interrupt("Stop."); sp.haptic(); sp.chime()
    assert capsys.readouterr().out == ""
    sp.log_to_stdout = True
    sp.say("Chair.")
    assert "[speech] Chair." in capsys.readouterr().out


def test_chime_is_a_short_sound_and_respects_mute(monkeypatch):
    sp = make(monkeypatch)
    monkeypatch.setattr(speech.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(speech.shutil, "which", lambda name: "/usr/bin/" + name)
    assert sp.chime() and FakeProc.started[-1].cmd[0] == "afplay" and FakeProc.started[-1].cmd[-1] == speech.CHIME
    assert not sp.is_speaking()                        # an earcon is not speech: it does not block the microphone
    sp.enabled = False
    n = len(FakeProc.started)
    assert not sp.chime() and len(FakeProc.started) == n
    sp.enabled, sp.backend = True, "print"
    assert not sp.chime()
