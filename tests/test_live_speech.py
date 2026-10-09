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
