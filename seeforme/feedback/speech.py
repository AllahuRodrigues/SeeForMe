"""Text-to-speech output. Uses macOS `say` when available, pyttsx3 otherwise, and
falls back to printing. Speech runs in a background thread so it never blocks
the camera loop; a new message is dropped if one is still playing (the runner's
audio-channel model is enforced in the policy loop, not here); interrupt() cuts it, haptic() plays a short
system sound on macOS, chime() a short earcon (the wake-word acknowledgement), and a disabled speaker only prints.

Voice commands need to know when the speaker talks, so the microphone never obeys the demo's own voice:
is_speaking() and speech_ended_at (time.monotonic() when the last message was first seen to have ended; it is checked
at every scheduler decision and every audio block, so it errs late, i.e. towards ignoring the microphone longer).
log_to_stdout=False leaves printing to the caller (scripts/run_live.py prints one aligned line per message)."""
from __future__ import annotations

import math
import platform
import shutil
import subprocess
import threading
import time
from typing import Optional

CHIME = "/System/Library/Sounds/Pop.aiff"


class Speaker:
    def __init__(self, enabled: bool = True, rate: int = 200, voice: Optional[str] = None, log_to_stdout: bool = True):
        self.enabled = enabled
        self.rate = rate
        self.voice = voice
        self.log_to_stdout = log_to_stdout
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None     # pyttsx3
        self._ended_at = -math.inf
        self._lock = threading.Lock()
        self.backend = "print"
        if enabled and platform.system() == "Darwin" and shutil.which("say"):
            self.backend = "say"
        elif enabled:
            try:
                import pyttsx3  # noqa: F401
                self.backend = "pyttsx3"
            except Exception:
                self.backend = "print"

    def _log(self, line: str) -> None:
        if self.log_to_stdout:
            print(line, flush=True)

    def is_speaking(self) -> bool:
        with self._lock:
            return self._alive_locked()

    def busy(self) -> bool:
        return self.is_speaking()

    @property
    def speech_ended_at(self) -> float:
        """time.monotonic() when the last message ended (now while one is playing; -inf if nothing was said)."""
        with self._lock:
            return time.monotonic() if self._alive_locked() else self._ended_at

    def say(self, text: str) -> None:
        """Speak unless muted or already speaking (the channel model lives in the policy loop)."""
        self._log(f"[speech] {text}" if self.enabled else f"[muted] {text}")
        if not self.enabled:
            return
        if self.backend == "say":
            with self._lock:
                if self._alive_locked():
                    return
                self._start_say(text)
        elif self.backend == "pyttsx3":
            def _run():
                import pyttsx3
                eng = pyttsx3.init(); eng.setProperty("rate", self.rate); eng.say(text); eng.runAndWait()
            th = threading.Thread(target=_run, daemon=True)
            with self._lock:
                self._thread = th
            th.start()

    def interrupt(self, text: str) -> None:
        """Cut the current message and speak this one (an INTERRUPT cue): unlike stop() + say(), the old process is
        gone before the new one starts, so the interrupting message is never dropped."""
        self._log(f"[interrupt] {text}" if self.enabled else f"[muted] {text}")
        if not self.enabled:
            return
        if self.backend == "say":
            with self._lock:
                self._stop_locked()
                self._start_say(text)
        else:
            log, self.log_to_stdout = self.log_to_stdout, False     # already printed as [interrupt]
            try:
                self.stop()
                self.say(text)
            finally:
                self.log_to_stdout = log

    def haptic(self) -> None:
        """A laptop has no vibration motor: play a short system sound instead (macOS), else print."""
        self._log("[haptic] buzz")
        if self.enabled and platform.system() == "Darwin" and shutil.which("afplay"):
            snd = "/System/Library/Sounds/Tink.aiff"
            try:
                subprocess.Popen(["afplay", "-v", "0.6", snd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass

    def chime(self) -> bool:
        """A short earcon (macOS Pop) that acknowledges the wake word with nothing to wait for. False when no sound was
        played (muted, no audio backend, not macOS): the caller then says something short instead."""
        self._log("[chime]")
        if not self.enabled or self.backend == "print" or platform.system() != "Darwin" or not shutil.which("afplay"):
            return False
        try:
            subprocess.Popen(["afplay", "-v", "0.5", CHIME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            return False

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    # ---- internals (call with the lock held) ----------------------------------------------------------------
    def _alive_locked(self) -> bool:
        alive = False
        if self._proc is not None:
            if self._proc.poll() is None:
                alive = True
            else:
                self._proc = None
                self._ended_at = time.monotonic()
        if self._thread is not None:
            if self._thread.is_alive():
                alive = True
            else:
                self._thread = None
                self._ended_at = time.monotonic()
        return alive

    def _start_say(self, text: str) -> None:
        cmd = ["say", "-r", str(self.rate)]
        if self.voice:
            cmd += ["-v", self.voice]
        self._proc = subprocess.Popen(cmd + [text])

    def _stop_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._ended_at = time.monotonic()
        self._proc = None
