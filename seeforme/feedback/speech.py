"""Text-to-speech output. Uses macOS `say` when available, pyttsx3 otherwise, and
falls back to printing. Speech runs in a background thread so it never blocks
the camera loop; a new message is dropped if one is still playing (the runner's
audio-channel model is enforced in the policy loop, not here); interrupt() cuts it, haptic() plays a short
system sound on macOS, and a disabled speaker only prints."""
from __future__ import annotations

import platform
import shutil
import subprocess
import threading
from typing import Optional


class Speaker:
    def __init__(self, enabled: bool = True, rate: int = 200, voice: Optional[str] = None):
        self.enabled = enabled
        self.rate = rate
        self.voice = voice
        self._proc: Optional[subprocess.Popen] = None
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

    def busy(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def say(self, text: str) -> None:
        """Speak unless muted or already speaking (the channel model lives in the policy loop)."""
        print(f"[speech] {text}" if self.enabled else f"[muted] {text}", flush=True)
        if not self.enabled:
            return
        if self.backend == "say":
            with self._lock:
                if self._proc is not None and self._proc.poll() is None:
                    return
                self._start_say(text)
        elif self.backend == "pyttsx3":
            def _run():
                import pyttsx3
                eng = pyttsx3.init(); eng.setProperty("rate", self.rate); eng.say(text); eng.runAndWait()
            threading.Thread(target=_run, daemon=True).start()

    def interrupt(self, text: str) -> None:
        """Cut the current message and speak this one (an INTERRUPT cue): unlike stop() + say(), the old process is
        gone before the new one starts, so the interrupting message is never dropped."""
        print(f"[interrupt] {text}" if self.enabled else f"[muted] {text}", flush=True)
        if not self.enabled:
            return
        if self.backend == "say":
            with self._lock:
                self._stop_locked()
                self._start_say(text)
        else:
            self.stop()
            self.say(text)

    def haptic(self) -> None:
        """A laptop has no vibration motor: play a short system sound instead (macOS), else print."""
        print("[haptic] buzz", flush=True)
        if self.enabled and platform.system() == "Darwin" and shutil.which("afplay"):
            snd = "/System/Library/Sounds/Tink.aiff"
            try:
                subprocess.Popen(["afplay", "-v", "0.6", snd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    # ---- internals (call with the lock held) ----------------------------------------------------------------
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
        self._proc = None
