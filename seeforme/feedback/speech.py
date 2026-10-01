"""Text-to-speech output. Uses macOS `say` when available, pyttsx3 otherwise, and
falls back to printing. Speech runs in a background thread so it never blocks
the camera loop; a new message is dropped if one is still playing (the runner's
audio-channel model is enforced in the policy loop, not here)."""
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
        print(f"[speech] {text}", flush=True)
        if self.backend == "say":
            with self._lock:
                if self._proc is not None and self._proc.poll() is None:
                    return
                cmd = ["say", "-r", str(self.rate)]
                if self.voice:
                    cmd += ["-v", self.voice]
                self._proc = subprocess.Popen(cmd + [text])
        elif self.backend == "pyttsx3":
            def _run():
                import pyttsx3
                eng = pyttsx3.init(); eng.setProperty("rate", self.rate); eng.say(text); eng.runAndWait()
            threading.Thread(target=_run, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
