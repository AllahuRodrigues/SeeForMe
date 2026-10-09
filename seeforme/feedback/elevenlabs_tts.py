"""ElevenLabs text-to-speech for the live pipeline and the web server.

The key is read from the environment / .env (`ELEVENLABS_API_KEY` or `elevenlabs_API_KEY`) and
never leaves the server side: the browser asks our own `/api/tts` endpoint. Audio is cached per
message (navigation cues repeat a lot) and played with `afplay` on macOS. `interrupt()` kills the
player so an urgent cue can cut a running message, matching the scheduler's INTERRUPT action.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

import requests

DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"      # "Sarah" (female, clear, neutral); available on the free tier
MODEL_ID = "eleven_flash_v2_5"               # lowest latency tier


def load_api_key() -> Optional[str]:
    for name in ("ELEVENLABS_API_KEY", "elevenlabs_API_KEY", "ELEVEN_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from dotenv import dotenv_values
        here = Path(__file__).resolve().parents[2]
        for env_path in (here / ".env", Path.cwd() / ".env"):
            if env_path.exists():
                vals = dotenv_values(env_path)
                for name in ("ELEVENLABS_API_KEY", "elevenlabs_API_KEY", "ELEVEN_API_KEY"):
                    if vals.get(name):
                        return vals[name]
    except Exception:
        pass
    return None


class ElevenLabsTTS:
    def __init__(self, api_key: Optional[str] = None, voice_id: str = DEFAULT_VOICE, model_id: str = MODEL_ID,
                 cache_dir: Optional[str] = None, speed: float = 1.0):
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise RuntimeError("no ElevenLabs API key (set ELEVENLABS_API_KEY or put it in .env)")
        self.voice_id, self.model_id, self.speed = voice_id, model_id, speed
        self.cache = Path(cache_dir or Path.home() / ".cache" / "seeforme" / "tts")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"xi-api-key": self.api_key, "accept": "audio/mpeg"})

    def synthesize(self, text: str) -> bytes:
        key = hashlib.sha1(f"{self.voice_id}|{self.model_id}|{self.speed}|{text}".encode()).hexdigest()
        f = self.cache / f"{key}.mp3"
        if f.exists():
            return f.read_bytes()
        r = self.session.post(f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
                              params={"output_format": "mp3_22050_32", "optimize_streaming_latency": 3},
                              json={"text": text, "model_id": self.model_id,
                                    "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.15, "use_speaker_boost": True, "speed": self.speed}},
                              timeout=15)
        r.raise_for_status()
        f.write_bytes(r.content)
        return r.content


class ElevenLabsSpeaker:
    """Drop-in for feedback.speech.Speaker: say() / busy() / stop() / interrupt()."""

    backend = "elevenlabs"

    def __init__(self, enabled: bool = True, **kw):
        self.enabled = enabled
        self.tts = ElevenLabsTTS(**kw)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._pending = False

    def busy(self) -> bool:
        with self._lock:
            return self._pending or (self._proc is not None and self._proc.poll() is None)

    def say(self, text: str, interrupt: bool = False) -> None:
        print(f"[speech] {text}", flush=True)
        if not self.enabled:
            return
        if interrupt:
            self.stop()
        elif self.busy():
            return
        with self._lock:
            self._pending = True

        def _run():
            try:
                audio = self.tts.synthesize(text)
                path = self.tts.cache / "last.mp3"
                path.write_bytes(audio)
                with self._lock:
                    self._pending = False
                    self._proc = subprocess.Popen(["afplay", str(path)])
            except Exception as e:  # network / quota: fall back to macOS say so the user still hears the cue
                print(f"[speech] elevenlabs failed ({e}); using say", flush=True)
                with self._lock:
                    self._pending = False
                    self._proc = subprocess.Popen(["say", "-r", "200", text])
        threading.Thread(target=_run, daemon=True).start()

    def interrupt(self, text: str) -> None:
        self.say(text, interrupt=True)

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
