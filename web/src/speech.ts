// Speech output (browser TTS or ElevenLabs through our server), haptics, and the audio-channel model.
import type { Cue } from "./types";

export type VoiceBackend = "browser" | "elevenlabs";

export class Speech {
  enabled = true;
  backend: VoiceBackend = "browser";
  private audio: HTMLAudioElement | null = null;
  private pending = false;
  private cache = new Map<string, string>();
  busyUntil = -1e9;
  current: Cue | null = null;
  log: Array<[number, number, Cue]> = [];
  constructor(private serverUrl: () => string, private onSpoken?: (text: string) => void) {}

  free(t: number) { return t >= this.busyUntil && !this.pending && !(this.audio && !this.audio.paused && !this.audio.ended); }
  remaining(t: number) { return Math.max(0, this.busyUntil - t); }

  /** Register the cue on the channel (even when muted, so the scheduler's model of the channel holds). */
  speak(cue: Cue, t: number, interrupt = false) {
    if (interrupt && this.log.length && this.busyUntil > t) this.log[this.log.length - 1][1] = t;
    this.busyUntil = t + cue.duration; this.current = cue; this.log.push([t, t + cue.duration, cue]);
    this.onSpoken?.(cue.message);
    if (!this.enabled) return;
    if (interrupt) this.stop();
    if (this.backend === "elevenlabs") this.playEleven(cue.message).catch(() => this.playBrowser(cue.message));
    else this.playBrowser(cue.message);
  }
  private playBrowser(text: string) {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text); u.rate = 1.15; u.lang = "en-US";
    window.speechSynthesis.speak(u);
  }
  private async playEleven(text: string) {
    this.pending = true;
    try {
      let url = this.cache.get(text);
      if (!url) {
        const r = await fetch(`${this.serverUrl()}/api/tts`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) });
        if (!r.ok) throw new Error(`tts ${r.status}`);
        url = URL.createObjectURL(await r.blob()); this.cache.set(text, url);
      }
      this.audio = new Audio(url); await this.audio.play();
    } finally { this.pending = false; }
  }
  stop() {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    if (this.audio) { this.audio.pause(); this.audio = null; }
  }
  reset() { this.stop(); this.busyUntil = -1e9; this.current = null; this.log = []; }
}

export function haptic(pattern: number | number[] = [80, 60, 80]) {
  try { if ("vibrate" in navigator) navigator.vibrate(pattern); } catch { /* ignore */ }
}

/** Short earcon for haptic cues on devices without vibration (laptops). */
let ctx: AudioContext | null = null;
export function earcon(freq = 660, ms = 120) {
  try {
    ctx ??= new AudioContext();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = freq; o.connect(g); g.connect(ctx.destination);
    g.gain.setValueAtTime(0.0001, ctx.currentTime); g.gain.exponentialRampToValueAtTime(0.4, ctx.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + ms / 1000);
    o.start(); o.stop(ctx.currentTime + ms / 1000 + 0.02);
  } catch { /* ignore */ }
}
