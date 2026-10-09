// Speech output (browser TTS or ElevenLabs through our server), haptics, and the audio-channel model.
import type { Cue } from "./types";

export type VoiceBackend = "browser" | "elevenlabs";
/** A cue is never spoken late: ElevenLabs audio has TTS_WAIT_MS to arrive before the browser voice says it instead, and a
 *  cue that still has not started TTS_LATE_MS after it was issued is dropped. */
const TTS_WAIT_MS = 1500, TTS_LATE_MS = 2000;

/** One AudioContext for the whole app: created/resumed by Speech.unlock() inside the Start gesture (iOS keeps a
 *  context created outside a gesture suspended), then reused by earcon(). */
let ctx: AudioContext | null = null;
function audioCtx(): AudioContext | null {
  try { ctx ??= new AudioContext(); } catch { return null; }
  if (ctx.state !== "running") void ctx.resume().catch(() => {});
  return ctx;
}

export class Speech {
  enabled = true;
  backend: VoiceBackend = "browser";
  private audio: HTMLAudioElement | null = null;
  private pending = false;
  private gen = 0; // bumped by stop()/reset(): an ElevenLabs fetch that resolves after that is dropped
  private cache = new Map<string, string>();
  busyUntil = -1e9;
  current: Cue | null = null;
  log: Array<[number, number, Cue]> = [];
  /** onSpoken: `urgent` marks an interrupt cue (critical hazard), not an answer that cuts in (sayNow) */
  constructor(private serverUrl: () => string, private onSpoken?: (text: string, urgent: boolean) => void) {}

  free(t: number) { return t >= this.busyUntil && !this.pending && !(this.audio && !this.audio.paused && !this.audio.ended); }

  /** Must run inside a user gesture (Start tap): unlocks audio playback and browser TTS on iOS/Safari. */
  unlock() {
    try {
      const a = new Audio("data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQxAADB8AhSmxhIIEVCSiJrDCQBTcu3UrAIwUdkRgQbFAZC1CQEwTJ9mjRvBA4UOLD8nKVOWfh+UlK3z/177OXrfOdKl7pyn3Xf//WreyTRUoAWgBgkOAGbZHBgG1OF6zM82DWbZaUmMBptgQhGjsyYqc9ae9XPAx/jRXw4V5JdQ1Zfzp3LnpVf/w0uSvtY2rnc3dFA7W1Me9jUYc/HYm2gi7ZYa9HhqfNXb+dr3W9qr++qrl+rtaf1qYKf///+9Ms3gXnmP3S8TgxEqmj3xJ9iI7kR2Hbp5p26Vq0h3SBAAAAA");
      a.volume = 0.01; void a.play().catch(() => {});
      if ("speechSynthesis" in window) { const u = new SpeechSynthesisUtterance(" "); u.volume = 0; window.speechSynthesis.speak(u); }
      audioCtx();
    } catch { /* ignore */ }
  }

  /** Speak free text (answers, confirmations) right away, cutting whatever is playing. */
  sayNow(text: string, t: number) {
    this.speak({ t, trackId: -1, label: "answer", message: text, action: "speak", duration: Math.max(1.2, text.split(/\s+/).length * 0.35), utility: 0, severity: 0, distance: null, ttc: null, slots: [] }, t, true);
  }
  remaining(t: number) { return Math.max(0, this.busyUntil - t); }

  /** Register the cue on the channel (even when muted, so the scheduler's model of the channel holds). */
  speak(cue: Cue, t: number, interrupt = false) {
    if (interrupt && this.log.length && this.busyUntil > t) this.log[this.log.length - 1][1] = t;
    this.busyUntil = t + cue.duration; this.current = cue; this.log.push([t, t + cue.duration, cue]);
    this.onSpoken?.(cue.message, cue.action === "interrupt");
    if (!this.enabled) return;
    if (interrupt) this.stop();
    // an interrupt cannot wait for a TTS round trip: the browser voice says it unless its ElevenLabs audio is cached
    if (this.backend === "elevenlabs" && (cue.action !== "interrupt" || this.cache.has(cue.message))) void this.playEleven(cue.message);
    else this.playBrowser(cue.message);
  }
  private playBrowser(text: string) {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.lang = "en-US";
    const voices = window.speechSynthesis.getVoices();
    const pick = voices.find((v) => /Samantha|Google US English|Karen|Moira|Zira|Jenny/i.test(v.name)) ?? voices.find((v) => v.lang.startsWith("en") && /female/i.test(v.name)) ?? null;
    if (pick) u.voice = pick;
    window.speechSynthesis.speak(u);
  }
  private async playEleven(text: string) {
    const gen = ++this.gen, t0 = performance.now();
    const stale = () => gen !== this.gen || !this.enabled; // muted, stopped or superseded while awaiting
    this.pending = true;
    try {
      let url = this.cache.get(text);
      if (!url) {
        // a slow or hung TTS request must not hold the channel or delay the cue: time out and fall back to the browser voice
        const r = await fetch(`${this.serverUrl()}/api/tts`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }), signal: AbortSignal.timeout(TTS_WAIT_MS) });
        if (!r.ok) throw new Error(`tts ${r.status}`);
        url = URL.createObjectURL(await r.blob()); this.cache.set(text, url);
      }
      if (stale() || performance.now() - t0 > TTS_LATE_MS) return;   // too late to be useful: dropped, never spoken late
      if (this.audio) this.audio.pause();
      const a = new Audio(url); this.audio = a; await a.play();
    } catch { if (!stale() && performance.now() - t0 <= TTS_LATE_MS) this.playBrowser(text); }
    finally { if (gen === this.gen) this.pending = false; }
  }
  stop() {
    this.gen++; this.pending = false;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    if (this.audio) { this.audio.pause(); this.audio = null; }
  }
  reset() { this.stop(); this.busyUntil = -1e9; this.current = null; this.log = []; }
}

export function haptic(pattern: number | number[] = [80, 60, 80]) {
  try { if ("vibrate" in navigator) navigator.vibrate(pattern); } catch { /* ignore */ }
}

/** Short earcon for haptic cues on devices without vibration (laptops). */
export function earcon(freq = 660, ms = 120) {
  try {
    const ac = audioCtx(); if (!ac) return;
    const o = ac.createOscillator(), g = ac.createGain();
    o.frequency.value = freq; o.connect(g); g.connect(ac.destination);
    g.gain.setValueAtTime(0.0001, ac.currentTime); g.gain.exponentialRampToValueAtTime(0.4, ac.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + ms / 1000);
    o.start(); o.stop(ac.currentTime + ms / 1000 + 0.02);
  } catch { /* ignore */ }
}
