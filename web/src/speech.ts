// Speech output (browser TTS or ElevenLabs through our server), haptics, and the audio-channel model.
import type { Cue } from "./types";
import { PlaybackLedger, type BudgetHolder } from "./playback";
import type { ClipVoice } from "./clipvoice";
import type { PlannedClip } from "./voicepack";

/** natural: the recorded voice pack, and the browser voice for any sentence the pack does not hold. */
export type VoiceBackend = "natural" | "browser" | "elevenlabs";
/** A cue is never spoken late: ElevenLabs audio has TTS_WAIT_MS to arrive before the browser voice says it instead, and a
 *  cue that still has not started TTS_LATE_MS after it was issued is dropped. */
const TTS_WAIT_MS = 1500, TTS_LATE_MS = 2000;

/** When the app last made a sound (performance.now() ms): the voice input ignores what it hears while the app speaks
 *  and for a moment after, so the app never hears itself. */
let soundUntilMs = 0;
const markSound = (ms = 0) => { soundUntilMs = Math.max(soundUntilMs, performance.now() + ms); };

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
  /** The recorded voice (main.ts sets it and loads it at start-up). Until it has loaded, the browser voice speaks. */
  clips: ClipVoice | null = null;
  private audio: HTMLAudioElement | null = null;
  private silent: HTMLAudioElement | null = null;   // the unlock clip, kept for iOS (see unlock)
  private pending = false;
  private gen = 0; // bumped by stop()/reset(): an ElevenLabs fetch that resolves after that is dropped
  private cache = new Map<string, string>();
  busyUntil = -1e9;
  current: Cue | null = null;
  log: Array<[number, number, Cue]> = [];
  /** What each cue really played (start, end, how it ended), settled against the scheduler's speech budget. */
  readonly ledger: PlaybackLedger;
  /** onSpoken: `urgent` marks an interrupt cue (critical hazard), not an answer that cuts in (sayNow).
   *  budgetOf: the scheduler whose speech budget the cues were charged to (main.ts swaps schedulers, hence a getter). */
  constructor(private serverUrl: () => string, private onSpoken?: (text: string, urgent: boolean) => void,
              budgetOf: () => BudgetHolder | null | undefined = () => null) { this.ledger = new PlaybackLedger(budgetOf); }

  /** what was said (text, performance.now() ms when it was issued), newest last: the voice input's echo guard */
  private said: Array<[string, number]> = [];

  /** Free once the estimated end has passed and no cue is still sounding (a voice slower than the estimate keeps it busy). */
  free(t: number) { return t >= this.busyUntil && !this.pending && !(this.audio && !this.audio.paused && !this.audio.ended) && !this.ledger.active(); }

  /** Is the app making a sound now, or did it stop less than `tailMs` ago? (browser voice, ElevenLabs audio, earcons) */
  audible(tailMs = 400): boolean {
    const now = performance.now();
    const playing = this.pending || (this.audio != null && !this.audio.paused && !this.audio.ended) || !!this.clips?.playing
      || ("speechSynthesis" in globalThis && (globalThis as any).speechSynthesis.speaking);
    if (playing) markSound();
    return playing || now < soundUntilMs + tailMs;
  }
  /** Texts spoken in the last `ms` milliseconds (counted from when they were issued plus their spoken length). */
  recent(ms = 3000): string[] {
    const now = performance.now();
    this.said = this.said.filter(([text, at]) => now - at < ms + 400 * text.split(/\s+/).length);
    return this.said.map(([text]) => text);
  }

  /** Must run inside a user gesture (Start tap): unlocks audio playback and browser TTS on iOS/Safari. */
  unlock() {
    try {
      const a = new Audio("data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQxAADB8AhSmxhIIEVCSiJrDCQBTcu3UrAIwUdkRgQbFAZC1CQEwTJ9mjRvBA4UOLD8nKVOWfh+UlK3z/177OXrfOdKl7pyn3Xf//WreyTRUoAWgBgkOAGbZHBgG1OF6zM82DWbZaUmMBptgQhGjsyYqc9ae9XPAx/jRXw4V5JdQ1Zfzp3LnpVf/w0uSvtY2rnc3dFA7W1Me9jUYc/HYm2gi7ZYa9HhqfNXb+dr3W9qr++qrl+rtaf1qYKf///+9Ms3gXnmP3S8TgxEqmj3xJ9iI7kR2Hbp5p26Vq0h3SBAAAAA");
      a.volume = 0.01; void a.play().catch(() => {});
      this.silent ??= a;
      if ("speechSynthesis" in window) { const u = new SpeechSynthesisUtterance(" "); u.volume = 0; window.speechSynthesis.speak(u); }
      audioCtx();
      // iOS plays Web Audio as "ambient" sound, which the ring/silent switch mutes; speech for a blind user must not be.
      // Where the page can name its audio session it asks for "playback"; elsewhere a silent, looping media element
      // (media elements are never "ambient") keeps the session in playback while the app is open.
      const session = (navigator as unknown as { audioSession?: { type: string } }).audioSession;
      if (session) session.type = "playback";
      else if (this.silent && /iP(hone|ad|od)|Macintosh.*Mobile/.test(navigator.userAgent)) { this.silent.loop = true; void this.silent.play().catch(() => {}); }
    } catch { /* ignore */ }
  }

  /** Speak free text (answers, confirmations) right away, cutting whatever is playing. */
  sayNow(text: string, t: number) {
    this.speak({ t, trackId: -1, label: "answer", message: text, action: "speak", duration: Math.max(1.2, text.split(/\s+/).length * 0.35), utility: 0, severity: 0, distance: null, ttc: null, slots: [] }, t, true, false);
  }
  remaining(t: number) { return Math.max(0, this.busyUntil - t); }

  /** Register the cue on the channel (even when muted, so the scheduler's model of the channel holds).
   *  charged: the scheduler spent speech budget on this cue (false for answers), so its playback is settled against it. */
  speak(cue: Cue, t: number, interrupt = false, charged = true) {
    if (interrupt && this.log.length && this.busyUntil > t) this.log[this.log.length - 1][1] = t;
    this.busyUntil = t + cue.duration; this.current = cue; this.log.push([t, t + cue.duration, cue]);
    this.onSpoken?.(cue.message, cue.action === "interrupt");
    if (!this.enabled) return;
    if (interrupt) this.stop();
    if (cue.message) { this.said.push([cue.message, performance.now()]); if (this.said.length > 12) this.said.shift(); }
    markSound(cue.duration * 1000);   // until its estimated end, even before the voice actually starts
    // an interrupt cannot wait for a TTS round trip: the browser voice says it unless its ElevenLabs audio is cached
    // the recorded voice needs no round trip and no synthesis, so it serves interrupts too; a sentence it does not
    // hold, or a pack that has not loaded yet, goes to the browser voice at once (a cue never waits for a voice)
    const plan = this.backend === "natural" ? this.clips?.plan(cue.message) ?? null : null;
    const actx = plan ? audioCtx() : null;
    if (plan && actx && actx.state === "running") { this.playClips(plan, actx, this.ledger.issue(cue.message, cue.duration, "natural", charged), t); return; }
    const id = this.ledger.issue(cue.message, cue.duration, this.backend === "elevenlabs" ? "elevenlabs" : "browser", charged);
    if (this.backend === "elevenlabs" && (cue.action !== "interrupt" || this.cache.has(cue.message))) void this.playEleven(cue.message, id);
    else this.playBrowser(cue.message, id);
  }
  /** Recorded clips: the length is known exactly before the first sample, so the channel model is exact too. */
  private playClips(plan: PlannedClip[], actx: AudioContext, id: number, t: number) {
    const secs = this.clips!.seconds(plan);
    this.busyUntil = t + secs;
    if (this.log.length) this.log[this.log.length - 1][1] = t + secs;
    markSound(secs * 1000);
    this.clips!.play(plan, actx, () => this.ledger.started(id), (completed) => { markSound(); this.ledger.ended(id, completed ? "completed" : "interrupted"); });
  }
  private playBrowser(text: string, id: number) {
    if (!("speechSynthesis" in window)) { this.ledger.ended(id, "failed"); return; }
    const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.lang = "en-US";
    const voices = window.speechSynthesis.getVoices();
    const pick = voices.find((v) => /Samantha|Google US English|Karen|Moira|Zira|Jenny/i.test(v.name)) ?? voices.find((v) => v.lang.startsWith("en") && /female/i.test(v.name)) ?? null;
    if (pick) u.voice = pick;
    u.onstart = () => this.ledger.started(id);
    u.onend = () => { markSound(); this.ledger.ended(id, "completed"); };
    u.onerror = (e) => {
      markSound();
      const why = (e as SpeechSynthesisErrorEvent).error;
      this.ledger.ended(id, why === "interrupted" || why === "canceled" ? "interrupted" : "failed");
    };
    window.speechSynthesis.speak(u);
  }
  private async playEleven(text: string, id: number) {
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
      if (stale() || performance.now() - t0 > TTS_LATE_MS) { this.ledger.ended(id, "cancelled"); return; }   // too late to be useful: dropped, never spoken late
      if (this.audio) this.audio.pause();
      const a = new Audio(url); this.audio = a;
      a.onplaying = () => this.ledger.started(id);
      // the element knows how much it played; whichever of pause / ended comes first settles the cue
      const done = () => { markSound(); this.ledger.ended(id, a.ended || (a.duration > 0 && a.currentTime >= a.duration - 0.05) ? "completed" : "interrupted", a.currentTime); };
      a.onended = a.onpause = done;
      a.onerror = () => this.ledger.ended(id, "failed", a.currentTime || 0);
      await a.play();
    } catch {
      if (!stale() && performance.now() - t0 <= TTS_LATE_MS) this.playBrowser(text, id);
      else this.ledger.ended(id, "failed");
    }
    finally { if (gen === this.gen) this.pending = false; }
  }
  stop() {
    this.gen++; this.pending = false;
    this.ledger.abandonAll();           // whatever was sounding is charged for what was heard, the rest is refunded
    this.clips?.stop();
    soundUntilMs = performance.now();   // silent from now on (the estimated end of a cut cue no longer applies)
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    if (this.audio) { this.audio.pause(); this.audio = null; }
  }
  reset() { this.stop(); this.busyUntil = -1e9; this.current = null; this.log = []; this.ledger.clear(); }
}

export function haptic(pattern: number | number[] = [80, 60, 80]) {
  try { if ("vibrate" in navigator) navigator.vibrate(pattern); } catch { /* ignore */ }
}

/** Short earcon for haptic cues on devices without vibration (laptops), and the voice input's acknowledgement. */
export function earcon(freq = 660, ms = 120, delayMs = 0) {
  try {
    const ac = audioCtx(); if (!ac) return;
    markSound(delayMs + ms + 20);
    if (delayMs > 0) { setTimeout(() => earcon(freq, ms), delayMs); return; }
    const o = ac.createOscillator(), g = ac.createGain();
    o.frequency.value = freq; o.connect(g); g.connect(ac.destination);
    g.gain.setValueAtTime(0.0001, ac.currentTime); g.gain.exponentialRampToValueAtTime(0.4, ac.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + ms / 1000);
    o.start(); o.stop(ac.currentTime + ms / 1000 + 0.02);
  } catch { /* ignore */ }
}
