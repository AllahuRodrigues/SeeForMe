// Plays the recorded voice pack (voicepack.ts) through Web Audio: one sprite file, decoded once, and each sentence is a
// few clips scheduled back to back on the audio clock, so the joins are sample-accurate and the length of a sentence is
// known exactly before it starts. Nothing here synthesises speech; a sentence the pack does not hold is not played.
import { planClips, planSeconds, type PlannedClip, type VoiceIndex } from "./voicepack";

/** Silence kept around each clip when it is cut out of the sprite: MP3 decoders differ by a few tens of milliseconds in
 *  where the audio starts, and the sprite has 200 ms of silence between clips, so this margin can never reach a neighbour. */
const PRE_S = 0.03, POST_S = 0.03;
/** Scheduling a hair ahead of the audio clock avoids a clipped first sample on a busy main thread. */
const LEAD_S = 0.03;

export interface ClipHandle { durationS: number; stop(): void }
type Decode = (bytes: ArrayBuffer) => Promise<AudioBuffer>;

/** Decode at the pack's own rate: a 48 kHz context would hold the sprite twice as large in memory. */
const decodeAt24k: Decode = (bytes) => new OfflineAudioContext(1, 1, 24000).decodeAudioData(bytes);

export class ClipVoice {
  index: VoiceIndex | null = null;
  private buffer: AudioBuffer | null = null;
  private loading: Promise<boolean> | null = null;
  private live: Array<{ nodes: AudioBufferSourceNode[]; end: (completed: boolean) => void }> = [];

  /** `base`: folder with index.json and the sprite, e.g. "/voice/af_heart". */
  constructor(private base: string, private fetcher: typeof fetch = (...a) => fetch(...a), private decode: Decode = decodeAt24k) {}

  get ready() { return this.buffer != null && this.index != null; }
  get playing() { return this.live.length > 0; }

  /** Fetch and decode the pack once. Resolves false when it cannot be loaded; the app then keeps its other voice. */
  load(): Promise<boolean> {
    this.loading ??= (async () => {
      try {
        const index = await (await this.fetcher(`${this.base}/index.json`)).json() as VoiceIndex;
        const bytes = await (await this.fetcher(`${this.base}/${index.sprite}`)).arrayBuffer();
        this.buffer = await this.decode(bytes); this.index = index;
        return true;
      } catch { this.loading = null; return false; }
    })();
    return this.loading;
  }

  /** The clips that say `message`, or null when the pack is not loaded or does not hold every part of it. */
  plan(message: string): PlannedClip[] | null {
    const idx = this.index;
    return idx && this.buffer ? planClips(message, (k) => k in idx.clips) : null;
  }

  /** Exact playing time of a plan, margins included. */
  seconds(plan: readonly PlannedClip[]): number {
    return planSeconds(plan, this.index!) + plan.length * PRE_S + POST_S;
  }

  /** Schedule a plan on `ctx`. onStart fires as the first clip begins; onEnd fires once, with whether it played to the end. */
  play(plan: readonly PlannedClip[], ctx: AudioContext, onStart: () => void, onEnd: (completed: boolean) => void): ClipHandle {
    const buffer = this.buffer!, clips = this.index!.clips;
    const nodes: AudioBufferSourceNode[] = [];
    let when = ctx.currentTime + LEAD_S, done = false;
    const entry = { nodes, end: (completed: boolean) => {
      if (done) return;
      done = true; clearTimeout(startTimer);
      this.live = this.live.filter((e) => e !== entry);
      for (const n of nodes) { n.onended = null; if (!completed) { try { n.stop(); } catch { /* not started yet */ } } }
      onEnd(completed);
    } };
    for (const c of plan) {
      const [start, dur] = clips[c.key];
      const node = ctx.createBufferSource();
      node.buffer = buffer; node.connect(ctx.destination);
      node.start(when, Math.max(0, start - PRE_S), dur + PRE_S + POST_S);
      nodes.push(node);
      when += PRE_S + dur + c.gapAfterS;
    }
    nodes[nodes.length - 1].onended = () => entry.end(true);
    const startTimer = setTimeout(() => { if (!done) onStart(); }, LEAD_S * 1000);
    this.live.push(entry);
    return { durationS: this.seconds(plan), stop: () => entry.end(false) };
  }

  /** Cut everything that is sounding or scheduled. */
  stop() { for (const e of [...this.live]) e.end(false); }
}
