// Learned scheduler (GRU + RL) in the browser: a port of seeforme/ml/scheduler_model.py SchedulerPolicy in mode "rl".
//
// Per decision step (same order as Python):
//   1. refill the speech budget, compute the 29 EventFeatures per track (SchedulerBase.candidates, Python's _features)
//   2. keep candidates with confidence >= minConfidence; append each one's feature vector to its track's history (last K)
//   3. run the exported model (public/models/scheduler_rl.onnx) on the right-aligned windows
//   4. legality per candidate: speak needs a free channel, interrupt a busy one (not the track being spoken), neither within
//      recentBlock of the track's last announcement; haptic needs hapticRepeat since the track's last buzz
//   5. one argmax over [speak/interrupt/haptic log-odds against each candidate's own silence logit (illegal = -1e9)]
//      followed by the learned global silence bias: the first maximum wins, as torch.argmax
//   6. the chosen candidate's message uses the slots the model's slot head turns on (sigmoid > 0.5) plus "object"
// `decideVectors` (steps 2-5) is the function web/parity/run.mjs checks against Python decisions.
import type * as OrtNs from "onnxruntime-web";
import { SLOTS, compose, speechSeconds } from "./compose";
import { FEATURE_NAMES, SchedulerBase, featureVector, type Candidate, type PCACSConfig } from "./pcacs";
import type { Camera } from "./tracker";
import type { Action, ChannelState, Cue, Track } from "./types";

export const ACTIONS = ["silence", "speak", "interrupt", "haptic"] as const;
const MASKED = Math.fround(-1e9);

/** public/config/scheduler.json, written by scripts/export_web_scheduler.py */
export interface SchedulerMeta {
  version: string; model_url: string; K: number; mode: string; silence_bias: number;
  features: string[]; actions: string[]; slots: string[]; source_sha256?: string; onnx_sha256?: string;
}
const RUN_TIMEOUT_MS = 2000;

export interface ModelOutput { action: Float32Array; slots: Float32Array; silenceBias: number }
/** windows: n x K x 29 float32, right-aligned; mask: n x K (1 = real step) */
export type RunModel = (windows: Float32Array, mask: Float32Array, n: number, K: number) => Promise<ModelOutput>;

export interface DecisionInput { trackId: number; vector: ArrayLike<number>; lastAnnounced: number | null; lastHaptic: number | null }
export interface Decision {
  candidate: number;        // index into the inputs, -1 = silence
  action: Action;
  scores: number[];         // per candidate: best legal log-odds (-Infinity when nothing is legal)
  best: Action[];           // per candidate: the action behind its score
  value: number;            // the winning entry of the joint categorical
  margin: number;           // winner minus runner-up (how close the decision was)
  logits: Float32Array; slotLogits: Float32Array; silenceBias: number;
}

/** Returns why the exported model cannot drive this build of the app, or null when it can. */
export function checkMeta(meta: SchedulerMeta): string | null {
  if (meta.mode !== "rl") return `mode ${meta.mode} (only rl is supported)`;
  if (!Number.isInteger(meta.K) || meta.K < 1) return `bad window length K=${meta.K}`;
  if (meta.features.length !== FEATURE_NAMES.length || meta.features.some((f, i) => f !== FEATURE_NAMES[i])) return "feature list differs from this app's EventFeatures order";
  if (meta.actions.join() !== ACTIONS.join()) return "action list differs";
  if (meta.slots.join() !== SLOTS.join()) return "slot list differs";
  return null;
}

/** Which of silence / speak / interrupt / haptic the policy may choose for this candidate (SchedulerPolicy.decide). */
export function legalActions(inp: DecisionInput, t: number, ch: ChannelState, cfg: PCACSConfig, useHaptic = true): boolean[] {
  const recent = inp.lastAnnounced != null && t - inp.lastAnnounced < cfg.recentBlock;
  return [true, !ch.busy && !recent, ch.busy && !recent && ch.currentTrack !== inp.trackId,
    useHaptic && (inp.lastHaptic == null || t - inp.lastHaptic >= cfg.hapticRepeat)];
}

/** The RL rule: argmax over [candidate x {speak, interrupt, haptic}] log-odds (float32, illegal = -1e9) + [silence bias]. */
export function rlChoose(logits: Float32Array, legal: boolean[][], silenceBias: number) {
  const n = legal.length, scores: number[] = [], best: Action[] = [];
  let top = -Infinity, second = -Infinity, arg = -1;
  const offer = (v: number, idx: number) => {
    if (v > top) { second = top; top = v; arg = idx; } else if (v > second) second = v;
  };
  for (let i = 0; i < n; i++) {
    let s = -Infinity, sa: Action = "silence";
    for (let j = 1; j < 4; j++) {
      const v = legal[i][j] ? Math.fround(logits[i * 4 + j] - logits[i * 4]) : MASKED;
      offer(v, i * 3 + j - 1);
      if (v > s) { s = v; sa = ACTIONS[j]; }
    }
    scores.push(s > -1e8 ? s : -Infinity); best.push(s > -1e8 ? sa : "silence");
  }
  offer(Math.fround(silenceBias), 3 * n);
  const candidate = arg === 3 * n ? -1 : Math.floor(arg / 3);
  return { candidate, action: (candidate < 0 ? "silence" : ACTIONS[(arg % 3) + 1]) as Action, scores, best, value: top, margin: top - second };
}

/** Slots the model's slot head turns on for candidate i: "object" + every other slot with sigmoid(logit) > 0.5. */
export function slotsFromLogits(slotLogits: Float32Array, i: number): string[] {
  const out = ["object"];
  SLOTS.forEach((s, j) => { if (s !== "object" && Math.fround(1 / (1 + Math.exp(-slotLogits[i * SLOTS.length + j]))) > 0.5) out.push(s); });
  return out;
}

export class LearnedScheduler extends SchedulerBase {
  readonly kind = "learned" as const;
  readonly K: number;
  useHaptic = true;
  /** per-track history of the last K feature vectors (float32, as Python's np.float32 window) */
  hist = new Map<number, Float32Array[]>();
  seen = new Map<number, number>();
  step = 0;
  lastDecision: Decision | null = null;
  silenceBias: number;

  constructor(cfg: PCACSConfig, cam: Camera, public meta: SchedulerMeta, private run: RunModel) {
    super(cfg, cam);
    this.K = meta.K; this.silenceBias = meta.silence_bias;
  }
  reset() { super.reset(); this.hist.clear(); this.seen.clear(); this.step = 0; this.lastDecision = null; }

  /** Steps 2-5 on given feature vectors (inputs in track order, already confidence-filtered). Updates the histories. */
  async decideVectors(inputs: DecisionInput[], t: number, ch: ChannelState): Promise<Decision | null> {
    this.step++;
    if (!inputs.length) return null;
    const K = this.K, D = FEATURE_NAMES.length, n = inputs.length;
    const windows = new Float32Array(n * K * D), mask = new Float32Array(n * K);
    inputs.forEach((inp, i) => {
      if (inp.vector.length !== D) throw new Error(`feature vector has ${inp.vector.length} values, expected ${D}`);
      let h = this.hist.get(inp.trackId);
      if (!h) { h = []; this.hist.set(inp.trackId, h); }
      h.push(Float32Array.from(inp.vector));
      if (h.length > K) h.shift();
      this.seen.set(inp.trackId, this.step);
      const off = K - h.length;
      h.forEach((row, k) => { windows.set(row, (i * K + off + k) * D); mask[i * K + off + k] = 1; });
    });
    if (this.step % 20 === 0) {
      for (const [tid, s] of [...this.seen]) if (this.step - s > 20) { this.hist.delete(tid); this.seen.delete(tid); }
    }
    if (!windows.every(Number.isFinite)) throw new Error("non-finite feature value");
    // a model run that never returns must not freeze the camera loop: time out and let the caller fall back
    const out = await Promise.race([this.run(windows, mask, n, K),
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error("scheduler model timed out")), RUN_TIMEOUT_MS))]);
    if (out.action.length !== n * 4 || out.slots.length !== n * SLOTS.length) throw new Error("scheduler model returned unexpected shapes");
    // NaN logits would never be chosen and the app would go quiet without saying why
    if (!out.action.every(Number.isFinite) || !out.slots.every(Number.isFinite) || !Number.isFinite(out.silenceBias)) throw new Error("scheduler model returned non-finite values");
    this.silenceBias = out.silenceBias;
    const legal = inputs.map((inp) => legalActions(inp, t, ch, this.cfg, this.useHaptic));
    const d: Decision = { ...rlChoose(out.action, legal, out.silenceBias), logits: out.action, slotLogits: out.slots, silenceBias: out.silenceBias };
    this.lastDecision = d;
    return d;
  }

  /** Same contract as PCACS.decide (tracks, time, channel -> one cue or null), asynchronous because the model runs in ONNX Runtime. */
  async decide(tracks: Track[], t: number, ch: ChannelState): Promise<Cue | null> {
    const c = this.cfg;
    this.refill(t);
    this.lastAction = "silence";
    const obs = this.candidates(tracks, t, ch).filter((f) => f.track.confidence >= c.minConfidence);
    const d = await this.decideVectors(obs.map((f) => ({ trackId: f.track.id, vector: featureVector(f), lastAnnounced: f.track.lastAnnounced, lastHaptic: f.track.lastHaptic })), t, ch);
    if (!d) { this.lastCandidates = []; return null; }
    obs.forEach((f, i) => { f.score = d.scores[i]; f.best = d.best[i]; this.applySlots(f, slotsFromLogits(d.slotLogits, i)); });
    this.lastCandidates = [...obs].sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
    if (d.candidate < 0) return null;
    this.lastAction = d.action;
    return this.makeCue(obs[d.candidate], t, d.action, d.scores[d.candidate]);
  }

  /** SchedulerPolicy._apply_slots: re-compose the message with the model's slots. */
  applySlots(f: Candidate, slots: string[]) {
    const tr = f.track;
    f.slots = slots;
    f.message = compose(slots, { label: tr.label, kind: tr.kind, text: tr.text, distance: f.distance, depthConf: f.depthConf, angle: f.angle,
      relSpeed: -f.ownSpeed, lateralSpeed: f.moving ? f.lateralSpeed : 0, lateral: tr.lateral, hedged: f.hedged, units: this.cfg.units });
    f.duration = speechSeconds(f.message);
  }
}

type Ort = typeof OrtNs;

/** RunModel over an ONNX Runtime session (browser or node). */
export function ortRunner(ort: Ort, session: OrtNs.InferenceSession): RunModel {
  let busy: Promise<unknown> = Promise.resolve();
  return (windows, mask, n, K) => {
    const D = windows.length / (n * K);
    // one run at a time: ORT sessions do not take concurrent run() calls
    const p = busy.then(() => session.run({ windows: new ort.Tensor("float32", windows, [n, K, D]), mask: new ort.Tensor("float32", mask, [n, K]) }))
      .then((r) => ({ action: r.action_logits.data as Float32Array, slots: r.slot_logits.data as Float32Array, silenceBias: (r.silence_bias.data as Float32Array)[0] }));
    busy = p.catch(() => {});
    return p;
  };
}

/** Browser: load ONNX Runtime from /ort/ (the same raw bundle the detector worker uses, single-threaded wasm on the main
 *  thread; the model is ~100 KB), create the session and check it against the metadata with one warm-up run. */
export async function loadLearnedRunner(meta: SchedulerMeta, timeoutMs = 20000): Promise<RunModel> {
  const why = checkMeta(meta);
  if (why) throw new Error(why);
  const work = (async () => {
    const rawImport = new Function("u", "return import(u)") as (u: string) => Promise<Ort>;
    const ort = await rawImport(new URL("/ort/ort.webgpu.bundle.min.mjs", location.origin).href);
    ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(meta.model_url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    const run = ortRunner(ort, session);
    const K = meta.K, D = FEATURE_NAMES.length;
    const mask = new Float32Array(K); mask[K - 1] = 1;
    const out = await run(new Float32Array(K * D), mask, 1, K);
    if (out.action.length !== 4 || out.slots.length !== SLOTS.length) throw new Error("model outputs have the wrong shape");
    if (!out.action.every(Number.isFinite) || !Number.isFinite(out.silenceBias)) throw new Error("model outputs are not finite");
    if (Math.abs(out.silenceBias - meta.silence_bias) > 1e-5) throw new Error("model and scheduler.json disagree (stale cache?)");
    return run;
  })();
  let timer = 0;
  const timeout = new Promise<never>((_, rej) => { timer = window.setTimeout(() => rej(new Error(`model did not load within ${timeoutMs / 1000} s`)), timeoutMs); });
  try { return await Promise.race([work, timeout]); } finally { clearTimeout(timer); }
}
