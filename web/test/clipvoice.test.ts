// The recorded voice (web/src/clipvoice.ts) and its place in Speech: a sentence the pack holds is played from clips on
// the audio clock with an exactly known length; anything else, or a pack that has not loaded, goes to the browser voice
// at once; cutting a sentence settles the speech budget for what was heard.
import { test } from "node:test";
import assert from "node:assert/strict";
import { ClipVoice } from "../src/clipvoice";
import { Speech } from "../src/speech";
import { COMMA_GAP_S } from "../src/voicepack";
import type { Cue } from "../src/types";

let clock = 0;
Object.defineProperty(globalThis, "performance", { value: { now: () => clock }, configurable: true });

interface FakeNode { buffer: unknown; onended: (() => void) | null; started: [number, number, number] | null; stopped: boolean; connect(): void; start(w: number, o: number, d: number): void; stop(): void }
class FakeAudioContext {
  currentTime = 10; state = "running"; destination = {}; nodes: FakeNode[] = [];
  createBufferSource(): FakeNode {
    const n: FakeNode = { buffer: null, onended: null, started: null, stopped: false, connect() {}, start(w, o, d) { n.started = [w, o, d]; }, stop() { n.stopped = true; } };
    this.nodes.push(n); return n;
  }
  resume() { return Promise.resolve(); }
}
const actx = new FakeAudioContext();
(globalThis as any).AudioContext = function () { return actx; };

class FakeSynth { queue: Array<{ text: string }> = []; speaking = false; getVoices() { return []; } speak(u: { text: string }) { this.queue.push(u); } cancel() { this.queue = []; } }
class FakeUtterance { rate = 1; lang = ""; voice: unknown = null; onstart: unknown = null; onend: unknown = null; onerror: unknown = null; constructor(public text: string) {} }

const INDEX = { voice: "test", sprite: "sprite.mp3", clips: { "m:car": [1.0, 0.5], "f:ahead": [2.0, 0.4], "f:car": [3.0, 0.6], "s:Sound on.": [4.0, 0.8] } };
const fetcher = (async (url: string) => ({ json: async () => INDEX, arrayBuffer: async () => new ArrayBuffer(8), url })) as unknown as typeof fetch;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const near = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
const cue = (message: string, duration: number, action: "speak" | "interrupt" = "speak"): Cue =>
  ({ t: 0, trackId: 1, label: "car", message, action, duration, utility: 1, severity: 1, distance: null, ttc: null, slots: [] } as Cue);

async function rig(load = true) {
  clock = 0; actx.nodes = []; actx.currentTime = 10;
  const synth = new FakeSynth();
  const g = globalThis as any;
  g.window = g; g.speechSynthesis = synth; g.SpeechSynthesisUtterance = FakeUtterance;
  const policy = { budget: 6.0, cfg: { budgetCapacity: 6.0 } };
  const speech = new Speech(() => "", undefined, () => policy);
  speech.backend = "natural";
  speech.clips = new ClipVoice("/voice/test", fetcher, async () => ({ duration: 60 }) as AudioBuffer);
  if (load) assert.equal(await speech.clips.load(), true);
  return { synth, policy, speech };
}

test("a pack that cannot be fetched leaves the voice unloaded and can be retried", async () => {
  let calls = 0;
  const v = new ClipVoice("/voice/x", (async () => { calls++; throw new Error("offline"); }) as unknown as typeof fetch);
  assert.equal(await v.load(), false); assert.equal(v.ready, false); assert.equal(v.plan("Car."), null);
  assert.equal(await v.load(), false); assert.equal(calls, 2);
});

test("a recorded sentence plays from clips, back to back on the audio clock, with an exactly known length", async () => {
  const { synth, policy, speech } = await rig();
  policy.budget -= 1.2;
  speech.speak(cue("Car, ahead.", 1.2), 100);
  assert.equal(synth.queue.length, 0);                                  // no synthetic voice
  const [a, b] = actx.nodes;
  near(a.started![0], 10.03); near(a.started![1], 0.97); near(a.started![2], 0.56);          // "Car," with its margins
  near(b.started![0], 10.03 + 0.03 + 0.5 + COMMA_GAP_S); near(b.started![1], 1.97);          // "ahead." after the comma pause
  const exact = 0.5 + 0.4 + COMMA_GAP_S + 2 * 0.03 + 0.03;
  near(speech.busyUntil, 100 + exact); near(speech.remaining(100), exact);
  clock = 30; await sleep(45); clock = 30 + exact * 1000; b.onended!();  // started 30 ms after scheduling, then played out
  assert.equal(speech.ledger.records[0].outcome, "completed"); assert.equal(speech.ledger.records[0].backend, "natural");
  near(policy.budget, 6.0 - exact);                                     // the 1.2 s estimate was settled to the real length
});

test("a fixed app sentence is one clip", async () => {
  const { speech } = await rig();
  speech.sayNow("Sound on.", 0);
  assert.equal(actx.nodes.length, 1); near(actx.nodes[0].started![1], 3.97);
});

test("a sentence the pack does not hold goes to the device voice whole", async () => {
  const { synth, speech } = await rig();
  speech.speak(cue("Sign reads Exit, ahead.", 1.4), 0);
  assert.equal(actx.nodes.length, 0);
  assert.deepEqual(synth.queue.map((u) => u.text), ["Sign reads Exit, ahead."]);
  assert.equal(speech.ledger.records[0].backend, "browser");
});

test("a cue never waits for the pack: before it has loaded, the device voice speaks at once", async () => {
  const { synth, speech } = await rig(false);
  speech.speak(cue("Car, ahead.", 1.2, "interrupt"), 0, true);
  assert.equal(actx.nodes.length, 0); assert.equal(synth.queue.length, 1);
});

test("cutting a recorded sentence stops its clips and charges only what was heard", async () => {
  const { policy, speech } = await rig();
  policy.budget -= 1.2;
  speech.speak(cue("Car, ahead.", 1.2), 0);
  clock = 30; await sleep(45); clock = 30 + 400;
  speech.stop();
  assert.ok(actx.nodes.every((n) => n.stopped)); assert.equal(speech.clips!.playing, false);
  assert.equal(speech.ledger.records[0].outcome, "interrupted");
  near(policy.budget, 6.0 - 0.4);
});

test("the device voice setting never uses the pack", async () => {
  const { synth, speech } = await rig();
  speech.backend = "browser";
  speech.speak(cue("Car, ahead.", 1.2), 0);
  assert.equal(actx.nodes.length, 0); assert.equal(synth.queue.length, 1);
});
