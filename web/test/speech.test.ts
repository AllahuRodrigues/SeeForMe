// Speech (web/src/speech.ts) on a fake browser voice and a fake clock: every way a spoken cue can end must settle the
// scheduler's speech budget once, against the seconds the voice really played. The scheduler has already charged the
// estimate when Speech.speak is called, so each test charges it first, the way pcacs.ts does.
import { test } from "node:test";
import assert from "node:assert/strict";
import { Speech } from "../src/speech";
import type { Cue } from "../src/types";

let clock = 0;
Object.defineProperty(globalThis, "performance", { value: { now: () => clock }, configurable: true });

class FakeUtterance {
  rate = 1; lang = ""; voice: unknown = null; volume = 1;
  onstart: (() => void) | null = null; onend: (() => void) | null = null; onerror: ((e: { error: string }) => void) | null = null;
  constructor(public text: string) {}
}
class FakeSynth {
  queue: FakeUtterance[] = []; cancels = 0;
  get speaking() { return this.queue.length > 0; }
  getVoices() { return []; }
  speak(u: FakeUtterance) { this.queue.push(u); }
  /** like a browser: whatever is queued or sounding ends with an "interrupted" error */
  cancel() { this.cancels++; const q = this.queue; this.queue = []; for (const u of q) u.onerror?.({ error: "interrupted" }); }
  start(i = 0) { this.queue[i].onstart?.(); }
  finish(i = 0) { const [u] = this.queue.splice(i, 1); u.onend?.(); }
  fail(i = 0, error = "synthesis-failed") { const [u] = this.queue.splice(i, 1); u.onerror?.({ error }); }
}

function rig(budget = 6.0) {
  clock = 0;
  const synth = new FakeSynth();
  const g = globalThis as any;
  g.window = g; g.speechSynthesis = synth; g.SpeechSynthesisUtterance = FakeUtterance;
  const policy = { budget, cfg: { budgetCapacity: 6.0 } };
  const speech = new Speech(() => "", undefined, () => policy);
  /** the scheduler charges the estimate, then hands the cue to the voice */
  const say = (message: string, duration: number, t: number, action: "speak" | "interrupt" = "speak") => {
    policy.budget -= duration;
    speech.speak(cue(message, duration, t, action), t, action === "interrupt");
  };
  return { synth, policy, speech, say };
}
const cue = (message: string, duration: number, t: number, action: "speak" | "interrupt" = "speak"): Cue =>
  ({ t, trackId: 1, label: "car", message, action, duration, utility: 1, severity: 1, distance: null, ttc: null, slots: [] } as Cue);
const near = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("a cue that the voice finishes early gives the unplayed seconds back", () => {
  const { synth, policy, speech, say } = rig();
  say("Car, ahead.", 2.0, 0);
  clock = 100; synth.start(); clock = 1300; synth.finish();
  near(policy.budget, 6.0 - 1.2);
  assert.equal(speech.ledger.records[0].outcome, "completed");
});

test("an interrupt cue cuts the one playing: the cut cue is charged only for what was heard", () => {
  const { synth, policy, speech, say } = rig();
  say("Bench, slightly left.", 2.0, 0);
  clock = 0; synth.start();
  clock = 500; say("Car, close ahead.", 1.0, 0.5, "interrupt");
  assert.equal(synth.cancels, 1);
  assert.deepEqual(speech.ledger.records.map((r) => r.outcome), ["interrupted", null]);
  near(policy.budget, 6.0 - 0.5 - 1.0);                       // 0.5 s heard of the first, the second still estimated
  synth.start(); clock = 1400; synth.finish();
  near(policy.budget, 6.0 - 0.5 - 0.9);
});

test("a voice slower than the estimate keeps the channel busy until it really ends, and the overrun is debt", () => {
  const { synth, policy, speech, say } = rig(1.0);
  say("Stairs going down ahead.", 1.0, 0);
  clock = 0; synth.start();
  clock = 1500; assert.equal(speech.free(1.5), false);        // past the estimate, still sounding
  clock = 1800; synth.finish();
  assert.equal(speech.free(1.8), true); near(policy.budget, -0.8);
});

test("a cue the browser fails to speak is refunded in full", () => {
  const { synth, policy, speech, say } = rig();
  say("Door, right.", 1.2, 0);
  synth.fail();
  near(policy.budget, 6.0); assert.equal(speech.ledger.records[0].outcome, "failed");
  delete (globalThis as any).speechSynthesis;                 // a browser with no voice at all
  say("Door, right.", 1.2, 3);
  near(policy.budget, 6.0); assert.equal(speech.ledger.records[1].outcome, "failed");
});

test("stop (pause, mute, goodbye) settles what was sounding and leaves nothing open", () => {
  const { synth, policy, speech, say } = rig();
  say("Person, ahead.", 2.0, 0);
  clock = 0; synth.start(); clock = 700;
  speech.stop();
  near(policy.budget, 6.0 - 0.7); assert.equal(speech.ledger.active(), false);
  assert.equal(speech.ledger.records[0].outcome, "interrupted");
});

test("an answer is measured but never spends or refunds the scheduler's budget", () => {
  const { synth, policy, speech } = rig(2.0);
  speech.sayNow("The exit is behind you.", 0);
  clock = 0; synth.start(); clock = 400; speech.stop();
  near(policy.budget, 2.0);
  assert.equal(speech.ledger.records[0].charged, false); near(speech.ledger.records[0].playedS, 0.4);
});

test("a muted app plays nothing and records nothing: the screen reader delivers the cue", () => {
  const { synth, policy, speech, say } = rig();
  speech.enabled = false;
  say("Car, ahead.", 2.0, 0);
  assert.equal(synth.queue.length, 0); assert.equal(speech.ledger.records.length, 0); near(policy.budget, 4.0);
});

test("a session reset stops the voice and forgets the playback log", () => {
  const { synth, speech, say } = rig();
  say("Car, ahead.", 2.0, 0); synth.start();
  speech.reset();
  assert.equal(speech.ledger.records.length, 0); assert.equal(speech.free(0), true);
});
