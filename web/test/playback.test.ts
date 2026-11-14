// Measured speech accounting (web/src/playback.ts): the scheduler charges its speech budget with an estimated cue length
// when it decides; these tests check that the charge is settled once against what the voice actually played, for every way
// a cue can end (completed, cut off, cancelled before it started, failed, never reported back), and that a log of playback
// events leaves the budget where the measured seconds say it should be.
import { test } from "node:test";
import assert from "node:assert/strict";
import { PlaybackLedger, type BudgetHolder } from "../src/playback";

function rig(budget = 3.0, capacity = 6.0) {
  const holder: BudgetHolder = { budget, cfg: { budgetCapacity: capacity } };
  let ms = 0;
  const ledger = new PlaybackLedger(() => holder, () => ms);
  return { holder, ledger, at: (t: number) => { ms = t; } };
}
const near = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

test("a cue that plays as long as estimated changes nothing", () => {
  const { holder, ledger, at } = rig();
  const id = ledger.issue("Car, ahead.", 2.0, "browser");
  at(100); ledger.started(id);
  at(2100); near(ledger.ended(id, "completed"), 0);
  near(holder.budget, 3.0);
  assert.equal(ledger.records[0].outcome, "completed"); near(ledger.records[0].playedS, 2.0);
});

test("a shorter cue refunds the unplayed seconds, never above the capacity", () => {
  const { holder, ledger, at } = rig(3.0);
  const id = ledger.issue("Car.", 2.0, "browser");
  at(0); ledger.started(id); at(1200);
  near(ledger.ended(id, "completed"), 0.8); near(holder.budget, 3.8);
  holder.budget = 5.9;
  const id2 = ledger.issue("Car.", 2.0, "browser");
  at(2000); ledger.started(id2); at(2500);
  near(ledger.ended(id2, "completed"), 0.1); near(holder.budget, 6.0);     // 1.5 s unplayed, only 0.1 s fits
});

test("a cue that runs long becomes debt", () => {
  const { holder, ledger, at } = rig(0.5);
  const id = ledger.issue("Stairs going down, two steps ahead.", 1.5, "browser");
  at(0); ledger.started(id); at(2500);
  near(ledger.ended(id, "completed"), -1.0); near(holder.budget, -0.5);
});

test("a cue cut off part way is charged only for what was heard", () => {
  const { holder, ledger, at } = rig(3.0);
  const id = ledger.issue("Person, slightly left.", 2.0, "browser");
  at(0); ledger.started(id); at(600);
  near(ledger.ended(id, "interrupted"), 1.4); near(holder.budget, 4.4);
  assert.equal(ledger.records[0].outcome, "interrupted");
});

test("a cue cancelled or failed before any sound is refunded in full", () => {
  const { holder, ledger } = rig(3.0);
  near(ledger.ended(ledger.issue("Door.", 1.0, "elevenlabs"), "cancelled"), 1.0);
  near(ledger.ended(ledger.issue("Door.", 1.0, "browser"), "failed"), 1.0);
  near(holder.budget, 5.0);
  assert.deepEqual(ledger.records.map((r) => r.playedS), [0, 0]);
});

test("each cue is settled once: a late end event after a cut changes nothing", () => {
  const { holder, ledger, at } = rig(3.0);
  const id = ledger.issue("Car.", 2.0, "browser");
  at(0); ledger.started(id); at(500); ledger.ended(id, "interrupted");
  at(900); near(ledger.ended(id, "completed"), 0);
  near(holder.budget, 4.5); assert.equal(ledger.records[0].outcome, "interrupted");
});

test("speech that was never charged (an answer) is measured but does not touch the budget", () => {
  const { holder, ledger, at } = rig(3.0);
  const id = ledger.issue("The exit is behind you.", 2.0, "browser", false);
  at(0); ledger.started(id); at(900); near(ledger.ended(id, "interrupted"), 0);
  near(holder.budget, 3.0); near(ledger.records[0].playedS, 0.9);
});

test("the voice may report its own played seconds (audio element position)", () => {
  const { holder, ledger, at } = rig(3.0);
  const id = ledger.issue("Car.", 2.0, "elevenlabs");
  at(0); ledger.started(id); at(5000);                      // wall clock stalled in a background tab
  near(ledger.ended(id, "completed", 1.6), 0.4); near(holder.budget, 3.4);
});

test("an end without a start is not measured: a completed cue keeps its estimate, a cut one counts as silent", () => {
  const { holder, ledger } = rig(3.0);
  near(ledger.ended(ledger.issue("Car.", 2.0, "browser"), "completed"), 0);
  assert.equal(ledger.records[0].measured, false); near(holder.budget, 3.0);
  near(ledger.ended(ledger.issue("Car.", 2.0, "browser"), "interrupted"), 2.0);
});

test("stop settles everything open: started cues as interrupted, unstarted ones as cancelled", () => {
  const { holder, ledger, at } = rig(1.0);
  const a = ledger.issue("Car.", 2.0, "browser"); ledger.issue("Pole.", 1.0, "elevenlabs");
  at(0); ledger.started(a); at(500);
  near(ledger.abandonAll(), 2.5); near(holder.budget, 3.5);
  assert.deepEqual(ledger.records.map((r) => r.outcome), ["interrupted", "cancelled"]);
  assert.equal(ledger.active(), false);
});

test("the channel is busy while a cue is open, and a cue whose end never arrives is closed by the watchdog", () => {
  const { holder, ledger, at } = rig(3.0);
  const id = ledger.issue("Car.", 2.0, "browser");
  at(0); ledger.started(id);
  at(2500); assert.equal(ledger.active(), true);            // running past its estimate: still busy
  at(60_000); assert.equal(ledger.active(), false);         // no end event ever came
  assert.equal(ledger.records[0].outcome, "lost"); near(holder.budget, 3.0);   // assumed to have played as estimated
  const never = ledger.issue("Pole.", 1.0, "browser");
  at(120_000); assert.equal(ledger.active(), false);
  assert.equal(ledger.records[1].outcome, "lost"); near(holder.budget, 4.0);   // never started: refunded
  near(ledger.ended(never, "completed"), 0);
});

test("a log of playback events leaves the budget where the measured seconds put it", () => {
  // [ms, event, cue, estimate or outcome]: the kind of trace Speech produces in a browser session
  const log: Array<[number, "issue" | "start" | "end" | "stop", string, number | string]> = [
    [0, "issue", "a", 1.8], [120, "start", "a", 0], [1700, "end", "a", "completed"],          // played 1.58
    [4000, "issue", "b", 2.2], [4100, "start", "b", 0], [4900, "issue", "c", 1.0], [4900, "stop", "", 0],   // b cut at 0.8, c cancelled
    [5000, "issue", "d", 1.0], [5050, "start", "d", 0], [6650, "end", "d", "completed"],          // played 1.6 (overrun 0.6)
    [9000, "issue", "e", 1.4], [9000, "end", "e", "failed"],
  ];
  const { holder, ledger, at } = rig(6.0);
  const ids = new Map<string, number>();
  let charged = 0;
  for (const [ms, ev, cue, x] of log) {
    at(ms);
    if (ev === "issue") { holder.budget -= x as number; charged += x as number; ids.set(cue, ledger.issue(cue, x as number, "browser")); }
    else if (ev === "start") ledger.started(ids.get(cue)!);
    else if (ev === "end") ledger.ended(ids.get(cue)!, x as "completed" | "failed");
    else ledger.abandonAll();
  }
  const s = ledger.summary();
  near(s.playedS, 1.58 + 0.8 + 0 + 1.6 + 0); near(s.reservedS, charged);
  near(holder.budget, 6.0 - s.playedS);                     // estimates are gone: only measured seconds were spent
  assert.deepEqual(s.outcomes, { completed: 2, interrupted: 1, cancelled: 1, failed: 1, lost: 0 });
  near(s.cutShare, 3 / 5); near(s.overrunS, 0.6);
});

test("a reset forgets the session", () => {
  const { ledger } = rig();
  ledger.ended(ledger.issue("Car.", 1.0, "browser"), "failed");
  ledger.clear();
  assert.equal(ledger.records.length, 0); assert.equal(ledger.summary().cues, 0);
});
