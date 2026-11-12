// The cane consumer: the server's "cane" summary is validated, turned into one obstacle cue while navigating (never
// still, never at a desk, never when a detected object explains it, never "clear"), and goes through the gate like any
// object: said once, held while nothing changes, said again when it is materially closer; very close straight ahead
// it is a haptic stop.
import { test } from "node:test";
import assert from "node:assert/strict";
import { CaneConsumer, caneMessage, parseCane } from "../src/cane";
import { AnnouncementGate } from "../src/gate";
import { Camera } from "../src/tracker";
import type { CaneSummary, ChannelState, Track } from "../src/types";

const FREE: ChannelState = { busy: false, remaining: 0, currentUtility: 0, currentSeverity: 0, currentTrack: null };
const summary = (d: number, extra: Partial<{ sector: string; level: string; confidence: number; suggest: string | null; bbox: number[] }> = {}): unknown => ({
  clear_m: { left: 3.5, ahead: d, right: null },
  obstacle: { sector: extra.sector ?? "ahead", distance_m: d, level: extra.level ?? "body", confidence: extra.confidence ?? 0.8, bbox: extra.bbox ?? [0.4, 0.3, 0.6, 0.9] },
  suggest: extra.suggest === undefined ? "left" : extra.suggest,
});
const ctx = (t: number, over: Partial<Parameters<CaneConsumer["propose"]>[0]> = {}) =>
  ({ t, tracks: [] as Track[], navigating: true, still: false, desk: false, channelBusy: false, units: "m" as const, ...over });

test("parseCane validates the server's summary", () => {
  const ok = parseCane(summary(1.0))!;
  assert.equal(ok.obstacle?.sector, "ahead"); assert.equal(ok.obstacle?.distance_m, 1.0); assert.equal(ok.suggest, "left");
  assert.equal(ok.clear_m.right, null);
  assert.equal(parseCane(null), null);
  assert.equal(parseCane(undefined), null);
  assert.equal(parseCane({ clear_m: {}, obstacle: { sector: "up", distance_m: 1, level: "body", confidence: 1, bbox: [0, 0, 1, 1] }, suggest: null })!.obstacle, null);
  assert.equal(parseCane({ clear_m: {}, obstacle: { sector: "ahead", distance_m: "1", level: "body", confidence: 1, bbox: [0, 0, 1, 1] }, suggest: "maybe" })!.suggest, null);
});

test("messages name the obstacle, its height, distance and the clearer side, and never say 'clear'", () => {
  const o = parseCane(summary(1.0))!.obstacle!;
  assert.equal(caneMessage(o, "left"), "Obstacle ahead, about 1 meter, clearer on your left.");
  assert.equal(caneMessage({ ...o, level: "head", distance_m: 2.1 }, null), "Something at head height ahead, about 2 meters.");
  assert.equal(caneMessage({ ...o, level: "ground", sector: "right", distance_m: 0.8 }, null), "Low obstacle on your right, right in front of you.");
  assert.equal(caneMessage(o, "stop"), "Stop. Obstacle ahead, about 1 meter.");
  for (const s of [null, "left", "right", "stop"] as const) for (const level of ["ground", "body", "head"] as const) {
    assert.doesNotMatch(caneMessage({ ...o, level }, s), /\bclear\b/i);
  }
});

/** The app's loop: server answers at ~4 Hz, a decision every 0.5 s through the gate. */
function run(gate: AnnouncementGate, cane: CaneConsumer, t0: number, t1: number, dist: (t: number) => number | null, over: (t: number) => object = () => ({})) {
  const said: Array<[number, string, string]> = [];
  for (let k = 0; t0 + k * 0.25 <= t1 + 1e-9; k++) {
    const t = +(t0 + k * 0.25).toFixed(3);
    const d = dist(t);
    cane.update(d == null ? null : (parseCane(summary(d)) as CaneSummary), t);
    gate.observe(cane.pseudoTracks(t), t);
    if (Math.abs(t * 2 - Math.round(t * 2)) < 1e-6) {
      const cue = cane.propose(ctx(t, over(t)));
      const out = gate.filter(cue ? [cue] : [], cane.pseudoTracks(t), t, FREE);
      if (out) said.push([t, out.action, out.message]);
    }
  }
  return said;
}

test("an obstacle is said once, held while nothing changes, and said again when materially closer", () => {
  const cam = new Camera(), gate = new AnnouncementGate(cam), cane = new CaneConsumer(cam);
  const said = run(gate, cane, 0, 12, (t) => (t < 6 ? 3.0 : Math.max(1.6, 3.0 - 0.5 * (t - 6))));
  assert.ok(said.length >= 2, JSON.stringify(said));
  assert.equal(said[0][2], "Obstacle ahead, about 3 meters, clearer on your left.");
  assert.ok(said[0][0] <= 1.0, `first said at ${said[0][0]}`);
  assert.equal(said.filter(([t]) => t < 6).length, 1, "held while static");
  assert.ok(said[1][0] >= 6 && /about 2 meters|about 1\.5 meters/.test(said[1][2]), JSON.stringify(said[1]));
});

test("never while still, at a desk, outside navigation, or when a detected object explains it", () => {
  const cam = new Camera();
  for (const over of [{ still: true }, { desk: true }, { navigating: false }]) {
    const gate = new AnnouncementGate(cam), cane = new CaneConsumer(cam);
    assert.deepEqual(run(gate, cane, 0, 4, () => 2.0, () => over), [], JSON.stringify(over));
  }
  const person: Track = { id: 5, label: "person", kind: "object", bbox: [0.35, 0.2, 0.65, 0.95], confidence: 0.9, firstSeen: 0, lastSeen: 4, hits: 40, misses: 0,
    confirmed: true, lastAnnounced: null, lastAnnouncedBBox: null, announceCount: 0, lastHaptic: null, bboxHistory: [], distance: 2.1, depthConf: 0.9,
    lateral: 0, relSpeed: 0, lateralSpeed: 0, ttc: Infinity, pathProb: 0.5, distHistory: [] };
  const gate = new AnnouncementGate(cam), cane = new CaneConsumer(cam);
  assert.deepEqual(run(gate, cane, 0, 4, () => 2.0, () => ({ tracks: [person] })), []);
  assert.match(cane.note, /explained by person/);
});

test("no summary or a stale one proposes nothing (silence, never 'clear')", () => {
  const cam = new Camera(), cane = new CaneConsumer(cam);
  cane.update(parseCane(summary(2.0)), 0);
  assert.equal(cane.propose(ctx(2.0)), null);   // 2 s old
  cane.update(null, 2.1);
  assert.equal(cane.propose(ctx(2.2)), null);
});

test("very close straight ahead: a haptic stop, at most once per 2 s", () => {
  const cam = new Camera(), gate = new AnnouncementGate(cam), cane = new CaneConsumer(cam);
  const said = run(gate, cane, 0, 4, () => 0.5);
  assert.ok(said.length >= 1 && said.every(([, a]) => a === "haptic"), JSON.stringify(said));
  for (let i = 1; i < said.length; i++) assert.ok(said[i][0] - said[i - 1][0] >= 2 - 1e-9);
});
