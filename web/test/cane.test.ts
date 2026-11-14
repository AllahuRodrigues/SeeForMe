// The cane consumer: the server's "cane" summary is validated, turned into one obstacle cue while navigating (never
// still, never at a desk, never when a detected object explains it, never "clear" or "clearer"), and goes through the
// gate like any object: said once, held while nothing changes, said again when it is materially closer; very close
// straight ahead it is a haptic stop. What it says depends on the travel aid (cane, guide dog, neither). Its own repeat
// rules (seeforme/live/cane.py): no repeat within 2.5 s unless the obstacle entered the 1 m band, ground and body are one
// obstacle, one stop per obstacle and nothing spoken for 3 s after it; the speech channel is modelled as the app keeps it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AIDS, CANE_CFG, CaneConsumer, aidAdvice, blockedSide, caneMessage, parseCane, steadyApproach, type TravelAid } from "../src/cane";
import { AnnouncementGate, GATE_DEFAULTS } from "../src/gate";
import { Camera } from "../src/tracker";
import type { CaneSummary, ChannelState, Cue, Track } from "../src/types";

const FREE: ChannelState = { busy: false, remaining: 0, currentUtility: 0, currentSeverity: 0, currentTrack: null };
type Extra = Partial<{ sector: string; level: string; confidence: number; suggest: string | null; bbox: number[]; left: number | null; right: number | null }>;
const summary = (d: number, extra: Extra = {}): unknown => ({
  clear_m: { left: extra.left === undefined ? 3.5 : extra.left, ahead: d, right: extra.right === undefined ? null : extra.right },
  obstacle: { sector: extra.sector ?? "ahead", distance_m: d, level: extra.level ?? "body", confidence: extra.confidence ?? 0.8, bbox: extra.bbox ?? [0.4, 0.3, 0.6, 0.9] },
  suggest: extra.suggest === undefined ? "left" : extra.suggest,
});
const ctx = (t: number, over: Partial<Parameters<CaneConsumer["propose"]>[0]> = {}) =>
  ({ t, tracks: [] as Track[], navigating: true, still: false, desk: false, channelBusy: false, units: "m" as const, aid: "none" as TravelAid, ...over });

test("parseCane validates the server's summary", () => {
  const ok = parseCane(summary(1.0))!;
  assert.equal(ok.obstacle?.sector, "ahead"); assert.equal(ok.obstacle?.distance_m, 1.0); assert.equal(ok.suggest, "left");
  assert.equal(ok.clear_m.right, null);
  assert.equal(parseCane(null), null);
  assert.equal(parseCane(undefined), null);
  assert.equal(parseCane({ clear_m: {}, obstacle: { sector: "up", distance_m: 1, level: "body", confidence: 1, bbox: [0, 0, 1, 1] }, suggest: null })!.obstacle, null);
  assert.equal(parseCane({ clear_m: {}, obstacle: { sector: "ahead", distance_m: "1", level: "body", confidence: 1, bbox: [0, 0, 1, 1] }, suggest: "maybe" })!.suggest, null);
});

test("messages name the obstacle, its height and distance, and what blocks; never a clear or clearer side", () => {
  const o = parseCane(summary(1.0))!.obstacle!;
  assert.equal(caneMessage(o, "right"), "Obstacle ahead, about 1 meter, and on your right.");
  assert.equal(caneMessage(o, null), "Obstacle ahead, about 1 meter.");
  assert.equal(caneMessage({ ...o, level: "head", distance_m: 2.1 }, null), "Something at head height ahead, about 2 meters.");
  assert.equal(caneMessage({ ...o, level: "head", distance_m: 1.2 }, "left"), "Something at head height ahead, about 1 meter.");
  assert.equal(caneMessage({ ...o, level: "ground", sector: "right", distance_m: 0.8 }, null), "Low obstacle on your right, right in front of you.");
  assert.equal(caneMessage(o, "stop"), "Stop. Obstacle ahead, about 1 meter.");
  for (const s of [null, "left", "right", "stop"] as const) for (const level of ["ground", "body", "head"] as const) for (const d of [0.5, 1.0, 2.0, 3.5]) {
    assert.doesNotMatch(caneMessage({ ...o, level, distance_m: d }, s), /\bclear|\bopen\b|\bfree\b|\bsafe\b|go (?:left|right)/i);
  }
});

test("path choice reports the side that blocks, never the side the server saw open", () => {
  const o = parseCane(summary(1.0))!.obstacle!;
  // the server prefers the left (seen open); the right was seen blocked at 1.3 m: "and on your right"
  assert.equal(blockedSide(o, { suggest: "left", clear_m: { left: 3.5, ahead: 1.0, right: 1.3 } }), "right");
  assert.equal(blockedSide(o, { suggest: "right", clear_m: { left: 0.9, ahead: 1.0, right: 4.0 } }), "left");
  // the other side was not seen, or is open as well: nothing to report (silence about a side is not "clear")
  assert.equal(blockedSide(o, { suggest: "left", clear_m: { left: 3.5, ahead: 1.0, right: null } }), null);
  assert.equal(blockedSide(o, { suggest: "left", clear_m: { left: 3.5, ahead: 1.0, right: 3.0 } }), null);
  // not blocked near enough, head level, a side obstacle, no suggestion
  assert.equal(blockedSide({ ...o, distance_m: 2.0 }, { suggest: "left", clear_m: { left: 4, ahead: 2, right: 1 } }), null);
  assert.equal(blockedSide({ ...o, level: "head" }, { suggest: "left", clear_m: { left: 4, ahead: 1, right: 1 } }), null);
  assert.equal(blockedSide({ ...o, sector: "left" }, { suggest: "left", clear_m: { left: 4, ahead: 1, right: 1 } }), null);
  assert.equal(blockedSide(o, { suggest: null, clear_m: { left: 4, ahead: 1, right: 1 } }), null);
});

/** The app's loop: server answers at ~4 Hz, a decision every 0.5 s through the gate. `sum(t)`: the server's summary (null:
 *  no depth map). Returns [t, action, message] of what passed. */
function run(gate: AnnouncementGate, cane: CaneConsumer, t0: number, t1: number, sum: (t: number) => unknown | null, over: (t: number) => object = () => ({})) {
  const said: Array<[number, string, string]> = [];
  for (let k = 0; t0 + k * 0.25 <= t1 + 1e-9; k++) {
    const t = +(t0 + k * 0.25).toFixed(3);
    const raw = sum(t);
    cane.update(raw == null ? null : (parseCane(raw) as CaneSummary), t);
    gate.observe(cane.pseudoTracks(t), t);
    if (Math.abs(t * 2 - Math.round(t * 2)) < 1e-6) {
      const cue = cane.propose(ctx(t, over(t)));
      const out = gate.filter(cue ? [cue] : [], cane.pseudoTracks(t), t, FREE);
      if (out) said.push([t, out.action, out.message]);
    }
  }
  return said;
}
const fresh = () => { const cam = new Camera(); return { gate: new AnnouncementGate(cam), cane: new CaneConsumer(cam) }; };
/** a walker closing on an obstacle at 1 m/s from d0 (stops at 0.3 m) */
const walkUp = (d0: number, level = "body", extra: Extra = {}) => (t: number) => summary(Math.max(0.3, d0 - t), { level, ...extra });

test("no aid: an obstacle is said once, held while nothing changes, and said again when materially closer", () => {
  const { gate, cane } = fresh();
  const said = run(gate, cane, 0, 12, (t) => summary(t < 6 ? 3.0 : Math.max(1.6, 3.0 - 0.5 * (t - 6))));
  assert.ok(said.length >= 2, JSON.stringify(said));
  assert.equal(said[0][2], "Obstacle ahead, about 3 meters.");
  assert.ok(said[0][0] <= 1.0, `first said at ${said[0][0]}`);
  assert.equal(said.filter(([t]) => t < 6).length, 1, "held while static");
  assert.ok(said[1][0] >= 6 && /about 2 meters|about 1\.5 meters/.test(said[1][2]), JSON.stringify(said[1]));
});

test("never while still, at a desk, outside navigation, or when a detected object explains it", () => {
  for (const aid of ["none", "cane", "guide-dog"] as const) {
    for (const over of [{ still: true }, { desk: true }, { navigating: false }]) {
      const { gate, cane } = fresh();
      assert.deepEqual(run(gate, cane, 0, 4, walkUp(3.0, "head"), () => ({ ...over, aid })), [], `${aid} ${JSON.stringify(over)}`);
    }
  }
  const person: Track = { id: 5, label: "person", kind: "object", bbox: [0.35, 0.2, 0.65, 0.95], confidence: 0.9, firstSeen: 0, lastSeen: 4, hits: 40, misses: 0,
    confirmed: true, lastAnnounced: null, lastAnnouncedBBox: null, announceCount: 0, lastHaptic: null, bboxHistory: [], distance: 2.1, depthConf: 0.9,
    lateral: 0, relSpeed: 0, lateralSpeed: 0, ttc: Infinity, pathProb: 0.5, distHistory: [] };
  const { gate, cane } = fresh();
  assert.deepEqual(run(gate, cane, 0, 4, () => summary(2.0), () => ({ tracks: [person] })), []);
  assert.match(cane.note, /explained by person/);
});

test("no summary or a stale one proposes nothing (silence, never 'clear')", () => {
  const cam = new Camera(), cane = new CaneConsumer(cam);
  cane.update(parseCane(summary(2.0)), 0);
  assert.equal(cane.propose(ctx(2.0)), null);   // 2 s old
  cane.update(null, 2.1);
  assert.equal(cane.propose(ctx(2.2)), null);
});

test("no aid: very close straight ahead is a haptic stop, once per obstacle (again only after it was away)", () => {
  // cane.py: once per obstacle, re-armed after nothing was within 1 m ahead for 2 s (the gate alone buzzed every 2 s)
  const { gate, cane } = fresh();
  const said = run(gate, cane, 0, 6, () => summary(0.5));
  assert.ok(said.length === 1 && said[0][1] === "haptic", JSON.stringify(said));
  const b = fresh();
  const back = run(b.gate, b.cane, 0, 8, (t) => summary(t < 3 || t >= 6 ? 0.5 : 3.0));
  assert.deepEqual(back.filter(([, a]) => a === "haptic").map(([t]) => t >= 6), [false, true], JSON.stringify(back));
});

// ---- travel aid -----------------------------------------------------------------------------------------------------------
test("steady approach: a walker closing in, not a noisy range or one that stands still", () => {
  const ramp = (d0: number, v: number, noise = 0) => Array.from({ length: 7 }, (_, i): [number, number] => [i * 0.25, d0 - v * i * 0.25 + (i % 2 ? noise : -noise)]);
  assert.equal(steadyApproach(ramp(2.0, 1.0), 1.5), true);
  assert.equal(steadyApproach(ramp(2.0, 0.1), 1.5), false, "too slow");
  assert.equal(steadyApproach(ramp(2.0, 0.0), 1.5), false, "standing at it");
  assert.equal(steadyApproach(ramp(2.5, 0.6, 0.3), 1.5), false, "a range that jumps about");
  assert.equal(steadyApproach(ramp(2.0, 1.0).slice(0, 2), 0.25), false, "two maps are not enough");
});

test("cane: body level is said between 1.2 and 2.5 m, nothing nearer, and the stop comes in a steady approach", () => {
  const { gate, cane } = fresh();
  const said = run(gate, cane, 0, 4, walkUp(4.0), () => ({ aid: "cane" }));
  const words = said.filter(([, a]) => a === "speak"), stops = said.filter(([, a]) => a === "haptic");
  assert.ok(words.length >= 1, JSON.stringify(said));
  for (const [t] of words) { const d = 4.0 - t; assert.ok(d <= 2.5 + 1e-9 && d >= 1.2 - 1e-9, `said at ${d.toFixed(2)} m: ${JSON.stringify(said)}`); }
  assert.equal(stops.length, 1, JSON.stringify(said));
  assert.ok(4.0 - stops[0][0] < 0.6 && 4.0 - stops[0][0] >= 0.3, `stop at ${4.0 - stops[0][0]} m`);
  // with no aid the same walk is said from 4 m
  const b = fresh();
  const none = run(b.gate, b.cane, 0, 4, walkUp(4.0), () => ({ aid: "none" }));
  assert.ok(none[0][0] <= 0.5 + 1e-9 && none[0][1] === "speak", JSON.stringify(none));
});

test("cane: a low obstacle within reach is the cane's; farther only with a steady approach; never a stop", () => {
  // walking past a kerb-like low obstacle the depth map keeps at 1.0 m (the user walks alongside it)
  const a = fresh();
  assert.deepEqual(run(a.gate, a.cane, 0, 6, () => summary(1.0, { level: "ground" }), () => ({ aid: "cane" })), []);
  assert.match(a.cane.note, /reach/);
  // walked up to from 3 m: said once beyond reach, nothing within it, no stop for a low obstacle
  const b = fresh();
  const said = run(b.gate, b.cane, 0, 3, walkUp(3.0, "ground"), () => ({ aid: "cane" }));
  assert.ok(said.length >= 1 && said.every(([t, a]) => a === "speak" && 3.0 - t >= 1.2 - 1e-9 && 3.0 - t <= 2.5 + 1e-9), JSON.stringify(said));
  assert.match(said[0][2], /^Low obstacle ahead/);
  // a low "obstacle" at 2 m whose range jumps about (a noisy floor patch) while walking: no steady approach, silence
  const c = fresh();
  assert.deepEqual(run(c.gate, c.cane, 0, 6, (t) => summary(2.0 + (Math.round(t * 4) % 2 ? 0.35 : -0.35), { level: "ground" }), () => ({ aid: "cane" })), []);
});

test("cane and guide dog: head level is always said (within 2.5 m); the dog leaves ground and body to the dog", () => {
  for (const aid of ["cane", "guide-dog"] as const) {
    const a = fresh();
    const head = run(a.gate, a.cane, 0, 4, walkUp(4.0, "head"), () => ({ aid }));
    const first = head.find(([, act]) => act === "speak");
    assert.ok(first, `${aid}: head level never said ${JSON.stringify(head)}`);
    assert.ok(4.0 - first![0] <= 2.5 + 1e-9 && 4.0 - first![0] >= 2.0, `${aid}: head said at ${4.0 - first![0]} m`);
    assert.match(first![2], /^Something at head height ahead/);
    // a static head-height obstacle within reach (a branch the user stands under) is said too: a cane passes under it
    const s = fresh();
    assert.ok(run(s.gate, s.cane, 0, 2, () => summary(1.0, { level: "head" }), () => ({ aid })).some(([, act]) => act === "speak"), aid);
  }
  for (const level of ["ground", "body"]) {
    const d = fresh();
    const said = run(d.gate, d.cane, 0, 4, walkUp(4.0, level), () => ({ aid: "guide-dog" }));
    assert.ok(said.every(([, a]) => a === "haptic"), `guide dog, ${level}: ${JSON.stringify(said)}`);
    assert.equal(said.length, level === "body" ? 1 : 0, `guide dog, ${level}: the stop only for body level`);
  }
});

test("cane and guide dog: no stop for something the user stands at (no steady approach); no aid: the stop as before", () => {
  for (const aid of ["cane", "guide-dog"] as const) {
    const { gate, cane } = fresh();
    assert.deepEqual(run(gate, cane, 0, 4, () => summary(0.5, { level: "body" }), () => ({ aid })), [], aid);
  }
  const { gate, cane } = fresh();
  assert.ok(run(gate, cane, 0, 4, () => summary(0.5, { level: "ground" }), () => ({ aid: "none" })).some(([, a]) => a === "haptic"));
});

test("cane and guide dog: only the corridor ahead is spoken about; no aid: the sides as before", () => {
  const side = (t: number) => summary(Math.max(0.8, 2.4 - 0.5 * t), { sector: "left", level: "head", bbox: [0.05, 0.1, 0.25, 0.4], suggest: null });
  for (const aid of ["cane", "guide-dog"] as const) {
    const { gate, cane } = fresh();
    assert.deepEqual(run(gate, cane, 0, 3, side, () => ({ aid })), [], aid);
  }
  const { gate, cane } = fresh();
  const said = run(gate, cane, 0, 3, side, () => ({ aid: "none" }));
  assert.match(said[0]?.[2] ?? "", /^Something at head height on your left/);
});

test("cane: path choice beyond reach names the side that blocks", () => {
  const { gate, cane } = fresh();
  const said = run(gate, cane, 0, 3, (t) => summary(Math.max(1.25, 2.4 - 0.5 * t), { level: "body", suggest: "left", left: 3.8, right: 1.3 }), () => ({ aid: "cane" }));
  assert.ok(said.some(([, , m]) => / and on your right\.$/.test(m)), JSON.stringify(said));
  assert.ok(said.every(([, , m]) => !/left/.test(m)), "the open side is never named");
});

test("cane: a low obstacle in a steady approach keeps its level (the server gives a cane user no path choice for it)", () => {
  // seeforme/live/cane.py path_choice_applies: the server's suggest is null for a ground-level obstacle and a cane user,
  // and the live demo says the same "Low obstacle ahead" (tests/test_live_cane.py checks the Python side)
  const { gate, cane } = fresh();
  const said = run(gate, cane, 0, 3, (t) => summary(Math.max(1.25, 1.6 - 0.3 * t), { level: "ground", suggest: null, left: 1.5, right: 4.0 }), () => ({ aid: "cane" }));
  assert.ok(said.length && said.every(([, , m]) => /^Low obstacle ahead/.test(m) && !/on your/.test(m)), JSON.stringify(said));
});

// ---- repeats, with the speech channel as the app keeps it -----------------------------------------------------------------
/** As run(), but a spoken cue that passes holds the speech channel for its duration, as speech.speak does in the app, and
 *  every decision sees that channel: the cane gets channelBusy (and with `exact` the track that plays, channelTrack), the
 *  gate the ChannelState. Returns [t, action, message, distance] of what passed. */
function runSpoken(gate: AnnouncementGate, cane: CaneConsumer, t0: number, t1: number, sum: (t: number) => unknown | null,
  over: (t: number) => object = () => ({}), exact = false) {
  const said: Array<[number, string, string, number | null]> = [];
  let busyUntil = -1e9, playing: Cue | null = null;
  for (let k = 0; t0 + k * 0.25 <= t1 + 1e-9; k++) {
    const t = +(t0 + k * 0.25).toFixed(3);
    const raw = sum(t);
    cane.update(raw == null ? null : (parseCane(raw) as CaneSummary), t);
    gate.observe(cane.pseudoTracks(t), t);
    if (Math.abs(t * 2 - Math.round(t * 2)) < 1e-6) {
      const busy = t < busyUntil - 1e-9;
      const ch: ChannelState = { busy, remaining: busy ? busyUntil - t : 0, currentUtility: busy ? playing!.utility : 0,
        currentSeverity: busy ? playing!.severity : 0, currentTrack: busy ? playing!.trackId : null };
      const cue = cane.propose(ctx(t, { channelBusy: busy, ...(exact ? { channelTrack: ch.currentTrack } : {}), ...over(t) }));
      for (const out of gate.filterAll(cue ? [cue] : [], cane.pseudoTracks(t), t, ch)) {
        said.push([t, out.action, out.message, out.distance]);
        if (out.action !== "haptic") { busyUntil = t + out.duration; playing = out; }
      }
    }
  }
  return said;
}

test("walking up to a wall: no cane cue within 2.5 s of the last unless it entered the 1 m band (cane.py min_gap_s)", () => {
  // the gate passes an obstacle with a measured TTC under 2 s at every decision while that TTC keeps falling: before this
  // rule, "Obstacle ahead, about 4 meters" at 0.5 s was followed by "about 1.5 meters" at 2.5 s, as soon as speech was free
  for (const exact of [false, true]) for (const aid of ["none", "cane", "guide-dog"] as const) for (const speed of [1.0, 0.6, 1.4]) for (const level of ["body", "head"]) {
    const { gate, cane } = fresh();
    const said = runSpoken(gate, cane, 0, 8, (t) => summary(Math.max(0.3, 4.0 - speed * t), { level }), () => ({ aid }), exact);
    const label = `${aid} ${level} ${speed} m/s${exact ? " (channelTrack)" : ""}: ${JSON.stringify(said)}`;
    const words = said.filter(([, a]) => a !== "haptic");
    for (let i = 1; i < words.length; i++) {
      const [t, , , d] = words[i], [tp, , , dp] = words[i - 1];
      assert.ok(t - tp >= 2.5 - 1e-9 || (d! <= 1.0 && dp! > 1.0), label);
    }
    // still warned: something is said or buzzed before the wall, and with no aid the first words come at once
    assert.ok(said.length >= 1, label);
    if (aid === "none") assert.ok(words[0][0] <= 0.5 + 1e-9, label);
    if (aid !== "none" || level === "body") assert.ok(said.some(([, a]) => a === "haptic"), label);
  }
});

test("ground and body are one obstacle: a level that flips between maps is not news (cane.py _recall)", () => {
  // the gate keys its memory by level, so the flip was said again at once as a new obstacle
  const flip = (t: number) => summary(2.0, { level: Math.floor(t / 2) % 2 ? "ground" : "body", suggest: null });
  const { gate, cane } = fresh();
  const said = runSpoken(gate, cane, 0, 8, flip, () => ({ aid: "none" }), true);
  assert.equal(said.length, 1, JSON.stringify(said));
  assert.match(said[0][2], /^Obstacle ahead, about 2 meters\.$/);
  // without channelTrack the cane only knows the cue was probably said: the flip waits out the 2.5 s gap, after which the
  // gate's own rules decide (it is new to the gate)
  const p = fresh();
  const probable = runSpoken(p.gate, p.cane, 0, 8, flip, () => ({ aid: "none" }));
  assert.ok(probable.length >= 1 && probable.every(([t], i) => !i || t - probable[i - 1][0] >= 2.5 - 1e-9), JSON.stringify(probable));
  // ... while head level stays its own identity (a cane passes under it) and is said
  const b = fresh();
  const head = runSpoken(b.gate, b.cane, 0, 8, (t) => summary(2.0, { level: t < 4 ? "body" : "head", suggest: null }), () => ({ aid: "none" }));
  assert.equal(head.length, 2, JSON.stringify(head));
  assert.match(head[1][2], /^Something at head height ahead/);
});

test("after a stop nothing is spoken for 3 s (the buzz said it), and the stop is not repeated", () => {
  for (const aid of ["none", "cane", "guide-dog"] as const) for (const exact of [false, true]) {
    const { gate, cane } = fresh();
    const said = runSpoken(gate, cane, 0, 9, (t) => summary(Math.max(0.3, 4.0 - t), { level: "head" }), () => ({ aid }), exact);
    const stops = said.filter(([, a]) => a === "haptic");
    assert.equal(stops.length, 1, `${aid}: ${JSON.stringify(said)}`);
    assert.ok(said.every(([t, a]) => a === "haptic" || t < stops[0][0] || t >= stops[0][0] + 3 - 1e-9), `${aid}: ${JSON.stringify(said)}`);
  }
});

test("the cane takes a cue as said only when the channel plays it, so a held or crowded-out cue is not a repeat", () => {
  const feed = (cane: CaneConsumer, from: number, to: number, d = 2.0) => { for (let t = from; t <= to + 1e-9; t += 0.25) cane.update(parseCane(summary(d, { suggest: null })), +t.toFixed(3)); };
  const at = (cane: CaneConsumer, t: number, over: object) => cane.propose(ctx(t, over));
  // the gate held it (the channel stays free at the next decision): proposed again, no gap of the cane's own
  let cane = new CaneConsumer(new Camera());
  feed(cane, 0, 1.0);
  const first = at(cane, 1.0, {})!;
  assert.equal(first.action, "speak");
  feed(cane, 1.25, 1.5);
  assert.ok(at(cane, 1.5, { channelBusy: false }));
  // said (it plays at the next decision): held for 2.5 s, then until it is nearer or 10 s have passed
  cane = new CaneConsumer(new Camera());
  feed(cane, 0, 1.0);
  at(cane, 1.0, {});
  feed(cane, 1.25, 1.5);
  assert.equal(at(cane, 1.5, { channelBusy: true, channelTrack: first.trackId }), null);
  feed(cane, 1.75, 3.0);
  assert.equal(at(cane, 3.0, {}), null);
  assert.match(cane.note, /said 2\.0 s ago/);
  feed(cane, 3.25, 4.0);
  assert.equal(at(cane, 4.0, {}), null);
  assert.match(cane.note, /not nearer/);
  feed(cane, 4.25, 4.5, 1.6);   // 0.4 m nearer: news
  assert.ok(at(cane, 4.5, {}));
  // probably said (the channel is busy, which track plays is not known): the 2.5 s gap only, then the gate decides
  cane = new CaneConsumer(new Camera());
  feed(cane, 0, 1.0);
  at(cane, 1.0, {});
  feed(cane, 1.25, 1.5);
  assert.equal(at(cane, 1.5, { channelBusy: true }), null);
  feed(cane, 1.75, 3.0);
  assert.equal(at(cane, 3.0, {}), null);
  assert.match(cane.note, /said 2\.0 s ago/);
  feed(cane, 3.25, 3.5);
  assert.ok(at(cane, 3.5, {}));
  // another cue played instead (the app passes channelTrack): not said, proposed again as soon as the channel is free
  cane = new CaneConsumer(new Camera());
  feed(cane, 0, 1.0);
  at(cane, 1.0, {});
  feed(cane, 1.25, 1.5);
  assert.equal(at(cane, 1.5, { channelBusy: true, channelTrack: 7 }), null);
  feed(cane, 1.75, 3.0);
  assert.ok(at(cane, 3.0, {}));
});

test("a slow device: a decision after the cue would have ended still remembers it, as probably said", () => {
  // decisions 1.75 s or more apart (the cue "Obstacle ahead, about 2 meters." lasts about 1.75 s): the next decision cannot
  // see the cue on the channel any more. It used to count as not said, so no memory was ever kept on such a device and the
  // gate let the cue through at every decision again.
  const feed = (cane: CaneConsumer, from: number, to: number, d = 2.0) => { for (let t = from; t <= to + 1e-9; t += 0.25) cane.update(parseCane(summary(d, { suggest: null })), +t.toFixed(3)); };
  for (const exact of [false, true]) {
    const cane = new CaneConsumer(new Camera());
    feed(cane, 0, 1.0);
    const first = cane.propose(ctx(1.0))!;
    assert.equal(first.action, "speak");
    assert.ok(first.duration < 1.8, `${first.duration}`);
    feed(cane, 1.25, 3.0);
    assert.equal(cane.propose(ctx(3.0, exact ? { channelTrack: null } : {})), null, `${exact}`);   // 2.0 s later, channel free
    assert.match(cane.note, /said 2\.0 s ago/);
    // probably said only: after the 2.5 s gap the gate decides (no "until nearer or 10 s" hold)
    feed(cane, 3.25, 5.25);
    assert.ok(cane.propose(ctx(5.25, exact ? { channelTrack: null } : {})), `${exact}: ${cane.note}`);
  }
  // decisions 2.25 s apart: the memory outlives the 2 s forget rule while its sector keeps reporting the obstacle
  const cane = new CaneConsumer(new Camera());
  feed(cane, 0, 1.0);
  cane.propose(ctx(1.0));
  feed(cane, 1.25, 3.25);
  assert.equal(cane.propose(ctx(3.25)), null);
  assert.match(cane.note, /said 2\.\d s ago/);
});

test("every stop the cane counts as given is one the gate lets through", () => {
  // the cane disarms its stop and keeps quiet for 3 s when it proposes a stop, before the gate judges it; that is right only
  // while the gate's own gaps for a buzz are shorter than the cane's gap between stops
  assert.ok(GATE_DEFAULTS.hapticGap <= CANE_CFG.stopGapS && GATE_DEFAULTS.criticalGap <= CANE_CFG.stopGapS,
    `gate hapticGap ${GATE_DEFAULTS.hapticGap} s, criticalGap ${GATE_DEFAULTS.criticalGap} s, cane stopGapS ${CANE_CFG.stopGapS} s`);
  // stops as close together as the cane allows: walked up to from 1 m at 1 m/s, then nothing in view, every 3 s (re-armed
  // after 2 s with nothing within 1 m, 3 s after the last stop)
  const none = { clear_m: { left: null, ahead: null, right: null }, obstacle: null, suggest: null };
  for (const aid of ["none", "cane", "guide-dog"] as const) {
    const { gate, cane } = fresh();
    const proposed: number[] = [], passed: number[] = [];
    for (let k = 0; k <= 40; k++) {
      const t = k * 0.25, phase = t % 3.0;
      cane.update(parseCane(phase <= 0.75 ? summary(Math.max(0.3, 1.0 - phase), { level: "head", suggest: null }) : none), t);
      gate.observe(cane.pseudoTracks(t), t);
      if (k % 2) continue;
      const cue = cane.propose(ctx(t, { aid }));
      if (cue?.action !== "haptic") continue;
      proposed.push(t);
      if (gate.filterAll([cue], cane.pseudoTracks(t), t, FREE).includes(cue)) passed.push(t);
    }
    assert.deepEqual(proposed, [0.5, 3.5, 6.5, 9.5], aid);
    assert.deepEqual(passed, proposed, aid);
  }
});

test("a path answer ends with the user's own travel aid, never a cane for a guide dog user", () => {
  // web/src/main.ts answer(); seeforme/live/commands.py AID_ADVICE has the same words (tests/test_live_commands.py)
  assert.equal(aidAdvice("cane"), "keep using your cane");
  assert.equal(aidAdvice("guide-dog"), "keep working with your guide dog");
  assert.equal(aidAdvice("none"), "go carefully");
  for (const aid of AIDS) assert.ok(!/clear|safe/.test(aidAdvice(aid)), aid);
  assert.ok(!/cane/.test(aidAdvice("guide-dog")) && !/cane|dog/.test(aidAdvice("none")));
});

test("the app tells the cane which cue the speech channel plays (main.ts passes channelTrack to cane.propose)", () => {
  // without it a cane cue counted as said whenever the channel was busy at the next decision, also when the scheduler's
  // cue had crowded it out, and the "until nearer or 10 s" hold never applied (the `exact` cases above model both)
  const main = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../src/main.ts"), "utf8");
  const call = main.match(/cane\.propose\(\{([\s\S]*?)\}\)/);
  assert.ok(call, "main.ts calls cane.propose({ ... })");
  assert.match(call[1], /channelTrack\s*:\s*[^,}]*currentTrack/);
});
