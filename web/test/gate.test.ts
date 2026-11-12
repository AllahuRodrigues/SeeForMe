// The announcement gate on scripted track sequences (10 frames/s, a decision every 0.5 s, as in the app): repeats are
// held, news passes (closer, closing, side change, critical, new identity, FIND goal), memory survives a tracker id
// switch, the desk user is held until a real approach, vehicles are never background, a cue held for confirmation is
// said after a flicker, the user's reported "Person, close, ahead." loop at a laptop is held, a different object where a
// remembered one vanished is not held as its repeat, and a cane buzz and a spoken warning pass at the same decision.
import { test } from "node:test";
import assert from "node:assert/strict";
import { distancePhrase } from "../src/compose";
import { AnnouncementGate, sameSign } from "../src/gate";
import type { BBox, ChannelState, Cue, Track } from "../src/types";
import { FREE, Sim, boxAt, boxAtRange, type Obj } from "./sim";

const AHEAD = boxAt(0.5, 0.2, 0.3, 0.8);

test("a repeat of the same object in the same direction is held; the reminder comes after the hold", () => {
  const s = new Sim();
  s.run(0, 25, () => [{ id: 1, bbox: AHEAD, z: 3.0 }], (t) => (t >= 1 && (t - 1) % 3 === 0 ? 1 : null));
  const said = s.passes();
  assert.equal(said[0].t, 1); assert.equal(said[0].reason, "new");
  assert.deepEqual(said.map((l) => l.t), [1, 22]);   // offers at 4, 7, ... 19 held; 22 > 1 + 20 s hold
  assert.equal(said[1].reason, "reminder");
  const held = s.log.find((l) => l.t === 4)!;
  assert.equal(held.cue, null); assert.equal(held.reason, "repeat");
  assert.match(held.text, /^held: same person ahead, said 3 s ago/);
});

test("closer by 1 m and 25% passes", () => {
  const s = new Sim();
  // said at 4.0 m, then walks closer at 0.2 m/s (too slow for "closing")
  s.run(0, 10, (t) => [{ id: 1, bbox: AHEAD, z: t < 2 ? 4.0 : Math.max(2.6, 4.0 - 0.2 * (t - 2)), v: t < 2 ? 0 : -0.2, path: 0.6 }],
    (t) => (t === 1 || t >= 3 ? 1 : null));
  const said = s.passes();
  assert.equal(said[0].reason, "new");
  assert.equal(said[1].reason, "closer");
  assert.ok(said[1].t <= 8.0 + 1e-9, `closer said at ${said[1].t}`);   // 4.0 -> 3.0 m at t = 7
});

test("still closing in the path within 3 m passes", () => {
  const s = new Sim();
  // said at 2.5 m, still until t = 3, then it comes closer at 0.6 m/s (TTC stays above 2 s at first)
  const z = (t: number) => (t < 3 ? 2.5 : 2.5 - 0.6 * (t - 3));
  s.run(0, 4.5, (t) => [{ id: 1, bbox: AHEAD, z: z(t), v: t < 3 ? 0 : -0.6, path: 0.8 }], (t) => (t >= 1 ? 1 : null));
  const said = s.passes();
  assert.equal(said[0].t, 1);
  const second = said[1];
  assert.ok(second, "the approach is said again");
  assert.ok(["closing", "critical"].includes(second.reason), second.reason);
  assert.ok(second.t <= 4.0 + 1e-9, `said again at ${second.t}`);   // 0.375 m nearer at t = 3.63: by the next decision
});

test("changing side passes", () => {
  const s = new Sim();
  const left = boxAt(0.15, 0.15, 0.3, 0.8), right = boxAt(0.85, 0.15, 0.3, 0.8);
  s.run(0, 6, (t) => [{ id: 1, bbox: t < 3.5 ? left : right, z: 3 }], (t) => (t === 1 || t === 4 ? 1 : null));
  const said = s.passes();
  assert.deepEqual(said.map((l) => l.reason), ["new", "sector"]);
});

test("critical passes, at most once per object every 2 s unless it gets worse", () => {
  const s = new Sim();
  const z = (t: number) => (t < 2 ? 3.0 : Math.max(0.7, 3.0 - 1.2 * (t - 2)));
  s.run(0, 2.4, (t) => [{ id: 1, bbox: AHEAD, z: z(t), v: t < 2 ? 0 : -1.2, path: 0.9 }], (t) => (t === 1 ? 1 : null));
  // at 3.0 m and closing at 1.2 m/s the TTC is 1.5 s: critical
  s.frame(2.5, [{ id: 1, bbox: AHEAD, z: 2.4, v: -1.2, path: 0.9, ttc: 1.5 }]);
  assert.equal(s.decide(2.5, 1)?.trackId, 1); assert.equal(s.gate.last?.reason, "critical");
  s.frame(3.0, [{ id: 1, bbox: AHEAD, z: 2.35, v: -1.2, path: 0.9, ttc: 1.45 }]);
  assert.equal(s.decide(3.0, 1), null); assert.equal(s.gate.last?.reason, "critical_gap");
  s.frame(3.5, [{ id: 1, bbox: AHEAD, z: 1.7, v: -1.2, path: 0.9, ttc: 0.9 }]);   // worse: TTC down by 0.6 s
  assert.equal(s.decide(3.5, 1)?.trackId, 1); assert.equal(s.gate.last?.reason, "critical");
});

test("a new identity passes, and a second person in view at the same time is a separate object", () => {
  const s = new Sim();
  const p1 = boxAt(0.45, 0.15, 0.3, 0.8), p2 = boxAt(0.6, 0.12, 0.35, 0.75), chair = boxAt(0.3, 0.12, 0.6, 0.9);
  s.run(0, 1.4, () => [{ id: 1, bbox: p1, z: 3 }, { id: 2, bbox: p2, z: 4 }, { id: 3, label: "chair", bbox: chair, z: 2.5 }], (t) => (t === 1 ? 1 : null));
  s.frame(1.5, [{ id: 1, bbox: p1, z: 3 }, { id: 2, bbox: p2, z: 4 }, { id: 3, label: "chair", bbox: chair, z: 2.5 }]);
  assert.ok(s.decide(1.5, 2), "the second person is new");
  s.frame(2.0, [{ id: 1, bbox: p1, z: 3 }, { id: 2, bbox: p2, z: 4 }, { id: 3, label: "chair", bbox: chair, z: 2.5 }]);
  assert.ok(s.decide(2.0, 3), "the chair is new");
  s.frame(2.5, [{ id: 1, bbox: p1, z: 3 }, { id: 2, bbox: p2, z: 4 }, { id: 3, label: "chair", bbox: chair, z: 2.5 }]);
  assert.equal(s.decide(2.5, 1), null, "the first person again is a repeat");
});

test("a FIND goal passes even when it was said a moment ago, and is never background", () => {
  const s = new Sim({ background: true, holdStatic: 60 });
  const chair = boxAt(0.3, 0.12, 0.6, 0.9);
  s.run(0, 5, () => [{ id: 1, label: "chair", bbox: chair, z: 3.5 }]);   // static through the warm-up: background
  s.frame(5.5, [{ id: 1, label: "chair", bbox: chair, z: 3.5 }]);
  assert.equal(s.decide(5.5, 1), null); assert.equal(s.gate.last?.reason, "background");
  s.gate.setGoal(["chair"]);
  s.frame(6.0, [{ id: 1, label: "chair", bbox: chair, z: 3.5 }]);
  assert.ok(s.decide(6.0, 1), "the FIND goal is said");
  s.frame(6.5, [{ id: 1, label: "chair", bbox: chair, z: 3.5 }]);
  assert.equal(s.decide(6.5, 1), null, "and not repeated at once");
});

test("a tracker id switch keeps the memory (the same person is not new)", () => {
  const s = new Sim();
  s.run(0, 3, (t) => (t < 2.05 ? [{ id: 1, bbox: AHEAD, z: 3 }] : [{ id: 2, bbox: boxAt(0.51, 0.2, 0.31, 0.8), z: 3 }]), (t) => (t === 1 ? 1 : null));
  s.frame(3.5, [{ id: 2, bbox: boxAt(0.51, 0.2, 0.31, 0.8), z: 3 }]);
  assert.equal(s.decide(3.5, 2), null);
  assert.equal(s.gate.last?.reason, "repeat");
});

test("the desk user is held, then said on a real approach (no later than one decision)", () => {
  const s = new Sim({ background: true, operatorFilter: true, holdStatic: 60 });
  const user = boxAt(0.5, 0.45, 0.15, 1.0);   // centred, 85% of the frame tall, 0.8 m away
  const z = (t: number) => (t < 10 ? 0.8 + 0.02 * Math.sin(7 * t) : Math.max(0.3, 0.8 - 0.6 * (t - 10)));
  let releasedAt: number | null = null;
  s.run(0, 11.5, (t) => [{ id: 1, bbox: user, z: z(t), v: t < 10 ? 0 : -0.6, path: 0.9, ttc: t < 10 ? Infinity : undefined }], (t) => {
    if (t >= 10 && releasedAt == null && 0.8 - z(t) >= 0.35) releasedAt = t;   // a smoothed drop of max(0.35 m, 15%)
    return t >= 1 ? 1 : null;   // the scheduler offers the user at every decision
  });
  const said = s.passes();
  assert.equal(said.filter((l) => l.t < 10).length, 0, `nothing said about the still user: ${JSON.stringify(said.map((l) => [l.t, l.reason]))}`);
  assert.deepEqual([...new Set(s.log.filter((l) => l.t >= 1 && l.t < 10).map((l) => l.reason))], ["warmup", "operator"]);
  assert.ok(said.length >= 1, "the approach is said");
  assert.ok(releasedAt != null);
  assert.ok(said[0].t <= releasedAt! + 0.5 + 1e-9, `said at ${said[0].t}, drop reached at ${releasedAt}`);
  assert.ok(["approach", "critical"].includes(said[0].reason), said[0].reason);
});

test("the desk user leaning in (no depth, box jitter) is not announced", () => {
  const s = new Sim({ background: true, operatorFilter: true, holdStatic: 60 });
  s.run(0, 30, (t) => {
    const lean = t > 12 && t < 16 ? 0.04 : 0;   // a lean: the box grows a little
    const j = 0.01 * Math.sin(13 * t);
    return [{ id: 1, bbox: [0.28 - lean + j, 0.2 - lean, 0.72 + lean - j, 1.0] as BBox, z: null, path: 0.9 }];
  }, (t) => (t >= 1 && t % 3 === 1 ? 1 : null));
  assert.equal(s.passes().length, 0, JSON.stringify(s.passes().map((l) => [l.t, l.text])));
  assert.deepEqual([...new Set(s.log.filter((l) => l.t >= 1 && l.t % 3 === 1).map((l) => l.reason))], ["warmup", "operator"]);
});

test("a vehicle is never background", () => {
  const s = new Sim({ background: true, holdStatic: 60 });
  const car = boxAt(0.7, 0.3, 0.4, 0.8);
  s.run(0, 5, () => [{ id: 1, label: "car", bbox: car, z: 8 }]);
  s.frame(5.5, [{ id: 1, label: "car", bbox: car, z: 8 }]);
  const out = s.decide(5.5, 1, "speak", FREE, "Car, on your right.");
  assert.ok(out, `held: ${s.gate.last?.text}`);
  assert.equal(s.gate.last?.reason, "new");
});

test("a cue held for confirmation is said once the track is confirmed, even after a one-frame flicker", () => {
  const s = new Sim();
  const b = boxAt(0.5, 0.2, 0.3, 0.8);
  s.frame(0.95, [{ id: 1, bbox: b, z: 3, hits: 1 }]);
  s.frame(1.0, [{ id: 1, bbox: b, z: 3, hits: 2 }]);   // two matches 0.05 s apart: not confirmed yet
  assert.equal(s.decide(1.0, 1), null); assert.equal(s.gate.last?.reason, "unconfirmed");
  for (let t = 1.1; t < 1.55; t += 0.1) s.frame(+t.toFixed(2), []);   // the track flickers out before it is confirmed
  assert.equal(s.decide(1.5, null), null);
  // it comes back as a new track id (the tracker lost it) and is confirmed: the held cue is said without a new offer
  s.frame(1.6, [{ id: 7, bbox: b, z: 3, hits: 1 }]); s.frame(1.7, [{ id: 7, bbox: b, z: 3, hits: 2 }]); s.frame(1.8, [{ id: 7, bbox: b, z: 3, hits: 3 }]);
  s.frame(1.9, [{ id: 7, bbox: b, z: 3, hits: 4 }]); s.frame(2.0, [{ id: 7, bbox: b, z: 3, hits: 5 }]);
  const out = s.decide(2.0, null);
  assert.ok(out, "the held cue is said");
  assert.equal(out!.trackId, 7);
  assert.equal(s.gate.last?.reason, "new");
  assert.equal(out!.message, "Person, about 3 meters, ahead.");   // held for a second: said as it is now (gate/compose.ts)

  // confirmed, then out of view at the decision (a sighting 0.4 s ago): said at that decision, not a flicker later
  const f = new Sim();
  f.frame(0.95, [{ id: 1, bbox: b, z: 3, hits: 1 }]); f.frame(1.0, [{ id: 1, bbox: b, z: 3, hits: 2 }]);
  assert.equal(f.decide(1.0, 1), null);
  f.frame(1.1, [{ id: 1, bbox: b, z: 3 }]);
  for (let t = 1.2; t < 1.55; t += 0.1) f.frame(+t.toFixed(2), []);
  assert.equal(f.decide(1.5, null)?.trackId, 1);
});

test("the reported loop: 'Person, close, ahead.' every 3.6 s at 3.2 m (box guess, noisy TTC, id switches) is held", () => {
  // the user sat at a laptop; no metric depth, the box cut off at the bottom gives 3.2 m and an assumed TTC of ~2.6 s.
  // Walking settings (the worst case: the camera was not detected as still).
  const s = new Sim({ background: false, operatorFilter: false, holdStatic: 20 });
  let id = 1;
  const said: number[] = [], heldWhy = new Set<string>();
  for (let k = 0; 40 + k * 0.1 <= 62.6; k++) {
    const t = +(40 + k * 0.1).toFixed(2);
    if (Math.abs(t - 47.0) < 1e-6 || Math.abs(t - 54.0) < 1e-6) id++;   // the tracker switches ids now and then
    const j = 0.006 * Math.sin(17 * t), jj = 0.006 * Math.cos(11 * t);
    s.frame(t, [{ id, bbox: [0.3 + j, 0.33 + jj, 0.7 - jj, 1.0], z: null, path: 0.9, ttc: 2.6 + 0.3 * Math.sin(5 * t) }]);
    const offer = [43.0, 46.6, 48.6, 50.2, 53.8, 54.3, 57.4, 57.9, 61.0, 61.5].some((x) => Math.abs(x - t) < 1e-6);
    if (offer || Math.abs(t * 2 - Math.round(t * 2)) < 1e-6) {
      if (s.decide(t, offer ? id : null)) said.push(t);
      else if (offer) heldWhy.add(s.gate.last!.reason);
    }
  }
  assert.deepEqual(said, [43.0]);
  assert.deepEqual([...heldWhy], ["repeat"]);
});

test("haptic and interrupt: at most one per object every 2 s", () => {
  const s = new Sim();
  s.run(0, 1, () => [{ id: 1, bbox: AHEAD, z: 2.0, v: -0.5, path: 0.8 }]);
  s.frame(1.5, [{ id: 1, bbox: AHEAD, z: 1.9, v: -0.5, path: 0.8 }]);
  assert.ok(s.decide(1.5, 1, "haptic"));
  s.frame(2.5, [{ id: 1, bbox: AHEAD, z: 1.85, v: -0.5, path: 0.8 }]);
  assert.equal(s.decide(2.5, 1, "haptic"), null); assert.equal(s.gate.last?.reason, "haptic_gap");
  s.frame(3.6, [{ id: 1, bbox: AHEAD, z: 1.8, v: -0.5, path: 0.8 }]);
  assert.ok(s.decide(3.6, 1, "haptic"));

  // interrupts: a static object already said is a repeat even as an interrupt; a moving one cuts in, once per 2 s
  const busy: ChannelState = { busy: true, remaining: 1, currentUtility: 0.5, currentSeverity: 0.3, currentTrack: 99 };
  const u = new Sim();
  u.run(0, 1, () => [{ id: 1, bbox: AHEAD, z: 3 }, { id: 2, label: "bicycle", bbox: boxAt(0.2, 0.15, 0.4, 0.8), z: 5, v: -1.0 }], (t) => (t === 1 ? 1 : null));
  u.run(1.1, 3.4, (t) => [{ id: 1, bbox: AHEAD, z: 3 }, { id: 2, label: "bicycle", bbox: boxAt(0.2 + 0.02 * t, 0.15, 0.4, 0.8), z: 5 - (t - 1), v: -1.0 }],
    (t) => (t === 1.5 ? 2 : null));
  u.frame(3.5, [{ id: 1, bbox: AHEAD, z: 3 }, { id: 2, label: "bicycle", bbox: boxAt(0.27, 0.15, 0.4, 0.8), z: 2.6, v: -1.0 }]);
  assert.equal(u.decide(3.5, 1, "interrupt", busy), null); assert.equal(u.gate.last?.reason, "repeat");
  assert.ok(u.decide(3.5, 2, "interrupt", busy), u.gate.last?.text);   // said at 1.5, moving: cuts in after 2 s
  u.frame(4.0, [{ id: 1, bbox: AHEAD, z: 3 }, { id: 2, label: "bicycle", bbox: boxAt(0.28, 0.15, 0.4, 0.8), z: 2.5, v: -1.0 }]);
  assert.equal(u.decide(4.0, 2, "interrupt", busy), null);
});

test("a held object that later closes in is said by the gate itself, with a fresh message", () => {
  const s = new Sim({ background: true, holdStatic: 60 });
  const p = boxAt(0.5, 0.18, 0.3, 0.85);
  const z = (t: number) => (t < 6 ? 3.0 : Math.max(1.0, 3.0 - 0.8 * (t - 6)));
  // background (static through the warm-up), offered once at t = 5 and held; then it approaches and is not offered again
  s.run(0, 7.5, (t) => [{ id: 1, bbox: p, z: z(t), v: t < 6 ? 0 : -0.8, path: 0.9 }], (t) => (t === 5 ? 1 : null));
  const said = s.passes();
  assert.equal(said.length, 1, JSON.stringify(s.log.map((l) => [l.t, l.reason])));
  const cue = said[0].cue!;
  assert.equal(cue.source, "gate");
  // composed now, from where the person is now (the held cue said "Person, close, ahead.")
  assert.equal(cue.message, `Person, ${distancePhrase(cue.distance, 0.9)}, ahead.`);
  assert.ok(said[0].t <= 6.5 + 1e-9, `said at ${said[0].t}`);
});

test("signs: numbers tell them apart, OCR noise does not", () => {
  assert.equal(sameSign("room 204", "r0om 2o4"), true);
  assert.equal(sameSign("room 204", "room 206"), false);
  assert.equal(sameSign("exit", "exit 2"), false);
  assert.equal(sameSign("gate 12", "gate 13"), false);
});

// ---- id switches that are not: a different object where a remembered one vanished ---------------------------------------
/** A is said at t = 1 and vanishes at `vanish`; B appears `gap` s later where A was last seen and comes closer, and the
 *  scheduler offers B at every decision (a bare scheduler says a new track at its first offer). When B is first said. */
function successorRun(label: string, depth: boolean, a: (t: number) => number, vanish: number, gap: number, b0: number, vB: number, path = 0.8) {
  const s = new Sim();
  const bT = vanish + gap, dB = (t: number) => b0 - vB * (t - bT);
  s.run(0, 20, (t) => {
    const o: Obj[] = [];
    if (t <= vanish + 1e-9) o.push({ id: 1, label, bbox: boxAtRange(label, a(t)), z: depth ? a(t) : null, v: depth ? (a(t + 0.1) - a(t)) / 0.1 : 0, path });
    if (t >= bT - 1e-9 && dB(t) > 0.7) o.push({ id: 2, label, bbox: boxAtRange(label, dB(t)), z: depth ? dB(t) : null, v: depth ? -vB : 0, path });
    return o;
  }, (t) => (Math.abs(t - 1) < 1e-6 ? 1 : t >= bT - 1e-9 ? 2 : null));
  const firstOffer = s.log.find((l) => l.t >= bT - 1e-9)!.t;
  const said = s.log.find((l) => l.cue?.trackId === 2);
  return { s, firstOffer, said };
}

test("one object walks away and vanishes, another comes from there: the second is said at once, not held as a repeat", () => {
  // the reviewer's case: A said at 4 m (6 m for the car), recedes, vanishes at 12 m (25 m); B appears there 1 s later
  for (const [label, d0, d1, vB] of [["person", 4, 12, 1.5], ["car", 6, 25, 5]] as const) {
    for (const depth of [false, true]) {
      const { firstOffer, said } = successorRun(label, depth, (t) => (t <= 1 ? d0 : d0 + ((d1 - d0) * (t - 1)) / 4), 5, 1, d1, vB);
      assert.ok(said, `${label} depth=${depth}: B never said`);
      // one decision to confirm the new track, as for any new object
      assert.ok(said!.t <= firstOffer + 0.5 + 1e-9, `${label} depth=${depth}: B first said at ${said!.t} (${said!.reason}), offered from ${firstOffer}`);
    }
  }
});

test("a static object hidden for a moment, and an object at its place that then approaches: said at once", () => {
  for (const [label, d, vB] of [["person", 12, 1.5], ["car", 25, 5]] as const) {
    for (const depth of [false, true]) {
      // out of view for 1 s (a dropout or occlusion): back in the path, no farther -> "reacquired"
      const r = successorRun(label, depth, () => d, 5, 1, d, vB);
      assert.ok(r.said && r.said.t <= r.firstOffer + 0.5 + 1e-9, `${label} depth=${depth} gap 1 s: said at ${r.said?.t} (${r.said?.reason})`);
      assert.equal(r.said!.reason, "reacquired");
    }
  }
  // a switch-like gap (0.3 s), metric depth or a fast object: its closing gives it away within one more decision
  for (const [label, depth, d, vB] of [["person", true, 12, 1.5], ["car", false, 25, 5], ["car", true, 25, 5]] as const) {
    const r = successorRun(label, depth, () => d, 5, 0.3, d, vB, 0.1);
    assert.ok(r.said && r.said.t <= r.firstOffer + 1.0 + 1e-9, `${label} depth=${depth} gap 0.3 s: said at ${r.said?.t} (${r.said?.reason}), offered from ${r.firstOffer}`);
  }
});

test("a new track at a clearly different range where a remembered object vanished is a new object", () => {
  // A said at 4 m stays put and vanishes; 1 s later another person stands 9 m away in the same direction
  const r = successorRun("person", false, () => 4, 5, 1, 9, 0);
  assert.equal(r.said?.reason, "new");
  assert.ok(r.said!.t <= r.firstOffer + 0.5 + 1e-9);
});

test("a static object that only flickers out (id switch at the same place and range) is still a repeat", () => {
  for (const gap of [0.1, 0.5]) {
    const r = successorRun("person", false, () => 3, 5, gap, 3, 0);
    const held = r.s.log.filter((l) => l.t >= r.firstOffer && l.t < 20 && !l.cue).map((l) => l.reason);
    assert.equal(r.said, undefined, `gap ${gap}: said at ${r.said?.t} (${r.said?.reason})`);
    assert.ok(held.length >= 10 && held.every((x) => x === "repeat" || x === "unconfirmed"), JSON.stringify([...new Set(held)]));
  }
});

// ---- two channels ------------------------------------------------------------------------------------------------------
const cueFor = (tr: Track, t: number, action: Cue["action"], message: string, extra: Partial<Cue> = {}): Cue => ({
  t, trackId: tr.id, label: tr.label, message, action, duration: action === "haptic" ? 0 : 1.5, utility: 1.5, severity: 0.8,
  distance: tr.distance, depthConf: tr.depthConf, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: ["object"], ...extra });

test("a cane buzz and the scheduler's warning pass together (a haptic never hides a spoken cue)", () => {
  const s = new Sim();
  const carBox = (z: number): BBox => [0.4, 0.5 - 0.6 / z, 0.6, 0.5 + 0.6 / z];
  const objs = (t: number): Obj[] => [{ id: 1, label: "car", bbox: carBox(9 - 2 * t), z: 9 - 2 * t, v: -2, path: 0.9 },
    { id: -101, label: "obstacle", bbox: [0.45, 0.6, 0.55, 1.0], z: 0.5, path: 0.9 }];
  for (let k = 0; k <= 5; k++) s.frame(+(k * 0.1).toFixed(2), objs(k * 0.1));
  const vis = [...s.tracks.values()].filter((tr) => tr.misses === 0);
  const car = s.tracks.get(1)!, obs = s.tracks.get(-101)!;
  const buzz = cueFor(obs, 0.5, "haptic", "", { identity: "obstacle:ahead:body", source: "cane" });
  const warn = cueFor(car, 0.5, "speak", "Car ahead, approaching.");
  const out = s.gate.filterAll([buzz, warn], vis, 0.5, FREE);
  assert.deepEqual(out.map((c) => c.action), ["haptic", "speak"]);
  assert.equal(out[1], warn);
  assert.deepEqual(s.gate.verdicts.map((v) => v.reason), ["haptic", "new"]);
});

test("a second speech cue at one decision is not said: refunded, and said later if the scheduler does not repeat it", () => {
  const s = new Sim();
  const policy = { budget: 5, cfg: { budgetCapacity: 10, units: "m", sureConfidence: 0.6 }, announcedDist: new Map(), goalTags: [], egoSpeed: 1, lastCandidates: [] };
  s.gate.policy = policy as unknown as AnnouncementGate["policy"];
  const objs = (): Obj[] => [{ id: 1, bbox: boxAt(0.3, 0.15, 0.3, 0.8), z: 3 }, { id: 2, label: "chair", bbox: boxAt(0.7, 0.15, 0.5, 0.9), z: 2.5 }];
  for (let k = 0; k <= 5; k++) s.frame(+(k * 0.1).toFixed(2), objs());
  const vis = [...s.tracks.values()].filter((tr) => tr.misses === 0);
  const first = cueFor(s.tracks.get(1)!, 0.5, "speak", "Person, on your left.");
  const second = cueFor(s.tracks.get(2)!, 0.5, "speak", "Chair, on your right.");
  policy.budget -= first.duration + second.duration;   // the scheduler charged both
  const out = s.gate.filterAll([first, second], vis, 0.5, FREE);
  assert.deepEqual(out, [first]);
  assert.equal(s.gate.verdicts[1].reason, "channel");
  assert.equal(policy.budget, 5 - first.duration);       // the crowded-out cue's seconds came back
  s.frame(1.0, objs());
  const later = s.gate.filterAll([], [...s.tracks.values()].filter((tr) => tr.misses === 0), 1.0, FREE);
  assert.equal(later.length, 1);
  assert.equal(later[0].trackId, 2); assert.equal(later[0].source, "gate");
});
