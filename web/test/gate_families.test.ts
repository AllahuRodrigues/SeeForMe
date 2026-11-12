// The failure families the Python gate's verifier found (seeforme/live/gate.py, gate_verify1), each as a scripted case
// against web/src/gate.ts. The safety principle: never warn later about an approaching, in-path, vehicle or FIND-goal
// object than the bare scheduler would (allowance: one 0.5 s decision step, or at least 1.5 s before contact when the
// bare cue carries news).
//   1 desk     after the desk user leans in (or leaves), a stranger walking up, or a visitor walking up in front of the
//              user, is warned about as if no user were there: what is held as "the user" must be the user
//   2 commit   a cue the gate lets through is one the app acts on (one per channel); a cue it does not let through is
//              not remembered as said
//   3 bg       a background record does not adopt a different same-label newcomer (range and timing continuity), and a
//              cue held as background is said when its object is released as approaching, even seconds later
//   4 succ     a newcomer inherits no memory without range and position continuity, also when the tracker keeps the old
//              track id across a dropout
//   5 flicker  a cue held for confirmation under heavy flicker (2-frame sightings, a new track id each) is said in time;
//              the same for a FIND goal under dropouts
import { test } from "node:test";
import assert from "node:assert/strict";
import type { ChannelState, Cue } from "../src/types";
import { FREE, Sim, boxAt, boxAtRange, type Obj } from "./sim";

const DESK = { background: true, operatorFilter: true, holdStatic: 60 };   // main.ts at a desk (camera still)
const USER = boxAt(0.5, 0.45, 0.15, 1.0), LEAN = boxAt(0.5, 0.62, 0.04, 1.0);
const said = (s: Sim, id: number) => s.log.filter((l) => l.cue?.trackId === id);
const show = (s: Sim, from = 0) => JSON.stringify(s.log.filter((l) => l.t >= from).map((l) => [l.t, l.cue ? `SAID ${l.cue.trackId}` : "", l.reason]));

// ---- 1: the desk user ---------------------------------------------------------------------------------------------------------
/** A desk user (id 1, 0.7 m, depth or not) leans in `leans` times; for "stranger" they leave at t = 20 and a stranger (id 2)
 *  walks up to the camera from 5 m at 1 m/s from t = 22; for "visitor" the user stays and the visitor walks up in front of
 *  them. The scheduler offers the user every second and the newcomer at every decision once it is in view. `user` false:
 *  the control (no desk user at all). A chair on the left is in view throughout, so the warm-up ends at 3 s in both. */
function deskRun(variant: "stranger" | "visitor", depth: boolean, leans: number, user = true) {
  const s = new Sim(DESK), tIn = 22, d0 = 5, v = 1.0, contact = tIn + (d0 - 0.6) / v;
  const dN = (t: number) => d0 - v * (t - tIn);
  const leaning = (t: number) => (leans >= 1 && t >= 8 && t < 10) || (leans >= 2 && t >= 13 && t < 15) || (leans >= 3 && t >= 17 && t < 18.5);
  s.run(0, contact - 0.1, (t) => {
    const o: Obj[] = [{ id: 9, label: "chair", bbox: boxAt(0.12, 0.1, 0.55, 0.85), z: depth ? 3.5 : null, path: 0.1 }];
    if (user && (variant === "visitor" || t < 20)) {
      const lean = leaning(t);
      o.push({ id: 1, bbox: lean ? LEAN : USER, z: depth ? (lean ? 0.35 : 0.7) : null, path: 0.9 });
    }
    if (t >= tIn - 1e-9) o.push({ id: 2, bbox: boxAtRange("person", dN(t)), z: depth ? dN(t) : null, v: depth ? -v : 0, path: 0.9 });
    return o;
  }, (t) => (t >= tIn - 1e-9 ? 2 : user && t >= 1 && Math.abs(t - Math.round(t)) < 1e-6 && (variant === "visitor" || t < 20) ? 1 : null));
  return { s, contact, firstOffer: tIn };
}

for (const variant of ["stranger", "visitor"] as const) {
  test(`1 desk: after the desk user leans in, a ${variant} walking up is warned about as if no user were there`, () => {
    for (const depth of [true, false]) {
      const ctl = deskRun(variant, depth, 0, false);
      const ctlSaid = said(ctl.s, 2);
      assert.ok(ctlSaid.length >= 2, `control ${depth}: ${show(ctl.s, 22)}`);
      for (const leans of [0, 1, 3]) {
        const r = deskRun(variant, depth, leans);
        const got = said(r.s, 2), tag = `${variant} depth=${depth} leans=${leans}`;
        assert.ok(got.length && got[0].t <= r.firstOffer + 0.5 + 1e-9, `${tag}: first said ${got[0]?.t}: ${show(r.s, 22)}`);
        // every warning the control gives is given no later than one decision after it (the user's memory is the user's)
        for (const c of ctlSaid) {
          assert.ok(got.some((g) => g.t <= c.t + 0.5 + 1e-9 && g.t >= c.t - 2.5), `${tag}: control said it at ${c.t} (${c.reason}), desk run: ${show(r.s, 22)}`);
        }
        // and a warning about the newcomer within 2.5 m, at least 1.5 s before contact (the bare scheduler says it there)
        assert.ok(got.some((g) => g.t >= r.contact - 2.0 - 1e-9 - 0.5 && g.t <= r.contact - 1.4 + 1e-9), `${tag}: no warning near 2 m: ${show(r.s, 22)}`);
      }
    }
  });
}

test("1 desk: the user is still held after leaning in (one cue at most per deep lean), and an id switch of the user stays the user", () => {
  for (const depth of [true, false]) {
    const s = new Sim(DESK);
    let id = 1;
    s.run(0, 40, (t) => {
      if (Math.abs(t - 25) < 1e-6) id = 7;   // the tracker switches the user's id in place
      const lean = (t >= 8 && t < 10) || (t >= 30 && t < 32);
      return [{ id, bbox: lean ? LEAN : USER, z: depth ? (lean ? 0.55 : 0.7) : null, path: 0.9 }];
    }, (t) => (t >= 1 && Math.abs(t - Math.round(t)) < 1e-6 ? id : null));
    assert.equal(s.passes().length, 0, `depth=${depth}: ${JSON.stringify(s.passes().map((l) => [l.t, l.reason]))}`);
  }
});

test("1 desk: the user back from a break and settled at their place is the user again (said only on the way in)", () => {
  for (const depth of [true, false]) {
    const s = new Sim(DESK);
    // the user leaves at 20 and walks back in from 4 m at 30-33 s, then sits (a new track id)
    const back = (t: number) => Math.max(0.7, 4 - 1.1 * (t - 30));
    s.run(0, 70, (t) => {
      const o: Obj[] = [{ id: 9, label: "chair", bbox: boxAt(0.12, 0.1, 0.55, 0.85), z: depth ? 3.5 : null, path: 0.1 }];
      if (t < 20) o.push({ id: 1, bbox: USER, z: depth ? 0.7 : null, path: 0.9 });
      else if (t >= 30) {
        const d = back(t), box = d <= 0.7 + 1e-9 ? USER : boxAtRange("person", d);
        o.push({ id: 2, bbox: box, z: depth ? d : null, v: depth && d > 0.7 ? -1.1 : 0, path: 0.9 });
      }
      return o;
    }, (t) => (Math.abs(t - Math.round(t)) < 1e-6 && t >= 1 && (t < 20 || t >= 30) ? (t < 20 ? 1 : 2) : null));
    // walking in at 1.1 m/s is news (with depth: critical); once at rest for 3 s at the user's place it is the user
    const after = said(s, 2), settled = 30 + (4 - 0.7) / 1.1;
    assert.ok(after.length >= 1 && after.every((l) => l.t <= settled + 3.5 + 1e-9), `depth=${depth}: ${JSON.stringify(after.map((l) => [l.t, l.reason]))}`);
    assert.ok(s.log.some((l) => l.t > 40 && l.reason === "operator"), `depth=${depth}: not held as the user: ${show(s, 34)}`);
  }
});

// ---- 2: what passes is acted on, what does not is not remembered ----------------------------------------------------------
test("2 commit: at most one speech and one haptic cue per decision, and every cue the gate commits is returned", () => {
  const s = new Sim();
  const mk = (id: number, t: number, action: Cue["action"]): Cue => {
    const tr = s.tracks.get(id)!;
    return { t, trackId: id, label: tr.label, message: `${tr.label} ${id}`, action, duration: action === "haptic" ? 0 : 1.2, utility: 1 + id / 10,
      severity: 0.5, distance: tr.distance, depthConf: tr.depthConf, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: ["object"] };
  };
  const objs = (t: number): Obj[] => [1, 2, 3, 4].filter((i) => (Math.floor(t * 3) + i) % 5 !== 0)   // flicker some of them
    .map((i) => ({ id: i, label: i % 2 ? "person" : "chair", bbox: boxAt(0.15 + 0.22 * (i - 1), 0.12, 0.3, 0.85), z: 4 - 0.15 * t + 0.1 * i, v: -0.15, path: 0.5 }));
  let rng = 7;
  const rand = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let k = 0; k <= 200; k++) {
    const t = +(k * 0.1).toFixed(2);
    s.frame(t, objs(t));
    if (k % 5) continue;
    const vis = s.visible();
    const offered = vis.filter(() => rand() < 0.5).map((tr) => mk(tr.id, t, rand() < 0.25 ? "haptic" : rand() < 0.2 ? "interrupt" : "speak"));
    const busy: ChannelState = rand() < 0.3 ? { busy: true, remaining: 1, currentUtility: 0.5, currentSeverity: 0.3, currentTrack: 99 } : FREE;
    const out = s.gate.filterAll(offered, vis, t, busy);
    assert.ok(out.filter((c) => c.action === "haptic").length <= 1 && out.filter((c) => c.action !== "haptic").length <= 1, `${t}: ${JSON.stringify(out)}`);
    assert.equal(s.gate.verdicts.filter((v) => v.passed).length, out.length, `${t}: ${JSON.stringify(s.gate.verdicts)}`);
  }
});

test("2 commit: a cue crowded out of the speech channel is not remembered as said", () => {
  const s = new Sim();
  const objs = (): Obj[] => [{ id: 1, bbox: boxAt(0.3, 0.15, 0.3, 0.8), z: 3 }, { id: 2, label: "chair", bbox: boxAt(0.7, 0.15, 0.5, 0.9), z: 2.5 }];
  for (let k = 0; k <= 5; k++) s.frame(+(k * 0.1).toFixed(2), objs());
  const vis = s.visible();
  const cue = (id: number): Cue => ({ t: 0.5, trackId: id, label: s.tracks.get(id)!.label, message: "x", action: "speak", duration: 1, utility: 1, severity: 0.5,
    distance: 3, depthConf: 0.9, ttc: null, slots: ["object"] });
  assert.equal(s.gate.filterAll([cue(1), cue(2)], vis, 0.5, FREE).length, 1);
  // the chair was never said: offered on its own next time (the channel is busy, so no rescue took it), it is new
  s.frame(1.0, objs());
  const busy: ChannelState = { busy: true, remaining: 1, currentUtility: 0.5, currentSeverity: 0.3, currentTrack: 1 };
  s.gate.filterAll([], s.visible(), 1.0, busy);
  s.frame(1.5, objs());
  const out = s.gate.filterAll([{ ...cue(2), t: 1.5 }], s.visible(), 1.5, FREE);
  assert.equal(out.length, 1); assert.equal(s.gate.last?.reason, "new");
});

// ---- 3: background records ------------------------------------------------------------------------------------------------
/** A static background object (id 1) at bgD vanishes at t = 10; a newcomer of the same label (id 2) appears where it stood
 *  `gap` s later at d0 and, from `go`, comes closer at v m/s. The (learned) scheduler offers the newcomer once, at its
 *  first decision in view, and never again. */
function bgRun(label: string, depth: boolean, bgD: number, d0: number, go: number, v: number, gap = 0.3) {
  const s = new Sim({ background: true, holdStatic: 60 });
  const tIn = 10 + gap, dN = (t: number) => (t < go ? d0 : d0 - v * (t - go)), contact = go + (d0 - 0.6) / v;
  let offered = -1;
  s.run(0, contact - 0.1, (t) => {
    const o: Obj[] = [];
    if (t <= 10 + 1e-9) o.push({ id: 1, label, bbox: boxAtRange(label, bgD), z: depth ? bgD : null, path: 0.9 });
    if (t >= tIn - 1e-9) o.push({ id: 2, label, bbox: boxAtRange(label, dN(t)), z: depth ? dN(t) : null, v: depth && t >= go ? -v : 0, path: 0.9 });
    return o;
  }, (t) => {
    if (offered < 0 && t >= tIn + 0.15) { offered = t; return 2; }
    return null;
  });
  return { s, contact, offered };
}

test("3 bg: a background record does not adopt a different newcomer, and a held background cue is said when its object approaches", () => {
  for (const label of ["person", "chair"]) for (const depth of [false, true]) {
    // a different object (6 m where the background stood at 4 m) that walks at the user at once
    const a = bgRun(label, depth, 4, 6, 10.3, 1.0);
    const ga = said(a.s, 2)[0];
    assert.ok(ga && (ga.t <= a.offered + 0.5 + 1e-9 || ga.t <= a.contact - 1.5 + 1e-9), `${label} depth=${depth} different: ${show(a.s, 10)}`);
    assert.ok(ga.t <= a.offered + 0.5 + 1e-9, `${label} depth=${depth}: a different object is new (${ga.reason} at ${ga.t})`);
    // indistinguishable from the background object (same place and range): held, then said once it approaches, even 4 s later
    const b = bgRun(label, depth, 4, 4, 14.5, 1.0);
    const gb = said(b.s, 2)[0];
    assert.ok(gb && gb.t <= b.contact - 1.5 + 1e-9, `${label} depth=${depth} same range, approaches at 14.5 (contact ${b.contact}): ${show(b.s, 10)}`);
  }
});

// ---- 4: successors -----------------------------------------------------------------------------------------------------------
test("4 succ: a car that leaves and another that comes from farther away are two cars", () => {
  for (const depth of [true, false]) {
    const s = new Sim();
    const contact = 12 + (6.4 - 0.6) / 3;
    s.run(0, contact - 0.1, (t) => {
      const o: Obj[] = [];
      if (t <= 8) o.push({ id: 1, label: "car", bbox: boxAtRange("car", 3.3, 0.5 + 0.08 * Math.max(0, t - 6)), z: depth ? 3.3 : null, path: 0.6 });
      if (t >= 8.5) { const d = t < 12 ? 6.4 : 6.4 - 3 * (t - 12); o.push({ id: 2, label: "car", bbox: boxAtRange("car", d, 0.62), z: depth ? d : null, v: depth && t >= 12 ? -3 : 0, path: 0.6 }); }
      return o;
    }, (t) => (Math.abs(t - 1) < 1e-6 ? 1 : t >= 12 - 1e-9 && Math.abs(t - 12) < 1e-6 ? 2 : null));
    const g = said(s, 2)[0];
    assert.ok(g && g.t <= 12.5 + 1e-9, `depth=${depth}: car B ${show(s, 8)}`);
  }
});

test("4 succ: the tracker keeps the background person's id across a dropout, and a farther person approaches there: news", () => {
  // background person at 3.0 m (camera still); a 0.3 s dropout; the same track id now holds a person at 4.5 m who walks at
  // the user at 1.5 m/s with no metric depth. The scheduler offers it at every decision once it moves.
  for (const depth of [false, true]) {
    const s = new Sim({ background: true, holdStatic: 60 });
    const go = 8.0, d = (t: number) => (t < go ? 4.5 : 4.5 - 1.5 * (t - go)), contact = go + (4.5 - 0.6) / 1.5;
    s.run(0, contact - 0.1, (t) => {
      if (t < 7.0) return [{ id: 1, bbox: boxAtRange("person", 3.0), z: depth ? 3.0 : null, path: 0.9 }];
      if (t < 7.3) return [];
      return [{ id: 1, bbox: boxAtRange("person", d(t)), z: depth ? d(t) : null, v: depth && t >= go ? -1.5 : 0, path: 0.9 }];
    }, (t) => (s.visible().length && (t >= go - 1e-9 || (t >= 1 && Math.abs(t - Math.round(t)) < 1e-6)) ? 1 : null));
    const g = s.log.find((l) => l.t >= 7.3 && l.cue?.trackId === 1);
    assert.ok(g && (g.t <= go + 0.5 + 1e-9 || g.t <= contact - 1.5 + 1e-9), `depth=${depth}: ${show(s, 6.5)}`);
  }
});

// ---- 5: flicker ------------------------------------------------------------------------------------------------------------
/** A person walks at the user from d0 at 1 m/s and is seen only in 2-frame bursts every 0.6 s, each burst a new track id
 *  (the tracker hands the gate a track from its second match). The scheduler offers whatever is in view at a decision. */
function flickerRun(depth: boolean, goal: boolean, d0 = 6) {
  const s = new Sim({}, 2);
  if (goal) s.gate.setGoal(["person"]);
  const contact = (d0 - 0.6) / 1.0, d = (t: number) => d0 - t;
  let bursts = 0;
  s.run(0, contact - 0.1, (t) => {
    const k = Math.round((t - 0.4) / 0.6), b0 = 0.4 + 0.6 * k;
    if (k < 0 || t < b0 - 1e-6 || t > b0 + 0.1 + 1e-6) return [];
    if (Math.abs(t - b0) < 1e-6) bursts++;
    return [{ id: 100 + k, bbox: boxAtRange("person", d(t), 0.52), z: depth ? d(t) : null, v: depth ? -1 : 0, path: 0.9 }];
  }, (t) => s.visible()[0]?.id ?? null);
  const firstOffer = s.log.find((l) => l.reason !== "" || l.cue)?.t ?? Infinity;
  return { s, contact, firstOffer, bursts };
}

test("5 flicker: a person seen only in 2-frame sightings (a new id each) is said in time", () => {
  for (const depth of [false, true]) {
    const r = flickerRun(depth, false);
    const g = r.s.log.find((l) => l.cue);
    assert.ok(r.firstOffer < 1, `offered from ${r.firstOffer}`);
    assert.ok(g && (g.t <= r.firstOffer + 0.5 + 1e-9 || g.t <= r.contact - 1.5 + 1e-9), `depth=${depth} (contact ${r.contact}): ${show(r.s)}`);
  }
});

test("5 flicker: a FIND goal under heavy dropouts is said in time", () => {
  for (const depth of [false, true]) {
    const r = flickerRun(depth, true, 4.2);
    const g = r.s.log.find((l) => l.cue);
    assert.ok(g && (g.t <= r.firstOffer + 0.5 + 1e-9 || g.t <= r.contact - 1.5 + 1e-9), `depth=${depth} (contact ${r.contact}): ${show(r.s)}`);
  }
});
