// AnnouncementGate: a "what did I already tell the user?" memory in front of either scheduler. A port of the essentials
// of seeforme/live/gate.py (its module docstring and docs/live_demo.md hold the evidence behind every rule).
//
// Both schedulers decide per track id and forget quickly: a track is blocked for recentBlock (2.8 s) after a cue and its
// novelty recovers with tau = 10 s. At a laptop that turns the user's own body into "Person, close, ahead." every 3-6 s,
// and every tracker id switch looks like a new person. The gate remembers announcements by identity instead:
//
//   identity   the label (truck / bus -> car, ...) for objects; the normalised text for signs, where two OCR readings are
//              one sign only if their numbers are identical ("ROOM 204" / "R0OM 2O4" are one, "room 204" / "room 206" are
//              two); "obstacle:<sector>:<level>" for the cane's class-agnostic obstacles
//   sector     left | ahead | right of the box centre (+-20 deg, 5 deg hysteresis)
//   successor  a new track inherits a vanished track's memory (a tracker id switch or a dropout) only when it appears
//              within 5 s, where the old one was last seen (box centre within 0.3 of the frame), at about its range
//              (max(1 m, 25%): metric depth, else the box side the frame edge cuts in neither), never while both were in
//              view as separate boxes, and only if the old object had not moved away since it was said (by that same
//              margin). One that walked off and vanished, and another coming from there, are two objects: the second is
//              new to the user, so a memory of the first must not hold it. The same continuity (first seen where and soon
//              after the old one was last seen, overlapping it, at about its range) decides whether a new track joins a
//              background record or is the desk user; what its box does later does not matter, so a stranger who walks
//              up into the user's place is never the user.
//
// Safety principle: the gate only removes cues that carry no new safety information. It never makes a warning about an
// approaching, in-path, vehicle or FIND-goal object later than the bare scheduler would (allowance: one 0.5 s decision
// step, to confirm a track), and its silence never means "all clear".
//
// Two channels: every offered cue is judged, and one speech cue and one haptic cue may pass at the same decision, so a
// cane buzz never hides the scheduler's warning (a cue the gate does not say is refunded and kept for rescue).
//
// A cue passes when (in this order) it is
//   critical    measured TTC < 2 s in the walking path (passes unconfirmed; repeats once per object every 2 s unless worse)
//   approach    a background object or the desk user released because it measurably closes in: its next cue
//   new         no memory of this identity in this sector (setting a FIND goal drops the memory of what it names)
//   reacquired  a successor track after more than 0.8 s out of view (the tracker's 8 missed frames), in the walking path
//               or corridor and no farther than when last seen: it may have come closer unseen, or be another object
//   successor   a successor track that closes in (>= 0.5 m/s: metric depth, else its box grows) and is a vehicle or in
//               the path or corridor: maybe another object coming from where the first vanished
//               (both: within the successor's first 5 s, once, until it is said)
//   sector      the same track was last said in another sector
//   closer      nearer by at least max(1 m, 25%) than when last said
//   near        no measured depth: near, in the walking corridor, and its box grows (medians of 3 boxes, 0.5 s and 1 s back)
//   closing     in the path within 3 m, max(0.35 m, 15%) nearer than when last said (median over 0.3 s), still closing
//   reminder    the hold ran out: 20 s walking, 60 s still, 8 s for a moving object or a FIND goal
//   interrupt   the scheduler cuts in for it and it moves or approaches, at most once per object every 2 s
//   haptic      at most once per object and sector every 2 s
// and it is held as
//   unconfirmed fewer than 3 matches (or 2 spanning 0.15 s): kept by identity for up to 3 s and said once a track of that
//               identity is confirmed, even if the track flickered or switched id in between. Under heavy flicker (each
//               sighting a new id) the matches of its predecessors at the same place count, and a sighting within 0.8 s
//               stands in for a track out of view at the decision; a cue held over a decision is recomposed
//   warmup / background   camera still: an object seen in >= 3 samples over >= 1 s without closing is background until
//               it measurably closes in. Vehicles and bikes never, FIND goals never, an object in the path within 3 m is
//               said once first. A record's track clearly farther than its place (max(1 m, 25%)) is another object
//   operator    desk: when the camera is still at start-up, a centred person within 1.2 m (no depth: a box taller than 60%
//               of the frame) is the user, held until they measurably come closer (smoothed drop of max(0.35 m, 15%)).
//               Another track is the user only by continuity (an id switch; a duplicate box that overlapped the user's
//               from its first sighting) or once it has rested 3 s at the user's place and range (back from a break)
//   repeat      none of the above
// Rescue: when an object the gate held (or said) qualifies later and the scheduler does not offer it again, the gate says
// it itself with a freshly composed message; a held cue's seconds go back to the scheduler's speech budget. A cue held as
// background, the user or warm-up stays rescuable for 20 s, for an approach or a critical TTC only: a learned scheduler
// may offer an object once and never again.
//
// "Measured" TTC / speed: the track has metric depth (depthConf >= 0.4) with >= 4 depth samples spanning >= 1 s. Without
// depth the tracker assumes the user walks at 1 m/s, so a static person 3 m in front of a laptop gets "TTC 2.6 s".
import { compose, pyRound, speechSeconds, SLOTS } from "./compose";
import type { Candidate, PCACSConfig } from "./pcacs";
import { Camera, DEFAULT_SIZE, TYPICAL_SIZE, area, center, iou } from "./tracker";
import type { BBox, ChannelState, Cue, Track } from "./types";

export type Sector = "left" | "ahead" | "right";

export interface GateConfig {
  holdStatic: number;       // same object, same sector, nothing new: not repeated within this (20 s walking, 60 s still)
  holdMoving: number;       // ... for a moving object
  holdGoal: number;         // ... for a FIND goal
  closerM: number; closerFrac: number;            // `closer`: nearer by max(closerM, closerFrac x distance then)
  sectorDeg: number; sectorHystDeg: number;
  minHits: number; confirmAge: number;            // confirmed: minHits matches, or 2 matches spanning confirmAge
  pending: number;                                // a cue held for confirmation is kept this long
  hapticGap: number; interruptGap: number;        // per identity
  criticalTtc: number; criticalPath: number; criticalGap: number; criticalWorseTtc: number; criticalWorseM: number;
  minGap: number;                                 // no non-critical repeat (closer / sector / near / closing) sooner
  switchGap: number; switchDist: number;          // id switch: appeared within switchGap s, box centre within switchDist
  lostS: number;                                  // `reacquired`: out of view longer than this (the tracker's max age)
  movingSpeed: number; approachTtc: number; approachMin: number;
  warmup: number; staticDisp: number; staticSizeFrac: number; warmMinSamples: number; warmHoldSpan: number; bgMinSpan: number;
  background: boolean;      // camera still: static objects are background
  operatorFilter: boolean;  // desk webcam: the person in front of it is the user
  bgReleaseDisp: number; bgPathM: number; corridorHalf: number;
  operatorMaxM: number; operatorMinHeight: number; operatorNoDepthHeight: number; operatorCenterTol: number; operatorDisp: number; operatorDupIou: number;
  releaseM: number; releaseFrac: number; releaseSmooth: number; releaseFast: number; releaseMps: number; releaseSpan: number; releaseTtc: number;
  recapture: number; restSpreadM: number; restSizeFrac: number;
  nodepthNearM: number; nodepthApproach: number; nodepthGrow: number; nodepthBottomFrac: number;
  heldKeep: number; approachKeep: number;
  silenceKeep: number;      // a cue held as background / the user / warm-up is said when its object approaches within this
}

export const GATE_DEFAULTS: GateConfig = {
  holdStatic: 20, holdMoving: 8, holdGoal: 8, closerM: 1.0, closerFrac: 0.25, sectorDeg: 20, sectorHystDeg: 5,
  minHits: 3, confirmAge: 0.15, pending: 3, hapticGap: 2, interruptGap: 2,
  criticalTtc: 2.0, criticalPath: 0.5, criticalGap: 2.0, criticalWorseTtc: 0.5, criticalWorseM: 0.3,
  minGap: 2.5, switchGap: 5, switchDist: 0.3, lostS: 0.8, movingSpeed: 0.3, approachTtc: 6, approachMin: 0.5,
  warmup: 3, staticDisp: 0.03, staticSizeFrac: 0.2, warmMinSamples: 3, warmHoldSpan: 0.4, bgMinSpan: 1.0,
  background: false, operatorFilter: false, bgReleaseDisp: 0.15, bgPathM: 3.0, corridorHalf: 0.45,
  operatorMaxM: 1.2, operatorMinHeight: 0.4, operatorNoDepthHeight: 0.6, operatorCenterTol: 0.2, operatorDisp: 0.12, operatorDupIou: 0.6,
  releaseM: 0.35, releaseFrac: 0.15, releaseSmooth: 0.3, releaseFast: 0.6, releaseMps: 0.25, releaseSpan: 1.0, releaseTtc: 3.0,
  recapture: 3, restSpreadM: 0.15, restSizeFrac: 0.08,
  nodepthNearM: 2.0, nodepthApproach: 0.3, nodepthGrow: 0.10, nodepthBottomFrac: 0.15,
  heldKeep: 3, approachKeep: 5, silenceKeep: 20,
};

/** What the gate needs from the scheduler it wraps (PCACS and LearnedScheduler both have these). */
export interface GatedScheduler {
  budget: number; cfg: PCACSConfig; announcedDist: Map<number, number>; goalTags: string[]; egoSpeed: number; lastCandidates: Candidate[];
}

export interface GateVerdict {
  t: number; passed: boolean; reason: string; text: string; trackId: number; label: string; deferred: boolean; rescued: boolean;
}

// labels that are the same physical object to the user when a tracker id switch also flips the class
const LABEL_ALIAS: Record<string, string> = { truck: "car", bus: "car", van: "car", motorcycle: "bicycle", motorbike: "bicycle",
  scooter: "bicycle", bike: "bicycle", tram: "train", pedestrian: "person", people: "person" };
// identities (after LABEL_ALIAS) that are never background or held for the warm-up: they move under their own power
const VEHICLES = new Set(["car", "bicycle", "train"]);
const SILENCE = new Set(["warmup", "background", "operator"]);
// rescue: what the gate says itself, from a cue it held (the scheduler wanted it said) or one it let through earlier
const RESCUE_HELD: Record<string, number> = { critical: 5, approach: 4, successor: 4, near: 3, closing: 3, reacquired: 2, new: 1 };
const RESCUE_SAID: Record<string, number> = { approach: 4, successor: 4, near: 3 };
const RESCUE_SAID_APPROACH: Record<string, number> = { critical: 5, ...RESCUE_SAID };
// a cue held as background / the user / warm-up more than heldKeep ago (up to silenceKeep): only when its object approaches
const RESCUE_SILENCED: Record<string, number> = { critical: 5, approach: 4 };
const EDGE = 0.01;   // a box side within this of the frame edge is cut off (its size says nothing)
const HIST_S = 3.0;  // per-track samples the gate keeps (its own, frame-rate independent: the tracker keeps 15 boxes)

const isSpeech = (c: Cue) => c.action === "speak" || c.action === "interrupt";
const cutLR = (b: BBox) => b[0] <= EDGE || b[2] >= 1 - EDGE;
const cutTB = (b: BBox) => b[1] <= EDGE || b[3] >= 1 - EDGE;
const med = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const centreDist = (a: BBox, b: BBox) => { const [ax, ay] = center(a), [bx, by] = center(b); return Math.hypot(ax - bx, ay - by); };
const s1 = (x: number) => x.toFixed(1);
const secs = (x: number) => (x < 1 ? x.toFixed(1) : String(Math.round(x)));

// ---- signs and goals -----------------------------------------------------------------------------------------
export function normalizeText(s: string | null | undefined): string {
  return (s ?? "").toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "").replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
}
const OCR_TO_DIGIT: Record<string, string> = { o: "0", i: "1", l: "1", s: "5", z: "2", b: "8" };
const OCR_TO_LETTER: Record<string, string> = { "0": "o", "1": "i", "5": "s", "8": "b" };
/** (canonical text, its digits): a token at least half digits is a number ("2o4" -> "204"), digits in a word are letters. */
export function signKey(txt: string): [string, string] {
  const toks = txt.split(" ").filter(Boolean).map((tok) => {
    const nd = [...tok].filter((ch) => ch >= "0" && ch <= "9").length;
    if (nd && 2 * nd >= tok.length) return [...tok].map((ch) => OCR_TO_DIGIT[ch] ?? ch).join("");
    if (nd) return [...tok].map((ch) => OCR_TO_LETTER[ch] ?? ch).join("");
    return tok;
  });
  const canon = toks.join(" ");
  return [canon, [...canon].filter((ch) => ch >= "0" && ch <= "9").join("")];
}
function matching(a: string, b: string): number {
  // difflib.SequenceMatcher: longest common block (earliest in a, then b), recursively on both sides
  if (!a.length || !b.length) return 0;
  let best = 0, bi = 0, bj = 0, prev = new Array<number>(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array<number>(b.length + 1).fill(0);
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] !== b[j - 1]) continue;
      cur[j] = prev[j - 1] + 1;
      if (cur[j] > best) { best = cur[j]; bi = i - best; bj = j - best; }
    }
    prev = cur;
  }
  return best ? best + matching(a.slice(0, bi), b.slice(0, bj)) + matching(a.slice(bi + best), b.slice(bj + best)) : 0;
}
export const seqRatio = (a: string, b: string) => (a.length + b.length ? (2 * matching(a, b)) / (a.length + b.length) : 1);
/** Two normalised OCR readings are one sign only if their numbers are identical, their lengths similar and the rest differs by OCR noise. */
export function sameSign(a: string, b: string, ratio = 0.8): boolean {
  if (a === b) return true;
  const [ca, da] = signKey(a), [cb, db] = signKey(b);
  if (da !== db || Math.abs(ca.length - cb.length) > Math.max(1, pyRound(0.2 * Math.max(ca.length, cb.length)))) return false;
  return ca === cb || seqRatio(ca, cb) >= ratio;
}
/** The normalised goal ("room 206", "chair") names this label / sign text as a whole word or phrase ("seat": chair, bench). */
export function goalNames(goal: string, key: string): boolean {
  if (goal === "seat" && (key === "chair" || key === "bench")) return true;
  return !!goal && !!key && (goal === key || ` ${key} `.includes(` ${goal} `));
}
export function sectorOf(angle: number, deg = 20): Sector { return angle < -deg ? "left" : angle > deg ? "right" : "ahead"; }
export function inSector(angle: number, s: Sector, deg = 20, hyst = 0): boolean {
  if (s === "left") return angle < -deg + hyst;
  if (s === "right") return angle > deg - hyst;
  return angle >= -deg - hyst && angle <= deg + hyst;
}
export function materiallyCloser(then: number | null, now: number | null, m: number, frac: number): boolean {
  return then != null && now != null && then - now >= Math.max(m, frac * then);
}

// ---- state -------------------------------------------------------------------------------------------------------
interface Ident { kind: "object" | "text"; key: string; id: string }
interface Sample { t: number; bbox: BBox; z: number | null }   // z: metric depth (depthConf >= 0.4), else null
/** A track that took over a memory entry from a vanished one: when, how long the object was out of view, and the old
 *  track's last sample (`reacquired`: no farther than that). */
interface Adoption { t: number; gap: number; prev: Sample }
interface Said {
  ident: Ident; sector: Sector; angle: number; t: number; dist: number | null; trackId: number; bbox: BBox | null;
  z: number | null;       // metric depth when said (the box is `bbox`): "had it moved away since it was said?"
  lastBox: BBox | null;   // the box of its track the last time it was in view (id-switch position check)
  goal: boolean; seen: number; apart: Set<number>;
  critT: number | null; critTtc: number; critDist: number | null; cue: Cue | null; rng: number | null;
  adopt: Adoption | null; // the last id switch (or dropout) that handed this entry to another track
}
interface Warm {
  ident: Ident; kind: string; label: string; c0: [number, number]; h0: number; firstT: number; lastT: number; bbox: BBox;
  n: number; hits: number; maxDisp: number; maxSize: number; path: number; closing: boolean;
  dists: number[]; depths: number[]; widths: number[]; heights: number[]; areas: number[];
}
interface RestSample { t: number; d: number | null; w: number | null; h: number | null; a: number; c: [number, number] }
interface Rec {
  ident: Ident; kind: string; label: string; bbox0: BBox; bbox: BBox; distRef: number | null; wRef: number | null; hRef: number | null; aRef: number | null;
  trackIds: Set<number>; released: boolean; sayOnce: boolean; said: boolean; approach: boolean; rest: RestSample[];
  last: Sample | null;   // its object's last sighting (any of its tracks): where and when a successor must appear
}
interface Pending { cue: Cue; t: number; ident: Ident; sector: Sector; bbox: BBox | null; z: number | null }
interface Held { cue: Cue; t: number; reason: string; angle: number }
interface Check { ok: boolean; reason: string; entry: Said | null; info: string }

export class AnnouncementGate {
  cfg: GateConfig;
  /** the scheduler being wrapped (budget refunds, announced marks, goal features); main.ts swaps it with the scheduler */
  policy: GatedScheduler | null = null;
  stats = { passed: 0, held: 0, rescued: 0 };
  /** the verdict to show for the last decision: the first cue that passed, else the first one judged */
  last: GateVerdict | null = null;
  /** every verdict of the last decision, in the order the cues were judged */
  verdicts: GateVerdict[] = [];
  releases: Array<[number, number, string]> = [];
  private goalTags: string[] = [];
  private mem = new Map<string, Said[]>();
  private hapticT = new Map<string, number>();
  private critHaptic = new Map<string, [number, number, number | null]>();
  private pending = new Map<string, Pending>();
  private held = new Map<number, Held>();
  private approach = new Map<number, number>();
  private closingSince = new Map<number, number>();
  private hist = new Map<number, Sample[]>();
  private first = new Map<number, Sample>();   // each track's first sighting (adoption tests: where did it come from?)
  private seen = new Map<number, Track>();     // tracks seen in the last few seconds (confirmation across flicker)
  private rescuing = false;
  private t0: number | null = null;
  private warm = new Map<number, Warm>();
  private warmDone = false;
  private bg: Rec[] = [];
  private operator: Rec | null = null;

  constructor(public cam: Camera = new Camera(), cfg: Partial<GateConfig> = {}) { this.cfg = { ...GATE_DEFAULTS, ...cfg }; }

  reset() {
    this.mem.clear(); this.hapticT.clear(); this.critHaptic.clear(); this.pending.clear(); this.held.clear(); this.approach.clear();
    this.closingSince.clear(); this.hist.clear(); this.first.clear(); this.seen.clear(); this.releases = []; this.stats = { passed: 0, held: 0, rescued: 0 }; this.last = null; this.verdicts = [];
    this.rebaseline();
  }
  /** Forget the background / desk user and restart the warm-up (the camera became still somewhere new). Memory stays. */
  rebaseline() { this.t0 = null; this.warm.clear(); this.warmDone = false; this.bg = []; this.operator = null; }
  /** Change config fields at run time; returns the previous values. */
  setParams(p: Partial<GateConfig>): Partial<GateConfig> {
    const old: Record<string, unknown> = {};
    for (const k of Object.keys(p) as Array<keyof GateConfig>) old[k] = this.cfg[k];
    Object.assign(this.cfg, p);
    return old as Partial<GateConfig>;
  }
  /** A FIND goal: memory of the objects and signs it names is dropped, so the goal is said as soon as it is seen. */
  setGoal(tags: string[]) {
    const next = tags.map((g) => g.toLowerCase());
    if (next.length && next.join("|") !== this.goalTags.join("|")) {
      const goals = next.map(normalizeText).filter(Boolean);
      for (const [id, es] of [...this.mem]) if (goals.some((g) => goalNames(g, es[0]?.ident.key ?? ""))) this.mem.delete(id);
    }
    this.goalTags = next;
  }
  get hasUser() { return this.operator != null && !this.operator.released; }
  get backgroundCount() { return this.bg.filter((b) => !b.released).length; }

  // ---- per frame -----------------------------------------------------------------------------------------------
  /** Record one box / depth sample per visible track (call it every frame; `filter` calls it too) and run the warm-up. */
  observe(tracks: Track[], t: number) {
    for (const tr of tracks) {
      if (tr.misses > 0) continue;
      let h = this.hist.get(tr.id);
      if (!h) { h = []; this.hist.set(tr.id, h); }
      const ts = Math.min(t, tr.lastSeen);
      if (h.length && ts <= h[h.length - 1].t + 1e-9) continue;
      const smp: Sample = { t: ts, bbox: tr.bbox, z: tr.depthConf >= 0.4 && tr.distance != null ? tr.distance : null };
      h.push(smp);
      while (h.length > 2 && ts - h[0].t > HIST_S) h.shift();
      if (!this.first.has(tr.id)) this.first.set(tr.id, smp);
      this.seen.set(tr.id, tr);
      for (const rec of this.operator ? [this.operator, ...this.bg] : this.bg) if (rec.trackIds.has(tr.id)) rec.last = smp;
      if (!this.warmDone) this.warmSample(tr, ts);
    }
    for (const [id, h] of [...this.hist]) if (t - h[h.length - 1].t > 5) { this.hist.delete(id); this.first.delete(id); this.seen.delete(id); }
    if (!this.warmDone) {
      if (this.t0 == null && tracks.some((tr) => tr.misses === 0)) this.t0 = t;
      if (this.t0 != null && t >= this.t0 + this.cfg.warmup) this.finishWarmup();
    }
  }
  private warmSample(tr: Track, t: number) {
    if (this.t0 == null) this.t0 = t;
    let w = this.warm.get(tr.id);
    const b = tr.bbox, c = center(b), h = Math.max(1e-3, b[3] - b[1]);
    if (!w) {
      w = { ident: this.identityOf(tr), kind: tr.kind, label: tr.label, c0: c, h0: h, firstT: t, lastT: t, bbox: b, n: 0, hits: 0, maxDisp: 0, maxSize: 0,
        path: 0, closing: false, dists: [], depths: [], widths: [], heights: [], areas: [] };
      this.warm.set(tr.id, w);
    }
    w.n++; w.lastT = t; w.bbox = b; w.hits = Math.max(w.hits, tr.hits); w.path = tr.pathProb;
    w.maxDisp = Math.max(w.maxDisp, Math.hypot(c[0] - w.c0[0], c[1] - w.c0[1]));
    w.maxSize = Math.max(w.maxSize, Math.abs(h - w.h0) / w.h0);
    if (!cutLR(b)) w.widths.push(b[2] - b[0]);
    if (!cutTB(b)) w.heights.push(h);
    w.areas.push(area(b));
    if (tr.distance != null) { w.dists.push(tr.distance); if (tr.depthConf >= 0.4) w.depths.push(tr.distance); }
    if (this.measured(tr) && -tr.relSpeed >= this.cfg.releaseMps) w.closing = true;
  }

  // ---- the decision ------------------------------------------------------------------------------------------------
  /** The first cue that passes (see filterAll), or null. */
  filter(cues: Cue[], tracks: Track[], t: number, ch?: ChannelState): Cue | null {
    return this.filterAll(cues, tracks, t, ch)[0] ?? null;
  }
  /** What to act on at this decision: every offered cue is judged in order, and the first speech cue (speak / interrupt)
   *  and the first haptic cue that pass go through together (two channels). A cue whose channel an earlier one took is
   *  not said: refunded and kept for rescue like a held one. While the speech channel is free, a cue held for
   *  confirmation, then a rescue, may take it. `tracks`: what is in view now (the scheduler's tracks plus the cane's
   *  obstacle tracks). */
  filterAll(cues: Cue[], tracks: Track[], t: number, ch?: ChannelState): Cue[] {
    this.observe(tracks, t);
    this.prune(t);
    const visible = new Map<number, Track>();
    for (const tr of tracks) if (tr.misses === 0) visible.set(tr.id, tr);
    this.updateClosing(visible, t);
    this.trackRecords(visible, t);
    let idents: Map<number, string> | null = null;
    for (const es of this.mem.values()) for (const e of es) {
      const own = visible.get(e.trackId);
      if (!own) continue;
      e.seen = t; e.lastBox = own.bbox;
      // another track of the same identity in view now as a separate box: a second object, never an id switch later
      for (const o of visible.values()) {
        if (o === own || e.apart.has(o.id) || iou(o.bbox, own.bbox) >= 0.3) continue;
        idents ??= new Map([...visible].map(([id, tr]) => [id, this.identityOf(tr).id]));
        if (idents.get(o.id) === e.ident.id) e.apart.add(o.id);
      }
    }
    this.last = null; this.verdicts = [];
    const out: Cue[] = [];
    const taken = (c: Cue) => out.some((o) => isSpeech(o) === isSpeech(c));
    for (const cue of cues) {
      if (taken(cue)) { this.notSaid(cue, visible.get(cue.trackId) ?? null, t); continue; }
      const got = this.judge(cue, visible.get(cue.trackId) ?? null, visible, t);
      if (got) out.push(got);
    }
    if (!out.some(isSpeech)) {
      const more = this.flushPending(visible, t, ch, taken) ?? this.rescue(visible, t, ch);
      if (more) out.push(more);
    }
    this.last = this.verdicts.find((v) => v.passed) ?? this.verdicts[0] ?? null;
    return out;
  }
  /** Another cue took this one's channel at this decision: not said, so refunded and kept for rescue like a held cue. */
  private notSaid(cue: Cue, trk: Track | null, t: number) {
    this.stats.held++;
    this.verdicts.push({ t, passed: false, reason: "channel", text: `held: ${cue.label}, another cue took the ${isSpeech(cue) ? "speech" : "haptic"} channel`,
      trackId: cue.trackId, label: cue.label, deferred: false, rescued: false });
    this.refund(cue);
    if (isSpeech(cue) && cue.source !== "cane") this.held.set(cue.trackId, { cue, t, reason: "channel", angle: trk ? this.angleOf(trk) : 0 });
  }

  private judge(cue: Cue, trk: Track | null, visible: Map<number, Track>, t: number, deferred = false): Cue | null {
    const c = this.check(cue, trk, visible, t);
    this.verdicts.push({ t, passed: c.ok, reason: c.reason, text: c.ok ? `${this.rescuing ? "said by the gate, " : ""}${c.info}` : `held: ${c.info}`,
      trackId: cue.trackId, label: cue.label, deferred, rescued: this.rescuing });
    const ident = this.identityOf(trk, cue);
    if (c.ok) {
      this.stats.passed++;
      const prev = this.pending.get(ident.id);
      // a cue held for confirmation on this channel is never said now (a buzz does not replace words, nor words a buzz)
      if (prev && prev.cue !== cue && isSpeech(prev.cue) === isSpeech(cue)) { this.pending.delete(ident.id); this.refund(prev.cue); }
      if (isSpeech(cue)) this.held.delete(cue.trackId);
      if (cue.source === "cane") this.charge(cue, null, t);
      this.commit(cue, trk, c.entry, t, c.reason);
      return cue;
    }
    this.stats.held++;
    if (c.reason === "unconfirmed") {
      const prev = this.pending.get(ident.id);
      if (prev && prev.cue !== cue) this.refund(prev.cue);     // superseded by the newer cue for the same object
      const angle = trk ? this.angleOf(trk) : 0;
      this.pending.set(ident.id, { cue, t: prev?.t ?? t, ident, sector: sectorOf(angle, this.cfg.sectorDeg), bbox: trk?.bbox ?? null,
        z: trk && trk.depthConf >= 0.4 ? trk.distance : null });
    } else if (!this.rescuing) {                                // (a rescued cue was refunded when it was first held)
      this.refund(cue);
      if (isSpeech(cue) && cue.source !== "cane") this.held.set(cue.trackId, { cue, t, reason: c.reason, angle: trk ? this.angleOf(trk) : 0 });
    }
    return null;
  }

  /** Held for confirmation: said (if the scheduler has nothing better) once a track of the same identity in the same
   *  place is confirmed, within cfg.pending; a one-frame flicker or an id switch in between does not lose it. */
  private flushPending(visible: Map<number, Track>, t: number, ch?: ChannelState, taken: (c: Cue) => boolean = () => false): Cue | null {
    const busy = !!ch?.busy;
    for (const [key, p] of [...this.pending].sort((a, b) => b[1].cue.utility - a[1].cue.utility)) {
      if (t - p.t > this.cfg.pending + 1e-9) { this.pending.delete(key); this.refund(p.cue); continue; }
      if (taken(p.cue)) continue;                             // its channel is used at this decision: keep waiting
      const trk = this.pendingTrack(p, visible, t);
      if (!trk || !this.confirmed(trk)) continue;           // not seen at its place lately (flicker): keep waiting
      let action = p.cue.action;
      if (action === "interrupt" && !busy) action = "speak";
      else if (action === "speak" && busy) continue;         // wait for the channel
      this.pending.delete(key);
      // held for more than a decision: said as the object is now, not as it was
      const base = isSpeech(p.cue) && t - p.cue.t > 0.75 ? this.refreshed(p.cue, trk, t) : p.cue;
      const out = this.judge({ ...base, t, action, trackId: trk.id }, trk, visible, t, true);
      if (out) return out;
    }
    return null;
  }
  /** The track a held-for-confirmation cue is about now: its own, else one of its identity in the same place and at
   *  about the same range (an id switch); else, under flicker, the latest such track seen within lostS (out of view at
   *  this decision only). */
  private pendingTrack(p: Pending, visible: Map<number, Track>, t: number): Track | null {
    const g = this.cfg;
    const own = visible.get(p.cue.trackId);
    if (own) return own;
    const here = (tr: Track) => {
      if (this.identityOf(tr, p.cue.identity ? p.cue : undefined).id !== p.ident.id || !inSector(this.angleOf(tr), p.sector, g.sectorDeg, g.sectorHystDeg)) return false;
      if (!p.bbox) return true;
      if (centreDist(p.bbox, tr.bbox) > g.switchDist) return false;
      const r = this.rangePair({ t: p.t, bbox: p.bbox, z: p.z }, { t: tr.lastSeen, bbox: tr.bbox, z: tr.depthConf >= 0.4 ? tr.distance : null }, tr.label);
      return !r || Math.abs(r[1] - r[0]) <= Math.max(g.closerM, g.closerFrac * r[0]);
    };
    let best: Track | null = null, bestD = Infinity;
    for (const tr of visible.values()) {
      const d = p.bbox ? centreDist(p.bbox, tr.bbox) : 0;
      if (here(tr) && d < bestD) { best = tr; bestD = d; }
    }
    if (best) return best;
    for (const tr of this.seen.values()) {
      if (visible.has(tr.id) || t - tr.lastSeen > g.lostS + 1e-9 || tr.lastSeen < p.t - g.pending) continue;
      if (here(tr) && (!best || tr.lastSeen > best.lastSeen)) best = tr;
    }
    return best;
  }

  /** Say a cue the gate held (or one said earlier) itself when its object now qualifies for safety and the scheduler did
   *  not offer it again. At most one, the most urgent. */
  private rescue(visible: Map<number, Track>, t: number, ch?: ChannelState): Cue | null {
    const g = this.cfg;
    let best: { key: [number, number]; cue: Cue; trk: Track; reason: string } | null = null;
    for (const [tid, trk] of visible) {
      if (!this.confirmed(trk)) continue;
      const h = this.held.get(tid);
      let src: Cue, heldFor: string | null, late = false;
      if (h && t - h.t > 1e-9 && (t - h.t <= g.heldKeep + 1e-9 || (SILENCE.has(h.reason) && t - h.t <= g.silenceKeep + 1e-9))) {
        src = h.cue; heldFor = h.reason; late = t - h.t > g.heldKeep + 1e-9;
      } else {
        const e = this.ownEntry(tid);
        if (!e || !e.cue || e.cue.source === "cane") continue;
        src = e.cue; heldFor = null;
      }
      const cue = this.refreshed(src, trk, t);
      const c = this.check(cue, trk, visible, t);   // (may release it as approaching)
      const allowed = heldFor != null ? (late ? RESCUE_SILENCED : RESCUE_HELD) : this.approach.has(tid) ? RESCUE_SAID_APPROACH : RESCUE_SAID;
      // "new" only for a cue held as background / the user / warm-up, or one another cue's channel crowded out
      if (!c.ok || !(c.reason in allowed) || (c.reason === "new" && !(heldFor != null && (SILENCE.has(heldFor) || heldFor === "channel")))) continue;
      // said by the gate unasked only when the user walks straight into it
      if (c.reason === "near" && heldFor == null && !this.inCorridor(trk.label, trk.bbox, this.rangeNow(trk) ?? 3, true)) continue;
      const key: [number, number] = [allowed[c.reason], cue.utility || 0];
      if (!best || key[0] > best.key[0] || (key[0] === best.key[0] && key[1] > best.key[1])) best = { key, cue, trk, reason: c.reason };
    }
    if (!best) return null;
    let cue = best.cue;
    if (ch?.busy) {
      if (!["critical", "approach", "near"].includes(best.reason)) return null;
      cue = { ...cue, action: "interrupt" };
    } else cue = { ...cue, action: "speak" };
    this.rescuing = true;
    let out: Cue | null;
    try { out = this.judge(cue, best.trk, visible, t, true); } finally { this.rescuing = false; }
    if (out) { this.stats.rescued++; this.charge(out, best.trk, t); }
    return out;
  }

  private check(cue: Cue, trk: Track | null, visible: Map<number, Track>, t: number): Check {
    const g = this.cfg;
    const ident = this.identityOf(trk, cue);
    const angle = trk ? this.angleOf(trk) : 0;
    const sec = sectorOf(angle, g.sectorDeg);
    const name = ident.kind === "text" ? `sign "${ident.key}"` : ident.key.replace(/:/g, " ");
    const desc = `${name} ${sec}`;
    // background / desk-user bookkeeping first (it follows the user's box and releases approaching objects), so a track
    // held as the user or as background is critical only when it measurably closes in
    let silenced = trk ? this.backgroundReason(trk, t, visible) : null;
    if ((silenced === "warmup" || silenced === "background") && this.isGoal(cue, trk)) silenced = null;   // a FIND goal never is
    if (trk && this.critical(trk) && (silenced == null || -trk.relSpeed >= g.approachMin)) {
      const e = this.lookup(ident, angle, cue.trackId, trk, visible, t);
      const last = cue.action === "haptic" ? this.critHaptic.get(`${ident.id}|${sec}`) ?? null
        : e && e.critT != null ? [e.critT, e.critTtc, e.critDist] as [number, number, number | null] : null;
      const why = this.criticalGap(last, trk, t);
      if (why) return { ok: false, reason: "critical_gap", entry: e, info: `critical ${desc} ${why}` };
      return { ok: true, reason: "critical", entry: e, info: `critical: ${desc}, TTC ${s1(trk.ttc)} s` };
    }
    if (trk && !this.confirmed(trk)) return { ok: false, reason: "unconfirmed", entry: null, info: `${desc} not confirmed yet (${trk.hits} matches)` };
    if (silenced === "operator") return { ok: false, reason: silenced, entry: null, info: `${name} is you (desk camera)` };
    if (silenced === "background") return { ok: false, reason: silenced, entry: null, info: `${desc} is background (camera still)` };
    if (silenced === "warmup") return { ok: false, reason: silenced, entry: null, info: `${desc} static while the camera settles` };
    if (cue.action === "haptic") {
      const last = this.hapticT.get(`${ident.id}|${sec}`);
      if (last != null && t - last < g.hapticGap) return { ok: false, reason: "haptic_gap", entry: null, info: `buzzed for ${desc} ${secs(t - last)} s ago` };
      return { ok: true, reason: "haptic", entry: null, info: `buzz: ${desc}` };
    }
    const e = this.lookup(ident, angle, cue.trackId, trk, visible, t);
    const released = this.approach.get(cue.trackId);
    if (released != null && t - released <= g.approachKeep) return { ok: true, reason: "approach", entry: e, info: `${desc} is coming closer` };
    if (!e) return { ok: true, reason: "new", entry: null, info: `${this.isGoal(cue, trk) ? "FIND goal" : "new"}: ${desc}` };
    // a successor track (id switch or dropout) in its first seconds, not said since it took over the memory
    if (trk && e.adopt && e.t < e.adopt.t && e.trackId === trk.id && t - trk.firstSeen <= g.approachKeep + 1e-9) {
      const back = this.reacquired(trk, e.adopt);
      if (back) return { ok: true, reason: "reacquired", entry: e, info: `${desc} back after ${s1(e.adopt.gap)} s out of view, ${back}` };
      if (this.successorCloses(trk)) return { ok: true, reason: "successor", entry: e, info: `${desc} coming closer on a new track (maybe another ${name})` };
    }
    const ago = t - e.t;
    const said = `said ${secs(ago)} s ago`;
    if (e.trackId === cue.trackId && !inSector(angle, e.sector, g.sectorDeg, g.sectorHystDeg)) {
      return ago >= g.minGap ? { ok: true, reason: "sector", entry: e, info: `${name} moved ${sec} (was ${e.sector})` }
        : { ok: false, reason: "repeat", entry: e, info: `same ${desc}, ${said}` };
    }
    const dNow = trk ? trk.distance : cue.distance;
    if (materiallyCloser(e.dist, dNow, g.closerM, g.closerFrac) && ago >= g.minGap) {
      return { ok: true, reason: "closer", entry: e, info: `closer: ${desc}, ${s1(e.dist!)} -> ${s1(dNow!)} m` };
    }
    if (trk && ago >= g.minGap && this.nearWithoutDepth(trk)) {
      return { ok: true, reason: "near", entry: e, info: `near and closing: ${desc}, about ${s1(this.rangeNow(trk) ?? 0)} m (no metric depth)` };
    }
    if (trk && ago >= g.minGap && this.closingInPath(trk, e)) {
      return { ok: true, reason: "closing", entry: e, info: `closing in: ${desc}, ${s1(this.rangeNow(trk) ?? 0)} m (said at ${s1(e.rng ?? 0)} m)` };
    }
    const goal = this.isGoal(cue, trk), moving = !!trk && this.moving(trk);
    const hold = goal ? g.holdGoal : moving ? g.holdMoving : g.holdStatic;
    if (ago >= hold) return { ok: true, reason: "reminder", entry: e, info: `reminder: ${desc}, ${said} (hold ${secs(hold)} s)` };
    if (cue.action === "interrupt" && ago >= g.interruptGap && trk && (moving || this.approaching(trk, e.dist))) {
      return { ok: true, reason: "interrupt", entry: e, info: `interrupt: ${desc} ${moving ? "moving" : "approaching"}` };
    }
    return { ok: false, reason: "repeat", entry: e, info: `same ${desc}, ${said}` };
  }

  /** minHits matches, or 2 spanning confirmAge. Under heavy flicker each sighting may be a new track id that never gets
   *  there by itself, so the tracker's matches of its predecessors count too: tracks of the same identity at the same
   *  place (switchDist) that vanished before it appeared, within `pending`. */
  private confirmed(tr: Track) {
    const g = this.cfg;
    const enough = (hits: number, span: number) => hits >= g.minHits || (hits >= 2 && span >= g.confirmAge - 1e-9);
    if (enough(tr.hits, tr.lastSeen - tr.firstSeen)) return true;
    let hits = tr.hits, t0 = tr.firstSeen, id: string | null = null;
    for (const o of this.seen.values()) {
      if (o === tr || o.lastSeen >= tr.firstSeen - 1e-9 || tr.firstSeen - o.lastSeen > g.pending || centreDist(o.bbox, tr.bbox) > g.switchDist) continue;
      id ??= this.identityOf(tr).id;
      if (this.identityOf(o).id !== id) continue;
      hits += o.hits; t0 = Math.min(t0, o.firstSeen);
    }
    return enough(hits, tr.lastSeen - t0);
  }
  private critical(tr: Track) {
    return this.measured(tr) && Number.isFinite(tr.ttc) && tr.ttc < this.cfg.criticalTtc && tr.pathProb >= this.cfg.criticalPath;
  }
  private criticalGap(last: [number, number, number | null] | null, tr: Track, t: number): string {
    const g = this.cfg;
    if (!last || t - last[0] >= g.criticalGap) return "";
    const [t0, ttc0, d0] = last;
    if (Number.isFinite(ttc0) && tr.ttc <= ttc0 - g.criticalWorseTtc) return "";
    if (d0 != null && tr.distance != null && tr.distance <= d0 - g.criticalWorseM) return "";
    return `said ${secs(t - t0)} s ago and not worse`;
  }

  // ---- range and closing evidence (the gate's own samples) -----------------------------------------------------------
  private samples(tr: Track): Sample[] { return this.hist.get(tr.id) ?? [{ t: tr.lastSeen, bbox: tr.bbox, z: tr.depthConf >= 0.4 ? tr.distance : null }]; }
  /** Metric depth with >= 4 samples spanning >= 1 s (not the assumed walking speed, not the first noisy samples). */
  measured(tr: Track, span = 1.0): boolean {
    if (tr.depthConf < 0.4) return false;
    const zs = this.samples(tr).filter((s) => s.z != null);
    return zs.length >= 4 && zs[zs.length - 1].t - zs[0].t >= span - 1e-9;
  }
  private boxRanges(b: BBox, label: string): [number | null, number | null] {
    const [H, W] = TYPICAL_SIZE[label] ?? DEFAULT_SIZE;
    const w = b[2] - b[0], h = b[3] - b[1];
    return [h > 1e-3 && !cutTB(b) ? (this.cam.fy * H) / h : null, w > 1e-3 && !cutLR(b) ? (this.cam.fx * W) / w : null];
  }
  /** Two sightings' ranges on one scale: metric depth when both have it, else the box side the frame edge cuts in
   *  neither (box size ~ 1 / range, at `label`'s typical size); null when neither tells. */
  private rangePair(a: Sample, b: Sample, label: string): [number, number] | null {
    if (a.z != null && b.z != null) return [a.z, b.z];
    const [H, W] = TYPICAL_SIZE[label] ?? DEFAULT_SIZE;
    const w = (x: BBox) => Math.max(1e-3, x[2] - x[0]), h = (x: BBox) => Math.max(1e-3, x[3] - x[1]);
    if (!cutLR(a.bbox) && !cutLR(b.bbox)) return [(this.cam.fx * W) / w(a.bbox), (this.cam.fx * W) / w(b.bbox)];
    if (!cutTB(a.bbox) && !cutTB(b.bbox)) return [(this.cam.fy * H) / h(a.bbox), (this.cam.fy * H) / h(b.bbox)];
    return null;
  }
  /** Metric depth when there is any, else the box (median of the last 3): its width when the frame edge does not cut it
   *  (a floor obstacle a step away is cut off at the bottom), else its height, else the tracker's estimate. */
  rangeNow(tr: Track): number | null {
    if (tr.depthConf >= 0.4 && tr.distance != null) return tr.distance;
    const rs = this.samples(tr).slice(-3).map((s) => this.boxRanges(s.bbox, tr.label));
    const ws = rs.map((r) => r[1]), hs = rs.map((r) => r[0]);
    if (ws.every((x) => x != null)) return med(ws as number[]);
    if (hs.every((x) => x != null)) return med(hs as number[]);
    return tr.distance;
  }
  private inCorridor(label: string, b: BBox, dist: number, strict = false): boolean {
    const [, width] = TYPICAL_SIZE[label] ?? DEFAULT_SIZE;
    return Math.abs(this.cam.lateralFromU(center(b)[0], dist)) <= this.cfg.corridorHalf + (strict ? 0 : width / 2);
  }
  /** Closing speed (m/s) shown by the box growing (sides the frame edge does not cut; box size ~ 1 / range): the median of
   *  the last 3 boxes against the median of 3 boxes 0.5 s and 1 s back (whichever exist). Every lag must show at least
   *  nodepthGrow growth, so box jitter does not read as an approach; 0 otherwise. */
  boxClosing(tr: Track, rng: number): number {
    const h = this.samples(tr);
    if (h.length < 4) return 0;
    const end = h.slice(-3), t1 = end[end.length - 1].t;
    let speed = Infinity, lags = 0;
    for (const lag of [0.5, 1.0]) {
      let k = -1;
      for (let i = 0; i < h.length - 3; i++) if (h[i].t <= t1 - lag + 0.05) k = i;
      if (k < 0) continue;
      const start = h.slice(Math.max(0, k - 1), k + 2).filter((s) => !end.includes(s));
      const dt = t1 - h[k].t;
      if (dt < 0.3 || !start.length) continue;
      const logs: number[] = [];
      const dims = (ss: Sample[], f: (b: BBox) => number) => med(ss.map((s) => f(s.bbox)));
      if (![...start, ...end].some((s) => cutLR(s.bbox))) logs.push(Math.log(dims(end, (b) => b[2] - b[0]) / Math.max(1e-4, dims(start, (b) => b[2] - b[0]))));
      if (![...start, ...end].some((s) => cutTB(s.bbox))) logs.push(Math.log(dims(end, (b) => b[3] - b[1]) / Math.max(1e-4, dims(start, (b) => b[3] - b[1]))));
      if (!logs.length) continue;
      const grow = Math.exp(logs.reduce((a, b) => a + b, 0) / logs.length);
      lags++;
      if (grow < 1 + this.cfg.nodepthGrow) return 0;
      speed = Math.min(speed, (rng * (grow - 1)) / dt);
    }
    return lags ? speed : 0;
  }
  /** No metric depth (nothing is critical): near, in the walking path or corridor, and its box grows. */
  private nearWithoutDepth(tr: Track): boolean {
    const g = this.cfg;
    if (this.measured(tr)) return false;
    const rng = this.rangeNow(tr);
    if (rng == null) return false;
    if (!(tr.pathProb >= g.criticalPath || this.inCorridor(tr.label, tr.bbox, rng))) return false;
    if (!(rng < g.nodepthNearM || tr.bbox[3] >= 1 - g.nodepthBottomFrac)) return false;
    return this.boxClosing(tr, rng) >= g.nodepthApproach;
  }
  private smoothedDepth(tr: Track): number | null {
    const zs = this.samples(tr).filter((s) => s.z != null);
    if (!zs.length) return tr.distance;
    const t1 = zs[zs.length - 1].t;
    return med(zs.filter((s) => s.t >= t1 - this.cfg.releaseSmooth - 1e-9).map((s) => s.z as number));
  }
  /** Closing speed over the last ~0.5 s: metric depth, else the box growth. */
  private shortClosing(tr: Track): number {
    if (tr.depthConf >= 0.4) {
      const zs = this.samples(tr).filter((s) => s.z != null);
      if (zs.length < 2) return 0;
      const last = zs[zs.length - 1], prev = zs.filter((s) => s.t <= last.t - 0.45);
      if (!prev.length) return 0;
      const p = prev[prev.length - 1];
      return ((p.z as number) - (last.z as number)) / Math.max(1e-3, last.t - p.t);
    }
    const rng = this.rangeNow(tr);
    return rng != null ? this.boxClosing(tr, rng) : 0;
  }
  /** In the walking path or corridor within bgPathM, max(releaseM, releaseFrac x range) nearer than when last said
   *  (median over releaseSmooth) and still closing. */
  private closingInPath(tr: Track, e: Said): boolean {
    const g = this.cfg;
    if (e.rng == null || tr.distance == null) return false;
    const rng = this.rangeNow(tr);
    if (rng == null || rng >= g.bgPathM) return false;
    if (!(tr.pathProb >= g.criticalPath || this.inCorridor(tr.label, tr.bbox, rng))) return false;
    const m = this.measured(tr);
    const zs = this.samples(tr).filter((s) => s.z != null);
    const d = m ? this.smoothedDepth(tr) ?? rng : rng, dLast = m && zs.length ? (zs[zs.length - 1].z as number) : rng;
    const thr = Math.max(g.releaseM, g.releaseFrac * e.rng);
    const fast = this.shortClosing(tr);
    if (e.rng - d >= thr - 1e-9 && (fast >= g.releaseMps || (m && -tr.relSpeed >= g.releaseMps))) return true;
    return m && e.rng - dLast >= thr - 1e-9 && fast >= g.releaseFast;
  }
  /** `reacquired`: the successor came back after more than lostS out of view, in the walking path or corridor, no
   *  farther than its predecessor was last seen (max(0.1 m, 5%) for box jitter). What to show, or "". */
  private reacquired(tr: Track, a: Adoption): string {
    const g = this.cfg;
    if (a.gap <= g.lostS + 1e-9) return "";
    const h = this.samples(tr);
    const pair = this.rangePair(a.prev, h[h.length - 1], tr.label);
    if (!pair || pair[1] > pair[0] + Math.max(0.1, 0.05 * pair[0])) return "";
    if (!(tr.pathProb >= g.criticalPath || this.inCorridor(tr.label, tr.bbox, this.rangeNow(tr) ?? pair[1]))) return "";
    return `${s1(pair[0])} -> ${s1(pair[1])} m`;
  }
  /** `successor`: the successor closes in (>= approachMin over ~0.5 s: metric depth, else its box grows) and is a vehicle
   *  or in the walking path or corridor. */
  private successorCloses(tr: Track): boolean {
    const g = this.cfg, rng = this.rangeNow(tr);
    if (rng == null) return false;
    if (!(this.isVehicle(this.identityOf(tr)) || tr.pathProb >= g.criticalPath || this.inCorridor(tr.label, tr.bbox, rng))) return false;
    return this.shortClosing(tr) >= g.approachMin;
  }
  private moving(tr: Track): boolean {
    const g = this.cfg;
    if (this.samples(tr).length < 4) return false;
    const z = tr.distance ?? 3;
    if (Math.abs(tr.lateralSpeed) > g.movingSpeed + 0.05 * z) return true;
    return this.measured(tr) && Math.abs(tr.relSpeed) > g.movingSpeed;
  }
  private approaching(tr: Track, dRef: number | null): boolean {
    const g = this.cfg;
    if (materiallyCloser(dRef, tr.distance, g.closerM, g.closerFrac)) return true;
    return this.measured(tr) && Number.isFinite(tr.ttc) && tr.ttc < g.approachTtc && -tr.relSpeed >= g.approachMin;
  }
  private isGoal(cue: Cue, tr: Track | null): boolean {
    const f = this.policy?.lastCandidates.find((c) => c.track.id === cue.trackId);
    if (f && f.goal > 0) return true;
    const tags = this.goalTags.map(normalizeText).filter(Boolean);
    if (!tags.length) return false;
    const label = (tr?.label ?? cue.label ?? "").toLowerCase();
    const keys = [label, LABEL_ALIAS[label] ?? label];
    const txt = normalizeText(tr?.text);
    if (txt) keys.push(txt);
    return tags.some((g) => keys.some((k) => goalNames(g, k)));
  }

  // ---- identity memory -----------------------------------------------------------------------------------------------
  private angleOf(tr: Track) { return this.cam.angleDeg(center(tr.bbox)[0]); }
  identityOf(tr: Track | null, cue?: Cue): Ident {
    if (cue?.identity) return { kind: "object", key: cue.identity, id: `object|${cue.identity}` };
    const kind = tr?.kind ?? "object";
    if (kind === "text") {
      const txt = normalizeText(tr?.text);
      if (txt && !this.mem.has(`text|${txt}`)) {
        for (const es of this.mem.values()) {   // OCR variants of the same sign ("ROOM 204" / "R0OM 2O4")
          const id = es[0]?.ident;
          if (id && id.kind === "text" && sameSign(id.key, txt)) return id;
        }
      }
      return { kind: "text", key: txt, id: `text|${txt}` };
    }
    const label = (tr?.label ?? cue?.label ?? "object").toLowerCase();
    const key = LABEL_ALIAS[label] ?? label;
    return { kind: "object", key, id: `object|${key}` };
  }
  /** The memory entry this cue is about: its own track's, else (id switch) the most recent entry of the same identity in
   *  the same sector whose track vanished shortly before this one appeared and that this track can continue
   *  (successorOf). The entry adopts the new track id, so a chain of switches stays one object. */
  private lookup(ident: Ident, angle: number, trackId: number, tr: Track | null, visible: Map<number, Track>, t: number): Said | null {
    const g = this.cfg, es = this.mem.get(ident.id);
    if (!es?.length) return null;
    const own = es.filter((e) => e.trackId === trackId);
    if (own.length) return own.reduce((a, b) => (b.t > a.t ? b : a));
    let best: Said | null = null, bestSwitch: Omit<Adoption, "t"> | null = null;
    for (const e of es) {
      if (!inSector(angle, e.sector, g.sectorDeg, g.sectorHystDeg)) continue;
      const other = visible.get(e.trackId);
      // the remembered object is still in view as a separate track: a second object (unless a duplicate box)
      if (other && tr && other !== tr && iou(other.bbox, tr.bbox) < 0.3) continue;
      const sw = !other && tr ? this.successorOf(e, tr) : null;
      if (!other && tr && !sw) continue;
      if (!best || e.t > best.t) { best = e; bestSwitch = sw; }
    }
    if (!best) return null;
    if (!visible.has(best.trackId)) {
      best.trackId = trackId;
      if (bestSwitch) best.adopt = { t, ...bestSwitch };
    }
    return best;
  }
  /** Can track `tr` continue memory entry `e`, whose track is out of view (a tracker id switch, or a dropout)? It must
   *  appear within switchGap of the old track's last sighting, near where that was (switchDist), never in view together
   *  with it as a separate box, at about its range (max(closerM, closerFrac)), and the old object must not have moved
   *  away from where it was said by that margin: one that walked off and vanished, and another coming from there, are
   *  two objects, and the second is news. How long it was out of view and the old track's last sample, or null. */
  private successorOf(e: Said, tr: Track): Omit<Adoption, "t"> | null {
    const g = this.cfg;
    if (tr.firstSeen - e.seen > g.switchGap) return null;     // appeared long after: a different object
    if (e.apart.has(tr.id)) return null;                      // was in view together with it: a second object
    const ph = this.hist.get(e.trackId), sh = this.hist.get(tr.id);
    const ref = e.lastBox ?? e.bbox ?? tr.bbox;
    const prev: Sample = ph?.length ? ph[ph.length - 1] : { t: e.seen, bbox: ref, z: null };
    const first: Sample = sh?.length ? sh[0] : { t: tr.firstSeen, bbox: tr.bbox, z: tr.depthConf >= 0.4 ? tr.distance : null };
    if (first.t - prev.t > g.switchGap + 1e-9) return null;
    if (centreDist(prev.bbox, first.bbox) > g.switchDist) return null;               // not where the old one was
    if (prev.t >= first.t - 1e-9 && iou(prev.bbox, first.bbox) < 0.3) return null;   // both in one frame: two objects
    const near = (r: [number, number] | null) => !r || Math.abs(r[1] - r[0]) <= Math.max(g.closerM, g.closerFrac * r[0]);
    if (!near(this.rangePair(prev, first, tr.label))) return null;                     // not at the range it had
    const away = e.bbox ? this.rangePair({ t: e.t, bbox: e.bbox, z: e.z }, prev, tr.label) : null;
    if (away && away[1] - away[0] >= Math.max(g.closerM, g.closerFrac * away[0])) return null;   // it had moved away
    return { gap: Math.max(0, first.t - prev.t), prev };
  }
  private ownEntry(trackId: number): Said | null {
    let best: Said | null = null;
    for (const es of this.mem.values()) for (const e of es) if (e.trackId === trackId && (!best || e.t > best.t)) best = e;
    return best;
  }
  private commit(cue: Cue, tr: Track | null, e: Said | null, t: number, reason: string) {
    const g = this.cfg, ident = this.identityOf(tr, cue);
    const angle = tr ? this.angleOf(tr) : 0, sec = sectorOf(angle, g.sectorDeg);
    if (cue.action === "haptic") {
      this.hapticT.set(`${ident.id}|${sec}`, t);
      if (reason === "critical" && tr) this.critHaptic.set(`${ident.id}|${sec}`, [t, tr.ttc, tr.distance]);
      return;
    }
    if (!isSpeech(cue)) return;
    this.approach.delete(cue.trackId);           // the approach was said
    for (const b of this.bg) if (b.trackIds.has(cue.trackId)) b.said = true;   // in the path and near: said once
    const dist = tr?.distance ?? cue.distance, goal = this.isGoal(cue, tr);
    const z = tr && tr.depthConf >= 0.4 ? tr.distance : null;
    if (!e) {
      e = { ident, sector: sec, angle, t, dist, trackId: cue.trackId, bbox: tr?.bbox ?? null, z, lastBox: tr?.bbox ?? null, goal, seen: t,
        apart: new Set(), critT: null, critTtc: Infinity, critDist: null, cue: null, rng: null, adopt: null };
      const es = this.mem.get(ident.id) ?? [];
      es.push(e); this.mem.set(ident.id, es);
    } else {
      if (!inSector(angle, e.sector, g.sectorDeg, g.sectorHystDeg)) e.sector = sec;
      Object.assign(e, { angle, t, dist, trackId: cue.trackId, goal, seen: t });
      if (tr) { e.bbox = tr.bbox; e.lastBox = tr.bbox; e.z = z; }
    }
    e.cue = cue;
    e.rng = tr ? this.rangeNow(tr) : dist;
    if (reason === "critical" && tr) { e.critT = t; e.critTtc = tr.ttc; e.critDist = tr.distance; }
  }
  private prune(t: number) {
    const g = this.cfg, keep = 2 * Math.max(g.holdStatic, g.holdMoving, g.holdGoal);   // older entries pass anyway
    for (const [id, es] of [...this.mem]) {
      const kept = es.filter((e) => t - e.t <= keep).sort((a, b) => a.t - b.t).slice(-8);
      if (kept.length) this.mem.set(id, kept); else this.mem.delete(id);
    }
    for (const [k, v] of [...this.hapticT]) if (t - v > g.hapticGap) this.hapticT.delete(k);
    for (const [k, v] of [...this.critHaptic]) if (t - v[0] > g.criticalGap) this.critHaptic.delete(k);
    for (const [k, v] of [...this.held]) if (t - v.t > (SILENCE.has(v.reason) ? g.silenceKeep : g.heldKeep)) this.held.delete(k);
    for (const [k, v] of [...this.approach]) if (t - v > g.approachKeep) this.approach.delete(k);
  }

  // ---- background and the desk user ------------------------------------------------------------------------------
  private inPathNear(label: string, b: BBox, dist: number | null, pathProb: number): boolean {
    if (dist == null || dist >= this.cfg.bgPathM) return false;
    return pathProb >= this.cfg.criticalPath || this.inCorridor(label, b, dist);
  }
  private isVehicle(ident: Ident) { return ident.kind === "object" && VEHICLES.has(ident.key); }
  private warmClosing(w: Warm): boolean {
    const g = this.cfg;
    if (w.closing) return true;
    if (w.depths.length >= 3) {
      const k = Math.max(1, Math.floor(w.depths.length / 3));
      const first = med(w.depths.slice(0, k)), last = med(w.depths.slice(-k));
      return first - last >= Math.max(g.releaseM, g.releaseFrac * first);
    }
    for (const xs of [w.widths, w.heights]) {
      if (xs.length < 3) continue;
      const k = Math.max(1, Math.floor(xs.length / 3));
      if (med(xs.slice(-k)) >= (1 + g.nodepthGrow) * med(xs.slice(0, k))) return true;
    }
    return false;
  }
  private seenEnough(w: Warm, span: number) {
    return w.n >= 2 && Math.max(w.n, w.hits) >= this.cfg.warmMinSamples && w.lastT - w.firstT >= span - 1e-9;
  }
  private warmStatic(w: Warm, span: number) {
    const g = this.cfg;
    return this.seenEnough(w, span) && !this.isVehicle(w.ident) && w.maxDisp <= g.staticDisp && w.maxSize <= g.staticSizeFrac && !this.warmClosing(w);
  }
  /** The user in front of a desk webcam: a still, centred person within operatorMaxM (metric depth), or, without depth,
   *  a box taller than operatorNoDepthHeight of the frame (the user's box is cut off at the bottom, so its height says "far"). */
  private operatorLike(w: Warm, b: BBox): boolean {
    const g = this.cfg;
    if (w.kind !== "object" || w.ident.key !== "person" || !this.seenEnough(w, g.warmHoldSpan)) return false;
    if (Math.abs(center(b)[0] - 0.5) > g.operatorCenterTol || w.maxDisp > g.operatorDisp) return false;
    const h = b[3] - b[1];
    if (w.depths.length) return med(w.depths) <= g.operatorMaxM && h >= g.operatorMinHeight;
    return h >= g.operatorNoDepthHeight;
  }
  private record(w: Warm, tid: number, sayOnce = false, said = false): Rec {
    const half = Math.max(1, Math.floor(w.n / 2));   // a reference size only from a side the frame rarely cut
    return { ident: w.ident, kind: w.kind, label: w.label, bbox0: w.bbox, bbox: w.bbox, distRef: w.depths.length ? med(w.depths) : null,
      wRef: w.widths.length >= half ? med(w.widths) : null, hRef: w.heights.length >= half ? med(w.heights) : null,
      aRef: w.areas.length ? med(w.areas) : null, trackIds: new Set([tid]), released: false, sayOnce, said, approach: false, rest: [],
      last: this.hist.get(tid)?.at(-1) ?? null };
  }
  private finishWarmup() {
    const g = this.cfg;
    this.warmDone = true;
    const tEnd = (this.t0 ?? 0) + g.warmup;
    let bestOp: [number, Warm, number] | null = null;
    const bg: Array<[number, Warm, boolean, boolean]> = [];
    for (const [tid, w] of this.warm) {
      if (!(this.seenEnough(w, Math.max(g.bgMinSpan, 0.5 * g.warmup)) && w.lastT >= tEnd - 1.0)) continue;
      if (g.operatorFilter && this.operatorLike(w, w.bbox)) {
        const h = w.bbox[3] - w.bbox[1];
        if (!bestOp || h > bestOp[0]) bestOp = [h, w, tid];
      }
      if (this.warmStatic(w, g.bgMinSpan)) {
        const once = this.inPathNear(w.label, w.bbox, w.dists.length ? med(w.dists) : null, w.path);
        bg.push([tid, w, once, once && this.ownEntry(tid) != null]);
      }
    }
    const opTid = bestOp ? bestOp[2] : null;
    for (const [tid, w, once, said] of bg) if (tid !== opTid) this.bg.push(this.record(w, tid, once, said));
    if (bestOp) this.operator = this.record(bestOp[1], bestOp[2]);
    this.warm.clear();
  }
  private updateClosing(visible: Map<number, Track>, t: number) {
    for (const tid of [...this.closingSince.keys()]) if (!visible.has(tid)) this.closingSince.delete(tid);
    for (const [tid, tr] of visible) {
      if (this.measured(tr) && -tr.relSpeed >= this.cfg.releaseMps) { if (!this.closingSince.has(tid)) this.closingSince.set(tid, t); }
      else this.closingSince.delete(tid);
    }
  }
  /** How much nearer than its warm-up reference (m), and the reference: metric depth when there is depth now and was in
   *  the warm-up, else the box (median of the last 3) against its reference size. `latest`: the last sample only. */
  private drop(tr: Track, rec: Rec, latest = false): [number | null, number | null] {
    const zs = this.samples(tr).filter((s) => s.z != null);
    if (tr.depthConf >= 0.4 && rec.distRef != null && zs.length) {
      const d = latest ? (zs[zs.length - 1].z as number) : this.smoothedDepth(tr) ?? rec.distRef;
      return [rec.distRef - d, rec.distRef];
    }
    const boxes = this.samples(tr).slice(latest ? -1 : -3).map((s) => s.bbox);
    const [H, W] = TYPICAL_SIZE[rec.label] ?? DEFAULT_SIZE;
    let ratio: number, ref: number | null;
    if (rec.wRef && !boxes.some(cutLR)) { ratio = med(boxes.map((b) => b[2] - b[0])) / rec.wRef; ref = (this.cam.fx * W) / rec.wRef; }
    else if (rec.hRef && !boxes.some(cutTB)) { ratio = med(boxes.map((b) => b[3] - b[1])) / rec.hRef; ref = (this.cam.fy * H) / rec.hRef; }
    else if (rec.aRef) {
      ratio = Math.sqrt(med(boxes.map(area)) / rec.aRef);
      const rs = this.boxRanges(rec.bbox0, rec.label).filter((r): r is number => r != null);
      ref = rs.length ? Math.min(...rs) : null;
    } else return [null, null];
    if (rec.distRef != null) ref = rec.distRef;
    if (ref == null || ratio <= 1e-6) return [null, null];
    return [ref * (1 - 1 / ratio), ref];
  }
  /** Rest tracking: a background object or the user at rest for `recapture` s has its reference moved there (a lean or
   *  settling a little nearer is not an approach later), and one released as approaching is held again. */
  private trackRecords(visible: Map<number, Track>, t: number) {
    const g = this.cfg;
    if (!this.warmDone) return;
    for (const rec of [...(this.operator ? [this.operator] : []), ...this.bg]) {
      if (rec.released && !rec.approach) continue;   // moved away from its place: an ordinary object for good
      const tr = [...rec.trackIds].map((id) => visible.get(id)).find((x): x is Track => !!x);
      if (!tr || this.closingSince.has(tr.id) || this.shortClosing(tr) >= g.releaseMps) { rec.rest = []; continue; }
      const b = tr.bbox;
      rec.rest.push({ t, d: tr.depthConf >= 0.4 ? this.smoothedDepth(tr) : null, w: cutLR(b) ? null : b[2] - b[0], h: cutTB(b) ? null : b[3] - b[1], a: area(b), c: center(b) });
      if (t - rec.rest[0].t < g.recapture - 1e-9) continue;
      while (rec.rest.length > 1 && t - rec.rest[1].t >= g.recapture - 1e-9) rec.rest.shift();
      const r = this.atRest(rec.rest);
      if (!r) continue;
      const [d, w, h, a] = r;
      if (d != null) rec.distRef = d;
      rec.wRef = w ?? rec.wRef; rec.hRef = h ?? rec.hRef; rec.aRef = a; rec.bbox0 = b;
      if (rec.released) {
        rec.released = false; rec.approach = false;
        if (rec === this.operator) rec.bbox = b;
        for (const id of rec.trackIds) this.approach.delete(id);
        this.releases.push([t, tr.id, `${rec === this.operator ? "user" : "background"} at rest again`]);
      }
    }
  }
  private atRest(ss: RestSample[]): [number | null, number | null, number | null, number] | null {
    const g = this.cfg, n = ss.length;
    if (n < 3) return null;
    if (Math.max(...ss.map((s) => Math.hypot(s.c[0] - ss[0].c[0], s.c[1] - ss[0].c[1]))) > 2 * g.staticDisp) return null;
    const meds: Array<number | null> = [];
    for (const k of ["d", "w", "h"] as const) {
      const xs = ss.map((s) => s[k]).filter((x): x is number => x != null);
      if (xs.length < n) { meds.push(null); continue; }
      const spread = k === "d" ? Math.max(...xs) - Math.min(...xs) : Math.max(...xs) / Math.max(1e-6, Math.min(...xs)) - 1;
      if (spread > (k === "d" ? g.restSpreadM : g.restSizeFrac)) return null;
      meds.push(med(xs));
    }
    const areas = ss.map((s) => s.a);
    if (meds.every((m) => m == null) && Math.max(...areas) / Math.max(1e-6, Math.min(...areas)) - 1 > 2 * g.restSizeFrac) return null;
    return [meds[0], meds[1], meds[2], med(areas)];
  }
  /** Why a background object or the desk user is released as approaching, or "". The user needs a sustained (smoothed)
   *  drop or a measured approach: a box that jitters or a lean toward the laptop is not one. */
  private releaseWhy(tr: Track, rec: Rec, t: number): string {
    const g = this.cfg;
    const [drop, ref] = this.drop(tr, rec);
    if (drop != null && ref != null && drop >= Math.max(g.releaseM, g.releaseFrac * ref) - 1e-9) return `${drop.toFixed(2)} m nearer than its ${s1(ref)} m reference`;
    if (ref != null && this.measured(tr)) {
      const [last] = this.drop(tr, rec, true), fast = this.shortClosing(tr);
      if (last != null && last >= Math.max(g.releaseM, g.releaseFrac * ref) - 1e-9 && fast >= g.releaseFast) return `${last.toFixed(2)} m nearer, closing ${s1(fast)} m/s`;
    }
    if (rec !== this.operator && !this.measured(tr) && this.nearWithoutDepth(tr)) return "looming in the walking path (no metric depth)";
    const since = this.closingSince.get(tr.id);
    if (since != null && t - since >= g.releaseSpan - 1e-9) return `closing ${(-tr.relSpeed).toFixed(2)} m/s for ${s1(t - since)} s`;
    if (this.measured(tr) && Number.isFinite(tr.ttc) && tr.ttc < g.releaseTtc && -tr.relSpeed >= g.approachMin) return `TTC ${s1(tr.ttc)} s, closing`;
    return "";
  }
  private release(rec: Rec, tr: Track, t: number, approach: boolean, why: string) {
    rec.released = true; rec.approach = approach; rec.rest = [];
    this.releases.push([t, tr.id, `${rec === this.operator ? "user" : "background"} released: ${why}`]);
    if (approach) for (const id of new Set([...rec.trackIds, tr.id])) this.approach.set(id, t);
  }
  /** A track's first sighting (the gate's own sample; the track itself when it predates the gate's history). */
  private firstOf(tr: Track): Sample {
    return this.first.get(tr.id) ?? { t: tr.firstSeen, bbox: tr.bbox, z: tr.depthConf >= 0.4 && tr.distance != null ? tr.distance : null };
  }
  /** Could a track first seen as `first` be the object last seen as `last` (a tracker id switch, a duplicate box)? It
   *  appeared within switchGap of that sighting, overlapping it, at about its range (within max(m, frac x range); no
   *  range evidence either way passes). Its box now does not matter: one that walks up into the old box later is
   *  another object. */
  private continues(last: Sample | null, first: Sample, label: string, m: number, frac: number, minIou = 0.3): boolean {
    if (!last || first.t - last.t > this.cfg.switchGap + 1e-9 || iou(last.bbox, first.bbox) < minIou) return false;
    const r = this.rangePair(last, first, label);
    return !r || Math.abs(r[1] - r[0]) <= Math.max(m, frac * r[0]) + 1e-9;
  }
  /** The background record this track belongs to: its own, or one of the same identity it continues (continues()). */
  private matchBackground(tr: Track): Rec | null {
    const g = this.cfg;
    for (const b of this.bg) if (b.trackIds.has(tr.id)) return b;
    const id = this.identityOf(tr).id, first = this.firstOf(tr);
    for (const b of this.bg) {
      if (b.released || b.ident.id !== id || !this.continues(b.last, first, b.label, g.closerM, g.closerFrac)) continue;
      b.trackIds.add(tr.id);
      return b;
    }
    return null;
  }
  private matchOperator(tr: Track, visible: Map<number, Track>): boolean {
    const g = this.cfg, op = this.operator;
    if (!op || op.released || tr.kind !== "object" || this.identityOf(tr).key !== "person") return false;
    if (!op.trackIds.has(tr.id)) {
      const mine = [...op.trackIds].map((id) => visible.get(id)).filter((x): x is Track => !!x);
      const first = this.firstOf(tr);
      // the user's own track is in view: this is another person (a visitor), unless the two boxes are one detection
      // twice, which overlapped from its first sighting at the user's range (a visitor walking up in front does not)
      if (mine.length) {
        return mine.some((m) => iou(m.bbox, tr.bbox) >= g.operatorDupIou
          && this.continues(this.hist.get(m.id)?.at(-1) ?? null, first, "person", g.releaseM, g.releaseFrac, g.operatorDupIou));
      }
      // a tracker id switch of the user: first seen where and when the user was last seen, at the user's range; or the
      // user back from a break, settled at their place. A stranger who walks up to the camera after the user left was
      // first seen farther away and is not at rest: never the user.
      const h = tr.bbox[3] - tr.bbox[1];
      if (!(h >= 0.8 * Math.min(g.operatorMinHeight, op.bbox[3] - op.bbox[1]) && Math.abs(center(tr.bbox)[0] - 0.5) <= g.operatorCenterTol + 0.1
        && (this.continues(op.last, first, "person", g.releaseM, g.releaseFrac) || (iou(op.bbox, tr.bbox) >= 0.3 && this.settledAt(tr, op))))) return false;
      op.trackIds.add(tr.id);
    }
    op.bbox = tr.bbox;          // follow the user as they shift in the chair
    return true;
  }
  /** At rest for `recapture` s (its own samples: atRest) at the record's range (within max(releaseM, releaseFrac)). */
  private settledAt(tr: Track, rec: Rec): boolean {
    const g = this.cfg, h = this.samples(tr), t1 = h[h.length - 1].t;
    const ss = h.filter((x) => x.t >= t1 - g.recapture - 1e-9);
    if (t1 - ss[0].t < g.recapture - 0.5 - 1e-9) return false;
    const rest = ss.map((x) => ({ t: x.t, d: x.z, w: cutLR(x.bbox) ? null : x.bbox[2] - x.bbox[0], h: cutTB(x.bbox) ? null : x.bbox[3] - x.bbox[1], a: area(x.bbox), c: center(x.bbox) }));
    if (!this.atRest(rest)) return false;
    const [drop, ref] = this.drop(tr, rec);
    return drop == null || ref == null || Math.abs(drop) <= Math.max(g.releaseM, g.releaseFrac * ref) + 1e-9;
  }
  /** "warmup" / "background" / "operator" when the cue must be held for that reason, else null. Releases a background
   *  object or the user that closes in (an approach release: its next cue passes). */
  private backgroundReason(tr: Track, t: number, visible: Map<number, Track>): string | null {
    const g = this.cfg;
    if (!(g.background || g.operatorFilter)) return null;
    if (!this.warmDone) {
      const w = this.warm.get(tr.id);
      if (!w) return null;
      const isStatic = g.background && this.warmStatic(w, g.warmHoldSpan) && !this.inPathNear(tr.label, tr.bbox, tr.distance, tr.pathProb);
      const opLike = g.operatorFilter && this.operatorLike(w, tr.bbox);
      return isStatic || opLike ? "warmup" : null;
    }
    if (g.operatorFilter && this.operator && this.matchOperator(tr, visible)) {
      const why = this.releaseWhy(tr, this.operator, t);
      if (why) { this.release(this.operator, tr, t, true, why); return null; }
      return "operator";
    }
    if (g.background) {
      const b = this.matchBackground(tr);
      if (b && !b.released) {
        const why = this.releaseWhy(tr, b, t);
        if (why) { this.release(b, tr, t, true, why); return null; }
        // clearly farther than its place: another object the tracker matched to its track after a dropout (or it walked
        // off): an ordinary object from now on, news to the user since the background was never said
        const [drop, ref] = this.drop(tr, b);
        if (drop != null && ref != null && -drop >= Math.max(g.closerM, g.closerFrac * ref) - 1e-9) { this.release(b, tr, t, false, `${(-drop).toFixed(1)} m farther than its place`); return null; }
        const [cx, cy] = center(tr.bbox), [x0, y0] = center(b.bbox0);
        if (Math.hypot(cx - x0, cy - y0) > g.bgReleaseDisp) { this.release(b, tr, t, false, "moved"); return null; }
        b.bbox = tr.bbox;
        if (b.sayOnce && !b.said) return null;   // in the walking path and near: said once before it is silenced
        return "background";
      }
    }
    return null;
  }

  // ---- scheduler bookkeeping -------------------------------------------------------------------------------------
  /** A held cue is never spoken: its seconds go back to the scheduler's speech budget (its "announced" marks stay, so the
   *  scheduler's own recent-block and novelty decay keep it from offering the same repeat at every step). */
  private refund(cue: Cue) {
    const p = this.policy;
    if (!p || !isSpeech(cue) || cue.source === "cane") return;
    p.budget = Math.min(p.cfg.budgetCapacity, p.budget + (cue.duration || 0));
  }
  /** The gate said a cue itself (rescue, or a cane obstacle): charge the speech budget and mark the track announced. */
  private charge(cue: Cue, tr: Track | null, t: number) {
    const p = this.policy;
    if (p && isSpeech(cue)) p.budget = Math.max(-p.cfg.budgetCapacity, p.budget - (cue.duration || 0));
    if (!tr) return;
    tr.lastAnnounced = t; tr.lastAnnouncedBBox = tr.bbox; tr.announceCount++;
    if (p && tr.distance != null) p.announcedDist.set(tr.id, tr.distance);
  }
  /** A fresh cue about the object as it is now (compose.ts, with at least its distance and direction). */
  private refreshed(src: Cue, tr: Track, t: number): Cue {
    const p = this.policy, units = p?.cfg.units ?? "m", ego = p?.egoSpeed ?? 1;
    const d = this.rangeNow(tr) ?? tr.distance;
    const own = tr.depthConf >= 0.4 ? -(tr.relSpeed + ego) : 0;
    const dist = d ?? 6, moving = Math.abs(tr.lateralSpeed) > 0.4 + 0.05 * dist || own > 0.5;
    const slots = SLOTS.filter((s) => s === "object" || s === "distance" || s === "direction" || src.slots.includes(s));
    const message = compose(slots, { label: tr.label, kind: tr.kind, text: tr.text, distance: d, depthConf: tr.depthConf, angle: this.angleOf(tr),
      relSpeed: -own, lateralSpeed: moving ? tr.lateralSpeed : 0, lateral: tr.lateral, hedged: tr.confidence < (p?.cfg.sureConfidence ?? 0.6), units });
    const ttc = this.measured(tr) && Number.isFinite(tr.ttc) ? tr.ttc : src.ttc;
    return { ...src, t, message, slots: [...slots], duration: speechSeconds(message), distance: d, depthConf: tr.depthConf, ttc, source: "gate" };
  }
}
