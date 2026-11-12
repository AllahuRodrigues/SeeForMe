// Cane: class-agnostic obstacle cues from the "cane" summary of POST /api/depth (server/app.py).
//
// The detector only knows 80 COCO classes: poles, walls, doors left ajar, bins, low branches and kerbs are invisible to
// it. The server reads the metric depth map like a white cane sweeps: free distance per image sector within 4 m and the
// nearest obstacle in the walking corridor, with its height level. This module turns that into one proposed cue per
// decision, only while navigating (not in a still or desk setup, where the depth map mostly sees the desk), only when no
// detected track explains the obstacle (the scheduler already speaks about those), and always through the announcement
// gate with identity "obstacle:<sector>:<level>", so a wall ahead is said once, again when it gets materially closer,
// and not every half second.
//
// Each identity is kept as a pseudo-track (negative id) with its own distance history, so the gate's rules (confirmed,
// closer, closing, critical with a measured TTC) apply to it as to any object.
//   "Obstacle ahead, about 1 meter, and on your right."   "Something at head height ahead, about 2 meters."
//   very close straight ahead: a haptic stop (a buzz, an earcon on laptops)
// It never says or implies that a path is clear. Path choice names what blocks ("... and on your right": that side was
// seen blocked too), never a side that looks open, because drop-offs (kerbs, stairs down) are not detected; the side that
// is not named is left to the user's cane. A missing or failed summary proposes nothing, and silence is not "all clear".
//
// Travel aid (Settings "I travel with", as seeforme/live/cane.py's aid). The BLV-Road-Nav focus group (Islam et al.,
// arXiv 2407.16777) wanted the assistant quiet about what the user's own cane can already touch, and warnings about
// "objects that pick you before you pick them" (head height) and what a cane finds too late:
//   cane       ground level: nothing within the cane's reach (1.2 m), farther only with a steady approach; body level from
//              1.2 to 2.5 m ahead (time to plan a detour), nothing nearer; head level always (within 2.5 m: a cane passes
//              under it); the stop only for body or head level within 0.6 m and a steady approach
//   guide-dog  the dog takes the user around ground and body obstacles in the path: head level and the stop only
//   none       every cue as before (any sector within 4 m, the stop for any level)
// Path choice follows the server's `suggest`, which applies the same rule as the live demo (cane.py path_choice_applies):
// with a cane only for a body obstacle beyond its reach (a low one stays "Low obstacle ahead"), never with a dog.
// With a cane or a dog only the corridor ahead is spoken about. A steady approach: over the last 1.5 s at least 3 maps
// spanning 0.4 s whose least-squares range closes at 0.25 m/s or more, none far off that line and none clearly farther
// than the one before (a walker closing on something, not a noisy floor patch). The aid changes only this layer: the
// scheduler's cues about detected objects (people, vehicles, chairs, the FIND goal) are the same for every aid.
//
// Repeats (seeforme/live/cane.py Cane.propose and commit, the same numbers). The gate keys its memory by level and lets an
// obstacle with a measured TTC under 2 s through at every decision while that TTC keeps falling, so walking up to a wall
// gave a cane cue whenever speech was free. The cane therefore keeps its own memory per identity, ground and body sharing
// one (the level of one obstacle may flip between maps): an identity is not proposed again within 2.5 s of being said
// unless it has since entered the 1 m band, and after that only once it is nearer by max(0.35 m, 15%) (40% with a cane or
// a dog, below head height) or 10 s have passed; it is forgotten when its sector shows nothing for 2 s or when it is seen
// clearly farther (a new obstacle). The stop is once per obstacle (re-armed after nothing was within 1 m ahead for 2 s),
// at most one per 3 s, and nothing is spoken for 3 s after it (the buzz said it). The gate still judges every cue.
//
// The cane cannot see what the gate let through. A spoken proposal counts as said when, at the next decision, the speech
// channel plays it (ctx.channelTrack is its pseudo-track's id); a free channel means it was not said (every cane message
// lasts at least 1.2 s, longer than one decision step). When the caller does not pass channelTrack, a busy channel may be
// another cue that crowded this one out, so the cue counts only as probably said: the 2.5 s gap applies to it, the "until
// nearer or 10 s" rule does not (it could hold back a warning that was never spoken), and after the gap the gate's own
// closer and closing rules decide. A haptic stop counts as given when it is proposed. A cue is proposed only once its
// pseudo-track is confirmed by the gate's rule (3 maps, or 2 spanning 0.15 s), which costs no time: the gate would hold an
// unconfirmed cue and say it at a later decision by itself, past this memory (a ground / body flip was then said as a new
// obstacle, and a one-map blip would have counted as a stop).
import { distancePhrase, speechSeconds } from "./compose";
import { GATE_DEFAULTS } from "./gate";
import { Camera, center } from "./tracker";
import type { BBox, CaneLevel, CaneObstacle, CaneSector, CaneSummary, Cue, Track } from "./types";

const SECTORS: CaneSector[] = ["left", "ahead", "right"];
const LEVELS: CaneLevel[] = ["ground", "body", "head"];
/** What the user travels with (Settings "I travel with"; the server's form field `aid`). */
export type TravelAid = "cane" | "guide-dog" | "none";
export const AIDS: TravelAid[] = ["cane", "guide-dog", "none"];
export const DEFAULT_AID: TravelAid = "cane";
export const isAid = (x: unknown): x is TravelAid => AIDS.includes(x as TravelAid);
/** How an answer about the path ends ("I can miss obstacles, so ..."): the user's own aid, never "clear"
 *  (seeforme/live/commands.py AID_ADVICE says the same). */
const AID_ADVICE: Record<TravelAid, string> = { cane: "keep using your cane", "guide-dog": "keep working with your guide dog", none: "go carefully" };
export const aidAdvice = (aid: TravelAid): string => AID_ADVICE[aid] ?? "keep using your cane or guide dog";
export const CANE_CFG = {
  maxAge: 1.0,          // s: a summary older than this proposes nothing
  minConfidence: 0.5,   // obstacle confidence needed to speak
  horizon: 4.0,         // m (no aid)
  hapticM: 0.6,         // straight ahead nearer than this: a haptic stop instead of words
  stale: 1.5,           // s: a pseudo-track not reported for this long is forgotten
  explainOverlap: 0.3,  // a detected track covering this share of the obstacle box explains it ...
  explainDepthM: 0.7,   // ... unless its metric depth disagrees by more than max(this, 35%)
  blockedM: 1.5,        // path choice: the corridor is blocked within this ...
  clearerByM: 1.0,      // ... and the side the server prefers is clear max(2 m, this beyond the obstacle): the other side,
                        // seen nearer than that, is named as blocked too
  // travel aid (module comment)
  reachM: 1.2,          // a long cane's reach: ground and body obstacles nearer than this are the cane's
  detourM: 2.5,         // with a cane or a dog, obstacles are said out to this (time to plan a detour)
  steadyS: 1.5,         // a steady approach: least squares over this window ...
  steadyMps: 0.25,      // ... closing at least this fast ...
  steadyResidM: 0.15,   // ... every map within this + steadyResidFrac x range of the line, none this much farther than
  steadyResidFrac: 0.05, // ... the map before
  sectorHistS: 2.0,     // per-sector range history kept for the steady-approach test
  // repeats (module comment; cane.py CaneConfig min_gap_s, near_m, closer_m, closer_frac, aid_closer_frac, remind_s,
  // forget_s, stop_gap_s, stop_rearm_s)
  minGapS: 2.5,         // an identity said less than this ago is not proposed again ...
  nearM: 1.0,           // ... unless it has since entered this band
  closerM: 0.35,        // after the gap: again only once nearer by max(closerM, closerFrac x the range when said) ...
  closerFrac: 0.15,
  aidCloserFrac: 0.4,   // ... (this share with a cane or a guide dog, for ground and body level) ...
  remindS: 10,          // ... or this long after it was said
  forgetS: 2.0,         // an identity whose sector showed nothing this long is forgotten (the next sighting is new)
  stopGapS: 3.0,        // nothing spoken, and no second stop, this long after a stop
  stopRearmS: 2.0,      // one stop per obstacle: re-armed after nothing was within nearM ahead for this long
};

const num = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
/** The "cane" field of a /api/depth reply, validated; null when absent, null or malformed (the app then proposes nothing). */
export function parseCane(raw: unknown): CaneSummary | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const cm = (r.clear_m && typeof r.clear_m === "object" ? r.clear_m : {}) as Record<string, unknown>;
  const clear_m = { left: num(cm.left), ahead: num(cm.ahead), right: num(cm.right) };
  let obstacle: CaneObstacle | null = null;
  const o = r.obstacle as Record<string, unknown> | null | undefined;
  if (o && typeof o === "object") {
    const d = num(o.distance_m), c = num(o.confidence), b = o.bbox;
    const sector = SECTORS.find((s) => s === o.sector), level = LEVELS.find((l) => l === o.level);
    const box = Array.isArray(b) && b.length === 4 && b.every((x) => num(x) != null) ? (b.map((x) => Math.max(0, Math.min(1, x as number))) as BBox) : null;
    if (d != null && d > 0 && c != null && sector && level && box && box[2] > box[0] && box[3] > box[1]) {
      obstacle = { sector, distance_m: d, level, confidence: Math.max(0, Math.min(1, c)), bbox: box };
    }
  }
  const suggest = r.suggest === "left" || r.suggest === "right" || r.suggest === "stop" ? r.suggest : null;
  return { clear_m, obstacle, suggest };
}

/** The side that blocks too when the corridor ahead is blocked within blockedM: the server's `suggest` names the side it
 *  saw open (left / right), and the other side, if it was seen at all, is blocked when it is not clear max(2 m,
 *  clearerByM beyond the obstacle). null: no path choice to report. Never the open side: drop-offs are not seen. */
export function blockedSide(o: CaneObstacle, summary: Pick<CaneSummary, "clear_m" | "suggest">): "left" | "right" | null {
  const c = CANE_CFG;
  if (o.sector !== "ahead" || o.level === "head" || o.distance_m > c.blockedM) return null;
  if (summary.suggest !== "left" && summary.suggest !== "right") return null;
  const other = summary.suggest === "left" ? "right" : "left";
  const v = summary.clear_m[other];
  return v != null && v < Math.max(2.0, o.distance_m + c.clearerByM) ? other : null;
}

/** "Obstacle ahead, about 1 meter, and on your right." / "Something at head height on your right, about 2 meters." /
 *  "Stop. Obstacle ahead, about 1 meter." (`blocked`: blockedSide; "stop": a spoken stop) */
export function caneMessage(o: CaneObstacle, blocked: "left" | "right" | "stop" | null, units: "m" | "ft" = "m"): string {
  const dir = o.sector === "ahead" ? "ahead" : `on your ${o.sector}`;
  const what = o.level === "head" ? `Something at head height ${dir}` : o.level === "ground" ? `Low obstacle ${dir}` : `Obstacle ${dir}`;
  const d = distancePhrase(o.distance_m, 0.9, units);
  if ((blocked === "left" || blocked === "right") && o.level !== "head") return `Obstacle ${dir}${d ? `, ${d},` : ""} and on your ${blocked}.`;
  const msg = `${[what, d].filter(Boolean).join(", ")}.`;
  return blocked === "stop" ? `Stop. ${msg}` : msg;
}

/** Least squares over the last steadyS: at least 3 ranges spanning 0.4 s closing at steadyMps or more, every one near the
 *  line and none clearly farther than the one before (seeforme/live/cane.py _SectorTrack.steady_approach). */
export function steadyApproach(hist: Array<[number, number]>, t: number): boolean {
  const c = CANE_CFG;
  const h = hist.filter(([ti]) => t - ti <= c.steadyS + 1e-6);
  if (h.length < 3 || h[h.length - 1][0] - h[0][0] < 0.4 - 1e-9) return false;
  const n = h.length, t0 = h[0][0];
  const tm = h.reduce((a, p) => a + (p[0] - t0), 0) / n, dm = h.reduce((a, p) => a + p[1], 0) / n;
  let sxy = 0, sxx = 0;
  for (const [ti, d] of h) { sxy += (ti - t0 - tm) * (d - dm); sxx += (ti - t0 - tm) ** 2; }
  if (sxx < 1e-12) return false;
  const k = sxy / sxx, b = dm - k * tm;
  const ds = h.map((p) => p[1]).sort((x, y) => x - y);
  const median = n % 2 ? ds[(n - 1) / 2] : (ds[n / 2 - 1] + ds[n / 2]) / 2;
  const tol = c.steadyResidM + c.steadyResidFrac * median;
  if (-k < c.steadyMps) return false;
  for (let i = 0; i < n; i++) {
    if (Math.abs(h[i][1] - (k * (h[i][0] - t0) + b)) > tol + 1e-9) return false;
    if (i && h[i][1] - h[i - 1][1] > tol + 1e-9) return false;
  }
  return true;
}

/** Why a cue about this obstacle is left to the user's travel aid, or "" when it is news for them (module comment). */
export function aidHolds(aid: TravelAid, o: CaneObstacle, steady: boolean): string {
  const c = CANE_CFG;
  if (aid === "none") return "";
  if (o.sector !== "ahead") return `beside the path (${aid === "cane" ? "cane" : "guide dog"} user)`;
  if (o.distance_m > c.detourM) return "beyond 2.5 m";
  if (o.level === "head") return "";
  if (aid === "guide-dog") return "the guide dog leads around it";
  if (o.distance_m < c.reachM) return "within the cane's reach";
  if (o.level === "body" || steady) return "";
  return "low obstacle without a steady approach";
}

export const caneIdentity = (o: Pick<CaneObstacle, "sector" | "level">) => `obstacle:${o.sector}:${o.level}`;
const pseudoId = (o: Pick<CaneObstacle, "sector" | "level">) => -(100 + 3 * SECTORS.indexOf(o.sector) + LEVELS.indexOf(o.level));

function overlapShare(obj: BBox, of: BBox): number {   // share of `of` covered by `obj`
  const w = Math.max(0, Math.min(obj[2], of[2]) - Math.max(obj[0], of[0])), h = Math.max(0, Math.min(obj[3], of[3]) - Math.max(obj[1], of[1]));
  const a = Math.max(1e-6, (of[2] - of[0]) * (of[3] - of[1]));
  return (w * h) / a;
}
/** A detected, confirmed object track explains the obstacle when its box covers it (or its centre) and, if both have
 *  metric depth, the distances agree. */
export function explainedBy(o: CaneObstacle, tracks: Track[]): Track | null {
  const [ox, oy] = center(o.bbox);
  for (const tr of tracks) {
    if (tr.id < 0 || tr.kind !== "object" || tr.misses > 0) continue;
    const b = tr.bbox;
    const covers = overlapShare(b, o.bbox) >= CANE_CFG.explainOverlap || (ox >= b[0] && ox <= b[2] && oy >= b[1] && oy <= b[3]);
    if (!covers) continue;
    if (tr.depthConf >= 0.4 && tr.distance != null && Math.abs(tr.distance - o.distance_m) > Math.max(CANE_CFG.explainDepthM, 0.35 * o.distance_m)) continue;
    return tr;
  }
  return null;
}

export interface CaneContext {
  t: number; tracks: Track[]; navigating: boolean; still: boolean; desk: boolean; channelBusy: boolean; units: "m" | "ft";
  /** what the user travels with (default: a white cane) */
  aid?: TravelAid;
  /** the track id of the cue the speech channel plays now (ChannelState.currentTrack, null when it is free), if the caller
   *  passes it: a cane cue then counts as said exactly when it is the one playing at the next decision. Without it, a
   *  busy channel at the next decision counts as probably said, which only the 2.5 s gap uses (module comment). */
  channelTrack?: number | null;
}
/** A cane identity said at time t, at range d; seen: the last time its sector reported an obstacle; sure: known to have
 *  been said (channelTrack, or a stop), not only probably (a busy channel; module comment). */
interface Said { t: number; d: number; seen: number; sure: boolean }

export class CaneConsumer {
  last: { t: number; summary: CaneSummary } | null = null;
  /** why the last proposal was not made (diagnostics) */
  note = "";
  private tracks = new Map<string, Track>();
  /** per sector: the reported obstacle's range over the last seconds, whatever its level (a level that flips between maps
   *  is one obstacle to the steady-approach test) */
  private sectorHist = new Map<CaneSector, Array<[number, number]>>();
  /** the identity memory (module comment): caneIdentity -> when and how near it was said */
  private said = new Map<string, Said>();
  /** the last spoken cue proposed, until the next decision tells whether it was said */
  private offer: { key: string; trackId: number; t: number; d: number; until: number } | null = null;
  private stopT = -Infinity;
  private stopArmed = true;
  private nearT = -Infinity;   // the last summary with an obstacle within nearM ahead
  constructor(public cam: Camera = new Camera()) {}

  reset() {
    this.last = null; this.tracks.clear(); this.sectorHist.clear(); this.note = "";
    this.said.clear(); this.offer = null; this.stopT = -Infinity; this.stopArmed = true; this.nearT = -Infinity;
  }
  /** Steady approach of the obstacle reported in this sector (module comment). */
  steady(sector: CaneSector, t: number): boolean { return steadyApproach(this.sectorHist.get(sector) ?? [], t); }

  /** A server answer for the frame taken at time t (null: no depth map). Older answers than the last are ignored. */
  update(summary: CaneSummary | null, t: number) {
    if (this.last && t <= this.last.t) return;
    for (const tr of this.tracks.values()) tr.misses++;
    const o0 = summary?.obstacle;
    if (o0 && o0.sector === "ahead" && o0.distance_m <= CANE_CFG.nearM && o0.confidence >= CANE_CFG.minConfidence) this.nearT = t;
    if (!this.stopArmed && t - this.nearT >= CANE_CFG.stopRearmS - 1e-9) this.stopArmed = true;
    if (!summary) { this.last = null; return; }
    this.last = { t, summary };
    for (const [k, h] of this.sectorHist) {
      while (h.length && t - h[0][0] > CANE_CFG.sectorHistS) h.shift();
      if (!h.length) this.sectorHist.delete(k);
    }
    const o = summary.obstacle;
    if (!o) return;
    for (const [k, e] of this.said) if (k.startsWith(`obstacle:${o.sector}:`)) e.seen = t;
    const sh = this.sectorHist.get(o.sector) ?? [];
    sh.push([t, o.distance_m]); this.sectorHist.set(o.sector, sh);
    const key = caneIdentity(o);
    let tr = this.tracks.get(key);
    if (!tr || t - tr.lastSeen > CANE_CFG.stale) {
      tr = { id: pseudoId(o), label: "obstacle", kind: "object", bbox: o.bbox, confidence: o.confidence, firstSeen: t, lastSeen: t, hits: 0, misses: 0,
        confirmed: false, lastAnnounced: null, lastAnnouncedBBox: null, announceCount: 0, lastHaptic: null, bboxHistory: [], distance: null, depthConf: 0,
        lateral: 0, relSpeed: 0, lateralSpeed: 0, ttc: Infinity, pathProb: 0, distHistory: [] };
      this.tracks.set(key, tr);
    }
    const z = o.distance_m, x = this.cam.lateralFromU(center(o.bbox)[0], z);
    tr.hits++; tr.misses = 0; tr.lastSeen = t; tr.bbox = o.bbox; tr.confidence = o.confidence; tr.confirmed = tr.hits >= 2;
    tr.distance = z; tr.depthConf = Math.max(0.45, Math.min(0.95, o.confidence)); tr.lateral = x;
    tr.bboxHistory.push([t, o.bbox]); if (tr.bboxHistory.length > 15) tr.bboxHistory.shift();
    tr.distHistory.push([t, z, x]);
    while (tr.distHistory.length > 2 && t - tr.distHistory[0][0] > 2.0) tr.distHistory.shift();
    const h = tr.distHistory;
    if (h.length >= 4 && h[h.length - 1][0] - h[0][0] >= 1.0 - 1e-9) {
      const n = h.length, tm = h.reduce((s, p) => s + p[0], 0) / n, zm = h.reduce((s, p) => s + p[1], 0) / n;
      let a = 0, b = 0;
      for (const p of h) { a += (p[0] - tm) * (p[1] - zm); b += (p[0] - tm) ** 2; }
      tr.relSpeed = b > 1e-9 ? a / b : 0;
    } else tr.relSpeed = 0;
    tr.ttc = -tr.relSpeed >= 0.12 ? Math.max(0, (z - 0.6) / -tr.relSpeed) : Infinity;
    tr.pathProb = o.sector === "ahead" ? 0.9 : 0.2;
  }

  /** The pseudo-tracks to show the gate this frame (the current obstacle's, plus recently seen ones for its memory). */
  pseudoTracks(t: number): Track[] {
    for (const [k, tr] of [...this.tracks]) if (t - tr.lastSeen > CANE_CFG.stale) this.tracks.delete(k);
    return [...this.tracks.values()];
  }

  /** Was the last spoken proposal said? Decided at the next decision while it would still be playing (module comment); a
   *  decision that comes later cannot tell, and the cue counts as not said. */
  private settle(ctx: CaneContext) {
    const f = this.offer;
    if (!f || ctx.t <= f.t + 1e-9) return;
    this.offer = null;
    if (ctx.t >= f.until - 1e-9) return;
    const sure = ctx.channelTrack !== undefined;
    if (sure ? ctx.channelTrack === f.trackId : ctx.channelBusy) this.said.set(f.key, { t: f.t, d: f.d, seen: f.t, sure });
  }
  /** Why this obstacle is not news yet (cane.py Cane.propose), or "": the memory of its identity, for ground and body also
   *  of the other of the two in its sector; none when it is clearly farther than when said (a new obstacle). */
  private repeatHold(o: CaneObstacle, t: number, aid: TravelAid): string {
    const c = CANE_CFG, key = caneIdentity(o), d = o.distance_m;
    const sibling = o.level === "head" ? null : caneIdentity({ sector: o.sector, level: o.level === "ground" ? "body" : "ground" });
    let e = this.said.get(key) ?? (sibling ? this.said.get(sibling) : undefined);
    if (e && d > e.d + Math.max(0.5, 0.25 * e.d)) { this.said.delete(key); e = undefined; }
    if (!e) return "";
    const band = d <= c.nearM && c.nearM < e.d;
    if (t - e.t < c.minGapS - 1e-9 && !band) return `said ${(t - e.t).toFixed(1)} s ago`;
    if (!e.sure) return "";
    const frac = aid === "none" || o.level === "head" ? c.closerFrac : c.aidCloserFrac;
    if (d > e.d - Math.max(c.closerM, frac * e.d) + 1e-9 && t - e.t < c.remindS - 1e-9) return "said, not nearer";
    return "";
  }

  /** One obstacle cue for the gate, or null (see the module comment for when). */
  propose(ctx: CaneContext): Cue | null {
    const c = CANE_CFG;
    this.settle(ctx);
    for (const [k, e] of this.said) if (ctx.t - e.seen > c.forgetS + 1e-9) this.said.delete(k);
    const no = (why: string) => { this.note = why; return null; };
    if (!ctx.navigating) return no("not navigating");
    if (ctx.desk) return no("desk setup");
    if (ctx.still) return no("camera still");
    const s = this.last;
    if (!s || ctx.t - s.t > c.maxAge) return no("no recent depth");
    const o = s.summary.obstacle;
    if (!o) return no("no obstacle");
    if (o.confidence < c.minConfidence || o.distance_m > c.horizon) return no("obstacle uncertain");
    const by = explainedBy(o, ctx.tracks);
    if (by) return no(`explained by ${by.label}`);
    const tr = this.tracks.get(caneIdentity(o));
    if (!tr || tr.misses > 0) return no("not current");
    const g = GATE_DEFAULTS;
    if (!(tr.hits >= g.minHits || (tr.hits >= 2 && tr.lastSeen - tr.firstSeen >= g.confirmAge - 1e-9))) return no("not confirmed yet");
    const aid = ctx.aid ?? DEFAULT_AID, steady = this.steady(o.sector, ctx.t);
    // the stop: with no aid for anything this near straight ahead; with a cane or a dog only at body or head level and in
    // a steady approach (the cane or the dog finds what is low; something the user stands at is not closing in). Once per
    // obstacle, at most one per stopGapS.
    const hapticStop = o.sector === "ahead" && o.distance_m < c.hapticM && (aid === "none" || (o.level !== "ground" && steady))
      && this.stopArmed && ctx.t - this.stopT >= c.stopGapS - 1e-9;
    if (!hapticStop) {
      const why = aidHolds(aid, o, steady);
      if (why) return no(why);
    }
    if (!hapticStop && ctx.channelBusy) return no("channel busy");
    if (!hapticStop) {
      if (ctx.t - this.stopT < c.stopGapS - 1e-9) return no("just stopped (the buzz said it)");
      const why = this.repeatHold(o, ctx.t, aid);
      if (why) return no(why);
    }
    const message = caneMessage(o, blockedSide(o, s.summary), ctx.units), key = caneIdentity(o);
    this.note = "";
    const cue: Cue = { t: ctx.t, trackId: tr.id, label: "obstacle", message: hapticStop ? "" : message, action: hapticStop ? "haptic" : "speak",
      duration: hapticStop ? 0 : speechSeconds(message), utility: 1 + (c.horizon - o.distance_m) / c.horizon, severity: o.level === "head" ? 0.7 : 0.6,
      distance: o.distance_m, depthConf: tr.depthConf, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: ["object", "distance", "direction"],
      identity: key, source: "cane" };
    if (hapticStop) {
      // the stop is the warning about what is ahead: no spoken cue about it right after (cane.py Cane.commit)
      this.stopT = ctx.t; this.stopArmed = false;
      this.said.set(key, { t: ctx.t, d: o.distance_m, seen: ctx.t, sure: true });
    } else this.offer = { key, trackId: tr.id, t: ctx.t, d: o.distance_m, until: ctx.t + cue.duration };
    return cue;
  }
}
