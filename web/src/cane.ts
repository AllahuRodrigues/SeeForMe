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
//   "Obstacle ahead, about 1 meter, clearer on your left."   "Something at head height ahead, about 2 meters."
//   under 0.6 m straight ahead: a haptic stop (a buzz, an earcon on laptops)
// It never says or implies that a path is clear: "clearer on your left" compares two sides, it is not "clear". A missing
// or failed summary proposes nothing, and silence is not "all clear".
import { distancePhrase, speechSeconds } from "./compose";
import { Camera, center } from "./tracker";
import type { BBox, CaneLevel, CaneObstacle, CaneSector, CaneSummary, Cue, Track } from "./types";

const SECTORS: CaneSector[] = ["left", "ahead", "right"];
const LEVELS: CaneLevel[] = ["ground", "body", "head"];
export const CANE_CFG = {
  maxAge: 1.0,          // s: a summary older than this proposes nothing
  minConfidence: 0.5,   // obstacle confidence needed to speak
  horizon: 4.0,         // m
  hapticM: 0.6,         // straight ahead nearer than this: a haptic stop instead of words
  stale: 1.5,           // s: a pseudo-track not reported for this long is forgotten
  explainOverlap: 0.3,  // a detected track covering this share of the obstacle box explains it ...
  explainDepthM: 0.7,   // ... unless its metric depth disagrees by more than max(this, 35%)
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

/** "Obstacle ahead, about 1 meter, clearer on your left." / "Something at head height on your right, about 2 meters." */
export function caneMessage(o: CaneObstacle, suggest: CaneSummary["suggest"], units: "m" | "ft" = "m"): string {
  const dir = o.sector === "ahead" ? "ahead" : `on your ${o.sector}`;
  const what = o.level === "head" ? `Something at head height ${dir}` : o.level === "ground" ? `Low obstacle ${dir}` : `Obstacle ${dir}`;
  const parts = [what];
  const d = distancePhrase(o.distance_m, 0.9, units);
  if (d) parts.push(d);
  if (suggest === "left" || suggest === "right") parts.push(`clearer on your ${suggest}`);
  const msg = `${parts.join(", ")}.`;
  return suggest === "stop" ? `Stop. ${msg}` : msg;
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
}

export class CaneConsumer {
  last: { t: number; summary: CaneSummary } | null = null;
  /** why the last proposal was not made (diagnostics) */
  note = "";
  private tracks = new Map<string, Track>();
  constructor(public cam: Camera = new Camera()) {}

  reset() { this.last = null; this.tracks.clear(); this.note = ""; }

  /** A server answer for the frame taken at time t (null: no depth map). Older answers than the last are ignored. */
  update(summary: CaneSummary | null, t: number) {
    if (this.last && t <= this.last.t) return;
    for (const tr of this.tracks.values()) tr.misses++;
    if (!summary) { this.last = null; return; }
    this.last = { t, summary };
    const o = summary.obstacle;
    if (!o) return;
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

  /** One obstacle cue for the gate, or null (see the module comment for when). */
  propose(ctx: CaneContext): Cue | null {
    const c = CANE_CFG;
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
    const hapticStop = o.sector === "ahead" && o.distance_m < c.hapticM;
    if (!hapticStop && ctx.channelBusy) return no("channel busy");
    const message = caneMessage(o, s.summary.suggest, ctx.units);
    this.note = "";
    return { t: ctx.t, trackId: tr.id, label: "obstacle", message: hapticStop ? "" : message, action: hapticStop ? "haptic" : "speak",
      duration: hapticStop ? 0 : speechSeconds(message), utility: 1 + (c.horizon - o.distance_m) / c.horizon, severity: o.level === "head" ? 0.7 : 0.6,
      distance: o.distance_m, depthConf: tr.depthConf, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: ["object", "distance", "direction"],
      identity: caneIdentity(o), source: "cane" };
  }
}
