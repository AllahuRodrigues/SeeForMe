// Scripted track sequences for the announcement gate's tests (10 frames/s, a decision every 0.5 s, as in the app). Not a
// test file itself: gate.test.ts and gate_families.test.ts import it.
import { AnnouncementGate, type GateConfig } from "../src/gate";
import { Camera, TYPICAL_SIZE } from "../src/tracker";
import type { BBox, ChannelState, Cue, Track } from "../src/types";

export const FREE: ChannelState = { busy: false, remaining: 0, currentUtility: 0, currentSeverity: 0, currentTrack: null };
export const cam = new Camera();
/** box centred at u (0..1) with the given width and top / bottom */
export const boxAt = (u: number, w: number, top: number, bottom: number): BBox => [u - w / 2, top, u + w / 2, bottom];

export interface Obj { id: number; label?: string; bbox: BBox; z?: number | null; v?: number; path?: number; ttc?: number; hits?: number; text?: string }

export class Sim {
  gate: AnnouncementGate;
  tracks = new Map<number, Track>();
  log: Array<{ t: number; cue: Cue | null; reason: string; text: string }> = [];
  /** the app's tracker hands the gate only tracks with at least nInit matches (web/src/tracker.ts: 2) */
  constructor(cfg: Partial<GateConfig> = {}, public nInit = 1) { this.gate = new AnnouncementGate(cam, cfg); }
  visible(): Track[] { return [...this.tracks.values()].filter((tr) => tr.misses === 0 && tr.hits >= this.nInit); }

  /** One frame: the listed objects are seen (metric depth z, or none), every other track misses. */
  frame(t: number, objs: Obj[]): Track[] {
    const seen = new Set<number>();
    for (const o of objs) {
      seen.add(o.id);
      let tr = this.tracks.get(o.id);
      if (!tr) {
        tr = { id: o.id, label: o.label ?? "person", kind: o.text ? "text" : "object", text: o.text, bbox: o.bbox, confidence: 0.8, firstSeen: t, lastSeen: t, hits: 0, misses: 0,
          confirmed: false, lastAnnounced: null, lastAnnouncedBBox: null, announceCount: 0, lastHaptic: null, bboxHistory: [], distance: null, depthConf: 0,
          lateral: 0, relSpeed: 0, lateralSpeed: 0, ttc: Infinity, pathProb: 0.3, distHistory: [] };
        this.tracks.set(o.id, tr);
      }
      tr.bbox = o.bbox; tr.lastSeen = t; tr.hits = o.hits ?? tr.hits + 1; tr.misses = 0; tr.confirmed = tr.hits >= 2;
      const depth = o.z != null;
      tr.distance = depth ? o.z! : cam.distanceFromBBox(o.bbox, tr.label);
      tr.depthConf = depth ? 0.9 : 0.25;
      tr.relSpeed = o.v ?? 0;
      // as the tracker: a measured TTC from the closing speed; without depth it assumes the user walks at 1 m/s
      tr.ttc = o.ttc ?? (depth ? (-tr.relSpeed >= 0.12 ? Math.max(0, (tr.distance - 0.6) / -tr.relSpeed) : Infinity) : Math.max(0, tr.distance - 0.6));
      tr.pathProb = o.path ?? 0.3;
      tr.bboxHistory.push([t, o.bbox]); if (tr.bboxHistory.length > 15) tr.bboxHistory.shift();
      tr.distHistory.push([t, tr.distance, 0]); if (tr.distHistory.length > 25) tr.distHistory.shift();
    }
    for (const [id, tr] of this.tracks) if (!seen.has(id)) tr.misses++;
    const visible = this.visible();
    this.gate.observe(visible, t);
    return visible;
  }
  /** A decision: the scheduler offers a cue about track `id` (or nothing); returns what the gate lets through. */
  decide(t: number, id: number | null, action: Cue["action"] = "speak", ch: ChannelState = FREE, message = "Person, close, ahead."): Cue | null {
    const visible = this.visible();
    const cues: Cue[] = [];
    if (id != null) {
      const tr = this.tracks.get(id)!;
      cues.push({ t, trackId: id, label: tr.label, message, action, duration: action === "haptic" ? 0 : 1.2, utility: 1, severity: 0.5,
        distance: tr.distance, depthConf: tr.depthConf, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: ["object", "distance", "direction"] });
    }
    const out = this.gate.filter(cues, visible, t, ch);
    this.log.push({ t, cue: out, reason: this.gate.last?.reason ?? "", text: this.gate.last?.text ?? "" });
    return out;
  }
  /** Run frames from t0 to t1 (10 fps); `objs(t)` gives what is seen, `offer(t)` what the scheduler offers at each decision. */
  run(t0: number, t1: number, objs: (t: number) => Obj[], offer: (t: number) => number | null = () => null, action: Cue["action"] = "speak") {
    for (let k = 0; t0 + k * 0.1 <= t1 + 1e-9; k++) {
      const t = +(t0 + k * 0.1).toFixed(3);
      this.frame(t, objs(t));
      if (Math.abs(t * 2 - Math.round(t * 2)) < 1e-6) this.decide(t, offer(t), action);
    }
  }
  passes() { return this.log.filter((l) => l.cue); }
}

/** a box `label` would have at range d (its typical size), centred at u */
export function boxAtRange(label: string, d: number, u = 0.5): BBox {
  const [H, W] = TYPICAL_SIZE[label];
  const h = Math.min(0.98, (cam.fy * H) / d), w = Math.min(0.98, (cam.fx * W) / d);
  return [u - w / 2, 0.5 - h / 2, u + w / 2, 0.5 + h / 2];
}
