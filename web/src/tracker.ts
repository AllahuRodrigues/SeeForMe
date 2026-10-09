// IoU tracker with metric dynamics: a port of seeforme/tracking + seeforme/geometry/dynamics.
import type { BBox, Detection, Track } from "./types";

export const TYPICAL_SIZE: Record<string, [number, number]> = {
  person: [1.7, 0.5], chair: [0.9, 0.5], door: [2.1, 0.9], car: [1.5, 1.8], bicycle: [1.0, 0.6], motorcycle: [1.1, 0.8],
  bus: [3.0, 2.5], truck: [2.5, 2.4], dog: [0.5, 0.4], bench: [0.9, 1.6], "dining table": [0.75, 1.4], couch: [0.9, 1.8],
  backpack: [0.45, 0.35], suitcase: [0.7, 0.45], "potted plant": [0.6, 0.4], "fire hydrant": [0.8, 0.3], "stop sign": [0.75, 0.75],
  "traffic light": [1.0, 0.35], umbrella: [1.0, 1.0], text: [0.3, 0.8],
};
const DEFAULT_SIZE: [number, number] = [0.8, 0.5];

export class Camera {
  constructor(public hfovDeg = 70, public aspect = 4 / 3, public heightM = 1.35, public pitchDeg = 15) {}
  // same operation order as Python (math.radians / math.degrees), so the angle feature matches bit for bit
  get fx() { return 0.5 / Math.tan((this.hfovDeg * (Math.PI / 180)) / 2); }
  get fy() { return this.fx * this.aspect; }
  lateralFromU(u: number, z: number) { return ((u - 0.5) * z) / this.fx; }
  angleDeg(u: number) { return (180 / Math.PI) * Math.atan2(u - 0.5, this.fx); }
  distanceFromBBox(b: BBox, label: string) {
    const h = Math.max(1e-3, b[3] - b[1]);
    const [H] = TYPICAL_SIZE[label] ?? DEFAULT_SIZE;
    return Math.max(0.3, (this.fy * H) / h);
  }
}

export function iou(a: BBox, b: BBox) {
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]), ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}
export const area = (b: BBox) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
export const center = (b: BBox): [number, number] => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];

function phi(x: number) { // standard normal CDF
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}
function slope(pts: Array<[number, number, number]>, idx: 1 | 2) {
  const n = pts.length; if (n < 2) return 0;
  const tm = pts.reduce((s, p) => s + p[0], 0) / n, vm = pts.reduce((s, p) => s + p[idx], 0) / n;
  let num = 0, den = 0;
  for (const p of pts) { num += (p[0] - tm) * (p[idx] - vm); den += (p[0] - tm) ** 2; }
  return den > 1e-9 ? num / den : 0;
}

export interface DynCfg { corridorHalf: number; contact: number; historyS: number; minApproach: number; defaultUserSpeed: number; sigma: number; sigmaPerS: number; ema: number; minSamples: number }
export const DYN: DynCfg = { corridorHalf: 0.45, contact: 0.6, historyS: 2.0, minApproach: 0.12, defaultUserSpeed: 1.0, sigma: 0.25, sigmaPerS: 0.12, ema: 0.35, minSamples: 4 };

export function updateDynamics(tr: Track, det: Detection | null, t: number, cam: Camera, c = DYN) {
  let z: number, conf: number;
  if (det && det.distance_m != null && det.depth_conf > 0) { z = det.distance_m; conf = det.depth_conf; }
  else { z = cam.distanceFromBBox(tr.bbox, tr.label); conf = 0.25; }
  if (tr.distance != null && tr.depthConf > 0) z = 0.5 * z + 0.5 * tr.distance;
  const [cx] = center(tr.bbox);
  const x = cam.lateralFromU(cx, z);
  tr.distance = z; tr.depthConf = conf; tr.lateral = x;
  tr.distHistory.push([t, z, x]);
  while (tr.distHistory.length > 2 && t - tr.distHistory[0][0] > c.historyS) tr.distHistory.shift();
  if (tr.distHistory.length > 25) tr.distHistory.shift();
  if (tr.distHistory.length >= c.minSamples) {
    const vz = slope(tr.distHistory, 1), vx = slope(tr.distHistory, 2);
    tr.relSpeed = tr.relSpeed !== 0 ? c.ema * vz + (1 - c.ema) * tr.relSpeed : vz;
    tr.lateralSpeed = tr.lateralSpeed !== 0 ? c.ema * vx + (1 - c.ema) * tr.lateralSpeed : vx;
  }
  // ttc & path probability
  const [, w] = TYPICAL_SIZE[tr.label] ?? DEFAULT_SIZE;
  let approach = -tr.relSpeed;
  if (tr.distHistory.length < c.minSamples || tr.depthConf < 0.4) approach = c.defaultUserSpeed;
  const ttc = approach >= c.minApproach ? Math.max(0, (z - c.contact) / approach) : Infinity;
  tr.ttc = ttc;
  const horizon = Number.isFinite(ttc) ? Math.min(ttc, 6) : 6;
  const xArr = x + tr.lateralSpeed * horizon;
  const sigma = c.sigma + c.sigmaPerS * horizon + (tr.depthConf < 0.4 ? 0.2 : 0);
  const half = c.corridorHalf + w / 2;
  let p = phi((half - xArr) / sigma) - phi((-half - xArr) / sigma);
  if (!Number.isFinite(ttc)) p *= 0.5;
  const vx = tr.lateralSpeed;
  if (Math.abs(vx) > 0.3 && x * vx < 0) {
    const tau = -x / vx, zThen = z + tr.relSpeed * tau;
    if (tau >= 0 && tau <= 2.5 && zThen < 3.5) p = Math.max(p, 0.95 - 0.1 * tau);
  }
  tr.pathProb = Math.max(0, Math.min(1, p));
}

export class Tracker {
  tracks = new Map<number, Track>();
  nextId = 1;
  constructor(public cam: Camera, public iouThr = 0.3, public maxAge = 8, public nInit = 2, public confEma = 0.6) {}

  reset() { this.tracks.clear(); this.nextId = 1; }

  update(dets: Detection[], t: number): Track[] {
    const ids = [...this.tracks.keys()];
    const used = new Set<number>(), matchedT = new Set<number>();
    // greedy by affinity
    const pairs: Array<[number, number, number]> = [];
    ids.forEach((tid, i) => dets.forEach((d, j) => {
      const tr = this.tracks.get(tid)!;
      if (d.kind !== tr.kind) return;
      let a = iou(d.bbox, tr.bbox);
      if (d.kind === "object" && d.label !== tr.label) a *= 0.5;
      if (a >= this.iouThr) pairs.push([a, i, j]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    for (const [, i, j] of pairs) {
      if (matchedT.has(i) || used.has(j)) continue;
      matchedT.add(i); used.add(j);
      const tr = this.tracks.get(ids[i])!, d = dets[j];
      tr.confidence = this.confEma * d.confidence + (1 - this.confEma) * tr.confidence;
      tr.bbox = d.bbox; tr.bboxHistory.push([t, d.bbox]); if (tr.bboxHistory.length > 15) tr.bboxHistory.shift();
      tr.lastSeen = t; tr.hits++; tr.misses = 0;
      if (d.kind === "text" && d.text) tr.text = d.text;
      if (d.label !== tr.label && d.confidence > tr.confidence) tr.label = d.label;
      if (tr.hits >= this.nInit) tr.confirmed = true;
      updateDynamics(tr, d, t, this.cam);
    }
    dets.forEach((d, j) => {
      if (used.has(j)) return;
      const tr: Track = {
        id: this.nextId++, label: d.label, kind: d.kind, text: d.text, bbox: d.bbox, confidence: d.confidence, firstSeen: t, lastSeen: t,
        hits: 1, misses: 0, confirmed: this.nInit <= 1, lastAnnounced: null, lastAnnouncedBBox: null, announceCount: 0, lastHaptic: null,
        bboxHistory: [[t, d.bbox]], distance: null, depthConf: 0, lateral: 0, relSpeed: 0, lateralSpeed: 0, ttc: Infinity, pathProb: 0, distHistory: [],
      };
      updateDynamics(tr, d, t, this.cam);
      this.tracks.set(tr.id, tr);
    });
    ids.forEach((tid, i) => {
      if (matchedT.has(i)) return;
      const tr = this.tracks.get(tid)!;
      tr.misses++;
      if (tr.misses > this.maxAge) this.tracks.delete(tid);
    });
    return [...this.tracks.values()].filter((tr) => tr.misses === 0 && tr.confirmed);
  }
}
