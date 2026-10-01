// P-CACS: predictive, action-aware cue scheduler (port of seeforme/cues/pcacs.py, linear variant).
import { compose, speechSeconds } from "./compose";
import type { Action, ChannelState, Cue, Track } from "./types";
import { area, center, Camera } from "./tracker";

export const EVENT: Record<string, { sev: number; act: number; slots: string[] }> = {
  stairs: { sev: 1.0, act: 1.0, slots: ["distance", "action"] }, curb: { sev: 0.8, act: 1.0, slots: ["distance", "action"] },
  vehicle: { sev: 1.0, act: 1.0, slots: ["motion", "direction"] }, bike: { sev: 0.9, act: 1.0, slots: ["motion", "direction"] },
  hazard_static: { sev: 0.7, act: 0.8, slots: ["distance"] }, barrier: { sev: 0.75, act: 0.85, slots: ["distance"] },
  person_moving: { sev: 0.5, act: 0.6, slots: ["motion"] }, person_static: { sev: 0.5, act: 0.5, slots: ["distance"] },
  door: { sev: 0.45, act: 0.6, slots: ["distance", "detail"] }, landmark: { sev: 0.4, act: 0.5, slots: ["distance", "direction"] },
  sign_nav: { sev: 0.45, act: 0.7, slots: ["direction"] }, goal: { sev: 0.7, act: 0.9, slots: ["distance", "direction"] },
  decor: { sev: 0, act: 0, slots: [] },
};
export const LABEL_EVENT: Record<string, string> = {
  stairs: "stairs", curb: "curb", car: "vehicle", bus: "vehicle", truck: "vehicle", motorcycle: "vehicle", bicycle: "bike", scooter: "bike",
  door: "door", elevator: "landmark", person: "person_static", cone: "barrier", barrier: "barrier", "wet floor sign": "barrier",
  chair: "hazard_static", bench: "hazard_static", box: "hazard_static", cart: "hazard_static", backpack: "hazard_static", suitcase: "hazard_static",
  "trash can": "hazard_static", "potted plant": "hazard_static", pole: "hazard_static", "dining table": "hazard_static",
  "fire hydrant": "hazard_static", umbrella: "hazard_static", dog: "person_moving", couch: "hazard_static", bed: "hazard_static",
};
const CRITICAL = /\b(exit|stop|danger|caution|warning|emergency|fire|wet floor|do not enter|stairs?|elevator|restroom|toilet|push|pull|entrance|closed|room \d+|gate \w+|floor \d|level \d|cross(ing)?|walk)\b/i;
const DECOR = /\b(sale|% off|welcome|menu|coffee|special|follow us|www\.|\.com|event|concert|club|happy|free|buy|join)\b/i;
export function textImportance(s: string) { if (!s) return 0; if (CRITICAL.test(s)) return 0.95; if (DECOR.test(s)) return 0.08; return /\d{3,4}/.test(s) ? 0.7 : 0.3; }

export interface PCACSWeights { wHazard: number; wTtc: number; wAction: number; wGoal: number; wNovelty: number; wUncert: number; wText: number; lamSpeech: number; lamQueue: number }
export interface PCACSConfig {
  weights: PCACSWeights; thresholdSpeak: number; thresholdInterrupt: number; interruptMargin: number;
  hapticTtc: number; hapticMinPath: number; hapticMinSeverity: number; hapticAfterSpeech: number; hapticRepeat: number;
  budgetCapacity: number; budgetRefill: number; urgentBypass: number; minConfidence: number; sureConfidence: number;
  ttcReact: number; ttcHorizon: number; reannounceMove: number; reannounceTau: number; recentBlock: number;
  decisionInterval: number; verbosity: "quiet" | "balanced" | "detailed"; units: "m" | "ft";
}
export const DEFAULT_CFG: PCACSConfig = {
  weights: { wHazard: 1.43, wTtc: 0.42, wAction: 1.06, wGoal: 2.5, wNovelty: 0.4, wUncert: 0.9, wText: 1.52, lamSpeech: 0.48, lamQueue: 1.13 },
  thresholdSpeak: 0.92, thresholdInterrupt: 2.14, interruptMargin: 0.54, hapticTtc: 1.5, hapticMinPath: 0.6, hapticMinSeverity: 0.5,
  hapticAfterSpeech: 3.0, hapticRepeat: 1.0, budgetCapacity: 6, budgetRefill: 0.6, urgentBypass: 0.57, minConfidence: 0.35,
  sureConfidence: 0.6, ttcReact: 1.5, ttcHorizon: 9.7, reannounceMove: 3.0, reannounceTau: 10, recentBlock: 2.8,
  decisionInterval: 0.5, verbosity: "balanced", units: "m",
};

export interface Candidate {
  track: Track; etype: string; severity: number; pathProb: number; hazard: number; urgency: number; actionability: number; goal: number;
  novelty: number; uncertainty: number; speechCost: number; queueCost: number; distance: number; depthConf: number; angle: number;
  ownSpeed: number; moving: boolean; ttc: number; textImportance: number; isText: boolean; slots: string[]; message: string; duration: number; hedged: boolean; utility: number;
}

export class PCACS {
  budget: number; lastT: number | null = null; egoSpeed = 1.0; goalTags: string[] = []; lastCandidates: Candidate[] = [];
  announcedDist = new Map<number, number>(); lastAction: Action = "silence";
  constructor(public cfg: PCACSConfig = DEFAULT_CFG, public cam = new Camera()) { this.budget = cfg.budgetCapacity; }
  reset() { this.budget = this.cfg.budgetCapacity; this.lastT = null; this.egoSpeed = 1; this.announcedDist.clear(); this.lastCandidates = []; }
  setGoal(tags: string[]) { this.goalTags = tags.map((t) => t.toLowerCase()); }

  private urgency(tr: Track) {
    const c = this.cfg;
    const ttc = Number.isFinite(tr.ttc) ? tr.ttc : Math.max(0, ((tr.distance ?? 6) - 0.6) / 1.0);
    return Math.max(0, Math.min(1, (c.ttcHorizon - ttc) / (c.ttcHorizon - c.ttcReact)));
  }
  private goal(tr: Track) {
    if (!this.goalTags.length) return 0;
    const words = new Set<string>([tr.label.toLowerCase()]);
    if (tr.text) { words.add(tr.text.toLowerCase()); tr.text.toLowerCase().split(/\s+/).forEach((w) => words.add(w)); }
    if (tr.label === "chair" || tr.label === "bench") words.add("seat");
    return this.goalTags.some((g) => words.has(g) || (tr.text ?? "").toLowerCase().includes(g)) ? 1 : 0;
  }
  private novelty(tr: Track, t: number) {
    if (tr.lastAnnounced == null) return 1;
    let n = 1 - Math.exp(-(t - tr.lastAnnounced) / this.cfg.reannounceTau);
    const d0 = this.announcedDist.get(tr.id);
    if (d0 != null && tr.distance != null && d0 - tr.distance >= this.cfg.reannounceMove) n = Math.max(n, Math.min(1, (d0 - tr.distance) / (2 * this.cfg.reannounceMove)));
    if (tr.lastAnnouncedBBox) { const [a, b] = center(tr.lastAnnouncedBBox), [x, y] = center(tr.bbox); if (Math.hypot(x - a, y - b) > 0.25) n = Math.max(n, 0.6); }
    return n / (1 + 0.4 * tr.announceCount);
  }
  private updateEgo(tracks: Track[]) {
    const v = tracks.filter((tr) => tr.kind === "object" && tr.depthConf >= 0.5 && tr.distance != null && tr.distance < 12 && Math.abs(tr.lateralSpeed) < 0.5 && tr.distHistory.length >= 4)
      .map((tr) => -tr.relSpeed).sort((a, b) => a - b);
    if (v.length) this.egoSpeed = 0.7 * this.egoSpeed + 0.3 * Math.max(0, Math.min(2, v[Math.floor(v.length / 2)]));
  }
  private slotsFor(etype: string, moving: boolean, angle: number) {
    const c = this.cfg; const base = ["object", ...(EVENT[etype]?.slots ?? ["distance"])];
    if (moving && !base.includes("motion") && !["sign_nav", "goal", "landmark"].includes(etype)) base.push("motion");
    if (Math.abs(angle) > (c.verbosity === "quiet" ? 25 : 12) && !base.includes("direction")) base.push("direction");
    if (c.verbosity === "quiet") return base.filter((s) => s !== "detail");
    if (c.verbosity === "detailed") for (const s of ["direction", "detail"]) if (!base.includes(s)) base.push(s);
    return base;
  }

  features(tracks: Track[], t: number, ch: ChannelState): Candidate[] {
    const c = this.cfg, w = c.weights;
    this.updateEgo(tracks);
    const raw: Candidate[] = [];
    for (const tr of tracks) {
      const etype0 = tr.kind === "text" ? "sign_nav" : (LABEL_EVENT[tr.label] ?? "decor");
      const spec = EVENT[etype0];
      const distance = tr.distance ?? 6, depthConf = tr.depthConf;
      const [cx] = center(tr.bbox); const angle = this.cam.angleDeg(cx);
      const ownSpeed = depthConf >= 0.4 ? -(tr.relSpeed + this.egoSpeed) : 0;
      const moving = Math.abs(tr.lateralSpeed) > 0.4 + 0.05 * distance || ownSpeed > 0.5;
      let severity: number, pathProb: number, actionability: number, ti = 0; const isText = tr.kind === "text";
      if (isText) {
        ti = textImportance(tr.text ?? "");
        const legible = Math.min(1, (tr.bbox[3] - tr.bbox[1]) / 0.05), facing = Math.abs(angle) < 30 ? 1 : 0.3;
        severity = 0.45 * ti; pathProb = legible * facing; actionability = EVENT.sign_nav.act * ti;
      } else { severity = spec.sev; pathProb = tr.pathProb; actionability = spec.act; }
      const hazard = severity * pathProb;
      const urgency = hazard > 0 ? this.urgency(tr) : 0;
      const goal = this.goal(tr);
      const etype = goal > 0 ? "goal" : etype0;
      const novelty = this.novelty(tr, t);
      const uncertainty = 1 - tr.confidence * (0.5 + 0.5 * depthConf);
      const slots = this.slotsFor(etype, moving, angle);
      const hedged = tr.confidence < c.sureConfidence;
      const message = compose(slots, { label: tr.label, kind: tr.kind, text: tr.text, distance, depthConf, angle, relSpeed: -ownSpeed, lateralSpeed: moving ? tr.lateralSpeed : 0, lateral: tr.lateral, hedged, units: c.units });
      const duration = speechSeconds(message);
      raw.push({ track: tr, etype, severity, pathProb, hazard, urgency, actionability, goal, novelty, uncertainty, speechCost: duration / 3, queueCost: 0,
        distance, depthConf, angle, ownSpeed, moving, ttc: Number.isFinite(tr.ttc) ? Math.min(10, tr.ttc) : 10, textImportance: ti, isText, slots, message, duration, hedged, utility: 0 });
    }
    const hz = raw.map((f) => f.hazard * f.urgency);
    raw.forEach((f, i) => {
      const others = hz.filter((_, j) => j !== i);
      f.queueCost = f.speechCost * (others.length ? Math.max(...others) : 0);
      const near = f.distance <= 9 ? 1 : 0, isT = f.isText ? 1 : 0;
      const value = w.wHazard * f.hazard * f.urgency + w.wTtc * f.hazard * Math.max(0, f.urgency - 0.5) * 2
        + w.wAction * f.actionability * f.pathProb * (0.4 + 0.6 * f.urgency) * (1 - isT)
        + w.wText * f.textImportance * f.pathProb * near * isT + w.wGoal * f.goal * near;
      f.utility = value * (0.35 + 0.65 * f.novelty) - w.wUncert * f.uncertainty - w.lamSpeech * f.speechCost - w.lamQueue * f.queueCost;
    });
    raw.sort((a, b) => b.utility - a.utility);
    return raw;
  }

  private makeCue(f: Candidate, t: number, action: Action): Cue {
    const tr = f.track;
    if (action === "speak" || action === "interrupt") {
      tr.lastAnnounced = t; tr.lastAnnouncedBBox = tr.bbox; tr.announceCount++;
      this.announcedDist.set(tr.id, tr.distance ?? 0);
      this.budget = Math.max(-this.cfg.budgetCapacity, this.budget - f.duration);
    } else tr.lastHaptic = t;
    return { t, trackId: tr.id, label: tr.label, message: action === "haptic" ? "" : f.message, action, duration: action === "haptic" ? 0 : f.duration,
      utility: f.utility, severity: f.severity, distance: tr.distance, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: f.slots };
  }

  decide(tracks: Track[], t: number, ch: ChannelState): Cue | null {
    const c = this.cfg;
    if (this.lastT != null) this.budget = Math.min(c.budgetCapacity, this.budget + c.budgetRefill * (t - this.lastT));
    this.lastT = t;
    const cands = this.features(tracks, t, ch);
    this.lastCandidates = cands; this.lastAction = "silence";
    for (const f of cands) {
      const tr = f.track;
      if (tr.confidence < c.minConfidence) continue;
      const imminent = f.severity >= c.hapticMinSeverity && f.pathProb >= c.hapticMinPath && f.ttc <= c.hapticTtc && f.depthConf >= 0.4;
      const recently = tr.lastAnnounced != null && t - tr.lastAnnounced < c.recentBlock;
      const hapticOk = imminent && (tr.lastHaptic == null || t - tr.lastHaptic >= c.hapticRepeat) && (tr.lastAnnounced == null || t - tr.lastAnnounced >= c.hapticAfterSpeech);
      if (recently) { if (hapticOk) { this.lastAction = "haptic"; return this.makeCue(f, t, "haptic"); } continue; }
      if (f.utility < c.thresholdSpeak) break;
      const urgent = f.hazard * f.urgency >= c.urgentBypass;
      if (!ch.busy) {
        if (!urgent && this.budget < f.duration) continue;
        this.lastAction = "speak"; return this.makeCue(f, t, "speak");
      }
      const critical = f.severity >= 0.9 && f.urgency >= 0.6 && f.pathProb >= 0.5 && ch.currentSeverity < 0.7;
      if (ch.currentTrack !== tr.id && f.severity > ch.currentSeverity && (critical || (f.utility >= c.thresholdInterrupt && f.utility >= ch.currentUtility + c.interruptMargin))) {
        this.lastAction = "interrupt"; return this.makeCue(f, t, "interrupt");
      }
      if (hapticOk) { this.lastAction = "haptic"; return this.makeCue(f, t, "haptic"); }
    }
    return null;
  }
}
export { area };
