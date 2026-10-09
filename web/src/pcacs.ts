// P-CACS: predictive, action-aware cue scheduler (port of seeforme/cues/pcacs.py).
// `SchedulerBase` computes the 29 EventFeatures exactly as Python's PCACSPolicy._features (same values, same order:
// see FEATURE_NAMES / featureVector); `PCACS` is the linear decision rule, `LearnedScheduler` (learned.ts) the GRU+RL one.
// web/parity/run.mjs replays Python track states through this file and checks every feature against Python.
import { compose, speechSeconds } from "./compose";
import type { Action, ChannelState, Cue, Track } from "./types";
import { area, center, Camera } from "./tracker";

/** seeforme/campus/world.py EVENT_SPECS (severity) + pcacs.py ACTIONABILITY / MIN_SLOTS. */
export const EVENT: Record<string, { sev: number; act: number }> = {
  stairs: { sev: 1.0, act: 1.0 }, curb: { sev: 0.8, act: 1.0 }, vehicle: { sev: 1.0, act: 1.0 }, bike: { sev: 0.9, act: 1.0 },
  hazard_static: { sev: 0.7, act: 0.8 }, barrier: { sev: 0.75, act: 0.85 }, person_moving: { sev: 0.5, act: 0.6 },
  person_static: { sev: 0.5, act: 0.5 }, door: { sev: 0.45, act: 0.6 }, landmark: { sev: 0.4, act: 0.5 }, sign_nav: { sev: 0.45, act: 0.7 },
  goal: { sev: 0.7, act: 0.9 }, unknown_obstacle: { sev: 0.6, act: 0.7 }, decor: { sev: 0, act: 0 }, sign_decor: { sev: 0, act: 0 },
};
/** pcacs.py MIN_SLOTS; an event type without an entry (decor, sign_decor) gets ["distance"], as in Python. */
export const MIN_SLOTS: Record<string, string[]> = {
  stairs: ["distance", "action"], curb: ["distance", "action"], vehicle: ["motion", "direction"], bike: ["motion", "direction"],
  hazard_static: ["distance"], barrier: ["distance"], person_moving: ["motion"], person_static: ["distance"], door: ["distance", "detail"],
  landmark: ["distance", "direction"], sign_nav: ["direction"], goal: ["distance", "direction"], unknown_obstacle: ["distance"],
};
/** seeforme/campus/world.py LABEL_EVENT, entry for entry. */
export const LABEL_EVENT: Record<string, string> = {
  stairs: "stairs", curb: "curb", car: "vehicle", bus: "vehicle", truck: "vehicle", motorcycle: "vehicle", bicycle: "bike", scooter: "bike",
  door: "door", elevator: "landmark", person: "person_static", cone: "barrier", barrier: "barrier", "wet floor sign": "barrier",
  chair: "hazard_static", bench: "hazard_static", box: "hazard_static", cart: "hazard_static", backpack: "hazard_static", suitcase: "hazard_static",
  "trash can": "hazard_static", "potted plant": "hazard_static", pole: "hazard_static", "dining table": "hazard_static",
  fence: "barrier", tree: "hazard_static", shelf: "hazard_static", machine: "hazard_static", "ground hazard": "hazard_static",
  "fire hydrant": "hazard_static", animal: "person_moving", sign: "landmark", crossing: "landmark",
};
// Up to v0.2.0 the web app also mapped umbrella / dog / couch / bed to hazards. Python maps them to "decor" (never
// announced; CampusBench even places couches as decor), and the learned scheduler was trained on Python's features, so
// the web now follows Python exactly. To announce dogs, add "dog" to LABEL_EVENT in seeforme/campus/world.py and here.
export const UNKNOWN_OBSTACLE_LABELS = new Set(["debris", "broken glass", "fallen sign", "wreckage", "rubble", "unknown object", "obstruction", "spill"]);

/** event_type_for(track) in pcacs.py */
export function eventType(tr: Pick<Track, "label" | "kind">): string {
  if (tr.kind === "text") return "sign_nav";
  return LABEL_EVENT[tr.label] ?? (UNKNOWN_OBSTACLE_LABELS.has(tr.label) ? "unknown_obstacle" : "decor");
}

// seeforme/cues/text_importance.py KeywordTextImportance (the learned n-gram classifier is Python-only)
const CRITICAL_PATTERNS = [
  /\bexit\b/, /\bstop\b/, /\bdanger\b/, /\bcaution\b/, /\bwarning\b/, /\bemergency\b/, /\bfire\b/, /\bwet\s*floor\b/, /\bdo\s*not\s*enter\b/,
  /\bkeep\s*out\b/, /\bstairs?\b/, /\belevator\b/, /\brestroom\b/, /\btoilet\b/, /\bmen\b/, /\bwomen\b/, /\bpush\b/, /\bpull\b/, /\bentrance\b/, /\bopen\b/,
  /\bclosed\b/, /\bout\s*of\s*order\b/, /\bno\s*entry\b/, /\bone\s*way\b/, /\bbus\s*stop\b/, /\bplatform\b/, /\bgate\s*\w+\b/, /\broom\s*\d+/,
  /\b\d{3,4}[a-z]?\b/, /\bfloor\s*\d/, /\blevel\s*\d/, /\bcross(ing)?\b/, /\bwalk\b/, /\bdont\s*walk\b/,
];
const DECORATIVE_PATTERNS = [
  /\bsale\b/, /\b%\s*off\b/, /\bwelcome\b/, /\bmenu\b/, /\bcoffee\b/, /\bspecial\b/, /\bfollow\s*us\b/, /\bwww\./, /\.com\b/, /\bevent\b/,
  /\bconcert\b/, /\bclub\b/, /\bhappy\b/, /\bbirthday\b/, /\bfree\b/, /\bnew\b/, /\bbuy\b/, /\bjoin\b/,
];
/** normalise_text: lower-case, strip accents (SAÍDA -> saida) and punctuation, collapse whitespace. */
export function normaliseText(s: string): string {
  s = s.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
  return s.replace(/[^a-z0-9%$.\s]/g, " ").replace(/\s+/g, " ").trim();
}
export function textImportance(text: string | null | undefined): number {
  if (!text) return 0;
  const s = normaliseText(text);
  if (!s) return 0;
  if (CRITICAL_PATTERNS.some((p) => p.test(s))) return 1.0;
  if (DECORATIVE_PATTERNS.some((p) => p.test(s))) return 0.05;
  const n = s.split(" ").length;
  return n <= 2 ? 0.35 : n <= 5 ? 0.2 : 0.1;
}

export interface PCACSWeights { wHazard: number; wTtc: number; wAction: number; wGoal: number; wNovelty: number; wUncert: number; wText: number; lamSpeech: number; lamQueue: number }
export interface PCACSConfig {
  weights: PCACSWeights; thresholdSpeak: number; thresholdInterrupt: number; interruptMargin: number;
  hapticTtc: number; hapticMinPath: number; hapticMinSeverity: number; hapticAfterSpeech: number; hapticRepeat: number;
  budgetCapacity: number; budgetRefill: number; urgentBypass: number; minConfidence: number; sureConfidence: number;
  ttcReact: number; ttcHorizon: number; reannounceMove: number; reannounceTau: number; recentBlock: number;
  decisionInterval: number; verbosity: "quiet" | "balanced" | "detailed"; units: "m" | "ft";
}
/** The tuned values of public/config/pcacs.json (scripts/export_web_config.py from configs/pcacs_tuned.yaml), so the app
 *  decides as Python does even when that file does not load. Re-copy them after re-tuning. */
export const DEFAULT_CFG: PCACSConfig = {
  weights: { wHazard: 1.4345838451670412, wTtc: 0.42464710889918655, wAction: 1.0565202797294195, wGoal: 1.8148035791469657, wNovelty: 0.39904830565667465,
    wUncert: 0.9047078367960077, wText: 1.5153253597080112, lamSpeech: 0.4840365373176261, lamQueue: 1.1321309592720592 },
  thresholdSpeak: 0.9189102379506626, thresholdInterrupt: 2.1414293827556574, interruptMargin: 0.5429629133852196, hapticTtc: 1.5, hapticMinPath: 0.6, hapticMinSeverity: 0.5,
  hapticAfterSpeech: 3.0, hapticRepeat: 1.0, budgetCapacity: 6.0, budgetRefill: 0.5963391537062852, urgentBypass: 0.573063215436029, minConfidence: 0.35,
  sureConfidence: 0.6, ttcReact: 1.5, ttcHorizon: 9.73126978643949, reannounceMove: 3.0, reannounceTau: 10.0, recentBlock: 2.8446761393410203,
  decisionInterval: 0.5, verbosity: "balanced", units: "m",
};

/** One candidate: every EventFeatures value (Python names in FEATURE_NAMES) plus what the decision and the UI need. */
export interface Candidate {
  track: Track; etype: string;
  severity: number; pathProb: number; hazard: number; urgency: number; actionability: number; goal: number; novelty: number; uncertainty: number;
  speechCost: number; queueCost: number; confidence: number; distance: number; depthConf: number; angle: number; relSpeed: number; lateralSpeed: number;
  ttc: number; growth: number; sinceAnnounced: number; announceCountN: number; textImportance: number; isText: boolean; moving: boolean;
  channelBusy: number; channelRemaining: number; budget: number; density: number; ownSpeed: number; egoSpeed: number;
  slots: string[]; message: string; duration: number; hedged: boolean;
  utility: number;          // linear P-CACS utility U
  score?: number;           // learned scheduler: best legal log-odds against silence (-Infinity when nothing is legal)
  best?: Action;            // learned scheduler: the action behind `score`
}

/** EventFeatures field order in seeforme/cues/pcacs.py (EventFeatures.names()); scheduler.json must list the same names. */
export const FEATURE_NAMES = [
  "severity", "path_prob", "hazard", "urgency", "actionability", "goal", "novelty", "uncertainty", "speech_cost", "queue_cost",
  "confidence", "distance", "depth_conf", "angle", "rel_speed", "lateral_speed", "ttc", "growth", "since_announced", "announce_count",
  "text_importance", "is_text", "moving", "channel_busy", "channel_remaining", "budget", "density", "own_speed", "ego_speed",
] as const;
/** EventFeatures.as_vector() */
export function featureVector(c: Candidate): number[] {
  return [c.severity, c.pathProb, c.hazard, c.urgency, c.actionability, c.goal, c.novelty, c.uncertainty, c.speechCost, c.queueCost,
    c.confidence, c.distance, c.depthConf, c.angle, c.relSpeed, c.lateralSpeed, c.ttc, c.growth, c.sinceAnnounced, c.announceCountN,
    c.textImportance, c.isText ? 1 : 0, c.moving ? 1 : 0, c.channelBusy, c.channelRemaining, c.budget, c.density, c.ownSpeed, c.egoSpeed];
}

/** Track.area_growth_rate(): relative bbox-area growth per second over the recent history (>0 means approaching). */
export function areaGrowthRate(tr: Track): number {
  const h = tr.bboxHistory;
  if (h.length < 2) return 0;
  const [t0, b0] = h[0], [t1, b1] = h[h.length - 1];
  const dt = t1 - t0;
  if (dt <= 1e-6) return 0;
  const a0 = area(b0), a1 = area(b1);
  if (a0 <= 1e-6) return 0;
  return (a1 - a0) / a0 / dt;
}

/** State and features shared by the linear and the learned scheduler (PCACSPolicy minus its decide()). */
export class SchedulerBase {
  budget: number; lastT: number | null = null; egoSpeed = 1.0; goalTags: string[] = []; lastCandidates: Candidate[] = [];
  announcedDist = new Map<number, number>(); lastAction: Action = "silence";
  constructor(public cfg: PCACSConfig = DEFAULT_CFG, public cam = new Camera()) { this.budget = cfg.budgetCapacity; }
  reset() { this.budget = this.cfg.budgetCapacity; this.lastT = null; this.egoSpeed = 1; this.announcedDist.clear(); this.lastCandidates = []; this.lastAction = "silence"; }
  setGoal(tags: string[]) { this.goalTags = tags.map((t) => t.toLowerCase()); }

  protected urgency(tr: Track) {
    const c = this.cfg;
    const ttc = Number.isFinite(tr.ttc) ? tr.ttc : Math.max(0, ((tr.distance ?? 6) - 0.6) / 1.0);
    return Math.max(0, Math.min(1, (c.ttcHorizon - ttc) / (c.ttcHorizon - c.ttcReact)));
  }
  protected goal(tr: Track) {
    if (!this.goalTags.length) return 0;
    const words = new Set<string>([tr.label.toLowerCase()]);
    if (tr.text) { words.add(tr.text.toLowerCase()); tr.text.toLowerCase().split(/\s+/).forEach((w) => words.add(w)); }
    if (tr.label === "chair" || tr.label === "bench") words.add("seat");
    // Web extension over Python's exact-word match: a multi-word tag ("room 204") also matches as a whole phrase inside
    // a longer sign ("Room 204 Lab"); whole words only, so "204" never matches a "1204" sign
    const text = (tr.text ?? "").toLowerCase();
    const phrase = (g: string) => new RegExp(`(^|[^\\p{L}\\p{N}])${g.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}($|[^\\p{L}\\p{N}])`, "u").test(text);
    return this.goalTags.some((g) => words.has(g) || (text !== "" && phrase(g))) ? 1 : 0;
  }
  protected novelty(tr: Track, t: number) {
    if (tr.lastAnnounced == null) return 1;
    let n = 1 - Math.exp(-(t - tr.lastAnnounced) / this.cfg.reannounceTau);
    const d0 = this.announcedDist.get(tr.id);
    if (d0 != null && tr.distance != null && d0 - tr.distance >= this.cfg.reannounceMove) n = Math.max(n, Math.min(1, (d0 - tr.distance) / (2 * this.cfg.reannounceMove)));
    let disp = 1.0;   // Track.displacement_since_announced()
    if (tr.lastAnnouncedBBox) { const [a, b] = center(tr.lastAnnouncedBBox), [x, y] = center(tr.bbox); disp = Math.sqrt((x - a) ** 2 + (y - b) ** 2); }
    if (disp > 0.25) n = Math.max(n, 0.6);
    return n / (1 + 0.4 * tr.announceCount);
  }
  protected updateEgo(tracks: Track[]) {
    const v = tracks.filter((tr) => tr.kind === "object" && tr.depthConf >= 0.5 && tr.distance != null && tr.distance < 12 && Math.abs(tr.lateralSpeed) < 0.5 && tr.distHistory.length >= 4)
      .map((tr) => -tr.relSpeed).sort((a, b) => a - b);
    if (v.length) this.egoSpeed = 0.7 * this.egoSpeed + 0.3 * Math.max(0, Math.min(2, v[Math.floor(v.length / 2)]));
  }
  protected slotsFor(etype: string, moving: boolean, angle: number) {
    const c = this.cfg; const base = ["object", ...(MIN_SLOTS[etype] ?? ["distance"])];
    if (moving && !base.includes("motion") && !["sign_nav", "goal", "landmark"].includes(etype)) base.push("motion");
    if (Math.abs(angle) > (c.verbosity === "quiet" ? 25 : 12) && !base.includes("direction")) base.push("direction");
    if (c.verbosity === "quiet") return base.filter((s) => s !== "detail");
    if (c.verbosity === "detailed") for (const s of ["direction", "detail"]) if (!base.includes(s)) base.push(s);
    return base;
  }
  /** budget += refill x elapsed (PCACSPolicy._refill) */
  protected refill(t: number) {
    const c = this.cfg;
    if (this.lastT != null) this.budget = Math.min(c.budgetCapacity, this.budget + c.budgetRefill * (t - this.lastT));
    this.lastT = t;
  }

  /** PCACSPolicy._features: one candidate per track, in track order (not sorted). */
  candidates(tracks: Track[], t: number, ch: ChannelState): Candidate[] {
    const c = this.cfg, w = c.weights;
    this.updateEgo(tracks);
    const raw: Candidate[] = [];
    const density = Math.min(1, tracks.length / 12);
    const channelBusy = ch.busy ? 1 : 0, channelRemaining = Math.min(1, ch.remaining / 3), budget = this.budget / c.budgetCapacity;
    for (const tr of tracks) {
      const etype0 = eventType(tr);
      const spec = EVENT[etype0];
      const distance = tr.distance ?? 6, depthConf = tr.depthConf;
      const [cx] = center(tr.bbox); const angle = this.cam.angleDeg(cx);
      const ownSpeed = tr.depthConf >= 0.4 ? -(tr.relSpeed + this.egoSpeed) : 0;   // > 0 = coming at the user
      const moving = Math.abs(tr.lateralSpeed) > 0.4 + 0.05 * distance || ownSpeed > 0.5;
      let severity: number, pathProb: number, actionability: number, ti = 0; const isText = tr.kind === "text";
      if (isText) {
        ti = textImportance(tr.text);
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
        confidence: tr.confidence, distance, depthConf, angle, relSpeed: tr.relSpeed, lateralSpeed: tr.lateralSpeed,
        ttc: Number.isFinite(tr.ttc) ? Math.min(10, tr.ttc) : 10, growth: Math.max(-1, Math.min(1, areaGrowthRate(tr))),
        sinceAnnounced: tr.lastAnnounced == null ? 1 : Math.min(1, (t - tr.lastAnnounced) / 10), announceCountN: Math.min(1, tr.announceCount / 3),
        textImportance: ti, isText, moving, channelBusy, channelRemaining, budget, density, ownSpeed, egoSpeed: this.egoSpeed,
        slots, message, duration, hedged, utility: 0 });
    }
    // queue cost needs the other candidates' urgency (remembered objects do not block the channel for others)
    const hz = raw.map((f) => (f.track.remembered ? 0 : f.hazard * f.urgency));
    raw.forEach((f, i) => {
      const others = hz.filter((_, j) => j !== i);
      f.queueCost = f.speechCost * (others.length ? Math.max(...others) : 0);
      const near = f.distance <= 9 ? 1 : 0, isT = f.isText ? 1 : 0;
      const value = w.wHazard * f.hazard * f.urgency + w.wTtc * f.hazard * Math.max(0, f.urgency - 0.5) * 2
        + w.wAction * f.actionability * f.pathProb * (0.4 + 0.6 * f.urgency) * (1 - isT)
        + w.wText * f.textImportance * f.pathProb * near * isT + w.wGoal * f.goal * near;
      f.utility = value * (0.35 + 0.65 * f.novelty) - w.wUncert * f.uncertainty - w.lamSpeech * f.speechCost - w.lamQueue * f.queueCost;
    });
    return raw;
  }
  /** Candidates sorted by linear utility (highest first). */
  features(tracks: Track[], t: number, ch: ChannelState): Candidate[] {
    return this.candidates(tracks, t, ch).sort((a, b) => b.utility - a.utility);
  }

  /** PCACSPolicy._make_cue: speak / interrupt book the announcement and spend budget, haptic only marks the track. */
  protected makeCue(f: Candidate, t: number, action: Action, utility = f.utility): Cue {
    const tr = f.track;
    if (action === "speak" || action === "interrupt") {
      tr.lastAnnounced = t; tr.lastAnnouncedBBox = tr.bbox; tr.announceCount++;
      this.announcedDist.set(tr.id, tr.distance ?? 0);
      this.budget = Math.max(-this.cfg.budgetCapacity, this.budget - f.duration);
    } else tr.lastHaptic = t;
    return { t, trackId: tr.id, label: tr.label, message: action === "haptic" ? "" : f.message, action, duration: action === "haptic" ? 0 : f.duration,
      utility, severity: f.severity, distance: tr.distance, depthConf: tr.depthConf, ttc: Number.isFinite(tr.ttc) ? tr.ttc : null, slots: f.slots };
  }
}

/** Linear P-CACS (PCACSPolicy.decide). */
export class PCACS extends SchedulerBase {
  readonly kind = "pcacs" as const;
  decide(tracks: Track[], t: number, ch: ChannelState): Cue | null {
    const c = this.cfg;
    this.refill(t);
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
