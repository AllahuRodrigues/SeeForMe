// Slot-based minimal message composer (port of seeforme/cues/compose.py; live tracks carry no scene `detail`).
// It must stay word-for-word identical to Python: the message length is the speech_cost feature the learned
// scheduler reads, and web/parity/run.mjs compares the cue messages of both implementations.
export const SLOTS = ["object", "distance", "direction", "motion", "action", "detail"] as const;

/** Python's round(): halves go to the even neighbour (round(2.5) == 2, while Math.round(2.5) == 3). */
export function pyRound(x: number): number {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

export function objectPhrase(label: string, kind: string, text?: string | null, hedged = false) {
  if (kind === "text") { const t = (text ?? "").trim(); const core = t ? `sign reads ${t}` : "a sign"; return hedged && t ? `possibly a sign, ${t}` : core; }
  label = label.replace(/_/g, " ");
  if (label === "unknown object" || label === "obstruction") return hedged ? "possibly an obstacle" : "an unknown obstacle";
  return hedged ? `possibly a ${label}` : label;
}
export function distancePhrase(d: number | null, depthConf = 1, units: "m" | "ft" = "m") {
  if (d == null) return "";
  if (depthConf < 0.45) { if (d < 1.2) return "right in front of you"; return d < 2.5 ? "very close" : d < 4.5 ? "close" : ""; }
  if (d < 1) return "right in front of you";
  if (units === "ft") { let ft = d * 3.28084; ft = ft < 10 ? pyRound(ft) : 5 * pyRound(ft / 5); return `about ${ft} feet`; }
  const s = d < 3 ? String(pyRound(d * 2) / 2) : String(pyRound(d));
  return `about ${s} meter${s !== "1" ? "s" : ""}`;
}
export function directionPhrase(angle: number) {
  if (Math.abs(angle) < 10) return "ahead";
  const side = angle > 0 ? "right" : "left";
  return Math.abs(angle) < 25 ? `slightly ${side}` : `on your ${side}`;
}
export function motionPhrase(relSpeed: number, lateralSpeed: number, lateral: number) {
  if (Math.abs(lateralSpeed) > 0.4 && lateral * lateralSpeed < 0) return `crossing from the ${lateral > 0 ? "right" : "left"}`;
  if (relSpeed < -1.5) return "approaching";
  if (Math.abs(lateralSpeed) > 0.4) return `passing on your ${lateral > 0 ? "right" : "left"}`;
  return "";
}
/** action_phrase(label, detail=None): without scene detail only a curb has an action ("step down"). */
export function actionPhrase(label: string) { return label === "curb" ? "step down" : ""; }
/** detail_phrase(label, detail=None) is empty for every live track. */
export function detailPhrase(_label: string) { return ""; }

export interface ComposeArgs { label: string; kind: "object" | "text"; text?: string | null; distance: number | null; depthConf: number; angle: number; relSpeed: number; lateralSpeed: number; lateral: number; hedged?: boolean; units?: "m" | "ft" }
export function compose(slots: readonly string[], a: ComposeArgs): string {
  const first = [objectPhrase(a.label, a.kind, a.text, a.hedged)];
  if (slots.includes("distance")) { const d = distancePhrase(a.distance, a.depthConf, a.units ?? "m"); if (d) first.push(d); }
  if (slots.includes("direction")) first.push(directionPhrase(a.angle));
  if (slots.includes("motion")) { const m = motionPhrase(a.relSpeed, a.lateralSpeed, a.lateral); if (m) first.push(m); }
  const sentences = [first.join(", ")];
  if (slots.includes("action")) { const s = actionPhrase(a.label); if (s) sentences.push(s); }
  if (slots.includes("detail")) { const s = detailPhrase(a.label); if (s) sentences.push(s); }
  const msg = sentences.filter(Boolean).map((s) => s[0].toUpperCase() + s.slice(1)).join(". ");
  return msg.endsWith(".") ? msg : msg + ".";
}
/** Python: max(min_gap, len(message.split()) * seconds_per_word). */
export const speechSeconds = (msg: string, minGap = 1.2, spw = 0.35) => Math.max(minGap, msg.split(/\s+/).filter(Boolean).length * spw);
