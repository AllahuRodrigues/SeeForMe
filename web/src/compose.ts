// Slot-based minimal message composer (port of seeforme/cues/compose.py).
export const SLOTS = ["object", "distance", "direction", "motion", "action", "detail"] as const;

export function objectPhrase(label: string, kind: string, text?: string, hedged = false) {
  if (kind === "text") { const t = (text ?? "").trim(); return t ? (hedged ? `possibly a sign, ${t}` : `sign reads ${t}`) : "a sign"; }
  return hedged ? `possibly a ${label}` : label;
}
export function distancePhrase(d: number | null, depthConf = 1, units: "m" | "ft" = "m") {
  if (d == null) return "";
  if (depthConf < 0.45) { if (d < 1.2) return "right in front of you"; return d < 2.5 ? "very close" : d < 4.5 ? "close" : ""; }
  if (d < 1) return "right in front of you";
  if (units === "ft") { let ft = d * 3.28084; ft = ft < 10 ? Math.round(ft) : 5 * Math.round(ft / 5); return `about ${ft} feet`; }
  const m = d < 3 ? Math.round(d * 2) / 2 : Math.round(d);
  const s = d < 3 ? String(m) : String(Math.round(m));
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
export interface ComposeArgs { label: string; kind: "object" | "text"; text?: string; distance: number | null; depthConf: number; angle: number; relSpeed: number; lateralSpeed: number; lateral: number; hedged?: boolean; units?: "m" | "ft" }
export function compose(slots: string[], a: ComposeArgs): string {
  const first = [objectPhrase(a.label, a.kind, a.text, a.hedged)];
  if (slots.includes("distance")) { const d = distancePhrase(a.distance, a.depthConf, a.units ?? "m"); if (d) first.push(d); }
  if (slots.includes("direction")) first.push(directionPhrase(a.angle));
  if (slots.includes("motion")) { const m = motionPhrase(a.relSpeed, a.lateralSpeed, a.lateral); if (m) first.push(m); }
  let msg = first.join(", ");
  msg = msg[0].toUpperCase() + msg.slice(1);
  return msg.endsWith(".") ? msg : msg + ".";
}
export const speechSeconds = (msg: string, minGap = 1.2, spw = 0.35) => Math.max(minGap, msg.split(/\s+/).length * spw);
