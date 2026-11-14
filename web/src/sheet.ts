// A bottom sheet for a <details data-sheet> on a phone: it rises from the bottom edge on a spring, follows the finger
// one to one while it is dragged, resists past its resting place, and on release goes where the gesture was heading
// (dismissed or back open), carrying the finger's velocity into the spring. Every motion starts from where the sheet is
// on screen, so it can be caught and reversed at any moment. Only transform and opacity change (--sheet-y, --sheet-dim).
// With reduced motion the sheet shows and hides without moving. On a wide screen the disclosure stays inline.

/** One step of a spring, by damping ratio (1 = no overshoot) and response in seconds (lower = snappier). */
export function springStep(x: number, v: number, target: number, dt: number, damping = 1, response = 0.35): [number, number] {
  const w = (2 * Math.PI) / response, k = w * w, c = 2 * damping * w;
  const n = Math.max(1, Math.ceil(dt / (1 / 240)));          // sub-steps keep a stiff spring stable on a slow frame
  const h = dt / n;
  for (let i = 0; i < n; i++) { v += (-k * (x - target) - c * v) * h; x += v * h; }
  return [x, v];
}

/** How far a release at `velocity` px/s travels on, by exponential decay (scroll-view deceleration). */
export function project(velocity: number, decelerationRate = 0.998): number {
  return (velocity / 1000) * decelerationRate / (1 - decelerationRate);
}

/** Progressive resistance past an edge: the further the pull, the less the sheet follows. */
export function rubberband(overshoot: number, dimension: number, constant = 0.55): number {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}

/** Where a released sheet goes: `y` pixels below its resting place, moving at `velocity` px/s (down is positive). */
export function releaseTarget(y: number, velocity: number, height: number): "open" | "dismiss" {
  return y + project(velocity) > height / 2 ? "dismiss" : "open";
}

/** Release velocity in px/s from the last [time ms, position px] samples; zero when the finger rested before lifting.
 *  Samples closer together than a frame say nothing about speed, and no finger moves faster than MAX_V. */
const MAX_V = 6000;
export function velocityOf(samples: ReadonlyArray<readonly [number, number]>, restMs = 80): number {
  if (samples.length < 2) return 0;
  const last = samples[samples.length - 1], prev = samples[samples.length - 2];
  if (last[0] - prev[0] > restMs) return 0;
  const first = samples[Math.max(0, samples.length - 5)];
  const dt = last[0] - first[0];
  if (dt < 8) return 0;
  return Math.max(-MAX_V, Math.min(MAX_V, ((last[1] - first[1]) / dt) * 1000));
}

const PHONE = "(max-width: 59.99rem)", STILL = "(prefers-reduced-motion: reduce)";

/** Give `details` sheet behaviour on a phone. Returns a function that removes it. */
export function attachSheet(details: HTMLDetailsElement): () => void {
  const sheet = details.querySelector<HTMLElement>(".sheet"), grab = details.querySelector<HTMLElement>("[data-sheet-grab]");
  const summary = details.querySelector<HTMLElement>("summary");
  if (!sheet || !summary) return () => {};
  const phone = matchMedia(PHONE), still = matchMedia(STILL);
  let y = 0, v = 0, raf = 0, guard = 0, last = 0, dragging = false, grabY = 0;
  let samples: Array<[number, number]> = [];
  const height = () => sheet.getBoundingClientRect().height || 1;
  const paint = () => {
    details.style.setProperty("--sheet-y", `${y.toFixed(1)}px`);
    details.style.setProperty("--sheet-dim", String(Math.max(0, Math.min(1, 1 - y / height()))));
  };
  const clear = () => { cancelAnimationFrame(raf); raf = 0; clearTimeout(guard); details.style.removeProperty("--sheet-y"); details.style.removeProperty("--sheet-dim"); y = 0; v = 0; };
  const finish = (to: "open" | "dismiss") => { clear(); if (to === "dismiss") { details.open = false; summary.focus({ preventScroll: true }); } };
  /** Spring from where the sheet is now, with the velocity it has now. */
  const animate = (to: "open" | "dismiss", damping: number, response: number) => {
    cancelAnimationFrame(raf); clearTimeout(guard);
    if (still.matches) return finish(to);
    const target = to === "open" ? 0 : height();
    guard = window.setTimeout(() => finish(to), 1500);       // a tab that gets no frames must still open and close
    last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      [y, v] = springStep(y, v, target, dt, damping, response);
      if (to === "dismiss" && y >= target) return finish(to);                         // off the screen: done, no settling
      if (Math.abs(y - target) < 0.5 && Math.abs(v) < 20) return finish(to);
      paint(); raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  };
  const close = () => { if (details.open && phone.matches) animate("dismiss", 1, 0.3); else details.open = false; };

  const onToggle = () => {
    if (!details.open || !phone.matches) return clear();
    y = height(); v = 0; paint();                             // arrives from the edge it will leave by
    animate("open", 1, 0.35);
  };
  const onSummary = (e: Event) => { if (details.open && phone.matches) { e.preventDefault(); close(); } };
  const onClose = (e: Event) => { if ((e.target as HTMLElement).closest("[data-sheet-close]")) { e.preventDefault(); close(); } };
  const onKey = (e: KeyboardEvent) => { if (e.key === "Escape" && details.open && phone.matches) { e.stopPropagation(); close(); } };
  const onDown = (e: PointerEvent) => {
    if (!phone.matches || (e.target as HTMLElement).closest("button")) return;
    cancelAnimationFrame(raf); raf = 0; clearTimeout(guard);  // catch it mid-flight, from where it is
    dragging = true; grabY = e.clientY - y; v = 0; samples = [[e.timeStamp, y]];
    grab!.setPointerCapture(e.pointerId);
  };
  const onMove = (e: PointerEvent) => {
    if (!dragging) return;
    const raw = e.clientY - grabY;
    y = raw >= 0 ? raw : -rubberband(-raw, height());         // down follows the finger; up resists
    samples.push([e.timeStamp, y]); if (samples.length > 8) samples.shift();
    paint();
  };
  const onUp = (e: PointerEvent) => {
    if (!dragging) return;
    dragging = false; samples.push([e.timeStamp, y]);
    v = velocityOf(samples);
    const to = releaseTarget(y, v, height());
    animate(to, to === "open" && Math.abs(v) > 200 ? 0.8 : 1, 0.3);   // a little bounce only when a flick carried it
  };

  details.addEventListener("toggle", onToggle); summary.addEventListener("click", onSummary);
  details.addEventListener("click", onClose); details.addEventListener("keydown", onKey);
  grab?.addEventListener("pointerdown", onDown); grab?.addEventListener("pointermove", onMove);
  grab?.addEventListener("pointerup", onUp); grab?.addEventListener("pointercancel", onUp);
  return () => {
    clear();
    details.removeEventListener("toggle", onToggle); summary.removeEventListener("click", onSummary);
    details.removeEventListener("click", onClose); details.removeEventListener("keydown", onKey);
    grab?.removeEventListener("pointerdown", onDown); grab?.removeEventListener("pointermove", onMove);
    grab?.removeEventListener("pointerup", onUp); grab?.removeEventListener("pointercancel", onUp);
  };
}
