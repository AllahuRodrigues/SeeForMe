// What is the camera doing: still, walking, or unclear? A port of the frame-based part of seeforme/live/activity.py
// (ActivityDetector), so the thresholds calibrated there on EgoBlind development footage and synthetic desk-webcam
// sequences carry over.
//
// Each frame is shrunk to a 160 px wide grey copy and compared with a reference at least 0.2 s old (the step of the
// calibration footage; the numbers are the same at 5, 15 or 30 fps, and it caps the cost at 5 measurements a second):
//   shift     global image shift by phase correlation (frame widths per second; dx > 0: the scene moves right)
//   energy    mean |difference| of the two mean-normalised frames / 255, per 0.2 s
//   flow      energy / mean |gradient|: roughly how many pixels the content moved per 0.2 s, whatever the texture or
//             lighting. The main still-vs-moving signal: walking moves everything, a desk webcam sees only the user move
//   coverage  share of the textured 3x3 tiles whose content moved (whole-frame motion)
//   forward   expansion rate (1/s) from the shifts of the four quadrants (walking forward makes the image expand)
//   bob       |vertical shift| rate (a hand-held phone bobs at every step)
// Every quantity is smoothed (EMA 0.35 per 0.2 s) and classified: still (shift < 0.05 w/s and flow < 0.7 or energy <
// 0.01), walking (moving with a forward, bob or coverage cue; a sustained turn counts as walking), unclear otherwise.
// Hysteresis: a new state must win 70% of the votes over 1.5 s (still) or 1 s (walking); leaving "still" is the safe
// direction and is quick: 60% of the votes over 0.6 s for any sign of motion. No frame for 2 s: unclear.
//
// The gate uses it for the hold times (20 s walking, 60 s still) and the background, main.ts for the desk detection
// (a camera still at start-up is a laptop webcam) and to keep the cane quiet while still.
export type Activity = "still" | "walking" | "unclear";
type State = "stationary" | "walking" | "turning" | "unknown";
const MOVING: State[] = ["walking", "turning"];
const STEP = 0.2;   // the nominal measurement step every per-step quantity is expressed in (calibration footage: 5 fps)

export const ACTIVITY_CFG = {
  width: 160, minGap: 0.2, maxGap: 1.0, stale: 2.0, minResponse: 0.05, maxRate: 3.0, minGradient: 2.0,
  ema: 0.35, dwell: 1.5, dwellMotion: 1.0, dwellLeave: 0.6, leaveFrac: 0.6, switchFrac: 0.7,
  stillFlow: 0.7, stillEnergy: 0.01, stillShift: 0.05, moveFlow: 1.2, moveShift: 0.15,
  turnRate: 0.2, turnConsistency: 0.5, walkForward: 0.1, walkBob: 0.06, walkCoverage: 0.85,
};
export type ActivityConfig = typeof ACTIVITY_CFG;

export interface MotionSample { dt: number; dx: number; dy: number; forward: number; energy: number; flow: number; coverage: number | null; response: number }

/** RGBA pixels -> float grey (BT.601 weights, as OpenCV's BGR2GRAY). */
export function grayFromRGBA(px: Uint8ClampedArray | Uint8Array, w: number, h: number): Float32Array {
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) g[i] = 0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2];
  return g;
}

// ---- FFT phase correlation -------------------------------------------------------------------------------------
// Radix-2 FFT with cached twiddle and bit-reversal tables; a frame's spectra are computed once and reused, since each
// frame is the reference of the next measurement.
const tables = new Map<number, { cos: Float64Array; sin: Float64Array; rev: Uint32Array }>();
function table(n: number) {
  let t = tables.get(n);
  if (!t) {
    const cos = new Float64Array(n >> 1), sin = new Float64Array(n >> 1), rev = new Uint32Array(n);
    for (let k = 0; k < n >> 1; k++) { cos[k] = Math.cos((2 * Math.PI * k) / n); sin[k] = Math.sin((2 * Math.PI * k) / n); }
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; rev[i] = j; }
    t = { cos, sin, rev };
    tables.set(n, t);
  }
  return t;
}
/** In-place FFT of `n` complex values at re/im[off + k * stride]. */
function fft(re: Float64Array, im: Float64Array, n: number, off: number, stride: number, inverse: boolean) {
  const { cos, sin, rev } = table(n), sgn = inverse ? 1 : -1;
  for (let i = 1; i < n; i++) {
    const j = rev[i];
    if (i < j) {
      const a = off + i * stride, b = off + j * stride;
      let x = re[a]; re[a] = re[b]; re[b] = x; x = im[a]; im[a] = im[b]; im[b] = x;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const cr = cos[k * step], ci = sgn * sin[k * step];
        const a = off + (i + k) * stride, b = a + half * stride;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
}
/** 2-D FFT of a w x h array whose rows from `rows` on are zero (padding: their transform is zero too). */
function fft2(re: Float64Array, im: Float64Array, w: number, h: number, inverse: boolean, rows = h) {
  for (let y = 0; y < rows; y++) fft(re, im, w, y * w, 1, inverse);
  for (let x = 0; x < w; x++) fft(re, im, h, x, w, inverse);
  if (inverse) { const n = w * h; for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}
const hanns = new Map<number, Float64Array>();
const hann = (n: number) => {
  let h = hanns.get(n);
  if (!h) { h = Float64Array.from({ length: n }, (_, i) => 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)))); hanns.set(n, h); }
  return h;
};
const pow2Above = (n: number) => 2 ** Math.ceil(Math.log2(Math.max(2, n)));

/** The windowed (Hanning, as cv2.phaseCorrelate), mean-removed crop (x0, y0, cw, ch) of a frame `fw` px wide, zero-padded to
 *  powers of two, in the frequency domain. */
export interface Spectrum { re: Float64Array; im: Float64Array; w: number; h: number }
export function spectrum(g: Float32Array, fw: number, x0: number, y0: number, cw: number, ch: number): Spectrum {
  const w = pow2Above(cw), h = pow2Above(ch), re = new Float64Array(w * h), im = new Float64Array(w * h);
  const hx = hann(cw), hy = hann(ch);
  let m = 0;
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) m += g[(y0 + y) * fw + x0 + x];
  m /= cw * ch;
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) re[y * w + x] = (g[(y0 + y) * fw + x0 + x] - m) * hx[x] * hy[y];
  fft2(re, im, w, h, false, ch);
  return { re, im, w, h };
}
export const MIN_PSR = 8;   // peak-to-sidelobe ratio below which a correlation peak is noise (a featureless wall)
/** Shift (px) of b relative to a: the peak of the inverse normalised cross-power spectrum, refined to sub-pixel by a
 *  parabola per axis. `response`: the peak's share as cv2.phaseCorrelate reports it (1 = a perfect shift), set to 0 when the peak
 *  does not stand out of the correlation surface (peak-to-sidelobe ratio < MIN_PSR: the zero-padding raises the noise
 *  floor above cv2's, and a featureless, noisy scene must not read as motion). */
export function correlate(a: Spectrum, b: Spectrum): [number, number, number] {
  const { w, h } = a, n = w * h, re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) {   // R = B conj(A) / |B conj(A)|: its inverse peaks at the shift of b relative to a
    const r = b.re[k] * a.re[k] + b.im[k] * a.im[k], i = b.im[k] * a.re[k] - b.re[k] * a.im[k], m = Math.hypot(r, i) || 1e-12;
    re[k] = r / m; im[k] = i / m;
  }
  fft2(re, im, w, h, true);
  let best = -Infinity, px = 0, py = 0, s = 0, s2 = 0;
  for (let k = 0; k < n; k++) { const v = re[k]; s += v; s2 += v * v; if (v > best) { best = v; px = k % w; py = (k / w) | 0; } }
  const mean = s / n, sd = Math.sqrt(Math.max(1e-18, s2 / n - mean * mean));
  const at = (dx: number, dy: number) => re[((py + dy + h) % h) * w + ((px + dx + w) % w)];
  let resp = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) resp += at(dx, dy);
  // sub-pixel: a parabola through the peak and its two neighbours on each axis (a 5x5 centroid is pulled by the
  // correlation of something moving inside a still scene, e.g. the user fidgeting in front of a desk webcam)
  const parab = (m: number, c0: number, p: number) => { const d = m - 2 * c0 + p; return d < 0 ? Math.max(-1, Math.min(1, (m - p) / (2 * d))) : 0; };
  let fx = px + parab(at(-1, 0), best, at(1, 0)), fy = py + parab(at(0, -1), best, at(0, 1));
  if (fx > w / 2) fx -= w;
  if (fy > h / 2) fy -= h;
  if ((best - mean) / sd < MIN_PSR) resp = 0;
  return Number.isFinite(fx) && Number.isFinite(fy) && Number.isFinite(resp) ? [fx, fy, resp] : [0, 0, 0];
}
/** Shift (px) of `cur` relative to `prev` inside the crop (x0, y0, cw, ch) of frames `fw` px wide (see correlate). */
export function phaseShift(prev: Float32Array, cur: Float32Array, fw: number, x0: number, y0: number, cw: number, ch: number): [number, number, number] {
  return correlate(spectrum(prev, fw, x0, y0, cw, ch), spectrum(cur, fw, x0, y0, cw, ch));
}
/** The five spectra measureMotion uses: the frame's centred 128 px wide column (full height; a 160 px row would pad to
 *  256 and double the cost for little gain) and its four quadrants. */
export function frameSpectra(g: Float32Array, W: number, H: number): Spectrum[] {
  const w2 = W >> 1, h2 = H >> 1, cw = Math.min(W, 128);
  return [spectrum(g, W, (W - cw) >> 1, 0, cw, H), ...[[0, 0], [0, w2], [h2, 0], [h2, w2]].map(([y0, x0]) => spectrum(g, W, x0, y0, w2, h2))];
}

/** Motion between two grey frames (W x H, same size) taken dt seconds apart (see the module comment). `spectra`: the
 *  frames' frameSpectra when already computed (prev's from the previous measurement); filled in for the caller. */
export function measureMotion(prev: Float32Array, cur: Float32Array, W: number, H: number, dt: number, c: ActivityConfig = ACTIVITY_CFG,
  spectra: { prev?: Spectrum[]; cur?: Spectrum[] } = {}): MotionSample {
  dt = Math.max(1e-3, dt);
  const perStep = STEP / dt, clip = (v: number) => Math.max(-c.maxRate, Math.min(c.maxRate, v));
  let mp = 0, mc = 0;
  for (let i = 0; i < W * H; i++) { mp += prev[i]; mc += cur[i]; }
  mp /= W * H; mc /= W * H;
  // per-pixel |difference| and Sobel gradient of the reference (borders replicated), summed per 3x3 tile
  const tileD = new Float64Array(9), tileG = new Float64Array(9), tileN = new Float64Array(9);
  let dSum = 0, gSum = 0;
  for (let y = 0; y < H; y++) {
    const ym = (y > 0 ? y - 1 : 0) * W, y0 = y * W, yp = (y < H - 1 ? y + 1 : y) * W, ty = Math.min(2, ((3 * y) / H) | 0) * 3;
    for (let x = 0; x < W; x++) {
      const xm = x > 0 ? x - 1 : 0, xp = x < W - 1 ? x + 1 : x, i = y0 + x;
      const d = Math.abs(cur[i] - mc - (prev[i] - mp));
      const gx = prev[ym + xp] + 2 * prev[y0 + xp] + prev[yp + xp] - prev[ym + xm] - 2 * prev[y0 + xm] - prev[yp + xm];
      const gy = prev[yp + xm] + 2 * prev[yp + x] + prev[yp + xp] - prev[ym + xm] - 2 * prev[ym + x] - prev[ym + xp];
      const g = (Math.abs(gx) + Math.abs(gy)) / 8;
      dSum += d; gSum += g;
      const tile = ty + Math.min(2, ((3 * x) / W) | 0);
      tileD[tile] += d; tileG[tile] += g; tileN[tile]++;
    }
  }
  const dMean = dSum / (W * H), gMean = gSum / (W * H);
  const energy = (dMean / 255) * perStep, flow = (dMean / Math.max(c.minGradient, gMean)) * perStep;
  let moved = 0, textured = 0;
  for (let k = 0; k < 9; k++) {
    const g = tileG[k] / Math.max(1, tileN[k]);
    if (g < c.minGradient) continue;
    textured++;
    if (((tileD[k] / Math.max(1, tileN[k])) / g) * perStep > 0.5) moved++;
  }
  // global shift (the centred column); expansion from the four quadrants (a zoom by 1 + k moves the quadrant centred at r by k r)
  const sp = spectra.prev ?? frameSpectra(prev, W, H), sc = frameSpectra(cur, W, H);
  spectra.prev = sp; spectra.cur = sc;
  let [sx, sy, resp] = correlate(sp[0], sc[0]);
  if (resp < c.minResponse) { sx = 0; sy = 0; }
  const w2 = W >> 1, h2 = H >> 1;
  const quads: Array<[number, number, boolean, number, number]> = [[0, 0], [0, w2], [h2, 0], [h2, w2]].map(([qy0, qx0], k) => {
    const [qx, qy, qr] = correlate(sp[k + 1], sc[k + 1]);
    return [qx, qy, qr >= c.minResponse, qx0 + w2 / 2 - W / 2, qy0 + h2 / 2 - H / 2];
  });
  let forward = 0;
  if (quads.every((q) => q[2])) {
    const mx = quads.reduce((s, q) => s + q[0], 0) / 4, my = quads.reduce((s, q) => s + q[1], 0) / 4;
    const num = quads.reduce((s, [qx, qy, , rx, ry]) => s + (qx - mx) * rx + (qy - my) * ry, 0);
    const den = quads.reduce((s, [, , , rx, ry]) => s + rx * rx + ry * ry, 0);
    const k = den > 0 ? num / den : 0;
    forward = Math.log(Math.max(1e-3, 1 + k)) / dt;
  }
  return { dt, dx: clip(sx / W / dt), dy: clip(sy / W / dt), forward: clip(forward), energy, flow, coverage: textured >= 3 ? moved / textured : null, response: resp };
}

export class ActivityDetector {
  activity: Activity = "unclear";
  state: State = "unknown";
  since = 0;
  samples = 0;
  last: MotionSample | null = null;
  // smoothed features
  shift: [number, number] = [0, 0]; motion = 0; energy = 0; flow = 0; coverage = 0; forward = 0; bob = 0;
  private adx = 0;
  private hasCoverage = false;
  private ref: Float32Array | null = null; private refT = 0; private refW = 0; private refH = 0;
  private refSpectra: Spectrum[] | undefined;
  private lastT: number | null = null; private lastSampleT: number | null = null; private startT: number | null = null;
  private votes: Array<[number, State]> = [];

  constructor(public cfg: ActivityConfig = ACTIVITY_CFG) {}

  reset() {
    this.activity = "unclear"; this.state = "unknown"; this.since = 0; this.samples = 0; this.last = null;
    this.shift = [0, 0]; this.motion = this.energy = this.flow = this.coverage = this.forward = this.bob = this.adx = 0; this.hasCoverage = false;
    this.ref = null; this.refSpectra = undefined; this.lastT = this.lastSampleT = this.startT = null; this.votes = [];
  }

  /** One grey frame (W x H, ~160 px wide) at time t (s); returns the current activity. */
  update(gray: Float32Array, W: number, H: number, t: number): Activity {
    const c = this.cfg;
    if (this.lastT != null && t < this.lastT - 1e-6) this.reset();   // time went backwards: a new stream
    if (this.lastT == null) this.since = t;
    this.lastT = t;
    let s: MotionSample | null = null;
    if (!this.ref || this.refW !== W || this.refH !== H || t > this.refT + c.maxGap || t < this.refT) {
      this.ref = gray.slice(); this.refT = t; this.refW = W; this.refH = H; this.refSpectra = undefined;
    } else if (t - this.refT >= c.minGap - 1e-3) {
      const spectra: { prev?: Spectrum[]; cur?: Spectrum[] } = { prev: this.refSpectra };
      s = measureMotion(this.ref, gray, W, H, t - this.refT, c, spectra);
      this.ref = gray.slice(); this.refT = t; this.refSpectra = spectra.cur;   // the next measurement's reference
    }
    if (s) { this.last = s; this.lastSampleT = t; this.smooth(s); }
    else if (this.lastSampleT != null && t - this.lastSampleT > c.stale) {
      // no usable measurement for a while (camera stalled): stop trusting the old state
      this.reset(); this.lastT = t; this.since = t;
      return this.activity;
    }
    if (this.samples > 0) {
      this.startT ??= t;
      this.votes.push([t, this.classify()]);
      this.switchState(t);
    }
    return this.activity;
  }

  private smooth(s: MotionSample) {
    const a = Math.min(1, Math.max(0, 1 - (1 - this.cfg.ema) ** (Math.max(1e-3, s.dt) / STEP)));
    const first = this.samples === 0;
    const e = (old: number, v: number) => (first ? v : old + a * (v - old));
    this.shift = [e(this.shift[0], s.dx), e(this.shift[1], s.dy)];
    this.adx = e(this.adx, Math.abs(s.dx));
    this.motion = e(this.motion, Math.hypot(s.dx, s.dy));
    this.forward = e(this.forward, s.forward);
    this.bob = e(this.bob, Math.abs(s.dy));
    this.energy = e(this.energy, s.energy);
    this.flow = e(this.flow, s.flow);
    if (s.coverage != null) { this.coverage = this.hasCoverage ? this.coverage + a * (s.coverage - this.coverage) : s.coverage; this.hasCoverage = true; }
    this.samples++;
  }

  /** Instantaneous classification of the smoothed features (no dwell). */
  classify(): State {
    const c = this.cfg;
    if (this.samples === 0) return "unknown";
    const sdx = Math.abs(this.shift[0]);
    if (sdx >= c.turnRate && sdx >= c.turnConsistency * this.adx) return "turning";
    const still = this.motion < c.stillShift && (this.flow < c.stillFlow || this.energy < c.stillEnergy);
    const moving = this.motion >= c.moveShift || this.flow >= c.moveFlow;
    const walkCue = this.forward >= c.walkForward || this.bob >= c.walkBob || (this.hasCoverage && this.coverage >= c.walkCoverage);
    if (still) return "stationary";
    if (moving) return walkCue ? "walking" : "unknown";
    // between the thresholds: keep a settled state, but a forward or bob cue (slow walking) leaves "still"
    if (this.state === "stationary") return this.forward >= c.walkForward || this.bob >= c.walkBob ? "unknown" : "stationary";
    return this.state === "walking" ? "walking" : "unknown";
  }

  private switchState(t: number) {
    const c = this.cfg;
    const horizon = Math.max(c.dwell, c.dwellMotion);
    while (this.votes.length && this.votes[0][0] < t - horizon - 1e-9) this.votes.shift();
    if (!this.votes.length || this.startT == null) return;
    const window = (d: number) => (t - this.startT! < d - 1e-9 ? [] : this.votes.filter(([tv]) => tv >= t - d - 1e-9).map(([, a]) => a));
    const wins = (group: State[], w: State[], frac = c.switchFrac): State | null => {
      const inside = w.filter((a) => group.includes(a));
      if (!w.length || inside.length < frac * w.length) return null;
      return group.reduce((best, a) => (inside.filter((x) => x === a).length > inside.filter((x) => x === best).length ? a : best), group[0]);
    };
    const cur = this.state;
    const wm = window(c.dwellMotion), ws = window(c.dwell);
    let target: State | null;
    if (MOVING.includes(cur)) target = wins(MOVING.filter((a) => a !== cur), wm) ?? wins(["stationary"], ws);
    // leaving "still" is the safe direction: walking, turning and unknown votes all count against it, over a short window
    else if (cur === "stationary") target = wins(["walking", "turning", "unknown"], window(c.dwellLeave), c.leaveFrac);
    else target = wins(MOVING, wm) ?? wins(["stationary"], ws);
    if (target == null && cur !== "unknown") target = wins(["unknown"], ws);
    if (target != null && target !== cur) {
      this.state = target; this.since = t;
      this.activity = target === "stationary" ? "still" : MOVING.includes(target) ? "walking" : "unclear";
    }
  }

  describe(): string {
    const name = this.state === "turning" ? `turning ${this.shift[0] > 0 ? "left" : "right"}` : this.activity;
    return `${name} · shift ${this.shift[0].toFixed(2)},${this.shift[1].toFixed(2)} w/s · flow ${this.flow.toFixed(2)} · fwd ${this.forward.toFixed(2)}`;
  }
}
