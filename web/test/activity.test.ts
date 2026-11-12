// Camera activity on synthetic 160 x 120 grey frames (a smooth texture sampled at sub-pixel positions, so shifts and
// zooms are exact): phase correlation finds the shift, a zoom reads as forward motion, a still camera (sensor noise,
// exposure flicker, a person fidgeting) becomes "still" within ~2 s, and walking leaves "still" quickly.
import { test } from "node:test";
import assert from "node:assert/strict";
import { ActivityDetector, measureMotion, phaseShift } from "../src/activity";

const W = 160, H = 120;
let seed = 7;   // mulberry32: uniform in [-1, 1)
const noise = () => { seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return (((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5) * 2; };
// a fixed broadband texture (like a real scene): random values on 1, 3 and 7 px lattices, bilinearly interpolated, so it
// can be sampled at any sub-pixel position
const N = 512, fine = Float32Array.from({ length: N * N }, () => noise()), coarse = Float32Array.from({ length: N * N }, () => noise());
const lattice = (g: Float32Array, x: number, y: number) => {
  const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi, at = (a: number, b: number) => g[(((b % N) + N) % N) * N + (((a % N) + N) % N)];
  return (1 - fy) * ((1 - fx) * at(xi, yi) + fx * at(xi + 1, yi)) + fy * ((1 - fx) * at(xi, yi + 1) + fx * at(xi + 1, yi + 1));
};
const grain = Float32Array.from({ length: N * N }, () => noise());
// plus a fixed 1 px grain: real scenes have sharp edges at every scale (a smooth texture under sensor noise lets anything
// with a sharp edge, such as a person moving, dominate the phase correlation)
const tex = (x: number, y: number) => 128 + 45 * lattice(fine, x / 3 + 50, y / 3 + 50) + 60 * lattice(coarse, x / 7 + 20, y / 7 + 20) + 25 * lattice(grain, x + 9, y + 9);
/** the texture moved by (dx, dy) px and zoomed by `zoom` about the centre, plus sensor noise and a brightness offset */
function frame(dx = 0, dy = 0, zoom = 1, sigma = 1.5, gain = 0, blob?: [number, number]): Float32Array {
  const f = new Float32Array(W * H), cx = W / 2, cy = H / 2;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = tex(cx + (x - cx) / zoom - dx, cy + (y - cy) / zoom - dy) + gain + sigma * noise();
    // a moving dark blob (a person fidgeting in front of a desk webcam): local motion only
    if (blob && (x - blob[0]) ** 2 + (y - blob[1]) ** 2 < 15 ** 2) v = 40 + sigma * noise();
    f[y * W + x] = Math.max(0, Math.min(255, v));
  }
  return f;
}

test("phase correlation finds a known shift (content moved right 3 px, up 2 px)", () => {
  const [sx, sy, r] = phaseShift(frame(), frame(3, -2), W, 16, 28, 128, 64);
  assert.ok(Math.abs(sx - 3) < 0.5, `dx ${sx}`);
  assert.ok(Math.abs(sy + 2) < 0.5, `dy ${sy}`);
  assert.ok(r > 0.2, `response ${r}`);
});

test("a zoom (walking forward) reads as expansion; a pan as a shift without expansion", () => {
  const z = measureMotion(frame(), frame(0, 0, 1.06), W, H, 0.2);
  assert.ok(z.forward > 0.1, `forward ${z.forward}`);
  const p = measureMotion(frame(), frame(4, 0), W, H, 0.2);
  assert.ok(p.dx > 0.1, `dx ${p.dx}`);
  assert.ok(Math.abs(p.forward) < 0.1, `forward ${p.forward}`);
  assert.ok(p.flow > 1.2, `flow ${p.flow}`);
});

test("a still camera becomes 'still' within ~2 s (noise, exposure flicker, a fidgeting person do not move it)", () => {
  const a = new ActivityDetector();
  let stillAt: number | null = null;
  for (let k = 0; k <= 50; k++) {
    const t = k * 0.1;
    const g = frame(0, 0, 1, 1, 6 * Math.sin(3 * t), [80 + 6 * Math.sin(5 * t), 70]);
    if (a.update(g, W, H, t) === "still" && stillAt == null) stillAt = t;
  }
  assert.ok(stillAt != null && stillAt <= 2.5, `still at ${stillAt}`);
  assert.equal(a.activity, "still");
});

test("walking leaves 'still' within about a second, and is recognised as walking", () => {
  const a = new ActivityDetector();
  for (let k = 0; k <= 30; k++) a.update(frame(), W, H, k * 0.1);
  assert.equal(a.activity, "still");
  let leftAt: number | null = null, x = 0;
  for (let k = 31; k <= 60; k++) {
    const t = k * 0.1;
    x += 2.5;   // the scene slides and bobs at every step, as with a hand-held phone
    const g = frame(x % 60, 4 * Math.sin(2 * Math.PI * 1.8 * t), 1 + 0.002 * (k - 30));
    if (a.update(g, W, H, t) !== "still" && leftAt == null) leftAt = t;
  }
  assert.ok(leftAt != null && leftAt - 3.0 <= 1.2, `left still at ${leftAt}`);
  assert.equal(a.activity, "walking");
});

test("a featureless, noisy scene (a white wall) is still, not random motion", () => {
  const a = new ActivityDetector();
  for (let k = 0; k <= 40; k++) a.update(Float32Array.from({ length: W * H }, () => 200 + 1.5 * noise()), W, H, k * 0.1);
  assert.equal(a.activity, "still", a.describe());
});

test("no frames for a while: back to 'unclear'", () => {
  const a = new ActivityDetector();
  for (let k = 0; k <= 30; k++) a.update(frame(), W, H, k * 0.1);
  assert.equal(a.activity, "still");
  a.update(frame(), W, H, 3.2); a.update(frame(), W, H, 6.0);   // a 2.8 s gap (stalled camera)
  a.update(frame(), W, H, 6.1);
  assert.equal(a.activity, "unclear");
});
