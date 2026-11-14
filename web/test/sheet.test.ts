// The physics behind the settings sheet (web/src/sheet.ts): a spring described by a damping ratio and a response time,
// momentum projection to decide where a flick is going, rubber-banding past the edge, and the release decision.
import { test } from "node:test";
import assert from "node:assert/strict";
import { project, releaseTarget, rubberband, springStep, velocityOf } from "../src/sheet";

function settle(x: number, v: number, target: number, damping: number, response: number) {
  let max = -Infinity, min = Infinity, t = 0;
  while (t < 3) { [x, v] = springStep(x, v, target, 1 / 120, damping, response); max = Math.max(max, x); min = Math.min(min, x); t += 1 / 120; }
  return { x, v, max, min };
}

test("a critically damped spring reaches its target without passing it", () => {
  const r = settle(400, 0, 0, 1.0, 0.35);
  assert.ok(Math.abs(r.x) < 0.01 && Math.abs(r.v) < 0.1);
  assert.ok(r.min > -0.01);                                  // never below the target: no overshoot
});

test("an under-damped spring overshoots a little and still settles", () => {
  const r = settle(400, 0, 0, 0.8, 0.3);
  assert.ok(r.min < -1 && r.min > -40);
  assert.ok(Math.abs(r.x) < 0.05);
});

test("a spring starts from the velocity it is handed, so a drag and its animation have no seam", () => {
  const [x] = springStep(100, 600, 100, 1 / 60, 1.0, 0.35);  // already at the target, still moving at 600 px/s
  assert.ok(x > 100 + 5);
});

test("a shorter response is a snappier spring", () => {
  const after = (response: number) => { let x = 300, v = 0; for (let i = 0; i < 12; i++) [x, v] = springStep(x, v, 0, 1 / 60, 1, response); return x; };
  assert.ok(after(0.25) < after(0.5));
});

test("momentum projection: a flick travels on, and faster flicks travel further", () => {
  assert.equal(project(0), 0);
  assert.ok(Math.abs(project(1000) - 499) < 1);              // Apple's exponential-decay form at 0.998
  assert.ok(project(2000) > project(1000) && project(-1000) < 0);
});

test("rubber-banding resists more the further past the edge the finger goes, and never exceeds the pull", () => {
  const a = rubberband(50, 600), b = rubberband(200, 600), c = rubberband(1000, 600);
  assert.ok(a > 0 && a < 50 && b > a && b < 200 && c > b && c < 600);
  assert.ok(b / 200 < a / 50);                               // progressive
  assert.equal(rubberband(0, 600), 0);
});

test("release: the sheet goes where the gesture was heading, not where the finger stopped", () => {
  assert.equal(releaseTarget(60, 1400, 500), "dismiss");     // barely moved, flicked down
  assert.equal(releaseTarget(300, -1400, 500), "open");      // dragged far, flicked back up
  assert.equal(releaseTarget(120, 0, 500), "open");          // let go near the top
  assert.equal(releaseTarget(320, 0, 500), "dismiss");       // let go past halfway
});

test("release velocity comes from the last samples of the drag, in pixels per second", () => {
  assert.equal(velocityOf([]), 0);
  assert.equal(velocityOf([[0, 0]]), 0);
  assert.ok(Math.abs(velocityOf([[0, 0], [16, 8], [32, 16], [48, 24]]) - 500) < 1);
  assert.equal(velocityOf([[0, 0], [16, 8], [400, 8]]), 0);   // the finger paused before lifting: no fling
  assert.equal(velocityOf([[0, 0], [1, 300], [2, 600]]), 0);  // samples inside one frame say nothing about speed
  assert.equal(velocityOf([[0, 0], [10, 400], [20, 800]]), 6000);   // capped at what a finger can do
});
