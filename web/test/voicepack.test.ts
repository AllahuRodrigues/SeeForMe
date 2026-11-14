// The recorded voice pack (web/src/voicepack.ts): every sentence the scheduler, the gate and the cane layer can compose
// must map onto recorded clips, word for word, and anything the pack cannot say must be handed to another voice whole
// (never half recorded, half synthetic).
import { test } from "node:test";
import assert from "node:assert/strict";
import { COCO } from "../src/yolo";
import { SLOTS, compose } from "../src/compose";
import { caneMessage } from "../src/cane";
import type { CaneObstacle } from "../src/types";
import { FIXED_PHRASES, clipText, packKeys, planClips, planSeconds, type VoiceIndex } from "../src/voicepack";

const KEYS = new Set(packKeys());
const has = (k: string) => KEYS.has(k);
const keysOf = (msg: string) => planClips(msg, has)?.map((c) => c.key) ?? null;

test("a cue is planned fragment by fragment, the last one with a falling end", () => {
  assert.deepEqual(keysOf("Car, about 3 meters, slightly left."), ["m:car", "m:about 3 meters", "f:slightly left"]);
  assert.deepEqual(keysOf("Person."), ["f:person"]);
  assert.deepEqual(keysOf("Possibly a bicycle, close, on your right, approaching."),
    ["m:possibly a bicycle", "m:close", "m:on your right", "f:approaching"]);
});

test("a second sentence starts after a longer pause", () => {
  const plan = planClips("Stop. Obstacle ahead, about 1 meter.", has)!;
  assert.deepEqual(plan.map((c) => c.key), ["f:stop", "m:obstacle ahead", "f:about 1 meter"]);
  assert.ok(plan[0].gapAfterS > plan[1].gapAfterS && plan[1].gapAfterS > 0 && plan[2].gapAfterS === 0);
});

test("every sentence the composer can produce is in the pack", () => {
  const angles = [-60, -20, -5, 0, 8, 15, 30], dists = [null, 0.4, 0.9, 1.2, 1.74, 2.3, 2.9, 3.4, 7.5, 12, 26.4, 39.6];
  const motions: Array<[number, number, number]> = [[0, 0, 0], [-2, 0, 0], [0, 0.6, -1], [0, -0.6, 1], [0, 0.6, 1], [0, -0.6, -1]];
  let n = 0;
  for (const label of [...COCO, "obstacle", "unknown object", "curb"]) for (const hedged of [false, true]) for (const units of ["m", "ft"] as const)
    for (const depthConf of [0.3, 0.9]) for (const distance of dists) for (const angle of angles) for (const [relSpeed, lateralSpeed, lateral] of motions) {
      const msg = compose(SLOTS, { label, kind: "object", distance, depthConf, angle, relSpeed, lateralSpeed, lateral, hedged, units });
      if (planClips(msg, has) == null) assert.fail(`not in the pack: ${msg}`);
      n++;
    }
  assert.ok(n > 100_000);
});

test("every cane and guide-dog sentence is in the pack", () => {
  for (const level of ["ground", "body", "head"] as const) for (const sector of ["left", "ahead", "right"] as const)
    for (const blocked of [null, "left", "right", "stop"] as const) for (const units of ["m", "ft"] as const) for (const d of [0.5, 1, 1.6, 2.4, 3.2, 6]) {
      const msg = caneMessage({ sector, level, distance_m: d } as CaneObstacle, blocked, units);
      if (planClips(msg, has) == null) assert.fail(`not in the pack: ${msg}`);
    }
});

test("fixed app sentences are one recorded clip each", () => {
  assert.ok(FIXED_PHRASES.includes("Camera on. I'll tell you what matters.") && FIXED_PHRASES.includes("Paused. The camera is off."));
  for (const p of FIXED_PHRASES) assert.deepEqual(keysOf(p), [`s:${p}`]);
});

test("what the pack cannot say goes to another voice whole", () => {
  assert.equal(keysOf("Sign reads Exit, ahead."), null);
  assert.equal(keysOf("I can see two chairs and a table."), null);
  assert.equal(keysOf("Car, about 3 meters, near the blue door."), null);
  assert.equal(keysOf(""), null);
});

test("the text recorded for a clip carries the punctuation its position needs", () => {
  assert.equal(clipText("m:car"), "Car,");
  assert.equal(clipText("f:slightly left"), "Slightly left.");
  assert.equal(clipText("s:Camera on. I'll tell you what matters."), "Camera on. I'll tell you what matters.");
});

test("a plan's length is the clips plus the pauses between them", () => {
  const index: VoiceIndex = { voice: "x", sprite: "x.mp3", clips: { "m:car": [0, 0.5], "m:about 3 meters": [1, 0.9], "f:slightly left": [3, 0.7] } };
  const plan = planClips("Car, about 3 meters, slightly left.", (k) => k in index.clips)!;
  const gaps = plan.reduce((a, c) => a + c.gapAfterS, 0);
  assert.ok(Math.abs(planSeconds(plan, index) - (0.5 + 0.9 + 0.7 + gaps)) < 1e-9);
});
