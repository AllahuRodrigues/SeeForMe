// The camera's controls (web/src/camcontrol.ts) with the real CameraSource on a fake camera device (fakecam.ts) and a
// fake app that records what is said and shown: the Start button and Space, Flip, the camera picker, a video file, and
// the voice commands start, pause, goodbye, flip camera and front / back camera, in every overlap the review found
// (scenarios A to F of its race harness) and the ones next to them. A blind user cannot see the camera light, so each
// case checks that no stream stays live after "Paused. The camera is off." or "Goodbye", that a cancelled open says
// nothing ("Camera error", a second "Camera on"), and that the replies say what is true.
import { test } from "node:test";
import assert from "node:assert/strict";
import { CameraControl, type CameraApp } from "../src/camcontrol";
import { LAPTOP, PHONE, fakeDevice, settle, type FakeCamera } from "./fakecam";

const CAMERA_ON = "say: Camera on. I'll tell you what matters.";

/** The real controls on a fake device and a fake app. The app's server check and its "say after" wait for the test. */
function rig(cameras: FakeCamera[] = PHONE) {
  const dev = fakeDevice(cameras);
  const log: string[] = [];
  let running = false;
  const server: Array<() => void> = [];
  const later: Array<{ text: string; current: () => boolean }> = [];
  const app: CameraApp = {
    running: () => running,
    unlock: () => {},
    begin: () => { running = true; log.push("status: Camera on"); },
    newView: () => log.push("new view"),
    checkServer: () => new Promise<void>((r) => server.push(r)),
    announce: () => log.push(CAMERA_ON),
    devices: () => {},
    end: (status) => { running = false; log.push(`status: ${status}`); },
    status: (text) => log.push(`status: ${text}`),
    say: (text) => log.push(`say: ${text}`),
    sayAfter: (text, current) => later.push({ text, current }),
  };
  const ctl = new CameraControl(dev.cam, app);
  return {
    ...dev, ctl, log,
    running: () => running,
    /** A video file starts playing (main.ts file input). */
    async file() { ctl.useFile(); await dev.cam.playFile(new File([new Uint8Array(4)], "walk.mp4")); running = true; await settle(); },
    /** The server check answers (`speech`: then the app's speech ends, and what waited to be said is said). */
    async server(speech = true) {
      const r = server.shift(); if (!r) throw new Error("no server check is waiting");
      r(); await settle();
      if (speech) this.speechEnds();
    },
    /** The app finished speaking: a sayAfter whose `current()` still holds (and the camera runs) is said now. */
    speechEnds() { for (const l of later.splice(0)) if (running && l.current()) log.push(`say: ${l.text}`); },
    /** The camera runs: Start, the permission, the server. */
    async started() { ctl.toggle(); await settle(); await dev.allow(); await this.server(); log.length = 0; },
    count: (line: string) => log.filter((l) => l === line).length,
    errors: () => log.filter((l) => l.startsWith("status: Camera error")),
  };
}

test("A: 'flip camera' then 'pause' before getUserMedia answers: the camera stays off", async () => {
  const r = rig(); await r.started();
  r.ctl.flip(null);
  await settle();
  assert.equal(r.waiting(), 1);
  r.ctl.pause();
  await r.allow();
  assert.equal(r.running(), false);
  assert.deepEqual(r.live(), []);
  assert.deepEqual(r.log, ["say: Switching camera.", "status: Paused. The camera is off.", "say: Paused. The camera is off."]);
  assert.equal(r.ctl.pending, null);
});

test("B: 'flip camera', then 'front camera' while it opens, then 'pause': one switch, no stream left on", async () => {
  const r = rig(); await r.started();
  r.ctl.flip(null); await settle();
  r.ctl.flip("front");
  assert.equal(r.log.at(-1), "say: Still switching the camera.");
  await r.allow();
  assert.equal(r.calls(), 2);                                 // the start, and one switch
  assert.deepEqual(r.liveCameras(), ["front"]);
  r.ctl.pause();
  assert.deepEqual(r.live(), []);
  // the Flip button while a switch opens: the newer open wins, and only its stream stays on
  const b = rig(); await b.started();
  b.ctl.next(); await settle();
  b.ctl.next(); await settle();
  await b.allow(); await b.allow();
  assert.equal(b.live().length, 1);
  b.ctl.toggle();                                              // Stop
  assert.deepEqual(b.live(), []);
});

test("C: 'pause' while video.play() waits: no 'Camera error', no 'Camera on', no stream left on", async () => {
  for (const play of ["resolves", "rejects"] as const) {
    const r = rig();
    r.holdPlay();
    r.ctl.start(); await settle();
    await r.allow();
    r.ctl.pause();
    if (play === "resolves") await r.play(); else await r.failPlay();
    assert.deepEqual(r.errors(), [], play);
    assert.equal(r.count(CAMERA_ON), 0, play);
    assert.deepEqual([r.running(), r.live()], [false, []], play);
    assert.deepEqual(r.log, ["status: Paused. The camera is off.", "say: Paused. The camera is off."], play);
  }
});

test("D: the Start button while the permission prompt is open, then 'start': one camera, one 'Camera on'", async () => {
  const r = rig();
  r.ctl.toggle(); await settle();
  r.ctl.start();
  assert.equal(r.log.at(-1), "say: Still starting the camera.");
  await settle();
  assert.equal(r.calls(), 1);
  await r.allow(); await r.server();
  assert.equal(r.count(CAMERA_ON), 1);
  assert.deepEqual(r.live(), [1]);
  r.ctl.pause();
  assert.deepEqual([r.running(), r.live()], [false, []]);
});

test("E: 'start', 'pause' while the permission prompt is open, then 'start' again: no lock; the camera comes on once allowed", async () => {
  const r = rig();
  r.ctl.start(); await settle();
  r.ctl.pause();
  r.ctl.start();                                               // opens again, sharing the request that still waits
  r.ctl.start();
  r.ctl.flip("front");
  await settle();
  assert.deepEqual(r.log, ["status: Paused. The camera is off.", "say: Paused. The camera is off.",
    "say: Still starting the camera.", "say: Still starting the camera."]);
  assert.equal(r.ctl.pending, "start");
  assert.equal(r.calls(), 1);                                  // no second permission prompt
  await r.allow(); await r.server();
  assert.equal(r.count(CAMERA_ON), 1);
  assert.deepEqual([r.running(), r.live()], [true, [1]]);
  // a pause during the prompt and no new start: the stream is closed the moment it arrives
  const p = rig();
  p.ctl.start(); await settle();
  p.ctl.pause();
  await p.allow();
  assert.deepEqual([p.running(), p.live(), p.count(CAMERA_ON)], [false, [], 0]);
});

test("F: 'start', then the Start button while it opens: one camera, one 'Camera on'", async () => {
  const r = rig();
  r.ctl.start(); await settle();
  r.ctl.toggle();
  assert.equal(r.log.at(-1), "status: Still starting the camera. If the browser asks, allow camera access. Press Start again to cancel.");
  await settle();
  assert.equal(r.calls(), 1);
  await r.allow(); await r.server();
  assert.equal(r.count(CAMERA_ON), 1);
  r.ctl.pause();
  assert.deepEqual([r.running(), r.live()], [false, []]);
});

test("a pause or Stop while the server check waits: the camera is off and nothing says 'Camera on'", async () => {
  const r = rig();
  r.ctl.start(); await settle(); await r.allow();
  assert.equal(r.running(), true);
  r.ctl.pause();
  await r.server();
  assert.equal(r.count(CAMERA_ON), 0);
  assert.deepEqual([r.running(), r.live()], [false, []]);
  // pause and start again before the first server check answers: one "Camera on", from the second start
  const s = rig();
  s.ctl.start(); await settle(); await s.allow();
  s.ctl.pause();
  s.ctl.start(); await settle(); await s.allow();
  await s.server();                                            // the first start's check: its session is over
  assert.equal(s.count(CAMERA_ON), 0);
  await s.server();
  assert.equal(s.count(CAMERA_ON), 1);
  assert.deepEqual(s.live(), [2]);
});

test("goodbye while a camera opens (a start or a switch to a side) leaves it off, and no side is said", async () => {
  const r = rig();
  r.ctl.flip("front");                                         // the camera is off: starts it on that side
  r.ctl.quit();
  await r.allow();
  assert.deepEqual([r.running(), r.live()], [false, []]);
  assert.deepEqual(r.log, ["status: Stopped."]);
  const s = rig(); await s.started();
  s.ctl.flip("front"); await settle();
  s.ctl.quit();
  await s.allow();
  assert.deepEqual([s.running(), s.live()], [false, []]);
  assert.deepEqual(s.log, ["status: Stopped."]);
  // goodbye with nothing on or opening says nothing and stops nothing
  const t = rig();
  t.ctl.quit();
  assert.deepEqual(t.log, []);
});

test("front camera with the camera off starts it, then names only the side the browser reports, after 'Camera on'", async () => {
  const r = rig();
  r.ctl.flip("front");
  assert.equal(r.ctl.pending, "start");
  await settle(); await r.allow(); await r.server();
  assert.deepEqual(r.log, ["status: Camera on", CAMERA_ON, "say: This is the front camera."]);
  assert.deepEqual(r.liveCameras(), ["front"]);
  // paused after "Camera on" but before the side was said: the side is not said
  const p = rig();
  p.ctl.flip("front"); await settle(); await p.allow(); await p.server(false);
  p.ctl.pause();
  p.speechEnds();
  assert.equal(p.log.at(-1), "say: Paused. The camera is off.");
  // paused and started again before the side was said: that side belongs to the old start, and is not said
  const q = rig();
  q.ctl.flip("front"); await settle(); await q.allow(); await q.server(false);
  q.ctl.pause();
  q.ctl.start(); await settle(); await q.allow(); await q.server(false);
  q.speechEnds();
  assert.equal(q.log.at(-1), CAMERA_ON);
  assert.ok(!q.log.includes("say: This is the front camera."), q.log.join(" | "));
  // a laptop camera that does not say which way it faces: asked again, it is not reopened
  const l = rig(LAPTOP);
  l.ctl.flip("back"); await settle(); await l.allow(); await l.server();
  assert.equal(l.log.at(-1), "say: I asked for the back camera, but this camera does not say which way it faces.");
  l.ctl.flip("back");
  assert.equal(l.calls(), 1);
  assert.equal(l.log.at(-1), "say: I asked for the back camera, but this camera does not say which way it faces.");
});

test("front / back camera while the camera runs: already on that side does nothing; otherwise that side, as reported", async () => {
  const r = rig(); await r.started();
  r.ctl.flip("back");
  assert.deepEqual(r.log, ["say: Already on the back camera."]);
  r.ctl.flip("front"); await settle();
  assert.equal(r.ctl.pending, "switch");
  r.ctl.start();
  assert.equal(r.log.at(-1), "say: Already on.");
  await r.allow();
  assert.deepEqual(r.log.slice(-2), ["new view", "say: Switched to the front camera."]);
  assert.deepEqual(r.liveCameras(), ["front"]);
  // a phone without a front camera
  const b = rig([{ deviceId: "back", facing: "environment" }]); await b.started();
  b.ctl.flip("front"); await settle(); await b.allow();
  assert.equal(b.log.at(-1), "say: I couldn't find a front camera. This is the back camera.");
  b.ctl.flip("front");
  assert.equal(b.calls(), 2);                                  // the same request on the same camera: not reopened
});

test("the camera picker and the Flip button while a voice switch opens: the newer open wins, and no side is claimed", async () => {
  const r = rig(); await r.started();
  r.ctl.flip("front"); await settle();
  r.ctl.pick("back"); await settle();
  await r.allow(); await r.allow();
  assert.deepEqual(r.liveCameras(), ["back"]);
  assert.ok(!r.log.some((l) => l.includes("front camera")), r.log.join(" | "));
  assert.equal(r.ctl.pending, null);
  // the picker or Flip with the camera off does nothing
  const off = rig();
  off.ctl.pick("front"); off.ctl.next(); await settle();
  assert.equal(off.calls(), 0);
});

test("a video file while a start opens: no 'Camera on', no 'Camera error', and the voice no longer waits for that start", async () => {
  const r = rig();
  r.ctl.start(); await settle();
  await r.file();
  assert.equal(r.ctl.pending, null);
  r.ctl.flip("front");                                         // the video runs: a camera of that side replaces it
  assert.notEqual(r.log.at(-1), "say: Still starting the camera.");
  r.ctl.pause();
  await r.allow(); await r.allow();
  assert.deepEqual(r.live(), []);
  assert.deepEqual([r.count(CAMERA_ON), r.errors().length, r.ctl.pending], [0, 0, null]);
  // a video file while the server check waits: the camera's "Camera on" is not said over the video
  const v = rig();
  v.ctl.start(); await settle(); await v.allow();
  await v.file();
  await v.server();
  assert.equal(v.count(CAMERA_ON), 0);
  v.ctl.start();
  assert.equal(v.log.at(-1), "say: Already on.");             // the video runs
});

test("start, pause and goodbye with nothing opening; failures stop the camera and say so", async () => {
  const r = rig();
  r.ctl.pause();
  assert.deepEqual(r.log, ["say: The camera is already off."]);
  r.ctl.start(); await settle(); await r.allow(); await r.server();
  assert.equal(r.log.at(-1), CAMERA_ON);                       // a plain start names no side
  r.ctl.start();
  assert.equal(r.log.at(-1), "say: Already on.");
  r.ctl.flip(null);
  assert.equal(r.log.at(-1), "say: Switching camera.");
  await settle(); await r.allow();
  assert.equal(r.log.at(-1), "new view");
  // a switch that fails: the old camera is already closed, so the camera is off, and that is said
  r.ctl.flip("back"); await settle();
  await r.deny("Could not start video source", "NotReadableError");
  await r.deny("Could not start video source", "NotReadableError");
  assert.deepEqual(r.log.slice(-2), ["status: I couldn't switch the camera. The camera is off.", "say: I couldn't switch the camera. The camera is off."]);
  assert.deepEqual([r.running(), r.live()], [false, []]);
  // a start that fails says why, and by voice says what to do
  const s = rig();
  s.ctl.start(); await settle();
  await s.deny();                                             // denied: no second prompt
  assert.equal(s.calls(), 1);
  assert.deepEqual(s.log, ["status: Camera error: Permission denied. Use HTTPS or localhost and allow the camera.",
    "say: I couldn't start the camera. Press Start."]);
  assert.equal(s.ctl.pending, null);
});

test("the Start button twice while the permission prompt is open cancels the open; nothing is left running", async () => {
  const r = rig();
  r.ctl.toggle(); await settle();
  r.ctl.toggle();                                              // adopts the open, says how to cancel
  assert.equal(r.ctl.pending, "start");
  r.ctl.toggle();                                              // cancels it
  assert.equal(r.ctl.pending, null);
  assert.equal(r.log.at(-1), "status: Camera stopped.");
  await r.allow();
  assert.deepEqual([r.running(), r.live(), r.count(CAMERA_ON)], [false, [], 0]);
  // a later start opens normally
  r.ctl.toggle(); await settle(); await r.allow(); await r.server();
  assert.deepEqual([r.running(), r.live().length, r.count(CAMERA_ON)], [true, 1, 1]);
});
