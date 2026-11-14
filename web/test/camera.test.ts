// CameraSource (web/src/camera.ts) on a fake camera device: overlapping opens, a stop while getUserMedia or video.play()
// waits, and a flip while the device list comes. Before the generation number, a second start() overwrote this.stream and
// left the first stream running, and a stop() while play() waited made start() throw a TypeError that the app announced as
// "Camera error". Each case here failed on that code.
import { test } from "node:test";
import assert from "node:assert/strict";
import { CameraCancelled, isCancelled } from "../src/camera";
import { LAPTOP, fakeDevice, settle } from "./fakecam";

/** The outcome of a camera promise once it settles: "ok", "cancelled" or the error's message. */
function outcome<T>(p: Promise<T>): { result: () => string } {
  let r = "pending";
  p.then(() => { r = "ok"; }, (e) => { r = isCancelled(e) ? "cancelled" : String((e as Error)?.message ?? e); });
  return { result: () => r };
}

test("camera: a second start while the first waits for getUserMedia closes the first stream; the newer open wins", async () => {
  const d = fakeDevice();
  const a = outcome(d.cam.start({ facing: "environment" }));
  const b = outcome(d.cam.start({ facing: "user" }));
  await settle();
  assert.equal(d.calls(), 2);
  await d.allow(0); await d.allow(0);                      // the first answers first
  assert.equal(a.result(), "cancelled");
  assert.equal(b.result(), "ok");
  assert.deepEqual(d.liveCameras(), ["front"]);
  assert.equal(d.cam.stream?.getVideoTracks()[0].getSettings().facingMode, "user");
  // and when the newer one answers first
  const e = fakeDevice();
  const x = outcome(e.cam.start({ facing: "environment" }));
  const y = outcome(e.cam.start({ facing: "user" }));
  await settle();
  await e.allow(1); await e.allow(0);
  assert.deepEqual([x.result(), y.result()], ["cancelled", "ok"]);
  assert.deepEqual(e.liveCameras(), ["front"]);
});

test("camera: stop while getUserMedia waits (the permission prompt) closes the stream when it arrives", async () => {
  const d = fakeDevice();
  const a = outcome(d.cam.start());
  await settle();
  d.cam.stop();
  await d.allow();
  assert.equal(a.result(), "cancelled");
  assert.deepEqual(d.live(), []);
  assert.equal(d.cam.stream, null);
  // a denial that comes after the stop is no error either: nobody waits for that camera any more
  const e = fakeDevice();
  const b = outcome(e.cam.start());
  await settle();
  e.cam.stop();
  await e.deny();
  assert.equal(b.result(), "cancelled");
  assert.equal(e.calls(), 1);                               // and no retry with "any camera"
});

test("camera: stop while video.play() waits rejects as cancelled, never a TypeError, and leaves no stream on", async () => {
  for (const ending of ["resolves", "rejects"] as const) {
    const d = fakeDevice();
    d.holdPlay();
    const a = outcome(d.cam.start());
    await settle();
    await d.allow();
    d.cam.stop();
    if (ending === "resolves") await d.play(); else await d.failPlay();
    assert.equal(a.result(), "cancelled", ending);
    assert.deepEqual(d.live(), [], ending);
  }
  // a newer start while play() waits: the first rejects as cancelled, its stream closed; the newer one runs
  const e = fakeDevice();
  e.holdPlay();
  const x = outcome(e.cam.start({ facing: "environment" }));
  await settle(); await e.allow();
  const y = outcome(e.cam.start({ facing: "user" }));
  await settle();
  await e.failPlay();
  await e.allow(); await e.play();
  assert.deepEqual([x.result(), y.result()], ["cancelled", "ok"]);
  assert.deepEqual(e.liveCameras(), ["front"]);
});

test("camera: a start while the same request still waits shares it (one permission prompt); the newest open takes the stream", async () => {
  const d = fakeDevice();
  const a = outcome(d.cam.start({ facing: "environment" }));
  await settle();
  d.cam.stop();                                             // "pause" while the prompt is open
  const b = outcome(d.cam.start({ facing: "environment" })); // "start" again
  await settle();
  assert.equal(d.calls(), 1);
  await d.allow();
  assert.deepEqual([a.result(), b.result()], ["cancelled", "ok"]);
  assert.deepEqual(d.live(), [1]);
  assert.ok(d.cam.stream);
  // shared, then stopped: nobody takes it, and it is closed when it arrives
  const e = fakeDevice();
  const x = outcome(e.cam.start()), y = outcome(e.cam.start());
  await settle();
  e.cam.stop();
  await e.allow();
  assert.deepEqual([x.result(), y.result(), e.calls()], ["cancelled", "cancelled", 1]);
  assert.deepEqual(e.live(), []);
  // a request for another side is not shared
  const f = fakeDevice();
  void outcome(f.cam.start({ facing: "environment" })); void outcome(f.cam.start({ facing: "user" }));
  await settle();
  assert.equal(f.calls(), 2);
});

test("camera: no camera of the side asked for: one retry with any camera, but not for an open that was cancelled", async () => {
  const d = fakeDevice(LAPTOP);
  const a = outcome(d.cam.start({ facing: "environment" }));
  await settle();
  await d.deny("Requested device not found", "OverconstrainedError");
  assert.equal(d.calls(), 2);
  await d.allow();
  assert.equal(a.result(), "ok");
  assert.deepEqual(d.live(), [1]);
  // the retry belongs to the open that waits now: shared after a stop, it still happens
  const e = fakeDevice(LAPTOP);
  const x = outcome(e.cam.start({ facing: "environment" }));
  await settle(); e.cam.stop();
  const y = outcome(e.cam.start({ facing: "environment" }));
  await settle();
  await e.deny("Requested device not found", "OverconstrainedError");
  assert.equal(e.calls(), 2);
  await e.allow();
  assert.deepEqual([x.result(), y.result()], ["cancelled", "ok"]);
});

test("camera: a failure that is not a cancellation is reported, and leaves no stream on", async () => {
  const d = fakeDevice();
  const a = outcome(d.cam.start());
  await settle();
  await d.deny("Permission denied");                       // no retry with any camera after a denied prompt
  assert.equal(a.result(), "Permission denied");
  assert.equal(d.calls(), 1);
  const e = fakeDevice();
  e.holdPlay();
  const b = outcome(e.cam.start());
  await settle(); await e.allow();
  await e.failPlay("play() failed");
  assert.equal(b.result(), "play() failed");
  assert.deepEqual(e.live(), []);
  assert.equal(e.cam.stream, null);
});

test("camera: flip while the device list comes is cancelled by a stop, and opens nothing", async () => {
  const d = fakeDevice();
  void outcome(d.cam.start()); await settle(); await d.allow();
  d.holdList();
  const f = outcome(d.cam.flip());
  await settle();
  d.cam.stop();
  await d.list();
  assert.equal(f.result(), "cancelled");
  assert.equal(d.calls(), 1);
  assert.deepEqual(d.live(), []);
  // without the stop it flips to the other side
  const e = fakeDevice();
  void outcome(e.cam.start({ facing: "environment" })); await settle(); await e.allow();
  const g = outcome(e.cam.flip());
  await settle(); await e.allow();
  assert.equal(g.result(), "ok");
  assert.deepEqual(e.liveCameras(), ["front"]);
});

test("camera: a video file cancels an open in progress", async () => {
  const d = fakeDevice();
  const a = outcome(d.cam.start());
  await settle();
  await d.cam.playFile(new File([new Uint8Array(4)], "walk.mp4", { type: "video/mp4" }));
  await d.allow();
  assert.equal(a.result(), "cancelled");
  assert.deepEqual(d.live(), []);
  assert.ok(new CameraCancelled() instanceof Error);
});

test("camera: a denied prompt is not followed by a second prompt; a missing side falls back to any camera", async () => {
  const d = fakeDevice();
  const a = outcome(d.cam.start({ facing: "environment" }));
  await settle(); await d.deny();
  assert.equal(a.result(), "Permission denied");
  assert.equal(d.calls(), 1);                                  // no retry after NotAllowedError
  const e = fakeDevice();
  const b = outcome(e.cam.start({ facing: "environment" }));
  await settle(); await e.deny("no such camera", "OverconstrainedError");
  await settle(); await e.allow();
  assert.equal(b.result(), "ok");
  assert.equal(e.calls(), 2);
});

test("camera: a cancelled open does not change which side the next plain start opens", async () => {
  const d = fakeDevice();
  void outcome(d.cam.start({ facing: "environment" })); await settle(); await d.allow();
  const f = outcome(d.cam.start({ facing: "user" }));         // "front camera" ...
  await settle();
  d.cam.stop();                                               // ... then "pause" before it opens
  await d.allow();
  assert.equal(f.result(), "cancelled");
  assert.equal(d.cam.facing, "environment");
  void outcome(d.cam.start()); await settle(); await d.allow();
  assert.deepEqual(d.liveCameras(), ["back"]);
});
