// A fake camera device for the web tests (camera.test.ts, camcontrol.test.ts): the real CameraSource runs on a fake
// navigator.mediaDevices and a fake <video>, and getUserMedia, the device list and video.play() answer only when the test
// says so, as a permission prompt or a slow camera would. Every stream it hands out is kept, so a test can check that no
// camera is left on (its light, which a blind user cannot see).
import { CameraSource, type MediaDevicesLike } from "../src/camera";

export type FakeFacing = "user" | "environment" | null;
export interface FakeCamera { deviceId: string; facing: FakeFacing; label?: string }
interface Waiting<T> { ok: (v: T) => void; fail: (e: Error) => void }

/** Let every promise that can run on now run (setImmediate comes after all pending microtasks). */
export const settle = () => new Promise<void>((r) => setImmediate(r));

/** The default device: a phone with a back and a front camera that report their sides. */
export const PHONE: FakeCamera[] = [{ deviceId: "back", facing: "environment" }, { deviceId: "front", facing: "user" }];
/** A laptop with one webcam that does not say which way it faces. */
export const LAPTOP: FakeCamera[] = [{ deviceId: "webcam", facing: null }];

export function fakeDevice(cameras: FakeCamera[] = PHONE) {
  const streams: Array<{ id: number; camera: string; live: boolean }> = [];
  const gum: Array<Waiting<MediaStream> & { constraints: MediaStreamConstraints }> = [];
  const lists: Array<Waiting<MediaDeviceInfo[]>> = [];
  const plays: Array<Waiting<void>> = [];
  let holdPlay = false, holdList = false, calls = 0;

  /** The stream the browser would give for these constraints: the exact device, else the side asked for, else the first. */
  const open = (c: MediaStreamConstraints): MediaStream => {
    const v = (typeof c.video === "object" ? c.video : {}) as MediaTrackConstraints;
    const exact = (v.deviceId as ConstrainDOMStringParameters | undefined)?.exact;
    const ideal = (v.facingMode as ConstrainDOMStringParameters | undefined)?.ideal;
    const cam = cameras.find((k) => k.deviceId === exact) ?? cameras.find((k) => ideal && k.facing === ideal) ?? cameras[0];
    const rec = { id: streams.length + 1, camera: cam.deviceId, live: true };
    streams.push(rec);
    const track = {
      stop: () => { rec.live = false; },
      getSettings: (): MediaTrackSettings => ({ deviceId: cam.deviceId, width: 1280, height: 720, ...(cam.facing ? { facingMode: cam.facing } : {}) }),
    };
    return { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
  };
  const media = {
    getUserMedia: (c: MediaStreamConstraints) => {
      calls++;
      return new Promise<MediaStream>((ok, fail) => gum.push({ constraints: c, ok: (v) => ok(v), fail }));
    },
    enumerateDevices: () => {
      const list = cameras.map((k, i) => ({ kind: "videoinput", deviceId: k.deviceId, label: k.label ?? `camera ${i + 1}`, groupId: "" }) as MediaDeviceInfo);
      return holdList ? new Promise<MediaDeviceInfo[]>((ok, fail) => lists.push({ ok, fail })).then(() => list) : Promise.resolve(list);
    },
  } as unknown as MediaDevicesLike;
  const video = {
    srcObject: null as MediaStream | null, src: "", loop: false,
    play: () => (holdPlay ? new Promise<void>((ok, fail) => plays.push({ ok, fail })) : Promise.resolve()),
  } as unknown as HTMLVideoElement;
  const cam = new CameraSource(video, () => media);

  return {
    cam, video,
    /** getUserMedia answers its oldest waiting call (`i`: another one): the user allowed the camera. */
    async allow(i = 0) { const [g] = gum.splice(i, 1); if (!g) throw new Error("no getUserMedia call is waiting"); g.ok(open(g.constraints)); await settle(); },
    /** getUserMedia fails for its oldest waiting call (denied, or no such camera). */
    async deny(message = "Permission denied", name = "NotAllowedError", i = 0) {
      const [g] = gum.splice(i, 1); if (!g) throw new Error("no getUserMedia call is waiting");
      g.fail(Object.assign(new Error(message), { name })); await settle();
    },
    /** From now on video.play() waits for play() / failPlay(). */
    holdPlay() { holdPlay = true; },
    async play() { const [p] = plays.splice(0, 1); if (!p) throw new Error("no play() is waiting"); p.ok(); await settle(); },
    async failPlay(message = "The play() request was interrupted by a new load request.") {
      const [p] = plays.splice(0, 1); if (!p) throw new Error("no play() is waiting");
      p.fail(Object.assign(new Error(message), { name: "AbortError" })); await settle();
    },
    /** From now on the device list waits for list(). */
    holdList() { holdList = true; },
    async list() { const [l] = lists.splice(0, 1); if (!l) throw new Error("no device list is waiting"); l.ok([]); await settle(); },
    /** The ids of the streams still live (1 is the first stream handed out). */
    live: () => streams.filter((s) => s.live).map((s) => s.id),
    /** Which camera each live stream is from. */
    liveCameras: () => streams.filter((s) => s.live).map((s) => s.camera),
    /** How many times getUserMedia was called, and how many calls still wait. */
    calls: () => calls,
    waiting: () => gum.length,
    constraints: (i = 0) => gum[i]?.constraints,
  };
}
