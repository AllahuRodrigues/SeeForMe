// One camera abstraction for phone/tablet front & rear cameras, laptop webcams, external cams and video files.
// The picture is never mirrored, for any camera: the preview shows the frames as the camera delivers them, boxes are
// drawn where the detector found the objects, and "left" in speech is the left of the picture.
//
// Opening a camera takes time (getUserMedia, the permission prompt the first time, video.play()), and a blind user cannot
// see the camera light. So every start() and stop() advances a generation number, and an open whose number is no longer
// current when getUserMedia or play() answers closes the stream it got and rejects with CameraCancelled: a newer open, a
// stop or a video file always wins, and no stream is left running unseen. A start() while an earlier one still waits for
// getUserMedia with the same constraints shares that request instead of asking again (no second permission prompt), and
// the newest open takes the stream. web/test/camera.test.ts checks each of these.
export type Facing = "environment" | "user";

/** What a camera open rejects with when a newer open, a stop() or a video file came first: nothing went wrong, and nothing
 *  should be said about it. */
export class CameraCancelled extends Error {
  constructor() { super("The camera open was cancelled."); this.name = "CameraCancelled"; }
}
export const isCancelled = (e: unknown): e is CameraCancelled => e instanceof CameraCancelled;

/** The part of navigator.mediaDevices the camera uses (a fake one in the web tests). */
export type MediaDevicesLike = Pick<MediaDevices, "getUserMedia" | "enumerateDevices">;
/** A getUserMedia request in flight; `owner` is the newest open waiting for it. */
interface Request { key: string; stream: Promise<MediaStream>; owner: number }

const stopTracks = (s: MediaStream) => s.getTracks().forEach((t) => t.stop());
/** getUserMedia errors after which any camera is tried instead of the one asked for (no such side or device, or busy). */
const RETRY_ANY = new Set(["OverconstrainedError", "ConstraintNotSatisfiedError", "NotFoundError", "NotReadableError"]);

export class CameraSource {
  stream: MediaStream | null = null;
  facing: Facing = "environment";
  deviceId = "";
  private gen = 0;                        // advanced by every start() and stop(): an open whose number is old is cancelled
  /** What the open in progress asked for: a flip during it goes on from there (two quick flips come back), while a stop
   *  clears it, so a cancelled open never decides which camera the next plain start opens. */
  private want: { facing: Facing; deviceId: string } | null = null;
  private request: Request | null = null;
  constructor(public video: HTMLVideoElement, private media: () => MediaDevicesLike = () => navigator.mediaDevices) {}

  async listDevices(): Promise<MediaDeviceInfo[]> {
    try { return (await this.media().enumerateDevices()).filter((d) => d.kind === "videoinput"); } catch { return []; }
  }

  /** Open a camera, closing the one that runs and cancelling an open in progress. Rejects with CameraCancelled when a
   *  newer start(), a stop() or playFile() came before it finished (the stream it got is closed then); on any other
   *  failure the camera is closed too. */
  async start(opts: { facing?: Facing; deviceId?: string } = {}) {
    this.stop();
    const gen = this.gen;
    // facing and deviceId change only once this open succeeds (until then `want` holds them for a flip)
    const facing = opts.facing ?? this.facing, deviceId = opts.deviceId ?? "";
    this.want = { facing, deviceId };
    const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    const video: MediaTrackConstraints = deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode: { ideal: facing } };
    const key = JSON.stringify(video);
    let req = this.request;
    if (req?.key === key) req.owner = gen;   // the same request still waits (a permission prompt): share it
    else {
      const media = this.media();
      const r = { key, owner: gen } as Request;
      r.stream = media.getUserMedia({ video, audio: false })
        // laptops have no "environment" camera: retry with any camera, unless nobody waits for this request any more. Only
        // when the constraints could not be met: after a denied or dismissed prompt (NotAllowedError) a retry would ask again
        .catch((e) => {
          if (this.gen !== r.owner || !RETRY_ANY.has((e as Error)?.name)) throw e;
          return media.getUserMedia({ video: base, audio: false });
        });
      const done = () => { if (this.request === r) this.request = null; };
      r.stream.then(done, done);
      req = this.request = r;
    }
    let stream: MediaStream;
    try { stream = await req.stream; } catch (e) { if (this.gen !== gen) throw new CameraCancelled(); throw e; }
    if (this.gen !== gen) {
      // a newer open that shares this request takes the stream; otherwise nobody wants it, and it is closed now
      if (req.owner !== this.gen) stopTracks(stream);
      throw new CameraCancelled();
    }
    this.stream = stream;
    this.video.srcObject = stream; this.video.src = "";
    try { await this.video.play(); } catch (e) {
      if (this.gen !== gen) throw new CameraCancelled();   // the stop() or newer start() that came closed this stream
      this.stop();                                         // the picture will not play: no camera left on unseen
      throw e;
    }
    if (this.gen !== gen) throw new CameraCancelled();
    const s = stream.getVideoTracks()[0]?.getSettings();
    this.facing = (s?.facingMode as Facing | undefined) ?? facing; this.deviceId = deviceId; this.want = null;
    return s;
  }

  /** The next camera: the next device in the list when one was picked, otherwise the other side. */
  async flip() {
    const gen = this.gen;
    const from = this.want ?? { facing: this.facing, deviceId: this.deviceId };   // an open in progress counts
    const devices = await this.listDevices();
    if (this.gen !== gen) throw new CameraCancelled();   // stopped, or another camera opened, while the list came
    if (devices.length > 1 && from.deviceId) {
      const i = devices.findIndex((d) => d.deviceId === from.deviceId);
      return this.start({ deviceId: devices[(i + 1) % devices.length].deviceId });
    }
    return this.start({ facing: from.facing === "environment" ? "user" : "environment" });
  }

  playFile(file: File) {
    this.stop();
    this.video.srcObject = null; this.video.src = URL.createObjectURL(file); this.video.loop = true;
    return this.video.play();
  }

  /** Close the camera and cancel an open in progress (its stream is closed when it arrives). */
  stop() { this.gen++; this.want = null; this.stream?.getTracks().forEach((t) => t.stop()); this.stream = null; }
}
