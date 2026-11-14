// The camera's controls in the web app: the Start / Stop button and Space, the Flip button and F, the camera picker, and
// the voice commands "start", "pause", "goodbye", "flip camera" and "front camera" / "back camera". main.ts gives it the
// camera (camera.ts CameraSource) and a few hooks into the app; web/test/camcontrol.test.ts runs it on a fake camera device.
//
// Every path that opens a camera goes through CameraSource.start() (flip() calls it too), whose generation number cancels
// an open that a newer open, a stop or a video file overtook: the cancelled open rejects with CameraCancelled and closes
// its stream, and this module says nothing about it (no "Camera error", no "Camera on"). On top of that it keeps:
//   - the session, which every stop and video file ends: a start's later steps (the server check, "Camera on", the side
//     it asked for) happen only while no stop or video file came since it began, so nothing says "Camera on" after
//     "Paused. The camera is off." or "Goodbye" (a second start can only come after a stop, so two starts never share
//     a session);
//   - the open in progress, a start (the camera is off) or a switch (it runs). "Start", the Start button or Space while a
//     start is in progress adopt it: one getUserMedia, one "Camera on". A voice flip while an open is in progress opens
//     nothing and says which open it is waiting for. A pause, goodbye or Stop ends both. A start after a pause that cancelled
//     an open still waiting for permission opens again, and CameraSource shares the request that still waits (no second
//     prompt), so the camera comes on as soon as the user allows it.
import { isCancelled, type CameraSource } from "./camera";
import { facingOfSide, flipPlan, sideOfFacing, switchedReply, type CameraSide } from "./voice";

/** What the controls need from the camera (CameraSource). */
export type CameraDevice = Pick<CameraSource, "stream" | "facing" | "start" | "flip" | "stop" | "listDevices">;
type Settings = MediaTrackSettings | undefined;

/** What the controls need from the app (web/src/main.ts). */
export interface CameraApp {
  /** Does the camera (or a video file) run (state.running)? */
  running(): boolean;
  /** Let speech play (Speech.unlock): it works inside a user gesture (a button, a key). */
  unlock(): void;
  /** A start's camera is open: the app runs on it (state, the view's shape, "Camera on" in the status, the loop). */
  begin(s: Settings): void;
  /** Another camera's picture after a switch: what was seen and said before is about the old view. */
  newView(s: Settings): void;
  /** The server check after a start (it may take seconds). */
  checkServer(): Promise<void>;
  /** The start is complete and still current: "Server connected", and "Camera on. I'll tell you what matters." spoken. */
  announce(): void;
  /** The cameras the browser lists, for the camera picker. */
  devices(list: MediaDeviceInfo[]): void;
  /** The camera stopped (the module has closed it): the app stops, its speech stops, and the status says `status`. */
  end(status: string): void;
  status(text: string): void;
  say(text: string): void;
  /** Say `text` once the app has finished what it is saying, if `current()` still holds then. */
  sayAfter(text: string, current: () => boolean): void;
}

export type PendingOpen = "start" | "switch";
type How = "start" | "switch" | "next" | "pick";

export class CameraControl {
  private session = 0;                                        // advanced by every stop and video file
  private opening: { kind: PendingOpen; id: number } | null = null;
  private opens = 0;
  private cancelArmed = 0;                                    // the open a first Start press adopted (a second cancels it)
  // a side asked for by voice, and the track that answered it (another camera is another track: no need to forget it)
  private asked: { side: CameraSide; track: MediaStreamTrack } | null = null;
  constructor(private cam: CameraDevice, private app: CameraApp) {}

  /** The open in progress, null when none. */
  get pending(): PendingOpen | null { return this.opening?.kind ?? null; }
  /** The side the running camera reports (its track's facingMode), null when it does not say (most laptop webcams). */
  private side(): CameraSide | null { return sideOfFacing(this.track()?.getSettings().facingMode); }

  // ---- buttons and keys --------------------------------------------------------------------------------------------
  /** The Start / Stop button, and Space. */
  toggle(): void {
    if (this.app.running()) { this.stop("Camera stopped."); return; }
    this.app.unlock();
    if (this.pending === "start" && this.opening) {
      // the first press adopts the open in progress (no second camera); a second press on the same open cancels it, so a
      // keyboard user is never stuck behind a permission prompt they cannot find
      if (this.cancelArmed === this.opening.id) { this.stop("Camera stopped."); return; }
      this.cancelArmed = this.opening.id;
      this.app.status("Still starting the camera. If the browser asks, allow camera access. Press Start again to cancel.");
    } else void this.open("start", null, false);
  }
  /** The Flip button and F (the voice "flip camera" too): the next camera, replacing an open in progress. */
  next(byVoice = false): void { if (this.app.running()) void this.open("next", null, byVoice); }
  /** The camera picker: that camera, replacing an open in progress. */
  pick(deviceId: string): void { if (this.app.running()) void this.open("pick", null, false, deviceId); }
  /** Close the camera and cancel an open in progress; the status says `status`. */
  stop(status: string): void {
    this.session++; this.opening = null;
    this.cam.stop();
    this.app.end(status);
  }
  /** A video file replaces the camera (CameraSource.playFile closes it): an open in progress and a start's later steps end. */
  useFile(): void { this.session++; this.opening = null; }

  // ---- voice ---------------------------------------------------------------------------------------------------------
  /** "Start". */
  start(): void {
    if (this.app.running()) this.app.say("Already on.");
    else if (this.pending === "start") this.app.say("Still starting the camera.");
    else void this.open("start", null, true);
  }
  /** "Pause": in the web app the camera stops, also one still opening. */
  pause(): void {
    const msg = "Paused. The camera is off.";
    if (!this.app.running() && !this.pending) { this.app.say("The camera is already off."); return; }
    this.stop(msg);
    this.app.say(msg);
  }
  /** "Goodbye": the camera stops, also one still opening (main.ts says the goodbye). */
  quit(): void { if (this.app.running() || this.pending) this.stop("Stopped."); }
  /** "Flip camera" (side null: the next camera), "front camera" or "back camera" (voice.ts flipPlan). */
  flip(side: string | null): void {
    const s: CameraSide | null = side === "front" || side === "back" ? side : null;
    const plan = flipPlan(s, this.app.running(), this.side(), this.askedSide(), this.pending);
    if (plan.say) this.app.say(plan.say);
    if (plan.act === "next") this.next(true);
    else if ((plan.act === "switch" || plan.act === "start") && s) void this.open(plan.act, s, true);
  }

  // ---- opening ---------------------------------------------------------------------------------------------------------
  private track(): MediaStreamTrack | null { return this.cam.stream?.getVideoTracks()[0] ?? null; }
  /** The side a voice command asked for on the camera that runs now, null when none did. */
  private askedSide(): CameraSide | null {
    const tr = this.track();
    return this.asked && tr && this.asked.track === tr ? this.asked.side : null;
  }
  /** The one path that opens a camera. `how`: "start" (the camera is off; `side` null: as last time), "switch" (to `side`),
   *  "next" (CameraSource.flip) or "pick" (`deviceId`). After a start or switch to a side, the reply names the side the
   *  browser reports (switchedReply), never the side asked for. */
  private async open(how: How, side: CameraSide | null, byVoice: boolean, deviceId = ""): Promise<void> {
    const kind: PendingOpen = how === "start" ? "start" : "switch";
    const id = ++this.opens;
    this.opening = { kind, id };
    if (kind === "start") this.app.unlock();
    const session = this.session;
    let s: Settings;
    try {
      s = how === "next" ? await this.cam.flip()
        : how === "pick" ? await this.cam.start({ deviceId })
        : await this.cam.start({ facing: side ? facingOfSide(side) : this.cam.facing });
    } catch (e) {
      if (this.opening?.id === id) this.opening = null;
      if (isCancelled(e)) return;            // a pause, goodbye, Stop, video file or newer open came first: nothing to say
      if (kind === "start") {
        const why = String((e as Error)?.message ?? e).split("\n")[0];
        this.app.status(`Camera error: ${why}. Use HTTPS or localhost and allow the camera.`);
        if (byVoice) this.app.say("I couldn't start the camera. Press Start.");
      } else {
        // the old camera was closed before the new one was asked for: the camera is off now, and that is said
        const msg = "I couldn't switch the camera. The camera is off.";
        this.stop(msg);
        this.app.say(msg);
      }
      return;
    }
    // resolved: nothing stopped or replaced the camera while it opened (CameraSource would have cancelled it)
    if (this.opening?.id === id) this.opening = null;
    const track = this.track();
    const got = sideOfFacing(s?.facingMode);
    if (side && track) this.asked = { side, track };
    if (kind === "switch") {
      this.app.newView(s);
      if (side) this.app.say(switchedReply(side, got));
      return;
    }
    this.app.begin(s);
    await this.app.checkServer();
    // stopped while the server answered (Stop, "pause", "goodbye", a video file): "Camera on" would be false
    const current = () => this.session === session;
    if (!current()) return;
    this.app.announce();
    // after "Camera on ..." (say would cut it), and only about this camera
    if (side) this.app.sayAfter(switchedReply(side, got, true), () => current() && this.track() === track);
    this.app.devices(await this.cam.listDevices());
  }
}
