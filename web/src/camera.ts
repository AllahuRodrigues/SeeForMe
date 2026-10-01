// One camera abstraction for phone/tablet front & rear cameras, laptop webcams, external cams and video files.
export type Facing = "environment" | "user";

export class CameraSource {
  stream: MediaStream | null = null;
  facing: Facing = "environment";
  deviceId = "";
  constructor(public video: HTMLVideoElement) {}

  async listDevices(): Promise<MediaDeviceInfo[]> {
    try { return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput"); } catch { return []; }
  }

  async start(opts: { facing?: Facing; deviceId?: string } = {}) {
    this.stop();
    this.facing = opts.facing ?? this.facing; this.deviceId = opts.deviceId ?? "";
    const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    const video: MediaTrackConstraints = this.deviceId ? { ...base, deviceId: { exact: this.deviceId } } : { ...base, facingMode: { ideal: this.facing } };
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      // laptops have no "environment" camera: retry with any camera
      this.stream = await navigator.mediaDevices.getUserMedia({ video: base, audio: false });
    }
    this.video.srcObject = this.stream; this.video.src = "";
    await this.video.play();
    const s = this.stream.getVideoTracks()[0]?.getSettings();
    if (s?.facingMode) this.facing = s.facingMode as Facing;
    return s;
  }

  async flip() {
    const devices = await this.listDevices();
    if (devices.length > 1 && this.deviceId) {
      const i = devices.findIndex((d) => d.deviceId === this.deviceId);
      return this.start({ deviceId: devices[(i + 1) % devices.length].deviceId });
    }
    return this.start({ facing: this.facing === "environment" ? "user" : "environment" });
  }

  playFile(file: File) {
    this.stop();
    this.video.srcObject = null; this.video.src = URL.createObjectURL(file); this.video.loop = true;
    return this.video.play();
  }

  /** True when the picture is mirrored (front cameras), so left/right in cues must be swapped. */
  get mirrored() { return !!this.stream && this.facing === "user"; }

  stop() { this.stream?.getTracks().forEach((t) => t.stop()); this.stream = null; }
}
