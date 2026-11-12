// Optional Python server (server/app.py): metric depth (and the cane's obstacle summary), OCR, ElevenLabs TTS. The app
// works without it.
import { parseCane } from "./cane";
import type { BBox, CaneSummary, Detection } from "./types";

/** One /api/depth answer: per requested box [metres or null, confidence], and the cane summary (null: no depth map). */
export interface DepthReply { distances: Array<[number | null, number]>; cane: CaneSummary | null }

export class ServerClient {
  available = false;
  depthModel = "";
  tts = false;   // /api/health: ElevenLabs is usable (the server has an API key)
  constructor(public baseUrl: string) {}

  async check(): Promise<boolean> {
    try {
      const r = await fetch(`${this.baseUrl}/api/health`, { signal: AbortSignal.timeout(2500) });
      if (!r.ok) throw new Error();
      const j = await r.json(); this.available = true; this.depthModel = j.depth ?? ""; this.tts = j.tts === true; return true;
    } catch { this.available = false; this.tts = false; return false; }
  }

  private async frameBlob(canvas: HTMLCanvasElement, maxW = 384): Promise<Blob> {
    // the frame is copied synchronously (before the first await), so the caller may draw the next frame at once
    const scale = Math.min(1, maxW / canvas.width);
    const c = document.createElement("canvas"); c.width = Math.round(canvas.width * scale); c.height = Math.round(canvas.height * scale);
    c.getContext("2d")!.drawImage(canvas, 0, 0, c.width, c.height);
    return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("frame encoding failed"))), "image/jpeg", 0.7));
  }

  /** Metric depth for the boxes of this frame (an empty list still returns the cane summary). null: not answered. */
  async depth(canvas: HTMLCanvasElement, boxes: BBox[], outdoor = false): Promise<DepthReply | null> {
    if (!this.available) return null;
    const fd = new FormData();
    fd.append("image", await this.frameBlob(canvas), "f.jpg");
    fd.append("boxes", JSON.stringify(boxes));
    fd.append("outdoor", String(outdoor));
    const r = await fetch(`${this.baseUrl}/api/depth`, { method: "POST", body: fd, signal: AbortSignal.timeout(1500) });
    if (!r.ok) return null;
    const j = await r.json() as { distances?: Array<[number | null, number]>; cane?: unknown };
    const distances = Array.isArray(j.distances) && j.distances.length === boxes.length ? j.distances : boxes.map((): [null, number] => [null, 0]);
    return { distances, cane: parseCane(j.cane) };
  }

  async ocr(canvas: HTMLCanvasElement): Promise<Detection[]> {
    if (!this.available) return [];
    const fd = new FormData(); fd.append("image", await this.frameBlob(canvas, 800), "f.jpg");
    const r = await fetch(`${this.baseUrl}/api/ocr`, { method: "POST", body: fd, signal: AbortSignal.timeout(8000) });
    if (!r.ok) return [];
    const j = await r.json() as { texts: Array<{ text: string; conf: number; bbox: [number, number, number, number] }> };
    return j.texts.map((t) => ({ label: "text", kind: "text" as const, text: t.text, confidence: t.conf, bbox: t.bbox, depth_conf: 0 }));
  }
}
