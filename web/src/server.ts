// Optional Python server (server/app.py): metric depth, OCR, ElevenLabs TTS. The app works without it.
import type { Detection } from "./types";

export class ServerClient {
  available = false;
  depthModel = "";
  constructor(public baseUrl: string) {}

  async check(): Promise<boolean> {
    try {
      const r = await fetch(`${this.baseUrl}/api/health`, { signal: AbortSignal.timeout(2500) });
      if (!r.ok) throw new Error();
      const j = await r.json(); this.available = true; this.depthModel = j.depth ?? ""; return true;
    } catch { this.available = false; return false; }
  }

  private async frameBlob(canvas: HTMLCanvasElement, maxW = 384): Promise<Blob> {
    const scale = Math.min(1, maxW / canvas.width);
    const c = document.createElement("canvas"); c.width = Math.round(canvas.width * scale); c.height = Math.round(canvas.height * scale);
    c.getContext("2d")!.drawImage(canvas, 0, 0, c.width, c.height);
    return new Promise((res) => c.toBlob((b) => res(b!), "image/jpeg", 0.7));
  }

  /** Fills distance_m / depth_conf on the detections from the server's depth model. */
  async depth(canvas: HTMLCanvasElement, dets: Detection[], outdoor = false): Promise<boolean> {
    if (!this.available || !dets.length) return false;
    const fd = new FormData();
    fd.append("image", await this.frameBlob(canvas), "f.jpg");
    fd.append("boxes", JSON.stringify(dets.map((d) => d.bbox)));
    fd.append("outdoor", String(outdoor));
    const r = await fetch(`${this.baseUrl}/api/depth`, { method: "POST", body: fd, signal: AbortSignal.timeout(1500) });
    if (!r.ok) return false;
    const j = await r.json() as { distances: Array<[number | null, number]> };
    j.distances.forEach(([z, c], i) => { if (z != null) { dets[i].distance_m = z; dets[i].depth_conf = c; } });
    return true;
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
