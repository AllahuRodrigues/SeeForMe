// Pre/post-processing for YOLO11 ONNX (input 1x3x640x640, output 1x84x8400).
export const COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

export const INPUT = 640;

export interface RawBox { x1: number; y1: number; x2: number; y2: number; score: number; cls: number }

/** Letterbox an RGBA ImageData into a Float32 CHW tensor; returns the scale/pad so boxes map back. */
export function letterbox(img: ImageData): { data: Float32Array; scale: number; padX: number; padY: number } {
  const { width: w, height: h, data: px } = img;
  const scale = Math.min(INPUT / w, INPUT / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const padX = Math.floor((INPUT - nw) / 2), padY = Math.floor((INPUT - nh) / 2);
  const out = new Float32Array(3 * INPUT * INPUT).fill(114 / 255);
  const plane = INPUT * INPUT;
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h - 1, Math.floor(y / scale));
    for (let x = 0; x < nw; x++) {
      const sx = Math.min(w - 1, Math.floor(x / scale));
      const si = (sy * w + sx) * 4;
      const di = (y + padY) * INPUT + (x + padX);
      out[di] = px[si] / 255; out[plane + di] = px[si + 1] / 255; out[2 * plane + di] = px[si + 2] / 255;
    }
  }
  return { data: out, scale, padX, padY };
}

function iou(a: RawBox, b: RawBox): number {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1), ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** Decode 1x84x8400 output (cx,cy,w,h + 80 scores) with class-wise NMS. */
export function decode(out: Float32Array, conf = 0.25, iouThr = 0.45, maxDet = 50): RawBox[] {
  const n = 8400, nc = 80;
  const cands: RawBox[] = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) {
      const s = out[(4 + c) * n + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < conf) continue;
    const cx = out[i], cy = out[n + i], w = out[2 * n + i], h = out[3 * n + i];
    cands.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep: RawBox[] = [];
  for (const c of cands) {
    if (keep.length >= maxDet) break;
    if (keep.some((k) => k.cls === c.cls && iou(k, c) > iouThr)) continue;
    keep.push(c);
  }
  return keep;
}
