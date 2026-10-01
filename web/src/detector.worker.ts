// Web Worker: runs YOLO11n with onnxruntime-web (WebGPU when available, WASM otherwise).
import * as ort from "onnxruntime-web/webgpu";
import { COCO, INPUT, decode, letterbox } from "./yolo";

let session: ort.InferenceSession | null = null;
let backend = "";

async function init(modelUrl: string) {
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = (self as any).crossOriginIsolated ? Math.min(4, (self.navigator?.hardwareConcurrency ?? 2)) : 1;
  const tryEp = async (eps: string[]) => ort.InferenceSession.create(modelUrl, { executionProviders: eps as any, graphOptimizationLevel: "all" });
  try {
    if ((self.navigator as any).gpu) { session = await tryEp(["webgpu"]); backend = "webgpu"; }
  } catch (e) { session = null; }
  if (!session) { session = await tryEp(["wasm"]); backend = "wasm"; }
  // warm-up
  const z = new ort.Tensor("float32", new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]);
  await session.run({ [session.inputNames[0]]: z });
  (self as any).postMessage({ type: "ready", backend });
}

async function detect(id: number, img: ImageData, conf: number) {
  if (!session) return;
  const t0 = performance.now();
  const { data, scale, padX, padY } = letterbox(img);
  const input = new ort.Tensor("float32", data, [1, 3, INPUT, INPUT]);
  const res = await session.run({ [session.inputNames[0]]: input });
  const out = res[session.outputNames[0]].data as Float32Array;
  const boxes = decode(out, conf);
  const dets = boxes.map((b) => ({
    label: COCO[b.cls], confidence: b.score, kind: "object",
    bbox: [
      Math.max(0, (b.x1 - padX) / scale / img.width), Math.max(0, (b.y1 - padY) / scale / img.height),
      Math.min(1, (b.x2 - padX) / scale / img.width), Math.min(1, (b.y2 - padY) / scale / img.height),
    ], depth_conf: 0,
  }));
  (self as any).postMessage({ type: "result", id, dets, ms: performance.now() - t0, backend });
}

self.onmessage = (ev: MessageEvent) => {
  const m = ev.data;
  if (m.type === "init") init(m.modelUrl);
  else if (m.type === "detect") detect(m.id, m.img, m.conf ?? 0.25);
};
