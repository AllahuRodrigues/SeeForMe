// Web Worker: runs YOLO11n with onnxruntime-web (WebGPU when available, WASM otherwise).
// ORT is loaded as a raw, self-contained bundle from /ort/ (copied from node_modules, see public/ort/) instead of
// being bundled: Vite's dev transform injects its HMR client into node_modules code, which breaks inside a worker,
// and this way dev and production execute the identical runtime bytes.
import type * as OrtNs from "onnxruntime-web";
import { COCO, INPUT, decode, letterbox } from "./yolo";

type Ort = typeof OrtNs;
let ort: Ort;
let session: OrtNs.InferenceSession | null = null;
let backend = "";

const stage = (s: string) => (self as any).postMessage({ type: "stage", stage: s });

const withTimeout = <T,>(p: Promise<T>, ms: number, what: string) =>
  Promise.race([p, new Promise<T>((_, rej) => setTimeout(() => rej(new Error(`${what} timed out after ${ms} ms`)), ms))]);

async function init(modelUrl: string, threads?: number) {
  stage("loading runtime");
  // import() through Function so Vite neither rewrites it nor injects its HMR client into the worker (dev mode)
  const rawImport = new Function("u", "return import(u)") as (u: string) => Promise<Ort>;
  ort = await rawImport(new URL("/ort/ort.webgpu.bundle.min.mjs", self.location.origin).href);
  stage("runtime loaded");
  ort.env.wasm.wasmPaths = "/ort/";
  // threaded wasm needs crossOriginIsolated; under the Vite dev server the thread spawn hangs, so dev stays single-threaded
  const auto = (self as any).crossOriginIsolated && !import.meta.env.DEV ? Math.min(4, (self.navigator?.hardwareConcurrency ?? 2)) : 1;
  ort.env.wasm.numThreads = threads ?? auto;
  stage(`threads=${ort.env.wasm.numThreads}`);
  const tryEp = async (eps: string[]) => ort.InferenceSession.create(modelUrl, { executionProviders: eps as any, graphOptimizationLevel: "all" });
  try {
    if ((self.navigator as any).gpu) { stage("trying webgpu"); session = await withTimeout(tryEp(["webgpu"]), 15000, "webgpu"); backend = "webgpu"; }
  } catch (e) { stage(`webgpu unavailable (${String((e as Error).message).slice(0, 60)})`); session = null; }
  if (!session) { stage("creating wasm session"); session = await tryEp(["wasm"]); backend = "wasm"; }
  stage("session ready, warming up");
  // warm-up
  const z = new ort.Tensor("float32", new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]);
  await session.run({ [session.inputNames[0]]: z });
  (self as any).postMessage({ type: "ready", backend });
}

async function detect(id: number, img: ImageData, conf: number) {
  if (!session) throw new Error("detector session not ready");
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

// Always answer a detect request: a rejected session.run must not leave the main thread waiting forever.
self.onmessage = (ev: MessageEvent) => {
  const m = ev.data;
  if (m.type === "init") init(m.modelUrl, m.threads).catch((e) => (self as any).postMessage({ type: "error", message: String(e?.stack ?? e) }));
  else if (m.type === "detect") detect(m.id, m.img, m.conf ?? 0.25).catch((e) =>
    (self as any).postMessage({ type: "result", id: m.id, dets: [], ms: 0, backend, error: String(e?.message ?? e) }));
};
