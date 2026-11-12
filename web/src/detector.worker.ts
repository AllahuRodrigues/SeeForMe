// Web Worker: runs YOLO11 with onnxruntime-web. With WebGPU it runs YOLO11s (recall of the directly mapped COCO classes
// 0.85 against 0.80 for YOLO11n on the BLV-Road-Nav keyframes at the same precision, docs/live_demo.md); on WASM it runs
// YOLO11n, which keeps the frame rate usable on a phone CPU. Both have the same input (1x3x640x640) and output (1x84x8400).
// ORT is loaded as a raw, self-contained bundle from /ort/ (copied from node_modules, see public/ort/) instead of
// being bundled: Vite's dev transform injects its HMR client into node_modules code, which breaks inside a worker,
// and this way dev and production execute the identical runtime bytes.
import type * as OrtNs from "onnxruntime-web";
import { COCO, INPUT, decode, letterbox } from "./yolo";

type Ort = typeof OrtNs;
let ort: Ort;
let session: OrtNs.InferenceSession | null = null;
let backend = "";
let model = "";

const stage = (s: string) => (self as any).postMessage({ type: "stage", stage: s });
const modelName = (url: string) => (/yolo11s/.test(url) ? "YOLO11s" : /yolo11n/.test(url) ? "YOLO11n" : url.split("/").pop()?.split("?")[0] ?? "model");

const withTimeout = <T,>(p: Promise<T>, ms: number, what: string) =>
  Promise.race([p, new Promise<T>((_, rej) => setTimeout(() => rej(new Error(`${what} timed out after ${ms} ms`)), ms))]);

/** gpuModel: the model for WebGPU (YOLO11s); cpuModel: the WASM model (YOLO11n). wasmOnly: skip WebGPU (a fresh worker
 *  after WebGPU failed); threads: WASM threads (default: up to 4 when cross-origin isolated). */
async function init(gpuModel: string, cpuModel: string, threads?: number, wasmOnly = false) {
  stage("loading runtime");
  // import() through Function so Vite neither rewrites it nor injects its HMR client into the worker (dev mode)
  const rawImport = new Function("u", "return import(u)") as (u: string) => Promise<Ort>;
  ort = await rawImport(new URL("/ort/ort.webgpu.bundle.min.mjs", self.location.origin).href);
  stage("runtime loaded");
  ort.env.wasm.wasmPaths = "/ort/";
  const create = (url: string, eps: string[]) => ort.InferenceSession.create(url, { executionProviders: eps as any, graphOptimizationLevel: "all" });
  const gpu = (self.navigator as any).gpu;
  // ask for an adapter first: a browser can expose navigator.gpu without a usable GPU, and the larger model would be
  // downloaded for nothing
  const adapter = !wasmOnly && threads !== 1 && gpu ? await withTimeout(gpu.requestAdapter() as Promise<unknown>, 5000, "webgpu adapter").catch(() => null) : null;
  if (adapter) {
    // The GPU runs the model; WASM only handles a few CPU-side ops. The threaded WASM build hangs WebGPU session creation
    // inside a worker (measured in Chrome on an M1: 4 threads never finish, 1 thread compiles YOLO11n in 0.3 s), so one
    // thread. WASM can be initialised once per runtime, so if WebGPU fails a fresh worker runs threaded WASM instead.
    ort.env.wasm.numThreads = 1;
    stage(`trying webgpu (${modelName(gpuModel)})`);
    try { session = await withTimeout(create(gpuModel, ["webgpu"]), 40000, "webgpu"); backend = "webgpu"; model = modelName(gpuModel); }
    catch (e) { throw new Error(`webgpu: ${String((e as Error)?.message ?? e).slice(0, 80)}`); }
  } else {
    // threaded wasm needs crossOriginIsolated; under the Vite dev server the thread spawn hangs, so dev stays single-threaded
    const auto = (self as any).crossOriginIsolated && !import.meta.env.DEV ? Math.min(4, (self.navigator?.hardwareConcurrency ?? 2)) : 1;
    ort.env.wasm.numThreads = threads ?? auto;
    stage(`creating wasm session (${modelName(cpuModel)}, threads=${ort.env.wasm.numThreads})`);
    session = await create(cpuModel, ["wasm"]); backend = "wasm"; model = modelName(cpuModel);
  }
  stage("session ready, warming up");
  const z = new ort.Tensor("float32", new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]);
  await session.run({ [session.inputNames[0]]: z });
  (self as any).postMessage({ type: "ready", backend, model });
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
  if (m.type === "init") init(m.gpuModel ?? m.modelUrl, m.cpuModel ?? m.modelUrl, m.threads, !!m.wasmOnly).catch((e) => (self as any).postMessage({ type: "error", message: String(e?.stack ?? e) }));
  else if (m.type === "detect") detect(m.id, m.img, m.conf ?? 0.25).catch((e) =>
    (self as any).postMessage({ type: "result", id: m.id, dets: [], ms: 0, backend, error: String(e?.message ?? e) }));
};
