/// <reference types="vite/client" />

/** Name of the service-worker cache that holds this build's ONNX Runtime wasm files (defined in vite.config.ts). */
declare const __SEEFORME_ORT_CACHE__: string;
/** Detector model URLs with their content hash (?v=), and the cache YOLO11s is kept in (vite.config.ts). gpu: run with
 *  WebGPU (YOLO11s when the file exists), cpu: the WASM model (YOLO11n). */
declare const __SEEFORME_MODELS__: { gpu: string; cpu: string; gpuCache: string };
