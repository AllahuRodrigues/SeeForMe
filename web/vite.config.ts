import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";
import { VitePWA } from "vite-plugin-pwa";

const ISOLATION = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
/** ONNX Runtime version: its wasm files keep the same names across releases, so their cache is named after the version
 *  (a new ORT never runs with the old wasm; main.ts deletes the caches of earlier versions). */
const ORT_VERSION: string = JSON.parse(readFileSync(resolve(__dirname, "node_modules/onnxruntime-web/package.json"), "utf8")).version;
const ORT_CACHE = `seeforme-ort-${ORT_VERSION}`;

/** Dev-only: serve ONNX Runtime's wasm/mjs loader files raw from node_modules.
 *  ORT imports them dynamically; Vite's transform pipeline refuses module imports of /public files,
 *  so this middleware answers /ort/* before Vite sees the request. In production the same files are
 *  copied from public/ort/ as-is. */
function serveOrt(): Plugin {
  const dist = resolve(__dirname, "node_modules/onnxruntime-web/dist");
  const types: Record<string, string> = { ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm" };
  return {
    name: "seeforme-serve-ort",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0];
        if (!url.startsWith("/ort/")) return next();
        const file = resolve(dist, url.slice(5));
        if (!file.startsWith(dist) || !existsSync(file)) return next();
        const ext = file.slice(file.lastIndexOf("."));
        res.setHeader("Content-Type", types[ext] ?? "application/octet-stream");
        res.setHeader("Content-Length", statSync(file).size);
        res.setHeader("Cache-Control", "public, max-age=3600");
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  server: { port: 5173, host: true, headers: ISOLATION },
  preview: { headers: ISOLATION },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
  define: { __SEEFORME_ORT_CACHE__: JSON.stringify(ORT_CACHE) },
  plugins: [
    serveOrt(),
    VitePWA({
      // A release replaces everything it changed: every precached file carries a content revision, so a new
      // scheduler_rl.onnx / scheduler.json / pcacs.json / yolo11n.onnx is downloaded once and the old copy dropped
      // (cleanupOutdatedCaches). The new worker activates at once (skipWaiting + clientsClaim); main.ts registers it,
      // reloads the page when it takes over (after the camera is stopped) and says "Updated to version X" once.
      registerType: "autoUpdate",
      injectRegister: false,
      includeAssets: ["icons/icon.svg", "icons/icon-192.png", "icons/icon-512.png"],
      manifest: false,
      workbox: {
        // json: config/pcacs.json, config/scheduler.json, data/benchmark.json; onnx: models/*; mjs: the ORT loaders in ort/
        globPatterns: ["**/*.{js,mjs,css,html,svg,png,webmanifest,json,onnx}"],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        // models are requested as /models/x.onnx?v=<sha256 prefix> (cache-busting without a service worker); the
        // precache ignores that parameter and serves the revision of this release
        ignoreURLParametersMatching: [/^utm_/, /^fbclid$/, /^v$/],
        skipWaiting: true,
        clientsClaim: true,
        cleanupOutdatedCaches: true,
        // the 14-28 MB ORT wasm builds are cached on first use, per ORT version
        runtimeCaching: [{ urlPattern: /\/ort\/.*\.wasm$/, handler: "CacheFirst", options: { cacheName: ORT_CACHE, cacheableResponse: { statuses: [200] } } }],
      },
    }),
  ],
});
