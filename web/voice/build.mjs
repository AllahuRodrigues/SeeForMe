#!/usr/bin/env node
// Records the voice pack: every clip key of web/src/voicepack.ts, spoken by Kokoro-82M, trimmed, laid end to end in one
// sprite file with an index of where each clip starts and how long it is.
//
//   cd web/voice && npm install && node build.mjs [--voice af_heart] [--speed 1.0] [--only 20]
//
// Output: web/public/voice/<voice>/sprite.mp3 and index.json. Raw clips are cached in web/voice/.cache so a rerun only
// records what is new. Needs ffmpeg on PATH. The model (about 330 MB) is downloaded from Hugging Face on first use.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const here = dirname(fileURLToPath(import.meta.url));
const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : dflt; };
const VOICE = arg("voice", "af_heart"), SPEED = Number(arg("speed", "1.0")), ONLY = Number(arg("only", "0"));
const RATE = 24000, PAD_S = 0.2, LEAD_S = 0.02, TAIL_S = 0.04, FLOOR = 0.004;

// the pack's contents come from the app's own source, so the recording can never drift from what the app says
const esbuild = createRequire(resolve(here, "../package.json"))("esbuild");
const bundle = resolve(here, ".cache/voicepack.mjs");
mkdirSync(dirname(bundle), { recursive: true });
await esbuild.build({ entryPoints: [resolve(here, "../src/voicepack.ts")], bundle: true, format: "esm", platform: "node", outfile: bundle, logLevel: "warning" });
const { packKeys, clipText } = await import(pathToFileURL(bundle).href);
const keys = ONLY ? packKeys().slice(0, ONLY) : packKeys();

const cacheDir = resolve(here, ".cache", `${VOICE}_${SPEED}`);
mkdirSync(cacheDir, { recursive: true });
const cacheFile = (key) => resolve(cacheDir, createHash("sha1").update(clipText(key)).digest("hex").slice(0, 16) + ".f32");

/** Cut the silence before and after the speech, keep a short margin, and fade the edges. */
function trim(pcm) {
  let a = 0, b = pcm.length - 1;
  while (a < b && Math.abs(pcm[a]) < FLOOR) a++;
  while (b > a && Math.abs(pcm[b]) < FLOOR) b--;
  a = Math.max(0, a - Math.round(LEAD_S * RATE)); b = Math.min(pcm.length - 1, b + Math.round(TAIL_S * RATE));
  const out = pcm.slice(a, b + 1), fade = Math.min(Math.round(0.005 * RATE), out.length >> 1);
  for (let i = 0; i < fade; i++) { out[i] *= i / fade; out[out.length - 1 - i] *= i / fade; }
  return out;
}

const todo = keys.filter((k) => !existsSync(cacheFile(k)));
console.log(`${keys.length} clips, ${todo.length} to record with ${VOICE} at speed ${SPEED}`);
if (todo.length) {
  const { KokoroTTS } = await import("kokoro-js");
  const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", { dtype: "fp32", device: "cpu" });
  const t0 = Date.now();
  for (const [i, key] of todo.entries()) {
    const audio = await tts.generate(clipText(key), { voice: VOICE, speed: SPEED });
    if (audio.sampling_rate !== RATE) throw new Error(`unexpected sampling rate ${audio.sampling_rate}`);
    const pcm = trim(Float32Array.from(audio.audio));
    writeFileSync(cacheFile(key), Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength));
    if ((i + 1) % 25 === 0 || i === todo.length - 1) console.log(`  ${i + 1}/${todo.length} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  }
}

// lay the clips end to end with silence between them
const clips = {}, parts = [];
let at = PAD_S;
parts.push(new Float32Array(Math.round(PAD_S * RATE)));
for (const key of keys) {
  const buf = readFileSync(cacheFile(key));
  const pcm = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  clips[key] = [Number(at.toFixed(4)), Number((pcm.length / RATE).toFixed(4))];
  parts.push(pcm, new Float32Array(Math.round(PAD_S * RATE)));
  at += pcm.length / RATE + PAD_S;
}
const total = parts.reduce((n, p) => n + p.length, 0), all = new Float32Array(total);
let off = 0; for (const p of parts) { all.set(p, off); off += p.length; }

const outDir = resolve(here, "../public/voice", VOICE);
mkdirSync(outDir, { recursive: true });
const raw = resolve(here, ".cache/sprite.f32");
writeFileSync(raw, Buffer.from(all.buffer));
const ff = spawnSync("ffmpeg", ["-v", "error", "-y", "-f", "f32le", "-ar", String(RATE), "-ac", "1", "-i", raw,
  "-c:a", "libmp3lame", "-b:a", "48k", "-ar", String(RATE), "-ac", "1", resolve(outDir, "sprite.mp3")], { stdio: "inherit" });
if (ff.status !== 0) throw new Error("ffmpeg failed");
rmSync(raw);
writeFileSync(resolve(outDir, "index.json"), JSON.stringify({ voice: VOICE, speed: SPEED, model: "Kokoro-82M v1.0 (Apache-2.0)", sprite: "sprite.mp3", clips }));
const speech = Object.values(clips).reduce((s, c) => s + c[1], 0);
console.log(`wrote ${outDir}: ${keys.length} clips, ${speech.toFixed(0)} s of speech, sprite ${(total / RATE).toFixed(0)} s`);
