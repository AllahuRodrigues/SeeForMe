import { CameraSource } from "./camera";
import { PCACS, DEFAULT_CFG, type Candidate } from "./pcacs";
import { ServerClient } from "./server";
import { Speech, earcon, haptic } from "./speech";
import { Camera, Tracker } from "./tracker";
import type { Cue, Detection, Track } from "./types";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video"), overlay = $<HTMLCanvasElement>("overlay"), status = $("status");
const srLive = $("sr-live"), cueCard = $("cue-card"), cueKind = $("cue-kind"), cueText = $("cue-text"), cueMeta = $("cue-meta");
const lastCueEl = $("last-cue"), budgetFill = $("budget-fill"), budgetText = $("budget-text"), debugRows = $("debug-rows"), debugLog = $("debug-log");

const serverUrl = () => (location.port === "5173" ? `${location.protocol}//${location.hostname}:8000` : location.origin);
const server = new ServerClient(serverUrl());
const cam = new CameraSource(video);
const camModel = new Camera();
const tracker = new Tracker(camModel);
const policy = new PCACS({ ...DEFAULT_CFG, weights: { ...DEFAULT_CFG.weights } }, camModel);
fetch("/config/pcacs.json").then((r) => r.json()).then((c) => { Object.assign(policy.cfg, c, { weights: { ...policy.cfg.weights, ...c.weights }, verbosity: policy.cfg.verbosity, units: policy.cfg.units }); policy.budget = policy.cfg.budgetCapacity; }).catch(() => {});
const speech = new Speech(serverUrl, (text) => { srLive.textContent = text; });
const frameCanvas = document.createElement("canvas");
const fctx = frameCanvas.getContext("2d", { willReadFrequently: true })!;
const octx = overlay.getContext("2d")!;

type Mode = "navigate" | "find" | "read" | "ask";
const state = { mode: "navigate" as Mode, running: false, backend: "", detMs: 0, lastDecision: -1, t0: performance.now(), cues: [] as Cue[],
  textCache: [] as Detection[], lastOcr: -1e9, depthOk: false, lastDepth: -1e9, outdoor: false, debug: false, mirrored: false, lastCueT: -1e9 };
const now = () => (performance.now() - state.t0) / 1000;

// ---- detector worker -------------------------------------------------------
const worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
let inflight = false; let detId = 0;
const pending = new Map<number, (d: Detection[]) => void>();
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "ready") { state.backend = m.backend; setStatus(`Detector ready (${m.backend})`); }
  if (m.type === "result") { state.detMs = m.ms; pending.get(m.id)?.(m.dets); pending.delete(m.id); inflight = false; }
};
worker.postMessage({ type: "init", modelUrl: "/models/yolo11n.onnx" });
function detect(img: ImageData): Promise<Detection[]> {
  return new Promise((res) => { const id = ++detId; pending.set(id, res); inflight = true; worker.postMessage({ type: "detect", id, img, conf: 0.3 }, [img.data.buffer]); });
}

function setStatus(s: string) { status.textContent = s; }

// ---- main loop ---------------------------------------------------------------
async function loop() {
  if (!state.running) return;
  if (video.readyState >= 2 && !inflight) {
    const w = 640, h = Math.round((video.videoHeight / video.videoWidth) * 640) || 480;
    if (frameCanvas.width !== w || frameCanvas.height !== h) { frameCanvas.width = w; frameCanvas.height = h; }
    fctx.drawImage(video, 0, 0, w, h);
    const img = fctx.getImageData(0, 0, w, h);
    const t = now();
    let dets = await detect(img);
    if (state.mirrored) dets = dets.map((d) => ({ ...d, bbox: [1 - d.bbox[2], d.bbox[1], 1 - d.bbox[0], d.bbox[3]] as Detection["bbox"] }));
    // metric depth from the server at ~4 Hz (otherwise bbox-height guesses with low confidence)
    if (server.available && t - state.lastDepth > 0.25) {
      state.lastDepth = t;
      server.depth(frameCanvas, dets, state.outdoor).then((ok) => { state.depthOk = ok; }).catch(() => { state.depthOk = false; });
    }
    // periodic OCR through the server in Navigate/Find (text tracks persist between passes)
    if (server.available && t - state.lastOcr > 2.0 && state.mode !== "read") {
      state.lastOcr = t;
      server.ocr(frameCanvas).then((tx) => { state.textCache = tx; }).catch(() => {});
    }
    const tracks = tracker.update([...dets, ...state.textCache], t);
    step(tracks, t);
    draw(tracks);
  }
  requestAnimationFrame(loop);
}

function step(tracks: Track[], t: number) {
  if (t - state.lastDecision < policy.cfg.decisionInterval) return;
  state.lastDecision = t;
  const busy = !speech.free(t);
  const cur = busy ? speech.current : null;
  const cue = policy.decide(tracks, t, { busy, remaining: speech.remaining(t), currentUtility: cur?.utility ?? 0, currentSeverity: cur?.severity ?? 0, currentTrack: cur?.trackId ?? null });
  updateBudget();
  if (!cue) return;
  state.cues.push(cue); state.lastCueT = t;
  if (cue.action === "haptic") { haptic(); earcon(); showCue("BUZZ", "Obstacle imminent", cue); return; }
  speech.speak(cue, t, cue.action === "interrupt");
  if (cue.action === "interrupt") haptic([40]);
  showCue(cue.action.toUpperCase(), cue.message, cue);
  lastCueEl.textContent = cue.message;
}

function showCue(kind: string, text: string, cue: Cue) {
  cueKind.textContent = kind; cueText.textContent = text;
  const d = cue.distance != null ? `${policy.cfg.units === "ft" ? (cue.distance * 3.28).toFixed(0) + " ft" : cue.distance.toFixed(1) + " m"}` : "";
  cueMeta.textContent = [d, cue.ttc != null ? `TTC ${cue.ttc.toFixed(1)} s` : "", `U ${cue.utility.toFixed(2)}`].filter(Boolean).join(" · ");
  cueCard.classList.add("show"); cueCard.dataset.kind = cue.action;
  setTimeout(() => { if (now() - state.lastCueT > 2.4) cueCard.classList.remove("show"); }, 2600);
}
function updateBudget() {
  const f = Math.max(0, Math.min(1, policy.budget / policy.cfg.budgetCapacity));
  budgetFill.style.width = `${f * 100}%`; budgetText.textContent = `${Math.round(f * 100)}%`;
}

// ---- overlay ---------------------------------------------------------------------
function draw(tracks: Track[]) {
  const W = (overlay.width = video.clientWidth), H = (overlay.height = video.clientHeight);
  octx.clearRect(0, 0, W, H);
  const cands = new Map(policy.lastCandidates.map((c) => [c.track.id, c]));
  for (const tr of tracks) {
    const [x1, y1, x2, y2] = tr.bbox; const c = cands.get(tr.id);
    const hot = c && c.utility >= policy.cfg.thresholdSpeak;
    octx.strokeStyle = tr.kind === "text" ? "#f0b13b" : hot ? "#ff5a5a" : "#5ee0b4"; octx.lineWidth = hot ? 3 : 2;
    octx.strokeRect(x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
    const d = tr.distance != null ? (policy.cfg.units === "ft" ? `${(tr.distance * 3.28).toFixed(0)}ft` : `${tr.distance.toFixed(1)}m`) : "";
    const label = `${tr.kind === "text" ? tr.text : tr.label} ${d}${Number.isFinite(tr.ttc) ? ` ${tr.ttc.toFixed(1)}s` : ""}`;
    octx.font = "600 13px system-ui"; const tw = octx.measureText(label).width + 8;
    octx.fillStyle = "rgba(0,0,0,.65)"; octx.fillRect(x1 * W, Math.max(0, y1 * H - 18), tw, 18);
    octx.fillStyle = "#fff"; octx.fillText(label, x1 * W + 4, Math.max(13, y1 * H - 5));
  }
  setStatus(`${state.running ? "LIVE" : "paused"} · ${state.backend || "loading model"} · det ${state.detMs.toFixed(0)} ms · ${tracks.length} tracks · depth ${server.available ? (state.depthOk ? "server" : "…") : "bbox"}${speech.enabled ? "" : " · muted"}`);
  if (state.debug) renderDebug(policy.lastCandidates);
}
function renderDebug(cands: Candidate[]) {
  debugRows.innerHTML = cands.slice(0, 8).map((c) => `<tr><td>${c.track.kind === "text" ? c.track.text : c.track.label}</td><td>${c.distance.toFixed(1)}</td><td>${c.ttc >= 10 ? "–" : c.ttc.toFixed(1)}</td><td>${c.pathProb.toFixed(2)}</td><td>${c.hazard.toFixed(2)}</td><td>${c.urgency.toFixed(2)}</td><td><b>${c.utility.toFixed(2)}</b></td></tr>`).join("");
  debugLog.textContent = state.cues.slice(-6).map((c) => `${c.t.toFixed(1)}s ${c.action.padEnd(9)} ${c.message || "(buzz)"}`).join("\n");
}

// ---- ASK (rule-based over the current event graph) ----------------------------------
function answer(q: string): string {
  const tracks = [...tracker.tracks.values()].filter((t) => t.misses === 0 && t.confirmed);
  const near = [...tracks].sort((a, b) => (a.distance ?? 99) - (b.distance ?? 99));
  const desc = (t: Track) => `${t.kind === "text" ? "sign " + t.text : t.label}${t.distance != null ? `, about ${t.distance.toFixed(0)} meters` : ""}, ${Math.abs(camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2)) < 10 ? "ahead" : camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2) > 0 ? "to your right" : "to your left"}`;
  q = q.toLowerCase();
  const find = q.match(/(?:where(?:'s| is)|find|look for)\s+(?:the |a |an )?([a-z0-9 ]+)/);
  if (find) { const tags = find[1].trim().split(/\s+/); policy.setGoal([find[1].trim(), ...tags]); $<HTMLInputElement>("find-input").value = find[1].trim(); return `Looking for ${find[1].trim()}. I will tell you when I see it.`; }
  for (const key of ["stairs", "door", "exit", "person", "car", "chair", "elevator", "restroom", "sign"]) {
    if (q.includes(key)) {
      const hits = near.filter((t) => t.label.includes(key) || (t.text ?? "").toLowerCase().includes(key));
      return hits.length ? hits.slice(0, 2).map(desc).join(". ") + "." : `I don't see a ${key} right now.`;
    }
  }
  if (q.includes("left")) { const h = near.filter((t) => camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2) < -10); return h.length ? h.slice(0, 3).map(desc).join(". ") + "." : "Nothing detected on your left."; }
  if (q.includes("right")) { const h = near.filter((t) => camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2) > 10); return h.length ? h.slice(0, 3).map(desc).join(". ") + "." : "Nothing detected on your right."; }
  if (q.includes("detail") || q.includes("more")) { const c = state.cues.filter((c) => c.action !== "haptic").at(-1); return c ? `${c.message} Distance ${c.distance?.toFixed(1) ?? "unknown"} meters.` : "No recent cue."; }
  return near.length ? `I see ${near.length} things. ` + near.slice(0, 3).map(desc).join(". ") + "." : "I don't see anything notable right now.";
}
function say(text: string) {
  srLive.textContent = text; lastCueEl.textContent = text;
  speech.speak({ t: now(), trackId: -1, label: "answer", message: text, action: "speak", duration: Math.max(1.2, text.split(/\s+/).length * 0.35), utility: 0, severity: 0, distance: null, ttc: null, slots: [] }, now(), true);
}

// ---- UI wiring ----------------------------------------------------------------------
$("start").onclick = async () => {
  if (state.running) { state.running = false; cam.stop(); $("start").textContent = "Start camera"; return; }
  try {
    const s = await cam.start({ facing: cam.facing });
    camModel.aspect = (s?.width ?? 4) / (s?.height ?? 3);
    state.mirrored = cam.mirrored; state.running = true; state.t0 = performance.now(); tracker.reset(); policy.reset(); speech.reset();
    $("start").textContent = "Stop"; setStatus("Camera on"); loop();
    await server.check(); if (server.available) setStatus(`Server connected (depth: ${server.depthModel})`);
    const devices = await cam.listDevices(); const sel = $<HTMLSelectElement>("camera-select");
    sel.innerHTML = `<option value="">auto</option>` + devices.map((d, i) => `<option value="${d.deviceId}">${d.label || "camera " + (i + 1)}</option>`).join("");
  } catch (e) { setStatus(`Camera error: ${(e as Error).message}. Use HTTPS or localhost and allow the camera.`); }
};
$("flip").onclick = async () => { if (!state.running) return; await cam.flip(); state.mirrored = cam.mirrored; tracker.reset(); };
$<HTMLSelectElement>("camera-select").onchange = async (e) => { const id = (e.target as HTMLSelectElement).value; if (state.running) { await cam.start({ deviceId: id }); state.mirrored = cam.mirrored; tracker.reset(); } };
$<HTMLInputElement>("file-input").onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0]; if (!f) return;
  await cam.playFile(f); state.mirrored = false; state.running = true; state.t0 = performance.now(); tracker.reset(); policy.reset(); speech.reset();
  $("start").textContent = "Stop"; loop(); await server.check();
};
$("mute").onclick = () => { speech.enabled = !speech.enabled; $("mute").textContent = speech.enabled ? "Sound on" : "Muted"; $("mute").setAttribute("aria-pressed", String(!speech.enabled)); if (!speech.enabled) speech.stop(); };
$<HTMLSelectElement>("verbosity").onchange = (e) => { policy.cfg.verbosity = (e.target as HTMLSelectElement).value as any; };
$<HTMLSelectElement>("units").onchange = (e) => { policy.cfg.units = (e.target as HTMLSelectElement).value as any; };
$<HTMLSelectElement>("voice").onchange = (e) => { speech.backend = (e.target as HTMLSelectElement).value as any; };
$("debug-toggle").onclick = () => { state.debug = !state.debug; $("debug").classList.toggle("hidden", !state.debug); $("debug-toggle").setAttribute("aria-expanded", String(state.debug)); };
document.querySelectorAll<HTMLButtonElement>(".mode").forEach((b) => b.onclick = () => {
  state.mode = b.dataset.mode as Mode;
  document.querySelectorAll(".mode").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", String(x === b)); });
  $("find-form").classList.toggle("hidden", state.mode !== "find"); $("read-panel").classList.toggle("hidden", state.mode !== "read"); $("ask-form").classList.toggle("hidden", state.mode !== "ask");
  if (state.mode === "navigate") policy.setGoal([]);
});
$<HTMLFormElement>("find-form").onsubmit = (e) => { e.preventDefault(); const v = $<HTMLInputElement>("find-input").value.trim(); policy.setGoal(v ? [v, ...v.split(/\s+/)] : []); say(v ? `Looking for ${v}.` : "Find cleared."); };
$("find-clear").onclick = () => { $<HTMLInputElement>("find-input").value = ""; policy.setGoal([]); say("Find cleared."); };
$("read-now").onclick = async () => {
  if (!server.available) { say("Reading text needs the SeeForMe server. Start it with make serve."); return; }
  say("Reading."); const tx = await server.ocr(frameCanvas);
  say(tx.length ? tx.map((t) => t.text).join(". ") : "I can't read any text here.");
};
$<HTMLFormElement>("ask-form").onsubmit = (e) => { e.preventDefault(); const q = $<HTMLInputElement>("ask-input").value.trim(); if (q) say(answer(q)); };
$("ask-voice").onclick = () => {
  const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!SR) { say("Voice input is not available in this browser."); return; }
  const r = new SR(); r.lang = "en-US"; r.onresult = (ev: any) => { const q = ev.results[0][0].transcript; $<HTMLInputElement>("ask-input").value = q; say(answer(q)); }; r.start();
};
document.addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT") return;
  if (e.key === " ") { e.preventDefault(); $("start").click(); }
  if (e.key.toLowerCase() === "f") $("flip").click();
  if (e.key.toLowerCase() === "m") $("mute").click();
  if (e.key.toLowerCase() === "d") $("debug-toggle").click();
});
updateBudget();
setStatus("Press Start camera (or Space).");
