import { CameraSource } from "./camera";
import { distancePhrase } from "./compose";
import { LearnedScheduler, loadLearnedRunner, type SchedulerMeta } from "./learned";
import { PCACS, DEFAULT_CFG, type Candidate } from "./pcacs";
import { ServerClient } from "./server";
import { Speech, earcon, haptic } from "./speech";
import { Camera, Tracker } from "./tracker";
import type { Cue, Detection, Track } from "./types";
import { APP_VERSION } from "./version";
import { COCO } from "./yolo";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video"), overlay = $<HTMLCanvasElement>("overlay"), status = $("status");
const srLive = $("sr-live"), srAlert = $("sr-alert"), cueCard = $("cue-card"), cueKind = $("cue-kind"), cueText = $("cue-text"), cueMeta = $("cue-meta");
const lastCueEl = $("last-cue"), budgetFill = $("budget-fill"), budgetText = $("budget-text"), debugLog = $("debug-log");
const candRows = $("cand-rows"), actionNow = $("action-now"), actionWhy = $("action-why"), channelState = $("channel-state"), egoSpeedEl = $("ego-speed"), depthSrc = $("depth-src"), liveDot = $("live-dot"), diag = $("diag");

const serverUrl = () => (location.port === "5173" ? `${location.protocol}//${location.hostname}:8000` : location.origin);
const server = new ServerClient(serverUrl());
const cam = new CameraSource(video);
const camModel = new Camera();
const tracker = new Tracker(camModel);
// Two schedulers share one config (tuned weights, verbosity, units): the learned GRU + RL scheduler (default, the
// production model of the Python pipeline, run with ONNX Runtime) and the linear P-CACS, the fallback.
const cfg = { ...DEFAULT_CFG, weights: { ...DEFAULT_CFG.weights } };
const linear = new PCACS(cfg, camModel);
let learned: LearnedScheduler | null = null;
let policy: PCACS | LearnedScheduler = linear;
const isLearned = () => learned != null && policy === learned;
/** localStorage can throw (private mode, blocked storage): preferences are a convenience, never required */
const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};
type SchedulerChoice = "learned" | "pcacs";
let wanted: SchedulerChoice = store.get("seeforme:scheduler") === "pcacs" ? "pcacs" : "learned";
// DEFAULT_CFG holds the same tuned values; a failed load is still shown, so the two cannot drift apart unseen
let cfgDefaults = false;
const cfgReady = fetch("/config/pcacs.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`pcacs.json ${r.status}`))))
  .then((c) => { Object.assign(cfg, c, { weights: { ...cfg.weights, ...c.weights }, verbosity: cfg.verbosity, units: cfg.units }); linear.budget = cfg.budgetCapacity; if (learned) learned.budget = cfg.budgetCapacity; })
  .catch((e) => { cfgDefaults = true; console.warn(`pcacs.json not loaded, using built-in defaults: ${e}`); diag.textContent = "Scheduler config not loaded: using built-in defaults."; });
// Every spoken cue/answer updates the visible last-cue line; it goes to a live region only while the app's own voice is
// muted, so a screen-reader user never hears the same cue from both the app and the screen reader. Interrupt cues go to
// the assertive #sr-alert (a hazard must not queue behind what the screen reader is reading), the rest to the polite #sr-live.
const speech = new Speech(serverUrl, (text, urgent) => { lastCueEl.textContent = text; if (!speech.enabled) (urgent ? srAlert : srLive).textContent = text; });
const voiceSel = $<HTMLSelectElement>("voice");
/** The browser voice unless ElevenLabs was picked and the server says its TTS works (an API key is set). */
function pickVoice() { speech.backend = voiceSel.value === "elevenlabs" && server.available && server.tts ? "elevenlabs" : "browser"; }
const frameCanvas = document.createElement("canvas");
const fctx = frameCanvas.getContext("2d", { willReadFrequently: true })!;
const octx = overlay.getContext("2d")!;

type Mode = "navigate" | "find" | "read" | "ask";
const state = { mode: "navigate" as Mode, running: false, backend: "", detMs: 0, lastDecision: -1, t0: performance.now(), cues: [] as Cue[],
  textCache: [] as Detection[], lastOcr: -1e9, ocrT: -1e9, ocrFails: 0, depthOk: false, lastDepth: -1e9, outdoor: false, debug: true, mirrored: false, lastCueT: -1e9,
  decisions: 0, silentDecisions: 0, speechSeconds: 0 };
const now = () => (performance.now() - state.t0) / 1000;

// ---- detector worker -------------------------------------------------------
let worker: Worker;
let inflight = false; let detId = 0; let readyTimer = 0; let timeouts = 0; let errors = 0;
/** The one detector state the Start button, the status messages and the scheduler notes read ("ready" again after a restart). */
let detectorState: "loading" | "ready" | "failed" = "loading";
const DETECT_TIMEOUT_MS = 5000, MAX_TIMEOUTS = 3, MAX_ERRORS = 3;
const DETECTOR_OFF = "Object detection is not working, obstacle cues are off. Reload the page to retry.";
const pending = new Map<number, (d: Detection[]) => void>();
function startWorker(threads?: number) {
  if (worker) worker.terminate();
  detectorState = "loading"; inflight = false; errors = 0; pending.forEach((done) => done([])); pending.clear();
  worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (ev) => {
    const m = ev.data;
    // Startup stages are diagnostics (aria-hidden #diag); the role=status region is set once, when the detector is ready.
    if (m.type === "stage") diag.textContent = `Detector: ${m.stage}…`;
    if (m.type === "error") { if (threads !== 1) fallback(); else detectorFailed(m.message.split("\n")[0]); }
    if (m.type === "ready") {
      detectorState = "ready"; clearTimeout(readyTimer); state.backend = m.backend; diag.textContent = `Detector ready (${m.backend})`;
      setStatus(takeNotice() + (state.running ? "Detector ready." : "Ready. Press Start (or Space)."));
      if (state.running && speech.enabled) say("Object detection is ready.");   // the camera started (or ran) without it
      startLearned();
    }
    if (m.type === "result") {
      const done = pending.get(m.id); if (!done) return; // late answer to a request that already timed out
      pending.delete(m.id); inflight = false; timeouts = 0;
      if (!m.error) { errors = 0; if (typeof m.ms === "number") state.detMs = m.ms; }
      else {
        console.warn(`detector: ${m.error}`);
        // a session that throws on every frame (e.g. a lost WebGPU device) returns "nothing ahead" forever: restart it once, then give up out loud
        if (++errors >= MAX_ERRORS) {
          if (threads !== 1) { setStatus("Detector error, restarting."); startWorker(1); }
          else { worker.terminate(); detectorFailed(String(m.error).split("\n")[0].slice(0, 80)); }
        }
      }
      done(m.dets ?? []);
    }
  };
  worker.onerror = () => { if (threads !== 1) fallback(); else detectorFailed("the detector could not start"); };
  worker.postMessage({ type: "init", modelUrl: "/models/yolo11n.onnx", threads });
  clearTimeout(readyTimer);
  readyTimer = window.setTimeout(() => { if (detectorState === "loading") { if (threads !== 1) fallback(); else detectorFailed("the detector did not load in time"); } }, 25000);
}
/** Final failure: say so in the status region (and aloud while the camera runs), so silence is never mistaken for "nothing ahead". */
function detectorFailed(why: string) {
  clearTimeout(readyTimer); detectorState = "failed";
  setStatus(`Detector failed: ${why}. Object cues are off; reload the page to retry.`);
  if (state.running && speech.enabled) say(DETECTOR_OFF);
  startLearned();
}
function fallback() { diag.textContent = "Detector: retrying single-threaded…"; startWorker(1); }
startWorker();
function detect(img: ImageData): Promise<Detection[]> {
  if (detectorState !== "ready") return Promise.resolve([]);
  return new Promise((res) => {
    const id = ++detId; inflight = true;
    // a request the worker never answers resolves empty, so a hung run cannot freeze the loop
    const timer = window.setTimeout(() => {
      if (pending.delete(id)) inflight = false;
      res([]);
      // a hung session answers nothing ever again: restart it rather than return empty frames forever
      if (++timeouts >= MAX_TIMEOUTS) { timeouts = 0; setStatus("Detector stalled, restarting."); startWorker(1); }
    }, DETECT_TIMEOUT_MS);
    pending.set(id, (d) => { clearTimeout(timer); res(d); });
    worker.postMessage({ type: "detect", id, img, conf: 0.3 }, [img.data.buffer]);
  });
}

function setStatus(s: string) { status.textContent = s; }
/** A one-time notice ("Updated to version X") is also put in front of the next status message, so it is not lost
 *  when "Ready" replaces the status a moment later. */
let notice = "";
function takeNotice() { const n = notice; notice = ""; return n ? `${n} ` : ""; }

// ---- schedulers ------------------------------------------------------------------
const schedSelect = $<HTMLSelectElement>("scheduler"), schedInfo = $("sched-info"), schedTag = $("sched-tag"), formula = $("formula"), scoreHead = $("score-head");
const LINEAR_FORMULA = formula.innerHTML;
const learnedFormula = (b: number, K: number) => `per candidate: GRU over its last ${K} feature vectors → log-odds of speak / interrupt / haptic against silence (illegal actions masked); the largest acts if it beats the silence bias b = ${b.toFixed(2)}`;
function setGoal(tags: string[]) { linear.setGoal(tags); learned?.setGoal(tags); }
function resetSchedulers() { linear.reset(); learned?.reset(); }
/** Switch the active scheduler, carrying the speech budget, walking speed, goal and announcement memory over. */
function useScheduler(choice: SchedulerChoice, announce: boolean) {
  const next = choice === "learned" && learned ? learned : linear;
  if (next !== policy) {
    next.reset();
    next.budget = policy.budget; next.lastT = policy.lastT; next.egoSpeed = policy.egoSpeed;
    next.announcedDist = new Map(policy.announcedDist); next.goalTags = [...policy.goalTags];
    policy = next;
  }
  schedSelect.value = isLearned() ? "learned" : "pcacs";
  schedTag.textContent = isLearned() ? "GRU + RL" : "P-CACS";
  if (learned && isLearned()) formula.textContent = learnedFormula(learned.silenceBias, learned.K);
  else formula.innerHTML = LINEAR_FORMULA;
  scoreHead.textContent = isLearned() ? "L" : "U";
  scoreHead.title = isLearned() ? "best legal log-odds against silence" : "linear utility";
  if (announce) setStatus(isLearned() ? "Scheduler: learned (GRU + RL)." : "Scheduler: linear P-CACS.");
}
/** The learned scheduler cannot run: say so, keep the linear P-CACS, and mark the option unavailable. */
function learnedUnavailable(why: string) {
  useScheduler("pcacs", false);
  learned = null; learnedState = "failed";
  const opt = schedSelect.querySelector<HTMLOptionElement>('option[value="learned"]');
  if (opt) { opt.disabled = true; opt.textContent = "Learned (unavailable)"; }
  schedInfo.textContent = `Learned scheduler unavailable: ${why}. Using linear P-CACS.`;
  // a failed detector (often the same root cause) keeps the status region: it is the message that matters
  if (wanted === "learned" && detectorState !== "failed") setStatus(`Learned scheduler unavailable (${why}). Using linear P-CACS.`);
}
let learnedState: "idle" | "loading" | "ready" | "failed" = "idle";
/** Load the learned scheduler once the detector has fetched the ONNX Runtime files (so both share one download). */
function startLearned() {
  if (learnedState !== "idle") return;
  learnedState = "loading";
  void (async () => {
    try {
      const r = await fetch("/config/scheduler.json", { cache: "no-cache" });
      if (!r.ok) throw new Error(`scheduler.json ${r.status}`);
      const meta = (await r.json()) as SchedulerMeta;
      const run = await loadLearnedRunner(meta);
      await cfgReady;
      learned = new LearnedScheduler(cfg, camModel, meta, run);
      learned.goalTags = [...linear.goalTags];
      learnedState = "ready";
      schedInfo.textContent = `Learned scheduler: GRU + RL, model ${meta.version}, last ${meta.K} steps per track, silence bias ${meta.silence_bias.toFixed(2)}.`;
      noteModelVersion(meta.version);
      useScheduler(wanted, false);
    } catch (e) { learnedUnavailable(String((e as Error)?.message ?? e).split("\n")[0].slice(0, 120)); }
  })();
}
window.setTimeout(startLearned, 30000);   // the detector never reported back: load the scheduler anyway
schedSelect.value = wanted;   // until the model is ready the linear P-CACS runs (the header tag says which one is active)
schedSelect.onchange = () => {
  wanted = schedSelect.value === "pcacs" ? "pcacs" : "learned";
  store.set("seeforme:scheduler", wanted);
  if (wanted === "learned" && !learned) {
    if (learnedState === "failed") { setStatus("The learned scheduler is not available; using linear P-CACS."); schedSelect.value = "pcacs"; }
    else setStatus("The learned scheduler is still loading; it takes over when ready.");
    return;
  }
  useScheduler(wanted, true);
};

// ---- main loop ---------------------------------------------------------------
/** front camera: the preview is mirrored, so boxes from the raw frame are flipped to match it */
const mirror = (d: Detection): Detection => ({ ...d, bbox: [1 - d.bbox[2], d.bbox[1], 1 - d.bbox[0], d.bbox[3]] });
// OCR passes overlap (one every 2 s, each may take longer): an older answer, or one from before a camera change, never replaces a newer one
let ocrSeq = 0, ocrDone = 0;
/** New camera view (Start, flip, camera or video change): its signs are not the old view's. */
function dropText() { state.textCache = []; ocrDone = ocrSeq; }
async function loop() {
  if (!state.running) return;
  if (video.readyState >= 2 && !inflight) {
    const w = 640, h = Math.round((video.videoHeight / video.videoWidth) * 640) || 480;
    if (frameCanvas.width !== w || frameCanvas.height !== h) { frameCanvas.width = w; frameCanvas.height = h; }
    fctx.drawImage(video, 0, 0, w, h);
    const img = fctx.getImageData(0, 0, w, h);
    const t = now();
    let dets = await detect(img);
    if (state.mirrored) dets = dets.map(mirror);
    // metric depth from the server at ~4 Hz (otherwise bbox-height guesses with low confidence)
    if (server.available && t - state.lastDepth > 0.25) {
      state.lastDepth = t;
      server.depth(frameCanvas, dets, state.outdoor).then((ok) => { state.depthOk = ok; }).catch(() => { state.depthOk = false; });
    }
    // periodic OCR through the server in Navigate/Find (text tracks persist between passes); a failed pass clears the
    // signs, a result older than 3 s is not used, and three failures in a row mark the server lost
    if (server.available && t - state.lastOcr > 2.0 && state.mode !== "read") {
      state.lastOcr = t;
      const mir = state.mirrored, seq = ++ocrSeq;
      server.ocr(frameCanvas).then((tx) => { if (seq <= ocrDone) return; ocrDone = seq; state.textCache = mir ? tx.map(mirror) : tx; state.ocrT = now(); state.ocrFails = 0; })
        .catch(() => { if (seq <= ocrDone) return; ocrDone = seq; state.textCache = []; if (++state.ocrFails >= 3 && server.available) { server.available = false; pickVoice(); setStatus("Lost the SeeForMe server: text reading and metric depth are off."); } });
    }
    const tracks = tracker.update([...dets, ...(t - state.ocrT < 3.0 ? state.textCache : [])], t);
    await step(tracks, t);
    draw(tracks);
  }
  requestAnimationFrame(loop);
}

let deciding = false; let schedulerErrorShown = false;
async function step(tracks: Track[], t: number) {
  if (deciding || t - state.lastDecision < cfg.decisionInterval) return;
  deciding = true; state.lastDecision = t;
  // a scheduler error must never stop the camera loop, and must be said out loud once: silence would read as "nothing ahead"
  try { await decideAndAct(tracks, t); schedulerErrorShown = false; }
  catch (e) {
    console.error("scheduler:", e);
    if (!schedulerErrorShown) { schedulerErrorShown = true; setStatus(`Scheduler error, cues are off: ${String((e as Error)?.message ?? e).split("\n")[0].slice(0, 80)}`); }
  } finally { deciding = false; }
}
async function decideAndAct(tracks: Track[], t: number) {
  const busy = !speech.free(t);
  const cur = busy ? speech.current : null;
  const ch = { busy, remaining: speech.remaining(t), currentUtility: cur?.utility ?? 0, currentSeverity: cur?.severity ?? 0, currentTrack: cur?.trackId ?? null };
  let cue: Cue | null;
  try { cue = await policy.decide(tracks, t, ch); }
  catch (e) {
    if (!isLearned()) throw e;
    learnedUnavailable(`model error: ${String((e as Error)?.message ?? e).slice(0, 80)}`);
    cue = linear.decide(tracks, t, ch);
  }
  updateBudget();
  state.decisions++; if (!cue) state.silentDecisions++;
  renderAlgorithm(policy.lastCandidates, cue, busy, t);
  renderSession(t);
  if (!cue) return;
  state.cues.push(cue); state.lastCueT = t;
  if (cue.action !== "haptic") state.speechSeconds += cue.duration;
  if (cue.action === "haptic") { haptic(); earcon(); showCue("BUZZ", "Obstacle imminent", cue); return; }
  speech.speak(cue, t, cue.action === "interrupt");
  if (cue.action === "interrupt") haptic([40]);
  showCue(cue.action.toUpperCase(), cue.message, cue);
}

function showCue(kind: string, text: string, cue: Cue) {
  cueKind.textContent = kind; cueText.textContent = text;
  const d = cue.distance != null ? `${cfg.units === "ft" ? (cue.distance * 3.28).toFixed(0) + " ft" : cue.distance.toFixed(1) + " m"}` : "";
  cueMeta.textContent = [d, cue.ttc != null ? `TTC ${cue.ttc.toFixed(1)} s` : "", `${isLearned() ? "L" : "U"} ${cue.utility.toFixed(2)}`].filter(Boolean).join(" · ");
  cueCard.classList.add("show"); cueCard.dataset.kind = cue.action;
  setTimeout(() => { if (now() - state.lastCueT > 2.4) cueCard.classList.remove("show"); }, 2600);
}
function updateBudget() {
  const f = Math.max(0, Math.min(1, policy.budget / cfg.budgetCapacity));
  budgetFill.style.width = `${f * 100}%`; budgetText.textContent = `${Math.round(f * 100)}%`;
}

// ---- overlay ---------------------------------------------------------------------
function draw(tracks: Track[]) {
  const W = (overlay.width = video.clientWidth), H = (overlay.height = video.clientHeight);
  octx.clearRect(0, 0, W, H);
  const cands = new Map(policy.lastCandidates.map((c) => [c.track.id, c]));
  for (const tr of tracks) {
    const [x1, y1, x2, y2] = tr.bbox; const c = cands.get(tr.id);
    const hot = c != null && isHot(c);
    octx.strokeStyle = tr.kind === "text" ? "#f0b13b" : hot ? "#ff5a5a" : "#5ee0b4"; octx.lineWidth = hot ? 3 : 2;
    octx.strokeRect(x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H);
    const d = tr.distance != null ? (cfg.units === "ft" ? `${(tr.distance * 3.28).toFixed(0)}ft` : `${tr.distance.toFixed(1)}m`) : "";
    const label = `${tr.kind === "text" ? tr.text : tr.label} ${d}${Number.isFinite(tr.ttc) ? ` ${tr.ttc.toFixed(1)}s` : ""}`;
    octx.font = "600 13px system-ui"; const tw = octx.measureText(label).width + 8;
    octx.fillStyle = "rgba(0,0,0,.65)"; octx.fillRect(x1 * W, Math.max(0, y1 * H - 18), tw, 18);
    octx.fillStyle = "#fff"; octx.fillText(label, x1 * W + 4, Math.max(13, y1 * H - 5));
  }
  // Per-frame diagnostics are visual only (aria-hidden); live-region changes every frame overwhelm screen readers.
  diag.textContent = `${state.running ? "LIVE" : "paused"} · ${state.backend || "loading model"} · det ${state.detMs.toFixed(0)} ms · ${tracks.length} tracks · depth ${server.available ? (state.depthOk ? "server" : "…") : "bbox"}${speech.enabled ? "" : " · muted"}${cfgDefaults ? " · built-in config" : ""}`;
  liveDot.classList.toggle("live", state.running);
}
/** The value the active scheduler acts on: linear utility U, or the learned best legal log-odds L against silence. */
const candValue = (c: Candidate) => (isLearned() ? (c.score ?? -Infinity) : c.utility);
const threshold = () => (learned && isLearned() ? learned.silenceBias : cfg.thresholdSpeak);
const isHot = (c: Candidate) => (isLearned() ? candValue(c) > threshold() : candValue(c) >= threshold());
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
function renderAlgorithm(cands: Candidate[], cue: Cue | null, busy: boolean, t: number) {
  const thr = threshold();
  const maxU = Math.max(thr * 1.5, ...cands.map(candValue).filter(Number.isFinite));
  candRows.innerHTML = cands.length ? cands.slice(0, 7).map((c) => {
    const v = candValue(c), hot = isHot(c);
    const w = Math.max(2, Math.round((Math.max(0, Number.isFinite(v) ? v : 0) / maxU) * 44));
    const name = c.track.kind === "text" ? `“${(c.track.text ?? "").slice(0, 16)}”` : c.track.label;
    return `<tr class="${hot ? "hot" : ""}"><td>${name}</td><td>${c.distance.toFixed(1)}</td><td>${c.ttc >= 10 ? "–" : c.ttc.toFixed(1)}</td><td>${c.pathProb.toFixed(2)}</td><td>${c.hazard.toFixed(2)}</td><td>${c.urgency.toFixed(2)}</td><td>${c.novelty.toFixed(2)}</td><td>${c.speechCost.toFixed(2)}</td><td><span class="ubar" style="width:${w}px"></span>${fmt(v)}</td></tr>`;
  }).join("") : `<tr><td colspan="9" class="muted">no tracks</td></tr>`;
  const a = cue?.action ?? "silence";
  actionNow.textContent = a; actionNow.className = a;
  const top = cands[0];
  if (isLearned()) actionWhy.textContent = cue ? `L ${cue.utility.toFixed(2)} > b ${thr.toFixed(2)}` : top ? (Number.isFinite(candValue(top)) ? `best L ${candValue(top).toFixed(2)} ≤ b ${thr.toFixed(2)}` : "no legal action (just announced)") : "";
  else actionWhy.textContent = cue ? `U ${cue.utility.toFixed(2)} ≥ ${thr.toFixed(2)}` : top ? (top.utility < thr ? `best U ${top.utility.toFixed(2)} < ${thr.toFixed(2)}` : busy ? "channel busy" : "blocked (recent / budget)") : "";
  channelState.textContent = busy ? `busy · ${speech.remaining(t).toFixed(1)} s left` : "free";
  egoSpeedEl.textContent = `${policy.egoSpeed.toFixed(1)} m/s`;
  depthSrc.textContent = server.available ? (state.depthOk ? `server · ${server.depthModel}` : "server · waiting") : "bbox height (hedged)";
  debugLog.textContent = state.cues.slice(-5).map((c) => `${c.t.toFixed(1)}s ${c.action.padEnd(9)} ${c.message || "(buzz)"}`).join("\n");
}
function renderSession(t: number) {
  const spoken = state.cues.filter((c) => c.action !== "haptic");
  const min = Math.max(1 / 60, t / 60);
  $("s-cues").textContent = String(spoken.length);
  $("s-rate").textContent = (spoken.length / min).toFixed(1);
  $("s-occ").textContent = `${Math.round(100 * Math.min(1, state.speechSeconds / Math.max(1, t)))}%`;
  $("s-silent").textContent = `${Math.round(100 * (state.decisions ? state.silentDecisions / state.decisions : 1))}%`;
  $("s-haptic").textContent = String(state.cues.filter((c) => c.action === "haptic").length);
  $("s-interrupt").textContent = String(state.cues.filter((c) => c.action === "interrupt").length);
}
interface BenchRow { id: string; name: string; [k: string]: number | string }
let bench: BenchRow[] = [];
function renderBench() {
  const metric = $<HTMLSelectElement>("bench-metric").value;
  const lowerBetter = ["late_critical_rate", "audio_occupancy", "regret"].includes(metric);
  const vals = bench.map((b) => Number(b[metric]));
  const max = Math.max(...vals, 1e-6);
  $("bench-rows").innerHTML = bench.map((b) => {
    const v = Number(b[metric]);
    const cls = b.id === "pcacs_learned:rl" ? "hl" : b.id === "oracle" ? "ref" : "";
    return `<div class="b ${cls}"><span class="n">${b.name}</span><span class="t"><i style="width:${Math.max(1, (v / max) * 100)}%"></i></span><span class="v">${v.toFixed(2)}</span></div>`;
  }).join("") + `<p class="muted small">${lowerBetter ? "lower is better" : "higher is better"}</p>`;
}
fetch("/data/benchmark.json").then((r) => r.json()).then((b) => { bench = b; renderBench(); }).catch(() => {});
$<HTMLSelectElement>("bench-metric").onchange = renderBench;

// ---- FIND goal tags -------------------------------------------------------------------
const LABELS = new Set(COCO); // what the in-browser detector can see; everything else needs server OCR
const STOPWORDS = new Set(["a", "an", "the", "my", "your", "some", "any", "me", "to", "for", "of", "in", "on", "at", "and", "or", "please", "nearest", "closest"]);
const ALIASES: Record<string, string[]> = { seat: ["chair", "bench"], people: ["person"], bike: ["bicycle"], table: ["dining table"], sofa: ["couch"], plant: ["potted plant"], phone: ["cell phone"], television: ["tv"] };
const SERVER_NOTE = "Without the SeeForMe server I can only find common objects such as a chair, a person or a bottle. Signs and room numbers need the server.";
/** "the exit" -> [exit]; "room 204" -> [room 204, 204]; "a seat" -> [seat, chair, bench]. Stopwords and tokens shorter
 *  than 3 characters are dropped (a lone "a" would match almost any sign); aliases and plurals map to detector labels. */
function goalTags(raw: string): string[] {
  const words = raw.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter((w) => w && !STOPWORDS.has(w));
  // "room 204": the number identifies the goal, so a lone "room" (any Room 310 sign) is not a tag; "exit door" keeps "exit" and "door"
  const singles = words.length > 1 && words.some((w) => /\d/.test(w)) ? words.filter((w) => /\d/.test(w)) : words;
  const phrase = words.join(" ");
  // a lone one-digit number ("gate 7" -> "7") would match any sign with a 7, so it only counts inside the phrase
  const tags = [phrase, ...singles].filter((w) => w.length >= 3 || LABELS.has(w) || (/\d/.test(w) && (w.length >= 2 || w === phrase)));
  for (const w of words) { tags.push(...(ALIASES[w] ?? [])); if (w.endsWith("s") && LABELS.has(w.slice(0, -1))) tags.push(w.slice(0, -1)); }
  return [...new Set(tags)];
}
const objectGoal = (tags: string[]) => tags.some((g) => LABELS.has(g));
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/** whole-word match, so a "204" goal does not fire on a "1204" sign */
const hasWord = (hay: string, w: string) => new RegExp(`(^|[^\\p{L}\\p{N}])${escapeRe(w)}($|[^\\p{L}\\p{N}])`, "u").test(hay);
const matchesGoal = (t: Track, tags: string[]) => tags.some((g) => t.label === g || (t.kind === "text" && hasWord((t.text ?? "").toLowerCase(), g)));

// ---- ASK (rule-based over the current event graph) ----------------------------------
function answer(q: string): string {
  const tracks = [...tracker.tracks.values()].filter((t) => t.misses === 0 && t.confirmed);
  const near = [...tracks].sort((a, b) => (a.distance ?? 99) - (b.distance ?? 99));
  const dir = (t: Track) => { const a = camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2); return Math.abs(a) < 10 ? "straight ahead" : Math.abs(a) < 25 ? (a > 0 ? "slightly to your right" : "slightly to your left") : (a > 0 ? "on your right" : "on your left"); };
  const dist = (t: Track) => { if (t.distance == null) return ""; const d = t.distance; if (t.depthConf < 0.45) return d < 2.5 ? "very close" : d < 4.5 ? "fairly close" : "further away"; const m = cfg.units === "ft" ? `${Math.round(d * 3.28)} feet` : `${d < 3 ? (Math.round(d * 2) / 2) : Math.round(d)} meters`; return `about ${m} away`; };
  const article = (w: string) => (/^[aeiou]/i.test(w) ? "an" : "a");
  const desc = (t: Track) => t.kind === "text" ? `a sign that says ${t.text}, ${dir(t)}` : `${article(t.label)} ${t.label} ${dist(t)}${dist(t) ? ", " : ""}${dir(t)}`;
  const list = (ts: Track[]) => ts.length === 1 ? desc(ts[0]) : ts.slice(0, -1).map(desc).join(", ") + ", and " + desc(ts[ts.length - 1]);
  q = q.toLowerCase();
  const find = q.match(/(?:where(?:'s| is)|find|look for)\s+(?:the |a |an )?([a-z0-9 ]+)/);
  if (find) { const what = find[1].trim().replace(/[?.!]+$/, ""); const tags = goalTags(what);
    if (!tags.length) return "Tell me what to look for, for example a chair or the exit.";   // keep the current goal
    setGoal(tags); $<HTMLInputElement>("find-input").value = what;
    const hit = near.find((t) => matchesGoal(t, tags));
    return hit ? `I can see it now: ${desc(hit)}.` : `Okay, I'm looking for ${what}. I'll tell you as soon as I see it.${!server.available && !objectGoal(tags) ? ` ${SERVER_NOTE}` : ""}`; }
  if (/stop looking|cancel|never mind/.test(q)) { setGoal([]); return "Okay, I stopped looking."; }
  for (const key of ["stairs", "door", "exit", "person", "people", "car", "chair", "seat", "elevator", "restroom", "sign", "bike", "bicycle"]) {
    if (q.includes(key)) {
      const k = key === "people" ? "person" : key === "seat" ? "chair" : key === "bike" ? "bicycle" : key;
      const hits = near.filter((t) => t.label.includes(k) || (t.text ?? "").toLowerCase().includes(k));
      if (!hits.length) return `I could not confirm ${article(key)} ${key} in the current camera view.`;
      return hits.length === 1 ? `Yes, there's ${desc(hits[0])}.` : `Yes, I see ${hits.length}: ${list(hits.slice(0, 3))}.`;
    }
  }
  if (/\bleft\b/.test(q)) { const h = near.filter((t) => camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2) < -10); return h.length ? `On your left there's ${list(h.slice(0, 3))}.` : "I could not confirm anything on your left in the current camera view."; }
  if (/\bright\b/.test(q)) { const h = near.filter((t) => camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2) > 10); return h.length ? `On your right there's ${list(h.slice(0, 3))}.` : "I could not confirm anything on your right in the current camera view."; }
  if (/ahead|in front/.test(q)) { const h = near.filter((t) => Math.abs(camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2)) < 12); return h.length ? `Ahead of you there's ${list(h.slice(0, 3))}.` : "I could not confirm anything ahead in the current camera view. That does not mean it is clear."; }
  if (/detail|more|repeat|again|what did you say/.test(q)) {
    const c = state.cues.filter((c) => c.action !== "haptic").at(-1); if (!c) return "I haven't said anything yet.";
    // as in the cues: a number only with server depth, "very close" / "close" for a bounding-box guess, in the chosen units
    const p = distancePhrase(c.distance, c.depthConf ?? 0, cfg.units);
    return `I said: ${c.message}${p ? ` That's ${p}${p.startsWith("about") ? " away" : ""}.` : ""}`;
  }
  if (/clear|safe|path|way/.test(q)) { const h = near.filter((t) => t.pathProb > 0.5 && (t.distance ?? 99) < 6); return h.length ? `Careful, there's ${list(h.slice(0, 2))} in your path.` : "Nothing detected in your path for the next few meters. I can miss things, so check with your cane."; }
  if (!near.length) return "I could not confirm objects in the current camera view. This does not mean the area is clear.";
  return near.length === 1 ? `I see ${desc(near[0])}.` : `I see ${near.length} things. The closest are ${list(near.slice(0, 3))}.`;
}
/** Speak an answer/confirmation; speech's onSpoken updates the last-cue line and (only when muted) #sr-live, once. */
function say(text: string) {
  speech.sayNow(text, now());
}

// ---- UI wiring ----------------------------------------------------------------------
$("start").onclick = async () => {
  if (state.running) { state.running = false; cam.stop(); speech.stop(); $("start").textContent = "Start"; setStatus("Camera stopped."); if (reloadWhenIdle) location.reload(); return; }
  speech.unlock();
  try {
    const s = await cam.start({ facing: cam.facing });
    camModel.aspect = (s?.width ?? 4) / (s?.height ?? 3);
    state.mirrored = cam.mirrored; state.running = true; state.t0 = performance.now(); state.lastDecision = -1; state.lastOcr = state.lastDepth = -1e9; state.ocrFails = 0;
    tracker.reset(); resetSchedulers(); speech.reset(); dropText();
    // "I'll tell you what matters" only once the detector runs: a failed or loading one is said, never left as silence
    const notReady = () => (detectorState === "failed" ? DETECTOR_OFF : "Camera on. Object detection is still loading.");
    $("start").textContent = "Stop"; setStatus(detectorState === "ready" ? "Camera on" : notReady()); loop();
    await server.check(); pickVoice(); if (server.available && detectorState === "ready") setStatus(`Server connected (depth: ${server.depthModel})`);
    if (speech.enabled) say(detectorState === "ready" ? "Camera on. I'll tell you what matters." : notReady()); // muted: the status region already announced it
    const devices = await cam.listDevices(); const sel = $<HTMLSelectElement>("camera-select");
    sel.innerHTML = `<option value="">auto</option>` + devices.map((d, i) => `<option value="${d.deviceId}">${d.label || "camera " + (i + 1)}</option>`).join("");
  } catch (e) { setStatus(`Camera error: ${(e as Error).message}. Use HTTPS or localhost and allow the camera.`); }
};
$("flip").onclick = async () => { if (!state.running) return; await cam.flip(); state.mirrored = cam.mirrored; tracker.reset(); resetSchedulers(); dropText(); state.lastDecision = -1; };
$<HTMLSelectElement>("camera-select").onchange = async (e) => { const id = (e.target as HTMLSelectElement).value; if (state.running) { await cam.start({ deviceId: id }); state.mirrored = cam.mirrored; tracker.reset(); resetSchedulers(); dropText(); state.lastDecision = -1; } };
$<HTMLInputElement>("file-input").onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0]; if (!f) return;
  try { await cam.playFile(f); }
  catch (err) { setStatus(`Could not play that video: ${String((err as Error)?.message ?? err).split("\n")[0].slice(0, 80)}`); return; }
  state.mirrored = false; state.running = true; state.t0 = performance.now(); state.lastDecision = -1; state.lastOcr = state.lastDepth = -1e9; state.ocrFails = 0;
  tracker.reset(); resetSchedulers(); speech.reset(); dropText();
  $("start").textContent = "Stop"; setStatus(detectorState === "failed" ? DETECTOR_OFF : "Playing selected video."); loop(); await server.check(); pickVoice();
};
void server.check().then(pickVoice);
$("mute").onclick = () => { speech.enabled = !speech.enabled; $("mute").setAttribute("aria-pressed", String(!speech.enabled)); if (!speech.enabled) speech.stop(); };
$<HTMLSelectElement>("verbosity").onchange = (e) => { cfg.verbosity = (e.target as HTMLSelectElement).value as any; };
$<HTMLSelectElement>("units").onchange = (e) => { cfg.units = (e.target as HTMLSelectElement).value as any; };
voiceSel.onchange = async () => {
  if (voiceSel.value === "elevenlabs") await server.check();   // the server may have started since the page loaded
  pickVoice();
  if (voiceSel.value === "elevenlabs" && speech.backend !== "elevenlabs") setStatus("The ElevenLabs voice needs the SeeForMe server with an ElevenLabs API key. Using the browser voice.");
};
document.querySelectorAll<HTMLButtonElement>(".mode").forEach((b) => b.onclick = () => {
  state.mode = b.dataset.mode as Mode;
  document.querySelectorAll(".mode").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-pressed", String(x === b)); });
  $("find-form").classList.toggle("hidden", state.mode !== "find"); $("read-panel").classList.toggle("hidden", state.mode !== "read"); $("ask-form").classList.toggle("hidden", state.mode !== "ask");
  if (state.mode === "navigate") setGoal([]);
});
$<HTMLFormElement>("find-form").onsubmit = (e) => {
  e.preventDefault(); const v = $<HTMLInputElement>("find-input").value.trim(); const tags = goalTags(v);
  if (v && !tags.length) { say("Tell me what to look for, for example a chair or the exit."); return; }   // keep the current goal
  setGoal(tags);
  say(!v ? "Find cleared." : `Looking for ${v}.${!server.available && !objectGoal(tags) ? ` ${SERVER_NOTE}` : ""}`);
};
$("find-clear").onclick = () => { $<HTMLInputElement>("find-input").value = ""; setGoal([]); say("Find cleared."); };
$("read-now").onclick = async () => {
  if (!server.available) { say("Reading text needs the SeeForMe server. Start it with make serve."); return; }
  say("Reading.");
  try {
    const tx = await server.ocr(frameCanvas);
    say(tx.length ? tx.map((t) => t.text).join(". ") : "I can't read any text here.");
  } catch { say("Sorry, reading failed. Check the connection to the SeeForMe server and try again."); }
};
$<HTMLFormElement>("ask-form").onsubmit = (e) => { e.preventDefault(); const q = $<HTMLInputElement>("ask-input").value.trim(); if (q) say(answer(q)); };
function listen() {
  const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!SR) { say("Voice input isn't available in this browser. Type your question in Ask."); return; }
  speech.stop(); speech.unlock();
  const mic = $("mic"); mic.classList.add("listening"); mic.textContent = "Listening…"; setStatus("Listening…");
  const r = new SR(); r.lang = "en-US"; r.interimResults = false; r.maxAlternatives = 1;
  const done = () => { mic.classList.remove("listening"); mic.textContent = "🎤 Ask"; };
  r.onresult = (ev: any) => { const q = ev.results[0][0].transcript; $<HTMLInputElement>("ask-input").value = q; setStatus(`You asked: ${q}`); say(answer(q)); };
  r.onerror = (ev: any) => { done(); say(ev.error === "not-allowed" ? "I need microphone permission to hear you." : "Sorry, I didn't catch that."); };
  r.onend = done;
  r.start();
}
$("ask-voice").onclick = listen;
$("mic").onclick = listen;
$("test-voice").onclick = () => { speech.unlock(); say("Hi, this is SeeForMe. Stairs down, about 3 meters. Handrail left."); };
document.addEventListener("keydown", (e) => {
  // No shortcuts with modifiers (Cmd/Ctrl+F is the browser's find) or on key repeat, and never steal Space or letters from
  // buttons, links, <summary>, form controls or other focusable widgets (tabindex="-1" targets such as #controls still get them).
  if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  if (e.target instanceof HTMLElement && e.target.closest('button, input, textarea, select, a[href], summary, details, [contenteditable], [tabindex]:not([tabindex="-1"])')) return;
  if (e.key === " ") { e.preventDefault(); $("start").click(); }
  if (e.key.toLowerCase() === "f") $("flip").click();
  if (e.key.toLowerCase() === "m") $("mute").click();
  if (e.key.toLowerCase() === "a") listen();
});
updateBudget();

// ---- version and updates ----------------------------------------------------------------
$("app-version").textContent = `v${APP_VERSION}`;
let reloadWhenIdle = false;
/** "Updated to version X", once per release: the previous version is remembered in localStorage (0.2.0 never stored it,
 *  so its leftover "seeforme-models" cache marks an install of that release). Never shown on a first visit. */
async function noteAppVersion() {
  let prev = store.get("seeforme:version");
  let keys: string[] = [];
  try { keys = "caches" in window ? await caches.keys() : []; } catch { /* ignore */ }
  if (prev == null && keys.includes("seeforme-models")) prev = "0.2.0";
  if (prev != null && prev !== APP_VERSION) announceUpdate(`Updated to version ${APP_VERSION}.`);
  store.set("seeforme:version", APP_VERSION);
  // caches of earlier releases that this release's service worker no longer reads (0.2.0 runtime cache, older ORT wasm)
  for (const k of keys) if (k === "seeforme-models" || (k.startsWith("seeforme-ort-") && k !== __SEEFORME_ORT_CACHE__)) void caches.delete(k).catch(() => {});
}
function noteModelVersion(model: string) {
  const prev = store.get("seeforme:model");
  if (prev != null && prev !== model) announceUpdate(`Scheduler model updated to ${model}.`);
  store.set("seeforme:model", model);
}
function announceUpdate(msg: string) {
  $("app-version").textContent = `v${APP_VERSION} · updated`;
  if (detectorState !== "loading") { setStatus(`${msg} ${status.textContent ?? ""}`.trim()); return; }   // "Ready…" (or the failure) is already shown: keep it after
  notice = notice ? `${notice} ${msg}` : msg;       // otherwise the "Ready…" status repeats it (takeNotice)
  setStatus(notice);
}
/** The service worker (vite-plugin-pwa, autoUpdate: skipWaiting + clientsClaim) activates a new release at once and the
 *  page reloads when it takes over, but never in the middle of a walk: with the camera running the reload waits for Stop. */
function registerServiceWorker() {
  if (!("serviceWorker" in navigator) || import.meta.env.DEV) return;
  const hadController = navigator.serviceWorker.controller != null;
  let reloading = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (!hadController || reloading) return;   // first install: nothing to replace
    if (state.running) { reloadWhenIdle = true; setStatus("A new version is ready. It loads when you stop the camera."); return; }
    reloading = true; location.reload();
  });
  navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch((e) => console.warn("service worker:", e));
}
void noteAppVersion();
registerServiceWorker();
