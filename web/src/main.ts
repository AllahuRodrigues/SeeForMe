// The SeeForMe web app: camera -> YOLO11 (detector worker) -> tracker with metric dynamics (server depth when it runs)
// -> scheduler (learned GRU + RL, or the linear P-CACS) -> announcement gate (gate.ts: no repeats of what was just said,
// the desk user and still background held, approaching / critical / FIND goals always through) -> speech, haptics.
// The activity detector (activity.ts) picks the gate's hold times and finds a desk webcam; the cane consumer (cane.ts)
// adds class-agnostic obstacle cues from the server's depth map; voice.ts takes spoken commands.
import { ActivityDetector, grayFromRGBA, type Activity } from "./activity";
import { CameraSource } from "./camera";
import { CameraControl } from "./camcontrol";
import { CaneConsumer, DEFAULT_AID, aidAdvice, isAid, type TravelAid } from "./cane";
import { distancePhrase } from "./compose";
import { AnnouncementGate } from "./gate";
import { LearnedScheduler, loadLearnedRunner, type SchedulerMeta } from "./learned";
import { PCACS, DEFAULT_CFG, type Candidate } from "./pcacs";
import { ServerClient } from "./server";
import { Speech, earcon, haptic } from "./speech";
import { ClipVoice } from "./clipvoice";
import { Camera, Tracker, area, iou } from "./tracker";
import type { BBox, ChannelState, Cue, Detection, Track } from "./types";
import { APP_VERSION } from "./version";
import { HELP_TEXT, IDLE_TEXT, VoiceInput, findTarget, normaliseSpeech, parseCommand, type Command, type VoiceState } from "./voice";
import { COCO } from "./yolo";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
/** an element the page may not have (ids added in later page layouts) */
const opt = <T extends HTMLElement>(id: string) => document.getElementById(id) as T | null;
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
// The announcement gate wraps whichever scheduler is active; the activity detector and the cane feed it.
const gate = new AnnouncementGate(camModel);
gate.policy = policy;
const activity = new ActivityDetector();
const cane = new CaneConsumer(camModel);
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
const speech = new Speech(serverUrl, (text, urgent) => { lastCueEl.textContent = text; if (!speech.enabled) (urgent ? srAlert : srLive).textContent = text; },
  () => policy);   // measured playback is settled against the active scheduler's speech budget
const voiceSel = $<HTMLSelectElement>("voice");
/** The recorded natural voice by default; the browser voice on request; ElevenLabs only when the server says its TTS
 *  works (an API key is set). */
function pickVoice() {
  speech.backend = voiceSel.value === "elevenlabs" ? (server.available && server.tts ? "elevenlabs" : "browser") : voiceSel.value === "browser" ? "browser" : "natural";
}
// the recorded voice pack (about 3 MB) is fetched and decoded in the background; until then the browser voice speaks
speech.clips = new ClipVoice(`${import.meta.env.BASE_URL}voice/af_heart`);
void speech.clips.load();
pickVoice();
const frameCanvas = document.createElement("canvas");
const fctx = frameCanvas.getContext("2d", { willReadFrequently: true })!;
const smallCanvas = document.createElement("canvas");   // 160 px grey copy for the activity detector
const sctx = smallCanvas.getContext("2d", { willReadFrequently: true })!;
const octx = overlay.getContext("2d")!;

type Mode = "navigate" | "find" | "read" | "ask";
const state = { mode: "navigate" as Mode, running: false, backend: "", model: "", detMs: 0, lastDecision: -1, t0: performance.now(), cues: [] as Cue[],
  textCache: [] as Detection[], lastOcr: -1e9, ocrT: -1e9, ocrFails: 0, depthOk: false, lastDepth: -1e9, outdoor: false, debug: true, lastCueT: -1e9,
  decisions: 0, silentDecisions: 0, speechSeconds: 0, held: 0,
  activity: "unclear" as Activity, walkingSince: -1, desk: false, gateOn: true, aid: DEFAULT_AID as TravelAid };
const now = () => (performance.now() - state.t0) / 1000;

// ---- detector worker -------------------------------------------------------
let worker: Worker;
let inflight = false; let detId = 0; let readyTimer = 0; let timeouts = 0; let errors = 0;
/** The one detector state the Start button, the status messages and the scheduler notes read ("ready" again after a restart). */
let detectorState: "loading" | "ready" | "failed" = "loading";
const DETECT_TIMEOUT_MS = 5000, MAX_TIMEOUTS = 3, MAX_ERRORS = 3;
const DETECTOR_OFF = "Object detection is not working, obstacle cues are off. Reload the page to retry.";
const pending = new Map<number, (d: Detection[]) => void>();
/** Detector start-up ladder: WebGPU (YOLO11s) -> threaded WASM (YOLO11n) in a fresh worker -> single-threaded WASM ->
 *  failed, said out loud. `threads` / `wasmOnly` say which rung this worker is. */
let rung: { threads?: number; wasmOnly: boolean } = { wasmOnly: false };
function startWorker(threads?: number, wasmOnly = false) {
  if (worker) worker.terminate();
  rung = { threads, wasmOnly };
  detectorState = "loading"; inflight = false; errors = 0; pending.forEach((done) => done([])); pending.clear();
  const w = worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (ev) => {
    if (w !== worker) return;   // a superseded worker: its late messages must not skip a rung
    const m = ev.data;
    // Startup stages are diagnostics (aria-hidden #diag); the role=status region is set once, when the detector is ready.
    if (m.type === "stage") diag.textContent = `Detector: ${m.stage}…`;
    if (m.type === "error") fallback(m.message.split("\n")[0]);
    if (m.type === "ready") {
      detectorState = "ready"; clearTimeout(readyTimer); state.backend = m.backend; state.model = m.model ?? ""; diag.textContent = `Detector ready (${state.model}, ${m.backend})`;
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
          if (wasmOnly && threads === 1) { worker.terminate(); detectorFailed(String(m.error).split("\n")[0].slice(0, 80)); }
          else { setStatus("Detector error, restarting."); fallback(String(m.error)); }
        }
      }
      done(m.dets ?? []);
    }
  };
  worker.onerror = () => { if (w === worker) fallback("the detector could not start"); };
  // YOLO11s when WebGPU works, YOLO11n on WASM
  worker.postMessage({ type: "init", gpuModel: __SEEFORME_MODELS__.gpu, cpuModel: __SEEFORME_MODELS__.cpu, threads, wasmOnly });
  clearTimeout(readyTimer);
  // the first attempt may download the 36 MB WebGPU model and compile it; the WASM model is 10 MB
  readyTimer = window.setTimeout(() => { if (detectorState === "loading") fallback("the detector did not load in time"); }, wasmOnly ? 25000 : 60000);
}
/** Final failure: say so in the status region (and aloud while the camera runs), so silence is never mistaken for "nothing ahead". */
function detectorFailed(why: string) {
  clearTimeout(readyTimer); detectorState = "failed";
  setStatus(`Detector failed: ${why}. Object cues are off; reload the page to retry.`);
  if (state.running && speech.enabled) say(DETECTOR_OFF);
  startLearned();
}
function fallback(why: string) {
  console.warn(`detector: ${why}`);
  if (!rung.wasmOnly) { diag.textContent = "Detector: WebGPU unavailable, using WASM…"; startWorker(undefined, true); }
  else if (rung.threads !== 1) { diag.textContent = "Detector: retrying single-threaded…"; startWorker(1, true); }
  else detectorFailed(why.slice(0, 80));
}
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
      if (++timeouts >= MAX_TIMEOUTS) { timeouts = 0; setStatus("Detector stalled, restarting."); startWorker(1, true); }
    }, DETECT_TIMEOUT_MS);
    pending.set(id, (d) => { clearTimeout(timer); res(d); });
    worker.postMessage({ type: "detect", id, img, conf: 0.3 }, [img.data.buffer]);
  });
}

/** #status is a live region, so a screen reader reads each change aloud, and an open microphone can hear that ("Lost the
 *  SeeForMe server..." contains the wake phrase): the voice input's echo guard gets the recent ones (statusRecent). */
const statusSaid: Array<[string, number]> = [];
function setStatus(s: string) {
  status.textContent = s;
  statusSaid.push([s, performance.now()]); if (statusSaid.length > 8) statusSaid.shift();
}
/** Status texts set in the last `ms` milliseconds plus their reading time (as Speech.recent). */
const statusRecent = (ms = 3000) => statusSaid.filter(([s, at]) => performance.now() - at < ms + 400 * s.split(/\s+/).length).map(([s]) => s);
/** A one-time notice ("Updated to version X") is also put in front of the next status message, so it is not lost
 *  when "Ready" replaces the status a moment later. */
let notice = "";
function takeNotice() { const n = notice; notice = ""; return n ? `${n} ` : ""; }

// ---- schedulers ------------------------------------------------------------------
const schedSelect = $<HTMLSelectElement>("scheduler"), schedInfo = $("sched-info"), schedTag = $("sched-tag"), formula = $("formula"), scoreHead = $("score-head");
const LINEAR_FORMULA = formula.innerHTML;
const learnedFormula = (b: number, K: number) => `per candidate: GRU over its last ${K} feature vectors → log-odds of speak / interrupt / haptic against silence (illegal actions masked); the largest acts if it beats the silence bias b = ${b.toFixed(2)}`;
function setGoal(tags: string[]) { linear.setGoal(tags); learned?.setGoal(tags); gate.setGoal(tags); applyGateMode(); }
const hasGoal = () => linear.goalTags.length > 0;
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
  gate.policy = policy;   // the gate refunds held cues to (and charges its own cues from) the active scheduler's budget
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
  const opt_ = schedSelect.querySelector<HTMLOptionElement>('option[value="learned"]');
  if (opt_) { opt_.disabled = true; opt_.textContent = "Learned (unavailable)"; }
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
      schedInfo.textContent = `Learned scheduler: GRU + RL trained on CampusBench (synthetic campus scenes, not EgoBlind video), model ${meta.version}, last ${meta.K} steps per track, silence bias ${meta.silence_bias.toFixed(2)}.`;
      noteModelVersion(meta.version);
      useScheduler(wanted, false);
    } catch (e) { learnedUnavailable(String((e as Error)?.message ?? e).split("\n")[0].slice(0, 120)); }
  })();
}
window.setTimeout(startLearned, 30000);   // the detector never reported back: load the scheduler anyway
schedSelect.value = wanted;   // until the model is ready the linear P-CACS runs (the footer tag says which one is active)
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

// ---- the repeat filter (gate) and what the camera is doing ------------------------------------------------------
/** The gate's settings follow the activity: still -> 60 s hold and still objects as background (re-learnt each time the
 *  camera comes to rest); walking or unclear -> 20 s hold, no background. A FIND goal keeps the walking settings, so
 *  static objects may come back sooner while the user searches. */
function applyGateMode(prev?: Activity) {
  const still = state.activity === "still" && !hasGoal();
  gate.setParams({ holdStatic: still ? 60 : 20, background: still });
  if (still && prev !== undefined && prev !== "still") gate.rebaseline();
}
/** Desk webcam: a camera that is still in the first seconds after Start is a laptop on a desk, and the person in front of
 *  it is the user (gate operator filter). Walking with the camera for 2 s ends that. */
function onActivity(t: number) {
  const a = activity.activity;
  if (a !== state.activity) { const prev = state.activity; state.activity = a; applyGateMode(prev); }
  state.walkingSince = a === "walking" ? (state.walkingSince < 0 ? t : state.walkingSince) : -1;
  const warm = gate.cfg.warmup;
  if (!state.desk && a === "still" && t >= warm + 1.5 && t < warm + 15) {
    state.desk = true; gate.setParams({ operatorFilter: true }); gate.rebaseline();
  } else if (state.desk && state.walkingSince >= 0 && t - state.walkingSince >= 2) {
    state.desk = false; gate.setParams({ operatorFilter: false }); gate.rebaseline();
  }
}
function observeActivity(t: number) {
  const W = 160, H = Math.max(16, Math.round((frameCanvas.height / frameCanvas.width) * W));
  if (smallCanvas.width !== W || smallCanvas.height !== H) { smallCanvas.width = W; smallCanvas.height = H; }
  sctx.drawImage(frameCanvas, 0, 0, W, H);
  activity.update(grayFromRGBA(sctx.getImageData(0, 0, W, H).data, W, H), W, H, t);
  onActivity(t);
}
const gateToggle = opt<HTMLInputElement | HTMLButtonElement>("gate-toggle");
/** Bare scheduler (every cue it proposes is spoken) or the gate; without a "gate-toggle" control the gate is always on. */
function setGate(on: boolean, announce: boolean) {
  state.gateOn = on;
  if (gateToggle instanceof HTMLInputElement) gateToggle.checked = on;
  else gateToggle?.setAttribute("aria-pressed", String(on));
  gate.reset(); cane.reset();
  if (announce) { store.set("seeforme:gate", on ? "on" : "off"); setStatus(on ? "Repeat filter on." : "Repeat filter off: every cue the scheduler proposes is spoken."); }
}
if (gateToggle) {
  setGate(store.get("seeforme:gate") !== "off", false);
  if (gateToggle instanceof HTMLInputElement) gateToggle.onchange = () => setGate(gateToggle.checked, true);
  else gateToggle.onclick = () => setGate(!state.gateOn, true);
}

// ---- metric depth and the cane ------------------------------------------------------------------------------------
// The server answers ~100-300 ms after a frame is sent, when the tracker has moved on: its distances are kept per box
// and given to the detections of the next frames that overlap those boxes (scaled by how much the box grew since).
let depthCache: { t: number; boxes: BBox[]; dist: Array<[number | null, number]> } | null = null;
let depthSeq = 0, depthDone = 0, depthInflight = 0;
const caneWanted = () => state.gateOn && (state.mode === "navigate" || state.mode === "find") && state.activity !== "still" && !state.desk;
function applyDepth(dets: Detection[], t: number) {
  const c = depthCache;
  if (!c || t - c.t > 1.0) return;
  for (const d of dets) {
    if (d.kind !== "object") continue;
    let best = 0.3, k = -1;
    c.boxes.forEach((b, i) => { const v = iou(b, d.bbox); if (c.dist[i]?.[0] != null && v >= best) { best = v; k = i; } });
    if (k < 0) continue;
    const [z, conf] = c.dist[k];
    const grow = Math.sqrt(area(c.boxes[k]) / Math.max(1e-6, area(d.bbox)));   // a box twice the area: ~1/sqrt(2) as far
    d.distance_m = (z as number) * Math.min(1.33, Math.max(0.75, grow));
    d.depth_conf = t - c.t > 0.6 ? conf * 0.85 : conf;
  }
}
function requestDepth(dets: Detection[], t: number) {
  if (!server.available || t - state.lastDepth <= 0.25 || depthInflight >= 2) return;
  const boxes = dets.filter((d) => d.kind === "object").map((d) => d.bbox);
  if (!boxes.length && !caneWanted()) return;
  state.lastDepth = t; depthInflight++;
  const seq = ++depthSeq;
  server.depth(frameCanvas, boxes, state.outdoor, { walking: state.activity === "walking", aid: state.aid })
    .then((r) => {
      state.depthOk = r != null;
      if (!r || seq <= depthDone) return;   // an older answer never replaces a newer one
      depthDone = seq; depthCache = { t, boxes, dist: r.distances }; cane.update(r.cane, t);
    })
    .catch(() => { state.depthOk = false; })
    .finally(() => { depthInflight--; });
}

// ---- main loop ---------------------------------------------------------------
// The picture is never mirrored, for any camera: boxes are drawn where the detector found the objects and "left" in
// speech is the left of the picture.
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
    const t = now();
    try { observeActivity(t); } catch (e) { console.warn("activity:", e); }
    const img = fctx.getImageData(0, 0, w, h);
    const dets = await detect(img);
    applyDepth(dets, t);
    // metric depth (and the cane summary) from the server at ~4 Hz; otherwise box-height guesses with low confidence
    requestDepth(dets, t);
    // periodic OCR through the server in Navigate/Find (text tracks persist between passes); a failed pass clears the
    // signs, a result older than 3 s is not used, and three failures in a row mark the server lost
    if (server.available && t - state.lastOcr > 2.0 && state.mode !== "read") {
      state.lastOcr = t;
      const seq = ++ocrSeq;
      server.ocr(frameCanvas).then((tx) => { if (seq <= ocrDone) return; ocrDone = seq; state.textCache = tx; state.ocrT = now(); state.ocrFails = 0; })
        .catch(() => { if (seq <= ocrDone) return; ocrDone = seq; state.textCache = []; if (++state.ocrFails >= 3 && server.available) { server.available = false; pickVoice(); setStatus("Lost the SeeForMe server: text reading and metric depth are off."); } });
    }
    const tracks = tracker.update([...dets, ...(t - state.ocrT < 3.0 ? state.textCache : [])], t);
    gate.observe([...tracks, ...cane.pseudoTracks(t)], t);
    await step(tracks, t);
    draw(tracks);
  }
  requestAnimationFrame(loop);
}

let deciding = false; let schedulerErrorShown = false; let gateErrorShown = false;
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
  const ch: ChannelState = { busy, remaining: speech.remaining(t), currentUtility: cur?.utility ?? 0, currentSeverity: cur?.severity ?? 0, currentTrack: cur?.trackId ?? null };
  let proposed: Cue | null;
  try { proposed = await policy.decide(tracks, t, ch); }
  catch (e) {
    if (!isLearned()) throw e;
    learnedUnavailable(`model error: ${String((e as Error)?.message ?? e).slice(0, 80)}`);
    proposed = linear.decide(tracks, t, ch);
  }
  let acts: Cue[] = proposed ? [proposed] : [], gateWhy = "";
  if (state.gateOn) {
    try {
      const caneCue = cane.propose({ t, tracks, navigating: state.mode === "navigate" || state.mode === "find", still: state.activity === "still",
        desk: state.desk, channelBusy: busy, channelTrack: ch.currentTrack, units: cfg.units, aid: state.aid });
      // a haptic stop from the cane goes first; otherwise the scheduler's cue, then the cane's. The gate judges them all
      // and lets one haptic and one speech cue through together, so a cane buzz never hides the scheduler's warning.
      const offered = caneCue?.action === "haptic" ? [caneCue, ...(proposed ? [proposed] : [])] : [...(proposed ? [proposed] : []), ...(caneCue ? [caneCue] : [])];
      acts = gate.filterAll(offered, [...tracks, ...cane.pseudoTracks(t)], t, ch);
      gateWhy = gate.verdicts.map((v) => v.text).join(" · ");
      if (proposed && !acts.includes(proposed)) state.held++;
      gateErrorShown = false;
    } catch (e) {
      // the gate fails open: a bug in the repeat filter must never silence a cue
      console.error("gate:", e);
      acts = proposed ? [proposed] : [];
      if (!gateErrorShown) { gateErrorShown = true; setStatus("Repeat filter error: every cue is spoken."); }
    }
  }
  updateBudget();
  state.decisions++; if (!acts.length) state.silentDecisions++;
  renderAlgorithm(policy.lastCandidates, acts, proposed, busy, t, gateWhy);
  renderSession(t);
  // the buzz first, then the words (the cue card ends on the spoken cue)
  const buzz = acts.find((c) => c.action === "haptic"), words = acts.find((c) => c.action !== "haptic");
  for (const cue of [buzz, words]) {
    if (!cue) continue;
    state.cues.push(cue); state.lastCueT = t;
    if (cue.action === "haptic") { haptic(); earcon(); showCue("BUZZ", cue.source === "cane" ? "Obstacle very close" : "Obstacle imminent", cue); continue; }
    state.speechSeconds += cue.duration;
    speech.speak(cue, t, cue.action === "interrupt");
    if (cue.action === "interrupt" && !buzz) haptic([40]);   // (a vibration pattern would cut the buzz short)
    showCue(cue.action.toUpperCase(), cue.message, cue);
  }
}

function showCue(kind: string, text: string, cue: Cue) {
  cueKind.textContent = kind; cueText.textContent = text;
  const d = cue.distance != null ? `${cfg.units === "ft" ? (cue.distance * 3.28).toFixed(0) + " ft" : cue.distance.toFixed(1) + " m"}` : "";
  const src = cue.source === "cane" ? "depth map" : cue.source === "gate" ? "repeat filter" : `${isLearned() ? "L" : "U"} ${cue.utility.toFixed(2)}`;
  cueMeta.textContent = [d, cue.ttc != null ? `TTC ${cue.ttc.toFixed(1)} s` : "", src].filter(Boolean).join(" · ");
  cueCard.classList.add("show"); cueCard.dataset.kind = cue.action;
  setTimeout(() => { if (now() - state.lastCueT > 2.4) cueCard.classList.remove("show"); }, 2600);
}
function updateBudget() {
  const f = Math.max(0, Math.min(1, policy.budget / cfg.budgetCapacity));
  budgetFill.style.width = `${f * 100}%`; budgetText.textContent = `${Math.round(f * 100)}%`;
}

// ---- overlay ---------------------------------------------------------------------
/** Where the video picture is inside the element (object-fit contain / cover / fill), so boxes sit on the objects. */
function videoRect(W: number, H: number) {
  const vw = video.videoWidth || W, vh = video.videoHeight || H;
  const fit = getComputedStyle(video).objectFit;
  if (fit === "fill") return { x: 0, y: 0, w: W, h: H };
  const s = fit === "cover" ? Math.max(W / vw, H / vh) : Math.min(W / vw, H / vh);
  return { x: (W - vw * s) / 2, y: (H - vh * s) / 2, w: vw * s, h: vh * s };
}
function draw(tracks: Track[]) {
  const W = (overlay.width = video.clientWidth), H = (overlay.height = video.clientHeight);
  octx.clearRect(0, 0, W, H);
  const r = videoRect(W, H);
  const cands = new Map(policy.lastCandidates.map((c) => [c.track.id, c]));
  const box = (b: BBox, color: string, width: number, label: string, dashed = false) => {
    const x = r.x + b[0] * r.w, y = r.y + b[1] * r.h, w = (b[2] - b[0]) * r.w, h = (b[3] - b[1]) * r.h;
    octx.setLineDash(dashed ? [6, 4] : []); octx.strokeStyle = color; octx.lineWidth = width; octx.strokeRect(x, y, w, h); octx.setLineDash([]);
    octx.font = "600 13px system-ui"; const tw = octx.measureText(label).width + 8;
    octx.fillStyle = "rgba(0,0,0,.65)"; octx.fillRect(x, Math.max(0, y - 18), tw, 18);
    octx.fillStyle = "#fff"; octx.fillText(label, x + 4, Math.max(13, y - 5));
  };
  const fmtD = (d: number | null) => (d == null ? "" : cfg.units === "ft" ? `${(d * 3.28).toFixed(0)}ft` : `${d.toFixed(1)}m`);
  for (const tr of tracks) {
    const c = cands.get(tr.id), hot = c != null && isHot(c);
    box(tr.bbox, tr.kind === "text" ? "#f0b13b" : hot ? "#ff5a5a" : "#5ee0b4", hot ? 3 : 2,
      `${tr.kind === "text" ? tr.text : tr.label} ${fmtD(tr.distance)}${Number.isFinite(tr.ttc) ? ` ${tr.ttc.toFixed(1)}s` : ""}`);
  }
  const o = cane.last && now() - cane.last.t < 1 ? cane.last.summary.obstacle : null;
  if (o && state.gateOn) box(o.bbox, "#b48cff", 2, `${o.level === "head" ? "head-height " : ""}obstacle ${fmtD(o.distance_m)}`, true);
  // Per-frame diagnostics are visual only (aria-hidden); live-region changes every frame overwhelm screen readers.
  diag.textContent = `${state.running ? "LIVE" : "paused"} · ${state.model || "loading model"} ${state.backend} · det ${state.detMs.toFixed(0)} ms · ${tracks.length} tracks · depth ${server.available ? (state.depthOk ? "server" : "…") : "bbox"} · ${activity.activity}${state.desk ? " (desk)" : ""} · repeat filter ${state.gateOn ? `on (${state.held} held)` : "off"}${speech.enabled ? "" : " · muted"}${cfgDefaults ? " · built-in config" : ""}`;
  liveDot.classList.toggle("live", state.running);
}
/** The value the active scheduler acts on: linear utility U, or the learned best legal log-odds L against silence. */
const candValue = (c: Candidate) => (isLearned() ? (c.score ?? -Infinity) : c.utility);
const threshold = () => (learned && isLearned() ? learned.silenceBias : cfg.thresholdSpeak);
const isHot = (c: Candidate) => (isLearned() ? candValue(c) > threshold() : candValue(c) >= threshold());
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
const esc = (s: string) => s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]!);
function renderAlgorithm(cands: Candidate[], acts: Cue[], proposed: Cue | null, busy: boolean, t: number, gateWhy: string) {
  const thr = threshold();
  const maxU = Math.max(thr * 1.5, ...cands.map(candValue).filter(Number.isFinite));
  candRows.innerHTML = cands.length ? cands.slice(0, 7).map((c) => {
    const v = candValue(c), hot = isHot(c);
    const w = Math.max(2, Math.round((Math.max(0, Number.isFinite(v) ? v : 0) / maxU) * 44));
    const name = esc(c.track.kind === "text" ? `“${(c.track.text ?? "").slice(0, 16)}”` : c.track.label);
    return `<tr class="${hot ? "hot" : ""}"><td>${name}</td><td>${c.distance.toFixed(1)}</td><td>${c.ttc >= 10 ? "–" : c.ttc.toFixed(1)}</td><td>${c.pathProb.toFixed(2)}</td><td>${c.hazard.toFixed(2)}</td><td>${c.urgency.toFixed(2)}</td><td>${c.novelty.toFixed(2)}</td><td>${c.speechCost.toFixed(2)}</td><td><span class="ubar" style="width:${w}px"></span>${fmt(v)}</td></tr>`;
  }).join("") : `<tr><td colspan="9" class="muted">no tracks</td></tr>`;
  actionNow.textContent = acts.map((c) => c.action).join(" + ") || "silence";
  actionNow.className = (acts.find((c) => c.action !== "haptic") ?? acts[0])?.action ?? "silence";
  const top = cands[0];
  let why: string;
  if (isLearned()) why = proposed ? `L ${proposed.utility.toFixed(2)} > b ${thr.toFixed(2)}` : top ? (Number.isFinite(candValue(top)) ? `best L ${candValue(top).toFixed(2)} ≤ b ${thr.toFixed(2)}` : "no legal action (just announced)") : "";
  else why = proposed ? `U ${proposed.utility.toFixed(2)} ≥ ${thr.toFixed(2)}` : top ? (top.utility < thr ? `best U ${top.utility.toFixed(2)} < ${thr.toFixed(2)}` : busy ? "channel busy" : "blocked (recent / budget)") : "";
  // the gate's verdict: "held: same person ahead, said 4 s ago", "new: chair left", "said by the gate, closing in: ..."
  actionWhy.textContent = gateWhy ? (proposed && gateWhy.startsWith("held") ? gateWhy : [why, gateWhy].filter(Boolean).join(" · ")) : why;
  channelState.textContent = busy ? `busy · ${speech.remaining(t).toFixed(1)} s left` : "free";
  egoSpeedEl.textContent = `${policy.egoSpeed.toFixed(1)} m/s`;
  depthSrc.textContent = server.available ? (state.depthOk ? `server · ${server.depthModel}` : "server · waiting") : "bbox height (hedged)";
  debugLog.textContent = state.cues.slice(-5).map((c) => `${c.t.toFixed(1)}s ${c.action.padEnd(9)} ${c.message || "(buzz)"}${c.source && c.source !== "scheduler" ? ` [${c.source}]` : ""}`).join("\n");
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
  // measured playback: seconds the voice really played against the seconds the scheduler booked for those cues
  const pb = speech.ledger.summary();
  $("s-played").textContent = `${pb.playedS.toFixed(1)} / ${pb.reservedS.toFixed(1)} s`;
  $("s-cut").textContent = String(pb.cues - pb.outcomes.completed);
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
    return `<div class="b ${cls}"><span class="n">${esc(String(b.name))}</span><span class="t"><i style="width:${Math.max(1, (v / max) * 100)}%"></i></span><span class="v">${v.toFixed(2)}</span></div>`;
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
const visibleTracks = () => [...tracker.tracks.values()].filter((t) => t.misses === 0 && t.confirmed).sort((a, b) => (a.distance ?? 99) - (b.distance ?? 99));
const dirOf = (t: Track) => { const a = camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2); return Math.abs(a) < 10 ? "straight ahead" : Math.abs(a) < 25 ? (a > 0 ? "slightly to your right" : "slightly to your left") : (a > 0 ? "on your right" : "on your left"); };
const distOf = (t: Track) => { if (t.distance == null) return ""; const d = t.distance; if (t.depthConf < 0.45) return d < 2.5 ? "very close" : d < 4.5 ? "fairly close" : "further away"; const m = cfg.units === "ft" ? `${Math.round(d * 3.28)} feet` : `${d < 3 ? (Math.round(d * 2) / 2) : Math.round(d)} meters`; return `about ${m} away`; };
const article = (w: string) => (/^[aeiou]/i.test(w) ? "an" : "a");
const desc = (t: Track) => t.kind === "text" ? `a sign that says ${t.text}, ${dirOf(t)}` : `${article(t.label)} ${t.label} ${distOf(t)}${distOf(t) ? ", " : ""}${dirOf(t)}`;
const list = (ts: Track[]) => ts.length === 1 ? desc(ts[0]) : ts.slice(0, -1).map(desc).join(", ") + ", and " + desc(ts[ts.length - 1]);
/** Set a FIND goal (form, Ask, voice): what to say back. An empty or meaningless target keeps the current goal. */
function lookFor(what: string): string {
  const tags = goalTags(what);
  if (!tags.length) return "Tell me what to look for, for example a chair or the exit.";
  setGoal(tags); $<HTMLInputElement>("find-input").value = what;
  const hit = visibleTracks().find((t) => matchesGoal(t, tags));
  return hit ? `I can see it now: ${desc(hit)}.` : `Looking for ${what}. I'll tell you as soon as I see it.${!server.available && !objectGoal(tags) ? ` ${SERVER_NOTE}` : ""}`;
}
/** The cane's current obstacle as words, for answers ("the depth map shows an obstacle ahead, about 1 meter"). */
function caneNote(): string {
  const o = cane.last && now() - cane.last.t < 1.5 ? cane.last.summary.obstacle : null;
  if (!o || o.confidence < 0.5) return "";
  const d = distancePhrase(o.distance_m, 0.9, cfg.units);
  return `The depth map shows ${o.level === "head" ? "something at head height" : "an obstacle"} ${o.sector === "ahead" ? "ahead" : `on your ${o.sector}`}${d ? `, ${d}` : ""}.`;
}
function answer(q: string): string {
  const near = visibleTracks();
  q = q.toLowerCase();
  const find = q.match(/(?:where(?:'s| is)|find|look for)\s+(?:the |a |an )?([a-z0-9 ]+)/);
  if (find) { const what = find[1].trim().replace(/[?.!]+$/, ""); return lookFor(findTarget(normaliseSpeech(what)) || what); }
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
  if (/ahead|in front/.test(q)) { const h = near.filter((t) => Math.abs(camModel.angleDeg((t.bbox[0] + t.bbox[2]) / 2)) < 12); return h.length ? `Ahead of you there's ${list(h.slice(0, 3))}.` : `I could not confirm anything ahead in the current camera view. That does not mean it is clear.${caneNote() ? ` ${caneNote()}` : ""}`; }
  if (/detail|more|repeat|again|what did you say/.test(q)) {
    const c = state.cues.filter((c) => c.action !== "haptic").at(-1); if (!c) return "I haven't said anything yet.";
    // as in the cues: a number only with server depth, "very close" / "close" for a bounding-box guess, in the chosen units
    const p = distancePhrase(c.distance, c.depthConf ?? 0, cfg.units);
    return `I said: ${c.message}${p ? ` That's ${p}${p.startsWith("about") ? " away" : ""}.` : ""}`;
  }
  // never "clear": what was not detected may still be there
  if (/clear|safe|path|way/.test(q)) {
    const h = near.filter((t) => t.pathProb > 0.5 && (t.distance ?? 99) < 6);
    if (h.length) return `Careful, there's ${list(h.slice(0, 2))} in your path.${caneNote() ? ` ${caneNote()}` : ""}`;
    // the user's own aid ("I travel with"), as seeforme/live/commands.py answer()
    const note = caneNote();
    return note ? `${note} I can miss things, so ${aidAdvice(state.aid)}.`
      : `I did not detect anything in your path, but that does not mean it is clear. I can miss obstacles, so ${aidAdvice(state.aid)}.`;
  }
  if (!near.length) return `I could not confirm objects in the current camera view. This does not mean the area is clear.${caneNote() ? ` ${caneNote()}` : ""}`;
  return near.length === 1 ? `I see ${desc(near[0])}.` : `I see ${near.length} things. The closest are ${list(near.slice(0, 3))}.`;
}
/** Speak an answer/confirmation; speech's onSpoken updates the last-cue line and (only when muted) #sr-live, once. */
function say(text: string) {
  speech.sayNow(text, now());
}

// ---- camera, modes and the other controls ---------------------------------------------------------------------------
const app = opt("app"), view = opt("view");
/** A new view (Start, flip, another camera or video): nothing seen or said before is about it. */
function resetSession() {
  tracker.reset(); resetSchedulers(); gate.reset(); cane.reset(); activity.reset(); dropText();
  depthCache = null; state.activity = "unclear"; state.walkingSince = -1; state.desk = false;
  gate.setParams({ operatorFilter: false }); applyGateMode();
  state.lastDecision = -1; state.lastOcr = state.lastDepth = -1e9; state.ocrFails = 0;
}
function setRunning(on: boolean) {
  state.running = on;
  $("start").textContent = on ? "Stop" : "Start";
  if (app) app.dataset.running = String(on);
}
/** The stream's shape on #view (--vw / --vh), so the box keeps the picture's aspect and the overlay lines up. */
function noteVideoShape() {
  if (view && video.videoWidth && video.videoHeight) { view.style.setProperty("--vw", String(video.videoWidth)); view.style.setProperty("--vh", String(video.videoHeight)); }
}
video.addEventListener("loadedmetadata", noteVideoShape);
video.addEventListener("resize", noteVideoShape);
/** The camera's shape: directions in speech depend on it. */
const noteAspect = (s: MediaTrackSettings | undefined) => { camModel.aspect = (s?.width ?? 4) / (s?.height ?? 3); };
// "I'll tell you what matters" only once the detector runs: a failed or loading one is said, never left as silence
const notReady = () => (detectorState === "failed" ? DETECTOR_OFF : "Camera on. Object detection is still loading.");
/** Every way of opening or stopping the camera: Start / Stop and Space, Flip and F, the camera picker, and the voice
 *  commands (camcontrol.ts). A pause, goodbye, Stop or newer open always wins over an open in progress, so no camera
 *  stays on after "Paused. The camera is off." or "Goodbye", and a cancelled open says nothing. */
const camCtl = new CameraControl(cam, {
  running: () => state.running,
  unlock: () => speech.unlock(),
  begin: (s) => {
    noteAspect(s);
    state.t0 = performance.now(); setRunning(true);
    resetSession(); speech.reset(); noteVideoShape();
    setStatus(detectorState === "ready" ? "Camera on" : notReady()); loop();
  },
  newView: (s) => { noteAspect(s); resetSession(); noteVideoShape(); },
  checkServer: async () => { await server.check(); pickVoice(); },
  announce: () => {
    if (server.available && detectorState === "ready") setStatus(`Server connected (depth: ${server.depthModel})`);
    if (speech.enabled) say(detectorState === "ready" ? "Camera on. I'll tell you what matters." : notReady()); // muted: the status region already announced it
  },
  devices: (devices) => {
    $<HTMLSelectElement>("camera-select").innerHTML = `<option value="">auto</option>`
      + devices.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || "camera " + (i + 1))}</option>`).join("");
  },
  end: (msg) => {
    setRunning(false); speech.stop(); setStatus(msg);
    if (reloadWhenIdle) location.reload();
  },
  status: setStatus,
  say,
  sayAfter,
});
$("start").onclick = () => camCtl.toggle();
$("flip").onclick = () => camCtl.next();
$<HTMLSelectElement>("camera-select").onchange = (e) => camCtl.pick((e.target as HTMLSelectElement).value);
$<HTMLInputElement>("file-input").onchange = async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0]; if (!f) return;
  camCtl.useFile();   // an open in progress and a start's "Camera on" end: the file replaces the camera
  try { await cam.playFile(f); }
  catch (err) {
    // playFile closed the camera: stop the loop too, or it would keep announcing the last frame as if it were live
    camCtl.stop(`Could not play that video: ${String((err as Error)?.message ?? err).split("\n")[0].slice(0, 80)}`);
    return;
  }
  state.t0 = performance.now(); setRunning(true);
  resetSession(); speech.reset(); noteVideoShape();
  setStatus(detectorState === "failed" ? DETECTOR_OFF : "Playing selected video."); loop(); await server.check(); pickVoice();
};
void server.check().then(pickVoice);
function setMuted(muted: boolean) { speech.enabled = !muted; $("mute").setAttribute("aria-pressed", String(muted)); if (muted) speech.stop(); }
$("mute").onclick = () => setMuted(speech.enabled);
const verbositySel = $<HTMLSelectElement>("verbosity");
verbositySel.onchange = () => { cfg.verbosity = verbositySel.value as typeof cfg.verbosity; };
$<HTMLSelectElement>("units").onchange = (e) => { cfg.units = (e.target as HTMLSelectElement).value as typeof cfg.units; };
// "I travel with": what the depth map's obstacle cues leave to the user's own cane or dog (cane.ts); remembered. The
// scheduler's cues about detected objects do not depend on it.
const aidSel = opt<HTMLSelectElement>("aid");
const AID_SAY: Record<TravelAid, string> = {
  cane: "White cane: obstacle cues skip what your cane can reach and always warn at head height.",
  "guide-dog": "Guide dog: obstacle cues only warn at head height and buzz to stop.",
  none: "No cane or guide dog: every obstacle cue from the depth map is said.",
};
function setAid(aid: TravelAid, announce: boolean) {
  state.aid = aid;
  if (aidSel) aidSel.value = aid;
  if (announce) { store.set("seeforme:aid", aid); setStatus(AID_SAY[aid]); }
}
{ const saved = store.get("seeforme:aid"); setAid(isAid(saved) ? saved : DEFAULT_AID, false); }
if (aidSel) aidSel.onchange = () => setAid(isAid(aidSel.value) ? aidSel.value : DEFAULT_AID, true);
voiceSel.onchange = async () => {
  if (voiceSel.value === "elevenlabs") await server.check();   // the server may have started since the page loaded
  pickVoice();
  if (voiceSel.value === "elevenlabs" && speech.backend !== "elevenlabs") setStatus("The ElevenLabs voice needs the SeeForMe server with an ElevenLabs API key. Using the browser voice.");
};
function setMode(m: Mode) {
  state.mode = m;
  document.querySelectorAll<HTMLElement>(".mode").forEach((x) => { const on = x.dataset.mode === m; x.classList.toggle("active", on); x.setAttribute("aria-pressed", String(on)); });
  $("find-form").classList.toggle("hidden", m !== "find"); $("read-panel").classList.toggle("hidden", m !== "read"); $("ask-form").classList.toggle("hidden", m !== "ask");
  if (m === "navigate") setGoal([]);
}
document.querySelectorAll<HTMLButtonElement>(".mode").forEach((b) => b.onclick = () => setMode(b.dataset.mode as Mode));
$<HTMLFormElement>("find-form").onsubmit = (e) => {
  e.preventDefault(); const v = $<HTMLInputElement>("find-input").value.trim();
  if (!v) { setGoal([]); say("Find cleared."); return; }
  say(lookFor(findTarget(normaliseSpeech(v)) || v));   // as spoken targets: "room two oh four" is room 204, no "the"
};
function clearFind() { $<HTMLInputElement>("find-input").value = ""; setGoal([]); }
$("find-clear").onclick = () => { clearFind(); say("Find cleared."); };
async function readNow() {
  if (!server.available) { say("Reading text needs the SeeForMe server. Start it with make serve."); return; }
  if (!state.running) { say("The camera is off. Start it first."); return; }
  say("Reading.");
  try {
    const tx = await server.ocr(frameCanvas);
    say(tx.length ? tx.map((t) => t.text).join(". ") : "I can't read any text here.");
  } catch { say("Sorry, reading failed. Check the connection to the SeeForMe server and try again."); }
}
$("read-now").onclick = () => void readNow();
/** Typed text in Ask goes through the same parser as speech ("direct": no wake phrase needed): "help", "unmute", "where is
 *  room two oh four" do what they say; anything else is a question about the scene. */
$<HTMLFormElement>("ask-form").onsubmit = (e) => {
  e.preventDefault();
  const q = $<HTMLInputElement>("ask-input").value.trim();
  if (!q) return;
  const cmd = parseCommand(q, "direct");
  if (!cmd || cmd.intent === "ask" || cmd.intent === "wake") say(answer(cmd?.arg || q));
  else runCommand(cmd);
};
$("test-voice").onclick = () => { speech.unlock(); say("Hi, this is SeeForMe. Stairs down, about 3 meters. Handrail left."); };

// ---- voice commands ------------------------------------------------------------------------------------------------
const wakeBtn = opt<HTMLButtonElement>("wake"), voiceStateEl = opt("voice-state"), micBtn = $("mic");
let pinnedUntil = 0;   // the first-use privacy note stays on #voice-state for a while
const voice = new VoiceInput({
  onCommand: (cmd) => runCommand(cmd),
  onState: (s, text, important) => showVoiceState(s, text, important),
  appSpeaking: () => speech.audible(400),
  recentSpeech: () => [...speech.recent(3000), ...statusRecent(3000)],
  ack: () => { earcon(880, 70); earcon(1320, 90, 90); },
});
function showVoiceState(s: VoiceState, text: string, important: boolean) {
  micBtn.classList.toggle("listening", s === "talk");
  wakeBtn?.classList.toggle("listening", s === "follow");
  // the listener turned itself off (microphone blocked): the toggle and the remembered choice follow
  if (s === "blocked" && !voice.handsFree) { wakeBtn?.setAttribute("aria-pressed", "false"); store.set("seeforme:wake", "off"); }
  if (!important && (s === "idle" || s === "starting") && Date.now() < pinnedUntil) return;
  if (voiceStateEl) voiceStateEl.textContent = text;
  if (important) setStatus(text);
}
const VOICE_NOTICE = "Your browser's speech recognition may send what the microphone hears to its speech service (in Chrome: Google).";
/** The first use of the microphone (hands-free, Talk, the A key) says once, per browser, where the audio may go: in the
 *  status region and on #voice-state (pinned for a while; `example` only there, since the status region is read aloud
 *  while the microphone may listen). Returns whether it was shown now. */
function voiceNoticeOnce(text: string, example = ""): boolean {
  if (store.get("seeforme:voice-notice") != null) return false;
  store.set("seeforme:voice-notice", "1");
  pinnedUntil = Date.now() + 12000;
  if (voiceStateEl) voiceStateEl.textContent = [text, example].filter(Boolean).join(" ");
  setStatus(text);
  return true;
}
/** Hands-free on / off. Turning it on is the user gesture that opens the microphone, so "SeeForMe, start" works before
 *  the camera runs. The first time, it says where the audio may go. */
function setHandsFree(on: boolean) {
  if (on && !voice.supported) { voice.enable(); return; }   // explains (status + #voice-state)
  if (on) {
    speech.unlock();
    voice.enable();
    if (!voiceNoticeOnce(`Hands-free voice is on. ${VOICE_NOTICE}`, `${IDLE_TEXT}.`)) setStatus("Hands-free voice on.");
  } else voice.disable("off", "Voice off", true);
  wakeBtn?.setAttribute("aria-pressed", String(on));
  store.set("seeforme:wake", on ? "on" : "off");
}
if (wakeBtn) {
  if (!voice.supported) {
    wakeBtn.disabled = true;
    wakeBtn.title = "Needs a browser with speech recognition (Chrome, Edge or Safari)";
    if (voiceStateEl) voiceStateEl.textContent = "Hands-free voice needs a browser with speech recognition (Chrome, Edge or Safari). Type in Ask instead.";
  } else {
    wakeBtn.onclick = () => setHandsFree(!voice.handsFree);
    if (store.get("seeforme:wake") === "on") {
      // remembered: listen again once the microphone may open (permission already granted, or at the first tap or key)
      wakeBtn.setAttribute("aria-pressed", "true");
      if (voiceStateEl) voiceStateEl.textContent = "Hands-free voice: tap anywhere or press a key to start listening.";
      const arm = (ev?: Event) => {
        window.removeEventListener("pointerdown", arm, true); window.removeEventListener("keydown", arm, true);
        // the hands-free button and the V key toggle it themselves (arming first would make them turn it off)
        if (ev && ((ev instanceof KeyboardEvent && ev.key.toLowerCase() === "v") || (ev.target instanceof Node && wakeBtn.contains(ev.target)))) {
          wakeBtn.setAttribute("aria-pressed", "false");
          return;
        }
        if (!voice.handsFree && wakeBtn.getAttribute("aria-pressed") === "true") { speech.unlock(); voice.enable(); }
      };
      window.addEventListener("pointerdown", arm, true); window.addEventListener("keydown", arm, true);
      void navigator.permissions?.query({ name: "microphone" as PermissionName }).then((p) => { if (p.state === "granted") arm(); }).catch(() => {});
    }
  }
}
/** Push-to-talk (#mic, #ask-voice, the A key): one command, wake phrase optional. The very first press only says where
 *  the audio may go (a screen reader reading that note into the open microphone would be taken as the command). */
function talk() {
  if (!voice.supported) { say("Voice input isn't available in this browser. Type your question in Ask."); return; }
  speech.stop(); speech.unlock();
  if (voiceNoticeOnce(`${VOICE_NOTICE} Press Talk again to speak.`)) {
    if (speech.enabled) say("Talk uses your browser's speech recognition, which may send the audio to its speech service. Press Talk again to speak.");
    return;
  }
  voice.talk();
}
micBtn.onclick = talk;
$("ask-voice").onclick = talk;
document.addEventListener("visibilitychange", () => voice.setHidden(document.hidden));
const VERBOSITY_SAY: Record<string, string> = { detailed: "More detail.", quiet: "Less detail.", balanced: "Normal detail." };
/** Say `text` once the app has finished what it is saying (at most `maxMs` later), so it does not cut the reply before it;
 *  nothing when the camera stopped or `current()` no longer holds meanwhile. */
function sayAfter(text: string, current: () => boolean = () => true, maxMs = 6000) {
  const t0 = performance.now();
  const go = () => {
    if (!state.running || !current()) return;
    if ((speech.free(now()) && !speech.audible(0)) || performance.now() - t0 > maxMs) say(text);
    else setTimeout(go, 150);
  };
  go();
}
/** Every recognised command does what its button does and gets a short spoken confirmation. */
function runCommand(cmd: Command) {
  const a = cmd.arg ?? "";
  switch (cmd.intent) {
    case "start": camCtl.start(); break;
    // unlike the terminal demo, where safety cues go on while paused, pausing here stops the camera, also one still
    // opening, and says so plainly (arg "camera", "camera off", is the same here)
    case "pause": camCtl.pause(); break;
    case "quit":
      camCtl.quit();   // also a camera still opening: nothing may run after "Goodbye"
      say("Goodbye. Hands-free voice is off.");
      setHandsFree(false);
      break;
    case "find": setMode("find"); say(lookFor(a)); break;
    case "stop_find": clearFind(); say("Stopped looking."); break;
    case "read": void readNow(); break;
    case "describe": say(answer("what's around me")); break;
    case "ask": $<HTMLInputElement>("ask-input").value = a; say(answer(a)); break;
    case "repeat": say(answer("repeat")); break;
    case "mute":
      if (!speech.enabled) break;
      say("Sound off.");
      setTimeout(() => setMuted(true), 1200);   // after the confirmation
      break;
    case "unmute": setMuted(false); say("Sound on."); break;
    case "help": say(HELP_TEXT); break;
    case "mode":
      // scene and automatic are the activity detector's job in this app: both are Navigate here
      setMode(a === "read" ? "read" : "navigate");
      say(a === "read" ? "Read mode." : a === "scene" ? "Navigate mode. Scene descriptions follow automatically when you stand still." : a === "auto" ? "Automatic: Navigate mode." : "Navigate mode.");
      break;
    case "verbosity":
      if (a === "detailed" || a === "quiet" || a === "balanced") { verbositySel.value = a; cfg.verbosity = a; }
      say(VERBOSITY_SAY[a] ?? "Okay.");
      break;
    // "flip camera": the next camera; "front camera" / "back camera": that side (starting the camera when it is off),
    // unless the camera already reports that side or already answered the same request
    case "flip": camCtl.flip(cmd.arg); break;
    case "wake": break;   // handled by the listener (follow-up window)
  }
}

document.addEventListener("keydown", (e) => {
  // No shortcuts with modifiers (Cmd/Ctrl+F is the browser's find) or on key repeat, and never steal Space or letters from
  // buttons, links, <summary>, form controls or other focusable widgets (tabindex="-1" targets such as #controls still get them).
  if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  if (e.target instanceof HTMLElement && e.target.closest('button, input, textarea, select, a[href], summary, details, [contenteditable], [tabindex]:not([tabindex="-1"])')) return;
  const k = e.key.toLowerCase();
  if (e.key === " ") { e.preventDefault(); $("start").click(); }
  if (k === "f") $("flip").click();
  if (k === "m") $("mute").click();
  if (k === "a") talk();
  if (k === "v" && voice.supported) setHandsFree(!voice.handsFree);
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
  // caches of earlier releases that this release's service worker no longer reads (0.2.0 runtime cache, older ORT wasm,
  // an earlier YOLO11s)
  for (const k of keys) {
    if (k === "seeforme-models" || (k.startsWith("seeforme-ort-") && k !== __SEEFORME_ORT_CACHE__)
      || (k.startsWith("seeforme-yolo11s-") && k !== __SEEFORME_MODELS__.gpuCache)) void caches.delete(k).catch(() => {});
  }
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
