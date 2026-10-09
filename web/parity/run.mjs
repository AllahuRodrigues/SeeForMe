#!/usr/bin/env node
// Web parity check for the learned scheduler.
//
// Replays the Python decision steps recorded by scripts/web_parity.py through the TypeScript code the browser runs
// (src/pcacs.ts features, src/learned.ts windows + decision rule, src/compose.ts messages; bundled here with esbuild)
// and the exported model public/models/scheduler_rl.onnx (onnxruntime-web, wasm backend in node), and compares:
//   full path      TS features -> TS windows -> ONNX -> TS rule -> cue (action, track, message, slots, duration)
//   decision only  Python's feature vectors -> TS windows -> ONNX -> TS rule (isolates the model + rule)
//   features       the 29 TS feature values against Python's, every candidate of every step
//   state          speech budget and walking-speed estimate before every step
//   rule cases     random logits / channel states / track timings through Python's decide() with a stub model vs the
//                  TS rule (legalActions + rlChoose): interrupts, exact ties and every legality mask
//
//   cd web && node parity/run.mjs [fixture.json[.gz] ...] [--quiet]
// Without fixture arguments it reads parity/fixtures/*.json(.gz). Exit code 0 only when everything is identical
// (features within 1e-9, logits within 1e-4). The last line is "PARITY_JSON {...}" for scripts and tests.
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";
import { build } from "esbuild";
import * as ort from "onnxruntime-web";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, "..");
const argv = process.argv.slice(2);
const quiet = argv.includes("--quiet");
const log = (...a) => { if (!quiet) console.log(...a); };
const readJson = (f) => JSON.parse(f.endsWith(".gz") ? gunzipSync(readFileSync(f)).toString("utf8") : readFileSync(f, "utf8"));

// ---- compile the TypeScript the browser runs -----------------------------------------------------------------
mkdirSync(resolve(here, ".build"), { recursive: true });
const outfile = resolve(here, ".build/parity-bundle.mjs");
await build({
  stdin: { contents: 'export * from "./src/learned.ts"; export * from "./src/pcacs.ts"; export * from "./src/compose.ts"; export { Camera } from "./src/tracker.ts";',
    resolveDir: web, loader: "ts", sourcefile: "parity-entry.ts" },
  bundle: true, format: "esm", platform: "neutral", target: "es2022", outfile, logLevel: "warning", external: ["onnxruntime-web"],
});
const L = await import(`${pathToFileURL(outfile).href}?t=${Date.now()}`);

// ---- model, metadata, config ---------------------------------------------------------------------------------
const meta = readJson(resolve(web, "public/config/scheduler.json"));
const pcacsJson = readJson(resolve(web, "public/config/pcacs.json"));
const onnxBuf = readFileSync(resolve(web, "public/models/scheduler_rl.onnx"));
const onnxSha = createHash("sha256").update(onnxBuf).digest("hex");
const problems = [];
if (meta.onnx_sha256 !== onnxSha) problems.push(`scheduler.json onnx_sha256 ${String(meta.onnx_sha256).slice(0, 12)} != model file ${onnxSha.slice(0, 12)} (re-run scripts/export_web_scheduler.py)`);
const metaErr = L.checkMeta(meta);
if (metaErr) problems.push(`scheduler.json rejected by the app: ${metaErr}`);
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";
const session = await ort.InferenceSession.create(onnxBuf, { executionProviders: ["wasm"] });
const run = L.ortRunner(ort, session);

function makeCfg(over = {}) {
  const cfg = { ...L.DEFAULT_CFG, weights: { ...L.DEFAULT_CFG.weights } };
  // as main.ts: the tuned values from public/config/pcacs.json over the defaults
  Object.assign(cfg, pcacsJson, { weights: { ...cfg.weights, ...pcacsJson.weights }, verbosity: pcacsJson.verbosity ?? "balanced", units: pcacsJson.units ?? "m" }, over);
  return cfg;
}
const mkTrack = (d) => ({
  id: d.id, label: d.label, kind: d.kind, text: d.text ?? undefined, bbox: d.bbox, confidence: d.conf, firstSeen: 0, lastSeen: 0, hits: 2, misses: 0,
  confirmed: true, lastAnnounced: d.la, lastAnnouncedBBox: d.lab, announceCount: d.ac, lastHaptic: d.lh, bboxHistory: d.bh, distance: d.dist,
  depthConf: d.dconf, lateral: d.lat, relSpeed: d.vz, lateralSpeed: d.vx, ttc: d.ttc ?? Infinity, pathProb: d.pp,
  distHistory: Array.from({ length: d.nhist }, () => [0, 0, 0]), remembered: d.rem,
});
const mkCh = (c) => ({ busy: c.busy, remaining: c.remaining, currentUtility: c.cu, currentSeverity: c.cs, currentTrack: c.ct });

// ---- replay ---------------------------------------------------------------------------------------------------
let files = argv.filter((a) => !a.startsWith("--"));
if (!files.length) {
  const dir = resolve(here, "fixtures");
  files = readdirSync(dir).filter((f) => /\.json(\.gz)?$/.test(f)).sort().map((f) => resolve(dir, f));
}
const R = {
  fixtures: files.map((f) => f.replace(`${web}/`, "")), scenarios: 0, steps: 0, candidates: 0,
  full_agree: 0, decision_agree: 0, message_agree: 0, feature_max_abs_diff: 0, feature_worst: null, feature_f32_mismatch: 0, missing_candidates: 0,
  logits_max_abs_diff: 0, slot_logits_max_abs_diff: 0, min_margin: Infinity, budget_max_abs_diff: 0, ego_max_abs_diff: 0,
  python_actions: {}, ts_actions: {}, per_scenario: [], mismatches: [], problems,
  rule_cases: 0, rule_agree: 0, rule_actions: {},
};
const per = (name) => ({ name, steps: 0, full_agree: 0, decision_agree: 0, actions: {} });

for (const file of files) {
  const fx = readJson(file);
  if (fx.meta.source_sha256 !== meta.source_sha256)
    problems.push(`${file}: recorded with model ${String(fx.meta.source_sha256).slice(0, 12)}, web model is ${String(meta.source_sha256).slice(0, 12)}`);
  if (fx.meta.features.join() !== L.FEATURE_NAMES.join()) problems.push(`${file}: feature names differ from FEATURE_NAMES`);
  const p = fx.meta.pcacs;
  const pc = makeCfg();
  for (const [py, ts] of [["min_confidence", "minConfidence"], ["recent_block_s", "recentBlock"], ["haptic_repeat_s", "hapticRepeat"], ["ttc_horizon", "ttcHorizon"], ["budget_capacity_s", "budgetCapacity"]])
    if (p[py] !== pc[ts]) problems.push(`public/config/pcacs.json ${ts}=${pc[ts]} but Python ${py}=${p[py]} (re-run scripts/export_web_config.py)`);
  for (const sc of fx.scenarios) {
    R.scenarios++;
    const ps = per(sc.name);
    const cfg = makeCfg(sc.cfg ?? {});
    const cam = new L.Camera();
    const A = new L.LearnedScheduler(cfg, cam, meta, run);   // full path
    const B = new L.LearnedScheduler(cfg, cam, meta, run);   // decision only
    A.setGoal(sc.goal_tags); B.setGoal(sc.goal_tags);
    for (const [k, s] of sc.steps.entries()) {
      R.steps++; ps.steps++;
      R.budget_max_abs_diff = Math.max(R.budget_max_abs_diff, Math.abs(A.budget - s.budget_in));
      R.ego_max_abs_diff = Math.max(R.ego_max_abs_diff, Math.abs(A.egoSpeed - s.ego_in));
      const ch = mkCh(s.ch);
      const where = { scenario: sc.name, step: k, t: s.t };
      // full path
      A.lastDecision = null;
      const cue = await A.decide(s.tracks.map(mkTrack), s.t, ch);
      const tsVec = new Map(A.lastCandidates.map((c) => [c.track.id, L.featureVector(c)]));
      s.cands.forEach((id, i) => {
        R.candidates++;
        const v = tsVec.get(id);
        if (!v) { R.missing_candidates++; return; }
        s.vectors[i].forEach((pv, j) => {
          const d = Math.abs(pv - v[j]);
          if (Math.fround(pv) !== Math.fround(v[j])) R.feature_f32_mismatch++;
          if (d > R.feature_max_abs_diff) { R.feature_max_abs_diff = d; R.feature_worst = { ...where, track: id, feature: L.FEATURE_NAMES[j], python: pv, ts: v[j] }; }
        });
      });
      const py = s.py, tsAction = cue?.action ?? "silence", tsTrack = cue?.trackId ?? null;
      R.python_actions[py.action] = (R.python_actions[py.action] ?? 0) + 1;
      R.ts_actions[tsAction] = (R.ts_actions[tsAction] ?? 0) + 1;
      ps.actions[py.action] = (ps.actions[py.action] ?? 0) + 1;
      const fullOk = tsAction === py.action && tsTrack === py.track;
      if (fullOk) { R.full_agree++; ps.full_agree++; }
      const msgOk = fullOk && (cue?.message ?? "") === py.message && (cue ? cue.duration : 0) === py.duration && (cue?.slots ?? []).join() === py.slots.join();
      if (msgOk) R.message_agree++;
      // decision only, from Python's own feature vectors
      const byId = new Map(s.tracks.map((d) => [d.id, d]));
      const inputs = s.cands.map((id, i) => ({ trackId: id, vector: s.vectors[i], lastAnnounced: byId.get(id).la, lastHaptic: byId.get(id).lh }));
      const dB = await B.decideVectors(inputs, s.t, ch);
      const bAction = dB ? dB.action : "silence", bTrack = dB && dB.candidate >= 0 ? s.cands[dB.candidate] : null;
      const decOk = bAction === py.action && bTrack === py.track;
      if (decOk) { R.decision_agree++; ps.decision_agree++; }
      if (dB && s.logits) {
        s.logits.forEach((row, i) => row.forEach((v, j) => { R.logits_max_abs_diff = Math.max(R.logits_max_abs_diff, Math.abs(v - dB.logits[i * 4 + j])); }));
        s.slot_logits.forEach((row, i) => row.forEach((v, j) => { R.slot_logits_max_abs_diff = Math.max(R.slot_logits_max_abs_diff, Math.abs(v - dB.slotLogits[i * row.length + j])); }));
        R.min_margin = Math.min(R.min_margin, dB.margin);
      }
      if ((!fullOk || !decOk || !msgOk) && R.mismatches.length < 25)
        R.mismatches.push({ ...where, python: py, full: { action: tsAction, track: tsTrack, message: cue?.message ?? "", slots: cue?.slots ?? [], duration: cue?.duration ?? 0 },
          decision_only: { action: bAction, track: bTrack }, margin: dB?.margin ?? null });
    }
    R.per_scenario.push(ps);
  }
  // the decision rule alone, on logits Python's decide() was given by a stub model
  for (const [k, c] of (fx.rule_cases ?? []).entries()) {
    R.rule_cases++;
    const cfg = makeCfg();
    const ch = mkCh(c.ch);
    const legal = c.inputs.map((inp) => L.legalActions(inp, c.t, ch, cfg));
    const r = L.rlChoose(Float32Array.from(c.logits.flat()), legal, c.silence_bias);
    const track = r.candidate >= 0 ? c.inputs[r.candidate].trackId : null;
    R.rule_actions[c.py.action] = (R.rule_actions[c.py.action] ?? 0) + 1;
    if (r.action === c.py.action && track === c.py.track) R.rule_agree++;
    else if (R.mismatches.length < 25) R.mismatches.push({ rule_case: k, python: c.py, ts: { action: r.action, track } });
  }
}

const ok = problems.length === 0 && R.steps > 0 && R.full_agree === R.steps && R.rule_agree === R.rule_cases && R.decision_agree === R.steps && R.message_agree === R.steps
  && R.missing_candidates === 0 && R.feature_max_abs_diff <= 1e-9 && R.logits_max_abs_diff < 1e-4 && R.slot_logits_max_abs_diff < 1e-4
  && R.budget_max_abs_diff <= 1e-9 && R.ego_max_abs_diff <= 1e-9;
R.ok = ok;
if (!Number.isFinite(R.min_margin)) R.min_margin = null;

const pct = (a, b) => `${a}/${b} (${((100 * a) / Math.max(1, b)).toFixed(2)}%)`;
log(`web parity: ${R.scenarios} scenarios, ${R.steps} decision steps, ${R.candidates} candidates (model ${meta.version}, onnxruntime-web ${ort.env.versions?.web ?? "?"})`);
for (const ps of R.per_scenario) log(`  ${ps.name.padEnd(24)} steps ${String(ps.steps).padStart(4)}  full ${ps.full_agree === ps.steps ? "ok" : ps.full_agree}  decision-only ${ps.decision_agree === ps.steps ? "ok" : ps.decision_agree}  ${JSON.stringify(ps.actions)}`);
log(`  decisions, full path (TS features):        ${pct(R.full_agree, R.steps)}`);
log(`  decisions, decision only (Python features): ${pct(R.decision_agree, R.steps)}`);
log(`  cue messages / slots / durations:           ${pct(R.message_agree, R.steps)}`);
log(`  features: max |TS - Python| ${R.feature_max_abs_diff.toExponential(2)} over ${R.candidates} x 29 values; float32 mismatches ${R.feature_f32_mismatch}; missing candidates ${R.missing_candidates}`);
if (R.feature_worst && R.feature_max_abs_diff > 0) log(`    worst: ${JSON.stringify(R.feature_worst)}`);
log(`  model: max |ONNX Runtime web - PyTorch| action logits ${R.logits_max_abs_diff.toExponential(2)}, slot logits ${R.slot_logits_max_abs_diff.toExponential(2)}; closest decision margin ${R.min_margin?.toFixed(4)}`);
log(`  state: budget max diff ${R.budget_max_abs_diff.toExponential(2)}, walking speed max diff ${R.ego_max_abs_diff.toExponential(2)}`);
log(`  Python actions ${JSON.stringify(R.python_actions)}, TS actions ${JSON.stringify(R.ts_actions)}`);
if (R.rule_cases) log(`  decision rule on ${R.rule_cases} random-logit cases: ${pct(R.rule_agree, R.rule_cases)} (Python ${JSON.stringify(R.rule_actions)})`);
for (const pr of problems) log(`  PROBLEM: ${pr}`);
for (const m of R.mismatches.slice(0, 10)) log(`  MISMATCH ${JSON.stringify(m)}`);
log(ok ? "PARITY OK: TypeScript and Python decide the same on every step" : "PARITY FAILED");
console.log(`PARITY_JSON ${JSON.stringify(R)}`);
process.exit(ok ? 0 : 1);
