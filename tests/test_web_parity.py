"""Web parity: the browser runs the same learned scheduler as Python and decides the same.

* the exported model (web/public/models/scheduler_rl.onnx + web/public/config/scheduler.json) is current: it was made
  from models/scheduler/scheduler_rl.pt, lists EventFeatures in Python's order, and matches PyTorch (ONNX Runtime)
* web/parity/run.mjs replays Python SchedulerPolicy decision steps through the TypeScript feature code, the ONNX model
  (onnxruntime-web in node) and the TypeScript decision rule: every decision, message and feature must match
The node tests are skipped when Node.js or web/node_modules (esbuild, onnxruntime-web) are missing.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RL_CKPT = ROOT / "models" / "scheduler" / "scheduler_rl.pt"
ONNX = WEB / "public" / "models" / "scheduler_rl.onnx"
META = WEB / "public" / "config" / "scheduler.json"
FIXTURE = WEB / "parity" / "fixtures" / "parity_small.json.gz"
CAMPUS_TEST = ROOT / "data" / "campus" / "test"
NODE = shutil.which("node")
WEB_DEPS = all((WEB / "node_modules" / p).exists() for p in ("esbuild", "onnxruntime-web"))

needs_export = pytest.mark.skipif(not (RL_CKPT.exists() and ONNX.exists() and META.exists()),
                                  reason="no exported web scheduler (python scripts/export_web_scheduler.py)")
needs_node = pytest.mark.skipif(NODE is None or not WEB_DEPS, reason="needs Node.js and web/node_modules (cd web && npm install)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_parity(fixture: Path):
    p = subprocess.run([NODE, str(WEB / "parity" / "run.mjs"), str(fixture), "--quiet"], cwd=str(WEB),
                       capture_output=True, text=True, timeout=900)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("PARITY_JSON ")]
    assert lines, f"runner printed no result\nstdout: {p.stdout[-3000:]}\nstderr: {p.stderr[-3000:]}"
    return p.returncode, json.loads(lines[-1][len("PARITY_JSON "):])


def assert_identical(code: int, r: dict, min_steps: int):
    assert r["problems"] == [], r["problems"]
    assert r["steps"] >= min_steps
    assert r["full_agree"] == r["steps"], r["mismatches"][:5]           # TS features -> ONNX -> TS rule
    assert r["decision_agree"] == r["steps"], r["mismatches"][:5]       # Python features -> ONNX -> TS rule
    assert r["message_agree"] == r["steps"], r["mismatches"][:5]        # message, slots and duration of every cue
    assert r["missing_candidates"] == 0 and r["feature_f32_mismatch"] == 0
    assert r["feature_max_abs_diff"] <= 1e-9, r["feature_worst"]
    assert r["logits_max_abs_diff"] < 1e-4 and r["slot_logits_max_abs_diff"] < 1e-4
    assert r["budget_max_abs_diff"] <= 1e-9 and r["ego_max_abs_diff"] <= 1e-9
    assert r["rule_cases"] > 0 and r["rule_agree"] == r["rule_cases"]
    assert r["rule_actions"].get("interrupt", 0) > 0 and r["rule_actions"].get("haptic", 0) > 0
    assert code == 0 and r["ok"]


@needs_export
def test_web_export_is_current():
    from seeforme.cues.compose import SLOTS
    from seeforme.cues.pcacs import ACTIONS, EventFeatures
    from seeforme.ml.scheduler_model import SchedulerModel
    meta = json.loads(META.read_text())
    assert meta["source_sha256"] == sha256(RL_CKPT), "models/scheduler/scheduler_rl.pt changed: run scripts/export_web_scheduler.py"
    assert meta["onnx_sha256"] == sha256(ONNX)
    assert meta["model_url"].endswith(f"?v={meta['onnx_sha256'][:12]}")
    assert meta["features"] == EventFeatures.names() and meta["actions"] == list(ACTIONS) and meta["slots"] == list(SLOTS)
    m = SchedulerModel.load(str(RL_CKPT))
    assert meta["K"] == m.K and meta["mode"] == "rl"
    assert abs(meta["silence_bias"] - float(m.net.silence_bias.detach()[0])) < 1e-7


@needs_export
def test_onnx_matches_pytorch():
    ort = pytest.importorskip("onnxruntime")
    import numpy as np
    from seeforme.ml.scheduler_model import SchedulerModel
    m = SchedulerModel.load(str(RL_CKPT))
    rng = np.random.default_rng(3)
    lists, X, M = [], np.zeros((64, m.K, 29), np.float32), np.zeros((64, m.K), np.float32)
    for i in range(64):
        L = int(rng.integers(1, m.K + 1))
        w = rng.normal(0, 1, (L, 29)).astype(np.float32)
        lists.append(w.tolist()); X[i, m.K - L:] = w; M[i, m.K - L:] = 1
    la, _, ls = m.forward(lists)          # the Python path: left-aligned windows + pack_padded_sequence
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    a, s, b = sess.run(["action_logits", "slot_logits", "silence_bias"], {"windows": X, "mask": M})
    assert np.abs(a - la.numpy()).max() < 1e-4 and np.abs(s - ls.numpy()).max() < 1e-4
    assert abs(float(b[0]) - float(m.net.silence_bias.detach()[0])) < 1e-6


@needs_export
@needs_node
def test_committed_fixture_replays_identically():
    """3 CampusBench test scenarios + a synthetic busy-channel run + 400 random-logit rule cases (committed)."""
    code, r = run_parity(FIXTURE)
    assert_identical(code, r, min_steps=250)


@needs_export
@needs_node
@pytest.mark.skipif(not CAMPUS_TEST.exists(), reason="no CampusBench test split (scripts/make_campus.py)")
def test_fresh_recording_replays_identically(tmp_path):
    """Record now from Python (one test scenario per scene kind, two synthetic runs, rule cases) and replay in TS."""
    out = tmp_path / "parity.json.gz"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "web_parity.py"), "--kinds", "", "--per-kind", "1", "--synthetic", "2",
                    "--rule-cases", "300", "--out", str(out)], check=True, cwd=str(ROOT), capture_output=True, timeout=900,
                   env={**os.environ, "OMP_NUM_THREADS": "2"})
    code, r = run_parity(out)
    assert_identical(code, r, min_steps=11 * 60)
