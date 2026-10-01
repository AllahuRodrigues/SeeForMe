"""Export the perception model for mobile / web deployment and measure CPU latency.

  * ONNX  (opset 12-17): runs in onnxruntime, onnxruntime-web (browser) and most
    mobile runtimes; this is the artefact a browser demo or an Android app needs.
  * Core ML: the iOS / macOS artefact (needs coremltools; skipped if unavailable).

The CACS layer itself has no weights to export beyond six numbers
(configs/cacs_tuned.yaml) or a tiny logistic model, so it ports to any language.
"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
from _common import MODELS, RESULTS, dump_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--skip-coreml", action="store_true")
args = ap.parse_args()

from ultralytics import YOLO
out_dir = MODELS / "export"; out_dir.mkdir(parents=True, exist_ok=True)
report = {"source": args.model, "imgsz": args.imgsz, "exports": {}}
m = YOLO(args.model)

# ---- ONNX ------------------------------------------------------------------
try:
    p = Path(m.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False, opset=17))
    dst = out_dir / p.name; shutil.move(str(p), dst)
    entry = {"path": str(dst.relative_to(MODELS.parent)), "size_mb": round(dst.stat().st_size / 2**20, 2)}
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        x = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        for _ in range(5):
            sess.run(None, {name: x})
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); sess.run(None, {name: x}); ts.append((time.perf_counter() - t0) * 1000)
        entry["onnxruntime_cpu_ms_mean"] = round(float(np.mean(ts)), 1)
        entry["onnxruntime_cpu_ms_p95"] = round(float(np.percentile(ts, 95)), 1)
        entry["output_shape"] = [o.shape for o in sess.get_outputs()]
    except Exception as e:  # onnxruntime not installed
        entry["onnxruntime"] = f"not measured: {type(e).__name__}: {e}"
    report["exports"]["onnx"] = entry
except Exception as e:
    report["exports"]["onnx"] = {"error": f"{type(e).__name__}: {e}"}

# ---- Core ML ---------------------------------------------------------------
if not args.skip_coreml:
    try:
        p = Path(m.export(format="coreml", imgsz=args.imgsz, nms=True))
        dst = out_dir / p.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(p), dst)
        size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) if dst.is_dir() else dst.stat().st_size
        report["exports"]["coreml"] = {"path": str(dst.relative_to(MODELS.parent)), "size_mb": round(size / 2**20, 2)}
    except Exception as e:
        report["exports"]["coreml"] = {"error": f"{type(e).__name__}: {e}"}

dump_json(report, RESULTS / "export_report.json")
print(json.dumps(report, indent=1))
