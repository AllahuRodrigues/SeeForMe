"""SeeForMe server: metric depth, OCR and ElevenLabs TTS for the web app, and static hosting of web/dist.

    .venv/bin/python server/app.py                 # http://localhost:8000
    .venv/bin/python server/app.py --ssl           # https://<your-mac-ip>:8000 for phones/tablets (self-signed cert)

The browser does detection, tracking and scheduling itself; it calls here only for what it
cannot run locally yet (Depth Anything V2 metric depth, EasyOCR) and for the ElevenLabs voice,
whose key must stay server-side.

POST /api/depth also returns "cane": the class-agnostic obstacle summary (seeforme/live/cane.py) of the same depth
map, for what the detector cannot name (walls, doors, poles, low and head-height obstacles):

    "cane": null | {"clear_m": {"left"|"ahead"|"right": metres | null},       free distance within 4 m; null = cannot tell
                    "obstacle": null | {"sector", "distance_m", "level": "ground"|"body"|"head", "confidence", "bbox"},
                    "suggest": null | "left" | "right" | "stop"}

It is one depth map, with no memory: the caller applies persistence (the live demo waits for 3 maps over 0.5 s),
decides walking vs standing still, and never reads clear_m out as "clear". Two optional form fields shape "suggest"
(the obstacle and clear_m are reported whatever they say):
    walking=true    "stop" (something within 0.6 m ahead) is returned only then: one map cannot tell a walker about to
                    hit something from a seated user whose laptop camera sees the desk 0.5 m away, and the live demo
                    stops only for what the user walked up to. Without it suggest is "left", "right" or null.
    aid=cane        what the user travels with: cane (default), guide-dog or none (seeforme/live/cane.py, the BLV-Road-Nav
                    focus group). cane: "stop" only for a body or head-level obstacle, "left" / "right" only for a body
                    obstacle beyond the cane's reach (1.2 m); guide-dog: "stop" only for body or head level, never
                    "left" / "right" (the dog leads); none: every suggestion. Anything else is a 400.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seeforme.live.cane import AIDS, analyze as cane_analyze  # noqa: E402
from seeforme.perception.depth import DepthAnythingEstimator, INDOOR, OUTDOOR, object_distances  # noqa: E402
from seeforme.types import Detection  # noqa: E402

app = FastAPI(title="SeeForMe server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def cross_origin_isolation(request, call_next):
    """COOP/COEP make the page crossOriginIsolated, which WebAssembly multi-threading needs."""
    resp = await call_next(request)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return resp
STATE: dict = {"depth": {}, "ocr": None, "tts": None, "tts_error": None}


def depth_model(outdoor: bool) -> DepthAnythingEstimator:
    key = "outdoor" if outdoor else "indoor"
    if key not in STATE["depth"]:
        STATE["depth"][key] = DepthAnythingEstimator(OUTDOOR if outdoor else INDOOR, input_size=252)
    return STATE["depth"][key]


def _decode(data: bytes) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "bad image")
    return img


def tts_usable() -> bool:
    """ElevenLabs loaded, or not tried yet and an API key is set (the web app picks the browser voice otherwise)."""
    if STATE["tts"] is not None:
        return True
    if STATE["tts_error"] is not None:
        return False
    try:
        from seeforme.feedback.elevenlabs_tts import load_api_key
    except Exception:
        return False
    return load_api_key() is not None


@app.get("/api/health")
def health():
    return {"ok": True, "depth": "depth-anything-v2-metric-small", "ocr": True, "tts": tts_usable()}


@app.post("/api/depth")
async def api_depth(image: UploadFile = File(...), boxes: str = Form("[]"), outdoor: str = Form("false"),
                    walking: str = Form(""), aid: str = Form("cane")):
    aid = aid.strip().lower() or "cane"
    if aid not in AIDS:
        raise HTTPException(400, f"aid must be one of {', '.join(AIDS)}")
    img = _decode(await image.read())
    bxs = json.loads(boxes)
    dets = [Detection(label="object", confidence=1.0, bbox=tuple(b)) for b in bxs]
    t0 = time.perf_counter()
    est = depth_model(outdoor.lower() == "true")
    depth = est(img)
    object_distances(depth, dets)
    return {"distances": [[d.distance_m, d.depth_conf] for d in dets], "ms": round((time.perf_counter() - t0) * 1000, 1),
            "device": est.device, "cane": cane_summary(depth, walking=walking.strip().lower() == "true", aid=aid)}


def cane_summary(depth, walking: bool = False, aid: str = "cane") -> dict | None:
    """The "cane" field: None when there is no usable depth map (or the analysis fails: distances still go back). A
    "stop" suggestion only for a client that says the user is walking, and only what the aid leaves to us (module
    docstring)."""
    if depth is None or getattr(depth, "size", 0) == 0:
        return None
    try:
        fr = cane_analyze(depth)
        if fr.floor is None:
            return None                                               # no valid depth at all: nothing to tell
        out = fr.as_contract(aid=aid)
        if out["suggest"] == "stop" and not walking:
            out["suggest"] = None
        return out
    except Exception as e:      # never lose the distances over the cane
        print(f"cane analysis failed: {type(e).__name__}: {e}", flush=True)
        return None


@app.post("/api/ocr")
async def api_ocr(image: UploadFile = File(...)):
    if STATE["ocr"] is None:
        from seeforme.perception import EasyOcrReader
        STATE["ocr"] = EasyOcrReader()
    img = _decode(await image.read())
    t0 = time.perf_counter()
    dets = STATE["ocr"].read(img)
    return {"texts": [{"text": d.text, "conf": d.confidence, "bbox": list(d.bbox)} for d in dets],
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/api/tts")
async def api_tts(body: dict):
    text = (body or {}).get("text", "").strip()
    if not text:
        raise HTTPException(400, "no text")
    if STATE["tts"] is None:
        try:
            from seeforme.feedback.elevenlabs_tts import ElevenLabsTTS
            STATE["tts"] = ElevenLabsTTS()
        except Exception as e:
            STATE["tts_error"] = str(e)
            raise HTTPException(503, f"ElevenLabs unavailable: {e}")
    try:
        audio = STATE["tts"].synthesize(text[:300])
    except Exception as e:
        raise HTTPException(502, f"ElevenLabs error: {e}")
    return Response(content=audio, media_type="audio/mpeg")


DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
else:
    @app.get("/")
    def index():
        return JSONResponse({"message": "web app not built; run `cd web && npm run build`, or use the Vite dev server on :5173"})


def ensure_cert(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    crt, key = d / "cert.pem", d / "key.pem"
    if not crt.exists():
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt),
                        "-days", "365", "-subj", "/CN=seeforme.local"], check=True, capture_output=True)
    return crt, key


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ssl", action="store_true", help="self-signed HTTPS so phones can open the camera")
    ap.add_argument("--warm", action="store_true", help="load the depth model at startup")
    args = ap.parse_args()
    if args.warm:
        depth_model(False)
    kw = {}
    if args.ssl:
        crt, key = ensure_cert(ROOT / "server" / "certs")
        kw = {"ssl_certfile": str(crt), "ssl_keyfile": str(key)}
    uvicorn.run(app, host=args.host, port=args.port, **kw)
