"""Ingest the owner's chest-mounted iPhone recordings (issue #9; recording protocol in docs/recording-protocol.md).

    python scripts/ingest_own.py                      # every new file in data/own/raw/
    python scripts/ingest_own.py --only 2026-11-14_akw-to-sss_take1.MOV

For each raw video (data/own/raw/, git-ignored, never uploaded):
  1. ffmpeg: upright 640-px-wide H.264 copy at the source frame rate, and a 16 kHz mono WAV of the narration;
  2. Vosk (offline, Apache-2.0, models/vosk/vosk-model-small-en-us-0.15): word-level timestamps; spoken keywords
     ("stairs down", "door", "curb", "crossing", "clear") become timed in-situ labels (seeforme.own.ingest);
  3. privacy: every person found by the detector gets the top of their box blurred in every frame before the
     processed video is written (data/own/processed/<name>.mp4, no audio track).
Writes data/own/processed/<name>.labels.json (words, events, duration, blur stats). Raw audio stays local.
"""
import argparse, json, subprocess, wave
from pathlib import Path

import cv2

from _common import ROOT
from seeforme.own.ingest import blur_regions, head_region, parse_narration

RAW, OUT = ROOT / "data" / "own" / "raw", ROOT / "data" / "own" / "processed"
VOSK = ROOT / "models" / "vosk" / "vosk-model-small-en-us-0.15"

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--no-blur", action="store_true", help="debug only; never share an unblurred output")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def transcribe(wav: Path):
    from vosk import KaldiRecognizer, Model
    model = Model(str(VOSK))
    wf = wave.open(str(wav), "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    words = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            words += json.loads(rec.Result()).get("result", [])
    words += json.loads(rec.FinalResult()).get("result", [])
    return [{"word": w["word"], "start": w["start"], "end": w["end"], "conf": w.get("conf", 1.0)} for w in words]


files = [RAW / args.only] if args.only else sorted(p for p in RAW.glob("*") if p.suffix.lower() in (".mov", ".mp4", ".m4v"))
detector = None
for src in files:
    name = src.stem
    mp4, wav, lab = OUT / f"{name}.mp4", OUT / f"{name}.wav", OUT / f"{name}.labels.json"
    if lab.exists():
        print(f"skip {name} (done)"); continue
    tmp = OUT / f"{name}.unblurred.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale={args.width}:-2", "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", str(tmp)], check=True)        # ffmpeg applies the rotation tag
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    words = transcribe(wav)
    events = parse_narration(words)
    cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = blurred = 0
    if not args.no_blur and detector is None:
        import os
        from seeforme.perception.detector import YoloDetector
        wdir = ROOT / "weights"                     # git-ignored; the person detector's weights live here, not in scripts/
        here = os.getcwd(); os.chdir(wdir)
        try:
            detector = YoloDetector()
        finally:
            os.chdir(here)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if not args.no_blur:
            heads = [head_region(d.bbox) for d in detector.detect(fr) if d.label == "person"]
            if heads:
                fr = blur_regions(fr, heads); blurred += 1
        writer.write(fr); n += 1
    cap.release(); writer.release(); tmp.unlink()
    lab.write_text(json.dumps({"source": src.name, "fps": fps, "frames": n, "duration_s": n / fps, "frames_with_blur": blurred,
                               "blurred": not args.no_blur, "words": words, "events": events}, indent=1))
    print(f"{name}: {n / fps:.0f} s, {len(words)} words, {len(events)} events "
          f"({', '.join(e['phrase'] for e in events[:8])}{' ...' if len(events) > 8 else ''}), people blurred in {blurred} frames")
