"""Blind human review of the weak EgoBlind events (M3, issue #5). Everything stays on this machine.

    python scripts/review_events.py frames --split train                 # extract thumbnails around each question
    python scripts/review_events.py serve  --split train --reviewer ar   # review at http://127.0.0.1:8765
    python scripts/review_events.py serve  --split train --reviewer r2 --second-rater   # 20% sample for agreement
    python scripts/review_events.py merge  --split train --reviewer ar   # write human_verdict into the review CSV
    python scripts/review_events.py kappa  --split train --a ar --b r2   # Cohen's kappa on the overlap

Reads results/egoblind/review_events_<split>.csv (scripts/diagnose_egoblind_recall.py). Model columns are never shown.
Frames and verdicts live in data/review/<split>/ (git-ignored: frames are EgoBlind video, CC BY-NC).
Keys in the page: 1 speak, 2 silent, 3 wrong object, 4 unclear, left/right arrows to move.
"""
import argparse, csv, html, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from _common import ROOT
from seeforme.egoblind.review import (VERDICTS, blind_row, cohen_kappa, latest_verdicts, make_verdict, merge_verdicts,
                                      sample_second_rater)

REVIEW = ROOT / "data" / "review"
STEP_S, BEFORE_S, AFTER_S, WIDTH = 0.75, 3.0, 1.0, 480

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["frames", "serve", "merge", "kappa"])
ap.add_argument("--split", default="train")
ap.add_argument("--reviewer", default="")
ap.add_argument("--second-rater", action="store_true", help="review only the seeded 20% agreement sample")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--a", default=""); ap.add_argument("--b", default="")
args = ap.parse_args()

csv_path = ROOT / "results" / "egoblind" / f"review_events_{args.split}.csv"
if not csv_path.exists():
    sys.exit(f"{csv_path} missing: run scripts/diagnose_egoblind_recall.py --split {args.split}")
rows = list(csv.DictReader(open(csv_path)))
out_dir = REVIEW / args.split
store = out_dir / "verdicts.jsonl"


def window_of(r):
    try:
        a, b = (float(x) for x in r["window"].split("-"))
        return a, b
    except Exception:
        return float(r["t_q"]) - 3.0, float(r["t_q"])


def video_path(r):
    sc = ROOT / "data" / "egoblind" / "scenarios" / args.split / f"{int(float(r['video_id']))}.json"
    if sc.exists():
        p = Path(json.loads(sc.read_text()).get("video", ""))
        if p.exists():
            return p
        cand = ROOT / "data" / "egoblind" / "videos" / p.parent.name / p.name
        if cand.exists():
            return cand
    return None


def load_records():
    return [json.loads(l) for l in open(store)] if store.exists() else []


if args.cmd == "frames":
    import cv2
    done = 0
    for r in rows:
        b = blind_row(r); d = out_dir / "frames" / b["key"].replace(":", "_")
        if (d / "meta.json").exists():
            continue
        vp = video_path(r)
        if vp is None:
            print(f"   no video for {b['key']}"); continue
        cap = cv2.VideoCapture(str(vp))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
        w0, _ = window_of(r); tq = float(r["t_q"])
        t, times = max(0.0, w0 - BEFORE_S), []
        d.mkdir(parents=True, exist_ok=True)
        while t <= min(dur, tq + AFTER_S) + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if ok:
                h, w = fr.shape[:2]
                fr = cv2.resize(fr, (WIDTH, int(h * WIDTH / w)))
                cv2.imwrite(str(d / f"{len(times):02d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                times.append(round(t, 2))
            t += STEP_S
        cap.release()
        (d / "meta.json").write_text(json.dumps({"times": times}))
        done += 1
        if done % 25 == 0:
            print(f"   {done} events", flush=True)
    print(f"frames in {out_dir / 'frames'}")

elif args.cmd == "merge":
    if not args.reviewer:
        sys.exit("--reviewer is required")
    v = latest_verdicts(load_records(), reviewer=args.reviewer)
    merged = merge_verdicts(rows, v)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(merged)
    print(f"{len(v)} of {len(rows)} events have a human verdict from {args.reviewer} -> {csv_path}")

elif args.cmd == "kappa":
    recs = load_records()
    a = {k: r["verdict"] for k, r in latest_verdicts(recs, reviewer=args.a).items()}
    b = {k: r["verdict"] for k, r in latest_verdicts(recs, reviewer=args.b).items()}
    k = cohen_kappa(a, b)
    print(f"overlap {len(set(a) & set(b))} events, kappa = {k if k is None else round(k, 3)}")

else:
    if not args.reviewer:
        sys.exit("--reviewer is required (your initials)")
    events = [blind_row(r) for r in rows]
    if args.second_rater:
        keep = set(sample_second_rater([e["key"] for e in events], 0.2, seed=0))
        events = [e for e in events if e["key"] in keep]
    for e in events:
        m = out_dir / "frames" / e["key"].replace(":", "_") / "meta.json"
        e["times"] = json.loads(m.read_text())["times"] if m.exists() else []
        e["win"] = list(window_of(e))
    out_dir.mkdir(parents=True, exist_ok=True)
    page = (Path(__file__).parent / "review_ui.html").read_text()
    page = page.replace("__DATA__", json.dumps({"events": events, "reviewer": args.reviewer, "split": args.split,
                                                  "verdicts": list(VERDICTS)}).replace("</", "<\\/"))
    page = page.replace("__TITLE__", html.escape(f"Event review: {args.split}"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if self.path == "/verdicts":
                v = latest_verdicts(load_records(), reviewer=args.reviewer)
                return self._send(200, json.dumps(v).encode(), "application/json")
            if self.path.startswith("/frames/"):
                p = (out_dir / self.path.lstrip("/")).resolve()
                if p.is_file() and (out_dir / "frames").resolve() in p.parents:
                    return self._send(200, p.read_bytes(), "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/verdict":
                return self._send(404, b"", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                win = body.get("window")
                v = make_verdict(body["key"], body["verdict"], args.reviewer, window=tuple(win) if win else None,
                                 note=body.get("note", ""))
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, str(e).encode(), "text/plain")
            with open(store, "a") as f:
                f.write(json.dumps(v) + "\n")
            self._send(200, json.dumps(v).encode(), "application/json")

    print(f"reviewing {len(events)} events as {args.reviewer}: http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
