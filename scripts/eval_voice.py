"""How many spoken commands survive the real recogniser, and how often does chatter wake it?

Every command phrase of tests/data/voice_commands.json (the hands-free "wake" cases with an intent) and 30 natural
variants written for this eval (NATURAL: how people actually phrase them, "SeeForMe, where is the nearest exit") are
synthesised with several macOS voices (`say -o x.wav --data-format=LEI16@16000`), clean and with mild pink noise mixed
in, and each WAV goes through the live demo's own code path: VoskBackend(source=wav) -> VoiceListener.hear (the
hands-free rules: wake phrase required, one follow-up command after a bare wake word) -> parse_command. Measured per
recogniser mode (VoskBackend recogniser=open | grammar | hybrid | snap, seeforme/live/voice.py).

Paired: every utterance is decoded once by VoskBackend(recogniser="hybrid", second_pass="always"), which records both
readings of each end-pointed utterance (the grammar's and the open vocabulary's) and hands the hybrid's choice to the
listener. The other modes are scored on the same two readings, through the same selection functions (hybrid_text,
snap_text) and a fresh VoiceListener: grammar = the grammar reading, open = the open reading, snap = snap_text. So the
modes differ only in how they use the same decodes, and the open and grammar passes are not run twice per mode on a
machine that has little CPU to spare. A real VoskBackend(recogniser="open") run, with its own end-pointing, on
--open-check voices checks that the open reading of the paired run is what the open mode alone hears.

    command accuracy   the listener produced the expected command, intent and argument both right (the last command of
                       the utterance; a bare "SeeForMe" must give "wake")
    false wakes        20 sentences that are not commands but contain "see", "for", "me" and their sound-alikes
                       (NEGATIVES; 4 of them near-misses such as "Let's see, for me it's fine") and the shared vectors'
                       ignore cases: any command, even a bare wake, counts
    decode time        recogniser CPU seconds per second of audio (the live demo shares the CPU with YOLO and depth)

The parser alone on the same texts (no audio) is reported too, to tell recognition errors from parser gaps. Synthesised
speech is cleaner and more uniform than a user in a street; read the numbers as a comparison of the modes, not as the
accuracy a user will get. The default recogniser is chosen on these same recordings (the four VOICES): the report gives
paired sign tests of its lead and whether it holds on both halves of the phrases. A run on voices none of which is in
VOICES is a held-out run: its report compares DEFAULT_RECOGNISER with the other modes instead of choosing one, and
--attach-held-out adds its summary to the main report (the voices are held out; the phrases are the same). Nothing is
downloaded: the Vosk model must be in models/vosk/ (make voice-setup).

    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_voice.py                  # writes results/live/voice_eval.{md,json}
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_voice.py --voices Samantha --no-noise --limit 10 --no-write   # smoke
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_voice.py --voices Rishi,Tessa,Fred --open-check "" --out DIR
                                                     # held-out voices: writes DIR/voice_eval.{md,json}
    .venv/bin/python scripts/eval_voice.py --report-only --attach-held-out DIR/voice_eval.json
                                                     # adds the held-out summary to results/live/voice_eval.{md,json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from seeforme.live.voice import (DEFAULT_RECOGNISER, VoiceListener, VoskBackend, load_vosk_model,  # noqa: E402
                                 parse_command, snap_text, vosk_status)

OUT = ROOT / "results" / "live"
VECTORS = ROOT / "tests" / "data" / "voice_commands.json"
# en_US, en_GB, en_AU, en_IE: macOS voices that ship with the system. The default recogniser was chosen on these four;
# a run on none of them is a held-out run (held_out_run)
VOICES = ("Samantha", "Daniel", "Karen", "Moira")
ACCENT = {"Samantha": "en_US", "Daniel": "en_GB", "Karen": "en_AU", "Moira": "en_IE", "Rishi": "en_IN", "Tessa": "en_ZA",
          "Fred": "en_US", "Aman": "en_IN", "Tara": "en_IN", "Kathy": "en_US", "Ralph": "en_US", "Albert": "en_US"}
MODES = ("open", "grammar", "hybrid", "snap")
SAMPLERATE = 16000

# natural ways to say the commands, written for this eval (not used to build the grammar): (text, intent, arg)
NATURAL: List[Tuple[str, str, Optional[str]]] = [
    ("SeeForMe, where is the nearest exit", "find", "exit"),
    ("SeeForMe, what's in front of me", "ask", "what's in front of me"),
    ("See for me, can you find the stairs", "find", "stairs"),
    ("See for me, please find a chair", "find", "chair"),
    ("See for me, where's the bathroom", "find", "bathroom"),
    ("See for me, I'm looking for the elevator", "find", "elevator"),
    ("See for me, take me to room two oh five", "find", "room 205"),
    ("See for me, look for gate fifteen", "find", "gate 15"),
    ("See for me, find the bus stop", "find", "bus stop"),
    ("See for me, help me find the door", "find", "door"),
    ("See for me, what do you see now", "describe", None),
    ("See for me, describe the room", "describe", None),
    ("See for me, what is around me", "describe", None),
    ("See for me, read the sign for me", "read", None),
    ("See for me, what does it say", "read", None),
    ("See for me, stop talking", "pause", None),
    ("See for me, hold on a second", "pause", None),
    ("See for me, turn the sound off", "mute", None),
    ("See for me, sound on", "unmute", None),
    ("See for me, say that again please", "repeat", None),
    ("See for me, what can I say", "help", None),
    ("See for me, start again", "start", None),
    ("See for me, keep going", "start", None),
    ("See for me, shut down", "quit", None),
    ("See for me, is there a bench nearby", "ask", "is there a bench nearby"),
    ("See for me, is the way ahead clear", "ask", "is the way ahead clear"),
    ("See for me, switch to reading mode", "mode", "read"),
    ("See for me, give me more details", "verbosity", "detailed"),
    ("See for me, stop looking for the door", "stop_find", None),
    ("See for me, flip the camera", "flip", None),
]
# not commands, but full of "see", "for", "me" and their sound-alikes: nothing may wake (kind: soft | near)
NEGATIVES: List[Tuple[str, str]] = [
    ("I'll see you for lunch tomorrow, call me.", "soft"),
    ("Can you see the bus coming for me?", "soft"),
    ("Wait for me at the corner, I will see you there.", "soft"),
    ("Let me see, four men walked past the door.", "soft"),
    ("She came to see me before the meeting.", "soft"),
    ("For me, the best part was the music.", "soft"),
    ("I see what you mean, thanks for telling me.", "soft"),
    ("Give me a minute to look for my keys.", "soft"),
    ("We went to see a movie for my birthday.", "soft"),
    ("Tell me what you see in the picture.", "soft"),
    ("The seats in front are for members only.", "soft"),
    ("See you later, and save a seat for me.", "soft"),
    ("Do you see four or five people over there?", "soft"),
    ("Is this coffee for me or for you?", "soft"),
    ("I can see the station from here.", "soft"),
    ("Come and see the view, it is lovely.", "soft"),
    ("Let's see, for me it's fine.", "near"),
    ("The sea for me is too cold in winter.", "near"),
    ("I saw four meals on the table.", "near"),
    ("Say four more words, then stop.", "near"),
]


def held_out_run(voices: List[str]) -> bool:
    """True when none of `voices` is one the default recogniser was chosen on (VOICES)."""
    return bool(voices) and not ({v.split()[0] for v in voices} & set(VOICES))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def synthesize(text: str, voice: str, work: Path) -> Path:
    """16 kHz mono LEI16 WAV of `text` in `voice` (cached by voice and text)."""
    key = hashlib.sha1(f"{voice}\0{text}".encode()).hexdigest()[:16]
    out = work / f"{voice.split()[0]}_{key}.wav"
    if not out.exists():
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=work)       # parallel calls never share a temporary file
        os.close(fd)
        for attempt in range(3):                                  # `say` can stall for minutes on a loaded machine
            try:
                subprocess.run(["say", "-v", voice, "-o", tmp, f"--data-format=LEI16@{SAMPLERATE}", text], check=True,
                               timeout=120)
                break
            except subprocess.TimeoutExpired:
                if attempt == 2:
                    os.unlink(tmp)
                    raise
        os.replace(tmp, out)
    return out


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise, unit RMS: a fan, traffic hum, a room."""
    f = np.fft.rfftfreq(n)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(np.maximum(f, 1.0 / n))
    x = np.fft.irfft(spec, n)
    return x / max(1e-9, float(np.sqrt(np.mean(x ** 2))))


def load(path: Path, snr_db: Optional[float], seed: int) -> np.ndarray:
    """The WAV as float samples, with pink noise at `snr_db` (relative to the speech RMS) over the whole clip, also
    the silence the backend pads around it (the noise floor is continuous in a room)."""
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768.0
    lead, tail = int(0.3 * SAMPLERATE), int(0.6 * SAMPLERATE)      # (the backend adds 0.5 s and 1 s of silence)
    x = np.concatenate([np.zeros(lead), x, np.zeros(tail)])
    if snr_db is not None:
        voiced = x[np.abs(x) > 0.01]
        rms = float(np.sqrt(np.mean(voiced ** 2))) if len(voiced) else 0.05
        x = x + pink(len(x), np.random.default_rng(seed)) * rms / (10 ** (snr_db / 20))
    return np.clip(x, -1.0, 1.0)


class _Still:
    """A listener clock that does not move: the follow-up window after a bare wake word stays open across the
    utterance, as it would in real time (decoding here can be slower than real time on a loaded machine)."""

    def __call__(self) -> float:
        return 0.0


def _listen(texts: List[str]) -> List:
    """The hands-free listener on the final results of one utterance, as VoskBackend._final hands them on."""
    lis = VoiceListener(None, clock=_Still())
    for t in texts:
        t = " ".join(w for w in (t or "").split() if w != "[unk]").strip()
        if t:
            lis.hear(t, source="vosk", audio=True)
    return lis.poll()


def _scored(cmds: List, heard: List[str], cpu: float) -> Dict:
    last = next((c for c in reversed(cmds) if c.intent != "wake"), cmds[-1] if cmds else None)
    return {"heard": heard, "commands": [[c.intent, c.arg] for c in cmds], "got": [last.intent, last.arg] if last else None,
            "decode_s": cpu}


def recognise(samples: np.ndarray, model) -> Dict[str, Dict]:
    """One utterance through the live code path (VoskBackend hybrid, both readings recorded -> VoiceListener), scored
    for every mode on those readings."""
    lis = VoiceListener(None, clock=_Still())
    vb = lis.add(VoskBackend(source=samples, recogniser="hybrid", model=model, second_pass="always"))
    lis.start()
    vb.join(timeout=600)
    raw = vb.raw
    out = {"hybrid": _scored(lis.poll(), vb.results, vb.decode_s)}
    g = [e["text"] for e in raw]
    o = [e["second"] or "" for e in raw]
    out["grammar"] = _scored(_listen(g), g, sum(e["cpu"] for e in raw))
    out["open"] = _scored(_listen(o), o, sum(e["cpu_second"] for e in raw))
    snap, cpu = [], 0.0
    for e in raw:
        ot = e["second"] or ""
        oc = parse_command(ot, require_wake=False) if ot else None
        need = bool(ot) and (oc is None or oc.intent == "ask")
        snap.append(snap_text(ot, e["text"] if need else None))
        cpu += e["cpu_second"] + (e["cpu"] if need else 0.0)
    out["snap"] = _scored(_listen(snap), snap, cpu)
    for r in out.values():
        r["error"] = vb.error
    return out


def recognise_open(samples: np.ndarray, model) -> Dict:
    """The real open mode alone (its own end-pointing), for the check."""
    lis = VoiceListener(None, clock=_Still())
    vb = lis.add(VoskBackend(source=samples, recogniser="open", model=model))
    lis.start()
    vb.join(timeout=600)
    return _scored(lis.poll(), vb.results, vb.decode_s)


def items(vectors: dict) -> List[Dict]:
    out = []
    for c in vectors["wake"]:
        if not c["text"].strip():
            continue
        if c["intent"] is None:
            out.append({"set": "negative_vector", "text": c["text"], "expect": None, "kind": "vector"})
        else:
            out.append({"set": "vector", "text": c["text"], "expect": [c["intent"], c["arg"]]})
    out += [{"set": "natural", "text": t, "expect": [i, a]} for t, i, a in NATURAL]
    out += [{"set": "negative", "text": t, "expect": None, "kind": k} for t, k in NEGATIVES]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voices", default=",".join(VOICES))
    ap.add_argument("--open-check", default="Samantha", help="voices whose clean speech also goes through the real open "
                    "mode alone (empty: none)")
    ap.add_argument("--snr", type=float, default=20.0, help="pink noise level for the noisy copy (dB below the speech)")
    ap.add_argument("--no-noise", action="store_true", help="clean speech only")
    ap.add_argument("--limit", type=int, default=0, help="first N phrases only (smoke test)")
    ap.add_argument("--work", default="", help="folder for the synthesised WAVs (default: a temporary folder; reused if given)")
    ap.add_argument("--vosk-model", default=None)
    ap.add_argument("--say-jobs", type=int, default=6, help="parallel `say` processes for the synthesis")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="folder for voice_eval.json and voice_eval.md (default results/live)")
    ap.add_argument("--report-only", action="store_true", help="re-render OUT/voice_eval.md from OUT/voice_eval.json")
    ap.add_argument("--attach-held-out", default="", help="with --report-only: the voice_eval.json of a held-out run; "
                    "its summary is stored in OUT/voice_eval.json and reported at the end of OUT/voice_eval.md")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.attach_held_out and not args.report_only:
        raise SystemExit("--attach-held-out needs --report-only")
    if args.report_only:
        res = json.loads((out_dir / "voice_eval.json").read_text())
        if args.attach_held_out:
            res["held_out_check"] = held_out_summary(json.loads(Path(args.attach_held_out).read_text()), res)
            (out_dir / "voice_eval.json").write_text(json.dumps(res, indent=1))
            print(f"wrote {out_dir / 'voice_eval.json'}")
        (out_dir / "voice_eval.md").write_text(report(res))
        print(f"wrote {out_dir / 'voice_eval.md'}")
        return
    ok, why = vosk_status(args.vosk_model)
    if not ok or shutil.which("say") is None:
        raise SystemExit(f"needs vosk and its model ({why}) and macOS say")
    voices = [v for v in args.voices.split(",") if v]
    modes = list(MODES)
    check_voices = [v for v in args.open_check.split(",") if v in voices]
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="seeforme_voice_"))
    work.mkdir(parents=True, exist_ok=True)
    conds = [("clean", None)] + ([] if args.no_noise else [(f"pink {args.snr:g} dB", args.snr)])
    its = items(json.loads(VECTORS.read_text()))
    if args.limit:
        its = its[:args.limit]
    t0 = time.perf_counter()
    load0 = round(os.getloadavg()[0], 1)

    # the parser alone on the texts (what the recogniser would need to hear exactly)
    for it in its:
        p = parse_command(it["text"], require_wake=True)
        it["parser"] = [p.intent, p.arg] if p else None

    print(f"synthesising {len(its)} phrases x {len(voices)} voices into {work} ...", flush=True)
    # `say` mostly waits on the speech service (about 0.7 s of CPU in 4.5 s on a loaded machine): a few at a time
    from concurrent.futures import ThreadPoolExecutor
    unique = sorted({(it["text"], v) for it in its for v in voices})
    with ThreadPoolExecutor(max_workers=args.say_jobs) as pool:
        made = dict(zip(unique, pool.map(lambda k: synthesize(k[0], k[1], work), unique)))
    wavs = {(i, v): made[(it["text"], v)] for i, it in enumerate(its) for v in voices}
    model = load_vosk_model(args.vosk_model)
    rows, checks = [], []

    def judged(r, it):
        return (r["got"] == it["expect"]) if it["expect"] is not None else (not r["commands"])
    tm = time.perf_counter()
    for n_done, ((i, v), wav) in enumerate(wavs.items()):
        it = its[i]
        for ci, (cname, snr) in enumerate(conds):
            x = load(wav, snr, seed=i * 31 + ci)
            audio_s = round((len(x) + 1.5 * SAMPLERATE) / SAMPLERATE, 2)        # (+ the backend's silence around it)
            for mode, r in recognise(x, model).items():
                rows.append({"mode": mode, "set": it["set"], "kind": it.get("kind"), "text": it["text"], "voice": v,
                             "cond": cname, "expect": it["expect"], "parser": it["parser"], "ok": bool(judged(r, it)),
                             "audio_s": audio_s, **r})
            if v in check_voices and snr is None:
                ro = recognise_open(x, model)
                paired = next(r for r in rows[::-1] if r["mode"] == "open" and r["text"] == it["text"] and r["voice"] == v)
                checks.append({"text": it["text"], "voice": v, "ok_alone": bool(judged(ro, it)), "ok_paired": paired["ok"],
                               "same_command": ro["got"] == paired["got"] and bool(ro["commands"]) == bool(paired["commands"]),
                               "heard_alone": ro["heard"], "heard_paired": paired["heard"]})
        if (n_done + 1) % 50 == 0:
            print(f"{n_done + 1}/{len(wavs)} recordings, {time.perf_counter() - tm:.0f} s", flush=True)
    del model

    def acc(sel) -> Dict:
        k, n = sum(r["ok"] for r in sel), len(sel)
        return {"n": n, "correct": k, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}

    def fw(sel) -> Dict:
        k, n = sum(bool(r["commands"]) for r in sel), len(sel)
        return {"n": n, "false_wakes": k, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}
    res_modes = {}
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        pos = [r for r in mr if r["expect"] is not None]
        neg = [r for r in mr if r["expect"] is None]
        errs = Counter((r["text"], json.dumps(r["expect"]), json.dumps(r["got"]), " / ".join(r["heard"])) for r in pos if not r["ok"])
        res_modes[mode] = {
            "commands": acc(pos), "vectors": acc([r for r in pos if r["set"] == "vector"]),
            "natural": acc([r for r in pos if r["set"] == "natural"]),
            "by_voice": {v: acc([r for r in pos if r["voice"] == v]) for v in voices},
            "by_condition": {c: acc([r for r in pos if r["cond"] == c]) for c, _ in conds},
            "parser_ok_but_heard_wrong": sum(1 for r in pos if not r["ok"] and r["parser"] == r["expect"]),
            "false_wakes": fw(neg), "false_wakes_soft": fw([r for r in neg if r["kind"] == "soft"]),
            "false_wakes_near": fw([r for r in neg if r["kind"] == "near"]),
            "false_wakes_vectors": fw([r for r in neg if r["kind"] == "vector"]),
            "false_wake_examples": sorted({(r["text"], " / ".join(r["heard"])) for r in neg if r["commands"]})[:20],
            "decode_rtf": round(sum(r["decode_s"] for r in mr) / max(1e-9, sum(r["audio_s"] for r in mr)), 3),
            "errors": [{"text": t, "expect": json.loads(e), "got": json.loads(g), "heard": h, "count": c}
                       for (t, e, g, h), c in errs.most_common(40)],
            "backend_errors": sorted({r["error"] for r in mr if r["error"]}),
        }
    nat = [it for it in its if it["set"] == "natural"]
    vec = [it for it in its if it["set"] == "vector"]
    parser_text = {"vectors": f"{sum(it['parser'] == it['expect'] for it in vec)}/{len(vec)}",
                   "natural": f"{sum(it['parser'] == it['expect'] for it in nat)}/{len(nat)}",
                   "natural_misses": [{"text": it["text"], "expect": it["expect"], "parser": it["parser"]}
                                      for it in nat if it["parser"] != it["expect"]],
                   # negatives whose exact words already wake the parser ("sea for me" is a wake mis-hearing by design)
                   "negatives_parser_wakes": [it["text"] for it in its if it["expect"] is None and it["parser"] is not None]}
    # the choice: the most commands right without more false wakes than the open vocabulary; ties go to less CPU
    base_fw = res_modes["open"]["false_wakes"]["rate"] if "open" in res_modes else None
    allowed = [m for m in modes if base_fw is None or res_modes[m]["false_wakes"]["rate"] <= base_fw + 1e-9] or modes
    best = max(allowed, key=lambda m: (res_modes[m]["commands"]["rate"] or 0.0, -res_modes[m]["decode_rtf"]))
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.perf_counter() - t0, 1),
           "settings": {"voices": voices, "conditions": [c for c, _ in conds], "modes": modes, "samplerate": SAMPLERATE,
                        "model": str(Path(args.vosk_model or "models/vosk/vosk-model-small-en-us-0.15")),
                        "load_average": [load0, round(os.getloadavg()[0], 1)], "cpus": os.cpu_count()},
           "phrases": {"vectors": len(vec), "natural": len(nat), "negatives": sum(1 for it in its if it["set"] == "negative"),
                       "negative_vectors": sum(1 for it in its if it["set"] == "negative_vector")},
           "parser_on_text": parser_text, "modes": res_modes, "best": best,
           "open_check": {"voices": check_voices, "n": len(checks), "same_command": sum(c["same_command"] for c in checks),
                          "ok_alone": sum(c["ok_alone"] for c in checks), "ok_paired": sum(c["ok_paired"] for c in checks),
                          "differences": [c for c in checks if not c["same_command"]][:20]},
           "rows": [{k: r[k] for k in ("mode", "set", "text", "voice", "cond", "expect", "got", "heard", "ok")} for r in rows]}
    if held_out_run(voices):
        # (only on a held-out run, so the default run's output is unchanged)
        res["settings"].update({"held_out": True, "default": DEFAULT_RECOGNISER, "selection_voices": list(VOICES),
                                "code": code_digest(), "command": " ".join(
                                    ([f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}"] if os.environ.get("OMP_NUM_THREADS") else [])
                                    + [".venv/bin/python", "scripts/eval_voice.py"] + [_quote(a) for a in sys.argv[1:]])})
    for m in modes:
        x = res_modes[m]
        print(f"{m:8s} commands {x['commands']['correct']}/{x['commands']['n']} ({x['commands']['rate']}) | vectors "
              f"{x['vectors']['rate']} natural {x['natural']['rate']} | false wakes {x['false_wakes']['false_wakes']}/"
              f"{x['false_wakes']['n']} | decode RTF {x['decode_rtf']}")
    print(f"best: {best}")
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "voice_eval.json").write_text(json.dumps(res, indent=1))
        (out_dir / "voice_eval.md").write_text(report(res))
        print(f"wrote {out_dir / 'voice_eval.md'}")


def _quote(a: str) -> str:
    return a if a and all(c.isalnum() or c in "-_./,=" for c in a) else "'" + a.replace("'", "'\\''") + "'"


def _diff(a: Optional[float], b: Optional[float]) -> str:
    return "-" if a is None or b is None else f"{(a - b) * 100:+.1f}"


def _rate(a: Dict) -> str:
    return "-" if a.get("rate") is None else f"{a['rate'] * 100:.0f}%"


def _pc(a: Dict) -> str:
    if not a["n"]:
        return "-"
    lo, hi = a["ci95"]
    return f"{a['rate'] * 100:.1f}% ({a['correct'] if 'correct' in a else a['false_wakes']}/{a['n']}; {lo * 100:.0f}-{hi * 100:.0f}%)"


def paired(rows: List[Dict], a: str, b: str, sets: Tuple[str, ...] = ("vector", "natural")) -> Tuple[int, int, float]:
    """Mode a against mode b on the same recordings of `sets` (default: the commands; ("negative", "negative_vector"):
    the negatives, where right means no false wake): (a right where b is wrong, the reverse, the exact two-sided sign
    test's p on those discordant pairs)."""
    ok: Dict[Tuple, Dict[str, bool]] = {}
    for r in rows:
        if r["set"] in sets:
            ok.setdefault((r["set"], r["text"], r["voice"], r["cond"]), {})[r["mode"]] = bool(r["ok"])
    both = [v for v in ok.values() if a in v and b in v]
    x, y = sum(v[a] and not v[b] for v in both), sum(v[b] and not v[a] for v in both)
    n = x + y
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def half_winners(rows: List[Dict], modes: List[str]) -> List[str]:
    """The mode with the most commands right on each half of the phrases (split by a hash of the text, fixed)."""
    out = []
    for h in (0, 1):
        right: Counter = Counter()
        for r in rows:
            if r["set"] in ("vector", "natural") and int(hashlib.sha256(r["text"].encode()).hexdigest(), 16) % 2 == h:
                right[r["mode"]] += bool(r["ok"])
        out.append(max(modes, key=lambda m: right[m]))
    return out


def in_sample(res: Dict, best: str, others: List[str]) -> str:
    """The choice's caveat: it was made on the reported recordings. How much of the lead is beyond the noise of this
    set (paired sign tests), and whether it holds on both halves of the phrases."""
    rows, M = res.get("rows", []), res["modes"]
    if not rows:
        return "In-sample: the rule chose the mode on these same recordings; there is no held-out set."
    pairs = {m: paired(rows, best, m) for m in others}
    noisy = [m for m in others if pairs[m][2] > 0.05]
    clear = [m for m in others if pairs[m][2] <= 0.05]
    base_fw = M["open"]["false_wakes"]["rate"] if "open" in M else None
    allowed = [m for m in res["settings"]["modes"] if base_fw is None or M[m]["false_wakes"]["rate"] <= base_fw + 1e-9]
    held = ("none of them is held out (a check on voices the choice was not made on is under Held-out voices at the "
            "end)" if res.get("held_out_check") else "there is no held-out set")
    return (f"In-sample: the rule chose the mode on these same recordings and {held}, so every figure "
            "above is one the choice was made on (and the hybrid's wake rule was revised after a first look at these "
            "negatives, see Limits). Paired on the same command recordings, "
            + "; ".join(f"{best} is right where {m} is wrong {x} times and the reverse {y} times (sign test p = {p:.2g})"
                        for m, (x, y, p) in pairs.items()) + ". "
            + (f"Against {' and '.join(noisy)} the difference is within the noise of this set (p > 0.05): no evidence "
               "either way on new speech. " if noisy else "")
            + (f"Against {' and '.join(clear)} it is clear. " if clear else "")
            + "Split by phrase into two fixed halves, the most commands right go to "
            + " and ".join(half_winners(rows, res["settings"]["modes"])) + " (among the modes with no more false wakes "
            "than the open vocabulary: " + " and ".join(half_winners(rows, allowed)) + ").")


NEG_SETS = ("negative", "negative_vector")
# the files whose change makes a held-out run stale: the parser, the recogniser and the shared vectors
CODE_FILES = ("seeforme/live/voice.py", "seeforme/live/commands.py", "tests/data/voice_commands.json")


def code_digest() -> str:
    """A digest of CODE_FILES and this script's phrases (not of its report code, which may change without changing a
    result)."""
    h = hashlib.sha1()
    for f in CODE_FILES:
        h.update((ROOT / f).read_bytes())
    h.update(json.dumps([NATURAL, NEGATIVES]).encode())
    return h.hexdigest()[:12]


def held_out_pairs(rows: List[Dict], default: str, others: List[str]) -> Dict:
    """The default against each other mode on the same recordings: commands, and negatives (right = no false wake)."""
    out = {}
    for m in others:
        x, y, p = paired(rows, default, m)
        fx, fy, fp = paired(rows, default, m, NEG_SETS)
        out[m] = {"commands": {"default_only": x, "other_only": y, "p": p},
                  "negatives": {"default_only": fx, "other_only": fy, "p": fp}}
    return out


def held_out_text(default: str, pairs: Dict) -> str:
    """The paired sign tests of a held-out run, in words."""
    return ("Paired on the same command recordings, "
            + "; ".join(f"{default} is right where {m} is wrong {q['commands']['default_only']} times and the reverse "
                        f"{q['commands']['other_only']} times (sign test p = {q['commands']['p']:.2g})"
                        for m, q in pairs.items())
            + ". On the negatives, "
            + "; ".join(f"{default} stays silent where {m} wakes {q['negatives']['default_only']} times and the reverse "
                        f"{q['negatives']['other_only']} times (p = {q['negatives']['p']:.2g})" for m, q in pairs.items())
            + ".")


def held_out_summary(h: Dict, main: Dict) -> Dict:
    """What the main report carries of a held-out run (its voice_eval.json): per mode commands right and false wakes
    with Wilson intervals, the default against each other mode (paired sign tests), and what the selection rule would
    pick on these voices alone (reported, never used to choose)."""
    st = h["settings"]
    if not st.get("held_out"):
        raise SystemExit("not a held-out run: its voices include one the default recogniser was chosen on")
    if h["phrases"] != main["phrases"] or st["conditions"] != main["settings"]["conditions"]:
        raise SystemExit("the held-out run used other phrases or conditions than the main run")
    if st.get("code") != code_digest():
        raise SystemExit("the parser, recogniser or phrases changed since the held-out run: run it again")
    default, modes = st["default"], st["modes"]
    keep = ("commands", "vectors", "natural", "false_wakes", "false_wakes_soft", "false_wakes_near", "by_voice",
            "by_condition", "decode_rtf")
    return {"generated": h["generated"], "seconds": h["seconds"], "command": st["command"], "voices": st["voices"],
            "conditions": st["conditions"], "selection_voices": st["selection_voices"], "default": default,
            "load_average": st["load_average"], "code": st["code"],
            "modes": {m: {k: h["modes"][m][k] for k in keep} for m in modes},
            "paired": held_out_pairs(h["rows"], default, [m for m in modes if m != default]),
            "rule_pick": h["best"]}


def held_out_section(hc: Dict, M: Dict) -> List[str]:
    """The "Held-out voices" section of the main report."""
    d = hc["default"]
    L = ["## Held-out voices", "",
         f"Held out: the default mode ({d}) was chosen on {', '.join(hc['selection_voices'])}. This check uses "
         + ", ".join(f"{v} ({ACCENT.get(v.split()[0], '?')})" for v in hc["voices"])
         + ", voices the choice was not made on; nothing was chosen or tuned on them. The phrases, the conditions ("
         + " and ".join(hc["conditions"]) + ") and the code path are the same as above, so the voices are held out and "
         f"the phrases are not. Generated {hc['generated']} ({hc['seconds'] / 60:.1f} min) by `{hc['command']}`, then "
         "added here by `.venv/bin/python scripts/eval_voice.py --report-only --attach-held-out <out>/voice_eval.json`.",
         "",
         "| mode | commands right | shared vectors | natural variants | false wakes (all) | false wakes, near-misses |",
         "|---|---|---|---|---|---|"]
    for m, x in hc["modes"].items():
        L.append(f"| {m}{' (default)' if m == d else ''} | {_pc(x['commands'])} | {_pc(x['vectors'])} | "
                 f"{_pc(x['natural'])} | {_pc(x['false_wakes'])} | {_pc(x['false_wakes_near'])} |")
    L += ["", "| mode | " + " | ".join(hc["voices"]) + " | " + " | ".join(hc["conditions"]) + " |",
          "|---|" + "---|" * (len(hc["voices"]) + len(hc["conditions"]))]
    for m, x in hc["modes"].items():
        L.append(f"| {m} | " + " | ".join(_rate(x["by_voice"][v]) for v in hc["voices"]) + " | "
                 + " | ".join(_rate(x["by_condition"][c]) for c in hc["conditions"]) + " |")
    hd, md = hc["modes"][d], M.get(d)
    L += ["", held_out_text(d, hc["paired"]),
          "",
          (f"Against the {len(hc['selection_voices'])} selection voices above, {d} gets "
           f"{_diff(hd['commands']['rate'], md['commands']['rate'])} points of commands right and "
           f"{_diff(hd['false_wakes']['rate'], md['false_wakes']['rate'])} points of false wakes on the held-out voices; "
           f"the selection voices' rate of commands right ({_rate(md['commands'])}) is "
           + _where(md["commands"]["rate"], hd["commands"]["ci95"]) + " the held-out 95% interval, and their false-wake "
           f"rate ({_rate(md['false_wakes'])}) is " + _where(md["false_wakes"]["rate"], hd["false_wakes"]["ci95"])
           + " it. Commands right on the held-out voices against the selection voices, per mode: "
           + ", ".join(f"{m} {_diff(x['commands']['rate'], M[m]['commands']['rate'])}" for m, x in hc["modes"].items()
                       if m in M) + " points. " if md else "")
          + f"The selection rule applied to the held-out voices alone would pick {hc['rule_pick']} (reported only; the "
          "default was not chosen again).", ""]
    return L


def _where(x: Optional[float], ci: List[Optional[float]]) -> str:
    lo, hi = ci
    if x is None or lo is None:
        return "not comparable with"
    return "above" if x > hi else "below" if x < lo else "within"


def report(res: Dict) -> str:
    st, M = res["settings"], res["modes"]
    modes = st["modes"]
    held = bool(st.get("held_out"))
    L = ["# Voice commands through the real recogniser" + (": held-out voices" if held else ""),
         ""]
    if held:
        L += [f"Held out: the default mode ({st['default']}) was chosen on {', '.join(st['selection_voices'])} "
              "(results/live/voice_eval.md); these voices were not used to choose or tune anything. The phrases are the "
              "same, so only the voices are held out.", ""]
    L += [f"Generated {res['generated']} by `{st['command'] if held else 'python scripts/eval_voice.py'}` "
         f"({res['seconds'] / 60:.1f} min; load average "
         f"{st['load_average'][0]} at the start and {st['load_average'][1]} at the end on {st['cpus']} cores). Vosk "
         f"`{Path(st['model']).name}` (offline, 16 kHz), fed through the live demo's code path: `VoskBackend(source=...)` -> "
         "`VoiceListener` (hands-free: the wake phrase is required, a bare \"SeeForMe\" opens a 5 s window for one command) -> "
         "`parse_command`.",
         "",
         f"Speech: macOS `say` voices {', '.join(f'{v} ({ACCENT.get(v, "?")})' for v in st['voices'])}, "
         f"{' and '.join(st['conditions'])} (pink noise relative to the speech level, over the silence around it too). "
         f"Phrases: the {res['phrases']['vectors']} hands-free command vectors of `tests/data/voice_commands.json`, "
         f"{res['phrases']['natural']} natural variants written for this eval (`NATURAL` in the script), and as negatives "
         f"{res['phrases']['negatives']} sentences full of \"see\", \"for\", \"me\" (4 near-misses such as \"Let's see, for me it's "
         f"fine\") plus the {res['phrases']['negative_vectors']} ignore cases of the vectors. A command is right when intent and "
         "argument both match; any command from a negative, even a bare wake, is a false wake.",
         "",
         "Recogniser modes (`VoskBackend(recogniser=...)`, `seeforme/live/voice.py`):",
         "",
         "- **open**: the model's whole vocabulary (what the demo used before).",
         "- **grammar**: `KaldiRecognizer` with a phrase list of the command vocabulary plus `[unk]` (`grammar_phrases()`, "
         "built from the parser's patterns and the FIND targets, not from these test phrases; Vosk makes a bigram model of it).",
         "- **hybrid**: the grammar recogniser hears the words; an utterance in which it heard the wake phrase or `[unk]` (a "
         "word outside the vocabulary) is decoded again by the open recogniser, which decides whether the wake phrase was "
         "said (the grammar, a model of commands that all start with it, forces chatter into \"see for me\") and whose reading "
         "is used for a command with a word the grammar did not know (a FIND target outside the list).",
         "- **snap**: the other way round: the open recogniser; an utterance that does not read as a command (\"see for me "
         "mood\") is decoded again with the grammar, whose reading snaps the near-miss to the command words when it is a "
         "command and adds no wake phrase the open reading lacked.",
         "",
         "Paired design: each recording is decoded once by `VoskBackend(recogniser=\"hybrid\", second_pass=\"always\")`, which "
         "records the grammar's and the open vocabulary's reading of every end-pointed utterance; each mode is scored on those "
         "two readings through its own selection (`hybrid_text`, `snap_text`) and a fresh `VoiceListener`, so the modes differ "
         "only in how they use the same decodes. "
         + (f"Check: the open mode run alone (its own end-pointing) on {', '.join(res['open_check']['voices'])}'s clean speech gave the "
            f"same command as the paired open reading on {res['open_check']['same_command']} of {res['open_check']['n']} recordings "
            f"(right alone {res['open_check']['ok_alone']}, paired {res['open_check']['ok_paired']})." if res.get("open_check", {}).get("n") else ""),
         "",
         "| mode | commands right | shared vectors | natural variants | false wakes (all) | false wakes, plain chatter | false wakes, near-misses | decode CPU / audio s |",
         "|---|---|---|---|---|---|---|---|"]
    for m in modes:
        x = M[m]
        L.append(f"| {m}{' (default)' if m == (st['default'] if held else res['best']) else ''} | {_pc(x['commands'])} | {_pc(x['vectors'])} | "
                 f"{_pc(x['natural'])} | {_pc(x['false_wakes'])} | {_pc(x['false_wakes_soft'])} | {_pc(x['false_wakes_near'])} | "
                 f"{x['decode_rtf']} |")
    L += ["", "Commands right by voice and by condition:", "",
          "| mode | " + " | ".join(st["voices"]) + " | " + " | ".join(st["conditions"]) + " |",
          "|---|" + "---|" * (len(st["voices"]) + len(st["conditions"]))]
    for m in modes:
        x = M[m]
        L.append(f"| {m} | " + " | ".join(_rate(x['by_voice'][v]) for v in st["voices"]) + " | "
                 + " | ".join(_rate(x['by_condition'][c]) for c in st["conditions"]) + " |")
    best = st["default"] if held else res["best"]
    others = [m for m in modes if m != best]
    against = "; ".join(f"against {m}: {_diff(M[best]['commands']['rate'], M[m]['commands']['rate'])} points of commands, "
                        f"{_diff(M[best]['false_wakes']['rate'], M[m]['false_wakes']['rate'])} points of false wakes, decode "
                        f"{M[best]['decode_rtf']} vs {M[m]['decode_rtf']} CPU s per audio s" for m in others) + "."
    if held:
        L += ["", f"**Held out, default {best}** (not chosen on these voices): " + against, "",
              held_out_text(best, held_out_pairs(res["rows"], best, others))
              + f" The selection rule applied to these voices alone would pick {res['best']} (reported only; nothing "
              "is chosen on a held-out run).", ""]
    else:
        L += ["", f"**Choice: {best}** (now `DEFAULT_RECOGNISER`): the most commands right among the modes with no more "
              "false wakes than the open vocabulary, ties to the cheaper decoder. " + against,
              "",
              in_sample(res, best, others),
              ""]
    L += ["Negatives whose exact words already wake the parser (any recogniser that hears them right wakes): "
          + ("; ".join(f"\"{t}\"" for t in res["parser_on_text"].get("negatives_parser_wakes", [])) or "none") + ".",
          "",
          f"Parser alone on the texts (no audio): shared vectors {res['parser_on_text']['vectors']}, natural variants "
          f"{res['parser_on_text']['natural']}"
          + (": the natural variants it misses as text are parser gaps, not recognition errors: "
             + "; ".join(f"\"{e['text']}\" -> {e['parser']} (wanted {e['expect']})" for e in res["parser_on_text"]["natural_misses"])
             if res["parser_on_text"]["natural_misses"] else "") + ".",
          ""]
    for m in modes:
        x = M[m]
        L += [f"## {m}", "",
              f"- {x['parser_ok_but_heard_wrong']} of the {x['commands']['n'] - x['commands']['correct']} wrong commands were parsed "
              "right as text: recognition errors.",
              "- False wakes: " + (("; ".join(f"\"{t}\" heard as \"{h}\"" for t, h in x["false_wake_examples"])) or "none") + ".",
              ""]
        if x["errors"]:
            L += ["Most frequent errors (all voices and conditions):", "", "| said | wanted | got | heard | times |", "|---|---|---|---|---|"]
            for e in x["errors"][:15]:
                L.append(f"| {e['text']} | {e['expect']} | {e['got']} | {e['heard'] or '(nothing)'} | {e['count']} |")
            L.append("")
        if x["backend_errors"]:
            L += [f"Backend errors: {'; '.join(x['backend_errors'])}", ""]
    L += ["## Limits", "",
          "Synthesised voices are clear, steady and free of accent drift, reverberation and street noise; the pink noise is a "
          "mild stand-in for a room. A user's own voice outdoors will do worse in every mode, probably more for the open "
          "vocabulary. The phrases are short commands; long questions (\"is there a bench nearby\") lean on the open recogniser "
          "in hybrid mode. The grammar was written from the parser's patterns before the eval ran. One design change followed a "
          "first look at these negatives (grammar alone woke on 11 of 46 utterances of two voices; the open reading of the same "
          "11 woke on 4, all near-misses whose words contain the wake phrase): the hybrid now lets the open recogniser decide "
          "whether the wake phrase was said. The snap mode and its rule existed before that look. No threshold or phrase list "
          "was fitted to the negatives. Decode CPU is process time on a shared machine; the open vocabulary's share is what "
          "the live demo pays per second of speech.", ""]
    if res.get("held_out_check") and not held:
        L += held_out_section(res["held_out_check"], M)
    return "\n".join(L)


if __name__ == "__main__":
    main()
