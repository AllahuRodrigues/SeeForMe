"""Record Python learned-scheduler decisions so the web app can prove it decides the same (web/parity/run.mjs).

The production scheduler (models/scheduler/scheduler_rl.pt, SchedulerPolicy mode "rl", the policy scripts call
"pcacs_learned:rl") runs closed loop on CampusBench scenarios, plus synthetic stress sequences that put many tracks on a
busy channel (so interrupt, haptic and the legality masks are exercised, which the campus runs rarely do). For every
decision step the recording holds
    t, the channel state, and the state of every track passed to decide(), taken BEFORE the call
    Python's answer: the 29 EventFeatures per candidate (confidence-filtered, track order), the model's action and slot
    logits, the decision (action, track id, message, slots, duration) and the budget / walking speed around the step
The node runner rebuilds the tracks from that state and runs the TypeScript feature code, the ONNX model (onnxruntime-web)
and the TypeScript decision rule, then compares
    full path       TS features -> TS windows -> ONNX -> TS rule -> cue message      vs Python
    decision only   Python's feature vectors -> TS windows -> ONNX -> TS rule       vs Python
    features        the 29 TS values vs Python's on every candidate of every step

Text tracks use the keyword text-importance rules (the web app has no learned text classifier); CampusBench evaluation
uses the learned one, which only changes text_importance on sign tracks.

    python scripts/web_parity.py                        # web/parity/fixtures/parity_small.json.gz (committed fixture)
    python scripts/web_parity.py --per-kind 1 --synthetic 3 --out /tmp/p.json --check
    python scripts/web_parity.py --all --synthetic 3 --out /tmp/p.json --check   # all 66 test scenarios
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_common import CAMPUS, ROOT, load_campus_cfg, load_campus_split  # noqa: E402
from seeforme.campus.runner import run_on_campus  # noqa: E402
from seeforme.campus.world import LABEL_EVENT  # noqa: E402
from seeforme.cues.pcacs import ChannelState, EventFeatures  # noqa: E402
from seeforme.cues.text_importance import KeywordTextImportance  # noqa: E402
from seeforme.ml.scheduler_model import SchedulerModel, SchedulerPolicy  # noqa: E402
from seeforme.types import Track  # noqa: E402

MODEL = ROOT / "models" / "scheduler" / "scheduler_rl.pt"
DEFAULT_OUT = ROOT / "web" / "parity" / "fixtures" / "parity_small.json.gz"
RUNNER = ROOT / "web" / "parity" / "run.mjs"


def _f(x):
    """JSON-safe float (repr round-trips exactly); inf / nan / None -> None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def dump_track(tr: Track) -> dict:
    bh = list(tr.bbox_history)
    ends = [bh[0], bh[-1]] if len(bh) >= 2 else bh[:1]     # area_growth_rate only reads the first and the last entry
    return {
        "id": int(tr.track_id), "label": tr.label, "kind": tr.kind, "text": tr.text,
        "bbox": [float(v) for v in tr.bbox], "conf": float(tr.confidence),
        "dist": _f(tr.distance_m), "dconf": float(tr.depth_conf), "lat": float(tr.lateral_m),
        "vz": float(tr.rel_speed_mps), "vx": float(tr.lateral_speed_mps), "ttc": _f(tr.ttc_s), "pp": float(tr.path_prob),
        "nhist": len(tr.dist_history),                          # the walking-speed estimate only reads the length
        "bh": [[float(t), [float(v) for v in b]] for t, b in ends],
        "la": _f(tr.last_announced), "lab": [float(v) for v in tr.last_announced_bbox] if tr.last_announced_bbox else None,
        "ac": int(tr.announce_count), "lh": _f(tr.last_haptic), "rem": bool(tr.remembered),
    }


def dump_ch(ch: ChannelState) -> dict:
    return {"busy": bool(ch.busy), "remaining": float(ch.remaining_s), "cu": float(ch.current_utility),
            "cs": float(ch.current_severity), "ct": None if ch.current_track is None else int(ch.current_track)}


class SpyModel:
    """Wraps SchedulerModel to keep the logits of the last forward pass (the policy reads .K, .net, .forward)."""

    def __init__(self, m: SchedulerModel):
        self.m, self.K, self.net, self.last = m, m.K, m.net, None

    def forward(self, windows, grad: bool = False):
        out = self.m.forward(windows, grad)
        self.last = out
        return out


class Recorder(SchedulerPolicy):
    """SchedulerPolicy that records every decide() call (inputs before, outputs after)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.steps = []

    def decide(self, tracks, t, ch=None, scene_ctx=None):
        ch = ch or ChannelState()
        rec = {"t": float(t), "ch": dump_ch(ch), "tracks": [dump_track(tr) for tr in tracks],
               "budget_in": float(self.budget), "ego_in": float(self.ego_speed)}
        self.model.last = None
        cues = super().decide(tracks, t, ch, scene_ctx)
        cands = [tr.track_id for tr in tracks if tr.confidence >= self.cfg.min_confidence]
        rec["cands"] = [int(i) for i in cands]
        rec["vectors"] = [[float(v) for v in self.last_features[i].as_vector()] for i in cands]
        if self.model.last is not None:
            la, _, ls = self.model.last
            rec["logits"] = [[float(v) for v in row] for row in la.numpy().tolist()]
            rec["slot_logits"] = [[float(v) for v in row] for row in ls.numpy().tolist()]
        cue = cues[0] if cues else None
        rec["py"] = {"action": cue.action if cue else "silence", "track": int(cue.track_id) if cue else None,
                     "message": cue.message if cue else "", "slots": list(cue.slots) if cue else [],
                     "duration": float(cue.duration_s) if cue else 0.0, "utility": float(cue.utility) if cue else None}
        rec["budget_out"], rec["ego_out"] = float(self.budget), float(self.ego_speed)
        self.steps.append(rec)
        return cues


def make_policy(model: SchedulerModel, pcfg, goal_tags) -> Recorder:
    return Recorder(SpyModel(model), pcfg, mode="rl", goal_tags=goal_tags, text_model=KeywordTextImportance(),
                    name="pcacs_learned:rl")


# ---------------------------------------------------------------------------
# synthetic stress sequences
# ---------------------------------------------------------------------------

# taxonomy labels + COCO labels Python maps to "decor" (dog, couch, toaster) + open-vocabulary unknown obstacles
SYN_LABELS = sorted(set(LABEL_EVENT) | {"person", "car", "curb", "stairs", "debris", "unknown object", "chair", "dog", "couch", "toaster"})
# no phrase-only goal matches ("Room 204 Lab" for "room 204"): the web Find matches those on purpose, Python does not
SYN_TEXTS = ["EXIT", "FIRE EXIT", "ROOM 204", "SALE 50% OFF", "STAIRS", "Café Menu", "WELCOME", "DO NOT ENTER", "Salle 204", "B12", "SAÍDA"]


def synthetic_run(model: SchedulerModel, pcfg, seed: int, n_steps: int = 120, n_tracks: int = 9):
    """A random walk over track states on an often busy channel, through the real SchedulerPolicy.decide."""
    rng = random.Random(seed)
    goal = ["exit", "seat", "room 204"] if seed % 2 == 0 else []
    pol = make_policy(model, pcfg, goal)
    pol.reset()
    tracks = []
    for i in range(n_tracks):
        text_kind = i < 2
        lab = "text" if text_kind else rng.choice(SYN_LABELS)
        tracks.append(Track(track_id=i + 1, label=lab, kind="text" if text_kind else "object", bbox=(0.4, 0.4, 0.6, 0.7),
                            confidence=0.8, first_seen=0.0, last_seen=0.0, text=rng.choice(SYN_TEXTS) if text_kind else None))
    for k in range(n_steps):
        t = 0.5 * (k + 1)
        live = []
        for tr in tracks:
            if rng.random() < 0.15:
                continue
            if tr.distance_m is None or tr.distance_m < 0.7 or rng.random() < 0.03:
                tr.distance_m = rng.uniform(1.0, 14.0)
                tr.announce_count = rng.choice([0, 0, 1, 2, 4])
            tr.distance_m = max(0.3, tr.distance_m - rng.uniform(-0.2, 1.2))
            tr.depth_conf = rng.choice([0.0, 0.25, 0.5, 0.6, 0.9])
            cx = min(0.95, max(0.05, (tr.bbox[0] + tr.bbox[2]) / 2 + rng.uniform(-0.08, 0.08)))
            h = min(0.9, max(0.02, 1.2 / max(0.5, tr.distance_m) * rng.uniform(0.3, 1.2)))
            w = h * rng.uniform(0.2, 0.9)
            tr.bbox = (cx - w / 2, max(0.0, 0.6 - h / 2), cx + w / 2, min(1.0, 0.6 + h / 2))
            tr.bbox_history.append((t, tr.bbox))
            tr.confidence = rng.choice([0.2, 0.34, 0.35, 0.5, 0.59, 0.6, 0.75, 0.95])
            tr.lateral_m = rng.uniform(-3, 3)
            tr.rel_speed_mps = rng.uniform(-3.5, 0.5)
            tr.lateral_speed_mps = rng.choice([0.0, 0.0, rng.uniform(-1.5, 1.5)])
            approach = -tr.rel_speed_mps
            tr.ttc_s = max(0.0, (tr.distance_m - 0.6) / approach) if approach >= 0.12 else float("inf")
            tr.path_prob = rng.choice([0.0, 0.1, 0.5, 0.65, 0.9, 1.0, rng.random()])
            tr.dist_history.append((t, tr.distance_m, tr.lateral_m))
            tr.remembered = rng.random() < 0.08
            tr.last_seen = t
            live.append(tr)
        busy = rng.random() < 0.45
        cur = rng.choice([None, -1] + [tr.track_id for tr in live]) if busy else None
        ch = ChannelState(busy=busy, remaining_s=rng.uniform(0, 3.5) if busy else 0.0, current_utility=rng.uniform(-1, 4) if busy else 0.0,
                          current_severity=rng.choice([0.0, 0.5, 0.9]) if busy else 0.0, current_track=cur)
        pol.decide(live, t, ch)
    return pol.steps, goal


class StubModel:
    """Returns given logits instead of running the network: drives SchedulerPolicy.decide's own rule code (legality
    masks, joint argmax, silence bias) through cases the trained model rarely produces, e.g. interrupts."""

    def __init__(self, K: int):
        import torch
        self.K, self.next = K, None
        self.net = type("Net", (), {})()
        self.net.silence_bias = torch.zeros(1)

    def forward(self, windows, grad: bool = False):
        import torch
        la, ls = self.next
        assert la.shape[0] == len(windows)
        return la, torch.zeros(len(windows)), ls


def rule_cases(pcfg, n_cases: int, seed: int, K: int) -> list:
    """Random logits / channel / track timing through the real Python decide(); the TS rule must pick the same."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    stub = StubModel(K)
    pol = SchedulerPolicy(stub, pcfg, mode="rl", text_model=KeywordTextImportance())
    out = []
    for c in range(n_cases):
        t = 100.0 + 0.5 * c
        n = int(rng.integers(1, 7))
        tracks = []
        for i in range(n):
            tr = Track(track_id=int(rng.integers(1, 40)) * 10 + i, label="person", kind="object", bbox=(0.4, 0.3, 0.6, 0.9),
                       confidence=0.9, first_seen=0.0, last_seen=t)
            tr.distance_m, tr.depth_conf = 3.0, 0.8
            tr.last_announced = [None, t - 0.5, t - pcfg.recent_block_s, t - 2.0, t - 30.0][int(rng.integers(0, 5))]
            tr.last_haptic = [None, t - 0.4, t - pcfg.haptic_repeat_s, t - 5.0][int(rng.integers(0, 4))]
            tracks.append(tr)
        la = rng.normal(0, 2.0, (n, 4)).astype(np.float32)
        la[:, 2] += rng.choice([0.0, 3.0], n).astype(np.float32)          # often a strong interrupt
        if rng.random() < 0.1:
            la[1 % n] = la[0]                                               # exact ties: the first maximum must win
        ls = rng.normal(0, 2.0, (n, 6)).astype(np.float32)
        sb = np.float32(rng.choice([-2.0, 0.0, 1.9996519088745117, 4.0, rng.normal(1, 2)]))
        busy = bool(rng.random() < 0.5)
        ch = ChannelState(busy=busy, remaining_s=1.0 if busy else 0.0,
                          current_track=(None if not busy else [None, tracks[0].track_id, -1][int(rng.integers(0, 3))]))
        stub.next = (torch.from_numpy(la), torch.from_numpy(ls))
        stub.net.silence_bias = torch.tensor([sb])
        pol.reset()
        pre = [{"trackId": tr.track_id, "lastAnnounced": _f(tr.last_announced), "lastHaptic": _f(tr.last_haptic)} for tr in tracks]
        cues = pol.decide(tracks, t, ch)
        cue = cues[0] if cues else None
        out.append({"t": t, "ch": dump_ch(ch), "inputs": pre, "logits": [[float(v) for v in r] for r in la.tolist()],
                    "silence_bias": float(sb), "py": {"action": cue.action if cue else "silence", "track": int(cue.track_id) if cue else None}})
    return out


# ---------------------------------------------------------------------------

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(args) -> dict:
    import torch
    torch.set_num_threads(2)
    cfg = load_campus_cfg()
    model = SchedulerModel.load(str(MODEL))
    scs = load_campus_split(args.split)
    if not args.all:
        by_kind = {}
        for sc in scs:
            by_kind.setdefault(sc.kind, []).append(sc)
        kinds = sorted(by_kind) if not args.kinds else [k for k in args.kinds.split(",") if k in by_kind]
        scs = [sc for k in kinds for sc in by_kind[k][: args.per_kind]]
    out = {"meta": {"model": str(MODEL.relative_to(ROOT)), "source_sha256": sha256(MODEL), "K": model.K,
                    "features": EventFeatures.names(), "campus": str(CAMPUS.relative_to(ROOT)), "split": args.split,
                    "text_model": "keyword", "recorded": _dt.datetime.now().isoformat(timespec="seconds"),
                    "pcacs": {k: getattr(cfg.pcacs, k) for k in ("min_confidence", "recent_block_s", "haptic_repeat_s", "use_haptic",
                                                                  "use_slots", "verbosity", "units", "ttc_horizon", "budget_capacity_s")}},
           "scenarios": []}
    for sc in scs:
        pol = make_policy(model, cfg.pcacs, sc.goal_tags)
        run_on_campus(pol, sc, cfg, seed=sc.seed)
        out["scenarios"].append({"name": sc.name, "kind": sc.kind, "goal_tags": list(sc.goal_tags), "seed": sc.seed,
                                 "cfg": {}, "steps": pol.steps})
    variants = [{}, {"verbosity": "detailed", "units": "ft"}, {"verbosity": "quiet"}]
    for j in range(args.synthetic):
        v = variants[j % len(variants)]
        pcfg = replace(cfg.pcacs, **v)
        steps, goal = synthetic_run(model, pcfg, seed=1000 + j, n_steps=args.synthetic_steps)
        out["scenarios"].append({"name": f"synthetic_{j}", "kind": "synthetic", "goal_tags": goal, "seed": 1000 + j,
                                 "cfg": {"verbosity": pcfg.verbosity, "units": pcfg.units}, "steps": steps})
    if args.rule_cases:
        out["rule_cases"] = rule_cases(cfg.pcacs, args.rule_cases, seed=7, K=model.K)
    return out


def summarise(rec: dict) -> str:
    from collections import Counter
    acts = Counter(s["py"]["action"] for sc in rec["scenarios"] for s in sc["steps"])
    n = sum(len(sc["steps"]) for sc in rec["scenarios"])
    nc = sum(len(s["cands"]) for sc in rec["scenarios"] for s in sc["steps"])
    rc = Counter(c["py"]["action"] for c in rec.get("rule_cases", []))
    return (f"{len(rec['scenarios'])} scenarios, {n} decision steps, {nc} candidate feature vectors, Python actions {dict(acts)}; "
            f"{sum(rc.values())} rule cases {dict(rc)}")


def run_node(fixture: Path, extra=()) -> int:
    node = shutil.which("node")
    if not node:
        print("node not found: install Node.js to run the TypeScript side", file=sys.stderr)
        return 2
    return subprocess.call([node, str(RUNNER), str(fixture), *extra], cwd=str(ROOT / "web"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-kind", type=int, default=1, help="scenarios per scene kind")
    ap.add_argument("--kinds", default="finding,crossing,stairs", help="comma-separated kinds (empty = all 11)")
    ap.add_argument("--all", action="store_true", help="every scenario of the split")
    ap.add_argument("--synthetic", type=int, default=1, help="synthetic stress sequences (busy channel, interrupts, haptics)")
    ap.add_argument("--synthetic-steps", type=int, default=100)
    ap.add_argument("--rule-cases", type=int, default=400, help="random-logit cases for the decision rule alone (interrupt, ties, masks)")
    ap.add_argument("--check", action="store_true", help="run web/parity/run.mjs on the recording")
    args = ap.parse_args()
    rec = record(args)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rec, separators=(",", ":")).encode()
    if out.suffix == ".gz":
        import gzip
        data = gzip.compress(data, compresslevel=9, mtime=0)
    out.write_bytes(data)
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} KB): {summarise(rec)}")
    if args.check:
        sys.exit(run_node(out))


if __name__ == "__main__":
    main()
