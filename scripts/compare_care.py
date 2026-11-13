"""E2: does the CARE decision rule beat the threshold rule on identical predictions at a matched speech budget?

    python scripts/compare_care.py --model logistic

Both policies see the same candidates and the same frozen usefulness scores (models/decision_models/<model>.pt); only
the decision differs. Threshold: speak the top legal candidate when its score clears the adaptive tau (cue-rate
controller, 2 cues/min). CARE: utility = p_helpful - cost_speech_per_s * duration, silence = 0, under a strict speech
ledger whose refill equals the threshold policy's measured spoken seconds per minute on `train` (matched budget).
The speech-cost weight is chosen on the `train` part; everything is reported on `finetune` (development). `select`,
`holdout` and `test` are never loaded. Writes results/care/e2.{json,md}.
"""
import argparse, random
from collections import defaultdict

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, source_recordings
from seeforme.campus.metrics import evaluate_run
from seeforme.egoblind import egoblind_extras
from seeforme.online.decision_models import DecisionModel, FrozenScorerPolicy
from seeforme.online.stream import run_stream
from seeforme.scheduling.care import CareWeights
from seeforme.scheduling.care_policy import CarePolicy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="logistic")
ap.add_argument("--costs", default="0,0.05,0.1,0.2,0.3")
ap.add_argument("--capacity", type=float, default=6.0)
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

cfg = load_campus_cfg()
src = source_recordings()
stream = []
for s in ("train", "finetune"):
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    stream += part
dur = {sc.video_id: sc.duration for sc in stream}
model = DecisionModel.load(ROOT / "models" / "decision_models" / f"{args.model}.pt")
g = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else 0.0


def run(pol):
    rows = []
    for sc, log in run_stream(pol, stream, cfg, p_tap=0.5, seed=0):
        r = evaluate_run(log, sc).as_dict(); r.update(egoblind_extras(log, sc))
        r.update({"video_id": sc.video_id, "split": sc.split}); rows.append(r)
    return {r["video_id"]: r for r in rows}


def pooled(rows, vids):
    t = defaultdict(float)
    for v in vids:
        r = rows[v]
        ne, nq = g(r, "n_events"), g(r, "n_quiet")
        t["ev"] += ne; t["hit"] += round(g(r, "important_cue_recall") * ne)
        t["q"] += nq; t["qb"] += round(g(r, "quiet_violation_rate") * nq)
        t["sp"] += g(r, "n_spoken"); t["min"] += dur[v] / 60; t["sps"] += g(r, "audio_occupancy") * dur[v]
    d = lambda a, b: a / b if b else float("nan")
    return {"recall": d(t["hit"], t["ev"]), "hits": t["hit"], "events": t["ev"], "quiet_violation": d(t["qb"], t["q"]),
            "cues_per_min": d(t["sp"], t["min"]), "speech_s_per_min": d(t["sps"], t["min"])}


split_of = {sc.video_id: sc.split for sc in stream}
tr = [v for v in dur if split_of[v] == "train"]; ft = [v for v in dur if split_of[v] == "finetune"]
base = run(FrozenScorerPolicy(model, target_per_min=2.0, tau0=0.6, seed=0, name="threshold"))
b_tr = pooled(base, tr)
refill = b_tr["speech_s_per_min"] / 60.0
print(f"threshold train: {b_tr}  -> CARE ledger refill {refill * 60:.2f} s/min", flush=True)
care = {}
for c in [float(x) for x in args.costs.split(",")]:
    care[c] = run(CarePolicy(model, capacity_s=args.capacity, refill_per_s=refill, weights=CareWeights(cost_speech_per_s=c), name=f"care_c{c:g}"))
    print(f"care cost {c:g} train: {pooled(care[c], tr)}", flush=True)
best_c = max(care, key=lambda c: (pooled(care[c], tr)["recall"], -pooled(care[c], tr)["quiet_violation"]))

rng = random.Random(0)


def boot(a, b, key):
    xs = []
    for _ in range(args.boot):
        vs = [rng.choice(ft) for _ in ft]
        x = pooled(a, vs)[key] - pooled(b, vs)[key]
        if x == x:
            xs.append(x)
    xs.sort()
    return [pooled(a, ft)[key] - pooled(b, ft)[key], xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


res = {"refill_s_per_min": refill * 60, "chosen_cost": best_c, "threshold": {"train": b_tr, "finetune": pooled(base, ft)}, "care": {}}
for c, rows in care.items():
    res["care"][f"{c:g}"] = {"train": pooled(rows, tr), "finetune": pooled(rows, ft),
                             "recall_vs_threshold": boot(rows, base, "recall"), "quiet_vs_threshold": boot(rows, base, "quiet_violation")}
out = ROOT / "results" / "care"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / "e2.json")
f = lambda r: f"{int(r['hits'])}/{int(r['events'])} = {r['recall']:.3f} | {r['quiet_violation']:.3f} | {r['cues_per_min']:.2f} | {r['speech_s_per_min']:.2f}"
L = [f"E2 on finetune ({len(ft)} videos): identical frozen `{args.model}` scores; CARE ledger refill {refill * 60:.2f} s/min (threshold's train speech rate), "
     f"capacity {args.capacity:g} s; speech cost chosen on train = {best_c:g}\n",
     "| policy | all-event recall | quiet violations | cues/min | speech s/min | recall vs threshold [95% CI] | quiet vs threshold [95% CI] |",
     "|---|---|---|---|---|---|---|", f"| threshold (adaptive tau) | {f(res['threshold']['finetune'])} | reference | reference |"]
for c, r in res["care"].items():
    rc, qc = r["recall_vs_threshold"], r["quiet_vs_threshold"]
    star = " (chosen)" if float(c) == best_c else ""
    L.append(f"| CARE, cost {c}{star} | {f(r['finetune'])} | {rc[0]:+.3f} [{rc[1]:+.3f}, {rc[2]:+.3f}] | {qc[0]:+.3f} [{qc[1]:+.3f}, {qc[2]:+.3f}] |")
(out / "e2.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
