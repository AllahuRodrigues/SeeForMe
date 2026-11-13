"""One-shot confirmation on `select` of the speak/silence model chosen on `finetune` (docs/online.md item 18).

    python scripts/confirm_decision_model.py --models logistic,gru_k8

Loads the frozen models saved by scripts/train_decision_models.py (no refitting, no tuning), streams
train -> finetune -> select exactly like the finetune replay (the earlier parts only let the budget controller settle),
and reports the `select` videos only: all-event and coverable recall, quiet-window violations, cues/min, speech
seconds/min, and a paired video bootstrap of each model minus logistic. Online greedy is included as the current
system. Run once; the result decides whether the chosen model is promoted. Holdout and test are never loaded.
"""
import argparse, csv, random
from collections import defaultdict

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, source_recordings
from seeforme.campus.metrics import evaluate_run
from seeforme.egoblind import egoblind_extras
from seeforme.online.decision_models import DecisionModel, FrozenScorerPolicy
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="logistic,gru_k8")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

cfg = load_campus_cfg()
src = source_recordings()
stream = []
for s in ("train", "finetune", "select"):
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    stream += part
dur = {sc.video_id: sc.duration for sc in stream}
names = [m for m in args.models.split(",") if m]
models = {m: DecisionModel.load(ROOT / "models" / "decision_models" / f"{m}.pt") for m in names}
rows = []
for seed in [int(s) for s in args.seeds.split(",")]:
    for nm in names + ["online_greedy"]:
        pol = (OnlinePolicy(mode="greedy", target_per_min=2.0, tau0=0.6, lr=0.5, seed=seed) if nm == "online_greedy"
               else FrozenScorerPolicy(models[nm], target_per_min=2.0, tau0=0.6, seed=seed, name=nm))
        for sc, log in run_stream(pol, stream, cfg, p_tap=0.5, seed=seed):
            if sc.split == "select":
                r = evaluate_run(log, sc).as_dict(); r.update(egoblind_extras(log, sc))
                r.update({"policy": nm, "seed": seed, "video_id": sc.video_id})
                rows.append(r)
        print(f"{nm} seed {seed} done", flush=True)

g = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else 0.0
tab = defaultdict(dict)
for r in rows:
    tab[r["policy"]][(r["seed"], r["video_id"])] = r


def pooled(pol, keys):
    t = defaultdict(float)
    for k in keys:
        r = tab[pol][k]
        ne, nc, nq = g(r, "n_events"), g(r, "n_coverable"), g(r, "n_quiet")
        t["ev"] += ne; t["hit"] += round(g(r, "important_cue_recall") * ne)
        t["cov"] += nc; t["cov_hit"] += round(g(r, "coverable_recall") * nc)
        t["q"] += nq; t["qb"] += round(g(r, "quiet_violation_rate") * nq)
        t["sp"] += g(r, "n_spoken"); t["min"] += dur[k[1]] / 60; t["sps"] += g(r, "audio_occupancy") * dur[k[1]]
    d = lambda a, b: a / b if b else float("nan")
    return {"all_event_recall": d(t["hit"], t["ev"]), "hits": t["hit"], "events": t["ev"], "coverable_recall": d(t["cov_hit"], t["cov"]),
            "quiet_violation": d(t["qb"], t["q"]), "cues_per_min": d(t["sp"], t["min"]), "speech_s_per_min": d(t["sps"], t["min"])}


keys = sorted(tab["online_greedy"])
vids = sorted({k[1] for k in keys})
res = {p: pooled(p, keys) for p in tab}
rng = random.Random(0)
for p in tab:
    if p == "logistic":
        continue
    diffs = {"all_event_recall": [], "quiet_violation": []}
    for _ in range(args.boot):
        vs = [rng.choice(vids) for _ in vids]
        ks = [(s, v) for v in vs for s in sorted({k[0] for k in keys})]
        a, b = pooled(p, ks), pooled("logistic", ks)
        for m in diffs:
            x = a[m] - b[m]
            if x == x:
                diffs[m].append(x)
    for m, xs in diffs.items():
        xs.sort()
        res[p][f"{m}_vs_logistic"] = [res[p][m] - res["logistic"][m], xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]

out = ROOT / "results" / "decision_models"
dump_json(res, out / "select_confirmation.json")
with open(out / "select_per_video.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(rows)
L = [f"One-shot confirmation on `select` ({len(vids)} videos; seeds {args.seeds} pooled; frozen models give identical cues on every seed)\n",
     "| policy | all-event recall | vs logistic [95% CI] | coverable recall | quiet violations | vs logistic [95% CI] | cues/min | speech s/min |",
     "|---|---|---|---|---|---|---|---|"]
for p, r in res.items():
    c = lambda m: "reference" if p == "logistic" else f"{r[m + '_vs_logistic'][0]:+.3f} [{r[m + '_vs_logistic'][1]:+.3f}, {r[m + '_vs_logistic'][2]:+.3f}]"
    L.append(f"| {p} | {int(r['hits'])}/{int(r['events'])} = {r['all_event_recall']:.3f} | {c('all_event_recall')} | {r['coverable_recall']:.3f} | "
             f"{r['quiet_violation']:.3f} | {c('quiet_violation')} | {r['cues_per_min']:.2f} | {r['speech_s_per_min']:.1f} |")
(out / "select_confirmation.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
