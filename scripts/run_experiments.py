"""Main experiment: baselines vs CACS (manual weights) vs learned CACS on the held-out test split."""
import argparse, csv, json, math
from _common import MODELS, RESULTS, dump_json, load_cfg, load_split
from seeforme.eval import aggregate, evaluate_cues, run_policy_on_scenario
from seeforme.ml.scene_context import SceneContextModel
from seeforme.pipeline import make_policy

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--policies", default="confidence_threshold,confidence_cooldown,cacs,cacs_learned:logreg,cacs_learned:gbm,cacs_learned:mlp_torch,cacs_dagger:logreg,cacs_dagger:gbm,cacs_dagger:mlp13,cacs_seq_il,cacs_seq_rl")
ap.add_argument("--repeats", type=int, default=3, help="detector-noise seeds per scenario")
args = ap.parse_args()

cfg = load_cfg()
scenarios = load_split(args.split)
scene_ctx_path = MODELS / "scene_context.joblib"
rows = []
for name in args.policies.split(","):
    for sc in scenarios:
        for rep in range(args.repeats):
            ctx = SceneContextModel.load(str(scene_ctx_path)) if (scene_ctx_path.exists() and name.startswith("cacs")) else None
            pol = make_policy(name, cfg, str(MODELS), scene_context=ctx)
            cues = run_policy_on_scenario(pol, sc, cfg, scene_context=ctx, seed=sc.seed + 17 * rep)
            m = evaluate_cues(cues, sc, name)
            rows.append(m)
    a = aggregate([r for r in rows if r.policy == name])[name]
    print(f"{name:26s} recall={a['important_cue_recall']:.3f} prec={a['cue_precision']:.3f} "
          f"load={a['audio_load_per_min']:5.1f}/min dup={a['duplicate_rate']:.3f} delay={a['mean_critical_delay']:.2f}s")

RESULTS.mkdir(exist_ok=True)
with open(RESULTS / f"per_scenario_{args.split}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r.as_dict())
agg = aggregate(rows)
dump_json(agg, RESULTS / f"summary_{args.split}.json")

# per-scenario-kind breakdown
kinds = sorted({r.scenario.rsplit("_", 1)[0] for r in rows})
by_kind = {}
for k in kinds:
    sub = [r for r in rows if r.scenario.rsplit("_", 1)[0] == k]
    by_kind[k] = aggregate(sub)
dump_json(by_kind, RESULTS / f"summary_by_kind_{args.split}.json")

# markdown table
lines = ["| policy | important-cue recall | cue precision | audio load (cues/min) | duplicate rate | mean critical delay (s) |",
         "|---|---|---|---|---|---|"]
for p, a in agg.items():
    d = a["mean_critical_delay"]
    lines.append(f"| {p} | {a['important_cue_recall']:.3f} ± {a['recall_sd']:.2f} | {a['cue_precision']:.3f} | "
                 f"{a['audio_load_per_min']:.1f} ± {a['audio_load_sd']:.1f} | {a['duplicate_rate']:.3f} | "
                 f"{'n/a' if d != d else f'{d:.2f}'} |")
(RESULTS / f"summary_{args.split}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
