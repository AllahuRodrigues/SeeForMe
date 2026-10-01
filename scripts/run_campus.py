"""CampusBench experiment: baselines vs CACS vs P-CACS (+ ablations, learned schedulers) on a split."""
import argparse, csv
from _common import dump_json
from campus_common import RESULTS_C, load_campus_cfg, load_campus_split, make_campus_policy, oracle_scores
from seeforme.campus.runner import run_on_campus
from seeforme.campus.metrics import evaluate_run, aggregate

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--policies", default="oracle,confidence_threshold,confidence_cooldown,cacs,pcacs")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg = load_campus_cfg()
scenarios = load_campus_split(args.split)
osc = oracle_scores(scenarios, cfg, repeats=args.repeats)
rows = []
for name in args.policies.split(","):
    for sc in scenarios:
        for rep in range(args.repeats):
            try:
                pol = make_campus_policy(name, cfg, sc)
            except FileNotFoundError as e:
                print(f"skip {name}: {e}"); break
            rows.append(evaluate_run(run_on_campus(pol, sc, cfg, seed=sc.seed + 17 * rep), sc))
    a = aggregate([r for r in rows if r.policy == name], osc).get(name)
    if a:
        print(f"{name:24s} AWrec={a['action_weighted_recall']:.3f} lateC={a['late_critical_rate']:.2f} lead={a['mean_lead_s']:.2f}s "
              f"prec={a['cue_precision']:.3f} occ={a['audio_occupancy']:.2f} cues={a['cues_per_min']:4.1f}/min hap={a['haptics_per_min']:.1f} "
              f"silP={a['silence_precision']:.2f} block={a['critical_blocking_rate']:.2f} minMsg={a['min_message_ratio']:.2f} regret={a['regret']:.2f}")

RESULTS_C.mkdir(parents=True, exist_ok=True)
tag = f"_{args.tag}" if args.tag else ""
with open(RESULTS_C / f"per_scenario_{args.split}{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys())); w.writeheader()
    for r in rows:
        w.writerow(r.as_dict())
agg = aggregate(rows, osc)
dump_json(agg, RESULTS_C / f"summary_{args.split}{tag}.json")
by_kind = {}
for k in sorted({r.kind for r in rows}):
    by_kind[k] = aggregate([r for r in rows if r.kind == k], osc)
dump_json(by_kind, RESULTS_C / f"summary_by_kind_{args.split}{tag}.json")
lines = ["| policy | action-weighted recall | late critical rate | mean lead (s) | cue precision | audio occupancy | cues/min | haptics/min | silence precision | critical blocking | min-message ratio | regret |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for p, a in agg.items():
    lines.append(f"| {p} | {a['action_weighted_recall']:.3f} ± {a['recall_sd']:.2f} | {a['late_critical_rate']:.2f} | {a['mean_lead_s']:.2f} | "
                 f"{a['cue_precision']:.3f} | {a['audio_occupancy']:.2f} | {a['cues_per_min']:.1f} | {a['haptics_per_min']:.1f} | "
                 f"{a['silence_precision']:.2f} | {a['critical_blocking_rate']:.2f} | {a['min_message_ratio']:.2f} | {a['regret']:.2f} |")
(RESULTS_C / f"summary_{args.split}{tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
