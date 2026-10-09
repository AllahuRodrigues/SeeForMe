"""Memory ledger expiry grid (Experiment B): TTL x decay on the scene families where memory matters.

    SEEFORME_CAMPUS=v2 python scripts/memory_grid.py --kinds occlusion,crowd,poor_capture --policies oracle,pcacs,pcacs_learned:rl
Writes results/campus_v2/memory_grid_<split>.{json,md}: action-weighted recall / precision per (ttl, decay) per kind.
The TTL / decay choice is made on val (the default); run --split test only to report the chosen setting.
"""
import argparse, itertools
from dataclasses import replace
from _common import dump_json
from campus_common import RESULTS_C, load_campus_cfg, load_campus_split, make_campus_policy
from seeforme.campus.metrics import evaluate_run
from seeforme.campus.runner import run_on_campus

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val", help="val to choose TTL / decay; test only to report the chosen setting")
ap.add_argument("--kinds", default="occlusion,crowd,poor_capture")
ap.add_argument("--policies", default="oracle,pcacs,pcacs_learned:rl")
ap.add_argument("--ttls", default="0,3,6,10")
ap.add_argument("--decays", default="0.8,0.95")
args = ap.parse_args()
cfg0 = load_campus_cfg()
kinds = args.kinds.split(",")
scenarios = [sc for sc in load_campus_split(args.split) if sc.kind in kinds]
grid = [(0.0, 0.0)] + [(t, d) for t in [float(x) for x in args.ttls.split(",") if float(x) > 0] for d in [float(x) for x in args.decays.split(",")]]
res = {}
for ttl, decay in grid:
    cfg = replace(cfg0, tracker=replace(cfg0.tracker, memory_ttl_s=ttl, memory_decay=decay))
    for name in args.policies.split(","):
        for sc in scenarios:
            m = evaluate_run(run_on_campus(make_campus_policy(name, cfg, sc), sc, cfg, seed=sc.seed), sc)
            res.setdefault(f"ttl{ttl:g}_decay{decay:g}", {}).setdefault(name, {}).setdefault(sc.kind, []).append((m.action_weighted_recall, m.cue_precision, m.late_critical_rate))
    print(f"ttl {ttl:g} decay {decay:g} done", flush=True)
summary = {k: {p: {kd: {"recall": sum(r[0] for r in rs) / len(rs), "precision": sum(r[1] for r in rs) / len(rs), "late_critical": sum(r[2] for r in rs) / len(rs)}
                      for kd, rs in kinds_d.items()} for p, kinds_d in pols.items()} for k, pols in res.items()}
dump_json(summary, RESULTS_C / f"memory_grid_{args.split}.json")
lines = ["| memory config | policy | " + " | ".join(f"{k} recall / precision" for k in kinds) + " |", "|---|---|" + "---|" * len(kinds)]
for k, pols in summary.items():
    for p, kd in pols.items():
        lines.append(f"| {k} | {p} | " + " | ".join(f"{kd[x]['recall']:.2f} / {kd[x]['precision']:.2f}" if x in kd else "-" for x in kinds) + " |")
(RESULTS_C / f"memory_grid_{args.split}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
