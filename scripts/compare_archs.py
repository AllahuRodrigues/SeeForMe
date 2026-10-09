"""The Irie comparison: parameter-matched history encoders on the same data, evaluated on the held-out v2 test
split, overall and per scene kind, with and without the tracker memory ledger.

    SEEFORME_CAMPUS=v2 python scripts/compare_archs.py --models models/arch_mlp1,models/arch_mlpk,models/arch_attn,models/arch_gru,models/arch_gru_mem
Writes results/campus_v2/arch_comparison_<which>.{json,md} (which = il or rl). Each model dir comes from
scripts/train_scheduler.py --arch ... --out ... Regret of a *_mem model is measured against the oracle run with the same
memory ledger (as run_campus.py --memory-ttl does), the others against the frame-only oracle.
"""
import argparse, json
from dataclasses import replace
from pathlib import Path
from _common import dump_json
from campus_common import RESULTS_C, load_campus_cfg, load_campus_split, oracle_scores, text_model
from seeforme.campus.metrics import aggregate, evaluate_run
from seeforme.campus.runner import run_on_campus
from seeforme.ml.scheduler_model import load_learned_scheduler

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="models/arch_mlp1,models/arch_mlpk,models/arch_attn,models/arch_gru,models/arch_gru_mem")
ap.add_argument("--which", default="il", choices=["il", "rl"], help="il or rl checkpoint")
ap.add_argument("--split", default="test")
ap.add_argument("--memory-ttl", type=float, default=6.0, help="ledger TTL used for the *_mem models and the +memory evaluation")
args = ap.parse_args()

cfg = load_campus_cfg()
scenarios = load_campus_split(args.split)
tm = text_model()
_osc_cache = {}


def oracle_for(c):
    """Oracle scores under the same tracker config (memory ledger on or off) as the model being scored."""
    key = (c.tracker.memory_ttl_s, c.tracker.memory_decay)
    if key not in _osc_cache:
        _osc_cache[key] = oracle_scores(scenarios, c)
    return _osc_cache[key]


rows_by, osc_by, meta = {}, {}, {}
for mdir in args.models.split(","):
    label, _, mdir = mdir.rpartition("=") if "=" in mdir else ("", "", mdir)
    d = Path(mdir)
    if not (d / "scheduler_report.json").exists():
        print(f"skip {mdir}: not trained yet"); continue
    rep = json.loads((d / "scheduler_report.json").read_text())
    name = label or d.name.replace("arch_", "")
    use_mem = name.endswith("_mem")
    c = cfg if not use_mem else replace(cfg, tracker=replace(cfg.tracker, memory_ttl_s=args.memory_ttl))
    meta[name] = {"arch": rep.get("arch", "gru"), "n_params": rep.get("n_params"), "memory_ttl": args.memory_ttl if use_mem else 0.0,
                  "which": args.which, "oracle_memory_ttl": c.tracker.memory_ttl_s,
                  "val_score": rep["il"]["val_score"] if args.which == "il" else rep.get("rl", {}).get("val", {}).get("score"),
                  "threshold": rep["il"]["threshold"] if args.which == "il" else None}
    osc_by[name] = oracle_for(c)
    rows_by[name] = [evaluate_run(run_on_campus(load_learned_scheduler(args.which, d, c, sc, text_model=tm, name=name), sc, c, seed=sc.seed), sc)
                     for sc in scenarios]
    a = aggregate(rows_by[name], osc_by[name])[name]
    print(f"{name:10s} ({meta[name]['arch']}, {meta[name]['n_params']} params) AWrec={a['action_weighted_recall']:.3f} lateC={a['late_critical_rate']:.2f} "
          f"prec={a['cue_precision']:.3f} occ={a['audio_occupancy']:.2f} silP={a['silence_precision']:.2f} regret={a['regret']:.2f}")
agg = {n: aggregate(rs, osc_by[n])[n] for n, rs in rows_by.items()}
by_kind = {}
for k in sorted({r.kind for rs in rows_by.values() for r in rs}):
    for n, rs in rows_by.items():
        rk = [r for r in rs if r.kind == k]
        if rk:
            by_kind.setdefault(k, {})[n] = aggregate(rk, osc_by[n])[n]
dump_json({"which": args.which, "split": args.split, "meta": meta, "overall": agg, "by_kind": by_kind}, RESULTS_C / f"arch_comparison_{args.which}.json")
names = [n for n in agg]
lines = [f"Checkpoint: {args.which}, split: {args.split}", "", "| model | encoder | params | action-weighted recall | late critical | precision | occupancy | silence precision | regret |", "|---|---|---|---|---|---|---|---|---|"]
for n in names:
    a = agg[n]; m = meta[n]
    lines.append(f"| {n} | {m['arch']}{' + memory ledger' if m['memory_ttl'] else ''} | {m['n_params']} | {a['action_weighted_recall']:.3f} | {a['late_critical_rate']:.2f} | "
                 f"{a['cue_precision']:.3f} | {a['audio_occupancy']:.2f} | {a['silence_precision']:.2f} | {a['regret']:.2f} |")
lines += ["", "Action-weighted recall per scene kind:", "", "| kind | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for k, a in by_kind.items():
    lines.append(f"| {k} | " + " | ".join(f"{a[n]['action_weighted_recall']:.2f}" if n in a else "-" for n in names) + " |")
(RESULTS_C / f"arch_comparison_{args.which}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
