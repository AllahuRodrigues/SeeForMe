"""Evaluate schedulers on cached EgoBlind videos against the question-derived weak events.

    python scripts/run_egoblind.py --split test
    python scripts/run_egoblind.py --split holdout --policies pcacs,pcacs_learned:rl,egoblind_learned:rl --noise p_miss=0.2,depth_rel=0.15 --tag degraded

Writes results/egoblind/summary_<split>[_tag].{json,md}, per_scenario_<split>.csv and summary_by_kind_<split>.json.
Per-video CampusBench metrics come from seeforme.campus.metrics (a video without events counts recall 1.0, so per-video
means are dominated by event-free videos); the headline numbers are therefore *pooled over events and cues*:

  recall_all        timely events / all should-speak events          (system: perception + scheduling)
  recall_coverable  ... / events the COCO detector + taxonomy can see (what scheduling could fix)
  recall_detected   ... / events the detector actually saw before t_q (scheduler alone)
  precision         first-and-timely cues / spoken cues   (lower bound: users asked about ~4 things per video)
  quiet_violation   quiet windows ("no obstacle") with a cue inside / quiet windows
  occupancy         speech time / video time;  cues_per_min over the whole split
"""
import argparse, csv
from _common import dump_json
from campus_common import load_campus_cfg
from egoblind_common import RESULTS_E, load_egoblind_split, make_egoblind_policy, parse_noise
from seeforme.campus.metrics import aggregate, evaluate_run, score
from seeforme.campus.oracle import CampusOracle
from seeforme.egoblind import aggregate_extras, egoblind_extras, run_on_egoblind

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--policies", default="oracle,confidence_threshold,confidence_cooldown,cacs,pcacs,pcacs_learned:il,pcacs_learned:rl,egoblind_learned:il,egoblind_learned:rl")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--kinds", default="", help="comma list, e.g. egoblind_safety,egoblind_quiet")
ap.add_argument("--noise", default="", help="replay perturbation, e.g. p_miss=0.2,depth_rel=0.15,p_blackout=0.05")
ap.add_argument("--tag", default="")
args = ap.parse_args()
if (args.noise or args.kinds or args.limit) and not args.tag:
    ap.error("--noise, --kinds and --limit change the results: give them a --tag so the clean summary is not overwritten")

cfg = load_campus_cfg()
noise = parse_noise(args.noise)
scenarios = load_egoblind_split(args.split, require_cache=True, limit=args.limit, kinds=[k for k in args.kinds.split(",") if k] or None)
if not scenarios:
    raise SystemExit("no cached EgoBlind scenarios for this split; run scripts/cache_egoblind_perception.py then build_egoblind_scenarios.py")
dur = {sc.name: sc.duration for sc in scenarios}
total_min = sum(dur.values()) / 60
n_events = sum(len(sc.speak_events()) for sc in scenarios)
n_quiet = sum(len(sc.quiet_events()) for sc in scenarios)
print(f"EgoBlind {args.split}: {len(scenarios)} cached videos, {n_events} should-speak events, {n_quiet} quiet windows, {total_min:.1f} min of video, noise={noise}")
osc = {sc.name: score(run_on_egoblind(CampusOracle(sc), sc, cfg, seed=sc.seed, noise=noise), sc) for sc in scenarios}


def pooled(rows_p, ex_p):
    cnt = lambda rows, rate, n: sum(int(round(getattr(r, rate) * getattr(r, n))) for r in rows if getattr(r, n))
    cnt_e = lambda rows, rate, n: sum(int(round(r[rate] * r[n])) for r in rows if r.get(n) and r.get(rate) is not None)
    n_ev, n_sp = sum(r.n_events for r in rows_p), sum(r.n_spoken for r in rows_p)
    n_cov, n_det, n_q = (sum(e[k] for e in ex_p) for k in ("n_coverable", "n_detected", "n_quiet"))
    occ = sum(r.audio_occupancy * dur[r.scenario] for r in rows_p) / sum(dur[r.scenario] for r in rows_p)
    div = lambda a, b: (a / b) if b else float("nan")
    return {"recall_all": div(cnt(rows_p, "important_cue_recall", "n_events"), n_ev),
            "recall_coverable": div(cnt_e(ex_p, "coverable_recall", "n_coverable"), n_cov),
            "recall_detected": div(cnt_e(ex_p, "detected_recall", "n_detected"), n_det),
            "precision": div(cnt(rows_p, "cue_precision", "n_spoken"), n_sp),
            "quiet_violation": div(cnt_e(ex_p, "quiet_violation_rate", "n_quiet"), n_q),
            "occupancy": occ, "cues_per_min": n_sp / total_min, "haptics_per_min": sum(r.n_haptic for r in rows_p) / total_min,
            "n_events": n_ev, "n_coverable": n_cov, "n_detected": n_det, "n_quiet": n_q, "n_spoken": n_sp}


rows, extras, summary = [], [], {}
for name in args.policies.split(","):
    rows_p, ex_p = [], []
    for sc in scenarios:
        try:
            pol = make_egoblind_policy(name, cfg, sc)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}"); break
        log = run_on_egoblind(pol, sc, cfg, seed=sc.seed, noise=noise)
        rows_p.append(evaluate_run(log, sc))
        ex = egoblind_extras(log, sc); ex.update({"policy": name, "scenario": sc.name, "kind": sc.kind}); ex_p.append(ex)
    if not rows_p:
        continue
    rows += rows_p; extras += ex_p
    a = aggregate(rows_p, osc)[name]
    x = aggregate_extras([{k: v for k, v in e.items() if k not in ("policy", "scenario", "kind")} for e in ex_p])
    p = pooled(rows_p, ex_p)
    summary[name] = {**p, "per_video": a, "per_video_extras": x}
    print(f"{name:22s} recall all/cover/det={p['recall_all']:.2f}/{p['recall_coverable']:.2f}/{p['recall_detected']:.2f} prec={p['precision']:.2f} "
          f"quietViol={p['quiet_violation']:.2f} occ={p['occupancy']:.2f} cues={p['cues_per_min']:4.1f}/min hap={p['haptics_per_min']:.1f} "
          f"lead={x['anticipation_lead_s']:.2f}s lateC={a['late_critical_rate']:.2f} silP={a['silence_precision']:.2f} minMsg={a['min_message_ratio']:.2f} regret={a['regret']:.2f}")

if not rows:
    raise SystemExit("no policy produced results (missing models or configs?); nothing written")
RESULTS_E.mkdir(parents=True, exist_ok=True)
tag = f"_{args.tag}" if args.tag else ""
with open(RESULTS_E / f"per_scenario_{args.split}{tag}.csv", "w", newline="") as f:
    ex_by = {(e["policy"], e["scenario"]): e for e in extras}
    fields = list(rows[0].as_dict().keys()) + [k for k in extras[0] if k not in ("policy", "scenario", "kind")]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in rows:
        d = r.as_dict(); d.update({k: v for k, v in ex_by[(r.policy, r.scenario)].items() if k not in ("policy", "scenario", "kind")}); w.writerow(d)
summary["_meta"] = {"split": args.split, "n_videos": len(scenarios), "n_events": n_events, "n_quiet": n_quiet, "noise": args.noise, "minutes": total_min,
                    "kinds": {k: sum(1 for sc in scenarios if sc.kind == k) for k in sorted({sc.kind for sc in scenarios})}}
dump_json(summary, RESULTS_E / f"summary_{args.split}{tag}.json")
by_kind = {k: aggregate([r for r in rows if r.kind == k], osc) for k in sorted({r.kind for r in rows})}
dump_json(by_kind, RESULTS_E / f"summary_by_kind_{args.split}{tag}.json")
lines = [f"EgoBlind {args.split}: {len(scenarios)} videos, {n_events} should-speak events, {n_quiet} quiet windows, {total_min:.1f} min"
         + (f", noise {args.noise}" if args.noise else "") + "\n",
         "| policy | recall (all events) | recall (coverable) | recall (detected) | precision | quiet-window violations | audio occupancy | cues/min | anticipation lead (s) | late critical rate | regret |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for p, s in summary.items():
    if p.startswith("_"):
        continue
    a, x = s["per_video"], s["per_video_extras"]
    lines.append(f"| {p} | {s['recall_all']:.2f} | {s['recall_coverable']:.2f} | {s['recall_detected']:.2f} | {s['precision']:.2f} | {s['quiet_violation']:.2f} | "
                 f"{s['occupancy']:.2f} | {s['cues_per_min']:.1f} | {x['anticipation_lead_s']:.2f} | {a['late_critical_rate']:.2f} | {a['regret']:.2f} |")
(RESULTS_E / f"summary_{args.split}{tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[1:]))
