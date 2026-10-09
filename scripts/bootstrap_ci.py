"""95% bootstrap intervals for the pooled EgoBlind metrics, resampling *videos* (the unit that was split).

    python scripts/bootstrap_ci.py results/egoblind/per_scenario_holdout_final.csv [--n 2000] [--seed 0]

For every policy in the per-video CSV written by scripts/run_egoblind.py it resamples the videos with replacement and
recomputes the pooled numbers the summary tables report: recall over all events, over coverable events, precision,
quiet-window violations and cues per minute. Paired differences against a reference policy use the same resampled
videos for both policies, so they compare the policies on identical events (--vs pcacs); --pairs a:b,c:d adds paired
differences a minus b for any other pairs. Writes <csv>.ci.{json,md}.
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--vs", default="pcacs", help="reference policy for paired differences")
ap.add_argument("--pairs", default="", help="extra paired differences, comma list of a:b (a minus b, same resampled videos)")
args = ap.parse_args()
rows = list(csv.DictReader(open(args.csv)))
f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan", "None") else float("nan")
by = defaultdict(dict)
for r in rows:
    by[r["policy"]][r["scenario"]] = r
videos = sorted({r["scenario"] for r in rows})


def pooled(pol, vids):
    t = defaultdict(float)
    for v in vids:
        r = by[pol].get(v)
        if r is None:
            continue
        ne, nc, nq, ns = f(r, "n_events"), f(r, "n_coverable"), f(r, "n_quiet"), f(r, "n_spoken")
        t["ev"] += ne; t["ev_hit"] += round(f(r, "important_cue_recall") * ne) if ne else 0
        t["cov"] += nc; t["cov_hit"] += round(f(r, "coverable_recall") * nc) if nc else 0
        t["q"] += nq; t["q_bad"] += round(f(r, "quiet_violation_rate") * nq) if nq else 0
        t["sp"] += ns; t["sp_ok"] += round(f(r, "cue_precision") * ns) if ns else 0
        dur_min = (ns / f(r, "cues_per_min")) if f(r, "cues_per_min") > 0 else None
        t["min"] += dur_min if dur_min else 0.0
    div = lambda a, b: a / b if b else float("nan")
    return {"recall_all": div(t["ev_hit"], t["ev"]), "recall_coverable": div(t["cov_hit"], t["cov"]), "precision": div(t["sp_ok"], t["sp"]),
            "quiet_violation": div(t["q_bad"], t["q"]), "cues": t["sp"]}


def ci(vals):
    vals = sorted(v for v in vals if v == v)
    if not vals:
        return [None, None]
    return [round(vals[int(0.025 * (len(vals) - 1))], 3), round(vals[int(0.975 * (len(vals) - 1))], 3)]


rng = random.Random(args.seed)
samples = [[rng.choice(videos) for _ in videos] for _ in range(args.n)]
out = {"csv": args.csv, "n_videos": len(videos), "n_boot": args.n, "unit": "video", "policies": {}}
for pol in by:
    point = pooled(pol, videos)
    boots = [pooled(pol, s) for s in samples]
    entry = {k: {"value": round(point[k], 3) if point[k] == point[k] else None, "ci95": ci([b[k] for b in boots])} for k in ("recall_all", "recall_coverable", "precision", "quiet_violation")}
    if args.vs in by and pol != args.vs:
        ref = [pooled(args.vs, s) for s in samples]
        for k in ("recall_all", "recall_coverable", "quiet_violation"):
            d = [b[k] - r[k] for b, r in zip(boots, ref)]
            entry[f"diff_vs_{args.vs}_{k}"] = {"value": round(point[k] - pooled(args.vs, videos)[k], 3) if point[k] == point[k] else None, "ci95": ci(d)}
    out["policies"][pol] = entry
KEYS = ("recall_all", "recall_coverable", "quiet_violation")
out["pairs"] = {}
for pair in filter(None, args.pairs.split(",")):
    # policy names contain colons (pcacs_learned:il), so try every split point
    a, b = next(((x, y) for i in range(1, pair.count(":") + 1)
                 for x, y in [(":".join(pair.split(":")[:i]), ":".join(pair.split(":")[i:]))] if x in by and y in by), (None, None))
    if a is None:
        raise SystemExit(f"--pairs {pair!r}: both policies must be in {sorted(by)}")
    pa, pb = pooled(a, videos), pooled(b, videos)
    ba, bb = [pooled(a, s) for s in samples], [pooled(b, s) for s in samples]
    out["pairs"][f"{a} - {b}"] = {k: {"value": round(pa[k] - pb[k], 3) if pa[k] == pa[k] and pb[k] == pb[k] else None,
                                      "ci95": ci([x[k] - y[k] for x, y in zip(ba, bb)])} for k in KEYS}
p = Path(args.csv)
Path(str(p) + ".ci.json").write_text(json.dumps(out, indent=1))
fmt = lambda e: "-" if e["value"] is None else f"{e['value']:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]"
lines = [f"95% bootstrap intervals over {len(videos)} videos ({args.n} resamples, seed {args.seed}) for `{p.name}`", "",
         f"| policy | recall (all) | recall (coverable) | precision | quiet-window violations | coverable recall minus {args.vs} |", "|---|---|---|---|---|---|"]
for pol, e in out["policies"].items():
    d = e.get(f"diff_vs_{args.vs}_recall_coverable")
    lines.append(f"| {pol} | {fmt(e['recall_all'])} | {fmt(e['recall_coverable'])} | {fmt(e['precision'])} | {fmt(e['quiet_violation'])} | {fmt(d) if d else '-'} |")
if out["pairs"]:
    lines += ["", "Paired differences (a minus b, same resampled videos)", "",
              "| a minus b | recall (all) | recall (coverable) | quiet-window violations |", "|---|---|---|---|"]
    lines += [f"| {k} | {fmt(e['recall_all'])} | {fmt(e['recall_coverable'])} | {fmt(e['quiet_violation'])} |" for k, e in out["pairs"].items()]
Path(str(p) + ".ci.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
