"""Which term of the P-CACS utility keeps real-video events below the speak threshold?

    python scripts/decompose_pcacs_utility.py --split train            # COCO caches
    SEEFORME_EGOBLIND_CACHE=data/egoblind/cache_ow python scripts/decompose_pcacs_utility.py --split holdout --tag ow

Replays the linear P-CACS on cached EgoBlind videos. For every coverable should-speak event, at every decision step inside
its window where a track carries the event's object, the utility is recomputed from the logged features and split into
value, novelty factor, uncertainty penalty (detector and depth confidence), speech cost and queue cost. Counterfactuals
ask whether the event would have crossed the speak threshold if one factor were ideal:

  no_uncertainty   uncertainty penalty removed
  depth_certain    depth confidence set to 1 (detector confidence kept)
  on_path          path probability 1 (the object is in the walking corridor)
  imminent         urgency 1 (time to contact inside the reaction horizon)
  on_path_imminent both
  free_speech      speech and queue costs removed

Writes results/egoblind/utility_decomposition_<split>[_tag].{json,md}.
"""
import argparse, math, statistics as st
from collections import Counter, defaultdict
from _common import dump_json
from campus_common import load_campus_cfg
from egoblind_common import RESULTS_E, load_egoblind_split, make_egoblind_policy
from seeforme.egoblind import run_on_egoblind

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--policy", default="pcacs")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()
cfg = load_campus_cfg()
scenarios = load_egoblind_split(args.split, require_cache=True, limit=args.limit)
CF = ["actual", "no_uncertainty", "depth_certain", "on_path", "imminent", "on_path_imminent", "free_speech"]


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (None, None)
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def value_of(f, w, path=None, urg=None):
    p = f.path_prob if path is None else path
    u = f.urgency if urg is None else urg
    hz = f.severity * p if (path is not None and not f.is_text) else f.hazard
    near = 1.0 if f.distance <= 9.0 else 0.0
    aware = (max(0.0, 1.0 - f.distance / 6.0) * (1.0 if abs(f.angle) < 25 else 0.4) * f.severity * (1 - f.is_text)) if w.w_aware > 0 else 0.0
    return (w.w_hazard * hz * u + w.w_ttc * hz * max(0.0, u - 0.5) * 2
            + w.w_action * f.actionability * p * (0.4 + 0.6 * u) * (1 - f.is_text)
            + w.w_text * f.text_importance * p * near * f.is_text + w.w_goal * f.goal * near + w.w_aware * aware)


def utilities(f, w):
    nov = 0.35 + 0.65 * f.novelty
    unc_pen = w.w_uncert * f.uncertainty
    costs = w.lam_speech * f.speech_cost + w.lam_queue * f.queue_cost
    base = value_of(f, w) * nov
    unc_depth1 = 1.0 - f.confidence * 1.0
    return {"actual": base - unc_pen - costs,
            "no_uncertainty": base - costs,
            "depth_certain": base - w.w_uncert * unc_depth1 - costs,
            "on_path": value_of(f, w, path=1.0) * nov - unc_pen - costs,
            "imminent": value_of(f, w, urg=1.0) * nov - unc_pen - costs,
            "on_path_imminent": value_of(f, w, path=1.0, urg=1.0) * nov - unc_pen - costs,
            "free_speech": base - unc_pen,
            "_parts": {"value": value_of(f, w), "novelty_factor": nov, "uncertainty_penalty": unc_pen, "costs": costs,
                       "path_prob": f.path_prob, "urgency": f.urgency, "depth_conf": f.depth_conf, "confidence": f.confidence,
                       "distance": f.distance, "severity": f.severity, "ttc": f.ttc}}


rows, mismatch = [], 0
for sc in scenarios:
    pol = make_egoblind_policy(args.policy, cfg, sc)
    w, thr = pol.cfg.weights, pol.cfg.threshold_speak
    events = {e.obj_id: e for e in sc.speak_events() if e.detail.get("coverable")}
    best = defaultdict(lambda: {k: -math.inf for k in CF})
    parts_at_best = {}

    def hook(t, tracks, obs, cue, ch, due):
        global mismatch
        for o in obs:
            oid = o.track.source_id
            if oid not in events or not events[oid].in_window(t):
                continue
            us = utilities(o.event, w)
            if abs(us["actual"] - o.score) > 1e-6:
                mismatch += 1
            b = best[oid]
            if us["actual"] > b["actual"]:
                parts_at_best[oid] = us["_parts"]
            for k in CF:
                b[k] = max(b[k], us[k])

    log = run_on_egoblind(pol, sc, cfg, seed=sc.seed, on_decision=hook)
    spoken = {c.source_id for c in log.spoken if c.source_id in events and events[c.source_id].in_window(c.t)}
    for oid, e in events.items():
        if oid not in best:
            continue
        rows.append({"video": sc.name, "obj_id": oid, "label": e.label, "spoken_timely": oid in spoken, "thr": thr,
                     **{k: best[oid][k] for k in CF}, **parts_at_best.get(oid, {})})

thr = rows[0]["thr"] if rows else float("nan")
silent = [r for r in rows if not r["spoken_timely"] and r["actual"] < r["thr"]]
summary = {"split": args.split, "policy": args.policy, "n_scored_events": len(rows), "n_below_threshold": len(silent),
           "threshold": thr, "recompute_mismatches": mismatch,
           "crosses_threshold_if": {k: sum(1 for r in silent if r[k] >= r["thr"]) for k in CF[1:]},
           "share_ci95": {k: wilson(sum(1 for r in silent if r[k] >= r["thr"]), len(silent)) for k in CF[1:]},
           "median_at_best_step": {k: (st.median([r[k] for r in silent if k in r]) if silent else None)
                                   for k in ("value", "novelty_factor", "uncertainty_penalty", "costs", "path_prob", "urgency",
                                             "depth_conf", "confidence", "distance", "ttc", "severity")},
           "labels_below_threshold": dict(Counter(r["label"] for r in silent).most_common(10))}
tag = f"_{args.tag}" if args.tag else ""
dump_json({"summary": summary, "rows": rows}, RESULTS_E / f"utility_decomposition_{args.split}{tag}.json")
n = len(silent)
lines = [f"# P-CACS utility decomposition on EgoBlind {args.split} ({len(scenarios)} videos)", "",
         f"{len(rows)} coverable events were scored at least once inside their window; {n} never reached the speak threshold "
         f"({thr:.2f}). Recomputed utility matched the policy's score at every step: {'yes' if mismatch == 0 else f'no ({mismatch} mismatches)'}.", "",
         "Each counterfactual changes one factor on the *same* events, so the rows compare interventions on identical events. "
         "These are oracle what-ifs that set the factor to its most favourable value for every event (depth confidence 1, path "
         "probability 1, urgency 1, penalty or cost removed): upper bounds on how far the threshold binds, not what a perfect "
         "estimator of the true geometry would give, and not results a real estimator has achieved.", "",
         "| if this factor were ideal | events that would cross the threshold | share of the silent events [95% Wilson] |", "|---|---|---|"]
for k in CF[1:]:
    c = summary["crosses_threshold_if"][k]; lo, hi = summary["share_ci95"][k]
    lines.append(f"| {k} | {c} | {c / n:.2f} [{lo:.2f}, {hi:.2f}] |" if n else f"| {k} | 0 | - |")
lines += ["", "Median feature values of the silent events at their best-scoring step:", "", "| feature | median |", "|---|---|"]
for k, v in summary["median_at_best_step"].items():
    lines.append(f"| {k} | {v:.2f} |" if v is not None else f"| {k} | - |")
lines += ["", "Labels of the silent events: " + ", ".join(f"{k} {v}" for k, v in summary["labels_below_threshold"].items())]
(RESULTS_E / f"utility_decomposition_{args.split}{tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
