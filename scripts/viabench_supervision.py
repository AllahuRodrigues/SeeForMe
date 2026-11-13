"""E3-V: is supervision the constraint? Same candidates and features, labelled by humans instead of by question proxies.

    python scripts/viabench_supervision.py

EgoBlind's weak labels mark when a blind user asked something. VIABench's proactive reminders mark, over whole
videos, when human annotators say an assistant should speak unprompted and what about. This script runs the same
replay on both (YOLO11 + depth cache, tracker, the base 29 candidate features) and asks how well a standardised
logistic regression separates should-speak candidates under each kind of label.

  EgoBlind   fit on `train`, score on `finetune`; labels as in feature_auc.py (inside the event window / near a question)
  VIABench   fit on `via_train`, score on `via_dev`; labels from seeforme.viabench.data.decision_label
             (1 = a reminder naming the candidate's object is active; 0 = no reminder of any kind within 3 s; else masked)

Reported for each: AUC with a bootstrap interval over held-out source videos; a class-only baseline (the detector
label as the only input), because both label rules depend on the object class; AUC inside `person` rows, which
removes the class effect; AUC inside each held-out video (does the score order moments, or only tell busy videos
from quiet ones); a label-permutation reference range; the other dataset's model scored on it (transfer); and for
VIABench a learning curve over the number of training videos. `via_test`, `select`, `holdout` and `test` are never
loaded. Writes results/viabench/supervision.{json,md}.
"""
import argparse, random
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, source_recordings
from viabench_common import load_viabench_split
from seeforme.eval.ranking import null_summary, within_group_auc
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream
from seeforme.viabench.data import decision_label

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--curve", default="10,20,40,80")
ap.add_argument("--curve-seeds", type=int, default=10)
ap.add_argument("--perm", type=int, default=200)
args = ap.parse_args()
rows = []


def ego_label(sc, o, t, announced):
    e = sc.event_for(o.track.source_id) if o.track.source_id is not None else None
    if e is not None and e.should_speak and e.in_window(t) and o.track.source_id not in announced:
        return 1
    return 0 if sc.counts_fp(t) else None


def via_label(sc, o, t, announced):
    return decision_label(t, o.track.label, sc.reminders)


class Dump(OnlinePolicy):
    labeller, dataset = None, ""

    def _learn_full_info(self, obs, X, t):
        sc = self._scenario
        for o in obs:
            y = type(self).labeller(sc, o, t, self._feedback.announced)
            if y is not None:
                rows.append({"ds": self.dataset, "split": sc.split, "video": sc.video_id, "y": int(y), "label": o.track.label,
                             "x": np.asarray(o.event.as_vector(), float)})


cfg = load_campus_cfg()
src = source_recordings()
ego = []
for s in ("train", "finetune"):
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    ego += part
via = load_viabench_split("via_train") + load_viabench_split("via_dev")
for name, scen, lab in (("egoblind", ego, ego_label), ("viabench", via, via_label)):
    D = type("Dump_" + name, (Dump,), {"labeller": staticmethod(lab)})
    pol = D(mode="full_info", lr=0.0, seed=0); pol.dataset = name
    for _ in run_stream(pol, scen, cfg, p_tap=0.5, seed=0):
        pass
    print(f"{name}: {len(scen)} videos, {sum(r['ds'] == name for r in rows)} labelled candidate rows", flush=True)

PARTS = {"egoblind": ("train", "finetune"), "viabench": ("via_train", "via_dev")}
sel = lambda ds, split: [r for r in rows if r["ds"] == ds and r["split"] == split]
XY = lambda rs: (np.stack([r["x"] for r in rs]), np.array([r["y"] for r in rs]))


def fit(rs):
    X, y = XY(rs)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(max_iter=5000, C=1.0).fit((X - mu) / sd, y)
    return lambda Z: m.decision_function((Z - mu) / sd)


def auc_ci(rs, score, seed=0):
    y = np.array([r["y"] for r in rs])
    by = defaultdict(list)
    for i, r in enumerate(rs):
        by[r["video"]].append(i)
    vids, rng, xs = sorted(by), random.Random(seed), []
    for _ in range(args.boot):
        idx = np.array([i for v in (rng.choice(vids) for _ in vids) for i in by[v]])
        if 0 < y[idx].sum() < len(idx):
            xs.append(roc_auc_score(y[idx], score[idx]))
    xs.sort()
    return [float(roc_auc_score(y, score)), xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


def within_ci(rs, score, seed=0):
    """Row-weighted mean AUC inside held-out videos, with a bootstrap over the videos that have both classes."""
    w = within_group_auc([r["y"] for r in rs], score, [r["video"] for r in rs])
    vids, rng, xs = sorted(w["per_group"]), random.Random(seed), []
    for _ in range(args.boot if vids else 0):
        pick = [rng.choice(vids) for _ in vids]
        xs.append(sum(w["per_group"][v] * w["group_rows"][v] for v in pick) / sum(w["group_rows"][v] for v in pick))
    xs.sort()
    ci = [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]] if xs else [float("nan")] * 2
    return {"auc": [w["auc"], *ci], "videos": w["groups"], "rows": w["rows"], "skipped_single_class": w["skipped"]}


def class_only(tr, te):
    top = [l for l, _ in Counter(r["label"] for r in tr).most_common(30)]
    oh = lambda rs: np.array([[float(r["label"] == l) for l in top] for r in rs])
    m = LogisticRegression(max_iter=2000).fit(oh(tr), [r["y"] for r in tr])
    return m.decision_function(oh(te))


res, models = {}, {}
for ds, (a, b) in PARTS.items():
    tr, te = sel(ds, a), sel(ds, b)
    f = models[ds] = fit(tr)
    Xe, ye = XY(te)
    s = f(Xe)
    person = [i for i, r in enumerate(te) if r["label"] == "person"]
    rng = np.random.default_rng(0)
    Xt, yt = XY(tr)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Zt, Ze = (Xt - mu) / sd, (Xe - mu) / sd
    null = [float(roc_auc_score(ye, LogisticRegression(max_iter=5000).fit(Zt, rng.permutation(yt)).decision_function(Ze)))
            for _ in range(args.perm)]
    res[ds] = {"train": {"split": a, "videos": len({r["video"] for r in tr}), "rows": len(tr), "positives": int(yt.sum())},
               "heldout": {"split": b, "videos": len({r["video"] for r in te}), "rows": len(te), "positives": int(ye.sum())},
               "auc": auc_ci(te, s), "auc_class_only": auc_ci(te, class_only(tr, te)),
               "auc_person_rows": auc_ci([te[i] for i in person], s[person]) if len({te[i]["y"] for i in person}) == 2 else None,
               "person_rows": {"rows": len(person), "positives": int(sum(te[i]["y"] for i in person))},
               "auc_within_video": within_ci(te, s),
               "auc_permuted_labels": null_summary(float(roc_auc_score(ye, s)), null),
               "positive_labels": dict(Counter(r["label"] for r in te if r["y"]).most_common(8))}
for ds, other in (("egoblind", "viabench"), ("viabench", "egoblind")):
    te = sel(ds, PARTS[ds][1])
    res[ds]["auc_transfer_from_other"] = auc_ci(te, models[other](XY(te)[0]))

tr, te = sel("viabench", "via_train"), sel("viabench", "via_dev")
by = defaultdict(list)
for r in tr:
    by[r["video"]].append(r)
vids, curve = sorted(by), {}
Xe, ye = XY(te)
for n in [int(x) for x in args.curve.split(",") if int(x) < len(vids)]:
    aucs = []
    for sd_ in range(args.curve_seeds):
        pick = random.Random(sd_).sample(vids, n)
        sub = [r for v in pick for r in by[v]]
        if len({r["y"] for r in sub}) == 2:
            aucs.append(float(roc_auc_score(ye, fit(sub)(Xe))))
    curve[n] = {"mean": float(np.mean(aucs)), "min": float(np.min(aucs)), "max": float(np.max(aucs)), "runs": len(aucs)}
res["viabench"]["learning_curve_train_videos"] = curve

# perception coverage of the human reminders (cached part of via_train + via_dev)
cov = defaultdict(lambda: [0, 0, 0])
for sc in via:
    seen = set()
    for _t, dets in sc.frames():
        seen.update(d.source_id for d in dets if d.source_id is not None)
    for e in sc.speak_events():
        c = cov[e.detail["task"]]
        c[0] += 1; c[1] += bool(e.detail["labels"]); c[2] += e.obj_id in seen
res["viabench"]["reminder_coverage_by_task"] = {k: {"reminders": v[0], "names_an_object": v[1], "detector_saw_it_in_window": v[2]}
                                                for k, v in sorted(cov.items(), key=lambda kv: -kv[1][0])}
tot = [sum(v[i] for v in cov.values()) for i in range(3)]
res["viabench"]["reminder_coverage"] = {"reminders": tot[0], "names_an_object": tot[1], "detector_saw_it_in_window": tot[2]}

out = ROOT / "results" / "viabench"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / "supervision.json")
f3 = lambda a: "n/a" if a is None else f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]"
L = ["E3-V: the same candidates and the base 29 features under two kinds of label (standardised logistic; intervals: bootstrap over held-out source videos)\n",
     "| | EgoBlind, question proxies | VIABench, human proactive reminders |", "|---|---|---|"]
E, V = res["egoblind"], res["viabench"]
row = lambda name, g: L.append(f"| {name} | {g(E)} | {g(V)} |")
row("fit on", lambda r: f"`{r['train']['split']}`: {r['train']['videos']} videos, {r['train']['rows']} rows, {r['train']['positives']} positive")
row("scored on", lambda r: f"`{r['heldout']['split']}`: {r['heldout']['videos']} videos, {r['heldout']['rows']} rows, {r['heldout']['positives']} positive")
row("AUC, base 29 features", lambda r: f"**{f3(r['auc'])}**")
row("AUC, object class only", lambda r: f3(r["auc_class_only"]))
row("AUC inside `person` rows", lambda r: f"{f3(r['auc_person_rows'])} ({r['person_rows']['positives']} positive of {r['person_rows']['rows']})")
row("AUC with the other dataset's model", lambda r: f3(r["auc_transfer_from_other"]))
row("AUC inside one video (row-weighted mean)", lambda r: f"{f3(r['auc_within_video']['auc'])} ({r['auc_within_video']['videos']} videos with both classes, "
                                                          f"{r['auc_within_video']['skipped_single_class']} skipped)")
row(f"AUC after permuting the training labels ({args.perm} draws)",
    lambda r: f"{r['auc_permuted_labels']['mean']:.3f} [{r['auc_permuted_labels']['lo']:.3f}, {r['auc_permuted_labels']['hi']:.3f}], p = {r['auc_permuted_labels']['p']:.3f}")
L += ["", "VIABench learning curve (AUC on `via_dev`, mean over seeded subsets of training videos): " +
      "; ".join(f"{n} videos {c['mean']:.3f} ({c['min']:.3f} to {c['max']:.3f})" for n, c in curve.items()) +
      f"; all {V['train']['videos']} videos {V['auc'][0]:.3f}.", "",
      f"Perception coverage of the human reminders (cached first part of {len(via)} videos): {tot[0]} reminders, {tot[1]} name an object "
      f"the label table knows, {tot[2]} have that object detected inside the reminder's window ({tot[2] / max(1, tot[0]):.0%} of all).", "",
      "| VIABench task | reminders | name an object | detector saw it in the window |", "|---|---|---|---|"]
for k, v in list(res["viabench"]["reminder_coverage_by_task"].items())[:14]:
    L.append(f"| {k} | {v['reminders']} | {v['names_an_object']} | {v['detector_saw_it_in_window']} |")
(out / "supervision.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
