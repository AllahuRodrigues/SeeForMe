"""E3-V at policy level: does a scorer trained on human reminders choose better than first-in first-out at equal speech?

    python scripts/viabench_policy.py

Stream: `via_train` then `via_dev` (held-out sources; `via_test` stays sealed). Everything is reported on `via_dev`.
Policies, on identical candidates:
  threshold + VIABench scorer   logistic on the base 29 features, fitted on `via_train` with the human reminder labels
  threshold + EgoBlind scorer   models/decision_models/logistic.pt (question proxies; the current production scorer)
  coverage-first                every confirmed candidate once, first-in first-out, under a ledger refilled at the
                                VIABench-scorer rule's own spoken seconds per minute on `via_train` (equal speech)
Controls, added after the first result to test what the gain is made of (same threshold rule, same rate controller):
  object class only             score = share of should-speak rows for the candidate's object class on `via_train`
  shuffled labels               the same logistic fitted on permuted `via_train` labels (--placebo draws)
Measures: reminders answered, i.e. a cue about an object the reminder names inside [start - 1 s, end + 0.5 s], over
all reminders and over reminders whose object the detector saw in that interval; the benchmark's own stricter rule
(inside [start, end]); cues with no reminder of any kind within 3 s, per minute (VIABench itself scores recall only);
cues and spoken seconds per minute. Intervals: bootstrap over `via_dev` videos.
Writes results/viabench/policy.{json,md}.
"""
import argparse, random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from viabench_common import load_viabench_split
from seeforme.online.decision_models import DecisionModel, FrozenScorerPolicy, Standardizer
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream
from seeforme.scheduling.coverage_policy import CoverageFirstPolicy
from seeforme.viabench.data import decision_label

ap = argparse.ArgumentParser()
ap.add_argument("--lead", type=float, default=1.0); ap.add_argument("--lag", type=float, default=0.5)
ap.add_argument("--guard", type=float, default=3.0)
ap.add_argument("--min-conf", type=float, default=0.35)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--placebo", type=int, default=3)
args = ap.parse_args()

cfg = load_campus_cfg()
train, dev = load_viabench_split("via_train"), load_viabench_split("via_dev")
stream = train + dev
rows = []


class Dump(OnlinePolicy):
    def _learn_full_info(self, obs, X, t):
        for o in obs:
            y = decision_label(t, o.track.label, self._scenario.reminders)
            if y is not None:
                rows.append((np.asarray(o.event.as_vector(), np.float32), int(y), o.track.label))


for _ in run_stream(Dump(mode="full_info", lr=0.0, seed=0), train, cfg, p_tap=0.5, seed=0):
    pass
X, y = np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])
std = Standardizer.fit(X)
Z = std(X[:, None, :], np.ones((len(X), 1), np.float32))[:, 0, :]


def fitted(yy, labels):
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(Z, yy)
    return DecisionModel("logistic", 1, std, coef=lr.coef_, intercept=float(lr.intercept_[0]),
                         meta={"labels": labels, "train_rows": int(len(yy)), "train_positives": int(yy.sum())})


class ClassPriorPolicy(FrozenScorerPolicy):
    """Control: the score is the training share of should-speak rows for the candidate's object class and nothing else."""
    prior, default = {}, 0.0

    def _scored(self, tracks, t, ch):
        obs, p, legal = super()._scored(tracks, t, ch)
        return obs, np.array([self.prior.get(o.track.label, self.default) for o in obs]), legal


via_model = fitted(y, "viabench_reminder")
base_rate, by_label = float(y.mean()), defaultdict(lambda: [0, 0])
for _, yi, lab in rows:
    by_label[lab][0] += yi; by_label[lab][1] += 1
ClassPriorPolicy.default = base_rate
ClassPriorPolicy.prior = {lab: (pos + 5 * base_rate) / (n + 5) for lab, (pos, n) in by_label.items()}   # shrunk to the base rate
ego_model = DecisionModel.load(ROOT / "models" / "decision_models" / "logistic.pt")
print(f"VIABench scorer fitted on {len(y)} rows ({int(y.sum())} positive) from {len(train)} videos", flush=True)

# which reminders had their object in view at all (independent of any policy)
detectable = {}
for sc in stream:
    hits = defaultdict(list)
    for t, dets in sc.frames():
        for d in dets:
            if d.confidence >= args.min_conf:
                hits[d.label].append(t)
    detectable[sc.video_id] = {r.qid for r in sc.reminders
                               if any(r.start - args.lead <= t <= r.end + args.lag for lab in r.labels for t in hits.get(lab, []))}


def run(pol):
    out = {}
    for sc, log in run_stream(pol, stream, cfg, p_tap=0.5, seed=0):
        cues = sorted(log.spoken, key=lambda c: c.t)
        ans = lambda r, a, b, cues=cues: any(c.label in r.labels and r.start - a <= c.t <= r.end + b for c in cues)
        rem = sc.reminders
        out[sc.video_id] = {"split": sc.split, "rem": len(rem), "det": len(detectable[sc.video_id]),
                            "hit": sum(ans(r, args.lead, args.lag) for r in rem), "hit_native": sum(ans(r, 0.0, 0.0) for r in rem),
                            "cues": len(cues), "speech": sum(c.duration_s for c in cues), "min": sc.duration / 60.0,
                            "unneeded": sum(not any(r.start - args.guard <= c.t <= r.end + args.guard for r in rem) for c in cues)}
    return out


def pooled(rs, vids):
    t = defaultdict(float)
    for v in vids:
        for k, x in rs[v].items():
            if k != "split":
                t[k] += x
    d = lambda a, b: a / b if b else float("nan")
    return {"reminders": int(t["rem"]), "detectable": int(t["det"]), "answered": int(t["hit"]), "recall": d(t["hit"], t["rem"]),
            "recall_of_detectable": d(t["hit"], t["det"]), "recall_native": d(t["hit_native"], t["rem"]),
            "unneeded_per_min": d(t["unneeded"], t["min"]), "unneeded_share": d(t["unneeded"], t["cues"]),
            "cues_per_min": d(t["cues"], t["min"]), "speech_s_per_min": d(t["speech"], t["min"])}


tr_v, dv_v = [sc.video_id for sc in train], [sc.video_id for sc in dev]
runs = {"threshold, VIABench scorer": run(FrozenScorerPolicy(via_model, target_per_min=2.0, tau0=0.6, seed=0, name="threshold_via"))}
rate = pooled(runs["threshold, VIABench scorer"], tr_v)["speech_s_per_min"]
print(f"threshold with the VIABench scorer speaks {rate:.2f} s/min on via_train: the first-in first-out budget", flush=True)
runs["threshold, EgoBlind scorer"] = run(FrozenScorerPolicy(ego_model, target_per_min=2.0, tau0=0.6, seed=0, name="threshold_ego"))
runs["coverage-first, equal speech"] = run(CoverageFirstPolicy(capacity_s=6.0, refill_per_s=rate / 60.0, name="coverage_first"))
runs["control: object class only"] = run(ClassPriorPolicy(via_model, target_per_min=2.0, tau0=0.6, seed=0, name="threshold_class_prior"))
for k in range(args.placebo):
    shuffled = fitted(np.random.default_rng(k).permutation(y), f"viabench_reminder_shuffled_{k}")
    runs[f"control: shuffled labels, draw {k + 1}"] = run(FrozenScorerPolicy(shuffled, target_per_min=2.0, tau0=0.6, seed=0, name=f"threshold_shuffled{k}"))
    print(f"shuffled draw {k + 1} done", flush=True)
rng = random.Random(0)


def boot(a, b, key):
    xs = []
    for _ in range(args.boot):
        vs = [rng.choice(dv_v) for _ in dv_v]
        x = pooled(a, vs)[key] - pooled(b, vs)[key]
        if x == x:
            xs.append(x)
    xs.sort()
    return [pooled(a, dv_v)[key] - pooled(b, dv_v)[key], xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


res = {"budget_s_per_min": rate, "videos": {"via_train": len(train), "via_dev": len(dev)}, "policies": {}}
ref = runs["coverage-first, equal speech"]
for name, rs in runs.items():
    res["policies"][name] = {"via_dev": pooled(rs, dv_v), "via_train": pooled(rs, tr_v)}
    if rs is not ref:
        res["policies"][name]["vs_fifo"] = {k: boot(rs, ref, k) for k in ("recall", "unneeded_per_min", "speech_s_per_min")}
main, cls = runs["threshold, VIABench scorer"], runs["control: object class only"]
res["viabench_scorer_vs_class_only"] = {k: boot(main, cls, k) for k in ("recall", "unneeded_per_min", "speech_s_per_min")}
res["class_prior_top"] = dict(sorted(ClassPriorPolicy.prior.items(), key=lambda kv: -kv[1])[:8])
out = ROOT / "results" / "viabench"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / "policy.json")
L = [f"E3-V at policy level, on `via_dev` ({len(dev)} held-out videos, first 100 s each; scorer fitted on {len(train)} `via_train` videos). "
     f"A reminder is answered by a cue about an object it names inside [start - {args.lead:g} s, end + {args.lag:g} s].\n",
     "| policy | reminders answered | of those the detector could see | benchmark's strict rule | cues with no reminder within 3 s, per min | cues/min | speech s/min |",
     "|---|---|---|---|---|---|---|"]
for name, r in res["policies"].items():
    p = r["via_dev"]
    L.append(f"| {name} | {p['answered']}/{p['reminders']} = {p['recall']:.3f} | {p['answered']}/{p['detectable']} = {p['recall_of_detectable']:.3f} | "
             f"{p['recall_native']:.3f} | {p['unneeded_per_min']:.2f} ({p['unneeded_share']:.0%} of cues) | {p['cues_per_min']:.2f} | {p['speech_s_per_min']:.2f} |")
L += ["", "Difference from first-in first-out at equal speech (95% interval, bootstrap over videos):", ""]
for name, r in res["policies"].items():
    if "vs_fifo" in r:
        a, b, c = r["vs_fifo"]["recall"], r["vs_fifo"]["unneeded_per_min"], r["vs_fifo"]["speech_s_per_min"]
        L.append(f"- {name}: reminders answered {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]; unneeded cues per min {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]; "
                 f"speech s/min {c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]")
a, b, c = (res["viabench_scorer_vs_class_only"][k] for k in ("recall", "unneeded_per_min", "speech_s_per_min"))
L += ["", f"VIABench scorer against the object-class-only control: reminders answered {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]; "
          f"unneeded cues per min {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]; speech s/min {c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]"]
(out / "policy.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
