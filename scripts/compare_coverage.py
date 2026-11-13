"""Coverage-first diagnostic against the production threshold rule on the EgoBlind dev stream.

    python scripts/compare_coverage.py

Question: if no learned score suppresses anything, how many should-speak events can the current candidates, the audio
channel and the speech budget deliver, and what does that cost in unnecessary speech? The coverage-first policy
(seeforme/scheduling/coverage_policy.py) announces every confirmed candidate once. It is a measuring instrument, not
a product policy.

Policies, all on identical candidates: `threshold` (frozen logistic, adaptive tau: the production rule); coverage-first
with the threshold's own measured speech budget (x1), with x2 and x4, and with no budget. The budget is the threshold's
spoken seconds per minute on `train`. Reported on `finetune` (out of sample for the logistic) and on train + finetune
(the coverage policy has no fitted parameter; the threshold is in-sample on the train part). Differences on `finetune`
carry a bootstrap interval over source recordings. `select`, `holdout` and `test` are never loaded.
Writes results/coverage/dev.{json,md}.
"""
import argparse, random
from collections import Counter, defaultdict

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, source_recordings
from seeforme.campus.metrics import evaluate_run
from seeforme.egoblind import egoblind_extras
from seeforme.eval.funnel import FunnelRecorder, classify
from seeforme.online.decision_models import DecisionModel, FrozenScorerPolicy
from seeforme.online.stream import run_stream
from seeforme.scheduling.coverage_policy import CoverageFirstPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="logistic")
ap.add_argument("--capacity", type=float, default=6.0)
ap.add_argument("--confirm", type=int, default=2)
ap.add_argument("--budgets", default="1,2,4,none", help="multiples of the threshold's speech rate; none = no budget")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--repeat-window", type=float, default=20.0)
args = ap.parse_args()

cfg = load_campus_cfg()
src = source_recordings()
stream = []
for s in ("train", "finetune"):
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    stream += part
det_times = {}
for sc in stream:
    d = defaultdict(list)
    for t, dets in sc.frames():
        for det in dets:
            if det.source_id is not None:
                d[det.source_id].append(t)
    det_times[sc.video_id] = d
g = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else 0.0
inw = lambda tr, t: tr.window[0] - 1e-6 <= t <= tr.window[1] + 1e-6


def run(pol):
    recs, rows = {}, {}
    for sc, log in run_stream(pol, stream, cfg, p_tap=0.5, seed=0,
                              decision_hook=lambda sc, _p=pol: recs.setdefault(sc.video_id, FunnelRecorder(_p))):
        r = evaluate_run(log, sc).as_dict(); r.update(egoblind_extras(log, sc))
        ne, nq = g(r, "n_events"), g(r, "n_quiet")
        spoken = sorted(log.spoken, key=lambda c: c.t)
        seen_src, last_label, rep_src, rep_lab = set(), {}, 0, 0
        for c in spoken:
            if c.source_id is not None:
                rep_src += c.source_id in seen_src; seen_src.add(c.source_id)
            rep_lab += c.label in last_label and c.t - last_label[c.label] <= args.repeat_window
            last_label[c.label] = c.t
        traces = [recs[sc.video_id].trace(e, det_times[sc.video_id].get(e.obj_id, [])) for e in sc.speak_events()] if sc.video_id in recs else []
        st = [classify(t) for t in traces]
        why = Counter()
        for tr, x in zip(traces, st):
            if x in ("scheduler", "timing"):
                rs = Counter(rr for tt, rr in tr.reasons if inw(tr, tt))
                why[rs.most_common(1)[0][0] if rs else "not_a_candidate_in_window"] += 1
        rows[sc.video_id] = {"split": sc.split, "ev": ne, "hit": round(g(r, "important_cue_recall") * ne), "q": nq,
                             "qb": round(g(r, "quiet_violation_rate") * nq), "sp": len(spoken), "min": sc.duration / 60,
                             "sps": g(r, "audio_occupancy") * sc.duration, "rep_src": rep_src, "src_cues": sum(c.source_id is not None for c in spoken),
                             "rep_lab": rep_lab, "covered": sum(x in ("scheduler", "timing", "delivered") for x in st),
                             "delivered": st.count("delivered"), "why": why}
    return rows


def pooled(rows, vids):
    t, why = defaultdict(float), Counter()
    for v in vids:
        for k, x in rows[v].items():
            if k == "why":
                why.update(x)
            elif k != "split":
                t[k] += x
    d = lambda a, b: a / b if b else float("nan")
    return {"events": int(t["ev"]), "hits": int(t["hit"]), "recall": d(t["hit"], t["ev"]), "covered": int(t["covered"]),
            "delivered": int(t["delivered"]), "covered_recall": d(t["delivered"], t["covered"]),
            "quiet_violation": d(t["qb"], t["q"]), "cues_per_min": d(t["sp"], t["min"]), "speech_s_per_min": d(t["sps"], t["min"]),
            "repeat_same_object": d(t["rep_src"], t["src_cues"]), "repeat_same_label": d(t["rep_lab"], t["sp"]),
            "missed_covered_by_reason": dict(why.most_common())}


vid_split = {sc.video_id: sc.split for sc in stream}
tr_v = [v for v in vid_split if vid_split[v] == "train"]; ft_v = [v for v in vid_split if vid_split[v] == "finetune"]
model = DecisionModel.load(ROOT / "models" / "decision_models" / f"{args.model}.pt")
runs = {"threshold": run(FrozenScorerPolicy(model, target_per_min=2.0, tau0=0.6, seed=0, name="threshold"))}
rate = pooled(runs["threshold"], tr_v)["speech_s_per_min"]
print(f"threshold speaks {rate:.2f} s/min on train: that is budget x1", flush=True)
for b in args.budgets.split(","):
    refill = None if b == "none" else float(b) * rate / 60.0
    name = "coverage-first, no budget" if refill is None else f"coverage-first, budget x{b}"
    runs[name] = run(CoverageFirstPolicy(capacity_s=args.capacity, refill_per_s=refill, confirm_n=args.confirm, name=name))
    print(name, {k: v for k, v in pooled(runs[name], ft_v).items() if k != "missed_covered_by_reason"}, flush=True)

groups = defaultdict(list)                                         # finetune videos by source recording
for v in ft_v:
    groups[src.get(v, f"v{v}")].append(v)
gk = sorted(groups)
rng = random.Random(0)


def boot(a, b, key):
    xs = []
    for _ in range(args.boot):
        vs = [v for k in (rng.choice(gk) for _ in gk) for v in groups[k]]
        x = pooled(a, vs)[key] - pooled(b, vs)[key]
        if x == x:
            xs.append(x)
    xs.sort()
    return [pooled(a, ft_v)[key] - pooled(b, ft_v)[key], xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


res = {"budget_x1_s_per_min": rate, "finetune_recordings": len(gk), "policies": {}}
for name, rows in runs.items():
    res["policies"][name] = {"finetune": pooled(rows, ft_v), "dev": pooled(rows, tr_v + ft_v)}
    if name != "threshold":
        res["policies"][name]["recall_vs_threshold"] = boot(rows, runs["threshold"], "recall")
        res["policies"][name]["quiet_vs_threshold"] = boot(rows, runs["threshold"], "quiet_violation")
out = ROOT / "results" / "coverage"
out.mkdir(parents=True, exist_ok=True)
dump_json(res, out / "dev.json")
pc = lambda x: "n/a" if x != x else f"{100 * x:.0f}%"
L = [f"Coverage-first diagnostic on the EgoBlind dev stream (confirm after {args.confirm} decisions, ledger capacity {args.capacity:g} s; "
     f"budget x1 = {rate:.2f} spoken s/min, the threshold rule's rate on train)\n"]
for part, title in (("finetune", f"`finetune` ({len(ft_v)} videos, {len(gk)} source recordings; out of sample for the threshold's logistic)"),
                    ("dev", f"train + finetune ({len(tr_v) + len(ft_v)} videos; the threshold is in-sample on the train part)")):
    L += [f"**{title}**", "", "| policy | events on time | of events with a candidate | quiet moments spoken over | cues/min | speech s/min | repeats, same object | repeats, same label in 20 s |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in res["policies"].items():
        p = r[part]
        L.append(f"| {name} | {p['hits']}/{p['events']} = {p['recall']:.3f} | {p['delivered']}/{p['covered']} = {p['covered_recall']:.3f} | "
                 f"{pc(p['quiet_violation'])} | {p['cues_per_min']:.2f} | {p['speech_s_per_min']:.2f} | {pc(p['repeat_same_object'])} | {pc(p['repeat_same_label'])} |")
    L.append("")
L += ["**Difference from the threshold rule on `finetune`** (bootstrap over source recordings, 95%)", "",
      "| policy | events on time | quiet moments spoken over |", "|---|---|---|"]
for name, r in res["policies"].items():
    if name != "threshold":
        a, b = r["recall_vs_threshold"], r["quiet_vs_threshold"]
        L.append(f"| {name} | {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}] | {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] |")
L += ["", "**Why events with a candidate were still missed** (train + finetune; main reason logged inside the event's window)", ""]
for name, r in res["policies"].items():
    L.append(f"- {name}: " + (", ".join(f"{k} {v}" for k, v in r["dev"]["missed_covered_by_reason"].items()) or "none"))
L += ["", "`events on time` uses the benchmark's recall (as in E2); `of events with a candidate` uses the failure funnel "
      "(a cue about the event's own object inside its window, over events whose object reached the scheduler in the window)."]
(out / "dev.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
