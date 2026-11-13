"""Event-failure funnel on the EgoBlind dev stream: where does each should-speak event get lost?

    python scripts/failure_funnel.py --policies pcacs,egoblind_learned:il,online_greedy --tag dev

For every should-speak event (train + finetune) the replay records when the event was detected, tracked, offered to
the scheduler as a candidate, legal to speak about, and spoken about (seeforme.eval.funnel), and attributes the event
to the first stage that lost it. Frozen policies run per video; online policies learn over the stream with simulated
one-sided feedback (p_tap 0.5), as in scripts/run_online.py. Development splits only.

Writes results/funnel/<tag>.{json,md} (funnel counts per policy, losses by stage and by question type / concept) and
results/funnel/events_<tag>.csv (one row per event and policy, for the replay viewer).
"""
import argparse, csv
from collections import Counter, defaultdict

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, make_egoblind_policy, source_recordings
from seeforme.egoblind import run_on_egoblind
from seeforme.eval.funnel import STAGES, EventTrace, classify, funnel_counts
from seeforme.eval.timing import TimedEvent, timing_metrics
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream

ap = argparse.ArgumentParser()
ap.add_argument("--stream", default="train,finetune")
ap.add_argument("--policies", default="pcacs,egoblind_learned:il,online_greedy")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default="dev")
args = ap.parse_args()
if any(s in ("holdout", "test") for s in args.stream.split(",")):
    ap.error("development splits only")

cfg = load_campus_cfg()
src = source_recordings()
scen = []
for s in args.stream.split(","):
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    scen += part
if args.limit:
    scen = scen[:args.limit]

# detection-level evidence does not depend on the policy: one pass over the cached frames
det_times = {}
for sc in scen:
    d = defaultdict(list)
    for t, dets in sc.frames():
        for det in dets:
            if det.source_id is not None:
                d[det.source_id].append(t)
    det_times[sc.video_id] = d
print(f"{len(scen)} videos, {sum(len(sc.speak_events()) for sc in scen)} should-speak events", flush=True)


def make_hook(rec, policy):
    block = getattr(getattr(policy, "cfg", None), "recent_block_s", 2.8)

    def hook(t, tracks, obs, cue, ch, due):
        for trk in tracks:
            if trk.source_id is not None:
                rec["track"][trk.source_id].append(t)
        for o in obs or []:
            sid = o.track.source_id
            if sid is None:
                continue
            rec["cand"][sid].append(t)
            la = o.track.last_announced
            if not ch.busy and (la is None or t - la >= block):
                rec["cand_legal"][sid].append(t)
        if cue is not None and cue.action != "haptic" and cue.source_id is not None:
            rec["cue"][cue.source_id].append(t)
    return hook


def new_rec():
    return {k: defaultdict(list) for k in ("track", "cand", "cand_legal", "cue")}


traces = defaultdict(list)
for name in [p for p in args.policies.split(",") if p]:
    print(f"-- {name}", flush=True)
    recs = {}
    if name.startswith("online_"):
        pol = OnlinePolicy(mode=name.split("_", 1)[1], seed=args.seed)
        # run_stream owns the feedback hook; wrap it to also record the funnel evidence
        import seeforme.online.stream as stream_mod
        orig = stream_mod.run_on_egoblind

        def wrapped(policy, sc, cfg_, seed=None, on_decision=None, noise=None, **kw):
            rec = recs.setdefault(sc.video_id, new_rec())
            mine = make_hook(rec, policy)

            def both(*a):
                mine(*a)
                if on_decision is not None:
                    on_decision(*a)
            return orig(policy, sc, cfg_, seed=seed, on_decision=both, noise=noise, **kw)
        stream_mod.run_on_egoblind = wrapped
        try:
            run_stream(pol, scen, cfg, p_tap=0.5, seed=args.seed)
        finally:
            stream_mod.run_on_egoblind = orig
    else:
        for sc in scen:
            try:
                pol = make_egoblind_policy(name, cfg, sc)
            except FileNotFoundError as e:
                print(f"   skip {name}: {e}"); break
            rec = recs.setdefault(sc.video_id, new_rec())
            run_on_egoblind(pol, sc, cfg, seed=sc.seed, on_decision=make_hook(rec, pol))
    for sc in scen:
        rec = recs.get(sc.video_id)
        if rec is None:
            continue
        for e in sc.speak_events():
            det = e.detail or {}
            traces[name].append(EventTrace(window=(e.earliest_useful_s, e.latest_safe_s), det=det_times[sc.video_id].get(e.obj_id, []),
                                           track=rec["track"].get(e.obj_id, []), cand=rec["cand"].get(e.obj_id, []),
                                           cand_legal=rec["cand_legal"].get(e.obj_id, []), cue=rec["cue"].get(e.obj_id, []),
                                           meta={"video": sc.video_id, "obj": e.obj_id, "split": sc.split, "label": e.label,
                                                 "ideal": e.ideal_s, "t_q": float(det.get("t_q", e.latest_safe_s - 0.5)),
                                                 "qtype": det.get("qtype", ""), "question": det.get("question", ""),
                                                 "window": f"{e.earliest_useful_s:.2f}-{e.latest_safe_s:.2f}"}))

out = ROOT / "results" / "funnel"
out.mkdir(parents=True, exist_ok=True)
res, rows = {}, []
for name, trs in traces.items():
    st = [classify(t) for t in trs]
    res[name] = {"funnel": funnel_counts(trs),
                 "lost_by_label": {s: dict(Counter(t.meta["label"] for t, x in zip(trs, st) if x == s).most_common(8)) for s in STAGES},
                 "lost_by_qtype": {s: dict(Counter(t.meta["qtype"] for t, x in zip(trs, st) if x == s)) for s in STAGES}}
    for t, x in zip(trs, st):
        first = lambda ts: f"{min(ts):.2f}" if ts else ""
        rows.append({"policy": name, **t.meta, "stage": x, "blocked": t.blocked, "first_det": first(t.det), "first_track": first(t.track),
                     "first_cand": first(t.cand), "first_cue": first(t.cue), "n_cues": len(t.cue)})
# D1 onset timing (docs/research/benchmarks.md): native anchor t_q with the EgoBlind window, and EgoVoice-style
# matching around the anticipation reference ideal_s
for name, trs in traces.items():
    cues = [(t, (tr.meta["video"], tr.meta["obj"])) for tr in trs for t in tr.cue]
    ev_q = [TimedEvent((tr.meta["video"], tr.meta["obj"]), (tr.meta["video"], tr.meta["obj"]), tr.meta["t_q"]) for tr in trs]
    ev_i = [TimedEvent((tr.meta["video"], tr.meta["obj"]), (tr.meta["video"], tr.meta["obj"]), tr.meta["ideal"]) for tr in trs]
    res[name]["timing_native"] = timing_metrics(ev_q, cues, early=10.0, late=0.5, taus=(0.5, 1.0, 2.0))
    res[name]["timing_egovoice"] = timing_metrics(ev_i, cues, early=2.5, late=1.5, taus=(0.5, 1.0, 2.0))
dump_json(res, out / f"{args.tag}.json")
with open(out / f"events_{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
lines = [f"Event-failure funnel on {args.stream} ({len(scen)} videos), seed {args.seed}\n",
         "| policy | N events | P detected | T tracked | C candidate | S spoken | D on time | lost: perception / tracking / candidate / scheduler (blocked) / timing |",
         "|---|---|---|---|---|---|---|---|"]
for name, r in res.items():
    f = r["funnel"]
    lines.append(f"| {name} | {f['N']} | {f['P']} | {f['T']} | {f['C']} | {f['S']} | {f['D']} | {f['lost_perception']} / {f['lost_tracking']} / "
                 f"{f['lost_candidate']} / {f['lost_scheduler']} ({f['blocked']}) / {f['lost_timing']} |")
lines += ["", "Onset timing (D1). native: cue within [t_q - 10 s, t_q + 0.5 s]; EgoVoice-style: within [-2.5 s, +1.5 s] of ideal_s.", "",
          "| policy | variant | recall | Start@0.5 / 1 / 2 (of matched) | Start@1 of all events | mean \\|Δ\\| (s) | mean signed Δ (s) |", "|---|---|---|---|---|---|---|"]
fmt = lambda x: "" if x is None else f"{x:.3f}"
for name, r in res.items():
    for v in ("timing_native", "timing_egovoice"):
        m = r[v]
        lines.append(f"| {name} | {v.split('_')[1]} | {fmt(m['recall'])} | {fmt(m['start@0.5'])} / {fmt(m['start@1'])} / {fmt(m['start@2'])} | "
                     f"{fmt(m['start@1_all'])} | {fmt(m['mean_abs_delta'])} | {fmt(m['mean_signed_delta'])} |")
for name, r in res.items():
    lines += ["", f"**{name}**, most common labels lost at each stage:"]
    for s in STAGES[:-1]:
        if r["lost_by_label"][s]:
            lines.append(f"- {s}: " + ", ".join(f"{k} {v}" for k, v in r["lost_by_label"][s].items()))
(out / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
