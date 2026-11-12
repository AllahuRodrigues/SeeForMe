"""M1: online learning of when to speak from one-sided feedback, on a time-ordered EgoBlind stream.

    python scripts/run_online.py --stream train,finetune                       # progressive validation (development)
    python scripts/run_online.py --stream select --tag select_tau              # choose hyperparameters on `select` only
    python scripts/run_online.py --stream holdout --final --tag final          # the one final run (never tune after it)

Videos are ordered by source recording, then clip id (one recording = one proxy user, clips in recording order).
Learning policies (`online_<mode>`) are ONE instance for the whole stream and learn from feedback as it arrives
(seeforme.online.feedback); frozen policies are rebuilt per video and ignore feedback. Every decision is made before
the feedback it causes, so the per-video rows are an honest online evaluation.

Writes results/online/per_video_<tag>.csv (bootstrap_ci.py compatible), summary_<tag>.{json,md}, curve_<tag>.csv.
"""
import argparse, csv, sys
from collections import defaultdict

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, make_egoblind_policy, parse_noise, source_recordings
from seeforme.campus.metrics import evaluate_run
from seeforme.egoblind import egoblind_extras, run_on_egoblind
from seeforme.online.policy import MODES, OnlinePolicy
from seeforme.online.stream import run_stream

PROTECTED = ("holdout", "test")
OUT = ROOT / "results" / "online"

ap = argparse.ArgumentParser()
ap.add_argument("--stream", default="train,finetune", help="comma list of splits, concatenated in this order")
ap.add_argument("--policies", default="pcacs,egoblind_learned:il,egoblind_learned:rl,online_greedy,online_apple,online_thompson,online_full_info")
ap.add_argument("--target", type=float, default=2.0, help="target cues per minute for the online budget controller")
ap.add_argument("--tau0", type=float, default=0.6)
ap.add_argument("--eps", type=float, default=0.25, help="share of the cue budget spent exploring (apple)")
ap.add_argument("--lr", type=float, default=0.5)
ap.add_argument("--implicit-neg", type=float, default=0.0, help="weight of the weak negative for a candidate nobody asked about")
ap.add_argument("--p-tap", type=float, default=0.5, help="probability that the user taps useful / not useful after a cue")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--noise", default="")
ap.add_argument("--final", action="store_true", help="required to touch holdout / test (run once, never tune afterwards)")
ap.add_argument("--tag", default="dev")
ap.add_argument("--curve-every", type=int, default=50)
ap.add_argument("--campus-epochs", type=int, default=1, help="warm-start passes over CampusBench v2 train for online_<mode>+campus (carry all) and +campusw (weights only)")
args = ap.parse_args()

splits = [s for s in args.stream.split(",") if s]
if any(s in PROTECTED for s in splits) and not args.final:
    ap.error(f"{PROTECTED} are evaluation-only: pass --final, and only after hyperparameters are frozen on `select`")

cfg = load_campus_cfg()
noise = parse_noise(args.noise)
src = source_recordings()
scenarios = []
for s in splits:
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    scenarios += part
if args.limit:
    scenarios = scenarios[:args.limit]
if not scenarios:
    raise SystemExit("no cached scenarios in the stream")
minutes = sum(sc.duration for sc in scenarios) / 60
print(f"stream {splits}: {len(scenarios)} videos, {minutes:.1f} min, {sum(len(sc.speak_events()) for sc in scenarios)} events, "
      f"{len(set(src.get(sc.video_id) for sc in scenarios))} proxy users; p_tap={args.p_tap} target={args.target}/min", flush=True)


def row_for(name, k, sc, log):
    r = evaluate_run(log, sc).as_dict()
    r.update({k2: v for k2, v in egoblind_extras(log, sc).items()})
    r.update({"policy": name, "stream_index": k, "video_id": sc.video_id, "split": sc.split, "proxy_user": src.get(sc.video_id, "")})
    return r


rows = []
for name in [p for p in args.policies.split(",") if p]:
    print(f"-- {name}", flush=True)
    if name.startswith("online_"):
        mode, _, warm = name.split("_", 1)[1].partition("+")
        if mode not in MODES or warm not in ("", "campus", "campusw"):
            sys.exit(f"unknown online policy {name}")
        pol = OnlinePolicy(mode=mode, target_per_min=args.target, tau0=args.tau0, eps=args.eps, lr=args.lr, seed=args.seed,
                           implicit_neg_w=args.implicit_neg)
        if warm:
            from seeforme.campus.world import CampusScenario
            from seeforme.online.pretrain import pretrain_on_campus
            sim = [CampusScenario.from_json(str(f)) for f in sorted((ROOT / "data" / "campus_v2" / "train").glob("*.json"))]
            stats = pretrain_on_campus(pol, sim, cfg, epochs=args.campus_epochs, seed=args.seed,
                                       carry="weights" if warm == "campusw" else "all")
            print(f"   warm start ({warm}): {stats}", flush=True)
        prog = lambda k, sc, log: (k + 1) % 100 == 0 and print(f"   {k + 1}/{len(scenarios)} tau={pol.tau:.3f} labels={pol.n_labels}", flush=True)
        for k, (sc, log) in enumerate(run_stream(pol, scenarios, cfg, p_tap=args.p_tap, seed=args.seed, noise=noise, on_video=prog)):
            rows.append(row_for(name, k, sc, log))
        dump_json(pol.to_dict(), OUT / f"model_{name}_{args.tag}.json")
    else:
        for k, sc in enumerate(scenarios):
            try:
                p = make_egoblind_policy(name, cfg, sc)
            except FileNotFoundError as e:
                print(f"   skip {name}: {e}"); break
            rows.append(row_for(name, k, sc, run_on_egoblind(p, sc, cfg, seed=sc.seed, noise=noise)))


def pooled(rs):
    cov = sum(r["n_coverable"] or 0 for r in rs)
    hit = sum(round((r["coverable_recall"] or 0) * (r["n_coverable"] or 0)) for r in rs)
    nq = sum(r["n_quiet"] or 0 for r in rs)
    bad = sum(round((r["quiet_violation_rate"] or 0) * (r["n_quiet"] or 0)) for r in rs)
    sp = sum(r["n_spoken"] for r in rs)
    mins = sum(next(sc.duration for sc in scenarios if sc.video_id == r["video_id"]) for r in rs) / 60
    return {"recall_coverable": hit / cov if cov else float("nan"), "quiet_violation": bad / nq if nq else float("nan"),
            "cues_per_min": sp / mins if mins else float("nan"), "n_coverable": cov, "n_quiet": nq, "n_spoken": sp}


OUT.mkdir(parents=True, exist_ok=True)
fields = list(rows[0].keys())
with open(OUT / f"per_video_{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
by = defaultdict(list)
for r in rows:
    by[r["policy"]].append(r)
summary = {p: pooled(rs) for p, rs in by.items()}
summary["_meta"] = {"stream": splits, "n_videos": len(scenarios), "minutes": minutes, "p_tap": args.p_tap, "target": args.target,
                    "tau0": args.tau0, "eps": args.eps, "lr": args.lr, "implicit_neg": args.implicit_neg, "seed": args.seed, "noise": args.noise, "final": args.final}
dump_json(summary, OUT / f"summary_{args.tag}.json")
with open(OUT / f"curve_{args.tag}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["policy", "videos_seen", "recall_coverable", "quiet_violation", "cues_per_min"])
    for p, rs in by.items():
        for end in range(args.curve_every, len(rs) + 1, args.curve_every):
            s = pooled(rs[end - args.curve_every:end])
            w.writerow([p, end, round(s["recall_coverable"], 4), round(s["quiet_violation"], 4), round(s["cues_per_min"], 3)])
lines = [f"Online stream {splits}: {len(scenarios)} videos, {minutes:.1f} min, p_tap={args.p_tap}, target {args.target}/min\n",
         "| policy | coverable recall | quiet violations | cues/min | coverable events |", "|---|---|---|---|---|"]
for p, s in summary.items():
    if not p.startswith("_"):
        lines.append(f"| {p} | {s['recall_coverable']:.3f} | {s['quiet_violation']:.3f} | {s['cues_per_min']:.2f} | {s['n_coverable']} |")
(OUT / f"summary_{args.tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
