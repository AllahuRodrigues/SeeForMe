"""M2 S3: does long-term appearance memory cut repeated cues without losing coverage?

    python scripts/eval_reid.py --stream train,finetune --horizons 0,120 --theta 0.85          # development effect
    python scripts/eval_reid.py --stream select --horizons 0,30,60,120,300 --theta 0.8,0.85,0.9 --tag select   # choose
    python scripts/eval_reid.py --stream holdout --final --horizons 0,<h> --theta <t> --tag final

Each policy runs with the re-identification gallery off (horizon 0) and on. Embeddings come from
data/egoblind_app/<video>.npz (scripts/cache_appearance.py); videos without them run with re-ID inert.

Repeat measures, per spoken cue:
    repeat_gt     the cue's weak event was already announced in this video. Detections are linked to an event by
                  label inside its window, so this also counts a cue about a different person or car (issue 6)
    repeat_track  the same tracker track was already announced in this video (a true re-announcement)
    repeat_label  a cue with the same label was spoken within the previous 60 s (the "door... door..." annoyance)
Match precision: among matches where both sides carry a ground-truth id, the share where the ids agree.
Paired bootstrap over videos for on minus off. Writes results/memory/reid_<tag>.{csv,json,md} and matches_<tag>.csv.
"""
import argparse, csv, random
from collections import defaultdict
from dataclasses import replace

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, make_egoblind_policy, source_recordings
from seeforme.campus.metrics import evaluate_run
from seeforme.egoblind import egoblind_extras, run_on_egoblind
from seeforme.egoblind.metrics import repeat_counts
from seeforme.online.policy import OnlinePolicy
from seeforme.online.stream import run_stream

PROTECTED = ("holdout", "test")
OUT = ROOT / "results" / "memory"
EMB = ROOT / "data" / "egoblind_app"

ap = argparse.ArgumentParser()
ap.add_argument("--stream", default="train,finetune")
ap.add_argument("--policies", default="pcacs,egoblind_learned:il,online_greedy")
ap.add_argument("--horizons", default="0,120", help="comma list of gallery horizons in seconds (0 = re-ID off)")
ap.add_argument("--theta", default="0.85", help="comma list of cosine thresholds")
ap.add_argument("--label-window", type=float, default=60.0)
ap.add_argument("--p-tap", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--final", action="store_true")
ap.add_argument("--tag", default="dev")
args = ap.parse_args()

splits = [s for s in args.stream.split(",") if s]
if any(s in PROTECTED for s in splits) and not args.final:
    ap.error(f"{PROTECTED} are evaluation-only: pass --final, and only after theta and the horizon are chosen on `select`")

base = load_campus_cfg()
src = source_recordings()
scenarios = []
for s in splits:
    part = load_egoblind_split(s, require_cache=True)
    part.sort(key=lambda sc: (src.get(sc.video_id, f"zz{sc.video_id:06d}"), sc.video_id))
    scenarios += part
if args.limit:
    scenarios = scenarios[:args.limit]
with_emb = sum(sc.cache().load_embeddings(EMB / f"{sc.video_id}.npz") for sc in scenarios)
print(f"stream {splits}: {len(scenarios)} videos, {with_emb} with embeddings", flush=True)

configs = [(0.0, None)] + [(float(h), float(th)) for h in args.horizons.split(",") if float(h) > 0 for th in args.theta.split(",")]


def variant(h, th):
    return "reid_off" if h == 0 else f"reid_h{h:g}_t{th:g}"


def repeats(log, sc):
    return repeat_counts(log.spoken, label_window=args.label_window)


rows, matches = [], []
for name in [p for p in args.policies.split(",") if p]:
    for h, th in configs:
        cfg = replace(base, tracker=replace(base.tracker, reid_horizon_s=h, reid_threshold=th or base.tracker.reid_threshold))
        tag = variant(h, th)
        print(f"-- {name} {tag}", flush=True)
        if name.startswith("online_"):
            pol = OnlinePolicy(mode=name.split("_", 1)[1], seed=args.seed)
            pairs = run_stream(pol, scenarios, cfg, p_tap=args.p_tap, seed=args.seed)
        else:
            pairs = []
            for sc in scenarios:
                try:
                    p = make_egoblind_policy(name, cfg, sc)
                except FileNotFoundError as e:
                    print(f"   skip {name}: {e}"); break
                pairs.append((sc, run_on_egoblind(p, sc, cfg, seed=sc.seed)))
        for sc, log in pairs:
            r = evaluate_run(log, sc).as_dict()
            r.update(egoblind_extras(log, sc))
            rc = repeats(log, sc)
            r.update({"policy": f"{name}@{tag}", "base_policy": name, "variant": tag, "video_id": sc.video_id, "split": sc.split,
                      "minutes": sc.duration / 60, "repeat_gt": rc["repeat_gt"], "repeat_track": rc["repeat_track"], "repeat_label": rc["repeat_label"], "n_reid": len(log.reid_matches),
                      "n_reid_announced": sum(m["announced"] for m in log.reid_matches)})
            rows.append(r)
            matches += [{"policy": name, "variant": tag, "video_id": sc.video_id, **m} for m in log.reid_matches]


def pooled(rs):
    cov = sum(r["n_coverable"] or 0 for r in rs)
    hit = sum(round((r["coverable_recall"] or 0) * (r["n_coverable"] or 0)) for r in rs)
    nq = sum(r["n_quiet"] or 0 for r in rs)
    bad = sum(round((r["quiet_violation_rate"] or 0) * (r["n_quiet"] or 0)) for r in rs)
    sp = sum(r["n_spoken"] for r in rs); mins = sum(r["minutes"] for r in rs)
    return {"recall_coverable": hit / cov if cov else float("nan"), "quiet_violation": bad / nq if nq else float("nan"),
            "cues_per_min": sp / mins if mins else float("nan"), "repeat_gt_share": sum(r["repeat_gt"] for r in rs) / sp if sp else float("nan"),
            "repeat_track_share": sum(r["repeat_track"] for r in rs) / sp if sp else float("nan"),
            "repeat_label_per_min": sum(r["repeat_label"] for r in rs) / mins if mins else float("nan"),
            "reid_per_min": sum(r["n_reid"] for r in rs) / mins if mins else float("nan"),
            "reid_announced": sum(r["n_reid_announced"] for r in rs), "n_coverable": cov, "n_spoken": sp}


def boot_diff(on, off, key, n, seed):
    """Paired bootstrap over videos of pooled(on)[key] - pooled(off)[key]."""
    vids = sorted(on)
    rng = random.Random(seed)
    point = pooled([on[v] for v in vids])[key] - pooled([off[v] for v in vids])[key]
    ds = []
    for _ in range(n):
        s = [rng.choice(vids) for _ in vids]
        ds.append(pooled([on[v] for v in s])[key] - pooled([off[v] for v in s])[key])
    ds = sorted(d for d in ds if d == d)
    return {"diff": point, "lo": ds[int(0.025 * len(ds))], "hi": ds[int(0.975 * len(ds)) - 1]}


by = defaultdict(list)
for r in rows:
    by[r["policy"]].append(r)
summary = {p: pooled(rs) for p, rs in by.items()}
diffs = {}
for name in {r["base_policy"] for r in rows}:
    off = {r["video_id"]: r for r in by.get(f"{name}@reid_off", [])}
    for h, th in configs[1:]:
        on = {r["video_id"]: r for r in by.get(f"{name}@{variant(h, th)}", [])}
        if on and off:
            diffs[f"{name}@{variant(h, th)}"] = {k: boot_diff(on, off, k, args.boot, args.seed)
                                                 for k in ("recall_coverable", "repeat_label_per_min", "cues_per_min", "quiet_violation")}
prec = defaultdict(lambda: [0, 0])
for m in matches:
    if m["old_sid"] is not None and m["new_sid"] is not None:
        k = f"{m['policy']}@{m['variant']}"
        prec[k][0] += m["old_sid"] == m["new_sid"]; prec[k][1] += 1
for k, (ok, n) in prec.items():
    summary[k]["match_precision_gt"] = ok / n if n else float("nan"); summary[k]["n_matches_gt"] = n
summary["_meta"] = {"stream": splits, "n_videos": len(scenarios), "with_embeddings": with_emb, "configs": configs,
                    "label_window": args.label_window, "final": args.final, "seed": args.seed}

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / f"reid_{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(rows)
if matches:
    with open(OUT / f"matches_{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(matches[0].keys())); w.writeheader(); w.writerows(matches)
dump_json({"summary": summary, "paired_diffs": diffs}, OUT / f"reid_{args.tag}.json")
lines = [f"Re-ID memory on {splits}: {len(scenarios)} videos ({with_emb} with embeddings)\n",
         "| policy | coverable recall | quiet viol. | cues/min | repeat (gt share) | repeat (same track share) | label repeats/min | re-IDs/min | re-IDs of announced objects | match precision (gt) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for p, s in summary.items():
    if not p.startswith("_"):
        mp = s.get("match_precision_gt")
        mp_cell = "" if mp is None else f"{mp:.2f} (n={s['n_matches_gt']})"
        lines.append(f"| {p} | {s['recall_coverable']:.3f} | {s['quiet_violation']:.3f} | {s['cues_per_min']:.2f} | {s['repeat_gt_share']:.3f} | {s['repeat_track_share']:.3f} | "
                     f"{s['repeat_label_per_min']:.3f} | {s['reid_per_min']:.2f} | {s['reid_announced']} | {mp_cell} |")
lines += ["", "Paired differences, on minus off (95% bootstrap over videos):", "", "| variant | Δ recall | Δ label repeats/min | Δ cues/min | Δ quiet viol. |", "|---|---|---|---|---|"]
for k, d in diffs.items():
    cell = lambda x: f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
    lines.append(f"| {k} | {cell(d['recall_coverable'])} | {cell(d['repeat_label_per_min'])} | {cell(d['cues_per_min'])} | {cell(d['quiet_violation'])} |")
(OUT / f"reid_{args.tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
