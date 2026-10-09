"""Tune the P-CACS weights and thresholds on the CampusBench *train* split by random search.

Objective: mean communication score (timely severity-weighted warnings minus off-window cues,
late penalties and audio occupancy), i.e. minimum regret against the oracle. Writes
configs/pcacs_tuned.yaml and results/campus/pcacs_tuning.json.

    --domain egoblind   tune on cached EgoBlind `train` replays (the 70% part of the source-recording split; weak
                        question-derived events); writes configs/pcacs_egoblind.yaml and results/egoblind/pcacs_tuning.json.
                        The pre-audit config is retired to results/egoblind/exploratory_2026-10-08/.
    --domain both       objective = mean of the two domains' scores.
"""
import argparse, json, random, time
from dataclasses import asdict, replace
import yaml
from _common import CONFIGS, dump_json
from campus_common import RESULTS_C, load_campus_split, oracle_scores, text_model
from seeforme.campus.runner import run_on_campus
from seeforme.campus.metrics import evaluate_run, aggregate
from seeforme.config import PipelineConfig
from seeforme.cues.pcacs import PCACSConfig, PCACSPolicy, PCACSWeights

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--subset", type=int, default=0, help="use only the first N train scenarios (0 = all)")
ap.add_argument("--domain", default="campus", choices=["campus", "egoblind", "both"])
args = ap.parse_args()

rng = random.Random(args.seed)
cfg = PipelineConfig()
tm = text_model()
domains = {}
if args.domain in ("campus", "both"):
    train = load_campus_split("train")
    domains["campus"] = (train[: args.subset] if args.subset else train, run_on_campus)
if args.domain in ("egoblind", "both"):
    from egoblind_common import RESULTS_E, load_egoblind_split
    from seeforme.egoblind import run_on_egoblind
    ego = load_egoblind_split("train", require_cache=True, limit=args.subset)
    if not ego:
        raise SystemExit("no cached EgoBlind train videos; run scripts/cache_egoblind_perception.py first")
    domains["egoblind"] = (ego, run_on_egoblind)
osc = {}
for name, (scs, run) in domains.items():
    from seeforme.campus.oracle import CampusOracle
    from seeforme.campus.metrics import score
    osc.update({sc.name: score(run(CampusOracle(sc), sc, cfg, seed=sc.seed), sc) for sc in scs})
print({k: len(v[0]) for k, v in domains.items()}, "train scenarios")


def evaluate(pc: PCACSConfig):
    per = []
    for name, (scs, run) in domains.items():
        rows = []
        for sc in scs:
            pol = PCACSPolicy(pc, text_model=tm, goal_tags=sc.goal_tags)
            rows.append(evaluate_run(run(pol, sc, cfg, seed=sc.seed), sc))
        per.append(aggregate(rows, osc)["pcacs"])
    a = dict(per[0])
    for k in a:
        if isinstance(a[k], (int, float)):
            a[k] = sum(p[k] for p in per) / len(per)
    return a


def sample(base: PCACSConfig) -> PCACSConfig:
    w = PCACSWeights(
        w_hazard=rng.uniform(1.0, 3.5), w_ttc=rng.uniform(0.0, 2.5), w_action=rng.uniform(0.2, 1.5),
        w_goal=rng.uniform(1.5, 3.5), w_novelty=rng.uniform(0.3, 1.2), w_uncert=rng.uniform(0.3, 1.5),
        w_text=rng.uniform(0.4, 1.8), lam_speech=rng.uniform(0.0, 0.8), lam_queue=rng.uniform(0.0, 1.5))
    return replace(base, weights=w, threshold_speak=rng.uniform(0.6, 1.8), threshold_interrupt=rng.uniform(1.6, 3.5),
                   interrupt_margin=rng.uniform(0.2, 1.0), budget_refill=rng.uniform(0.2, 0.6),
                   urgent_bypass=rng.uniform(0.25, 0.7), ttc_horizon=rng.uniform(5.0, 10.0),
                   recent_block_s=rng.uniform(1.5, 4.0))


t0 = time.time()
base = PCACSConfig()
best_cfg, best, history = base, evaluate(base), []
history.append({"trial": 0, **{k: v for k, v in best.items()}, "config": asdict(base)})
print(f"trial 0 (defaults): score={best['score']:.3f} regret={best['regret']:.3f} AWrecall={best['action_weighted_recall']:.3f} "
      f"prec={best['cue_precision']:.3f} occ={best['audio_occupancy']:.2f}")
for i in range(1, args.trials + 1):
    pc = sample(base)
    a = evaluate(pc)
    history.append({"trial": i, **{k: v for k, v in a.items()}, "config": asdict(pc)})
    if a["score"] > best["score"]:
        best, best_cfg = a, pc
        print(f"trial {i}: score={a['score']:.3f} regret={a['regret']:.3f} AWrecall={a['action_weighted_recall']:.3f} "
              f"prec={a['cue_precision']:.3f} occ={a['audio_occupancy']:.2f} lateC={a['late_critical_rate']:.2f} ({time.time()-t0:.0f}s)")
CONFIGS.mkdir(exist_ok=True)
out_cfg = CONFIGS / ("pcacs_tuned.yaml" if args.domain == "campus" else f"pcacs_{args.domain}.yaml")
out_res = (RESULTS_C if args.domain == "campus" else RESULTS_E) / "pcacs_tuning.json"
out_cfg.write_text(yaml.safe_dump(asdict(best_cfg), sort_keys=False))
dump_json({"domain": args.domain, "best": {k: v for k, v in best.items()}, "best_config": asdict(best_cfg), "history": history}, out_res)
print(f"best score {best['score']:.3f} (regret {best['regret']:.3f}) -> {out_cfg}")
