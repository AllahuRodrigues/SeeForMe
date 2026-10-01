"""Tune the P-CACS weights and thresholds on the CampusBench *train* split by random search.

Objective: mean communication score (timely severity-weighted warnings minus off-window cues,
late penalties and audio occupancy), i.e. minimum regret against the oracle. Writes
configs/pcacs_tuned.yaml and results/campus/pcacs_tuning.json.
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
args = ap.parse_args()

rng = random.Random(args.seed)
cfg = PipelineConfig()
train = load_campus_split("train")
if args.subset:
    train = train[: args.subset]
osc = oracle_scores(train, cfg)
tm = text_model()


def evaluate(pc: PCACSConfig):
    rows = []
    for sc in train:
        pol = PCACSPolicy(pc, text_model=tm, goal_tags=sc.goal_tags)
        rows.append(evaluate_run(run_on_campus(pol, sc, cfg), sc))
    a = aggregate(rows, osc)["pcacs"]
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
(CONFIGS / "pcacs_tuned.yaml").write_text(yaml.safe_dump(asdict(best_cfg), sort_keys=False))
dump_json({"best": {k: v for k, v in best.items()}, "best_config": asdict(best_cfg), "history": history},
          RESULTS_C / "pcacs_tuning.json")
print(f"best score {best['score']:.3f} (regret {best['regret']:.3f}) -> configs/pcacs_tuned.yaml")
