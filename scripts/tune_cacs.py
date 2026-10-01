"""Tune the CACS weights and threshold on the *training* split by random search.

Objective: F1 between important-cue recall and cue precision, which rewards
saying the important things and staying quiet otherwise. The tuned config is
written to configs/cacs_tuned.yaml and used by every later experiment on the
held-out test split.
"""
import argparse, dataclasses, json, random
from _common import CONFIGS, RESULTS, dump_json, load_split
from seeforme.config import CACSWeights, PipelineConfig
from seeforme.cues import CACSPolicy
from seeforme.cues.text_importance import LearnedTextImportance
from seeforme.eval import aggregate, evaluate_cues, run_policy_on_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

train = load_split("train")
text_model = LearnedTextImportance("models/text_importance.joblib")
rng = random.Random(args.seed)


def objective(cfg: PipelineConfig):
    rows = []
    for sc in train:
        pol = CACSPolicy(cfg.cacs, text_model=text_model)
        rows.append(evaluate_cues(run_policy_on_scenario(pol, sc, cfg), sc, "cacs"))
    a = aggregate(rows)["cacs"]
    r, p = a["important_cue_recall"], a["cue_precision"]
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return f1, a


base = PipelineConfig()
best_f1, best_agg = objective(base)
print(f"default weights: F1={best_f1:.3f} recall={best_agg['important_cue_recall']:.3f} prec={best_agg['cue_precision']:.3f} load={best_agg['audio_load_per_min']:.1f}")
history = [{"trial": -1, "f1": best_f1, **best_agg, "weights": base.cacs.weights.as_dict(), "threshold": base.cacs.threshold}]
best_cfg = base
for i in range(args.trials):
    cfg = PipelineConfig()
    cfg.cacs.weights = CACSWeights(alpha=rng.uniform(0, 1.5), beta=rng.uniform(0, 2.5), gamma=rng.uniform(0, 2),
                                   delta=rng.uniform(0, 3), eta=rng.uniform(0, 3), lam=rng.uniform(0.5, 4))
    cfg.cacs.threshold = rng.uniform(1.5, 4.5)
    cfg.cacs.duplicate_tau = rng.choice([4, 6, 8, 10, 12])
    f1, agg = objective(cfg)
    history.append({"trial": i, "f1": f1, **agg, "weights": cfg.cacs.weights.as_dict(), "threshold": cfg.cacs.threshold,
                    "duplicate_tau": cfg.cacs.duplicate_tau})
    if f1 > best_f1:
        best_f1, best_cfg, best_agg = f1, cfg, agg
        print(f"trial {i:3d}: F1={f1:.3f} recall={agg['important_cue_recall']:.3f} prec={agg['cue_precision']:.3f} "
              f"load={agg['audio_load_per_min']:.1f} thr={cfg.cacs.threshold:.2f} w={cfg.cacs.weights.as_dict()}")

CONFIGS.mkdir(exist_ok=True)
best_cfg.dump(str(CONFIGS / "cacs_tuned.yaml"))
dump_json({"best_f1": best_f1, "best": best_agg, "weights": best_cfg.cacs.weights.as_dict(),
           "threshold": best_cfg.cacs.threshold, "duplicate_tau": best_cfg.cacs.duplicate_tau,
           "history": history}, RESULTS / "tuning.json")
print("saved", CONFIGS / "cacs_tuned.yaml")
