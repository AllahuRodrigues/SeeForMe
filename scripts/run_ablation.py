"""Ablation study: switch individual CACS signals off and measure the effect."""
import argparse, copy, csv, json
from _common import MODELS, RESULTS, dump_json, load_cfg, load_split
from seeforme.cues import CACSPolicy
from seeforme.cues.text_importance import LearnedTextImportance, KeywordTextImportance
from seeforme.eval import aggregate, evaluate_cues, run_policy_on_scenario
from seeforme.ml.scene_context import SceneContextModel

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()

cfg = load_cfg()
scenarios = load_split(args.split)
text_model = LearnedTextImportance(str(MODELS / "text_importance.joblib"))
ctx_path = MODELS / "scene_context.joblib"

variants = {
    "cacs_full": {},
    "no_novelty": {"use_novelty": False},
    "no_proximity": {"use_proximity": False},
    "no_uncertainty": {"use_uncertainty": False},
    "no_temporal_memory": {"use_temporal_memory": False},
    "no_relevance": {"use_relevance": False},
    "no_text_importance": {"use_text": False},
    "no_scene_context": {"use_scene_context": False},
    "keyword_text_model": {"__text": "keyword"},
}
rows = []
for name, changes in variants.items():
    for sc in scenarios:
        for rep in range(args.repeats):
            c = copy.deepcopy(cfg)
            tm = text_model
            for k, v in changes.items():
                if k == "__text":
                    tm = KeywordTextImportance()
                else:
                    setattr(c.cacs, k, v)
            ctx = SceneContextModel.load(str(ctx_path)) if ctx_path.exists() else None
            pol = CACSPolicy(c.cacs, text_model=tm, scene_context=ctx)
            cues = run_policy_on_scenario(pol, sc, c, scene_context=ctx, seed=sc.seed + 17 * rep)
            rows.append(evaluate_cues(cues, sc, name))
    a = aggregate([r for r in rows if r.policy == name])[name]
    print(f"{name:22s} recall={a['important_cue_recall']:.3f} prec={a['cue_precision']:.3f} "
          f"load={a['audio_load_per_min']:5.1f}/min dup={a['duplicate_rate']:.3f} delay={a['mean_critical_delay']:.2f}s")

agg = aggregate(rows)
dump_json(agg, RESULTS / f"ablation_{args.split}.json")
with open(RESULTS / f"ablation_per_scenario_{args.split}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys())); w.writeheader()
    for r in rows: w.writerow(r.as_dict())
lines = ["| variant | important-cue recall | cue precision | audio load (cues/min) | duplicate rate | mean critical delay (s) |", "|---|---|---|---|---|---|"]
for p, a in agg.items():
    d = a["mean_critical_delay"]
    lines.append(f"| {p} | {a['important_cue_recall']:.3f} | {a['cue_precision']:.3f} | {a['audio_load_per_min']:.1f} | {a['duplicate_rate']:.3f} | {'n/a' if d != d else f'{d:.2f}'} |")
(RESULTS / f"ablation_{args.split}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
