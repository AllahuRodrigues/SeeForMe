"""Train the supervised cue-priority models (logistic regression, GBM, PyTorch MLP)."""
import json
from _common import MODELS, RESULTS, dump_json, load_cfg, load_split
from seeforme.cues.text_importance import LearnedTextImportance
from seeforme.ml.priority_model import train_priority_models

rep = train_priority_models(load_split("train"), load_split("val"), out_dir=str(MODELS), cfg=load_cfg(),
                            text_model=LearnedTextImportance(str(MODELS / "text_importance.joblib")))
dump_json(rep, RESULTS / "priority_model_report.json")
print(json.dumps(rep, indent=1))
