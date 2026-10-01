"""Train the supervised sign-importance classifier (T_i)."""
import json
from _common import MODELS, RESULTS, dump_json
from seeforme.ml.text_classifier import train_and_save

rep = train_and_save(str(MODELS / "text_importance.joblib"))
dump_json(rep, RESULTS / "text_classifier_report.json")
print(json.dumps({k: v for k, v in rep.items() if k != "unseen_examples"}, indent=1))
for s, p in rep["unseen_examples"]:
    print(f"  {s:20s} p(important)={p:.3f}")
