"""Score the sign-importance models on research/sign_taxonomy.json (89 strings, 13 categories) plus the
Portuguese / Spanish / French translations in seeforme/cues/sign_vocab.py, then train a v2 model that has seen
the taxonomy *minus* a held-out fold and report the gain on the held-out strings.

    python scripts/eval_sign_taxonomy.py            # -> results/sign_taxonomy_eval.{json,md}, models/text_importance_v2.joblib

v1 = models/text_importance.joblib (English templates only), keyword = the hand-written rules.
"""
import csv, json, random
from collections import defaultdict
from pathlib import Path
from _common import MODELS, RESULTS, ROOT, dump_json
from seeforme.cues.sign_vocab import taxonomy_rows
from seeforme.cues.text_importance import KeywordTextImportance, LearnedTextImportance, normalise_text
from seeforme.ml.text_classifier import generate_dataset, build_model

rows = taxonomy_rows()
csv_path = ROOT / "data" / "text_importance" / "sign_taxonomy.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["text", "label", "category", "language"]); w.writeheader(); w.writerows(rows)


from seeforme.cues.sign_vocab import DECOR_TRANSLATIONS, SIGN_TRANSLATIONS

# Group every string with its translations, so no sign intent has strings on both sides of a CV split ("EXIT" in train and
# "SALIDA" in test would make the held-out score a memory test), and exact duplicates after normalisation share a group.
EN_TO_INTENT = {normalise_text(t["en"]): f"sign:{k}" for k, t in SIGN_TRANSLATIONS.items() if "en" in t}
EN_TO_INTENT.update({normalise_text(t["en"]): f"decor:{k}" for k, t in DECOR_TRANSLATIONS.items() if "en" in t})


def group_of(r):
    cat = r["category"]
    if cat.startswith("translated:"):
        return "sign:" + cat.split(":", 1)[1]
    if cat.startswith("decor:"):
        return cat
    return EN_TO_INTENT.get(normalise_text(r["text"]), "text:" + normalise_text(r["text"]))


SUBSETS = {"all": lambda r: True, "English taxonomy": lambda r: r["language"] == "en" and ":" not in r["category"],
           "non-English (pt / es / fr)": lambda r: r["language"] != "en"}
for lang in ("en", "pt", "es", "fr"):
    SUBSETS[f"lang:{lang}"] = (lambda L: (lambda r: r["language"] == L))(lang)


def metrics(pairs):
    """pairs: (label, predicted 0/1). Accuracy, balanced accuracy (mean of the two class recalls), recall per class."""
    pos = [p for y, p in pairs if y == 1]; neg = [p for y, p in pairs if y == 0]
    rec1 = sum(pos) / len(pos) if pos else float("nan"); rec0 = (len(neg) - sum(neg)) / len(neg) if neg else float("nan")
    acc = sum(int(y == p) for y, p in pairs) / len(pairs)
    bal = (rec1 + rec0) / 2 if pos and neg else float("nan")
    return {"n": len(pairs), "n_important": len(pos), "accuracy": round(acc, 3), "balanced_accuracy": round(bal, 3) if bal == bal else None,
            "recall_important": round(rec1, 3) if pos else None, "recall_not_important": round(rec0, 3) if neg else None}


def evaluate(pred):
    """pred(row) -> 0/1 for every subset."""
    return {name: metrics([(int(r["label"]), int(pred(r))) for r in rows if f(r)]) for name, f in SUBSETS.items()}


class _Pipe:
    def __init__(self, m): self.m = m
    def score(self, t): return float(self.m.predict_proba([normalise_text(t)])[0, 1])


# 5-fold cross-validation over sign *intents*: train on the English templates + 4 folds of groups, score the 5th
groups = sorted({group_of(r) for r in rows})
rng = random.Random(0); rng.shuffle(groups)
fold_of = {g: i % 5 for i, g in enumerate(groups)}
cv_pred = {}
for k in range(5):
    X, y = generate_dataset(seed=k, extra_csv="/nonexistent")
    for i, r in enumerate(rows):
        if fold_of[group_of(r)] != k:
            X.append(normalise_text(r["text"])); y.append(int(r["label"]))
    m = build_model().fit(X, y)
    for i, r in enumerate(rows):
        if fold_of[group_of(r)] == k:
            cv_pred[i] = float(m.predict_proba([normalise_text(r["text"])])[0, 1])
idx_of = {id(r): i for i, r in enumerate(rows)}
kw, v1 = KeywordTextImportance(), LearnedTextImportance(str(MODELS / "text_importance.joblib"))
majority = int(sum(r["label"] for r in rows) * 2 >= len(rows))
results = {"n_rows": len(rows), "n_groups": len(groups), "cv": "5-fold, grouped by sign intent (a string and all its translations share a fold)",
           "always_important": evaluate(lambda r: 1), "majority_class": evaluate(lambda r: majority),
           "keyword": evaluate(lambda r: kw.score(r["text"]) >= 0.5), "learned_v1": evaluate(lambda r: v1.score(r["text"]) >= 0.5),
           "learned_v2_grouped_cv": evaluate(lambda r: cv_pred[idx_of[id(r)]] >= 0.5)}
# final v2 model on everything (templates + full taxonomy + translations) for deployment
from seeforme.ml.text_classifier import train_and_save
rep = train_and_save(str(MODELS / "text_importance_v2.joblib"), extra_csv=str(csv_path))
results["v2_model"] = {k: v for k, v in rep.items() if k != "unseen_examples"}
results["worst_v1"] = sorted(((round(v1.score(r["text"]), 2), r["text"], r["label"], r["language"]) for r in rows),
                            key=lambda x: abs(x[2] - x[0]), reverse=True)[:15]
dump_json(results, RESULTS / "sign_taxonomy_eval.json")
cols = [("always 'important'", "always_important"), ("keyword rules", "keyword"), ("learned v1 (English templates)", "learned_v1"),
        ("learned v2 (grouped CV)", "learned_v2_grouped_cv")]
lines = [f"{len(rows)} strings in {len(groups)} sign-intent groups; v2 is scored by 5-fold cross-validation grouped by intent "
         "(a sign and its translations are never split). Cells: accuracy / balanced accuracy (mean of the two class recalls; "
         "'-' when the subset has one class).", "",
         "| subset (n, important) | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
fmt = lambda m: f"{m['accuracy']:.2f} / " + (f"{m['balanced_accuracy']:.2f}" if m["balanced_accuracy"] is not None else "-")
for name in SUBSETS:
    m0 = results["keyword"][name]
    lines.append(f"| {name} ({m0['n']}, {m0['n_important']}) | " + " | ".join(fmt(results[k][name]) for _, k in cols) + " |")
lines += ["", "Recall of important / not-important strings in the non-English subset: "
          + "; ".join(f"{c} {results[k]['non-English (pt / es / fr)']['recall_important']:.2f} / {results[k]['non-English (pt / es / fr)']['recall_not_important']:.2f}"
                      for c, k in cols)]
(RESULTS / "sign_taxonomy_eval.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("hardest for v1:", results["worst_v1"][:8])
