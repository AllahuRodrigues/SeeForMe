"""Train the on-policy (DAgger) feature models and the deep context (GRU) model,
then fine-tune the GRU with REINFORCE. Writes models/deep_context/* and
models/deep_context_report.json (copied to results/).

  python scripts/train_deep_context.py            # full run (~10-15 min)
  python scripts/train_deep_context.py --quick    # smoke test
"""
import argparse, copy, json, shutil, time
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from _common import MODELS, RESULTS, dump_json, load_cfg, load_split
from seeforme.cues.cacs import LearnedCACS
from seeforme.cues.text_importance import LearnedTextImportance
from seeforme.ml.onpolicy import FEATURE_NAMES, OraclePolicy, collect, dagger
from seeforme.ml.priority_model import TorchMLPPriority
from seeforme.ml.sequence_model import SequenceCACS, SequenceModel, train_reinforce

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--dagger-iters", type=int, default=2)
ap.add_argument("--rl-epochs", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = load_cfg()
train, val = load_split("train"), load_split("val")
if args.quick:
    train, val = train[:6], val[:3]
    args.dagger_iters, args.rl_epochs = 1, 2
tm = LearnedTextImportance(str(MODELS / "text_importance.joblib"))
out = MODELS / "deep_context"; out.mkdir(parents=True, exist_ok=True)
t_start = time.time()

print("collecting oracle data ...")
data0 = collect(lambda sc: OraclePolicy(sc, cfg.cacs, seed=args.seed + sc.seed, text_model=tm), train, cfg, repeats=2, seed=args.seed)
print(f"  {len(data0)} rows, positive rate {data0.y.mean():.3f}, features {FEATURE_NAMES}")
report = {"n_oracle_rows": int(len(data0)), "pos_rate": float(data0.y.mean()), "features": FEATURE_NAMES, "dagger": {}}

factories = {
    "logreg": (lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"), 6),
    "gbm": (lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=args.seed), 6),
    "mlp13": (lambda: TorchMLPPriority(seed=args.seed, device="cpu", epochs=10 if args.quick else 40), 13),
}
for name, (mf, nf) in factories.items():
    print(f"DAgger: {name} ({nf} features)")
    best, hist = dagger(mf, lambda model, thr, sc: LearnedCACS(model, cfg.cacs, threshold=thr, n_features=nf, text_model=tm),
                        data0, train, val, cfg, n_iter=args.dagger_iters, n_features=nf, seed=args.seed)
    joblib.dump(best["model"], out / f"{name}.joblib")
    r = {"n_features": nf, "threshold": best["threshold"], "best_iter": best["iter"], "val_f1": best["f1"], "history": hist}
    if name == "logreg":
        r["learned_weights"] = {n: float(c) for n, c in zip(FEATURE_NAMES[:6], best["model"].coef_[0])}
    report["dagger"][name] = r

print("DAgger: GRU sequence model (imitation)")
best, hist = dagger(lambda: SequenceModel(K=8, epochs=8 if args.quick else 30, seed=args.seed),
                    lambda model, thr, sc: SequenceCACS(model, cfg.cacs, threshold=thr, mode="imitation", text_model=tm),
                    data0, train, val, cfg, n_iter=args.dagger_iters, n_features=13, seed=args.seed)
best["model"].save(str(out / "seq_il.pt"))
report["seq_il"] = {"threshold": best["threshold"], "best_iter": best["iter"], "val_f1": best["f1"], "history": hist, "K": 8}

print("REINFORCE fine-tuning of the sequence model")
seq_rl = copy.deepcopy(best["model"])
rl = train_reinforce(seq_rl, train, val, cfg, epochs=args.rl_epochs, seed=args.seed, text_model=tm)
seq_rl.save(str(out / "seq_rl.pt"))
report["seq_rl"] = {"val_f1": rl["best"]["f1"], "best": rl["best"], "history": rl["history"],
                    "silence_logit": float(seq_rl.net.silence.item())}
report["minutes"] = round((time.time() - t_start) / 60, 1)
dump_json(report, MODELS / "deep_context_report.json")
shutil.copy(MODELS / "deep_context_report.json", RESULTS / "deep_context_report.json")
print(json.dumps({k: (v if k != "dagger" else {n: {kk: vv for kk, vv in r.items() if kk != "history"} for n, r in v.items()})
                  for k, v in report.items() if k not in ("seq_il", "seq_rl")}, indent=1))
print("seq_il val F1", report["seq_il"]["val_f1"], "| seq_rl val F1", report["seq_rl"]["val_f1"], "| minutes", report["minutes"])
