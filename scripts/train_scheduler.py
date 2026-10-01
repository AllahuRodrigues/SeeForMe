"""Train the learned predictive scheduler on CampusBench: DAgger imitation of the oracle, then
REINFORCE fine-tuning with the benchmark score as reward. Writes models/scheduler/*.pt and
models/scheduler_report.json; results/campus/scheduler_training.json keeps the curves."""
import argparse, json, time
from _common import MODELS, dump_json
from campus_common import RESULTS_C, load_campus_cfg, load_campus_split, text_model
from seeforme.ml.scheduler_model import dagger, train_reinforce, closed_loop, SchedulerPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dagger-iters", type=int, default=2)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--rl-epochs", type=int, default=12)
ap.add_argument("--subset", type=int, default=0, help="first N train / val scenarios (smoke test)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = load_campus_cfg()
train, val = load_campus_split("train"), load_campus_split("val")
if args.subset:
    train, val = train[: args.subset], val[: max(2, args.subset // 2)]
tm = text_model()
t0 = time.time()
print(f"CampusBench: {len(train)} train / {len(val)} val scenarios")
print("imitation (DAgger)")
best, hist_il = dagger(train, val, cfg, K=args.K, n_iter=args.dagger_iters, epochs=args.epochs, text_model=tm, seed=args.seed, log=print)
out = MODELS / "scheduler"; out.mkdir(parents=True, exist_ok=True)
best["model"].save(str(out / "scheduler_il.pt"))
print(f"  best imitation iter {best['iter']} thr={best['threshold']} val score={best['score']:.3f} ({time.time()-t0:.0f}s)")
print("reward fine-tuning (REINFORCE)")
rl = train_reinforce(best["model"], train, val, cfg, epochs=args.rl_epochs, text_model=tm, seed=args.seed, log=print)
best["model"].save(str(out / "scheduler_rl.pt"))
report = {"K": args.K, "il": {"threshold": best["threshold"], "iter": best["iter"], "val_score": best["score"], "history": hist_il},
          "rl": {"val": rl["best"], "history": rl["history"]}, "seconds": time.time() - t0}
dump_json(report, MODELS / "scheduler_report.json")
dump_json(report, RESULTS_C / "scheduler_training.json")
print(f"done in {time.time()-t0:.0f}s -> {out}")
