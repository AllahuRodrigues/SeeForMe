"""Train the learned predictive scheduler on CampusBench: DAgger imitation of the oracle, then
REINFORCE fine-tuning with the benchmark score as reward. Writes models/scheduler/*.pt and
models/scheduler_report.json; results/campus[_<version>]/scheduler_training.json keeps the curves (a run with
--out models/<name> writes models/<name>/... and scheduler_training_<name>.json instead, so variants never overwrite it)."""
import argparse, json, os, time
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
ap.add_argument("--out", default=str(MODELS), help="model directory (default models/; variants go to models/<name>)")
ap.add_argument("--memory-ttl", type=float, default=0.0, help="train with the tracker memory ledger on (seconds; 0 = frame-only)")
ap.add_argument("--arch", default="gru", choices=["gru", "mlp1", "mlpk", "attn"], help="history encoder (parameter-matched)")
args = ap.parse_args()

cfg = load_campus_cfg()
if args.memory_ttl > 0:
    from dataclasses import replace
    cfg.tracker = replace(cfg.tracker, memory_ttl_s=args.memory_ttl)
train, val = load_campus_split("train"), load_campus_split("val")
if args.subset:
    train, val = train[: args.subset], val[: max(2, args.subset // 2)]
tm = text_model()
t0 = time.time()
print(f"CampusBench: {len(train)} train / {len(val)} val scenarios")
print("imitation (DAgger)")
best, hist_il = dagger(train, val, cfg, K=args.K, n_iter=args.dagger_iters, epochs=args.epochs, text_model=tm, seed=args.seed, log=print, arch=args.arch)
from pathlib import Path
out = Path(args.out) / "scheduler"; out.mkdir(parents=True, exist_ok=True)
best["model"].save(str(out / "scheduler_il.pt"))
print(f"  best imitation iter {best['iter']} thr={best['threshold']} val score={best['score']:.3f} ({time.time()-t0:.0f}s)")
print("reward fine-tuning (REINFORCE)")
rl = train_reinforce(best["model"], train, val, cfg, epochs=args.rl_epochs, text_model=tm, seed=args.seed, log=print)
best["model"].save(str(out / "scheduler_rl.pt"))
report = {"campus_version": os.environ.get("SEEFORME_CAMPUS") or "v1", "K": args.K, "arch": args.arch, "n_params": best["model"].n_params, "il": {"threshold": best["threshold"], "iter": best["iter"], "val_score": best["score"], "history": hist_il},
          "rl": {"val": rl["best"], "history": rl["history"]}, "seconds": time.time() - t0}
dump_json(report, Path(args.out) / "scheduler_report.json")
dump_json(report, RESULTS_C / ("scheduler_training.json" if Path(args.out).resolve() == MODELS.resolve() else f"scheduler_training_{Path(args.out).name}.json"))
print(f"done in {time.time()-t0:.0f}s -> {out}")
