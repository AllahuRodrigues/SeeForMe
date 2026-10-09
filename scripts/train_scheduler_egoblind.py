"""Train the learned scheduler on EgoBlind replays (optionally mixed with CampusBench), on the 70 / 15 / 5 / 10 split of
scripts/egoblind_common.py:

  imitation (DAgger) of the question-derived reference scheduler   on `train`     (70%)
  reward fine-tuning (REINFORCE, the communication score)          on `finetune`  (15%)
  every selection: DAgger iteration, IL threshold, RL silence bias and epoch   on `select` (5%)
  `holdout` (10%) and EgoBlind's official `test` are never touched here.
CampusBench, when mixed in with --train, contributes its `train` split to both training stages (--finetune defaults to
--train); its `val` split is used for selection only when --select includes campus (the default --select is egoblind).
models/egoblind is built with --train egoblind,campus and SEEFORME_CAMPUS unset (CampusBench v1) and selected on EgoBlind
`select` only.

    python scripts/train_scheduler_egoblind.py --train egoblind --out models/egoblind_only
    python scripts/train_scheduler_egoblind.py --train egoblind,campus --out models/egoblind
    python scripts/train_scheduler_egoblind.py --subset 6 --dagger-iters 1 --epochs 2 --rl-epochs 1   # smoke test

Writes models/egoblind/scheduler/scheduler_{il,rl}.pt and models/egoblind/scheduler_report.json, loadable as the
`egoblind_learned:il|rl` policies in scripts/run_egoblind.py (and anywhere make_live_policy takes model_dir=models/egoblind).
Training rollouts can be perturbed (--noise) for diversity; validation replays are never perturbed.
"""
import argparse, json, time
from _common import dump_json
from campus_common import RESULTS_C, load_campus_cfg, load_campus_split, text_model
from egoblind_common import MODELS_E, RESULTS_E, load_egoblind_split, parse_noise
from seeforme.egoblind import EgoBlindScenario, make_runner
from seeforme.ml.scheduler_model import dagger, train_reinforce

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="egoblind", help="comma list of domains: egoblind, campus")
ap.add_argument("--finetune", default="", help="domains for REINFORCE fine-tuning (default: same as --train): EgoBlind 'finetune', CampusBench 'train'")
ap.add_argument("--select", "--val", dest="select", default="egoblind",
                help="domains used for selection (DAgger iteration, IL threshold, RL silence bias and epoch): EgoBlind 'select', "
                     "CampusBench 'val'. EgoBlind 'holdout' and 'test' are never used here")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--dagger-iters", type=int, default=2)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--rl-epochs", type=int, default=12)
ap.add_argument("--subset", type=int, default=0, help="first N scenarios per domain (smoke test)")
ap.add_argument("--kinds", default="", help="restrict EgoBlind scenarios to these kinds")
ap.add_argument("--noise", default="", help="training-rollout perturbation, e.g. p_miss=0.1,depth_rel=0.1")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", default="gru", choices=["gru", "mlp1", "mlpk", "attn"], help="history encoder (parameter-matched)")
ap.add_argument("--out", default=str(MODELS_E))
args = ap.parse_args()

from pathlib import Path
cfg = load_campus_cfg()
kinds = [k for k in args.kinds.split(",") if k] or None


SPLIT_OF = {"train": {"egoblind": "train", "campus": "train"}, "finetune": {"egoblind": "finetune", "campus": "train"},
            "select": {"egoblind": "select", "campus": "val"}}


def domain(role, domains):
    """Scenarios of each domain's split for this role (SPLIT_OF)."""
    out = []
    for d in domains.split(","):
        if d == "egoblind":
            scs = load_egoblind_split(SPLIT_OF[role]["egoblind"], require_cache=True, kinds=kinds)
        elif d == "campus":
            scs = load_campus_split(SPLIT_OF[role]["campus"])
        else:
            raise SystemExit(f"unknown domain {d}")
        out += scs[: args.subset] if args.subset else scs
    return out


train, finetune, val = domain("train", args.train), domain("finetune", args.finetune or args.train), domain("select", args.select)
if not train or not finetune or not val:
    raise SystemExit("empty training, fine-tuning or selection set: build scenarios and caches first "
                     "(scripts/build_egoblind_scenarios.py, scripts/cache_egoblind_perception.py)")
assert not ({s.name for s in train} & {s.name for s in val}) and not ({s.name for s in finetune} & {s.name for s in val})
if args.subset:
    val = val[: max(2, args.subset // 2)]
n_ego = sum(isinstance(s, EgoBlindScenario) for s in train)
print(f"train: {len(train)} scenarios ({n_ego} EgoBlind videos, {len(train) - n_ego} CampusBench); fine-tune: {len(finetune)}; selection: {len(val)} "
      f"({sum(isinstance(s, EgoBlindScenario) for s in val)} EgoBlind); events in EgoBlind train: "
      f"{sum(len(s.speak_events()) for s in train if isinstance(s, EgoBlindScenario))}")
noise = parse_noise(args.noise)
train_names = {s.name for s in train} | {s.name for s in finetune}
run_noisy, run_clean = make_runner(noise), make_runner(None)


def run(policy, sc, cfg, seed=None, on_decision=None, **kw):
    return (run_noisy if (noise is not None and sc.name in train_names and on_decision is not None) else run_clean)(policy, sc, cfg, seed=seed, on_decision=on_decision, **kw)


tm = text_model()
t0 = time.time()
print("imitation (DAgger) of the question-derived reference scheduler")
best, hist_il = dagger(train, val, cfg, K=args.K, n_iter=args.dagger_iters, epochs=args.epochs, text_model=tm, seed=args.seed, log=print, run=run, arch=args.arch)
out = Path(args.out) / "scheduler"; out.mkdir(parents=True, exist_ok=True)
best["model"].save(str(out / "scheduler_il.pt"))
print(f"  best imitation iter {best['iter']} thr={best['threshold']} val score={best['score']:.3f} ({time.time() - t0:.0f}s)")
print("reward fine-tuning (REINFORCE)")
rl = train_reinforce(best["model"], finetune, val, cfg, epochs=args.rl_epochs, text_model=tm, seed=args.seed, log=print, run=run)
best["model"].save(str(out / "scheduler_rl.pt"))
import os
report = {"K": args.K, "arch": args.arch, "n_params": best["model"].n_params, "train_domains": args.train,
          "finetune_domains": args.finetune or args.train, "select_domains": args.select, "splits": SPLIT_OF,
          "campus_version": os.environ.get("SEEFORME_CAMPUS") or "v1",
          "n_train": len(train), "n_finetune": len(finetune), "n_select": len(val), "noise": args.noise, "seed": args.seed,
          "il": {"threshold": best["threshold"], "iter": best["iter"], "val_score": best["score"], "history": hist_il},
          "rl": {"val": rl["best"], "history": rl["history"]}, "seconds": time.time() - t0}
dump_json(report, Path(args.out) / "scheduler_report.json")
dump_json(report, RESULTS_E / f"scheduler_training_{Path(args.out).name}.json")
print(f"done in {time.time() - t0:.0f}s -> {out}")
