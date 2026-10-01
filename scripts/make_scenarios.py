"""Generate the synthetic benchmark: train / val / test splits of controlled scenarios."""
import argparse
from _common import DATA, make_benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--n-per-kind", type=int, default=6)
ap.add_argument("--duration", type=float, default=30.0)
ap.add_argument("--fps", type=float, default=10.0)
args = ap.parse_args()

splits = {"train": 100, "val": 5000, "test": 9000}
for split, seed0 in splits.items():
    out = DATA / split
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    n = args.n_per_kind if split != "val" else max(2, args.n_per_kind // 2)
    scs = make_benchmark(n_per_kind=n, seed0=seed0, duration=args.duration, fps=args.fps)
    for sc in scs:
        sc.to_json(str(out / f"{sc.name}.json"))
    n_ev = sum(len(sc.important_events()) for sc in scs)
    print(f"{split}: {len(scs)} scenarios, {n_ev} important events, {sum(len(s.objects) for s in scs)} objects -> {out}")
