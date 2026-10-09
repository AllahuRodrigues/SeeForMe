"""Re-run committed results and check they reproduce exactly (fixed seeds, cached perception).

    python scripts/repro_check.py            # CampusBench v2 test (3 policies) + EgoBlind val (2 policies), ~10 min
    python scripts/repro_check.py --quick    # 2 CampusBench policies only

Each check re-runs a script with a `repro` tag and compares every numeric metric of the named policies with the committed
summary. Writes results/repro_check.md and exits non-zero on any difference larger than 1e-9.
"""
import argparse, json, os, subprocess, sys
from _common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
PY = sys.executable
env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
env.pop("SEEFORME_EGOBLIND_CACHE", None)
checks = [("CampusBench v2 test", {**env, "SEEFORME_CAMPUS": "v2"},
           [PY, "scripts/run_campus.py", "--split", "test", "--repeats", "1", "--policies", "oracle,pcacs" if args.quick else "oracle,pcacs,pcacs_learned:rl", "--tag", "repro"],
           "results/campus_v2/summary_test_zeroshot.json", "results/campus_v2/summary_test_repro.json")]
if not args.quick:
    checks.append(("EgoBlind holdout (10% unseen split, COCO caches)", env,
                   [PY, "scripts/run_egoblind.py", "--split", "holdout", "--policies", "pcacs,egoblind_learned:il@models/egoblind_only", "--tag", "repro"],
                   "results/egoblind/summary_holdout_final.json", "results/egoblind/summary_holdout_repro.json"))
lines, bad = ["# Reproducibility check", ""], 0
for name, e, cmd, ref, new in checks:
    r = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True)
    if r.returncode != 0:
        lines.append(f"- {name}: FAILED to run (`{' '.join(cmd[1:])}`): {r.stderr.strip().splitlines()[-1] if r.stderr else ''}"); bad += 1; continue
    a, b = json.load(open(ROOT / ref)), json.load(open(ROOT / new))
    diffs, n = [], 0
    for pol, metrics in b.items():
        if pol not in a:
            continue
        flat = lambda d, p="": {f"{p}{k}": v for k, v in d.items() if isinstance(v, (int, float))} | {kk: vv for k, v in d.items() if isinstance(v, dict) for kk, vv in flat(v, f"{p}{k}.").items()}
        fa, fb = flat(a[pol]), flat(metrics)
        for k in sorted(set(fa) & set(fb)):
            n += 1
            x, y = fa[k], fb[k]
            if not (x == y or (x != x and y != y) or abs(x - y) <= 1e-9):
                diffs.append(f"{pol}.{k}: {x} -> {y}")
    bad += bool(diffs)
    lines.append(f"- {name}: {n} metrics compared against `{ref}`, " + ("all identical" if not diffs else f"{len(diffs)} differ: " + "; ".join(diffs[:8])))
    lines.append(f"  command: `{' '.join(cmd[1:])}`")
(ROOT / "results" / "repro_check.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if bad else 0)
