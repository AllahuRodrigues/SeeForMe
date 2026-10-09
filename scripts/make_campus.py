"""Generate the CampusBench train / val / test splits (data/campus/<split>/*.json), each file carrying
the full annotation (events with communication windows, minimum information and ideal cue)."""
import argparse
from _common import ROOT
from seeforme.campus import make_campus_scenario, CAMPUS_KINDS

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=6, help="scenarios per kind")
ap.add_argument("--val", type=int, default=3)
ap.add_argument("--test", type=int, default=6)
ap.add_argument("--duration", type=float, default=30.0)
ap.add_argument("--out", default=str(ROOT / "data" / "campus"), help="data/campus (v1 kinds) or data/campus_v2")
ap.add_argument("--kinds", default="v1", help="'v1' (the 11 original kinds), 'all' (v1 + contention, occlusion, wayfinding_airport, ood_aftermath) or a comma list")
ap.add_argument("--ood-test-only", action="store_true",
                help="(the default; still accepted) out-of-distribution kinds (ood_aftermath) only in the test split, never in train / val")
ap.add_argument("--ood-in-train", action="store_true",
                help="opt in: also generate OOD kinds in train / val (campus_common.load_campus_split refuses such train / val splits)")
args = ap.parse_args()
if args.ood_test_only and args.ood_in_train:
    ap.error("--ood-test-only and --ood-in-train are mutually exclusive")
ood_test_only = not args.ood_in_train

from pathlib import Path
from seeforme.campus import CAMPUS_KINDS_V1, OOD_KINDS
out = Path(args.out)
kinds = CAMPUS_KINDS_V1 if args.kinds == "v1" else (CAMPUS_KINDS if args.kinds == "all" else args.kinds.split(","))
n_speak = n_obj = 0
for split, n, seed0 in (("train", args.train, 2000), ("val", args.val, 3000), ("test", args.test, 4000)):
    d = out / split; d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    written = 0
    for kind in kinds:
        if ood_test_only and kind in OOD_KINDS and split != "test":
            continue
        k = CAMPUS_KINDS.index(kind)          # seeds depend on the kind's fixed index, so v1 scenarios are identical in v2
        for i in range(n):
            sc = make_campus_scenario(kind, seed0 + 100 * k + i, args.duration)
            sc.to_json(str(d / f"{sc.name}.json"))
            n_speak += len(sc.speak_events()); n_obj += len(sc.objects); written += 1
    print(f"{split}: {written} scenarios -> {d}")
print(f"{n_obj} objects, {n_speak} should-speak events in total")
