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
args = ap.parse_args()

out = ROOT / "data" / "campus"
n_speak = n_obj = 0
for split, n, seed0 in (("train", args.train, 2000), ("val", args.val, 3000), ("test", args.test, 4000)):
    d = out / split; d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    for k, kind in enumerate(CAMPUS_KINDS):
        for i in range(n):
            sc = make_campus_scenario(kind, seed0 + 100 * k + i, args.duration)
            sc.to_json(str(d / f"{sc.name}.json"))
            n_speak += len(sc.speak_events()); n_obj += len(sc.objects)
    print(f"{split}: {n * len(CAMPUS_KINDS)} scenarios -> {d}")
print(f"{n_obj} objects, {n_speak} should-speak events in total")
