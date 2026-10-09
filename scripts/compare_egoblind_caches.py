"""Coverage of the weak events under two perception caches (COCO YOLO11n vs open-vocabulary YOLO-World).

    python scripts/compare_egoblind_caches.py --split holdout --a data/egoblind/cache --b data/egoblind/cache_ow

Loads the split against each cache directory (coverability and detection are re-derived from the cache's vocabulary
and detections), restricts to the videos present in both, and reports how many should-speak events are coverable and
detected before the question under each, which concepts changed, and the detection histogram of the second cache.
Writes results/egoblind/cache_comparison_<split>.md.
"""
import argparse, importlib, os
from collections import Counter
import numpy as np
from _common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="holdout")
ap.add_argument("--a", default="data/egoblind/cache")
ap.add_argument("--b", default="data/egoblind/cache_ow")
ap.add_argument("--names", default="YOLO11n (COCO),YOLO-World (NAV_VOCAB)")
args = ap.parse_args()
names = args.names.split(",")


def load(cache_dir):
    os.environ["SEEFORME_EGOBLIND_CACHE"] = cache_dir
    import egoblind_common as ec
    importlib.reload(ec)
    return {sc.video_id: sc for sc in ec.load_egoblind_split(args.split, require_cache=True)}


A, B = load(args.a), load(args.b)
common = sorted(set(A) & set(B))


def stats(scs):
    c = Counter()
    for sc in scs:
        for e in sc.speak_events():
            c["events"] += 1
            if e.detail.get("coverable"):
                c["coverable"] += 1
                if e.detail.get("detected"):
                    c["detected"] += 1
            else:
                c[f"uncoverable:{e.detail.get('coverable_reason')}"] += 1
    return c


sa, sb = stats([A[v] for v in common]), stats([B[v] for v in common])
changed = []
for v in common:
    ea = {e.obj_id: e for e in A[v].speak_events()}
    for e in B[v].speak_events():
        a = ea[e.obj_id]
        if bool(a.detail.get("coverable")) != bool(e.detail.get("coverable")) or bool(a.detail.get("detected")) != bool(e.detail.get("detected")):
            changed.append((v, e.detail.get("concepts"), e.detail.get("labels"), a.detail.get("coverable"), a.detail.get("detected"),
                            e.detail.get("coverable"), e.detail.get("detected"), (e.detail.get("question") or "")[:70]))
labs = Counter()
for v in common:
    c = B[v].cache()
    for lab, n in zip(c.labels, np.bincount(c.det_label, minlength=len(c.labels))):
        labs[lab] += int(n)
lines = [f"# {args.split}: weak-event coverage under two perception caches ({len(common)} videos in both; {len(A)} / {len(B)} cached)", "",
         f"| | {names[0]} | {names[1]} |", "|---|---|---|"]
for k in ("events", "coverable", "detected"):
    lines.append(f"| {k} | {sa[k]} | {sb[k]} |")
for k in sorted(set(sa) | set(sb)):
    if k.startswith("uncoverable:"):
        lines.append(f"| {k} | {sa[k]} | {sb[k]} |")
lines += ["", f"Events whose coverability or detection changed ({len(changed)}):", "",
          "| video | concepts | labels | coverable a -> b | detected a -> b | question |", "|---|---|---|---|---|---|"]
lines += [f"| {v} | {','.join(c or [])} | {','.join(l or [])} | {ca} -> {cb} | {da} -> {db} | {q} |" for v, c, l, ca, da, cb, db, q in changed]
lines += ["", f"Detections by label in {names[1]} caches: " + ", ".join(f"{l} {n}" for l, n in labs.most_common(30))]
out = ROOT / "results" / "egoblind" / f"cache_comparison_{args.split}.md"
out.write_text("\n".join(lines) + "\n"); print("\n".join(lines)); print("->", out)
