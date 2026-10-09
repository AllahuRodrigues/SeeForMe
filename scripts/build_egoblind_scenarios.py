"""Turn the EgoBlind CSVs into weakly labelled scenarios (data/egoblind/scenarios/<split>/<video_id>.json).

    python scripts/build_egoblind_scenarios.py                 # all splits; refines windows where a perception cache exists
    python scripts/build_egoblind_scenarios.py --only-cached   # only videos that already have data/egoblind/cache/<id>.npz

Prints and saves (data/egoblind/scenarios/coverage.json) how many questions became should-speak events, quiet windows
or queries, how many events the current perception stack can cover at all, and which objects / concepts dominate.
"""
import argparse, collections, json
from _common import dump_json
from egoblind_common import CACHE, SCENARIOS, SPLITS, cache_path, load_egoblind_questions, video_path
from seeforme.egoblind import PerceptionCache, build_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default=",".join(SPLITS))
ap.add_argument("--only-cached", action="store_true")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

report = {}
for split in args.splits.split(","):
    by_video = load_egoblind_questions(split)
    agg = {"videos": 0, "videos_with_cache": 0, "videos_with_video": 0, "questions": 0, "present": 0, "clear": 0, "skip": 0, "coverable": 0,
           "detected": 0, "coverable_with_cache": 0, "uncoverable": collections.Counter(), "by_type": collections.Counter(),
           "kinds": collections.Counter(), "labels": collections.Counter(), "concepts": collections.Counter(), "sudden": 0}
    n = 0
    # drop scenario files of videos that left this split (the split or the labels changed); load_egoblind_split globs the dir
    for stale in (SCENARIOS / split).glob("*.json"):
        if stale.stem.isdigit() and int(stale.stem) not in by_video:
            stale.unlink()
    for vid, qs in sorted(by_video.items()):
        cp = cache_path(vid)
        cache = PerceptionCache.load(cp) if cp.exists() else None
        if args.only_cached and cache is None:
            continue
        vp = video_path(vid)
        sc = build_scenario(vid, qs, split, video=str(vp) if vp else "", cache=cache, cache_path=str(cp))
        if cache is not None:
            sc.stats["refined_from"] = cache.meta.get("built")
        sc.to_json(SCENARIOS / split / f"{vid}.json")
        st = sc.stats
        agg["videos"] += 1; agg["videos_with_cache"] += cache is not None; agg["videos_with_video"] += vp is not None
        agg["questions"] += st["n_questions"]; agg["present"] += st["present"]; agg["clear"] += st["clear"]; agg["skip"] += st["skip"]
        agg["coverable"] += st["coverable"]; agg["uncoverable"].update(st["uncoverable"]); agg["by_type"].update(st["by_type"])
        agg["kinds"][sc.kind] += 1
        if cache is not None:
            agg["detected"] += st.get("detected", 0); agg["coverable_with_cache"] += st.get("coverable_total", 0)
        for e in sc.speak_events():
            agg["sudden"] += e.sudden
            if e.detail.get("coverable"):
                agg["labels"][e.label] += 1
            else:
                for c in e.detail.get("concepts", [])[:1] or [e.label]:
                    agg["concepts"][c] += 1
        n += 1
        if args.limit and n >= args.limit:
            break
    agg["uncoverable"] = dict(agg["uncoverable"]); agg["by_type"] = dict(agg["by_type"]); agg["kinds"] = dict(agg["kinds"])
    agg["labels"] = dict(agg["labels"].most_common(15)); agg["concepts"] = dict(agg["concepts"].most_common(15))
    report[split] = agg
    print(f"[{split}] {agg['videos']} videos ({agg['videos_with_video']} with mp4, {agg['videos_with_cache']} with cache), {agg['questions']} questions: "
          f"{agg['present']} should-speak events, {agg['clear']} quiet windows, {agg['skip']} queries/skipped")
    print(f"   coverable by YOLO-COCO + taxonomy: {agg['coverable']}/{agg['present']}  uncoverable: {agg['uncoverable']}  sudden: {agg['sudden']}")
    if agg["videos_with_cache"]:
        print(f"   detected before t_q (cached videos): {agg['detected']}/{agg['coverable_with_cache']}")
    print(f"   kinds: {agg['kinds']}\n   top labels: {agg['labels']}\n   top uncoverable concepts: {agg['concepts']}")
dump_json(report, SCENARIOS / "coverage.json")
print(f"-> {SCENARIOS}/<split>/*.json, coverage.json")
