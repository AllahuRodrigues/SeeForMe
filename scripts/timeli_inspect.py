"""Summarise the TIMELI annotation files (Time-Aware Assistive Navigation; Kuribayashi, Shangguan and Ohn-Bar, ICRA 2026).

The files come from the "Data" Google Drive folder linked on https://timeli-icra.github.io/ (downloaded 2026-10-08;
dataset.croissant.json declares CC BY 4.0): train_labels.json (56,281 CARLA frames, 360 MB), test_labels.json (10,820
frames, 70 MB), real_test.json (1,059 YouTube frames, 4.9 MB). The images (32 GB), real-test frames (38 GB) and videos
(537 GB) were not downloaded; the two small code archives (Dataset_Generation, CarlaWalker) are under data/timeli/code.

    python scripts/timeli_inspect.py                    # -> results/timeli/inspect.{json,md}
    python scripts/timeli_inspect.py --splits test,real  # skip the 360 MB train file

How time and silence are represented. One record per frame at 1 FPS; `id` is the frame index and therefore the time in
seconds since the start of the session video `video_id`; frames are contiguous (ids 16..N for CARLA, 15..N for the real
set; the first frames only feed the history). The instruction for that second is `QA[0].a`: a JSON string
{"reason", "instruction"} for CARLA, plain text for the real set. Silence is the instruction "None" with the reason
`remain_silent` or `remain_silent_in_junction` (`speaking_reason` in CARLA, `spoken_reason` in the real set). `history`
holds the 15 previous seconds' instructions oldest first (index 14 = 1 s ago, "None" for silence). The clip at `video`
spans the 16 frames up to and including `id`. The ground-truth rule (Dataset_Generation/make_dataset.py,
`determine_to_talk`) is, in priority order: first frame -> constant_instruction; an instruction in the previous frame
-> remain_silent (never two in a row); junction entered / left -> enter_junction / exit_junction; a not-yet-conveyed
obstacle class in front -> obstacle_in_front; 6 to 9 (random) silent frames outside a junction -> constant_instruction
(or stop at the goal); plan changes from "walk straight" to a turn, or a left-right ambiguity -> direction_change;
otherwise remain_silent (remain_silent_in_junction inside a junction). "wall" is never conveyed and "pedestrian" may be
repeated after 3 frames; other obstacle classes are conveyed once per history window.
"""
import argparse, json, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from _common import RESULTS, ROOT, dump_json

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "timeli"))
ap.add_argument("--splits", default="train,test,real")
ap.add_argument("--out", default=str(RESULTS / "timeli"))
args = ap.parse_args()
DATA = Path(args.data); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
FILES = {"train": "train_labels.json", "test": "test_labels.json", "real": "real_test.json"}
SILENT = {"remain_silent", "remain_silent_in_junction"}


def instruction_of(rec):
    """(reason, instruction text or None, is_silent) for a CARLA (JSON answer) or real (plain answer) record."""
    a = rec["QA"][0]["a"]
    reason = rec.get("speaking_reason", rec.get("spoken_reason"))
    text = a
    if isinstance(a, str) and a.lstrip().startswith("{"):
        try:
            j = json.loads(a); reason = j.get("reason", reason); text = j.get("instruction")
        except json.JSONDecodeError:
            pass
    text = (text or "").strip()
    silent = text.lower() in ("none", "")
    return reason, (None if silent else text), silent


def runs_of_silence(flags):
    """Lengths of maximal runs of True in a boolean sequence."""
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def bucket(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7-9" if n <= 9 else "10+"


def describe_fields(rec):
    out = {}
    for k, v in rec.items():
        ex = json.dumps(v)
        out[k] = {"type": type(v).__name__, "example": ex[:110] + ("..." if len(ex) > 110 else "")}
    return out


def inspect(split, path):
    recs = json.loads(path.read_text())
    by = defaultdict(list)
    for r in recs:
        by[r["video_id"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["id"])
    gaps = Counter(); first_ids = Counter(); n_frames = []
    reasons = Counter(); actions = Counter(); junction = 0; spoken = 0; words = []
    silence_runs = []; inter = []; inconsistent = Counter(); first_frame_reason = Counter()
    conveyed_obj = Counter(); conveyed_dir = Counter(); cluster_obj = Counter(); cluster_dir = Counter(); det_labels = Counter()
    weather = Counter(); town = Counter(); history_ok = history_n = 0; pred_nonnull = 0; seen = Counter()
    for vid, rs in by.items():
        ids = [r["id"] for r in rs]; n_frames.append(len(rs)); first_ids[ids[0]] += 1
        for a, b in zip(ids, ids[1:]):
            gaps[b - a] += 1
        flags = []; last_t = None; prev = None
        for r in rs:
            reason, text, silent = instruction_of(r)
            reasons[reason] += 1; actions[r.get("action")] += 1; junction += bool(r.get("is_junction"))
            if r["id"] == ids[0]:
                first_frame_reason[reason] += 1
            if silent and reason not in SILENT:
                inconsistent[f"{reason} but no instruction"] += 1
            if not silent and reason in SILENT:
                inconsistent[f"{reason} but instruction given"] += 1
            if not silent:
                spoken += 1; words.append(len(text.split()))
                if last_t is not None:
                    inter.append(r["id"] - last_t)
                last_t = r["id"]
            flags.append(silent)
            if prev is not None and r.get("history"):
                history_n += 1; history_ok += (r["history"][-1] == (instruction_of(prev)[1] or "None"))
            prev = r
            for obj, dirs in (r.get("conveyed_obstacles") or {}).items():
                conveyed_obj[obj] += 1
                for d in dirs:
                    conveyed_dir[d] += 1
            for obj, dirs in (r.get("description_cluster") or {}).items():
                cluster_obj[obj] += 1
                for d in (dirs if isinstance(dirs, list) else [dirs]):
                    cluster_dir[str(d)] += 1
            od = r.get("object_detection")
            if isinstance(od, list):
                for o in od:
                    det_labels[o.get("label", o.get("class", "?")) if isinstance(o, dict) else str(o)[:20]] += 1
            m = r.get("meta") or {}
            weather[m.get("weather")] += 1; town[m.get("town")] += 1
            if r.get("prediction") not in (None, "None"):
                pred_nonnull += 1
            if "seen" in r:
                seen[str(r["seen"])] += 1
        silence_runs += runs_of_silence(flags)
    od0 = next((r["object_detection"] for r in recs if r.get("object_detection")), None)
    n = len(recs)
    return {
        "file": path.name, "bytes": path.stat().st_size, "records": n, "videos": len(by),
        "frames_per_video": {"min": min(n_frames), "mean": round(st.mean(n_frames), 1), "max": max(n_frames)},
        "first_frame_ids": dict(first_ids.most_common(3)), "id_gaps": dict(gaps.most_common(3)),
        "fields": describe_fields(recs[0]),
        "object_detection_element": (json.dumps(od0[0])[:200] if od0 else None),
        "reasons": dict(reasons.most_common()), "actions": dict(actions.most_common()),
        "junction_share": round(junction / n, 3), "spoken_share": round(spoken / n, 3),
        "instructions_per_minute": round(60 * spoken / n, 2),
        "instruction_words": {"mean": round(st.mean(words), 2), "sd": round(st.pstdev(words), 2), "max": max(words), "min": min(words)} if words else None,
        "silence_runs": {"n": len(silence_runs), "seconds_total": sum(silence_runs), "mean_s": round(st.mean(silence_runs), 2) if silence_runs else None,
                         "longest_s": max(silence_runs) if silence_runs else None, "by_length": dict(sorted(Counter(bucket(x) for x in silence_runs).items()))},
        "gap_between_instructions_s": {"mean": round(st.mean(inter), 2), "median": st.median(inter), "max": max(inter)} if inter else None,
        "first_frame_reason": dict(first_frame_reason), "reason_vs_instruction_inconsistencies": dict(inconsistent),
        "history_last_equals_previous_instruction": round(history_ok / history_n, 3) if history_n else None,
        "conveyed_obstacles": dict(conveyed_obj.most_common(15)), "conveyed_directions": dict(conveyed_dir),
        "description_cluster_objects": dict(cluster_obj.most_common(15)), "description_cluster_directions": dict(cluster_dir),
        "object_detection_labels": dict(det_labels.most_common(15)),
        "weather": dict(weather.most_common()), "town": dict(town.most_common()),
        "prediction_non_null": pred_nonnull, "seen": dict(seen),
    }


summary = {"source": "https://timeli-icra.github.io/ (Data folder on Google Drive)", "licence": "CC BY 4.0 (dataset.croissant.json)",
           "not_downloaded": {"images.tar.gz": "32 GB", "real_test.tar.gz": "38 GB", "videos.tar.gz": "537 GB", "TIMELI_Train_Eval.tar.gz": "14 MB (model code)"},
           "splits": {}}
for split in args.splits.split(","):
    path = DATA / FILES[split]
    if not path.exists():
        print(f"missing {path}"); continue
    print(f"reading {path.name} ...")
    summary["splits"][split] = inspect(split, path)
dump_json(summary, OUT / "inspect.json")

lines = ["# TIMELI annotation files: format and counts", "",
         "One record per frame at 1 FPS; `id` = seconds since the start of the session video; the instruction for that second is `QA[0].a` "
         "(JSON `{reason, instruction}` for CARLA, plain text for the real set); silence = instruction `None` with reason `remain_silent` / "
         "`remain_silent_in_junction`; `history` = the 15 previous seconds, oldest first. See the module docstring for the generator rules.", "",
         "| split | file | records (frames) | videos | frames / video | spoken share | instr. / min | words / instr. | silence runs (mean s, longest) | gap between instr. (median s) | junction share |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for split, s in summary["splits"].items():
    w = s["instruction_words"]; sr = s["silence_runs"]; g = s["gap_between_instructions_s"]
    lines.append(f"| {split} | {s['file']} ({s['bytes'] / 1e6:.0f} MB) | {s['records']} | {s['videos']} | {s['frames_per_video']['min']}-{s['frames_per_video']['max']} "
                 f"(mean {s['frames_per_video']['mean']}) | {s['spoken_share']} | {s['instructions_per_minute']} | {w['mean']} +/- {w['sd']} (max {w['max']}) | "
                 f"{sr['n']} ({sr['mean_s']}, {sr['longest_s']}) | {g['median'] if g else '-'} | {s['junction_share']} |")
lines += ["", "## Reasons (why the generator spoke or stayed silent)", "", "| split | " + " | ".join(sorted({k for s in summary['splits'].values() for k in s['reasons']}, key=str)) + " |"]
keys = sorted({k for s in summary["splits"].values() for k in s["reasons"]}, key=str)
lines.append("|---|" + "---|" * len(keys))
for split, s in summary["splits"].items():
    lines.append(f"| {split} | " + " | ".join(str(s["reasons"].get(k, 0)) for k in keys) + " |")
lines += ["", "## Fields (first record)", ""]
for split, s in summary["splits"].items():
    lines += [f"### {split}", "", "| field | type | example |", "|---|---|---|"]
    lines += [f"| `{k}` | {v['type']} | `{v['example'].replace('|', '/')}` |" for k, v in s["fields"].items()]
    lines += ["", f"Actions: {s['actions']}. Conveyed obstacles: {s['conveyed_obstacles']} (directions {s['conveyed_directions']}). "
              f"Scene objects (`description_cluster`): {s['description_cluster_objects']}. Detector labels (`object_detection`): {s['object_detection_labels']}. "
              f"Weather: {s['weather']}; town: {s['town']}. Silence runs by length (s): {s['silence_runs']['by_length']}. "
              f"First-frame reason: {s['first_frame_reason']}. Reason / instruction inconsistencies: {s['reason_vs_instruction_inconsistencies'] or 'none'}. "
              f"`history[-1]` equals the previous frame's instruction in {s['history_last_equals_previous_instruction']} of frames.", ""]
(OUT / "inspect.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:8 + len(summary["splits"]) + 6]))
print(f"wrote {OUT / 'inspect.json'} and {OUT / 'inspect.md'}")
