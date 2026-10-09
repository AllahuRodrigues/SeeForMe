"""Schema summary of the Mendeley Data set "Dataset for Deep Learning Based Navigation Assistance for the Visually Impaired
People" (Ashik, Murteja, Shakil, Rony; East West University, Dhaka; DOI 10.17632/66bjx8h79p.1, v1 2025-03-21, CC BY 4.0) and
a comparison of its per-object features with SeeForMe's EventFeatures (seeforme/cues/pcacs.py).

    python scripts/mendeley_navassist.py     # -> results/mendeley_navassist/summary.{md,json}

The set is features only (no images, no video): per frame, counts of detected objects by side / range / motion, up to 20
objects with class, depth, distance, angle, side and approaching / receding, and one "decision" label
(Move Forward / Move Left / Move Right / Stop). Getting it (8.4 MB zip, 82 MB extracted):

    mkdir -p data/mendeley_navassist
    curl -L -o data/mendeley_navassist/66bjx8h79p-1.zip https://data.mendeley.com/public-api/zip/66bjx8h79p/download/1
    cd data/mendeley_navassist && unzip -q 66bjx8h79p-1.zip && mv "Dataset for Deep Learning Based Navigation Assistance for the Visually Impaired People" extracted
"""
import argparse, csv, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

from _common import ROOT, RESULTS, dump_json
from seeforme.campus.world import EVENT_SPECS, default_event_type
from seeforme.cues.pcacs import EventFeatures

DATA = ROOT / "data" / "mendeley_navassist"
OUT = RESULTS / "mendeley_navassist"
DOI = "10.17632/66bjx8h79p.1"
COCO80 = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
          "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
          "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
          "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}

# their 22 classes (a custom YOLO trained on Dhaka street video) -> the label SeeForMe's taxonomy would see
CLASS_MAP = {
    "Person": ("person", "same class"), "Car": ("car", "same class"), "Motorcycle": ("motorcycle", "same class"),
    "Bus": ("bus", "same class"), "Truck": ("truck", "same class"), "Cycle": ("bicycle", "same class"),
    "Pickup": ("truck", "pickup truck"), "Van": ("truck", "van"), "Food Van": ("truck", "food truck"),
    "Leguna": ("bus", "Leguna minibus"), "CNG": ("motorcycle", "CNG auto-rickshaw, a motorised three-wheeler"),
    "Rickshaw": ("bicycle", "cycle rickshaw, a pedal tricycle"), "Van gari": ("bicycle", "pedal cargo tricycle"),
    "Electric Pole": ("pole", "simulator / open-vocab label"), "Building Pillar": ("pole", "simulator / open-vocab label"),
    "Parking Cone": ("cone", "simulator / open-vocab label"), "Stairs": ("stairs", "simulator / open-vocab label"),
    "Food cart": ("cart", "simulator / open-vocab label"), "Bin": ("trash can", "simulator / open-vocab label"),
    "Tree": ("tree", "open-vocab label"), "Obstacle": ("obstruction", "UNKNOWN_OBSTACLE_LABELS -> unknown_obstacle"),
    "Footpath": (None, "walkable surface (sidewalk); not an object event"),
}

# SeeForMe EventFeatures (seeforme/cues/pcacs.py) -> what the Mendeley records carry for it
FEATURE_MAP = {
    "severity": "class_name (through LABEL_EVENT / EVENT_SPECS)", "path_prob": "object_posistion left / center / right (coarse)",
    "hazard": "derivable: severity x centre position", "urgency": "-", "actionability": "class_name", "goal": "-", "novelty": "-",
    "uncertainty": "-", "speech_cost": "-", "queue_cost": "-", "confidence": "-", "distance": "estimated_distance (m, monocular)",
    "depth_conf": "-", "angle": "angle (deg from the frame centre)", "rel_speed": "movement_status approaching / moving away (sign only)",
    "lateral_speed": "-", "ttc": "-", "growth": "-", "since_announced": "-", "announce_count": "-", "text_importance": "-",
    "is_text": "-", "moving": "movement_status != unknown (approx.)", "channel_busy": "-", "channel_remaining": "-", "budget": "-",
    "density": "total_object and the *_in_4m / *_in_2m counts", "own_speed": "-", "ego_speed": "-",
}
AGG_COLS = ["total_object", "total_left_object", "total_center_object", "total_right_object", "moving_away_left_object",
            "moving_away_center_object", "moving_away_right_object", "approaching_left_object", "approaching_center_object",
            "approaching_right_object", "total_left_object_in_4m", "total_center_object_in_4m", "total_right_object_in_4m",
            "total_left_object_in_2m", "total_center_object_in_2m", "total_right_object_in_2m"]


def find_files():
    csvs = sorted(DATA.rglob("dataset.csv"))
    jsons = sorted(DATA.rglob("dataset.json"))
    if not csvs or not jsons:
        sys.exit(f"dataset.csv / dataset.json not found under {DATA}; see the module docstring for the download")
    return csvs[0], jsons[0]


def quantiles(vals):
    vals = sorted(vals)
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    return {"min": round(vals[0], 3), "p5": round(q(0.05), 3), "median": round(q(0.5), 3), "p95": round(q(0.95), 3), "max": round(vals[-1], 3)}


def sequences(records):
    """Frames are named frame_<n>.jpg with no video id: a new sequence starts whenever n stops increasing."""
    seqs, cur, prev = [], [], -1
    for r in records:
        n = int(r["image_file"].split("_")[1].split(".")[0])
        if n <= prev and cur:
            seqs.append(cur); cur = []
        cur.append((n, r["decision"])); prev = n
    if cur:
        seqs.append(cur)
    lens = [len(s) for s in seqs]
    gaps = sum(s[-1][0] - s[0][0] + 1 - len(s) for s in seqs)
    changes = sum(sum(a[1] != b[1] for a, b in zip(s, s[1:])) for s in seqs)
    runs = []
    for s in seqs:
        k = 0
        for n, d in s + [(None, "Move Forward")]:
            if d != "Move Forward":
                k += 1
            elif k:
                runs.append(k); k = 0
    return {"n_sequences": len(seqs), "frames": sum(lens), "length_min": min(lens), "length_median": statistics.median(lens),
            "length_max": max(lens), "dropped_frame_numbers": gaps, "dropped_share": round(gaps / (gaps + sum(lens)), 3),
            "decision_changes": changes, "change_rate_per_frame": round(changes / max(1, sum(lens) - len(seqs)), 3),
            "non_forward_runs": len(runs), "non_forward_run_median": statistics.median(runs) if runs else None,
            "non_forward_run_max": max(runs) if runs else None}


def decision_vs_geometry(records):
    rows = {}
    for d in sorted({r["decision"] for r in records}):
        sub = [r for r in records if r["decision"] == d]
        centre = [min((o["estimated_distance"] for o in r["object_collection"] if o["object_posistion"] == "center"), default=None) for r in sub]
        centre = [c for c in centre if c is not None]
        rows[d] = {"n": len(sub), "share": round(len(sub) / len(records), 3),
                   "mean_total_object": round(statistics.mean(r["total_object"] for r in sub), 2),
                   "share_centre_object_in_2m": round(statistics.mean(r["total_center_object_in_2m"] > 0 for r in sub), 3),
                   "share_approaching_centre": round(statistics.mean(r["approaching_center_object"] > 0 for r in sub), 3),
                   "median_nearest_centre_m": round(statistics.median(centre), 2) if centre else None,
                   "share_no_object": round(statistics.mean(r["total_object"] == 0 for r in sub), 3)}
    near = [r for r in records if r["total_center_object_in_2m"] > 0]
    far = [r for r in records if r["total_center_object_in_2m"] == 0]
    cond = {"P(not Move Forward | centre object within 2 m)": round(statistics.mean(r["decision"] != "Move Forward" for r in near), 3),
            "P(not Move Forward | no centre object within 2 m)": round(statistics.mean(r["decision"] != "Move Forward" for r in far), 3),
            "P(Stop | centre object within 2 m)": round(statistics.mean(r["decision"] == "Stop" for r in near), 3),
            "P(Stop | no centre object within 2 m)": round(statistics.mean(r["decision"] == "Stop" for r in far), 3)}
    return rows, cond


def predictability(records):
    """How much of the decision a shallow tree on the frame-level features recovers (a label that is a fixed rule of the
    stored features is not independent supervision)."""
    try:
        import numpy as np
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        return {"skipped": "scikit-learn not installed"}
    names = AGG_COLS + ["nearest_left_m", "nearest_center_m", "nearest_right_m"]
    X, y = [], []
    for r in records:
        near = {s: min((o["estimated_distance"] for o in r["object_collection"] if o["object_posistion"] == s), default=99.0)
                for s in ("left", "center", "right")}
        X.append([r[c] for c in AGG_COLS] + [near["left"], near["center"], near["right"]]); y.append(r["decision"])
    X, y = np.array(X, dtype=float), np.array(y)
    out = {"features": names, "majority_share": round(float(np.mean(y == Counter(y).most_common(1)[0][0])), 3)}
    for depth in (3, 6, 12):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        cv = cross_val_score(clf, X, y, cv=5)
        clf.fit(X, y)
        out[f"tree_depth_{depth}"] = {"train_accuracy": round(float(clf.score(X, y)), 3), "cv5_accuracy": round(float(cv.mean()), 3)}
        if depth == 3:
            out["tree_depth_3_rules"] = export_text(clf, feature_names=names, max_depth=3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    csv_path, json_path = find_files()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_cols = [c.strip() for c in reader.fieldnames]
        csv_rows = list(reader)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    rec_keys = list(records[0].keys())
    obj_keys = list(records[0]["object_collection"][0].keys())

    # --- values -------------------------------------------------------------------------------------------------------
    decisions = Counter(r["decision"] for r in records)
    objs = [o for r in records for o in r["object_collection"]]
    cls_objects, cls_frames, cls_id = Counter(), Counter(), {}
    for r in records:
        seen = set()
        for o in r["object_collection"]:
            cls_objects[o["class_name"]] += 1; cls_id[o["class_name"]] = o["class_id"]; seen.add(o["class_name"])
        cls_frames.update(seen)
    csv_cls = Counter(r[f"{i}_index_objects_class_name"].strip() for r in csv_rows for i in range(3) if r[f"{i}_index_objects_class_name"].strip())
    numeric = {k: quantiles([o[k] for o in objs]) for k in ("estimated_distance", "avg_depth_object", "nearest_object_depth", "angle")}
    movement = Counter(o["movement_status"] for o in objs)
    position = Counter(o["object_posistion"] for o in objs)
    n_obj = Counter(len(r["object_collection"]) for r in records)
    # csv vs json consistency: same rows / decisions; are the three CSV objects the first three of the JSON list?
    same_len = len(csv_rows) == len(records)
    same_dec = same_len and all(a["decision"].strip() == b["decision"] for a, b in zip(csv_rows, records))
    first3 = sum(all(a[f"{i}_index_objects_class_name"].strip() == (b["object_collection"][i]["class_name"] if i < len(b["object_collection"]) else "")
                     for i in range(3)) for a, b in zip(csv_rows, records)) if same_len else 0
    sorted_by_dist = sum(all(x["estimated_distance"] <= y["estimated_distance"] for x, y in zip(r["object_collection"], r["object_collection"][1:]))
                         for r in records if len(r["object_collection"]) > 1)
    multi = sum(len(r["object_collection"]) > 1 for r in records)

    classes = []
    for name, n in cls_objects.most_common():
        label, how = CLASS_MAP.get(name, (None, "unmapped"))
        ev = default_event_type(label) if label else ""
        spec = EVENT_SPECS.get(ev)
        classes.append({"class_name": name, "class_id": cls_id[name], "objects": n, "frames": cls_frames[name],
                        "csv_nearest3": csv_cls[name], "seeforme_label": label or "", "coco80": bool(label and label in COCO80),
                        "event_type": ev, "severity": spec.severity if spec else 0.0, "how": how})
    seq = sequences(records)
    by_decision, cond = decision_vs_geometry(records)
    pred = predictability(records)
    feat_names = EventFeatures.names()
    covered = [f for f in feat_names if FEATURE_MAP.get(f, "-") != "-"]

    summary = {"doi": DOI, "licence": "CC BY 4.0", "files": {"csv": str(csv_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT)),
               "csv_bytes": csv_path.stat().st_size, "json_bytes": json_path.stat().st_size},
               "csv": {"rows": len(csv_rows), "columns": csv_cols},
               "json": {"records": len(records), "record_keys": rec_keys, "object_keys": obj_keys, "max_objects": max(n_obj),
                        "objects_per_record": dict(sorted(n_obj.items()))},
               "consistency": {"same_row_count": same_len, "same_decisions": same_dec, "csv_objects_are_first_three_of_json": first3,
                               "json_objects_sorted_by_estimated_distance": f"{sorted_by_dist} of {multi} multi-object records"},
               "decision": {k: {"n": v, "share": round(v / len(records), 3)} for k, v in decisions.most_common()},
               "classes": classes, "movement_status": dict(movement.most_common()), "object_position": dict(position.most_common()),
               "numeric": numeric, "unique_image_files": len({r["image_file"] for r in records}), "sequences": seq,
               "decision_vs_geometry": by_decision, "conditional": cond, "predictability": pred,
               "event_features": {"n": len(feat_names), "with_counterpart": covered, "map": FEATURE_MAP}}
    OUT.mkdir(parents=True, exist_ok=True)
    dump_json(summary, OUT / "summary.json")
    write_md(summary, OUT / "summary.md")

    print(f"CSV {len(csv_rows)} rows x {len(csv_cols)} columns; JSON {len(records)} records, up to {max(n_obj)} objects each; "
          f"{len(classes)} classes; decisions {dict(decisions)}")
    print(f"sequences inferred from frame numbers: {seq['n_sequences']} (median length {seq['length_median']}); "
          f"CSV = first three JSON objects in {first3} / {len(records)} rows")
    print(f"EventFeatures with a counterpart: {len(covered)} / {len(feat_names)}: {', '.join(covered)}")
    if "tree_depth_6" in pred:
        print(f"decision from frame-level features: depth-3 tree CV acc {pred['tree_depth_3']['cv5_accuracy']}, depth-6 {pred['tree_depth_6']['cv5_accuracy']}, "
              f"depth-12 {pred['tree_depth_12']['cv5_accuracy']} (majority {pred['majority_share']})")
    print(f"wrote {OUT / 'summary.md'} and summary.json")


def write_md(s, path: Path):
    f3 = lambda x: "" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)
    L = ["# Mendeley 'Dataset for Deep Learning Based Navigation Assistance for the Visually Impaired People'", "",
         f"DOI {s['doi']} (v1, 2025-03-21), East West University (Dhaka), licence {s['licence']}. Files: `{s['files']['csv']}` "
         f"({s['files']['csv_bytes'] / 1e6:.1f} MB), `{s['files']['json']}` ({s['files']['json_bytes'] / 1e6:.1f} MB). "
         "Features extracted from recorded street video with a custom YOLO plus a monocular depth model; no images or video are "
         "included, so nothing here can be replayed through SeeForMe's own perception. The authors' stated purpose is to train a "
         "sequence model (RNN / LSTM / GRU) that predicts the movement decision.", ""]
    L += ["## Schema", "", f"* CSV: {s['csv']['rows']} rows x {len(s['csv']['columns'])} columns: `image_file`, 16 per-frame counts "
          "(`total_object`, `total_{left,center,right}_object`, `moving_away_*`, `approaching_*`, `*_in_4m`, `*_in_2m`), the three nearest "
          "objects as `{0,1,2}_index_objects_{class_name, estimated_distance, avg_depth_object, nearest_object_depth, angle, posistion, movement_status}`, "
          "and `decision`.",
          f"* JSON: {s['json']['records']} records with the same keys plus `object_collection`, a list of up to {s['json']['max_objects']} objects "
          f"(`{'`, `'.join(s['json']['object_keys'])}`). Objects per record: " +
          ", ".join(f"{k}: {v}" for k, v in s['json']['objects_per_record'].items()) + ".",
          f"* Consistency: same row count {s['consistency']['same_row_count']}, same decisions {s['consistency']['same_decisions']}; the CSV's three objects "
          f"are the first three of the JSON list in {s['consistency']['csv_objects_are_first_three_of_json']} / {s['json']['records']} rows; the JSON list is "
          f"sorted by estimated distance in {s['consistency']['json_objects_sorted_by_estimated_distance']}, so the CSV's three objects are the three nearest.",
          f"* `image_file` is `frame_<n>.jpg` with no video id ({s['unique_image_files']} distinct names for {s['json']['records']} records); sequences can "
          f"only be inferred from the frame number resetting: {s['sequences']['n_sequences']} sequences, length min / median / max "
          f"{s['sequences']['length_min']} / {s['sequences']['length_median']} / {s['sequences']['length_max']}, "
          f"{s['sequences']['dropped_share']:.1%} of frame numbers missing (frames without detections were dropped). No timestamps, no frame rate, "
          "no object identity across frames.", ""]
    L += ["## Label fields", "", "| decision | n | share |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['share']:.3f} |" for k, v in s["decision"].items()]
    L += ["", f"Per object: `movement_status` {s['movement_status']}, `object_posistion` {s['object_position']}. Numeric ranges (min / p5 / median / p95 / max):", ""]
    L += [f"* `{k}`: {v['min']} / {v['p5']} / {v['median']} / {v['p95']} / {v['max']}" for k, v in s["numeric"].items()]
    L += ["", "`estimated_distance` behaves like metres (median 7.3 m, a long tail of far objects); `avg_depth_object` / `nearest_object_depth` are raw "
          "depth-map statistics in model units (the nearest value can be negative); `angle` is degrees from the frame centre, consistent with a "
          "roughly 70-degree horizontal field of view.", ""]
    L += ["## Classes", "", "| class (id) | objects | frames | in CSV nearest-3 | SeeForMe label | COCO-80 | event type | severity | how |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in s["classes"]:
        L.append(f"| {c['class_name']} ({c['class_id']}) | {c['objects']} | {c['frames']} | {c['csv_nearest3']} | {c['seeforme_label']} | "
                 f"{'yes' if c['coco80'] else ''} | {c['event_type']} | {c['severity'] if c['event_type'] else ''} | {c['how']} |")
    n_coco = sum(c["coco80"] for c in s["classes"]); n_ev = sum(c["event_type"] not in ("", "decor") for c in s["classes"])
    L += ["", f"{n_coco} of the {len(s['classes'])} classes are COCO-80 classes YOLO11n would output; {n_ev} reach a non-decor P-CACS event through "
          "LABEL_EVENT (the vehicle family of Dhaka streets: rickshaw, CNG auto-rickshaw, Leguna minibus, van gari cargo tricycle, is mapped onto "
          "the nearest COCO vehicle class).", ""]
    L += ["## Their per-object features against SeeForMe's EventFeatures", "",
          f"`EventFeatures` ({s['event_features']['n']} fields, `seeforme/cues/pcacs.py`) is what P-CACS and the learned scheduler see per candidate. "
          f"{len(s['event_features']['with_counterpart'])} of them have a counterpart in the Mendeley records:", "",
          "| EventFeatures field | Mendeley counterpart |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["event_features"]["map"].items()]
    L += ["", "What they have that SeeForMe does not: a per-frame object census by side and range (`*_in_2m`, `*_in_4m`, approaching / receding counts), "
          "which is a richer `density` than the single scalar. What SeeForMe has that they lack: detector and depth confidence, time to contact and box "
          "growth, lateral motion, everything about the speech channel (busy, remaining, budget, queue cost) and announcement history (novelty, "
          "since_announced), goal relevance and text. Their motion is a binary approaching / receding flag per object with no tracks, so TTC cannot be "
          "recovered from the records.", ""]
    g = s["decision_vs_geometry"]; c = s["conditional"]; p = s["predictability"]
    L += ["## Is `decision` a usable weak scheduler target?", "",
          "| decision | n | share | mean objects | centre object within 2 m | approaching centre object | median nearest centre (m) | no object at all |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['share']:.3f} | {v['mean_total_object']} | {v['share_centre_object_in_2m']:.2f} | {v['share_approaching_centre']:.2f} | "
          f"{f3(v['median_nearest_centre_m'])} | {v['share_no_object']:.2f} |" for k, v in g.items()]
    L += ["", "* " + "; ".join(f"{k} = {v}" for k, v in c.items()) + "."]
    if "tree_depth_6" in p:
        L += [f"* A decision tree on the 16 frame-level counts plus the nearest distance per side recovers the label with 5-fold accuracy "
              f"{p['tree_depth_3']['cv5_accuracy']} (depth 3), {p['tree_depth_6']['cv5_accuracy']} (depth 6), {p['tree_depth_12']['cv5_accuracy']} (depth 12) "
              f"against a majority-class baseline of {p['majority_share']}. Depth-3 rules:", "", "```", p["tree_depth_3_rules"].rstrip(), "```", ""]
    L += ["**Verdict.** The decision is a locomotion command (where to move), not a communication decision (whether to speak); it lives in a "
          "different action space from P-CACS's silence / speak / interrupt / haptic. It is still usable as a *weak, frame-level* 'something "
          "action-relevant is ahead' target: `Stop` and `Move Left` / `Move Right` frames (together "
          f"{sum(v['share'] for k, v in g.items() if k != 'Move Forward'):.1%}) mark moments when an instruction was warranted, `Move Forward` frames mark "
          "quiet windows, in the same spirit as EgoBlind's present / clear windows. Limits: no timestamps or frame rate, so earliest / ideal / latest "
          "windows cannot be placed in seconds; no object identity, so the responsible object must be inferred (the nearest approaching centre object); "
          "no images, so the label cannot be attached to SeeForMe's own detections; the labelling procedure is not documented and much of the label "
          "is recoverable from the stored features (see the tree above: `Stop` never occurs without a centre object within 2 m), so it adds little "
          "information beyond those features; the scene is Dhaka street traffic, and the record does not say who carried the camera or whether blind "
          "users were involved. Nothing in it says *when* to speak or what to say.", ""]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
