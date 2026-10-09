"""Map the 90-object BLV-Road-Nav-Accessibility taxonomy (Islam et al., ASSETS 2024, arXiv 2407.16777) onto
(a) the COCO-80 classes YOLO11n can output and (b) SeeForMe's event types (LABEL_EVENT / EVENT_SPECS), then score
YOLO11n presence detection against the repo's labelled keyframes.

    python scripts/blvnav_taxonomy.py                  # -> results/blvnav/taxonomy_map.{json,md}, detector_eval.{json,md}
    python scripts/blvnav_taxonomy.py --skip-detector  # taxonomy map only (no model, no frames needed)
    python scripts/blvnav_taxonomy.py --reuse          # re-score from results/blvnav/detections.jsonl without the model

The repo (external/BLV-Road-Nav-Accessibility, git clone https://github.com/Shohan29531/BLV-Road-Nav-Accessibility)
holds the 90-object list, 1,795 keyframes (JPEG, mostly 1280x720) from 83 segments of 21 YouTube / Vimeo videos of
blind pedestrians, and presence labels (object x keyframe, 0 / 1) for 31 of the segments over the paper's earlier
79-object list. The videos themselves are not in the repo and no URLs are published; nothing is downloaded here.
"""
import argparse, csv, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

from _common import ROOT, RESULTS, dump_json, load_cfg
from seeforme.campus.world import EVENT_SPECS, LABEL_EVENT, default_event_type
try:   # open-vocabulary (YOLO-World) prompts -> labels, when the perception package provides them
    from seeforme.perception.detector import NAV_VOCAB
except ImportError:
    NAV_VOCAB = []
PROMPTS = defaultdict(list)
for _prompt, _label in NAV_VOCAB:
    PROMPTS[_label].append(_prompt)

REPO = ROOT / "external" / "BLV-Road-Nav-Accessibility"
OUT = RESULTS / "blvnav"
PAPER = ("Islam, Kabir, Pearce, Reza, Billah: A Dataset for Crucial Object Recognition in Blind and Low-Vision "
         "Individuals' Navigation, ASSETS 2024 (arXiv 2407.16777)")

# Table 3 of the paper. Groups 3, 5 and 7 are the ones the focus group rated most significant ("failure in detecting
# these objects can cause accident or injury").
GROUPS = {1: "Attributes of a sidewalk and driveway", 2: "Obstructions likely to be detected by a white cane",
          3: "Obstructions less likely to be detected by a white cane",
          4: "Objects that are too late to be detected by a white cane", 5: "Objects that pick you before you pick them",
          6: "Objects that provide navigational guidance", 7: "Objects not supposed to be on the sidewalk",
          8: "Moving objects sharing the sidewalk", 9: "Intersection", 10: "Objects on the road shoulder",
          11: "Objects on the road", 12: "Traffic signals and street signs",
          13: "Objects related to building exits and entrances", 14: "Indoor objects", 15: "Objects related to public transit"}
HIGH_PRIORITY = (3, 5, 7)
DECOR = ("decor", "sign_decor")

# One entry per line of Accessibility-related-objects.txt: (group, coco, match, via, analogue, ocr, note)
#   group     Table 3 group id (Clear Ice is in the 90-list but missing from the table as published; it was added after
#             the focus group as "black ice" and is placed with the other surface conditions of group 1)
#   coco      COCO-80 class names YOLO11n outputs for the object; match = "direct" (same object class) or "proxy" (a class
#             that fires on the object without naming it: a guide dog is a dog, the person in a wheelchair is a person)
#   via       a LABEL_EVENT key that is NOT a COCO class: a simulator label ('wet floor sign', 'scooter') or an
#             open-vocabulary label that NAV_VOCAB (YOLO-World) can produce ('stairs', 'curb', 'pole', 'fence', ...). The
#             P-CACS taxonomy has an event for it; the COCO detector cannot produce it
#   analogue  an event type assigned here by analogy when neither of the above applies
#   ocr       the object is a text sign that SeeForMe's OCR path turns into sign_nav / sign_decor
MAP = {
    # group 1
    "Accent Paving": (1, [], "", None, None, False, "surface attribute"),
    "Driveway (flat)": (1, [], "", None, None, False, "participants: the driveway ahead matters, its slope does not"),
    "Puddle": (1, [], "", "ground hazard", None, False, "open-vocab prompt 'puddle'"),
    "Raised Entryway": (1, [], "", None, "curb", False, "a step up at an entrance: closest spec is curb"),
    "Sidewalk": (1, [], "", None, None, False, "walkable surface, not an event"),
    "Sidewalk Pits": (1, [], "", "ground hazard", None, False, "open-vocab prompt 'pothole'"),
    "Sloped Driveway": (1, [], "", None, None, False, ""),
    "Tactile Paving": (1, [], "", None, None, False, "guidance surface"),
    "Brick Paving": (1, [], "", None, None, False, ""),
    "Cobblestone Paving": (1, [], "", None, None, False, ""),
    "Unpaved Sidewalk": (1, [], "", None, None, False, ""),
    "Wet Surface": (1, [], "", None, None, False, "surface condition; only the warning sign ('wet floor sign' -> barrier) is in LABEL_EVENT"),
    "Clear Ice": (1, [], "", None, None, False, "not in Table 3 as published; surface condition"),
    # group 2
    "Fire hydrant": (2, ["fire hydrant"], "direct", None, None, False, ""),
    "Gutter": (2, [], "", None, None, False, ""),
    "Vegetation": (2, [], "", None, None, False, "'potted plant' does not fire on street vegetation"),
    "Tree": (2, [], "", "tree", None, False, "open-vocab prompt 'tree trunk'"),
    "Brick Wall": (2, [], "", None, None, False, ""),
    "Fence": (2, [], "", "fence", None, False, ""),
    "Trash Bins": (2, [], "", "trash can", None, False, "COCO-80 has no trash can"),
    "Lamp Post": (2, [], "", "pole", None, False, ""),
    "Pole": (2, [], "", "pole", None, False, ""),
    "Mailbox": (2, [], "", None, "hazard_static", False, "sidewalk obstruction like a pole; no prompt"),
    # group 3 (high priority)
    "Closed Sidewalk": (3, [], "", "barrier", None, True, "barricade plus sign: the text is readable by OCR"),
    "Barrier Post": (3, [], "", "pole", None, False, "open-vocab prompt 'bollard'"),
    "Barrier Stump": (3, [], "", "pole", None, False, "short bollard; open-vocab prompt 'bollard'"),
    "Foldout Sign": (3, [], "", "sign", None, True, "A-frame sign: the 'sign' prompt reaches it only as a landmark (0.4); the simulator's 'wet floor sign' label (barrier, 0.75) is the better spec"),
    "Bench": (3, ["bench"], "direct", None, None, False, ""),
    # group 4
    "Train Tracks": (4, [], "", None, None, False, "'train' fires on a train, never on the track"),
    "Train Platform": (4, [], "", None, "curb", False, "platform edge: closest spec is curb, which understates the drop"),
    # group 5 (high priority)
    "Overhanging Tree Branches": (5, [], "", None, "hazard_static", False, "head-height obstruction; 'tree trunk' does not fire on branches; no event class of its own"),
    # group 6
    "Retaining Wall": (6, [], "", None, None, False, "guidance surface"),
    "Railing": (6, [], "", "fence", None, False, "open-vocab prompt 'railing'"),
    "Wall": (6, [], "", None, None, False, ""),
    "Curved Railing": (6, [], "", "fence", None, False, "open-vocab prompt 'railing'"),
    # group 7 (high priority)
    "Hose": (7, [], "", None, "unknown_obstacle", False, "an open-vocabulary 'obstruction' label would reach unknown_obstacle; no prompt today"),
    "Maintenance Vehicle": (7, ["truck", "car"], "proxy", None, None, False, "detected as a truck or car"),
    "Trash on Roads": (7, [], "", None, "unknown_obstacle", False, "UNKNOWN_OBSTACLE_LABELS has 'debris'; no prompt today"),
    "Snow": (7, [], "", None, None, False, "surface condition"),
    "Water Leakage": (7, [], "", None, "unknown_obstacle", False, "UNKNOWN_OBSTACLE_LABELS has 'spill'; no prompt today"),
    "Yard Waste": (7, [], "", None, "unknown_obstacle", False, "UNKNOWN_OBSTACLE_LABELS has 'debris'; no prompt today"),
    "Water Pipes": (7, [], "", None, "unknown_obstacle", False, "no prompt today"),
    # group 8
    "Person": (8, ["person"], "direct", None, None, False, "person_moving once the track moves"),
    "Bicycle": (8, ["bicycle"], "direct", None, None, False, ""),
    "Wheelchair": (8, ["person"], "proxy", None, None, False, "the user is detected as a person; the wheelchair has no class"),
    "Person with a Disability": (8, ["person"], "proxy", None, None, False, ""),
    "White Cane": (8, [], "", None, None, False, "identifies a blind pedestrian; not an obstacle"),
    "Dog": (8, ["dog"], "direct", None, None, False, "the COCO label 'dog' is decor; the open-vocab route maps dog -> 'animal' -> person_moving"),
    "Guide Dog": (8, ["dog"], "proxy", None, None, False, "detected as a dog (decor on the COCO route)"),
    "Street Vendor": (8, ["person"], "proxy", None, None, False, "the vendor is a person; the stall has no class"),
    # group 9
    "Pedestrian Crossing": (9, [], "", "crossing", None, False, "open-vocab prompt 'crosswalk'; the signal is 'Traffic Signals'"),
    "Sloped Curb": (9, [], "", "curb", None, False, "curb cut"),
    "Intersection": (9, [], "", None, None, False, ""),
    "Crosswalk": (9, [], "", "crossing", None, False, ""),
    "Curb": (9, [], "", "curb", None, False, ""),
    "Bridge": (9, [], "", None, None, False, ""),
    "Uncontrolled Crossing": (9, [], "", None, None, False, ""),
    # group 10
    "Road Shoulder": (10, [], "", None, None, False, ""),
    "Roadside Parking": (10, ["car"], "proxy", None, None, False, "parked cars are detected as cars; the parking zone is not"),
    "Parallel Parking Spot": (10, [], "", None, None, False, ""),
    "Paratransit Vehicle": (10, ["bus", "truck", "car"], "proxy", None, None, False, "detected as a bus, truck or car"),
    # group 11
    "Road": (11, [], "", None, None, False, ""),
    "Unpaved Road": (11, [], "", None, None, False, ""),
    "Bus": (11, ["bus"], "direct", None, None, False, ""),
    "Car": (11, ["car"], "direct", None, None, False, ""),
    "Motorcycle": (11, ["motorcycle"], "direct", None, None, False, ""),
    "Road Divider": (11, [], "", "barrier", None, False, "median / jersey barrier"),
    # group 12
    "Traffic Signals": (12, ["traffic light"], "direct", None, None, False, "COCO class with no LABEL_EVENT entry: decor; SeeForMe has no crossing-signal event"),
    "Stop Sign": (12, ["stop sign"], "direct", None, None, True, "the YOLO label maps to decor; the text STOP goes through the OCR path"),
    "Sign": (12, [], "", "sign", None, True, ""),
    "Sign Post": (12, [], "", "pole", None, True, "the post is a pole-type obstruction; the face is read by OCR"),
    "Push Button": (12, [], "", None, None, False, "pedestrian signal button; a FIND-mode goal rather than a hazard"),
    '"Use the Other Door" Sign': (12, [], "", "sign", None, True, ""),
    "Toilet Sign": (12, [], "", "sign", None, True, "pictogram plus text; a FIND-mode goal"),
    # group 13
    "Gate": (13, [], "", None, "door", False, "the 'door' prompt may or may not fire on a gate"),
    "Flush Door": (13, [], "", "door", None, False, ""),
    "Doorway": (13, [], "", "door", None, False, ""),
    # group 14
    "Counter": (14, ["dining table"], "proxy", None, None, False, "'dining table' fires on some counters"),
    "Elevator": (14, [], "", "elevator", None, False, ""),
    "Escalator": (14, [], "", "stairs", None, False, "open-vocab prompt 'escalator'"),
    "Fountain": (14, [], "", None, "hazard_static", False, ""),
    "Stairs": (14, [], "", "stairs", None, False, "severity 1.0 in EVENT_SPECS, no COCO class"),
    "Uneven Stairs": (14, [], "", "stairs", None, False, ""),
    "Table": (14, ["dining table"], "direct", None, None, False, ""),
    "Building": (14, [], "", None, None, False, ""),
    "Moving Walk": (14, [], "", None, "stairs", False, "moving surface; closest spec"),
    "Pillar": (14, [], "", "pole", None, False, ""),
    "Chair": (14, ["chair"], "direct", None, None, False, ""),
    # group 15
    "Bus Stop": (15, [], "", "sign", None, True, "the sign is readable by OCR; a FIND-mode goal"),
    "Turnstile": (15, [], "", "machine", None, False, "open-vocab prompt 'turnstile'"),
}
STATUSES = ("detected_and_scheduled", "detected_but_decor", "open_vocab", "taxonomy_only", "analogue", "ocr_only", "no_class_no_event")
STATUS_DESC = {
    "detected_and_scheduled": "YOLO11n outputs a class for it and LABEL_EVENT gives that class a non-decor event",
    "detected_but_decor": "YOLO11n outputs a class for it, but LABEL_EVENT treats that class as decor (never spoken)",
    "open_vocab": "no COCO class, but a NAV_VOCAB prompt of the open-vocabulary detector (YOLO-World) produces a LABEL_EVENT label for it",
    "taxonomy_only": "an event exists for a simulator label (wet floor sign, scooter, ...) that no detector produces",
    "analogue": "no label at all; an event type is assigned here by analogy",
    "ocr_only": "a text sign: reachable through the OCR path, invisible to the detector",
    "no_class_no_event": "neither a detector class nor an event type",
}


def canon(name: str) -> str:
    """Key for matching the 79-object CSV spellings to the 90-object list ('Slopped Curb', 'Mail box', 'Driveway(flat)')."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return {"sloppedcurb": "slopedcurb"}.get(key, key)


def load_objects():
    names = [l.strip() for l in (REPO / "Accessibility-related-objects.txt").read_text().splitlines() if l.strip()]
    missing, extra = [n for n in names if n not in MAP], [n for n in MAP if n not in names]
    if missing or extra:
        sys.exit(f"MAP out of sync with Accessibility-related-objects.txt: missing {missing}, extra {extra}")
    return names


def classify(name: str):
    g, coco, match, via, analogue, ocr, note = MAP[name]
    detector_event = default_event_type(coco[0]) if coco else ""
    if coco and detector_event not in DECOR:
        event, source, status = detector_event, f"LABEL_EVENT['{coco[0]}']", "detected_and_scheduled"
    elif via:
        event, source = LABEL_EVENT[via], f"LABEL_EVENT['{via}'] (not a COCO class)"
        status = "detected_but_decor" if coco else ("open_vocab" if PROMPTS.get(via) else "taxonomy_only")
    elif analogue:
        event, source, status = analogue, "analogue (assigned here)", "detected_but_decor" if coco else "analogue"
    elif ocr:
        event, source, status = "sign_nav", "OCR text path", "detected_but_decor" if coco else "ocr_only"
    elif coco:
        event, source, status = detector_event, f"LABEL_EVENT default for '{coco[0]}'", "detected_but_decor"
    else:
        event, source, status = "", "none", "no_class_no_event"
    spec = EVENT_SPECS.get(event)
    prompts = sorted({p for lab in ([via] if via else []) + coco for p in PROMPTS.get(lab, [])})
    return {"object": name, "group": g, "group_name": GROUPS[g], "high_priority": g in HIGH_PRIORITY,
            "coco_labels": coco, "coco_match": match or "none", "open_vocab_prompts": prompts, "detector_event": detector_event,
            "event_type": event, "event_source": source, "severity": spec.severity if spec else 0.0,
            "urgency": spec.urgency if spec else "none", "ocr": ocr, "status": status, "note": note}


def summarise(objs):
    def block(rows):
        c = Counter(o["status"] for o in rows)
        return {"n": len(rows), "coco_direct": sum(o["coco_match"] == "direct" for o in rows),
                "coco_proxy": sum(o["coco_match"] == "proxy" for o in rows),
                "detectable": sum(o["coco_match"] != "none" for o in rows),
                "detected_and_scheduled": c["detected_and_scheduled"], "detected_but_decor": c["detected_but_decor"],
                "open_vocab": c["open_vocab"], "taxonomy_only": c["taxonomy_only"], "analogue": c["analogue"],
                "ocr_only": c["ocr_only"], "no_class_no_event": c["no_class_no_event"],
                "open_vocab_reachable": sum(bool(o["open_vocab_prompts"]) for o in rows),
                "ocr_reachable": sum(o["ocr"] for o in rows),
                "has_event_type": sum(o["event_type"] not in ("",) + DECOR for o in rows)}
    out = {"all": block(objs), "high_priority_3_5_7": block([o for o in objs if o["high_priority"]]),
           "by_group": {g: {"name": GROUPS[g], **block([o for o in objs if o["group"] == g])} for g in GROUPS}}
    return out


def write_taxonomy_md(objs, summ, path: Path):
    L = ["# BLV-Road-Nav-Accessibility: the 90 objects against YOLO11n (COCO-80) and the P-CACS event taxonomy", "",
         f"Source: {PAPER}. Groups are Table 3 of the paper; groups 3, 5 and 7 are the focus group's most significant objects.",
         "YOLO11n = the detector in `seeforme/perception/detector.py` (COCO-80 classes); events = `LABEL_EVENT` / `EVENT_SPECS` in",
         "`seeforme/campus/world.py`. *direct* = the COCO class is the object; *proxy* = a COCO class fires on the object without naming it.", ""]
    a, h = summ["all"], summ["high_priority_3_5_7"]
    L += ["## Headline", "",
          f"* detectable by YOLO11n at all: **{a['detectable']} / 90** ({a['coco_direct']} direct, {a['coco_proxy']} proxy); "
          f"of these {a['detected_and_scheduled']} get a non-decor event and {a['detected_but_decor']} are decor in LABEL_EVENT",
          f"* high-priority groups 3, 5, 7: **{h['detectable']} / {h['n']}** detectable ({h['detected_and_scheduled']} scheduled); "
          f"{h['open_vocab']} reachable only through an open-vocabulary prompt, {h['taxonomy_only']} have an event for a label no detector "
          f"produces, {h['analogue']} by analogy, {h['no_class_no_event']} nothing",
          f"* reachable through a NAV_VOCAB prompt of the open-vocabulary detector (YOLO-World; {len(NAV_VOCAB)} prompts): "
          f"**{a['open_vocab_reachable']} / 90** ({h['open_vocab_reachable']} / {h['n']} high priority), of which {a['open_vocab']} have no COCO class at all",
          f"* taxonomy-only (event exists for a simulator label no detector produces): {a['taxonomy_only']}; analogue only: {a['analogue']}; "
          f"OCR only: {a['ocr_only']}; no class and no event: {a['no_class_no_event']}",
          f"* reachable through the OCR text path (signs): {a['ocr_reachable']}", ""]
    L += ["## Per group", "", "| group | parent concept | n | YOLO direct | YOLO proxy | scheduled | detected, decor | open-vocab only | taxonomy only | analogue | OCR only | nothing |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, b in list(summ["by_group"].items()) + [("3+5+7", {"name": "high priority", **h})]:
        tag = f"**{g}**" if g in HIGH_PRIORITY else str(g)
        L.append(f"| {tag} | {b['name']} | {b['n']} | {b['coco_direct']} | {b['coco_proxy']} | {b['detected_and_scheduled']} | "
                 f"{b['detected_but_decor']} | {b['open_vocab']} | {b['taxonomy_only']} | {b['analogue']} | {b['ocr_only']} | {b['no_class_no_event']} |")
    L += ["", "Status meanings: " + "; ".join(f"*{k}* = {v}" for k, v in STATUS_DESC.items()), ""]
    L += ["## All 90 objects", "", "| # | object | group | COCO class(es) | match | open-vocab prompt(s) | event type | via | severity | status | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, o in enumerate(objs, 1):
        g = f"**{o['group']}**" if o["high_priority"] else str(o["group"])
        L.append(f"| {i} | {o['object']} | {g} | {', '.join(o['coco_labels']) or ''} | {o['coco_match']} | {', '.join(o['open_vocab_prompts'])} | "
                 f"{o['event_type'] or ''} | {o['event_source'] if o['event_source'] != 'none' else ''} | {o['severity'] if o['event_type'] else ''} | "
                 f"{o['status']} | {o['note']} |")
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# keyframe presence labels and the detector
# ---------------------------------------------------------------------------------------------------------------------

def load_labels(names):
    """segment -> (frame indices, {object: [0/1/None per frame]}); CSV spellings are mapped onto the 90-object names."""
    by_key = {canon(n): n for n in names}
    segs, odd = {}, Counter()
    for f in sorted((REPO / "Ground Truth Annotation").glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].strip()]
        idx = [int(re.search(r"(\d+)", h).group(1)) for h in rows[0][1:] if h.strip()]
        table = {}
        for r in rows[1:]:
            name = by_key.get(canon(r[0]))
            if name is None:
                odd[f"unknown object {r[0]!r}"] += 1
                continue
            vals = []
            for v in (r[1:1 + len(idx)] + [""] * len(idx))[:len(idx)]:
                v = v.strip()
                if v in ("0", "1"):
                    vals.append(int(v))
                else:
                    vals.append(None); odd[f"cell value {v!r}"] += 1
            table[name] = vals
        segs[f.stem] = (idx, table)
    return segs, odd


def run_detector(segs, det_path: Path, conf_floor: float):
    """YOLO11n over every labelled keyframe; one JSON line per frame with the raw detections."""
    import cv2
    from seeforme.perception import YoloDetector
    cfg = load_cfg()
    cfg.perception.yolo_conf = min(cfg.perception.yolo_conf, conf_floor)
    det = YoloDetector(cfg.perception)
    todo = [(seg, i, REPO / "Video Frames" / f"{seg}-frame-{i}.jpeg") for seg, (idx, _) in segs.items() for i in idx]
    missing = [p.name for _, _, p in todo if not p.exists()]
    t0 = time.perf_counter()
    with open(det_path, "w") as fh:
        for k, (seg, i, p) in enumerate(todo):
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            dets = det.detect(img)
            fh.write(json.dumps({"segment": seg, "frame": i, "file": p.name, "w": img.shape[1], "h": img.shape[0],
                                 "dets": [[d.label, round(d.confidence, 4)] + [round(x, 4) for x in d.bbox] for d in dets]}) + "\n")
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(todo)} frames, {(time.perf_counter() - t0) / (k + 1) * 1000:.0f} ms/frame", flush=True)
    return {"model": cfg.perception.yolo_model, "device": det.device, "imgsz": cfg.perception.yolo_imgsz,
            "frames": len(todo) - len(missing), "missing_files": missing,
            "seconds": round(time.perf_counter() - t0, 1)}


def score(segs, det_path: Path, objs, thresholds):
    maxconf = {}   # (segment, frame) -> {label: max confidence}
    with open(det_path) as fh:
        for line in fh:
            r = json.loads(line)
            mc = defaultdict(float)
            for lab, c, *_ in r["dets"]:
                mc[lab] = max(mc[lab], c)
            maxconf[(r["segment"], r["frame"])] = mc
    mappable = [o for o in objs if o["coco_labels"]]
    present_in = Counter()
    for seg, (idx, table) in segs.items():
        for name, vals in table.items():
            present_in[name] += sum(v == 1 for v in vals)
    out = {"thresholds": {}, "frames_scored": len(maxconf), "segments": len(segs),
           "annotated_objects": sorted({n for _, t in segs.values() for n in t}),
           "frames_present": {n: present_in[n] for n in sorted(present_in)}}
    for thr in thresholds:
        per = {}
        micro = {"direct": Counter(), "mappable": Counter(), "all_annotated": Counter(), "all_annotated_direct": Counter()}
        hp_frames = hp_hit = 0
        for o in mappable:
            name, labs = o["object"], o["coco_labels"]
            c = Counter()
            for seg, (idx, table) in segs.items():
                vals = table.get(name)
                if vals is None:
                    continue
                for i, v in zip(idx, vals):
                    if v is None or (seg, i) not in maxconf:
                        continue
                    pred = max((maxconf[(seg, i)].get(l, 0.0) for l in labs), default=0.0) >= thr
                    c["tp" if (v and pred) else "fp" if pred else "fn" if v else "tn"] += 1
            if sum(c.values()) == 0:
                per[name] = {"annotated": False}
                continue
            per[name] = {"annotated": True, "group": o["group"], "coco_labels": labs, "match": o["coco_match"], **counts(c),
                         "support": c["tp"] + c["fn"], **prf(c)}
            for k in ("mappable",) + (("direct",) if o["coco_match"] == "direct" else ()):
                micro[k].update(c)
        # paper-style micro averages: every annotated object counts and the detector predicts 0 for the objects it has no
        # class for; "all_annotated" uses direct + proxy classes, "all_annotated_direct" only the 12 direct ones (the
        # protocol of Table 5 in the paper, N = 12)
        mapped = {o["object"]: o for o in mappable}
        direct = {n for n, o in mapped.items() if o["coco_match"] == "direct"}
        for seg, (idx, table) in segs.items():
            for name, vals in table.items():
                for i, v in zip(idx, vals):
                    if v is None or (seg, i) not in maxconf:
                        continue
                    if name not in mapped:
                        micro["all_annotated"]["fn" if v else "tn"] += 1
                    if name not in direct:
                        micro["all_annotated_direct"]["fn" if v else "tn"] += 1
        micro["all_annotated"].update(micro["mappable"])
        micro["all_annotated_direct"].update(micro["direct"])
        # frame-level recall on the high-priority groups: a labelled frame containing any group 3/5/7 object counts as a
        # hit when the detector fires a mapped class for at least one of the objects present
        for seg, (idx, table) in segs.items():
            for j, i in enumerate(idx):
                if (seg, i) not in maxconf:
                    continue
                present = [n for n, vals in table.items() if vals[j] == 1 and MAP[n][0] in HIGH_PRIORITY]
                if not present:
                    continue
                hp_frames += 1
                hp_hit += any(n in mapped and max((maxconf[(seg, i)].get(l, 0.0) for l in mapped[n]["coco_labels"]), default=0.0) >= thr
                              for n in present)
        out["thresholds"][str(thr)] = {"per_object": per, "micro": {k: {**counts(v), **prf(v)} for k, v in micro.items()},
                                       "high_priority_frames": {"frames_with_group_3_5_7_object": hp_frames, "frames_with_a_hit": hp_hit,
                                                                "recall": round(hp_hit / hp_frames, 3) if hp_frames else None}}
    return out


def counts(c):
    return {k: c[k] for k in ("tp", "fp", "fn", "tn")}


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": None if p is None else round(p, 3), "recall": None if r is None else round(r, 3),
            "f1": None if f is None else round(f, 3)}


def write_eval_md(ev, run, path: Path):
    fmt = lambda x: "" if x is None else f"{x:.2f}"
    L = ["# YOLO11n presence detection on the BLV-Road-Nav-Accessibility keyframes", "",
         f"{ev['frames_scored']} labelled keyframes from {ev['segments']} segments (presence labels 0 / 1 per object and keyframe, "
         f"{len(ev['annotated_objects'])} annotated objects of the paper's earlier list); detector `{run.get('model', '?')}` on "
         f"`{run.get('device', '?')}` at imgsz {run.get('imgsz', '?')}, cached in `results/blvnav/detections.jsonl`. A frame counts as a "
         "detection of an object when any of its COCO classes is present at or above the confidence threshold (boxes are not "
         "scored: the labels have no boxes). Only the objects that map to a COCO class can score; every other object is a miss. "
         "Proxy rows share a class with a direct object (a wheelchair user, a disabled person and a street vendor are all 'person'; "
         "a maintenance or paratransit vehicle is any truck / car / bus), so their false positives are every frame where the shared "
         "class is present for another reason.", ""]
    for thr, blk in ev["thresholds"].items():
        L += [f"## confidence >= {thr}", "", "| object | group | COCO class(es) | match | frames present | TP | FP | FN | precision | recall | F1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in blk["per_object"].items():
            if not r.get("annotated"):
                L.append(f"| {name} | {MAP[name][0]} | {', '.join(MAP[name][1])} | {MAP[name][2]} | not annotated | | | | | | |")
                continue
            g = f"**{r['group']}**" if r["group"] in HIGH_PRIORITY else str(r["group"])
            L.append(f"| {name} | {g} | {', '.join(r['coco_labels'])} | {r['match']} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                     f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} |")
        m = blk["micro"]
        L += ["", "| micro average over | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
        for k, label in (("direct", "direct COCO objects"), ("mappable", "direct + proxy objects"),
                         ("all_annotated_direct", "all annotated objects, direct classes only (paper-style, N = 12)"),
                         ("all_annotated", "all annotated objects, direct + proxy classes")):
            L.append(f"| {label} | {m[k]['tp']} | {m[k]['fp']} | {m[k]['fn']} | {fmt(m[k]['precision'])} | {fmt(m[k]['recall'])} | {fmt(m[k]['f1'])} |")
        hp = blk["high_priority_frames"]
        L += ["", f"High-priority groups 3, 5, 7: {hp['frames_with_group_3_5_7_object']} labelled frames contain at least one such object; "
              f"the detector fires a mapped class for one of them in {hp['frames_with_a_hit']} ({fmt(hp['recall'])} frame recall). "
              "Only Bench (direct) and Maintenance Vehicle (proxy truck / car) can score there; the rest have no COCO class.", ""]
    L += ["For comparison, Table 5 of the paper reports YOLOv7 on the same keyframes at N = 12 predictable objects, micro precision 0.94, "
          "recall 0.16, F1 0.27; the 'direct classes only' row is the matching protocol (all annotated objects, non-predictable ones always missed).", ""]
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-detector", action="store_true", help="write the taxonomy map only")
    ap.add_argument("--reuse", action="store_true", help="score from the cached detections.jsonl instead of running the model")
    ap.add_argument("--thresholds", default="0.25,0.5", help="confidence thresholds to score (comma separated)")
    args = ap.parse_args()
    if not REPO.exists():
        sys.exit(f"{REPO} missing: git clone https://github.com/Shohan29531/BLV-Road-Nav-Accessibility {REPO}")
    OUT.mkdir(parents=True, exist_ok=True)
    names = load_objects()
    objs = [classify(n) for n in names]
    summ = summarise(objs)
    dump_json({"paper": PAPER, "repo": str(REPO.relative_to(ROOT)), "groups": GROUPS, "high_priority_groups": list(HIGH_PRIORITY),
               "status_meanings": STATUS_DESC, "summary": summ, "objects": objs}, OUT / "taxonomy_map.json")
    write_taxonomy_md(objs, summ, OUT / "taxonomy_map.md")
    a, h = summ["all"], summ["high_priority_3_5_7"]
    print(f"{'group':6s} {'n':>3s} {'direct':>6s} {'proxy':>5s} {'sched':>5s} {'decor':>5s} {'ovocab':>6s} {'taxon':>5s} {'analog':>6s} {'ocr':>3s} {'none':>4s}  concept")
    for g, b in list(summ["by_group"].items()) + [("3+5+7", {"name": "high-priority groups", **h}), ("all", {"name": "", **a})]:
        print(f"{str(g):6s} {b['n']:3d} {b['coco_direct']:6d} {b['coco_proxy']:5d} {b['detected_and_scheduled']:5d} {b['detected_but_decor']:5d} "
              f"{b['open_vocab']:6d} {b['taxonomy_only']:5d} {b['analogue']:6d} {b['ocr_only']:3d} {b['no_class_no_event']:4d}  {b['name']}"
              f"{'  [high priority]' if g in HIGH_PRIORITY else ''}")
    print(f"\nYOLO11n (COCO-80) can output a class for {a['detectable']} of 90 objects ({a['coco_direct']} direct, {a['coco_proxy']} proxy); "
          f"{a['detected_and_scheduled']} of those get a non-decor event. High-priority groups 3/5/7: {h['detectable']} of {h['n']} detectable.\n"
          f"Open-vocabulary NAV_VOCAB ({len(NAV_VOCAB)} prompts) reaches {a['open_vocab_reachable']} of 90 ({h['open_vocab_reachable']} of {h['n']} high priority).")
    print(f"wrote {OUT / 'taxonomy_map.json'} and .md")
    if args.skip_detector:
        return
    segs, odd = load_labels(names)
    n_frames = sum(len(idx) for idx, _ in segs.values())
    print(f"\nlabels: {len(segs)} segments, {n_frames} labelled keyframes, {len({n for _, t in segs.values() for n in t})} objects; odd cells: {dict(odd) or 'none'}")
    det_path = OUT / "detections.jsonl"
    thresholds = [float(t) for t in args.thresholds.split(",")]
    if args.reuse and det_path.exists():
        run = json.loads((OUT / "detector_run.json").read_text()) if (OUT / "detector_run.json").exists() else {}
    else:
        run = run_detector(segs, det_path, min(thresholds))
        dump_json(run, OUT / "detector_run.json")
        print(f"detector: {run['frames']} frames in {run['seconds']} s on {run['device']}; missing files: {len(run['missing_files'])}")
    ev = score(segs, det_path, objs, thresholds)
    ev["run"] = run
    ev["label_oddities"] = dict(odd)
    dump_json(ev, OUT / "detector_eval.json")
    write_eval_md(ev, run, OUT / "detector_eval.md")
    for thr, blk in ev["thresholds"].items():
        m, hp = blk["micro"], blk["high_priority_frames"]
        print(f"conf>={thr}: direct P {m['direct']['precision']} R {m['direct']['recall']} F1 {m['direct']['f1']} | "
              f"mappable P {m['mappable']['precision']} R {m['mappable']['recall']} | paper-style (direct) P {m['all_annotated_direct']['precision']} "
              f"R {m['all_annotated_direct']['recall']} F1 {m['all_annotated_direct']['f1']} | high-priority frame recall {hp['recall']}")
    print(f"wrote {OUT / 'detector_eval.json'} and .md")


if __name__ == "__main__":
    main()
