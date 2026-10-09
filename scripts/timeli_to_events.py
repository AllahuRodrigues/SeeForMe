"""Convert TIMELI frame annotations (Kuribayashi, Shangguan and Ohn-Bar, ICRA 2026) into SeeForMe EventAnnotation records.

    python scripts/timeli_to_events.py                      # all splits -> data/timeli/events.jsonl, results/timeli/events_summary.{json,md}
    python scripts/timeli_to_events.py --splits test,real   # skip the 360 MB train file
    python scripts/timeli_to_events.py --min-quiet-s 1      # keep the 1 s gaps after every instruction as quiet windows too

One JSON line per event, flattened `EventAnnotation.as_dict()` plus `split`, `video_id`, `frame_id`, `t_video_s`, `clip`:

* a **speak event** per annotated instruction: `ideal_s` = the instruction's second (`t_video_s` = TIMELI frame `id`),
  `earliest_useful_s` = t - 1, `latest_safe_s` = t + 1, mirroring TIMELI's one-second timing tolerance (Timing F1 / AUC);
  `event_type` and `severity` from the reason (obstacle_in_front -> the obstacle's SeeForMe event type, direction_change ->
  goal 0.7, enter_junction -> landmark 0.8, exit_junction -> landmark 0.5, constant_instruction -> goal 0.3, stop -> goal
  1.0); `label` from `conveyed_obstacles` (CARLA) or from the instruction text (real set, via `match_concepts`);
  `ideal_cue` = the annotated instruction; `minimum_information` = the slots the instruction carries;
* a **quiet window** per maximal run of silent frames of at least `--min-quiet-s` seconds (`should_speak = False`,
  `[t_first - 0.5, t_last + 0.5]`, reason `remain_silent` or `remain_silent_in_junction` by majority). The 1 s gap that the
  generator forces after every instruction ("never two in a row") is dropped by default (`--min-quiet-s 2`).

What cannot be mapped (also written to the summary):
  - no object identity or track: CARLA labels name a class and a coarse side (front / left / right), the real set only the
    text; `obj_id` is a running index, `direction_at_ideal_deg` is coarse (0 / -45 / +45) and `distance_at_ideal_m` is None
    (obstacles are "within cane range", 1.5 m, by construction but no distance is stored);
  - no earliest-useful window: TIMELI has one timestamp per instruction with a +/-1 s tolerance, not an interval of usefulness,
    so `earliest_useful_s` / `latest_safe_s` are the tolerance, not the user's need; `t_contact`, `sudden` and
    `interrupt_current_speech` cannot be derived (`t_contact` None, `sudden` False, interrupt = severity >= 0.75);
  - quiet windows mean "the rule produced no trigger", not "the user confirmed the way is clear" as in EgoBlind: a cue inside
    one is redundant, not necessarily wrong (`detail.quiet_semantics = "no_trigger"`);
  - `constant_instruction` (periodic reassurance after 6-9 silent seconds) has no SeeForMe equivalent and is kept at
    severity 0.3 with `detail.reassurance = True` so schedulers that stay silent are not charged a miss by default;
  - the real set's reasons are inconsistent with the text in ~17% of frames (a silence labelled obstacle_in_front, an
    instruction labelled remain_silent); the text wins and `detail.reason_inferred` records it;
  - frames and videos are not on disk, so the events cannot be replayed through the perception stack yet; `clip` and
    `image` paths are kept for when they are.
"""
import argparse, json, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from _common import RESULTS, ROOT, dump_json
from seeforme.campus.world import EVENT_SPECS, LABEL_EVENT, EventAnnotation
from seeforme.egoblind.data import match_concepts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "timeli"))
ap.add_argument("--splits", default="train,test,real")
ap.add_argument("--out", default=str(ROOT / "data" / "timeli" / "events.jsonl"))
ap.add_argument("--summary", default=str(RESULTS / "timeli"))
ap.add_argument("--min-quiet-s", type=int, default=2, help="shortest run of silent frames kept as a quiet window")
args = ap.parse_args()
DATA = Path(args.data); OUT = Path(args.out); SUM = Path(args.summary); SUM.mkdir(parents=True, exist_ok=True)
FILES = {"train": "train_labels.json", "test": "test_labels.json", "real": "real_test.json"}
SILENT = ("remain_silent", "remain_silent_in_junction")
TOLERANCE_S = 1.0          # TIMELI scores timing with a one-second tolerance
QUIET_PAD_S = 0.5
LABEL_SOURCE = {"train": "timeli_rule", "test": "timeli_rule", "real": "timeli_human"}

# TIMELI obstacle classes (CARLA semantic clusters, lower-cased) -> SeeForMe labels
TIMELI_LABEL = {"pedestrian": "person", "vehicle": "car", "pole": "pole", "fence": "fence", "guard rail": "fence", "guard_rail": "fence",
                "building": "wall", "wall": "wall", "vegetation": "tree", "traffic sign": "sign", "traffic_sign": "sign", "object": "unknown object"}
# SeeForMe label -> event type when the label is announced as an obstacle in front (LABEL_EVENT covers the CampusBench labels)
OBSTACLE_TYPE = {**LABEL_EVENT, "person": "person_moving", "pole": "hazard_static", "fence": "barrier", "wall": "barrier", "tree": "hazard_static",
                 "sign": "hazard_static", "unknown object": "unknown_obstacle", "box": "hazard_static", "cart": "hazard_static",
                 "trash can": "hazard_static", "cone": "barrier", "obstacle": "unknown_obstacle", "ground hazard": "unknown_obstacle",
                 "stairs": "stairs", "curb": "curb", "door": "door", "animal": "hazard_static", "dog": "hazard_static", "shelf": "hazard_static",
                 "machine": "hazard_static", "elevator": "landmark"}
NOT_OBJECTS = {"road", "crossing", "water", "sign"}       # concepts that name the scene, not an obstacle (sign stays for CARLA traffic signs)
REASON_TYPE = {"direction_change": ("goal", 0.7, "medium", ("direction",)), "enter_junction": ("landmark", 0.8, "high", ("distance",)),
               "exit_junction": ("landmark", 0.5, "medium", ()), "constant_instruction": ("goal", 0.3, "low", ()),
               "constantly_instruct": ("goal", 0.3, "low", ()), "stop": ("goal", 1.0, "high", ("action",))}
SIDE_RE = re.compile(r"\b(?:on|to|at) (?:your|the) (left|right)\b|\b(left|right) side\b|\b(?:your|the) (left|right)\b")
FRONT_RE = re.compile(r"\b(?:ahead|in front|front|forward)\b")
COARSE_DEG = {"front": 0.0, "left": -45.0, "right": 45.0}


def instruction_of(rec):
    a = rec["QA"][0]["a"]
    reason = rec.get("speaking_reason", rec.get("spoken_reason"))
    text = a
    if isinstance(a, str) and a.lstrip().startswith("{"):
        try:
            j = json.loads(a); reason = j.get("reason", reason); text = j.get("instruction")
        except json.JSONDecodeError:
            pass
    text = (text or "").strip()
    return reason, (None if text.lower() in ("none", "") else text)


def objects_from_record(rec, text):
    """[(seeforme label, side)] named by the record: CARLA conveyed_obstacles first, else the instruction text."""
    out = []
    for obj, sides in (rec.get("conveyed_obstacles") or {}).items():
        lab = TIMELI_LABEL.get(obj.lower().replace("_", " "), obj.lower())
        out.append((lab, ("front" if "front" in sides else sides[0] if sides else "front")))
    out.sort(key=lambda o: o[1] != "front")                 # the obstacle ahead is the event; side objects are context
    if out or not text:
        return out
    coco, other = match_concepts(text)
    m = SIDE_RE.search(text.lower())
    side = next((g for g in (m.groups() if m else ()) if g), None) or ("front" if FRONT_RE.search(text.lower()) else "front")
    for lab in coco + [c for c in other if c not in NOT_OBJECTS]:
        out.append((lab, side))
    return out


def slots_for(text, etype, spec_slots):
    slots = ["object"] if etype not in ("goal", "landmark") else []
    tl = text.lower()
    if SIDE_RE.search(tl) or re.search(r"\b(?:left|right|o'?clock|your (?:one|two|three|nine|ten|eleven))\b", tl):
        slots.append("direction")
    if re.search(r"\b(?:stop|slow|wait|keep walking|continue|turn|go straight|walk straight|proceed)\b", tl):
        slots.append("action")
    if re.search(r"\b\d+(?:\.\d+)? ?(?:m|meters?|steps?|paces)\b", tl):
        slots.append("distance")
    for s in spec_slots:
        if s not in slots and s in ("motion",):
            slots.append(s)
    return tuple(dict.fromkeys(slots))


def speak_event(obj_id, rec, split, reason, text):
    t = float(rec["id"])
    objs = objects_from_record(rec, text)
    inferred = None
    if reason in SILENT or reason in (None, "None"):          # real-set inconsistencies: the text wins
        inferred = "obstacle_in_front" if objs else "constant_instruction"
        reason = inferred
    if reason == "obstacle_in_front" or (reason not in REASON_TYPE and objs):
        if objs:
            label, side = objs[0]
        else:
            label, side = "unknown object", "front"
        etype = OBSTACLE_TYPE.get(label, "unknown_obstacle")
        spec = EVENT_SPECS[etype]
        severity, urgency, spec_slots = spec.severity, spec.urgency, spec.slots
        if reason != "obstacle_in_front":
            inferred = inferred or "obstacle_in_front"
        reason_out = "obstacle_in_front"
    else:
        etype, severity, urgency, spec_slots = REASON_TYPE.get(reason, ("goal", 0.3, "low", ()))
        label, side = ("junction" if etype == "landmark" else "path"), "front"
        reason_out = reason
    moving = etype in ("person_moving", "vehicle", "bike")
    detail = {"split": split, "video_id": rec["video_id"], "frame_id": rec["id"], "clip": rec.get("video"), "image": rec.get("image"),
              "timeli_reason": reason_out, "reason_inferred": inferred, "action": rec.get("action"), "goal_coord": rec.get("goal_coord"),
              "is_junction": bool(rec.get("is_junction")), "objects": objs, "conveyed_obstacles": rec.get("conveyed_obstacles"),
              "scene_objects": rec.get("description_cluster"), "template": (rec["QA"][0].get("a_original") if rec.get("QA") else None),
              "weather": (rec.get("meta") or {}).get("weather"), "town": (rec.get("meta") or {}).get("town"),
              "label_source": LABEL_SOURCE[split], "direction_source": "coarse_relative_position", "within_cane_range_m": 1.5 if reason_out == "obstacle_in_front" else None,
              "reassurance": reason_out in ("constant_instruction", "constantly_instruct"), "coverable": etype not in ("goal", "landmark") and label in LABEL_EVENT}
    return EventAnnotation(obj_id, label, "object", etype, None, True, "", float(severity), urgency, severity >= 0.75,
                           round(t - TOLERANCE_S, 2), round(t, 2), round(t + TOLERANCE_S, 2), None,
                           1.0 if reason_out == "obstacle_in_front" else 0.0, None, COARSE_DEG.get(side, 0.0), moving, False,
                           slots_for(text, etype, spec_slots), text, detail)


def quiet_event(obj_id, run, split, video_id):
    t0, t1 = run[0]["id"], run[-1]["id"]
    reasons = Counter(instruction_of(r)[0] for r in run)
    in_junction = sum(bool(r.get("is_junction")) for r in run)
    reason = "remain_silent_in_junction" if reasons.get("remain_silent_in_junction", 0) > len(run) / 2 else "remain_silent"
    detail = {"split": split, "video_id": video_id, "frame_id": t0, "frame_id_last": t1, "n_frames": len(run), "junction_frames": in_junction,
              "reasons": dict(reasons), "label_source": LABEL_SOURCE[split], "quiet_semantics": "no_trigger", "coverable": False}
    return EventAnnotation(obj_id, "none", "object", "quiet", None, False, reason, 0.0, "none", False,
                           round(t0 - QUIET_PAD_S, 2), None, round(t1 + QUIET_PAD_S, 2), None, 0.0, None, None, False, False, (), "", detail)


stats = {}
unmapped = ["object identity / tracks (class + coarse side only)", "distance and time-to-contact (obstacles are 'within cane range' by construction)",
            "earliest-useful windows (one timestamp per instruction, +/-1 s tolerance)", "sudden events and interruption semantics",
            "user-confirmed clear paths (quiet windows mean 'no trigger fired')", "periodic reassurance (constant_instruction) has no SeeForMe event type",
            "frames and videos (not downloaded: 32 GB / 38 GB / 537 GB), so no perception replay yet"]
n_total = 0
with open(OUT, "w") as fout:
    for split in args.splits.split(","):
        path = DATA / FILES[split]
        if not path.exists():
            print(f"missing {path}"); continue
        print(f"reading {path.name} ...")
        recs = json.loads(path.read_text())
        by = defaultdict(list)
        for r in recs:
            by[r["video_id"]].append(r)
        c = Counter(); types = Counter(); labels = Counter(); quiet_s = 0; sev = []
        for video_id, rs in by.items():
            rs.sort(key=lambda r: r["id"])
            obj_id = 0; run = []
            for r in rs + [None]:                                  # the sentinel closes the last silent run
                reason, text = instruction_of(r) if r is not None else (None, "")
                if r is not None and text is None:
                    run.append(r); continue
                if run:                                            # an instruction (or the end of the video) closes a silent run
                    if len(run) >= args.min_quiet_s:
                        obj_id += 1
                        e = quiet_event(obj_id, run, split, video_id)
                        fout.write(json.dumps({**e.as_dict(), "split": split, "video_id": video_id, "frame_id": run[0]["id"], "t_video_s": float(run[0]["id"]), "clip": run[-1].get("video")}) + "\n")
                        c["quiet_windows"] += 1; quiet_s += len(run)
                    else:
                        c["quiet_gaps_dropped"] += 1
                    run = []
                if r is None:
                    break
                obj_id += 1
                e = speak_event(obj_id, r, split, reason, text)
                fout.write(json.dumps({**e.as_dict(), "split": split, "video_id": video_id, "frame_id": r["id"], "t_video_s": float(r["id"]), "clip": r.get("video")}) + "\n")
                c["speak_events"] += 1; c[f"reason:{e.detail['timeli_reason']}"] += 1; types[e.event_type] += 1; labels[e.label] += 1
                c["reason_inferred"] += bool(e.detail["reason_inferred"]); c["coverable"] += bool(e.detail["coverable"]); sev.append(e.severity)
            c["videos"] += 1; c["frames"] += len(rs)
        n_total += c["speak_events"] + c["quiet_windows"]
        stats[split] = {"videos": c["videos"], "frames_s": c["frames"], "speak_events": c["speak_events"], "quiet_windows": c["quiet_windows"],
                        "quiet_seconds": quiet_s, "quiet_gaps_dropped_(<min)": c["quiet_gaps_dropped"],
                        "speak_events_per_minute": round(60 * c["speak_events"] / c["frames"], 2), "mean_severity": round(st.mean(sev), 3) if sev else None,
                        "reasons": {k[7:]: v for k, v in c.items() if k.startswith("reason:")}, "event_types": dict(types.most_common()),
                        "labels": dict(labels.most_common(12)), "coverable_by_coco_taxonomy": c["coverable"], "reason_inferred_from_text": c["reason_inferred"]}
summary = {"output": str(OUT), "events": n_total, "min_quiet_s": args.min_quiet_s, "tolerance_s": TOLERANCE_S, "splits": stats, "cannot_map": unmapped}
dump_json(summary, SUM / "events_summary.json")
lines = ["# TIMELI -> SeeForMe events", "", f"`{OUT}`: {n_total} events (one JSON line each; speak events at the instruction second with a +/-{TOLERANCE_S:.0f} s window, "
         f"quiet windows for silent runs of at least {args.min_quiet_s} s).", "",
         "| split | videos | seconds | speak events | per min | quiet windows | quiet s | mean severity | coverable (COCO + taxonomy) | reason inferred from text |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for split, s in stats.items():
    lines.append(f"| {split} | {s['videos']} | {s['frames_s']} | {s['speak_events']} | {s['speak_events_per_minute']} | {s['quiet_windows']} | {s['quiet_seconds']} | "
                 f"{s['mean_severity']} | {s['coverable_by_coco_taxonomy']} | {s['reason_inferred_from_text']} |")
for split, s in stats.items():
    lines += ["", f"{split}: reasons {s['reasons']}; event types {s['event_types']}; labels {s['labels']}"]
lines += ["", "Cannot be mapped: " + "; ".join(unmapped) + "."]
(SUM / "events_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"wrote {OUT}, {SUM / 'events_summary.json'}, {SUM / 'events_summary.md'}")
