"""Score a cue log (JSONL from run_video.py) against a hand-made annotation file.

Two annotation formats:
  * legacy (docs/annotation_format.md): {"duration": 42.0, "events": [{"label": "chair", "t_start": 3.0, "t_end": 8.0}, ...]}
  * CampusBench (--campus, docs/campus_annotation.md): events with should_speak / severity /
    earliest_useful_s / latest_safe_s / minimum_information, as drafted by scripts/annotate_video.py.
"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("cue_log"); ap.add_argument("annotation")
ap.add_argument("--dup-window", type=float, default=6.0); ap.add_argument("--campus", action="store_true")
args = ap.parse_args()
cues = sorted((json.loads(l) for l in open(args.cue_log)), key=lambda c: c["t"])
ann = json.load(open(args.annotation))
events = ann["events"]
minutes = ann["duration"] / 60

if not args.campus:
    hits = 0; delays = []
    for ev in events:
        h = [c for c in cues if ev["t_start"] <= c["t"] < ev["t_end"] and (c["label"] == ev["label"] or (ev["label"] == "text" and c["label"] == "text"))]
        if h: hits += 1; delays.append(h[0]["t"] - ev["t_start"])
    last = {}; dup = 0; useful = 0
    for c in cues:
        key = c["label"]
        if key in last and c["t"] - last[key] < args.dup_window: dup += 1
        elif any(e["label"] == key and e["t_start"] <= c["t"] < e["t_end"] for e in events): useful += 1
        last[key] = c["t"]
    print(json.dumps({"important_cue_recall": hits / len(events) if events else None, "cue_precision": useful / len(cues) if cues else None,
                      "audio_load_per_min": len(cues) / minutes, "duplicate_rate": dup / len(cues) if cues else 0,
                      "mean_critical_delay": sum(delays) / len(delays) if delays else None, "n_cues": len(cues)}, indent=1))
    raise SystemExit

speak = [e for e in events if e.get("should_speak") and e.get("earliest_useful_s") is not None and e.get("latest_safe_s") is not None]
spoken = [c for c in cues if c.get("action", "speak") != "haptic"]
match = lambda c, e: c["label"] == e["label"] or (e["label"] == "text" and c["label"] == "text" and (e.get("text") or "").lower() in (c.get("message") or "").lower())
first = {}
for e in speak:
    h = [c for c in spoken if match(c, e) and e["first_seen_s"] - 0.5 <= c["t"] <= e["last_seen_s"] + 0.5] if "first_seen_s" in e else [c for c in spoken if match(c, e)]
    if h: first[id(e)] = h[0]
timely = {k: c for k, c in first.items() if any(id(e) == k and e["earliest_useful_s"] <= c["t"] <= e["latest_safe_s"] for e in speak)}
sev = sum(e["severity"] for e in speak)
awr = sum(e["severity"] for e in speak if id(e) in timely) / sev if sev else None
late = [e for e in speak if id(e) not in timely]
crit = [e for e in speak if e["severity"] >= 0.7]
useful = sum(1 for k, c in timely.items())
occ = sum(c.get("duration_s", max(1.2, 0.35 * len((c.get("message") or "").split()))) for c in spoken) / ann["duration"]
jac = []
for k, c in timely.items():
    e = next(e for e in speak if id(e) == k)
    req, got = set(e.get("minimum_information", [])), set(c.get("slots", ["object"]))
    jac.append(len(req & got) / len(req | got) if req | got else 1.0)
print(json.dumps({"n_events": len(speak), "action_weighted_recall": awr, "important_cue_recall": len(timely) / len(speak) if speak else None,
                  "late_rate": len(late) / len(speak) if speak else None,
                  "late_critical_rate": sum(1 for e in late if e["severity"] >= 0.7) / len(crit) if crit else None,
                  "cue_precision": useful / len(spoken) if spoken else None, "audio_occupancy": occ, "cues_per_min": len(spoken) / minutes,
                  "haptics_per_min": sum(1 for c in cues if c.get("action") == "haptic") / minutes,
                  "min_message_ratio": sum(jac) / len(jac) if jac else None}, indent=1))
