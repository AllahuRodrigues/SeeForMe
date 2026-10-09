"""Run the research stress matrix (research/stress_matrix.csv: 1,232 sampled modifier combinations over the 154
catalog scenarios) against the schedulers, on CampusBench scenes mapped from each catalog domain.

    python scripts/run_stress_matrix.py --sample 200 --policies pcacs,pcacs_learned:rl
    python scripts/run_stress_matrix.py --sample 400 --memory-ttl 6

Each row's modifiers become scenario / config overrides (light -> lighting, camera -> sway / blur / tilt / occlusion,
density -> scene family, speech -> external messages, empty budget or TTS delay, perception -> depth noise, track
loss, unknown objects, camera loss, reaction_time -> reaction_scale, language -> sign language). Rows whose scope is
ood_only are reported separately and never mixed into the main numbers. Output: results/campus_v2/stress_<tag>.{json,md}
with per-modifier means (which conditions hurt which policy), plus the per-row CSV.
"""
import argparse, csv, json, random, statistics as st, zlib
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from _common import ROOT, dump_json
from campus_common import load_campus_cfg, make_campus_policy
from seeforme.campus import make_campus_scenario, CAMPUS_KINDS
from seeforme.campus.metrics import evaluate_run, score
from seeforme.campus.oracle import CampusOracle
from seeforme.campus.runner import run_on_campus
from seeforme.campus.world import UNKNOWN_OBSTACLE_LABELS, WorldObject
from seeforme.cues.sign_vocab import translate

DOMAIN_KIND = {"indoor_hallway": "hallway", "stairs_levels": "stairs", "doors_entries": "doors", "floor_hazards": "temporary_hazard",
               "moving_people": "crowd", "wayfinding": "finding", "campus_outdoors": "courtyard", "text_and_signs": "finding",
               "retail_restaurant": "hallway", "transit_station": "wayfinding_airport", "airport_terminal": "wayfinding_airport",
               "hospital_clinic": "doors", "crowd_contention": "contention", "construction_detours": "temporary_hazard",
               "dynamic_hazards": "changing", "poor_capture": "poor_capture", "lost_observation": "occlusion",
               "multilingual_and_symbols": "wayfinding_airport", "social_context": "crowd", "street_edge": "crossing",
               "accident_aftermath": "ood_aftermath", "extreme_rare": "ood_aftermath"}
DENSITY_KIND = {"empty": "negative", "burst": "contention", "crowded": "crowd"}

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", default=str(ROOT / "research" / "stress_matrix.csv"))
ap.add_argument("--sample", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--policies", default="oracle,confidence_cooldown,pcacs,pcacs_learned:il,pcacs_learned:rl")
ap.add_argument("--memory-ttl", type=float, default=0.0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.matrix)))
rng = random.Random(args.seed)
rows = rng.sample(rows, min(args.sample, len(rows)))
cfg0 = load_campus_cfg()
if args.memory_ttl > 0:
    cfg0.tracker = replace(cfg0.tracker, memory_ttl_s=args.memory_ttl)


def build(row, seed):
    kind = DENSITY_KIND.get(row["density"], DOMAIN_KIND.get(row["domain"], "hallway"))
    if row["perception"] == "unknown_object" and kind not in ("ood_aftermath",):
        pass
    sc = make_campus_scenario(kind, seed)
    r = random.Random(seed)
    cfg = replace(cfg0); cfg.pcacs = replace(cfg0.pcacs); cfg.speech = replace(cfg0.speech)
    # light
    if row["light"] in ("dim", "night"):
        sc.lighting = "low"
    if row["light"] in ("glare", "backlit"):
        sc.motion_blur = max(sc.motion_blur, 0.25)
        for o in sc.objects: o.detectability *= 0.85
    # camera
    if row["camera"] == "handheld_pan":
        sc.yaw_jitter_deg = max(sc.yaw_jitter_deg, 6.0); sc.pitch_jitter_deg = max(sc.pitch_jitter_deg, 5.0)
    elif row["camera"] == "motion_blur":
        sc.motion_blur = max(sc.motion_blur, 0.5)
    elif row["camera"] == "camera_tilt":
        sc.camera = replace(sc.camera, pitch_deg=25.0)
    elif row["camera"] == "partial_occlusion":
        for o in sc.objects:
            if o.spec().severity > 0 and r.random() < 0.5:
                t0 = r.uniform(2, sc.duration - 6); o.occluded_windows = [(t0, t0 + r.uniform(1.0, 3.0))]
    # speech channel
    if row["speech"] == "busy_short":
        sc.distractor_speech = [(t, 3.0, 0.3) for t in range(3, int(sc.duration) - 3, 9)]
    elif row["speech"] == "busy_long":
        sc.distractor_speech = [(t, 8.0, 0.3) for t in range(3, int(sc.duration) - 8, 14)]
    elif row["speech"] == "budget_empty":
        cfg.pcacs = replace(cfg.pcacs, budget_capacity_s=1.5, budget_refill=0.15)
    elif row["speech"] == "tts_delay":
        cfg.speech = replace(cfg.speech, min_gap=cfg.speech.min_gap + 0.8)
    # perception
    if row["perception"] == "depth_noisy":
        sc.depth_noise_rel = 0.2
    elif row["perception"] == "track_loss":
        for o in sc.objects: o.detectability *= 0.6
    elif row["perception"] == "camera_unavailable":
        sc.blackout_windows = sc.blackout_windows + [(t, t + 1.5) for t in (r.uniform(3, 10), r.uniform(14, 24))]
    elif row["perception"] == "unknown_object":
        z = sc.user_z(r.uniform(8, 18)) + 6.0
        sc.objects.append(WorldObject(obj_id=len(sc.objects) + 1, label=r.choice(sorted(UNKNOWN_OBSTACLE_LABELS)), keyframes=[(0.0, r.uniform(-0.3, 0.3), z)],
                                      width=0.9, height=0.5, base_conf=0.5, conf_noise=0.15, detectability=0.7))
        sc.objects[-1]._scene = sc; sc._events = None
    # user goal / reaction time / language
    if row["user_goal"] == "read_sign":
        sc.distractor_speech = sorted(sc.distractor_speech + [(r.uniform(4, sc.duration - 10), 6.0, 0.3)])
    sc.reaction_scale = {"short": 0.6, "medium": 1.0, "long": 1.6}[row["reaction_time"]]
    lang = {"portuguese": "pt", "spanish": "es", "french": "fr"}.get(row["language"])
    if lang:
        for o in sc.objects:
            if o.kind == "text" and o.event_type == "sign_nav" and o.text:
                key = o.text.lower()
                if key in ("exit", "stairs", "elevator", "restroom", "fire exit", "emergency exit"):
                    o.text = translate(key, lang)
    sc._events = None
    return sc, cfg, kind


results = []
for i, row in enumerate(rows):
    # a stable hash: the built-in hash() of a str is salted per process, so seeds (and results) changed run to run
    seed = 7000 + int(row["instance_id"].split("-")[-1]) + 97 * (zlib.crc32(row["scenario_id"].encode()) % 50)
    sc, cfg, kind = build(row, seed)
    osc = score(run_on_campus(CampusOracle(sc), sc, cfg, seed=seed), sc)
    for name in args.policies.split(","):
        try:
            pol = make_campus_policy(name, cfg, sc)
        except FileNotFoundError:
            continue
        m = evaluate_run(run_on_campus(pol, sc, cfg, seed=seed), sc)
        results.append({**{k: row[k] for k in ("instance_id", "scenario_id", "domain", "case", "scope", "light", "camera", "density", "speech", "perception", "user_goal", "reaction_time", "language")},
                        "kind": kind, "policy": name, "n_events": m.n_events, "recall": m.action_weighted_recall, "late_critical": m.late_critical_rate,
                        "precision": m.cue_precision, "occupancy": m.audio_occupancy, "silence_precision": m.silence_precision,
                        "interrupts": m.n_interrupts, "haptics": m.n_haptic, "blocking": m.critical_blocking_rate, "score": m.score, "regret": osc - m.score})
    if (i + 1) % 25 == 0:
        print(f"{i + 1}/{len(rows)} rows", flush=True)

out = ROOT / "results" / "campus_v2"; out.mkdir(parents=True, exist_ok=True)
tag = f"_{args.tag}" if args.tag else ""
with open(out / f"stress_rows{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)


def means(rs):
    return {"n": len(rs), "recall": st.mean(r["recall"] for r in rs), "late_critical": st.mean(r["late_critical"] for r in rs),
            "precision": st.mean(r["precision"] for r in rs), "occupancy": st.mean(r["occupancy"] for r in rs),
            "regret": st.mean(r["regret"] for r in rs), "interrupts": st.mean(r["interrupts"] for r in rs), "haptics": st.mean(r["haptics"] for r in rs)}


summary = {}
for factor in ("speech", "perception", "camera", "light", "reaction_time", "language", "density", "scope"):
    for pol in args.policies.split(","):
        by = defaultdict(list)
        for r in results:
            # ood_only rows never enter the main per-factor means; the scope factor keeps them as its own level
            if r["policy"] == pol and r["n_events"] > 0 and (factor == "scope" or r["scope"] != "ood_only"):
                by[r[factor]].append(r)
        for level, rs in by.items():
            summary.setdefault(factor, {}).setdefault(level, {})[pol] = means(rs)
ood = {}
for pol in args.policies.split(","):
    rs = [r for r in results if r["policy"] == pol and r["n_events"] > 0 and r["scope"] == "ood_only"]
    if rs:
        ood[pol] = means(rs)
n_ood = sum(1 for row in rows if row["scope"] == "ood_only")
dump_json({"n_rows": len(rows), "n_rows_main": len(rows) - n_ood, "n_rows_ood": n_ood, "policies": args.policies.split(","),
           "memory_ttl": args.memory_ttl, "by_factor": summary, "ood_only": ood}, out / f"stress{tag}.json")
lines = [f"Stress matrix: {len(rows)} sampled specifications x {len(args.policies.split(','))} policies (memory TTL {args.memory_ttl}); "
         f"the {n_ood} ood_only specifications are excluded from every factor except scope and reported separately below\n"]
for factor, levels in summary.items():
    lines.append(f"\n### {factor}\n\n| level | n | " + " | ".join(f"{p} recall / lateC / occ / regret" for p in args.policies.split(",")) + " |")
    lines.append("|---|---|" + "---|" * len(args.policies.split(",")))
    for level in sorted(levels):
        cells = []
        for p in args.policies.split(","):
            s = levels[level].get(p)
            cells.append(f"{s['recall']:.2f} / {s['late_critical']:.2f} / {s['occupancy']:.2f} / {s['regret']:.2f}" if s else "-")
        n = next(iter(levels[level].values()))["n"]
        lines.append(f"| {level} | {n} | " + " | ".join(cells) + " |")
lines.append(f"\n### ood_only (reported separately)\n\n| policy | n | recall | late critical | occupancy | regret |\n|---|---|---|---|---|---|")
for p, s in ood.items():
    lines.append(f"| {p} | {s['n']} | {s['recall']:.2f} | {s['late_critical']:.2f} | {s['occupancy']:.2f} | {s['regret']:.2f} |")
(out / f"stress{tag}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:40]))
print(f"-> {out}/stress{tag}.md")
