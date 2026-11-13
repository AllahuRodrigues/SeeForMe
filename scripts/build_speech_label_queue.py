"""Build the stratified two-rater label queue (issue #5, rubric v4). Development splits only.

    python scripts/build_speech_label_queue.py --name priority160 --seed 7
    python scripts/review_events.py frames --split priority160
    python scripts/review_events.py serve  --split priority160 --reviewer r1      # second person: --reviewer r2
    python scripts/review_events.py kappa  --split priority160 --a r1 --b r2

Each item is one decision moment in a `train` or `finetune` video. The reviewer sees only the frames up to that
moment and the sentence the assistant would say, and records two labels: would an unprompted announcement help, and
is the sentence true of what is visible. The question the user asked, the derived labels, the answers and the stratum
are written to `derived:` columns, which the review page never shows.

Strata: door, stairs, pole (events the COCO detector misses), scene (whole-scene questions), covered_other, quiet
(the user asked and was told the way is clear), unasked (a confirmed detection with no question within 6 s: moments
that training has treated as silent without evidence). The proposed sentence is the live cue text for the event's
track at the last decision before the moment when the detector had it, else the label cue.

Writes results/egoblind/review_events_<name>.csv (git-ignored: it holds the answer key) and
results/egoblind/label_queue_<name>.{json,md} (counts only).
"""
import argparse, csv
from collections import defaultdict

from _common import dump_json, ROOT
from campus_common import load_campus_cfg
from egoblind_common import load_egoblind_split, make_egoblind_policy, source_recordings
from seeforme.egoblind import run_on_egoblind
from seeforme.egoblind.label_queue import DEV_SPLITS, QUOTAS, build_queue, stratum_of

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="priority160")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--max-per-recording", type=int, default=2)
ap.add_argument("--unasked-gap", type=float, default=6.0, help="seconds between an unasked moment and any question")
ap.add_argument("--confirmed", type=float, default=0.5, help="detector confidence of a confirmed candidate")
args = ap.parse_args()

cfg = load_campus_cfg()
src = source_recordings()
pool = []
for split in DEV_SPLITS:
    for sc in load_egoblind_split(split, require_cache=True):
        decisions = []                                            # (t, [candidate dicts]) in time order

        def hook(t, tracks, obs, cue, ch, due, _d=decisions):
            _d.append((t, [{"sid": o.track.source_id, "tid": o.track.track_id, "label": o.track.label,
                            "conf": float(o.track.confidence), "msg": o.message} for o in (obs or [])]))
        run_on_egoblind(make_egoblind_policy("pcacs", cfg, sc), sc, cfg, seed=sc.seed, on_decision=hook)
        base = {"video": sc.name, "video_id": sc.video_id, "source_split": split, "group": src.get(sc.video_id, f"v{sc.video_id:06d}")}

        def before(t_q, within=None):
            return [(t, c) for t, c in decisions if t <= t_q + 1e-6 and (within is None or t_q - t <= within)]

        asked = set()
        for e in sc.events():
            det = e.detail or {}
            t_q = float(det.get("t_q", e.latest_safe_s - 0.5))
            asked.add(t_q)
            concepts = det.get("concepts") or []
            claim, claim_src = "", "none"
            if e.should_speak:
                mine = [c for _, cs in before(t_q) for c in cs if c["sid"] == e.obj_id]
                if mine:
                    claim, claim_src = mine[-1]["msg"], "system"
                elif (e.ideal_cue or "").strip():
                    claim, claim_src = e.ideal_cue.strip(), "label"
            else:
                top = [c for _, cs in before(t_q, within=2.0)[-1:] for c in cs if c["conf"] >= args.confirmed]
                if top:
                    claim, claim_src = max(top, key=lambda c: c["conf"])["msg"], "system_top"
            pool.append({**base, "key": f"{sc.video_id}:{e.obj_id}", "obj_id": e.obj_id, "t_q": round(t_q, 2), "claim": claim,
                         "stratum": stratum_of(e.label, concepts, e.should_speak, e.reason),
                         "derived:should_speak": int(bool(e.should_speak)), "derived:label": e.label, "derived:concepts": ",".join(concepts),
                         "derived:question": det.get("question", ""), "derived:qtype": det.get("qtype", ""),
                         "derived:window": f"{e.earliest_useful_s:.2f}-{e.latest_safe_s:.2f}", "derived:claim_source": claim_src})
        qtimes = sorted(asked | set(sc.question_times) | set(sc.labelled_times))
        prev = {}
        for t, cs in decisions:                                   # first confirmed candidate far from every question
            now = {c["tid"]: c for c in cs if c["conf"] >= args.confirmed}
            both = [c for tid, c in now.items() if tid in prev]
            if t >= 3.0 and both and all(abs(t - q) >= args.unasked_gap for q in qtimes):
                c = max(both, key=lambda c: c["conf"])
                pool.append({**base, "key": f"{sc.video_id}:9001", "obj_id": 9001, "t_q": round(t, 2), "claim": c["msg"], "stratum": "unasked",
                             "derived:should_speak": "", "derived:label": c["label"], "derived:concepts": "", "derived:question": "",
                             "derived:qtype": "", "derived:window": "", "derived:claim_source": "system_top"})
                break
            prev = now

avail = defaultdict(int)
for p in pool:
    avail[p["stratum"]] += 1
picked, rep = build_queue(pool, QUOTAS, seed=args.seed, max_per_group=args.max_per_recording)
rep["available"] = dict(avail)
rep["claim_source"] = {k: sum(p["derived:claim_source"] == k for p in picked) for k in ("system", "label", "system_top", "none")}
rep["recordings"] = len({p["group"] for p in picked})

cols = ["sample_id", "video", "video_id", "obj_id", "t_q", "window", "source_split", "claim", "human_verdict", "human_note",
        "derived:stratum", "derived:should_speak", "derived:label", "derived:concepts", "derived:question", "derived:qtype",
        "derived:window", "derived:claim_source"]
out = ROOT / "results" / "egoblind"
with open(out / f"review_events_{args.name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for p in picked:                                              # the same window for every item: it must not hint at the stratum
        w.writerow({**p, "window": f"{max(0.0, p['t_q'] - 4.0):.2f}-{p['t_q']:.2f}", "derived:stratum": p["stratum"],
                    "human_verdict": "", "human_note": ""})
dump_json(rep, out / f"label_queue_{args.name}.json")
L = [f"Label queue `{args.name}` (seed {args.seed}; {rep['total']} decision moments from {rep['recordings']} source recordings; "
     f"train {rep['by_split'].get('train', 0)}, finetune {rep['by_split'].get('finetune', 0)}; at most {args.max_per_recording} per recording)\n",
     "| stratum | wanted | available | picked for the stratum | filled in from it | final |", "|---|---|---|---|---|---|"]
for s, q in QUOTAS.items():
    L.append(f"| {s} | {q} | {avail[s]} | {rep['picked'][s]} | {rep['filled'].get(s, 0)} | {rep['final'].get(s, 0)} |")
L += ["", "Proposed sentence: " + ", ".join(f"{k} {v}" for k, v in rep["claim_source"].items()) +
      " (system = live cue text for the event's own track; label = cue from the weak label because the detector missed it; "
      "system_top = cue for the most confident detection at that moment; none = no sentence, usefulness only).",
      "No human labels are included. `select`, `holdout`, `test` and own recordings are never sampled."]
(out / f"label_queue_{args.name}.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
