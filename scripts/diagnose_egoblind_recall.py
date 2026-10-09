"""Where does recall go on real EgoBlind video? (14-day plan, P1)

    python scripts/diagnose_egoblind_recall.py --split holdout --policies pcacs,pcacs_learned:rl,egoblind_learned:il,egoblind_learned:rl

For every should-speak weak event the policy is replayed with a decision hook and the event is put in exactly one bin:

  spoken_timely         first cue about the object inside [earliest_useful, latest_safe]           (recall)
  spoken_early          first cue before the window (then blocked as "recently announced")
  spoken_late           first cue after latest_safe
  not_coverable         concept the COCO detector + taxonomy cannot see (door, stairs, pole ...)   (perception vocabulary)
  not_detected          coverable but the detector never saw it before the question                (perception recall)
  never_tracked         detected, but no track carried the object at any decision step in the window (tracker)
  low_confidence        tracked, but below the policy's min_confidence at every step in the window
  below_threshold       scored at every step, never above the speak threshold                     (scheduler utility)
  budget_blocked        above threshold on a free channel, but the speech budget was empty
  channel_busy          above threshold only while the channel was busy, never interrupted
  other                 above threshold on a free channel with budget, and still silent (policy-internal rule)

Also writes results/egoblind/review_events_<split>.csv: one row per weak event with the question, answers, concepts,
window, coverability, detection and each policy's bin, for human review of the weak labels.
"""
import argparse, csv, math
from collections import Counter, defaultdict
from _common import dump_json
from campus_common import load_campus_cfg
from egoblind_common import RESULTS_E, load_egoblind_split, make_egoblind_policy, parse_noise
from seeforme.egoblind import run_on_egoblind

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="holdout")
ap.add_argument("--policies", default="pcacs,pcacs_learned:rl,egoblind_learned:il,egoblind_learned:rl")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--noise", default="")
ap.add_argument("--tag", default="")
args = ap.parse_args()
cfg = load_campus_cfg()
noise = parse_noise(args.noise)
scenarios = load_egoblind_split(args.split, require_cache=True, limit=args.limit)
BINS = ["spoken_timely", "spoken_early", "spoken_late", "not_coverable", "not_detected", "never_tracked", "low_confidence",
        "below_threshold", "budget_blocked", "channel_busy", "other"]


def threshold_of(policy):
    """The score a candidate must reach to be spoken. Linear P-CACS: threshold_speak. Learned scheduler, RL mode: the
    learned silence logit (it speaks when a candidate's log-odds against its own silence beat the global silence bias).
    Learned scheduler, imitation mode: the action-probability threshold, compared with 1 - P(silence), an upper bound
    on the best single action's probability, so 'below_threshold' there is exact and 'other' may hold near misses."""
    if hasattr(policy, "threshold") and hasattr(policy, "model"):
        if getattr(policy, "mode", "") == "imitation":
            return float(policy.threshold)
        return float(policy.model.net.silence_bias.detach().reshape(-1)[0])
    return getattr(getattr(policy, "cfg", None), "threshold_speak", 0.0)


def diagnose(policy, sc):
    events = {e.obj_id: e for e in sc.speak_events()}
    seen = defaultdict(lambda: {"tracked": False, "conf_ok": False, "max_score": -math.inf, "free_above": False, "busy_above": False,
                                "budget_ok_above": False})
    thr = threshold_of(policy)
    min_conf = getattr(getattr(policy, "cfg", None), "min_confidence", 0.0)

    def hook(t, tracks, obs, cue, ch, due):
        for oid, e in events.items():
            if not (e.earliest_useful_s <= t <= e.latest_safe_s):
                continue
            s = seen[oid]
            for tr in tracks:
                if tr.source_id == oid:
                    s["tracked"] = True
                    if tr.confidence >= min_conf:
                        s["conf_ok"] = True
            for o in obs:
                if o.track.source_id == oid and o.score is not None and o.score > -1e8:
                    s["max_score"] = max(s["max_score"], o.score)
                    if o.score >= thr:
                        if ch.busy:
                            s["busy_above"] = True
                        else:
                            s["free_above"] = True
                            budget = getattr(policy, "budget", None) if not hasattr(policy, "model") else None   # only P-CACS gates on the budget
                            if budget is None or not getattr(policy.cfg, "use_budget", False) or budget >= getattr(o, "duration", 0.0):
                                s["budget_ok_above"] = True

    log = run_on_egoblind(policy, sc, cfg, seed=sc.seed, noise=noise, on_decision=hook)
    first = {}
    for c in sorted(log.spoken, key=lambda c: c.t):
        if c.source_id in events and c.source_id not in first:
            first[c.source_id] = c
    out = {}
    for oid, e in events.items():
        d, s = e.detail, seen[oid]
        if oid in first:
            c = first[oid]
            b = "spoken_timely" if e.in_window(c.t) else ("spoken_early" if c.t < e.earliest_useful_s else "spoken_late")
        elif not d.get("coverable"):
            b = "not_coverable"
        elif not d.get("detected"):
            b = "not_detected"
        elif not s["tracked"]:
            b = "never_tracked"
        elif not s["conf_ok"]:
            b = "low_confidence"
        elif s["max_score"] < thr:
            b = "below_threshold"
        elif s["free_above"] and not s["budget_ok_above"]:
            b = "budget_blocked"
        elif s["busy_above"] and not s["free_above"]:
            b = "channel_busy"
        else:
            b = "other"
        out[oid] = (b, s["max_score"] if math.isfinite(s["max_score"]) else None)
    return out


rows = {}      # (scenario, obj_id) -> row
bins = {}
for name in args.policies.split(","):
    cnt = Counter(); cnt_cov = Counter()
    for sc in scenarios:
        try:
            pol = make_egoblind_policy(name, cfg, sc)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}"); break
        res = diagnose(pol, sc)
        for e in sc.speak_events():
            key = (sc.name, e.obj_id)
            r = rows.setdefault(key, {"video": sc.name, "video_id": sc.video_id, "obj_id": e.obj_id, "t_q": e.detail.get("t_q"),
                                      "window": f"{e.earliest_useful_s:.1f}-{e.latest_safe_s:.1f}", "qtype": e.detail.get("qtype"),
                                      "question": e.detail.get("question"), "answers": " || ".join(e.detail.get("answers", [])),
                                      "concepts": ",".join(e.detail.get("concepts", [])), "labels": ",".join(e.detail.get("labels", [])),
                                      "weak_reason": e.detail.get("weak_reason"), "coverable": e.detail.get("coverable"),
                                      "coverable_reason": e.detail.get("coverable_reason"), "detected": e.detail.get("detected"),
                                      "first_seen_s": e.detail.get("first_seen_s"), "severity": e.severity, "urgency": e.urgency,
                                      "human_verdict": "", "human_note": ""})
            b, ms = res[e.obj_id]
            r[f"bin:{name}"] = b; r[f"max_score:{name}"] = "" if ms is None else f"{ms:.2f}"
            cnt[b] += 1
            if e.detail.get("coverable"):
                cnt_cov[b] += 1
    if not cnt:
        continue
    bins[name] = {"all": dict(cnt), "coverable": dict(cnt_cov), "n_events": sum(cnt.values()), "n_coverable": sum(cnt_cov.values())}
    print(f"{name}: " + ", ".join(f"{b}={cnt[b]}" for b in BINS if cnt[b]))

tag = f"_{args.tag}" if args.tag else ""
dump_json(bins, RESULTS_E / f"recall_diagnosis_{args.split}{tag}.json")
lines = [f"# EgoBlind recall diagnosis ({args.split}: {len(scenarios)} videos, {len(rows)} should-speak events)", "",
         "| bin | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
for b in BINS:
    lines.append(f"| {b} | " + " | ".join(str(bins[p]["all"].get(b, 0)) for p in bins) + " |")
lines += ["", "Coverable events only:", "", "| bin | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
for b in BINS:
    if b != "not_coverable":
        lines.append(f"| {b} | " + " | ".join(str(bins[p]["coverable"].get(b, 0)) for p in bins) + " |")
(RESULTS_E / f"recall_diagnosis_{args.split}{tag}.md").write_text("\n".join(lines) + "\n")
cols = list(next(iter(rows.values())).keys()) if rows else []
with open(RESULTS_E / f"review_events_{args.split}{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for key in sorted(rows, key=lambda k: (int(k[0].split("_")[-1]) if k[0].split("_")[-1].isdigit() else k[0], k[1])):
        w.writerow(rows[key])
print("\n".join(lines))
print(f"-> results/egoblind/recall_diagnosis_{args.split}{tag}.{{json,md}}, review_events_{args.split}{tag}.csv")
