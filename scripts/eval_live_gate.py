"""Does the live-demo AnnouncementGate cut repeats and airtime without losing timely events?

Replays cached EgoBlind development videos (finetune + select) through the schedulers exactly like
scripts/run_egoblind.py (seeforme.egoblind.run_on_egoblind), each policy in up to three variants:

    bare      the scheduler alone
    gated     AnnouncementGate(scheduler, GateConfig())   default config, operator_filter False (no desk webcam here)
    adapter   gated + ActivityDetector (track-based: frame=None, the replays have no pixels) + TaskAdapter, both updated
              at every decision step, as the live demo's LiveController wires them: a mode sets the gate only (hold,
              background, read-mode muting, which the gate applies before it remembers a cue), never the scheduler's own
              settings, and the scheduler decides through the gate's decide (the channel it is shown is its own) (only
              if seeforme/live/activity.py exists)

    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_gate.py                        # finetune,select; COCO + open-vocab
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_gate.py --caches coco --limit 10 --no-write   # smoke test
    OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_live_gate.py --splits holdout --allow-final        # ONCE, at the end

Pooled metrics (as in scripts/run_egoblind.py): recall over coverable / all / detected events, precision, quiet-window
violations, cues and haptics per minute, audio occupancy, plus two repeat measures:

    dup_rate      THE MAIN REPEAT MEASURE. campus.metrics duplicate rate: spoken cues about the same ground-truth object
                  (source id) as a cue less than 6 s earlier, over all spoken cues. Independent of the gate's rules.
                  Only detections whose label an EgoBlind event names, inside that event's window, carry a source id,
                  so repeats about unlabelled objects are not counted.
    repeat_rate   secondary: share of spoken cues whose (label, direction sector) was already spoken in the previous 20 s
                  while the object is not at least 1 m closer than at that earlier cue (sector: left < -20 deg < ahead <
                  20 deg < right, camera angle of the box centre; text cues are keyed by their normalised text; an
                  unknown distance counts as "not closer"). This restates the gate's own rule (identity + sector + hold),
                  so a gated run scores well on it by construction; it is reported to show what is left, not as proof.

These are walking / hand-held clips: none of them is a desk-webcam session, so they neither show nor test the live
"Person." loop at a desk (that case is covered by tests/test_live_gate.py with the real tracker and dynamics).

and, per gated variant, the gate's stats counters and pass / hold reasons summed over videos. Timely events (the first
cue about an event inside its window) are compared with the bare run: every event a variant loses or gains is listed
with the reason (held by the gate and why, spoken before the window, or never offered by the scheduler in the window).

Writes results/live/gate_eval_dev.{md,json} (development splits) or results/live/gate_eval_<splits>.{md,json}.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (str(SCRIPTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEV_SPLITS = ("finetune", "select")
FINAL_SPLITS = ("holdout", "test")
CACHES = {"coco": ROOT / "data" / "egoblind" / "cache", "ow": ROOT / "data" / "egoblind" / "cache_ow"}
CACHE_NAMES = {"coco": "COCO detector cache (yolo11n; the live demo runs the same COCO classes with yolo11s)",
               "ow": "open-vocabulary detector cache (yolov8s-worldv2)"}
DEFAULT_POLICIES = "pcacs,pcacs_learned:rl,egoblind_learned:il@models/egoblind_only"
VARIANT_TITLE = {"bare": "bare", "gated": "gated", "adapter": "gated + adapter"}
HAVE_ACTIVITY = (ROOT / "seeforme" / "live" / "activity.py").exists()


# ---- the repeat metric ---------------------------------------------------------------------------------------------
def cue_key(cue, trk=None) -> str:
    """What the user hears the cue as being about: the object label, or the normalised text of a sign."""
    from seeforme.live.gate import normalize_text
    if getattr(cue, "kind", "object") == "text":
        raw = (getattr(trk, "text", None) if trk is not None else None) or cue.text or cue.message
        return "text:" + normalize_text(raw)
    return "object:" + (cue.label or "").lower()


def repeat_of(records: Sequence[Sequence], window_s: float = 20.0, closer_m: float = 1.0) -> List[Optional[int]]:
    """records: (t, key, sector, distance_m or None, ...) of the spoken cues in time order. A cue is a repeat when the
    most recent earlier cue with the same (key, sector) is at most window_s old and the object is not at least closer_m
    closer now than then (unknown distances count as not closer). Returns, per cue, the index of the cue it repeats
    or None."""
    last: Dict[Tuple[str, str], int] = {}
    out: List[Optional[int]] = []
    for i, r in enumerate(records):
        t, key, sec, d = r[0], r[1], r[2], r[3]
        j = last.get((key, sec))
        rep = None
        if j is not None and t - records[j][0] <= window_s + 1e-9:
            d_prev = records[j][3]
            closer = d_prev is not None and d is not None and (d_prev - d) >= closer_m - 1e-9
            rep = None if closer else j
        out.append(rep)
        last[(key, sec)] = i
    return out


def repeat_flags(records: Sequence[Sequence], window_s: float = 20.0, closer_m: float = 1.0) -> List[bool]:
    return [j is not None for j in repeat_of(records, window_s, closer_m)]


# ---- policy wrappers ---------------------------------------------------------------------------------------------
def make_wrappers():
    from seeforme.live.gate import AnnouncementGate

    class RecordingGate(AnnouncementGate):
        """AnnouncementGate that also logs every verdict with the cue's ground-truth source id (replays only)."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._verdicts: List[Dict] = []

        def reset(self) -> None:
            self._verdicts = []
            super().reset()

        def _judge(self, cue, trk, visible, t, deferred=False):
            out = super()._judge(cue, trk, visible, t, deferred)
            v = dict(self.last_verdict or {})
            v["source_id"] = cue.source_id
            self._verdicts.append(v)
            return out

    class AdaptedPolicy:
        """gate + ActivityDetector (track-based) + TaskAdapter, stepped at every decision.

        Order per decision: detector.update(None, t, tracks) -> adapter.step(activity, t, has_goal) (applies the mode's
        gate hold / background / read-mode muting on a change; never the scheduler's settings, as the LiveController)
        -> gate.decide (the scheduler shown its own channel; read mode: non-critical object cues held as muted)."""

        def __init__(self, gate, name: str):
            from seeforme.live.activity import ActivityDetector
            self.gate = gate
            self.name = name
            self.detector = ActivityDetector()
            self.adapter = None
            self.timeline: List[Tuple[float, str, str]] = []
            self.muted: List[Tuple[float, Optional[int]]] = []

        @property
        def decision_interval(self) -> float:
            return self.gate.decision_interval

        @property
        def supports_actions(self) -> bool:
            return self.gate.supports_actions

        @property
        def cfg(self):
            return self.gate.cfg

        def last_observations(self):
            return self.gate.last_observations()

        def reset(self) -> None:
            from seeforme.live.activity import TaskAdapter
            self.gate.reset()
            self.detector.reset()
            self.adapter = TaskAdapter(gate=self.gate, policy=None)   # applies 'navigate' to the gate at once
            self.timeline, self.muted = [], []

        def _pre(self, tracks, t) -> None:
            act = self.detector.update(None, t, tracks=tracks)
            has_goal = bool(getattr(self.gate.inner, "goal_tags", None))
            mode = self.adapter.step(act, t, has_goal=has_goal)
            self.timeline.append((t, mode, act.value))

        def _note_muted(self, n: int, t: float) -> None:
            for v in self.gate._verdicts[n:]:     # (the gate's read-mode holds of this decision)
                if v.get("reason") == "muted":
                    self.muted.append((t, v.get("source_id")))

        def decide(self, tracks, t, ch=None):
            self._pre(tracks, t)
            n = len(self.gate._verdicts)
            out = self.gate.decide(tracks, t, ch)
            self._note_muted(n, t)
            return out

        def select(self, tracks, t, ctx=None):
            self._pre(tracks, t)
            n = len(self.gate._verdicts)
            out = self.gate.select(tracks, t, ctx)
            self._note_muted(n, t)
            return out

    return RecordingGate, AdaptedPolicy


# ---- one replay ----------------------------------------------------------------------------------------------------
def run_one(policy_name: str, variant: str, sc, cfg, wrappers, repeat_window_s: float, closer_m: float,
            sector_deg: float) -> Dict:
    import egoblind_common as EC
    from seeforme.campus.metrics import evaluate_run
    from seeforme.egoblind import egoblind_extras, run_on_egoblind
    from seeforme.live.gate import GateConfig, sector_of
    from seeforme.types import bbox_center

    RecordingGate, AdaptedPolicy = wrappers
    pcfg = copy.deepcopy(cfg)        # (every run its own config: nothing one run changes leaks into another)
    pol = inner = EC.make_egoblind_policy(policy_name, pcfg, sc)
    gate = adapted = None
    if variant in ("gated", "adapter"):
        gate = pol = RecordingGate(inner, GateConfig(), name=f"{policy_name}+gate")
    if variant == "adapter":
        adapted = pol = AdaptedPolicy(gate, name=f"{policy_name}+gate+adapter")
    cam = sc.camera
    recs: List[Tuple[float, str, str, Optional[float]]] = []

    def on_decision(t, tracks, obs, cue, ch, due):
        if cue is None or cue.action == "haptic":
            return
        trk = next((tr for tr in tracks if tr.track_id == cue.track_id), None)
        u = bbox_center(trk.bbox)[0] if trk is not None else 0.5
        d = trk.distance_m if trk is not None and trk.distance_m is not None else cue.distance_m
        recs.append((float(t), cue_key(cue, trk), sector_of(cam.angle_deg_from_u(u), sector_deg),
                     None if d is None else float(d), cue.track_id, cue.message))

    log = run_on_egoblind(pol, sc, pcfg, seed=sc.seed, on_decision=on_decision)
    row = evaluate_run(log, sc)
    ex = egoblind_extras(log, sc)
    rof = repeat_of(recs, repeat_window_s, closer_m)
    passed_reason = {round(v["t"], 3): v["reason"] for v in (gate._verdicts if gate is not None else []) if v["passed"]}
    repeats = []
    for i, j in enumerate(rof):
        if j is None:
            continue
        a, b = recs[j], recs[i]
        repeats.append({"video": sc.video_id, "t": round(b[0], 2), "key": b[1], "sector": b[2],
                        "dist": None if b[3] is None else round(b[3], 2), "message": b[5],
                        "prev_t": round(a[0], 2), "prev_dist": None if a[3] is None else round(a[3], 2),
                        "prev_message": a[5], "same_track": a[4] == b[4],
                        "gate_reason": passed_reason.get(round(b[0], 3))})
    ev = {e.obj_id: e for e in sc.speak_events()}
    first: Dict[int, object] = {}
    for c in sorted(log.spoken, key=lambda c: c.t):
        if c.source_id in ev and c.source_id not in first:
            first[c.source_id] = c
    timely = {o for o, c in first.items() if ev[o].in_window(c.t)}
    out = {"row": row, "ex": ex, "n_spoken": len(log.spoken), "n_rec": len(recs), "n_repeat": len(repeats),
           "repeats": repeats,
           "timely": timely, "first": {o: (round(c.t, 2), c.message, c.action) for o, c in first.items()},
           "airtime_s": sum(b - a for a, b, c in log.channel_log if getattr(c, "policy", None) != "external")}
    if gate is not None:
        out["stats"] = dict(gate.stats)
        out["reasons"] = Counter(gate.reasons)
        out["verdicts"] = list(gate._verdicts)
        out["haptic_held"] = Counter(v["reason"] for v in gate._verdicts if v.get("action") == "haptic" and not v["passed"])
    if adapted is not None:
        out["timeline"] = list(adapted.timeline)
        out["muted"] = list(adapted.muted)
        out["mode_switches"] = max(0, len(adapted.adapter.history) - 1)
    return out


def pooled(results: List[Dict], dur: Dict[str, float], minutes: float) -> Dict:
    """Pooled over videos, the same computation as scripts/run_egoblind.py, plus the repeat / duplicate rates."""
    rows = [r["row"] for r in results]
    exs = [r["ex"] for r in results]
    cnt = lambda rate, n: sum(int(round(getattr(r, rate) * getattr(r, n))) for r in rows if getattr(r, n))
    cnt_e = lambda rate, n: sum(int(round(e[rate] * e[n])) for e in exs if e.get(n) and e.get(rate) is not None)
    div = lambda a, b: (a / b) if b else float("nan")
    n_ev, n_sp = sum(r.n_events for r in rows), sum(r.n_spoken for r in rows)
    n_cov, n_det, n_q = (sum(e[k] for e in exs) for k in ("n_coverable", "n_detected", "n_quiet"))
    occ = sum(r.audio_occupancy * dur[r.scenario] for r in rows) / sum(dur[r.scenario] for r in rows)
    n_rep, n_rec = sum(r["n_repeat"] for r in results), sum(r["n_rec"] for r in results)
    n_dup = sum(int(round(r.duplicate_rate * r.n_spoken)) for r in rows)
    n_timely = sum(len(r["timely"]) for r in results)
    n_timely_cov = cnt_e("coverable_recall", "n_coverable")
    return {"recall_all": div(cnt("important_cue_recall", "n_events"), n_ev),
            "recall_coverable": div(n_timely_cov, n_cov),
            "recall_detected": div(cnt_e("detected_recall", "n_detected"), n_det),
            "precision": div(cnt("cue_precision", "n_spoken"), n_sp),
            "quiet_violation": div(cnt_e("quiet_violation_rate", "n_quiet"), n_q),
            "occupancy": occ, "airtime_min": sum(r["airtime_s"] for r in results) / 60.0,
            "cues_per_min": n_sp / minutes, "haptics_per_min": sum(r.n_haptic for r in rows) / minutes,
            "repeat_rate": div(n_rep, n_rec), "n_repeats": n_rep, "repeats_per_min": n_rep / minutes,
            "dup_rate": div(n_dup, n_sp), "n_dups": n_dup, "dups_per_min": n_dup / minutes,
            "score_mean": sum(r.score for r in rows) / len(rows),
            "n_events": n_ev, "n_coverable": n_cov, "n_detected": n_det, "n_quiet": n_q, "n_spoken": n_sp,
            "n_timely": n_timely, "n_timely_coverable": n_timely_cov,
            "n_quiet_violated": cnt_e("quiet_violation_rate", "n_quiet"),
            "repeat_same_track": sum(1 for r in results for x in r["repeats"] if x["same_track"]),
            "repeat_gate_reasons": dict(Counter(x["gate_reason"] for r in results for x in r["repeats"] if x["gate_reason"])),
            "repeat_examples": [x for r in results for x in r["repeats"]][:20]}


def explain_change(sc, e, bare: Dict, var: Dict, variant: str, lost: bool) -> Dict:
    """Why an event's timeliness differs between the bare run and a variant run."""
    o = e.obj_id
    lo, hi = e.earliest_useful_s, e.latest_safe_s
    offers = [v for v in var.get("verdicts", []) if v.get("source_id") == o and lo - 1e-9 <= v["t"] <= hi + 1e-9]
    held = Counter(v["reason"] for v in offers if not v["passed"])
    muted = [t for t, sid in var.get("muted", []) if sid == o and lo <= t <= hi]
    b_first, v_first = bare["first"].get(o), var["first"].get(o)
    rec = {"video": sc.video_id, "obj_id": o, "label": e.label, "qtype": e.detail.get("qtype"),
           "coverable": bool(e.detail.get("coverable")), "severity": e.severity, "sudden": bool(e.sudden),
           "window": [lo, hi], "t_q": e.detail.get("t_q"), "question": e.detail.get("question"),
           "bare_first": b_first, "variant_first": v_first,
           "offers_in_window": [{"t": round(v["t"], 2), "passed": v["passed"], "reason": v["reason"], "info": v["info"]}
                                for v in offers][:12]}
    if lost:
        if v_first is not None and v_first[0] < lo:
            why = f"said before its window (t={v_first[0]:.1f} s < {lo:.1f} s), so the first cue was not timely"
        elif held:
            why = "held by the gate in its window: " + ", ".join(f"{k} x{n}" for k, n in held.most_common())
        elif muted:
            why = f"muted by the adapter's read mode x{len(muted)}"
        elif v_first is not None and v_first[0] > hi:
            why = f"said late (t={v_first[0]:.1f} s > {hi:.1f} s); the scheduler did not offer it in the window"
        else:
            why = ("never offered by the scheduler in its window (its state diverged from the bare run: other cues "
                   "spoken or held, budget, channel)")
    else:
        if b_first is not None and b_first[0] < lo:
            why = f"the bare run said it before its window (t={b_first[0]:.1f} s)"
        elif b_first is not None and b_first[0] > hi:
            why = f"the bare run said it late (t={b_first[0]:.1f} s)"
        else:
            why = "the bare run never said it (its channel / budget was spent on other cues)"
    rec["why"] = why
    if variant == "adapter" and var.get("timeline"):
        t_ref = (b_first[0] if (lost and b_first) else (v_first[0] if v_first else lo))
        prior = [x for x in var["timeline"] if x[0] <= t_ref + 1e-9]
        if prior:
            rec["mode_at_t"] = {"t": round(t_ref, 2), "mode": prior[-1][1], "activity": prior[-1][2]}
    return rec


# ---- report --------------------------------------------------------------------------------------------------------
def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def pct_change(a, b) -> str:
    if not a:
        return "n/a"
    return f"{100.0 * (b - a) / a:+.0f}%"


def write_md(summary: Dict, path: Path) -> str:
    meta = summary["_meta"]
    L: List[str] = []
    L.append("# Announcement gate on EgoBlind development replays\n")
    L.append(f"Generated {meta['generated']} by `{meta['command']}` ({meta['seconds']:.0f} s). "
             f"Splits: {', '.join(meta['splits'])}"
             + (" (development only; holdout and test were not loaded)." if meta["dev_only"] else
                " (INCLUDES FINAL SPLITS: evaluate holdout only once).") + "\n")
    L.append("Variants: **bare** is the scheduler alone; **gated** wraps it in `AnnouncementGate` with the default "
             "`GateConfig` (operator_filter off: the replays are walking clips, not a desk webcam); **gated + adapter** "
             "adds `ActivityDetector` in track-based mode (frame=None) and `TaskAdapter`, both stepped at every 0.5 s "
             "decision, with read-mode muting applied to the scheduler's cues before the gate.\n")
    L.append("Repeats are measured two ways. **Dup rate** (the main measure) is the `campus.metrics` duplicate rate: "
             "spoken cues about the same ground-truth object as a cue less than 6 s earlier, over all spoken cues; it does "
             "not depend on the gate's rules (only detections whose label an EgoBlind event names, inside that event's "
             "window, carry an object id, so repeats about unlabelled objects are not counted). The **label+sector repeat "
             f"rate** is secondary: the share of spoken cues whose (label, direction sector) was already spoken in the "
             f"previous {meta['repeat_window_s']:.0f} s while the object was not at least {meta['repeat_closer_m']:.0f} m "
             f"closer (sectors at +-{meta['sector_deg']:.0f} deg; unknown distance counts as not closer). It restates the "
             "gate's own rule, so the gated rows score well on it by construction. Recall, precision and quiet-window "
             "violations are pooled over events, cues and windows exactly as in `scripts/run_egoblind.py`; precision is "
             "a lower bound (only the questioned objects are labelled).\n")
    L.append("These replays are walking / hand-held clips: none is a desk-webcam session, so they neither show nor test "
             "the live \"Person.\" loop at a desk (operator_filter is off here). That case is covered by "
             "`tests/test_live_gate.py` (real tracker and dynamics, the user leaning in with depth jitter).\n")
    L.append("## Reading\n")
    L.extend(f"- {s}" for s in summary["reading"]["summary"])
    L.append("")
    for cache, cs in summary["caches"].items():
        m = cs["_meta"]
        L.append(f"## {CACHE_NAMES.get(cache, cache)}\n")
        L.append(f"{m['n_videos']} videos, {m['minutes']:.1f} min, {m['n_events']} should-speak events "
                 f"({m['n_coverable']} coverable by this detector), {m['n_quiet']} quiet windows "
                 f"(scenario files sha256 {m['scenario_sha256']}, newest {m['scenarios_newest']}).\n")
        for pol, ps in cs["policies"].items():
            L.append(f"### `{pol}`\n")
            L.append("| variant | recall (coverable) | recall (all) | precision | quiet-window violations | cues/min | "
                     "**dup rate (6 s)** | dups | label+sector repeat rate (gate's rule) | repeats | audio occupancy | "
                     "airtime (min) | haptics/min | score (mean) |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for v, s in ps["variants"].items():
                L.append(f"| {VARIANT_TITLE[v]} | {fmt(s['recall_coverable'], 3)} ({s['n_timely_coverable']}/{s['n_coverable']}) | "
                         f"{fmt(s['recall_all'], 3)} ({s['n_timely']}/{s['n_events']}) | {fmt(s['precision'], 3)} | "
                         f"{fmt(s['quiet_violation'], 3)} ({s['n_quiet_violated']}/{s['n_quiet']}) | {fmt(s['cues_per_min'])} | "
                         f"**{fmt(s['dup_rate'], 3)}** | {s['n_dups']} | {fmt(s['repeat_rate'], 3)} | {s['n_repeats']} | "
                         f"{fmt(s['occupancy'], 3)} | {fmt(s['airtime_min'])} | {fmt(s['haptics_per_min'])} | {fmt(s['score_mean'], 3)} |")
            gv = {v: s for v, s in ps["variants"].items() if "gate_stats" in s}
            if gv:
                L.append("")
                L.append("| gate counters (sum over videos) | passed | held: repeat | held: background | held: unconfirmed | "
                         "held: haptic | pass / hold reasons |")
                L.append("|---|---|---|---|---|---|---|")
                for v, s in gv.items():
                    g = s["gate_stats"]
                    rs = ", ".join(f"{k} {n}" for k, n in sorted(s["gate_reasons"].items(), key=lambda kv: -kv[1]))
                    L.append(f"| {VARIANT_TITLE[v]} | {g['passed']} | {g['suppressed_repeat']} | {g['suppressed_background']} | "
                             f"{g['suppressed_unconfirmed']} | {g['suppressed_haptic']} | {rs} |")
            L.append("")
            for v, s in ps["variants"].items():
                rr = ", ".join(f"{k} {n}" for k, n in sorted(s["repeat_gate_reasons"].items(), key=lambda kv: -kv[1]))
                L.append(f"- {VARIANT_TITLE[v]}: {s['n_dups']} dups (6 s); {s['n_repeats']} label+sector repeats, "
                         f"{s['repeat_same_track']} on the same track id"
                         + (f"; the gate passed them as: {rr}" if rr else "")
                         + (f"; haptics held by the gate: " + ", ".join(f"{k} {n}" for k, n in s["haptics_held"].items())
                            if s.get("haptics_held") else ""))
            ad = ps["variants"].get("adapter")
            if ad and ad.get("modes"):
                modes = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(ad["modes"].items(), key=lambda kv: -kv[1]))
                acts = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(ad["activities"].items(), key=lambda kv: -kv[1]))
                eps = (f"; {ad['still_episodes']} scene / read episodes (median {fmt(ad['still_episode_median_s'], 1)} s, "
                       f"max {fmt(ad['still_episode_max_s'], 1)} s), each entry rebaselines the gate" if ad.get("still_episodes") else "")
                read_note = ("" if "reading" in ad["activities"] else
                             " (read mode cannot trigger here: the cached replays have no OCR text tracks)")
                L.append(f"\nAdapter: decision steps per mode {modes}; activity {acts}; {ad['mode_switches']} mode changes "
                         f"({ad['mode_switches'] / m['minutes']:.2f}/min){eps}; {ad['n_muted']} cues muted by read mode{read_note}.")
            for v in ("gated", "adapter"):
                if v not in ps["variants"]:
                    continue
                lost, gained = ps["lost_vs_bare"].get(v, []), ps["gained_vs_bare"].get(v, [])
                tm = ps.get("timing_vs_bare", {}).get(v, {})
                tdesc = ""
                if tm.get("n_shared"):
                    tdesc = (f" Of the {tm['n_shared']} events timely in both, {tm['n_later']} were said later and "
                             f"{tm['n_earlier']} earlier than bare (mean shift {tm['mean_delay_s']:+.2f} s, max {tm['max_delay_s']:+.2f} s).")
                L.append(f"\n**{VARIANT_TITLE[v]} vs bare:** {len(lost)} timely event(s) lost, {len(gained)} gained.{tdesc}")
                for tag, lst in (("lost", lost), ("gained", gained)):
                    for r in lst:
                        bf = f"bare said {r['bare_first'][1]!r} at {r['bare_first'][0]:.1f} s" if r["bare_first"] else "bare never said it"
                        vf = f"{VARIANT_TITLE[v]} said {r['variant_first'][1]!r} at {r['variant_first'][0]:.1f} s" if r["variant_first"] else f"{VARIANT_TITLE[v]} never said it"
                        mode = f"; adapter mode {r['mode_at_t']['mode']} ({r['mode_at_t']['activity']})" if r.get("mode_at_t") else ""
                        L.append(f"- {tag}: video {r['video']}, {r['label']} ({r['qtype']}, "
                                 f"{'coverable' if r['coverable'] else 'not coverable'}), window {r['window'][0]:.1f}-{r['window'][1]:.1f} s: "
                                 f"{bf}; {vf}. Why: {r['why']}{mode}.")
            L.append("")
    L.append("## Per comparison\n")
    L.extend(f"- {s}" for s in summary["reading"]["per_comparison"])
    L.append("")
    text = "\n".join(L) + "\n"
    path.write_text(text)
    return text


def reading(summary: Dict) -> Dict[str, List[str]]:
    """Statements computed from the numbers (a rerun cannot leave a stale interpretation behind):
    'summary' (the short reading) and 'per_comparison' (one line per cache x policy x variant)."""
    per: List[str] = []
    agg: Dict[str, Dict] = {v: defaultdict(list) for v in ("gated", "adapter")}
    by_policy: Dict[str, List[str]] = defaultdict(list)
    bare_rep = bare_rep_other = 0
    bare_rep_pol: Dict[str, int] = defaultdict(int)
    nav_share: List[float] = []
    for cache, cs in summary["caches"].items():
        for pol, ps in cs["policies"].items():
            b = ps["variants"].get("bare")
            if b is None:
                continue
            bare_rep += b["n_repeats"]; bare_rep_other += b["n_repeats"] - b["repeat_same_track"]
            bare_rep_pol[pol] += b["n_repeats"]
            if "adapter" in ps["variants"]:
                nav_share.append(ps["variants"]["adapter"].get("modes", {}).get("navigate", 0.0))
            for v in ("gated", "adapter"):
                s = ps["variants"].get(v)
                if s is None:
                    continue
                lost, gained = ps["lost_vs_bare"].get(v, []), ps["gained_vs_bare"].get(v, [])
                tm = ps.get("timing_vs_bare", {}).get(v, {})
                a = agg[v]
                a["n"].append(1); a["lost"] += lost; a["gained"] += gained
                a["later"].append(tm.get("n_later", 0)); a["max_delay"].append(tm.get("max_delay_s") or 0.0)
                a["quiet_same"].append(s["n_quiet_violated"] == b["n_quiet_violated"])
                a["quiet_more"].append(s["n_quiet_violated"] > b["n_quiet_violated"])
                a["cues"].append(100.0 * (s["cues_per_min"] - b["cues_per_min"]) / b["cues_per_min"] if b["cues_per_min"] else 0.0)
                a["air"].append(100.0 * (s["airtime_min"] - b["airtime_min"]) / b["airtime_min"] if b["airtime_min"] else 0.0)
                a["rep_b"].append(b["n_repeats"]); a["rep_v"].append(s["n_repeats"])
                a["dup_b"].append(b["n_dups"]); a["dup_v"].append(s["n_dups"])
                a["prec_up"].append(s["precision"] >= b["precision"] - 1e-12 if s["n_spoken"] and b["n_spoken"] else True)
                a["reasons"].append(Counter(s.get("repeat_gate_reasons", {})))
                if v == "gated":
                    by_policy[pol].append(f"{cache} {pct_change(b['cues_per_min'], s['cues_per_min'])}")
                ev = ("no timely event lost" if not lost else
                      f"{len(lost)} timely event(s) lost (" + ", ".join("video %s %s" % (r["video"], r["label"]) for r in lost) + ")")
                if gained:
                    ev += f", {len(gained)} gained"
                per.append(f"{cache} / `{pol}` / {VARIANT_TITLE[v]}: cues/min {fmt(b['cues_per_min'])} -> {fmt(s['cues_per_min'])} "
                           f"({pct_change(b['cues_per_min'], s['cues_per_min'])}), airtime {fmt(b['airtime_min'])} -> "
                           f"{fmt(s['airtime_min'])} min ({pct_change(b['airtime_min'], s['airtime_min'])}), dup rate "
                           f"{fmt(b['dup_rate'], 3)} -> {fmt(s['dup_rate'], 3)} ({b['n_dups']} -> {s['n_dups']} dups), "
                           f"label+sector repeat rate {fmt(b['repeat_rate'], 3)} -> {fmt(s['repeat_rate'], 3)} "
                           f"({b['n_repeats']} -> {s['n_repeats']}), "
                           f"coverable recall {fmt(b['recall_coverable'], 3)} -> {fmt(s['recall_coverable'], 3)}, precision "
                           f"{fmt(b['precision'], 3)} -> {fmt(s['precision'], 3)}, quiet-window violations "
                           f"{b['n_quiet_violated']} -> {s['n_quiet_violated']}; {ev}.")
    out: List[str] = []
    for v in ("gated", "adapter"):
        a = agg[v]
        n = len(a["n"])
        if not n:
            continue
        lost, gained = a["lost"], a["gained"]
        lost_txt = ("no timely event lost" if not lost else
                    f"{len(lost)} timely event(s) lost ({sum(1 for r in lost if r['coverable'])} coverable): " +
                    "; ".join(f"video {r['video']} {r['label']} ({r['why']})" for r in lost))
        reasons = sum(a["reasons"], Counter())
        gk = Counter("bare said it late" if "late" in r["why"] else "bare said it before the window" if "before" in r["why"]
                     else "bare never said it" for r in gained)
        gained_txt = f"{len(gained)} gained" + (" (" + ", ".join(f"{k} x{c}" for k, c in gk.items()) + ")" if gained else "")
        out.append(
            f"**{VARIANT_TITLE[v]}** vs bare, {n} cache x policy comparisons: {lost_txt}; {gained_txt}; "
            f"{sum(a['later'])} events timely in both runs said later "
            f"(max shift {max(a['max_delay']):+.1f} s); quiet-window violations unchanged in {sum(a['quiet_same'])}/{n} "
            f"(more in {sum(a['quiet_more'])}). Cues/min {min(a['cues']):+.0f}% to {max(a['cues']):+.0f}%, airtime "
            f"{min(a['air']):+.0f}% to {max(a['air']):+.0f}%, dups (same object within 6 s) {sum(a['dup_b'])} -> "
            f"{sum(a['dup_v'])} ({pct_change(sum(a['dup_b']), sum(a['dup_v']))}), label+sector repeats (the gate's own "
            f"rule) {sum(a['rep_b'])} -> {sum(a['rep_v'])} ({pct_change(sum(a['rep_b']), sum(a['rep_v']))}); precision "
            f"not lower in {sum(a['prec_up'])}/{n}."
            + (f" The remaining label+sector repeats passed the gate as: " + ", ".join(f"{k} {c}" for k, c in reasons.most_common()) + "."
               if reasons else ""))
    cut_pol: Dict[str, List[float]] = defaultdict(list)
    for cs in summary["caches"].values():
        for pol, ps in cs["policies"].items():
            b, g = ps["variants"].get("bare"), ps["variants"].get("gated")
            if b and g and b["cues_per_min"]:
                cut_pol[pol].append((g["cues_per_min"] - b["cues_per_min"]) / b["cues_per_min"])
    top_cut = min(cut_pol, key=lambda p: sum(cut_pol[p]) / len(cut_pol[p])) if cut_pol else None
    top_rep = max(bare_rep_pol, key=bare_rep_pol.get) if bare_rep_pol else None
    if by_policy:
        out.append("Gated cues/min change by policy: " + "; ".join(f"`{p}` " + ", ".join(x) for p, x in by_policy.items())
                   + ". Label+sector repeats without the gate, summed over the caches: "
                   + ", ".join(f"`{p}` {n}" for p, n in bare_rep_pol.items())
                   + (": the gate removes the most where the scheduler repeats itself the most." if top_cut == top_rep else "."))
    if bare_rep:
        out.append(f"{bare_rep_other} of the {bare_rep} label+sector repeats without the gate ({100.0 * bare_rep_other / bare_rep:.0f}%) are "
                   "about a different track id than the cue they repeat (tracker id switches, or a second object with the same "
                   "label in the same direction): the scheduler's per-track memory cannot see them, the gate's identity memory can.")
    ga, aa = agg["gated"], agg["adapter"]
    if ga["n"] and aa["n"]:
        more = sum(1 for x, y in zip(aa["cues"], ga["cues"]) if x > y + 1e-9)
        nav = (sum(nav_share) / len(nav_share)) if nav_share else float("nan")
        out.append(f"The adapter keeps more cues than the plain gate in {more}/{len(aa['n'])} comparisons: on these replays "
                   f"it is in navigate mode (background rule off, 20 s hold) for {100 * nav:.0f}% of the decision steps "
                   "(track-based activity; the adapter lines give how often it changes mode).")
    bare = [ps["variants"]["bare"] for cs in summary["caches"].values() for ps in cs["policies"].values() if "bare" in ps["variants"]]
    if bare:
        dpm = [b["dups_per_min"] for b in bare]
        rpm = [b["repeats_per_min"] for b in bare]
        out.append(f"Without the gate the schedulers say the same object again within 6 s {min(dpm):.2f}-{max(dpm):.2f} times "
                   f"per minute on these replays ({min(rpm):.2f}-{max(rpm):.2f} label+sector repeats per minute); they are "
                   "hand-held / head-worn clips, mostly walking, where objects leave the view quickly. None of them is a "
                   "desk-webcam session (the user's own body in view for minutes, operator_filter on), so they do not show "
                   "the live \"Person.\" loop and say nothing about how much the gate saves there: they measure what the gate "
                   "costs (lost or delayed events) and removes on real walking footage.")
    return {"summary": out, "per_comparison": per}


# ---- main ----------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default=",".join(DEV_SPLITS), help="comma list of EgoBlind splits (default finetune,select)")
    ap.add_argument("--allow-final", action="store_true", help="permit the holdout / test splits (evaluate holdout ONCE)")
    ap.add_argument("--policies", default=DEFAULT_POLICIES)
    ap.add_argument("--variants", default="bare,gated,adapter" if HAVE_ACTIVITY else "bare,gated")
    ap.add_argument("--caches", default="coco,ow", help="coco (data/egoblind/cache), ow (data/egoblind/cache_ow) or paths")
    ap.add_argument("--limit", type=int, default=0, help="first N videos per split (smoke tests; needs --no-write or --out)")
    ap.add_argument("--repeat-window", type=float, default=20.0)
    ap.add_argument("--repeat-closer", type=float, default=1.0)
    ap.add_argument("--sector-deg", type=float, default=20.0)
    ap.add_argument("--out", default="", help="output stem (default results/live/gate_eval_dev or gate_eval_<splits>)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    final = [s for s in splits if s in FINAL_SPLITS]
    if final and not args.allow_final:
        ap.error(f"split(s) {final} are reserved for the final evaluation; pass --allow-final to run them "
                 "(holdout must be evaluated only once)")
    if final:
        print(f"WARNING: evaluating final split(s) {final}. The holdout split must be evaluated only once, at the very end; "
              "do not tune anything on these numbers.", flush=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANT_TITLE]
    if bad:
        ap.error(f"unknown variant(s) {bad}; choose from {list(VARIANT_TITLE)}")
    if "adapter" in variants and not HAVE_ACTIVITY:
        print("seeforme/live/activity.py not found: skipping the adapter variant")
        variants = [v for v in variants if v != "adapter"]
    if "bare" not in variants:
        variants = ["bare"] + variants       # the lost / gained comparison needs the bare run
    dev_only = set(splits) <= set(DEV_SPLITS)
    stem = Path(args.out) if args.out else ROOT / "results" / "live" / ("gate_eval_dev" if dev_only else "gate_eval_" + "_".join(splits))
    if args.limit and not (args.no_write or args.out):
        ap.error("--limit changes the results: add --no-write (or an explicit --out) so the full report is not overwritten")

    os.chdir(ROOT)                           # model specs such as egoblind_learned:il@models/egoblind_only are relative
    import egoblind_common as EC
    from campus_common import load_campus_cfg
    from seeforme.live.gate import GateConfig
    t_start = time.time()
    cfg = load_campus_cfg()
    wrappers = make_wrappers()
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    summary: Dict = {"caches": {}}
    for cache in [c.strip() for c in args.caches.split(",") if c.strip()]:
        cdir = CACHES.get(cache, Path(cache))
        if not Path(cdir).exists():
            print(f"skip cache {cache}: {cdir} not found")
            continue
        EC.CACHE = Path(cdir)                # load_egoblind_split reads the module-level CACHE at call time
        scenarios = []
        for s in splits:
            scenarios += EC.load_egoblind_split(s, require_cache=True, limit=args.limit)
        if not scenarios:
            print(f"skip cache {cache}: no cached scenarios for {splits}")
            continue
        import hashlib
        h = hashlib.sha256()
        newest = 0.0
        for s_ in splits:
            for f in sorted((EC.SCENARIOS / s_).glob("*.json")):
                h.update(f.name.encode()); h.update(f.read_bytes())
                newest = max(newest, f.stat().st_mtime)
        dur = {sc.name: sc.duration for sc in scenarios}
        minutes = sum(dur.values()) / 60.0
        n_events = sum(len(sc.speak_events()) for sc in scenarios)
        n_cov = sum(1 for sc in scenarios for e in sc.speak_events() if e.detail.get("coverable"))
        n_quiet = sum(len(sc.quiet_events()) for sc in scenarios)
        print(f"[{cache}] {len(scenarios)} videos, {minutes:.1f} min, {n_events} events ({n_cov} coverable), {n_quiet} quiet windows", flush=True)
        cs = {"_meta": {"cache_dir": str(Path(cdir).relative_to(ROOT)) if str(cdir).startswith(str(ROOT)) else str(cdir),
                        "n_videos": len(scenarios), "minutes": minutes, "n_events": n_events, "n_coverable": n_cov,
                        "n_quiet": n_quiet, "videos": sorted(sc.video_id for sc in scenarios),
                        "scenario_sha256": h.hexdigest()[:16],
                        "scenarios_newest": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest))},
              "policies": {}}
        for pol in policies:
            per_var: Dict[str, List[Dict]] = {}
            ps = {"variants": {}, "lost_vs_bare": {}, "gained_vs_bare": {}}
            for v in variants:
                t0 = time.time()
                try:
                    res = [run_one(pol, v, sc, cfg, wrappers, args.repeat_window, args.repeat_closer, args.sector_deg)
                           for sc in scenarios]
                except FileNotFoundError as e:
                    print(f"skip {pol}: {e}")
                    break
                per_var[v] = res
                s = pooled(res, dur, minutes)
                if v != "bare":
                    st = Counter(); rs = Counter()
                    for r in res:
                        st.update(r["stats"]); rs.update(r["reasons"])
                    s["gate_stats"] = {k: int(st.get(k, 0)) for k in ("passed", "suppressed_repeat", "suppressed_background",
                                                                      "suppressed_unconfirmed", "suppressed_haptic")}
                    s["gate_reasons"] = dict(rs)
                    s["haptics_held"] = dict(sum((r["haptic_held"] for r in res), Counter()))
                if v == "adapter":
                    modes = Counter(x[1] for r in res for x in r["timeline"])
                    acts = Counter(x[2] for r in res for x in r["timeline"])
                    n = sum(modes.values()) or 1
                    s["modes"] = {k: c / n for k, c in modes.items()}
                    s["activities"] = {k: c / n for k, c in acts.items()}
                    s["mode_switches"] = sum(r["mode_switches"] for r in res)
                    s["n_muted"] = sum(len(r["muted"]) for r in res)
                    eps = []                 # still-mode (scene / read) episodes: how long the adapter stays settled
                    for r in res:
                        start = None
                        for (t_, mode_, _a) in r["timeline"]:
                            if mode_ in ("scene", "read") and start is None:
                                start = t_
                            elif mode_ not in ("scene", "read") and start is not None:
                                eps.append(t_ - start); start = None
                        if start is not None:
                            eps.append(r["timeline"][-1][0] - start + 0.5)
                    eps.sort()
                    s["still_episodes"] = len(eps)
                    s["still_episode_median_s"] = eps[len(eps) // 2] if eps else None
                    s["still_episode_max_s"] = eps[-1] if eps else None
                ps["variants"][v] = s
                print(f"  {pol:42s} {VARIANT_TITLE[v]:16s} recall cov/all={fmt(s['recall_coverable'], 3)}/{fmt(s['recall_all'], 3)} "
                      f"prec={fmt(s['precision'], 3)} quiet={s['n_quiet_violated']}/{s['n_quiet']} cues/min={fmt(s['cues_per_min'])} "
                      f"occ={fmt(s['occupancy'], 3)} dup={fmt(s['dup_rate'], 3)} ({s['n_dups']}) "
                      f"repeat={fmt(s['repeat_rate'], 3)} ({s['n_repeats']}) "
                      f"{'' if v == 'bare' else 'gate=' + json.dumps(s['gate_stats'])} [{time.time() - t0:.0f} s]", flush=True)
            if "bare" not in per_var:
                continue
            for v in per_var:
                if v == "bare":
                    continue
                lost, gained = [], []
                for sc, b, r in zip(scenarios, per_var["bare"], per_var[v]):
                    ev = {e.obj_id: e for e in sc.speak_events()}
                    lost += [explain_change(sc, ev[o], b, r, v, True) for o in sorted(b["timely"] - r["timely"])]
                    gained += [explain_change(sc, ev[o], b, r, v, False) for o in sorted(r["timely"] - b["timely"])]
                ps["lost_vs_bare"][v], ps["gained_vs_bare"][v] = lost, gained
                delays = []
                for sc, b, r in zip(scenarios, per_var["bare"], per_var[v]):
                    for o in sorted(b["timely"] & r["timely"]):
                        delays.append({"video": sc.video_id, "obj_id": o, "bare_t": b["first"][o][0],
                                       "variant_t": r["first"][o][0], "delay_s": round(r["first"][o][0] - b["first"][o][0], 2)})
                ps.setdefault("timing_vs_bare", {})[v] = {
                    "n_shared": len(delays), "n_later": sum(1 for d in delays if d["delay_s"] > 0.01),
                    "n_earlier": sum(1 for d in delays if d["delay_s"] < -0.01),
                    "mean_delay_s": (sum(d["delay_s"] for d in delays) / len(delays)) if delays else None,
                    "max_delay_s": max((d["delay_s"] for d in delays), default=None),
                    "changed": [d for d in delays if abs(d["delay_s"]) > 0.01]}
                for x in lost:
                    print(f"    {VARIANT_TITLE[v]} LOST video {x['video']} {x['label']} window {x['window']}: {x['why']}")
                for x in gained:
                    print(f"    {VARIANT_TITLE[v]} gained video {x['video']} {x['label']}: {x['why']}")
            cs["policies"][pol] = ps
        summary["caches"][cache] = cs
    if not summary["caches"]:
        raise SystemExit("nothing evaluated (no cache found)")
    mode_settings = None
    if HAVE_ACTIVITY:
        from seeforme.live.activity import MODE_SETTINGS
        mode_settings = {k: asdict(v) for k, v in MODE_SETTINGS.items()}
    argv_list = list(sys.argv[1:] if argv is None else argv)
    summary["_meta"] = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "seconds": time.time() - t_start,
        "command": " ".join(["python scripts/eval_live_gate.py", *argv_list]),
        "splits": splits, "dev_only": dev_only, "policies": policies, "variants": variants,
        "repeat_window_s": args.repeat_window, "repeat_closer_m": args.repeat_closer, "sector_deg": args.sector_deg,
        "gate_config": asdict(GateConfig()),
        "mode_settings": mode_settings,
    }
    summary["reading"] = reading(summary)
    print("\n".join("READING: " + r for r in summary["reading"]["summary"]))
    if args.no_write:
        return 0
    stem.parent.mkdir(parents=True, exist_ok=True)
    jpath, mpath = stem.with_suffix(".json"), stem.with_suffix(".md")
    jpath.write_text(json.dumps(summary, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else float(o)))
    write_md(summary, mpath)
    print(f"wrote {jpath.relative_to(ROOT) if str(jpath).startswith(str(ROOT)) else jpath} and {mpath.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
