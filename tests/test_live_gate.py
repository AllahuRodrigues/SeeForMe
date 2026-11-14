"""AnnouncementGate (seeforme/live/gate.py): the live demo's memory of what was already said.

A worst-case fake scheduler offers a cue for the nearest track at every decision step (2 per second); the gate
must turn that into one announcement per object, survive tracker id switches, and still let approaching,
moving-to-another-side and critical objects through. No camera, no models."""
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from seeforme.campus.metrics import evaluate_run
from seeforme.config import PipelineConfig
from seeforme.cues.pcacs import ChannelState, PCACSConfig, PCACSPolicy
from seeforme.egoblind import egoblind_extras, run_on_egoblind
from seeforme.types import Cue, Track

try:
    from seeforme.live.gate import AnnouncementGate, GateConfig
    from seeforme.live import gate as _mod
except ImportError:   # seeforme/live/__init__.py also imports activity.py (built separately): load gate.py on its own
    _mod = sys.modules.get("seeforme.live.gate")
    if _mod is None:
        _spec = importlib.util.spec_from_file_location("seeforme.live.gate",
                                                       Path(__file__).resolve().parents[1] / "seeforme" / "live" / "gate.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["seeforme.live.gate"] = _mod
        _spec.loader.exec_module(_mod)
    AnnouncementGate, GateConfig = _mod.AnnouncementGate, _mod.GateConfig
measured, same_sign, _Said = _mod.measured, _mod.same_sign, _mod._Said

DT = 0.5   # decision interval: 2 decisions per second


class NearestPolicy:
    """Worst-case scheduler: a cue for the nearest track at every decision (optionally skipping tracks it announced in
    the last `recent_block_s`, like P-CACS), marking the track as announced and charging its speech budget."""
    name = "nearest"
    decision_interval = DT
    supports_actions = True

    def __init__(self, action="speak", recent_block_s=0.0):
        self.cfg = PCACSConfig()
        self.action, self.recent_block_s = action, recent_block_s
        self.budget = self.cfg.budget_capacity_s
        self.resets, self.goal = 0, None
        self.goal_tags = []

    def reset(self):
        self.resets += 1
        self.budget = self.cfg.budget_capacity_s

    def last_observations(self):
        return ["obs"]

    def set_goal(self, tags):
        self.goal = tags
        self.goal_tags = [g.lower() for g in (tags or [])]      # like PCACSPolicy.set_goal

    def decide(self, tracks, t, ch=None):
        cands = [tr for tr in tracks if self.action == "haptic" or tr.last_announced is None
                 or t - tr.last_announced >= self.recent_block_s]
        if not cands:
            return []
        tr = min(cands, key=lambda x: x.distance_m)
        if self.action == "haptic":
            tr.last_haptic = t
        else:
            tr.last_announced, tr.announce_count = t, tr.announce_count + 1
            self.budget -= 1.2
        return [Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind, message=f"{tr.label}, {tr.distance_m:.1f} m.",
                    score=1.0, policy=self.name, text=tr.text, action=self.action, duration_s=1.2 if self.action != "haptic" else 0.0,
                    distance_m=tr.distance_m, ttc_s=tr.ttc_s if math.isfinite(tr.ttc_s) else None, severity=0.5, utility=1.0)]

    def select(self, tracks, t, ctx=None):
        return [c for c in self.decide(tracks, t) if c.action != "haptic"]


class SelectOnly:
    name = "select_only"
    decision_interval = DT

    def reset(self):
        pass

    def select(self, tracks, t, ctx=None):
        return [Cue(t=t, track_id=tr.track_id, label=tr.label, kind=tr.kind, message=tr.label, score=1.0, policy=self.name)
                for tr in tracks[:1]]


class Scene:
    """Persistent tracks driven by a script; a new track starts at 2 hits (the tracker confirms at 2) and gains 5 per
    decision (10 fps), with depth and box history (as if seen for 1.2 s at its first distance and box) so speeds and
    TTC count as measured and the warm-up can judge it static."""

    def __init__(self):
        self.tracks = {}

    def at(self, t, specs):
        out = []
        for s in specs:
            trk = self.tracks.get(s["tid"])
            if trk is None:
                trk = Track(track_id=s["tid"], label=s.get("label", "person"), kind=s.get("kind", "object"), bbox=s["bbox"],
                            confidence=0.9, first_seen=t, last_seen=t, text=s.get("text"), hits=2, confirmed=True)
                for k in range(4):
                    trk.dist_history.append((t - 1.2 + 0.3 * k, s.get("dist", 3.0), 0.0))
                    trk.bbox_history.append((t - 1.2 + 0.3 * k, s["bbox"]))
                self.tracks[s["tid"]] = trk
            else:
                trk.hits += 5
            if "hits" in s:
                trk.hits = s["hits"]
            trk.bbox, trk.last_seen, trk.text = s["bbox"], t, s.get("text", trk.text)
            trk.bbox_history.append((t, trk.bbox))
            trk.distance_m, trk.depth_conf = s.get("dist", 3.0), s.get("depth_conf", 0.9)
            trk.rel_speed_mps, trk.lateral_speed_mps = s.get("rel", 0.0), s.get("lat", 0.0)
            trk.ttc_s, trk.path_prob = s.get("ttc", math.inf), s.get("path", 0.2)
            trk.dist_history.append((t, trk.distance_m, 0.0))
            out.append(trk)
        return out


def run(gate, script, t_end, t0=0.0):
    """Drive the gate over `script(t) -> track specs` and return the cues it let through."""
    scene, cues = Scene(), []
    n = int(round((t_end - t0) / DT))
    for i in range(n + 1):
        t = round(t0 + i * DT, 3)
        out = gate.decide(scene.at(t, script(t)), t, ChannelState())
        assert len(out) <= 1
        cues += out
    return cues


def box(cx, h=0.7, w=0.2, top=0.2):
    return (cx - w / 2, top, cx + w / 2, top + h)


AHEAD = box(0.5)


def test_static_person_said_once_and_reminded_only_after_hold():
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=1.5)], 40.0)
    assert len([c for c in cues if c.t < 30.0]) == 1          # 60 offers in 30 s, one spoken
    assert len(cues) == 2 and cues[1].t - cues[0].t >= gate.gate_cfg.hold_static_s
    assert gate.reasons["reminder"] == 1 and gate.stats["suppressed_repeat"] > 70
    assert cues[0].t <= 0.5                                    # confirmation costs at most one decision step


def test_track_id_switch_every_3s_is_still_one_person():
    gate, ids = AnnouncementGate(NearestPolicy(), GateConfig(background=False)), set()
    def scene(t):
        ids.add(1 + int(t // 3))
        return [dict(tid=1 + int(t // 3), bbox=AHEAD, dist=1.5)]
    cues = run(gate, scene, 29.5)
    assert len(cues) == 1 and len(ids) == 10                    # ten track ids, one announcement
    # a different car 30 s after the last one in the same place is a new car, not an id switch
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, label="car", bbox=AHEAD, dist=8.0)] if t < 5 else
               ([dict(tid=2, label="car", bbox=AHEAD, dist=8.0)] if t >= 25 else []), 30.0)
    assert [c.track_id for c in cues] == [1, 2]


def test_two_people_in_the_same_sector_are_two_objects():
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.45, h=0.5), dist=2.0)]
               + ([dict(tid=2, bbox=box(0.62, h=0.3), dist=4.0)] if t >= 5 else []), 20.0)
    assert sorted(c.track_id for c in cues) == [1, 2]


def test_approaching_person_reannounced_when_materially_closer():
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    z = lambda t: 6.0 - 0.5 * t
    cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.5, h=min(0.95, 1.6 / z(t))), dist=z(t), rel=-0.5,
                                     ttc=(z(t) - 0.6) / 0.5, path=0.3)], 9.0)
    d = [c.distance_m for c in cues]
    assert len(cues) >= 3 and gate.reasons["closer"] >= 2
    assert all(a - b >= gate.gate_cfg.closer_m for a, b in zip(d, d[1:]))     # each one materially closer
    assert d[0] > 5.0 and d[-1] <= 2.5


def test_critical_passes_rate_limited_but_a_worsening_hazard_always_passes():
    # a hazard that stays critical (measured TTC 1 s, closing 1 m/s) passes at once, even unconfirmed and with the
    # background rule on, then at most once per critical_gap_s (it used to pass at all 21 decisions in 10 s)
    gate = AnnouncementGate(NearestPolicy())
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=1.6, rel=-1.0, ttc=1.0, path=0.9, hits=1)], 10.0)
    ts = [c.t for c in cues]
    assert ts[0] == 0.0 and len(ts) == 6 and all(b - a >= gate.gate_cfg.critical_gap_s for a, b in zip(ts, ts[1:]))
    assert set(gate.reasons) == {"critical", "critical_gap"} and gate.stats["suppressed_repeat"] == 15
    # walking into it (1 m/s; said at once: in the path): every decision the TTC drops 0.5 s, then (TTC 0) the distance
    # 0.5 m: each one passes;
    # once it stops getting worse it is held for critical_gap_s
    z = lambda t: max(0.3, 3.0 - t)
    gate = AnnouncementGate(NearestPolicy())
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=z(t), rel=-1.0, ttc=max(0.0, z(t) - 0.6), path=0.9)], 3.5)
    assert [c.t for c in cues] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] and gate.reasons["critical_gap"] == 2
    # without depth the dynamics *assume* a 1 m/s walk (TTC 1.0 s for a static person): not critical, said once
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=1.6, ttc=1.0, path=0.9, depth_conf=0.25)], 10.0)
    assert len(cues) == 1 and gate.reasons["critical"] == 0 and gate.reasons["near"] == 0


def test_measured_needs_depth_and_a_one_second_span():
    trk = Track(track_id=1, label="person", kind="object", bbox=AHEAD, confidence=0.9, first_seen=0.0, last_seen=0.3)
    trk.depth_conf = 0.9
    for k in range(4):
        trk.dist_history.append((0.1 * k, 2.0, 0.0))             # 4 samples over 0.3 s (10 fps): a noisy slope
    assert not measured(trk)
    trk.dist_history.append((1.2, 2.0, 0.0))
    assert measured(trk)
    trk.depth_conf = 0.25                                       # box-size distance: TTC is assumed, not measured
    assert not measured(trk)


def test_unconfirmed_track_is_held_and_a_flicker_never_spoken():
    # (off to the right, out of the walking path, so it waits for confirmation; in the path it would not, see below)
    inner = NearestPolicy(recent_block_s=2.8)
    gate = AnnouncementGate(inner, GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.95), dist=4.0, hits=1)] if t == 0.0 else [], 3.0)
    assert cues == [] and 1 in gate._pending                     # kept for flicker_keep_s: it may come back (Cause 2)
    cues = run(gate, lambda t: [], 4.0, t0=3.5)
    assert cues == [] and gate.stats["suppressed_unconfirmed"] == 1 and gate.reasons["unconfirmed (dropped)"] == 1
    # the scheduler's budget is what it would be alone, which spent the cue (a refund would change what it says next)
    assert inner.budget == pytest.approx(inner.cfg.budget_capacity_s - 1.2)
    # a real object is said one step later even though the scheduler (recent block) does not offer it again
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.95), dist=4.0, **({"hits": 1} if t == 0.0 else {}))], 3.0)
    assert [c.t for c in cues] == [0.5] and gate.reasons["new (deferred)"] == 1


def test_operator_filter_ignores_the_user_but_not_a_visitor():
    def scene(t):
        j = 0.04 * math.sin(t)                                  # the user fidgets: more than static_disp
        user = dict(tid=1 if t < 20 else 2, bbox=(0.28 + j, 0.15, 0.72 + j, 0.95), dist=0.8)   # id switch at 20 s
        visitor = [dict(tid=10, bbox=(0.78, 0.35, 0.92, 0.75), dist=3.0)] if t >= 10 else []
        return [user] + visitor
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8), GateConfig(background=False, operator_filter=True))
    cues = run(gate, scene, 40.0)
    assert [c.track_id for c in cues] == [10] and gate.stats["suppressed_background"] > 10
    assert "user found" in gate.describe()
    plain = AnnouncementGate(NearestPolicy(recent_block_s=2.8), GateConfig(background=False))
    assert 1 in {c.track_id for c in run(plain, scene, 40.0)}


def test_background_is_quiet_until_it_approaches():
    def scene(t):
        chair = dict(tid=1, label="chair", bbox=(0.7, 0.5, 0.85, 0.8), dist=4.0 if t < 20 else 2.5)
        person = [dict(tid=2, bbox=box(0.3, h=0.5), dist=3.0)] if t >= 6 else []
        return [chair] + person
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8))
    cues = run(gate, scene, 30.0)
    said = [(c.label, c.t) for c in cues]
    assert [lab for lab, _ in said] == ["person", "chair"] and said[1][1] >= 20.0
    assert gate.reasons["background"] + gate.reasons["warmup"] > 3
    gate.rebaseline()                                           # a new warm-up starts at the next decision
    assert not gate._bg and gate._t0 is None


def test_sector_change_reannounces_with_hysteresis():
    def scene(t):
        if t < 10:
            cx = 0.5                                            # ahead
        elif t < 20:
            cx = 0.85                                           # 26 deg: right
        else:
            cx = 0.74 if int(t * 2) % 2 else 0.78               # 18.6 / 21.4 deg: jitter on the boundary
        return [dict(tid=1, bbox=box(cx), dist=2.0)]
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, scene, 30.0)
    assert len(cues) == 2 and 10.0 <= cues[1].t <= 10.5 and gate.reasons["sector"] == 1


def test_haptic_rate_limit():
    gate = AnnouncementGate(NearestPolicy(action="haptic"), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=2.5, ttc=2.5, path=0.9)], 10.0)
    ts = [c.t for c in cues]
    assert 3 <= len(ts) <= 4 and all(b - a >= gate.gate_cfg.haptic_gap_s for a, b in zip(ts, ts[1:]))
    assert gate.stats["suppressed_haptic"] > 10
    assert not gate._mem                                        # a buzz does not count as having said the object


def test_reset_clears_memory_and_resets_inner():
    inner = NearestPolicy()
    gate = AnnouncementGate(inner, GateConfig(background=False))
    spec = lambda t: [dict(tid=1, bbox=AHEAD, dist=2.0)]
    assert len(run(gate, spec, 5.0)) == 1
    gate.reset()
    assert inner.resets == 1 and gate.stats["passed"] == 0 and not gate._mem
    assert len(run(gate, spec, 5.0)) == 1                      # said again after the reset


def test_text_cues_keyed_by_normalised_text():
    def scene(t):
        if t < 5:
            return [dict(tid=1, kind="text", label="text", text="ROOM 204", bbox=box(0.5, h=0.1), dist=3.0)]
        if t < 10:                                              # id switch and an OCR variant of the same sign
            return [dict(tid=2, kind="text", label="text", text="Room 2O4!", bbox=box(0.5, h=0.1), dist=3.0)]
        return [dict(tid=3, kind="text", label="text", text="EXIT", bbox=box(0.5, h=0.1), dist=3.0)]
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, scene, 15.0)
    assert [c.text for c in cues] == ["ROOM 204", "EXIT"]


def test_goal_request_is_said_at_once_and_reminded_sooner():
    inner = NearestPolicy()
    gate = AnnouncementGate(inner, GateConfig(background=False))
    spec = lambda t: [dict(tid=1, label="chair", bbox=AHEAD, dist=3.0)]
    assert len(run(gate, spec, 4.0)) == 1
    gate.set_goal(["chair"])
    assert inner.goal == ["chair"] and not gate._mem            # forwarded; the chair's memory is dropped
    cues = run(gate, spec, 15.0, t0=5.0)                       # (a new scene: the chair's track is new again)
    assert [c.t for c in cues] == [5.0, 13.0]                   # FIND goal: at once (no confirmation step), reminded
                                                                # after hold_goal_s, not 30 s


def test_interface_delegation_and_mode_params():
    inner = NearestPolicy()
    inner.fallback_from = "pcacs_learned:rl"
    gate = AnnouncementGate(inner)
    assert gate.name == "nearest" and gate.decision_interval == DT and gate.supports_actions
    assert gate.cfg is inner.cfg and gate.fallback_from == "pcacs_learned:rl" and gate.last_observations() == ["obs"]
    gate.cfg.verbosity = "quiet"
    assert inner.cfg.verbosity == "quiet"
    gate.goal_tags = ["door"]                                   # an attribute the inner policy has is written through
    assert inner.goal_tags == ["door"]
    door = Cue(t=0.0, track_id=7, label="door", kind="object", message="door", score=1.0, policy="nearest")
    assert gate._is_goal(door, None)                            # ... and the gate's goal test sees it too
    gate.goal_tags = []
    assert not gate._is_goal(door, None)
    gate.decision_interval = 0.25
    assert inner.decision_interval == 0.25
    old = gate.set_mode_params(hold_static_s=5.0, background=False)
    assert old == {"hold_static_s": 30.0, "background": True} and gate.gate_cfg.hold_static_s == 5.0
    with pytest.raises(ValueError):
        gate.set_mode_params(not_a_field=1)
    assert "\n" not in gate.describe() and "passed" in gate.describe()
    assert AnnouncementGate(inner, name="gated").name == "gated"
    sel = AnnouncementGate(SelectOnly(), GateConfig(background=False))
    assert not sel.supports_actions
    sel.goal_tags = ["Door"]                                    # a policy without goal_tags: the gate keeps its own copy
    assert sel._is_goal(door, None)
    scene = Scene()
    said = [c for t in (0.0, 0.5, 1.0, 1.5) for c in sel.select(scene.at(t, [dict(tid=1, bbox=AHEAD)]), t)]
    assert len(said) == 1


def test_real_pcacs_keeps_the_egoblind_car_event():
    from tests.test_egoblind import synthetic_scenario
    sc, _ = synthetic_scenario()
    cfg = PipelineConfig()
    bare = run_on_egoblind(PCACSPolicy(cfg.pcacs), sc, cfg)
    gate = AnnouncementGate(PCACSPolicy(cfg.pcacs))
    gated = run_on_egoblind(gate, sc, cfg)
    car = next(e for e in sc.speak_events() if e.label == "car")
    first = lambda log: min((c.t for c in log.spoken if c.source_id == car.obj_id), default=None)
    assert first(bare) is not None and car.in_window(first(bare))           # the bare policy says it in time ...
    assert first(gated) == first(bare)                                      # ... and the gate does not delay it
    mb, mg = evaluate_run(bare, sc), evaluate_run(gated, sc)
    assert mg.important_cue_recall == mb.important_cue_recall == 1.0 and mg.n_spoken <= mb.n_spoken
    assert egoblind_extras(gated, sc)["coverable_recall"] == 1.0
    assert gated.policy == "pcacs" and not any(c.label == "chair" for c in gated.cues)


# ---- review fixes: numbered signs, the user leaning in, rate limits, id switches, depth off, startup ----------------
def test_numbered_signs_are_different_signs():
    assert same_sign("room 204", "room 2o4") and same_sign("room 204", "r0om 204") and same_sign("exit", "ex1t")
    for a, b in [("room 204", "room 206"), ("platform 1", "platform 2"), ("gate 12", "gate 13"), ("exit", "exit 2")]:
        assert not same_sign(a, b), (a, b)                      # SequenceMatcher alone merged all of these (>= 0.8)
    texts = ["ROOM 204", "ROOM 206", "Platform 1", "Platform 2", "Gate 12", "Gate 13", "EXIT", "EXIT 2"]
    def scene(t):                                               # one sign after the other, each a new track id
        i = int(t // 3)
        return [dict(tid=10 + i, kind="text", label="text", text=texts[i], bbox=box(0.5, h=0.1), dist=3.0)] if i < len(texts) else []
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    assert [c.text for c in run(gate, scene, 23.5)] == texts
    gate.set_goal(["206"])                                      # FIND 206 drops room 206's memory, not room 204's
    assert ("text", "room 204") in gate._mem and ("text", "room 206") not in gate._mem


def desk_session(policy, gcfg, z0=0.8, noise=0.05, fidget=0.01, walk_at=None, walk_mps=0.0, T=60.0, seed=0, bare=False,
                 depth=True, lean_m=0.25, depth_off=None):
    """The user at a desk webcam through the real IoUTracker + TrackDynamics (10 fps, decisions every 0.5 s): a
    fidget of `fidget` (normalised box units), depth jitter sigma `noise`, and a lean of `lean_m` toward the screen
    over 1 s every 20 s (held 2 s); from `walk_at` the user walks toward the camera at `walk_mps`. Returns the cues
    the gate let through and how many decisions saw the user's track as critical by measured TTC. `bare`: the
    scheduler's own cues, without the gate (the gate object is still returned, unused); `depth` False: no depth model;
    depth_off=(a, b): no depth model from a to b (run_live's 'd' key pressed twice)."""
    import random
    from seeforme.geometry import CameraModel, TrackDynamics
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.types import Detection
    rng, cfg = random.Random(seed), PipelineConfig()
    tracker = IoUTracker(cfg.tracker, dynamics=TrackDynamics(CameraModel()))
    gate = AnnouncementGate(policy, gcfg)
    gate.reset()
    cues, n_crit, next_t = [], 0, 0.0
    for i in range(int(T * 10)):
        t = i / 10
        ph = t % 20.0
        lean = lean_m * (min(1.0, ph - 5.0) if 5.0 <= ph < 8.0 else max(0.0, 9.0 - ph) if 8.0 <= ph < 9.0 else 0.0)
        z = z0 - lean + rng.gauss(0.0, noise)
        if walk_at is not None and t >= walk_at:
            z = max(0.3, z0 - walk_mps * (t - walk_at)) + rng.gauss(0.0, noise)
        j, grow = fidget * math.sin(0.7 * t), 0.16 * max(0.0, z0 - z)      # the box grows as the user comes closer
        bbox = (0.28 + j - grow, max(0.0, 0.12 - grow), 0.72 + j + grow, 0.98)
        on = depth and not (depth_off is not None and depth_off[0] <= t < depth_off[1])
        tracks = tracker.update([Detection("person", 0.9, bbox, distance_m=z if on else None,
                                           depth_conf=0.85 if on else 0.0)], t)
        if t + 1e-9 < next_t:
            continue
        next_t = t + 0.5
        n_crit += sum(1 for trk in tracks if gate._critical(trk))
        cues += (policy if bare else gate).decide(tracks, t, ChannelState())
    return cues, n_crit, gate


@pytest.mark.parametrize("z0,noise", [(0.8, 0.03), (0.8, 0.05), (1.0, 0.05)])
def test_user_leaning_in_at_a_desk_is_not_announced(z0, noise):
    for make in (lambda: NearestPolicy(), lambda: PCACSPolicy(PipelineConfig().pcacs)):
        cues, n_crit, gate = desk_session(make(), GateConfig(operator_filter=True), z0=z0, noise=noise)
        assert n_crit > 0                                       # the lean does give a measured TTC under 2 s ...
        assert cues == [], gate.last_reason                     # ... but the user's own body is never announced
        assert gate.reasons["critical"] == 0
    # without the operator filter a static user is background after the warm-up: said at most once (in the path)
    cues, _, gate = desk_session(NearestPolicy(), GateConfig(), z0=z0, noise=noise)
    assert len(cues) <= 1


@pytest.mark.parametrize("walk_mps", [0.5, 0.7])
def test_person_two_metres_from_a_desk_webcam_is_released_as_soon_as_they_approach(walk_mps):
    # 2 m from the webcam is not a laptop user (0.4-1.0 m away): not the operator but background (said once at the
    # start: in the walking path), held through two leans, then released and said within ~1 s of walking up to the
    # camera (the old gate held this "operator" until 0.5 m), with an always-offering scheduler and with one that
    # offers a track only every 2.8 s (the gate says the cue it held itself: P-CACS will not offer it again in time)
    for make in (lambda: NearestPolicy(), lambda: NearestPolicy(recent_block_s=2.8)):
        cues, _, gate = desk_session(make(), GateConfig(operator_filter=True), z0=2.0, noise=0.03,
                                     walk_at=30.0, walk_mps=walk_mps, T=40.0)
        before, walk = [c for c in cues if c.t < 30.0], [c for c in cues if c.t >= 30.0]
        assert gate._operator is None and len(before) <= 1
        assert walk and walk[0].t <= 31.5 and 2.0 - walk_mps * (walk[0].t - 30.0) >= 1.2   # contact at 32.0-32.8 s
        assert any(r.startswith("approach") for r in gate.reasons) and any("background" in r[2] for r in gate.releases)


def test_operator_released_by_a_real_approach_is_held_again_once_at_rest():
    # the user at 1.0 m (the operator) comes right up to the screen (0.5 m in 0.5 s) and stays there: said once (the
    # approach), then held again after recapture_s at rest, so one release is never a "Person." loop
    def session(policy):
        import random
        from seeforme.geometry import CameraModel, TrackDynamics
        from seeforme.tracking.tracker import IoUTracker
        from seeforme.types import Detection
        rng, tracker = random.Random(0), IoUTracker(PipelineConfig().tracker, dynamics=TrackDynamics(CameraModel()))
        gate = AnnouncementGate(policy, GateConfig(operator_filter=True))
        gate.reset()
        cues = []
        for i in range(600):
            t = i / 10
            z = (1.0 if t < 30.0 else max(0.5, 1.0 - (t - 30.0))) + rng.gauss(0.0, 0.03)
            grow = 0.16 * max(0.0, 1.0 - z)
            tracks = tracker.update([Detection("person", 0.9, (0.28 - grow, max(0.0, 0.12 - grow), 0.72 + grow, 0.98),
                                               distance_m=z, depth_conf=0.85)], t)
            if i % 5 == 0:
                cues += gate.decide(tracks, t, ChannelState())
        return cues, gate
    cues, gate = session(NearestPolicy(recent_block_s=2.8))
    assert [c.t for c in cues] == [30.5] and gate.reasons["approach"] == 1
    (t_rel, _, why), (t_back, _, back) = gate.releases
    assert t_rel == 30.5 and why.startswith("operator") and "nearer than its 1.0 m reference" in why
    assert back.startswith("operator re-captured") and 33.5 <= t_back <= 36.0 and not gate._operator.released


def test_interrupts_need_a_gap_and_a_moving_or_approaching_object():
    gate = AnnouncementGate(NearestPolicy(action="interrupt"), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=2.0)], 10.0)
    assert len(cues) == 1 and gate.reasons["interrupt"] == 0     # a static person: 1 of 21 interrupts (was 20)
    gate = AnnouncementGate(NearestPolicy(action="interrupt"), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=2.0, lat=0.8)], 10.0)
    ts = [c.t for c in cues]
    assert ts == [0.0, 2.5, 5.0, 7.5, 10.0] and gate.reasons["interrupt"] == 4   # every min_gap_s (it moves)


def test_second_object_seen_together_is_not_an_id_switch():
    # two people ahead at once; the scheduler only ever offers the nearer one until it leaves at 8 s. The farther one
    # was in view together with it, so it is a second person, not the first one's new track id
    def scene(t):
        return ([dict(tid=1, bbox=box(0.40, h=0.5), dist=2.0)] if t < 8 else []) + [dict(tid=2, bbox=box(0.62, h=0.3), dist=4.0)]
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, scene, 12.0)
    assert [c.track_id for c in cues] == [1, 2] and cues[1].t == 8.0
    # a real id switch (the new id appears as the old one vanishes) still inherits the memory, and so does an older
    # track the tracker re-acquires later (first seen long before, but never in view together with the newer id)
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    assert len(run(gate, lambda t: [dict(tid=1 if t < 8 else 2, bbox=box(0.40, h=0.5), dist=2.0)], 12.0)) == 1
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    tid = lambda t: 1 if t < 4 or t >= 8 else 2                  # 1 -> 2 -> 1 again (re-acquired, first_seen 0)
    assert len(run(gate, lambda t: [dict(tid=tid(t), bbox=box(0.40, h=0.5), dist=2.0)], 12.0)) == 1
    # two overlapping boxes of one object (a duplicate detection) are not "apart": the survivor inherits
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, lambda t: ([dict(tid=1, bbox=box(0.40, h=0.5), dist=2.0)] if t < 8 else [])
               + ([dict(tid=2, bbox=box(0.42, h=0.5), dist=2.1)] if t >= 4 else []), 12.0)
    assert [c.track_id for c in cues] == [1]


def test_without_depth_an_object_in_the_path_closing_in_is_reannounced():
    # depth off: TTC is assumed, nothing is critical. The chair said at 1.8 m is not repeated while the user stands
    # still (no "Person." loop at a desk), but once the box shows the user walking toward it, it is said again after
    # min_gap_s, before contact
    def scene(t):
        z = 1.8 if t < 10 else 1.8 - 0.4 * (t - 10)
        return [dict(tid=1, label="chair", bbox=AHEAD, dist=z, rel=0.0 if t < 10 else -0.4, ttc=z - 0.6, path=0.9,
                     depth_conf=0.25)]
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, scene, 13.0)
    assert [c.t for c in cues] == [0.0, 10.0, 12.5] and gate.reasons["near"] >= 1
    assert gate.reasons["critical"] == 0 and all(c.distance_m > 0.6 for c in cues)


def test_in_path_object_present_at_startup_is_said_once_with_background_on():
    def scene(t):
        return [dict(tid=1, label="chair", bbox=AHEAD, dist=2.0),                     # in the path, 2 m
                dict(tid=2, label="chair", bbox=(0.80, 0.50, 0.95, 0.80), dist=4.0)]  # off to the right, 4 m
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8))
    cues = run(gate, scene, 40.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 0.0)]      # at once (not after the warm-up), and never repeated
    assert gate.reasons["background"] > 5                       # the far chair, and the near one once it was said


def test_second_person_overlapping_the_user_is_not_taken_for_the_user():
    def scene(t):
        user = dict(tid=1, bbox=(0.28, 0.15, 0.72, 0.95), dist=0.8)
        visitor = [dict(tid=5, bbox=(0.40, 0.10, 0.80, 0.90), dist=1.5)] if t >= 10 else []   # IoU 0.56, centred
        return [user] + visitor
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8), GateConfig(background=False, operator_filter=True))
    cues = run(gate, scene, 20.0)
    assert [c.track_id for c in cues] == [5]
    assert gate._operator.track_ids == {1} and gate._operator.bbox == (0.28, 0.15, 0.72, 0.95)


def test_prune_keeps_the_entries_said_most_recently():
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    ident = ("object", "person")
    # created in this order; the first one was said again most recently (an id switch adopted it)
    gate._mem[ident] = [_Said(ident, "ahead", 0.0, t, 2.0, i, None) for i, t in enumerate([9.0, 0.0, 1, 2, 3, 4, 5, 6, 7, 8])]
    gate._prune(10.0)
    assert sorted(e.t for e in gate._mem[ident]) == [2, 3, 4, 5, 6, 7, 8, 9.0]


# ---- safety: the gated pipeline warns no later than the bare scheduler (+ one decision step), with the REAL P-CACS ----
def real_pcacs():
    """The live P-CACS exactly as scripts/run_live.py builds it (configs/default.yaml, the text-importance model)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from _common import load_cfg
    from seeforme.pipeline import make_live_policy
    return make_live_policy("pcacs", load_cfg())


SIZE = {"person": (1.7, 0.5), "chair": (0.9, 0.5), "car": (1.5, 1.8)}


def tracked_session(objs, wrap="bare", gcfg=None, setup="auto", mode="auto", depth=True, T=30.0, events=(), seed=0):
    """Objects (label, z(t), x(t), on(t)) projected through the camera model into the real IoUTracker + TrackDynamics
    at 10 fps (depth noise 3 cm), a decision every 0.5 s with real P-CACS. wrap: 'bare' (P-CACS alone), 'gate'
    (AnnouncementGate(gcfg)) or 'ctl' (LiveController(setup, mode)); events: (t, f(policy, gate, controller)).
    Returns [(t, label, true range)] of the spoken cues and the gate."""
    import random
    from seeforme.geometry import CameraModel, TrackDynamics
    from seeforme.live.controller import LiveController
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.types import Detection
    rng, cam = random.Random(seed), CameraModel()
    tracker = IoUTracker(PipelineConfig().tracker, dynamics=TrackDynamics(cam))
    pol, gate, ctl = real_pcacs(), None, None
    if wrap == "gate":
        dec = gate = AnnouncementGate(pol, gcfg or GateConfig(), camera=cam)
    elif wrap == "ctl":
        ctl = LiveController(pol, mode=mode, setup=setup, camera=cam, gate_cfg=gcfg)
        dec, gate = ctl.policy, ctl.gate
    else:
        dec = pol
    dec.reset()
    out, owner, next_t = [], {}, 0.0
    for i in range(int(T * 10)):
        t = i / 10
        for et, f in events:
            if abs(et - t) < 1e-6:
                f(pol, gate, ctl)
        dets, boxes = [], []
        for label, z, x, on in objs:
            if not on(t):
                continue
            H, W = SIZE[label]
            b = cam.clip(cam.project_box(x(t), z(t), H, W))
            if b is None:
                continue
            dets.append(Detection(label, 0.9, b, distance_m=max(0.2, z(t) + rng.gauss(0.0, 0.03)) if depth else None,
                                  depth_conf=0.85 if depth else 0.0))
            boxes.append((label, z(t), b))
        tracks = tracker.update(dets, t)
        for trk in tracks:
            best = max(boxes, key=lambda lb: _mod.bbox_iou(lb[2], trk.bbox), default=None)
            if best is not None and _mod.bbox_iou(best[2], trk.bbox) > 0.3:
                owner[trk.track_id] = best
        if ctl is not None:
            ctl.on_frame(None, t, tracks)
        if t + 1e-9 < next_t:
            continue
        next_t = t + 0.5
        for c in dec.decide(tracks, t, ChannelState()):
            label, _, _ = owner.get(c.track_id, (c.label, None, None))
            z = next((o[1](t) for o in objs if o[0] == label), None)
            out.append((t, label, z))
    return out, gate


def first_in(cues, lo, hi):
    return next((c for c in cues if lo - 1e-9 <= c[0] < hi - 1e-9), None)


GATED = [("gate", GateConfig(), {}), ("gate", GateConfig(operator_filter=True), {}),
         ("ctl", None, dict(setup="auto")), ("ctl", None, dict(setup="desk")), ("ctl", None, dict(setup="auto", mode="scene"))]


@pytest.mark.parametrize("walk_mps", [0.5, 0.7])
def test_real_pcacs_user_walking_up_to_the_webcam_is_announced_before_1m(walk_mps):
    # the desk scene of the "Person." report, but the user (2 m away: not a laptop user) walks up to the camera at 30 s;
    # P-CACS offers its critical cue once (then blocks the track for 2.8 s): the old gate held it as the operator and
    # never said it. Now said no later than the bare scheduler says it (+ one step); at 0.5 m/s while still at least
    # 1.0 m away (true range; at 0.7 m/s P-CACS itself first speaks at 0.95 m)
    bare, _, _ = desk_session(real_pcacs(), GateConfig(operator_filter=True), z0=2.0, noise=0.03, walk_at=30.0,
                              walk_mps=walk_mps, T=36.0, bare=True)
    tb = first_in([(c.t,) for c in bare], 30.0, 36.0)
    assert tb is not None
    for gcfg in (GateConfig(operator_filter=True), GateConfig()):
        cues, _, gate = desk_session(real_pcacs(), gcfg, z0=2.0, noise=0.03, walk_at=30.0, walk_mps=walk_mps, T=36.0)
        walk = [c for c in cues if c.t >= 30.0]
        assert walk and walk[0].t <= tb[0] + DT, gate.last_reason
        if walk_mps == 0.5:
            assert 2.0 - walk_mps * (walk[0].t - 30.0) >= 1.0 - 1e-9 and walk[0].distance_m > 1.0


def test_real_pcacs_person_standing_ahead_at_startup_then_approaching_is_announced():
    # a person stands centred 2 m ahead through the warm-up (large and centred: the old gate took them for the desk
    # user), then walks at the user at 0.5 m/s from 20 s (contact at 22.8 s)
    person = [("person", lambda t: 2.0 if t < 20.0 else max(0.3, 2.0 - 0.5 * (t - 20.0)), lambda t: 0.0, lambda t: True)]
    bare, _ = tracked_session(person, T=23.0)
    tb = first_in(bare, 20.0, 22.8)
    assert tb is not None
    for wrap, gcfg, kw in GATED:
        cues, gate = tracked_session(person, wrap, gcfg, T=23.0, **kw)
        tg = first_in(cues, 20.0, 22.8)
        assert tg is not None and tg[0] <= tb[0] + DT, (wrap, kw, cues, gate.releases, dict(gate.reasons))


def test_real_pcacs_car_that_is_the_first_object_seen_is_announced_before_contact():
    # a car comes into view 12 m away at 5 s, head-on at 5 m/s (contact 7.3 s), as the first thing the camera sees:
    # one box sample is not "static", and a vehicle is never warm-up or background
    car = [("car", lambda t: max(0.3, 12.0 - 5.0 * max(0.0, t - 5.0)), lambda t: 0.0, lambda t: t >= 5.0)]
    tc = 5.0 + 11.4 / 5.0
    bare, _ = tracked_session(car, T=7.5)
    tb = first_in(bare, 5.0, tc)
    assert tb is not None
    rebase = [(4.9, lambda pol, gate, ctl: gate.rebaseline())]       # also right after a rebaseline (scene / desk)
    for wrap, gcfg, kw in GATED:
        for ev in ((), rebase):
            cues, gate = tracked_session(car, wrap, gcfg, T=7.5, events=ev, **kw)
            tg = first_in(cues, 5.0, tc)
            assert tg is not None and tg[0] <= tb[0] + DT, (wrap, kw, cues, dict(gate.reasons))


def test_real_pcacs_find_goal_static_since_startup_is_announced_when_the_goal_is_set():
    # a chair 4 m away on the right, in view and static since startup (background), becomes the FIND goal at 10 s
    chair = [("chair", lambda t: 4.0, lambda t: 1.8, lambda t: True)]
    goal = [(10.0, lambda pol, gate, ctl: (ctl or gate or pol).set_goal(["chair"]))]
    bare, _ = tracked_session(chair, T=20.0, events=goal)
    tb = first_in(bare, 10.0, 20.0)
    assert tb is not None and tb[0] <= 10.5
    for wrap, gcfg, kw in GATED:
        cues, gate = tracked_session(chair, wrap, gcfg, T=20.0, events=goal, **kw)
        tg = first_in(cues, 10.0, 20.0)
        assert tg is not None and tg[0] <= tb[0] + DT, (wrap, kw, cues, dict(gate.reasons))
    # and a sign said before is said again at once when FIND names part of it ("room 206" -> "ROOM 206 LAB")
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    sign = lambda t: [dict(tid=1 if t < 3 else 2, kind="text", label="text", text="ROOM 206 LAB", bbox=box(0.5, h=0.1), dist=3.0)]
    assert len(run(gate, sign, 2.0)) == 1
    gate.set_goal(["Room 206"])
    # (at once: the new track id takes over the old one's matches, so it needs no confirmation step)
    assert ("text", "room 206 lab") not in gate._mem and [c.t for c in run(gate, sign, 6.0, t0=3.0)][:1] == [3.0]


@pytest.mark.parametrize("z0", [1.5, 1.8])
def test_real_pcacs_depth_off_chair_is_reannounced_before_contact(z0):
    # depth off: the chair in the path is said at the start; from 10 s the user walks into it at 0.5 m/s. Its box is
    # soon cut off at the bottom of the frame, so its box-height range *grows*; the width shows it closing. P-CACS
    # never offers it again (bare: silent until contact), the gate says it again itself before contact
    chair = [("chair", lambda t: z0 if t < 10.0 else max(0.3, z0 - 0.5 * (t - 10.0)), lambda t: 0.0, lambda t: True)]
    tc = 10.0 + (z0 - 0.6) / 0.5
    for wrap, gcfg, kw in [("gate", GateConfig(background=False), {})] + GATED:
        cues, gate = tracked_session(chair, wrap, gcfg, depth=False, T=tc, **kw)
        again = first_in(cues, 10.0, tc)
        assert cues and cues[0][0] < 10.0 and again is not None and again[0] <= tc - 0.5, (wrap, kw, cues, dict(gate.reasons))


# ---- third review: what the rescue may say, flickers, id switches, repeated leans ----------------------------------
class OfferAt(NearestPolicy):
    """A sparse scheduler (the learned one offers a static object every few seconds, then not while it creeps in): a
    cue for the nearest track only at the listed decision times, marked as announced like P-CACS."""

    def __init__(self, times, **kw):
        super().__init__(**kw)
        self.times = {round(x, 3) for x in times}

    def decide(self, tracks, t, ch=None):
        return super().decide(tracks, t, ch) if round(t, 3) in self.times else []


def test_held_object_that_comes_closer_is_said_by_the_gate():
    # Cause 1 (car repro): a car said at 5.6 m creeps toward the user from 12.4 s at 0.37 m/s (contact 26 s); the
    # scheduler offered it once more while it was parked (held: a repeat) and not again. The gate says that cue itself
    # once the car is materially closer (the old rescue dropped `closer`, and kept a held cue for 3 s only): in the
    # walking path within path_m, at the `closing` step (0.84 m at 5.6 m) while it closes
    z = lambda t: 5.6 if t < 12.4 else 5.6 - 0.37 * (t - 12.4)
    def scene(t):
        go = t >= 12.4
        return [dict(tid=1, label="car", bbox=box(0.46, h=min(0.95, 1.4 / z(t)), w=min(0.95, 1.7 / z(t))), dist=z(t),
                     rel=-0.37 if go else 0.0, ttc=(z(t) - 0.6) / 0.37 if go else math.inf, path=0.6)]
    for gcfg in (GateConfig(), GateConfig(background=False)):
        gate = AnnouncementGate(OfferAt([2.0, 10.0]), gcfg)
        ts = [c.t for c in run(gate, scene, 25.5)]
        assert ts[0] == 2.0 and not [x for x in ts if 2.0 < x < 12.4]          # parked: no repeats
        assert [x for x in ts if x > 12.4][:1] == [15.0], ts                    # 0.96 m closer, 11 s before contact
        assert gate.stats["rescued"] >= 1 and gate.reasons["closing (deferred)"] == 1


def test_rescue_finds_the_safety_reason_behind_a_sector_change():
    # Cause 1: the rules are checked in order and the rescue only saw the first one that held, so a person said on the
    # right, offered again (held: a repeat), who then steps in front of the user and closes in (a sector change first,
    # then `closing`) was never said again; the rescue now asks for the first reason it may say a cue for
    def scene(t):
        z = 2.8 if t < 5.0 else max(0.8, 2.8 - 0.4 * (t - 5.0))
        go = t >= 5.0
        return [dict(tid=1, bbox=box(0.62 if go else 0.85, h=min(0.95, 1.6 / z)), dist=z, rel=-0.4 if go else 0.0,
                     ttc=(z - 0.6) / 0.4 if go else math.inf, path=0.6 if go else 0.3)]
    gate = AnnouncementGate(OfferAt([0.5, 3.0]), GateConfig(background=False))
    cues = run(gate, scene, 8.0)
    assert cues[0].t == 0.5 and [c.t for c in cues[1:]][:1] == [6.5]       # 0.6 m nearer, within 3 m, closing
    assert gate.reasons["closing (deferred)"] == 1 and gate.reasons["sector"] == 0


@pytest.mark.parametrize("seen_before", [False, True])
def test_cue_held_for_confirmation_survives_a_flicker(seen_before):
    # Cause 2 (seed 1996): lighting dropouts; the chair (off the walking path: in it, it would not wait at all) is in
    # view at single decisions only, under a new track id each time (2 matches: not confirmed). The scheduler offers it once (then blocks it); the gate used to drop the held cue
    # as soon as the track missed a frame. Now the cue waits for the chair's return (a successor track of the same
    # identity, where it vanished) and is said then, refreshed: the successor's matches count towards confirmation.
    # Seen once before at the same place, it is confirmed at once
    visits = [7.5, 9.0, 10.5, 12.0] if seen_before else [9.0, 10.5, 12.0]
    z = lambda t: 4.2 - 0.8 * max(0.0, t - 7.5)
    def scene(t):
        return [dict(tid=10 + visits.index(t), label="chair", bbox=box(0.85, h=min(0.95, 0.9 / z(t))), dist=z(t))] if t in visits else []
    inner = OfferAt([9.0], recent_block_s=2.8)
    gate = AnnouncementGate(inner, GateConfig(background=False))
    cues = run(gate, scene, 12.5)
    assert [c.t for c in cues][:1] == ([9.0] if seen_before else [10.5])
    assert cues[0].track_id == 10 + visits.index(cues[0].t) and cues[0].distance_m == pytest.approx(z(cues[0].t))
    assert gate.reasons["unconfirmed (dropped)"] == 0


def test_object_close_in_the_path_does_not_wait_for_confirmation():
    # Cause 2 (seed 5246): in lighting dropouts a bicycle riding at the user is in view at one decision only, 2.5 m away
    # in the path, under a new track id (2 matches); the scheduler says it then (1.2 s before contact). Something in
    # the path cannot wait a decision step for confirmation (a new track's TTC is not measured: not critical); fourth
    # review (seeds 22385, 21845): neither can a person walking at the user from 3 m, seen in 2-frame sightings whose
    # follow-ups fall between decisions (the next one came 1.5 s later), nor a chair. Off the path it still waits
    def run_one(label, dist, cx):
        gate = AnnouncementGate(OfferAt([25.5]), GateConfig(background=False))
        return run(gate, lambda t: [dict(tid=7, label=label, bbox=box(cx, h=0.5), dist=dist, path=0.88 if cx == 0.5 else 0.1)]
                   if t == 25.5 else [], 26.5, t0=25.0)
    for label in ("bicycle", "person", "chair"):
        assert [c.t for c in run_one(label, 2.5, 0.5)] == [25.5] and [c.t for c in run_one(label, 6.0, 0.5)] == [25.5]
        assert run_one(label, 2.5, 0.95) == [] and run_one(label, 6.0, 0.95) == [], label


@pytest.mark.parametrize("walks", [True, False])
def test_background_person_lost_from_view_is_not_background_when_reacquired(walks):
    # Cause 3 (seed 1815): a person static through the warm-up (background) is lost in a lighting dropout and comes back
    # under a new track id, walking at the user; the scheduler offers it once, right then. It may have moved while out
    # of view, so it is not held as background; one that stays where it is is held again after recapture_s at rest
    z = lambda t: 4.1 if t < 12.8 or not walks else 4.1 - 1.26 * (t - 12.8)
    def scene(t):
        if 11.5 < t < 13.0:
            return []                                               # dropout: two decisions without it
        go = walks and t >= 12.8
        return [dict(tid=1 if t <= 11.5 else 2, bbox=box(0.45, h=min(0.95, 1.6 / z(t))), dist=z(t), rel=-1.26 if go else 0.0,
                     ttc=(z(t) - 0.6) / 1.26 if go else math.inf, path=0.6)]
    gate = AnnouncementGate(OfferAt([13.0, 20.0]))
    cues = run(gate, scene, 15.0 if walks else 21.0)
    assert [c.t for c in cues][:1] == [13.0] and any("lost from view" in r[2] for r in gate.releases)
    if not walks:                                                   # at rest again: background, the 20 s offer is held
        assert any("re-captured" in r[2] for r in gate.releases) and gate.last_verdict["reason"] == "background"


@pytest.mark.parametrize("dropout", [False, True])
def test_held_cue_follows_the_object_across_an_id_switch(dropout):
    # Cause 3 (seed 2299): a background person's cue is held at 20.5 s; then the person comes nearer under a new track id
    # (an id switch, or a dropout of a second) and is released: as approaching, or as lost from view for longer than
    # lost_s. The held cue (keyed by track id, it was lost) now follows the object, so the gate says it although the
    # scheduler does not offer it again
    z = lambda t: 3.0 if t < 20.5 else 3.0 - 0.5 * (t - 20.5)
    def scene(t):
        if dropout and t == 21.0:
            return []
        go = t >= 20.5
        return [dict(tid=1 if t <= 20.5 else 2, bbox=box(0.55, h=min(0.95, 1.6 / z(t))), dist=z(t), rel=-0.5 if go else 0.0,
                     ttc=(z(t) - 0.6) / 0.5 if go else math.inf, path=0.3)]
    gate = AnnouncementGate(OfferAt([20.5]))
    cues = run(gate, scene, 22.0)
    if dropout:
        # out of view right after its cue was held (a person in the path the user never heard of): the gate cannot
        # watch it come closer, so it says the held cue at once (_say_lost), and the successor is not said again
        assert [(c.t, c.track_id) for c in cues] == [(21.0, 1)] and 1 not in gate._held
        assert gate.reasons["lost (deferred)"] == 1 and any("lost from view" in r[2] for r in gate.releases)
        return
    assert [(c.t, c.track_id) for c in cues] == [(21.5, 2)] and 1 not in gate._held
    assert gate.reasons["approach (deferred)"] == 1 and any("nearer than its" in r[2] for r in gate.releases)


@pytest.mark.parametrize("lean_m", [0.35, 0.6])
def test_repeated_deep_leans_at_a_desk_cost_one_cue(lean_m):
    # Cause 4: the user (operator) leaning 0.35-0.6 m toward the webcam every 20 s was released and re-captured at
    # every lean ("Person, right in front of you." each time; bare P-CACS says it once). The operator remembers the
    # nearest point it came back from: the same lean again says nothing new. Coming right up to the screen (0.3 m,
    # nearer than the 0.65 m leans) still releases it at once (said when the scheduler offers it: bare P-CACS, having
    # said the user at the first lean, does not)
    for make in (lambda: PCACSPolicy(PipelineConfig().pcacs), lambda: NearestPolicy(recent_block_s=2.8)):
        cues, _, gate = desk_session(make(), GateConfig(operator_filter=True), z0=1.0, noise=0.03, lean_m=lean_m, T=100.0)
        assert len(cues) <= 1, ([c.t for c in cues], gate.releases)
    if lean_m > 0.5:
        return                                                      # (0.4 m leans: 0.3 m is not nearer by lean_margin_m)
    cues, _, gate = desk_session(NearestPolicy(recent_block_s=2.8), GateConfig(operator_filter=True), z0=1.0, noise=0.03,
                                 lean_m=lean_m, T=102.0, walk_at=100.0, walk_mps=1.0)
    assert len([c for c in cues if c.t < 100.0]) <= 1 and [c for c in cues if 100.0 <= c.t <= 101.0], gate.releases



# ---- fourth review: lean memory, read mode, same-label newcomers, flickers, repeats after dropouts ----------------
def boxes_session(policy, objs, gcfg=None, T=30.0, depth=True, seed=0):
    """Named objects (dicts: name, label, box(t) or x(t), z(t), on(t), drop(t)) through the real IoUTracker +
    TrackDynamics at 10 fps (depth noise 3 cm), a decision every 0.5 s: the policy alone (gcfg None) or gated.
    Returns [(t, name)] of the spoken cues, the gate, and the track ids each object was seen under."""
    import random
    from seeforme.geometry import CameraModel, TrackDynamics
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.types import Detection
    rng, cam = random.Random(seed), CameraModel()
    tracker = IoUTracker(PipelineConfig().tracker, dynamics=TrackDynamics(cam))
    dec = policy if gcfg is None else AnnouncementGate(policy, gcfg, camera=cam)
    dec.reset()
    out, owner, ids, next_t = [], {}, {}, 0.0
    for i in range(int(T * 10)):
        t = round(i / 10, 1)
        dets, named = [], []
        for o in objs:
            if not o.get("on", lambda t: True)(t) or o.get("drop", lambda t: False)(t):
                continue
            H, W = SIZE[o["label"]]
            b = o["box"](t) if "box" in o else cam.clip(cam.project_box(o["x"](t), o["z"](t), H, W))
            dets.append(Detection(o["label"], 0.9, b, distance_m=max(0.2, o["z"](t) + rng.gauss(0.0, 0.03)) if depth else None,
                                  depth_conf=0.85 if depth else 0.0))
            named.append((o["name"], b))
        tracks = tracker.update(dets, t)
        for trk in tracks:
            best = max(named, key=lambda nb: _mod.bbox_iou(nb[1], trk.bbox), default=None)
            if best is not None and _mod.bbox_iou(best[1], trk.bbox) > 0.3:
                owner[trk.track_id] = best[0]
                ids.setdefault(best[0], set()).add(trk.track_id)
        if t + 1e-9 < next_t:
            continue
        next_t = t + 0.5
        out += [(t, owner.get(c.track_id, "?")) for c in dec.decide(tracks, t, ChannelState()) if c.action != "haptic"]
    return out, (dec if gcfg is not None else None), ids


def _desk_user(z0, lean_m, on):
    """The desk user (box cut off at the bottom, as in desk_session) leaning lean_m toward the webcam once, at 5-10 s."""
    def lean(t):
        return lean_m * (min(1.0, t - 5.0) if 5.0 <= t < 9.0 else max(0.0, 10.0 - t) if 9.0 <= t < 10.0 else 0.0)
    def bbox(t):
        grow = 0.16 * lean(t)
        return (0.28 - grow, max(0.0, 0.12 - grow), 0.72 + grow, 0.98)
    return dict(name="user", label="person", z=lambda t: z0 - lean(t), box=bbox, on=on)


@pytest.mark.parametrize("visitor", [False, True])
def test_lean_memory_is_not_inherited_by_a_stranger_taken_for_the_user(visitor):
    # Critical (fourth review): after the desk user leaned in once, the operator remembered the lean and was released or
    # critical again only nearer than it. A stranger who walks up to the webcam after the user left (a new track over
    # the user's box) or a visitor walking up in front of the user (a second box over the user's) was taken for the
    # user and inherited that memory: never said, to contact. The lean memory now belongs to the track that rested
    # at the user's place, and someone farther than the user's place is not taken for the user at all
    # (leans deep enough to release the operator over its depth noise margin, so the lean memory exists)
    zu, lean_m, t_go = (0.8, 0.45, 66.0) if visitor else (0.7, 0.42, 66.0)
    z0, v = (3.5, 0.5) if visitor else (4.0, 1.0)
    zs = lambda t: max(0.3, z0 - v * max(0.0, t - t_go))
    stranger = dict(name="s", label="person", z=zs, x=lambda t: 0.0, on=lambda t: t >= (t_go - 1.0 if visitor else t_go))
    user = _desk_user(zu, lean_m, (lambda t: zs(t) > zu - 0.1) if visitor else (lambda t: t < 62.0))
    tc = t_go + (z0 - 0.6) / v
    for make in (lambda: PCACSPolicy(PipelineConfig().pcacs), lambda: NearestPolicy(recent_block_s=2.8)):
        bare, _, _ = boxes_session(make(), [user, stranger], T=tc + 0.5)
        tb = next((t for t, n in bare if n == "s" and t_go <= t < tc), None)
        assert tb is not None
        cues, gate, _ = boxes_session(make(), [user, stranger], GateConfig(operator_filter=True), T=tc + 0.5)
        tg = next((t for t, n in cues if n == "s" and t_go <= t < tc), None)
        assert tg is not None and tg <= tb + DT, (cues, gate.releases, dict(gate.reasons))
        assert gate._operator is not None and gate._operator.lean is not None     # (the user did lean in)
        assert len([t for t, n in cues if n == "user" and t < t_go]) <= 1          # ... and that cost one cue at most


@pytest.mark.xfail(strict=True, reason="unfinished gate round: the quick-closing release says the 7.0 s offer inside min_gap_s (issue 34)")
def test_read_mode_holds_a_muted_cue_before_it_is_remembered():
    # Major (fourth review, seed 22651): in read mode the gate committed a chair cue (said while TTC >= 6 s) that the
    # TaskAdapter's read filter then dropped; the scheduler's warning once the chair approached was held as a repeat of
    # a cue nobody heard. With objects_muted the adapter leaves the muting to the gate, which holds the cue *before*
    # remembering it. (gate_verify_r3: read mode never mutes an object coming at the user, by a TTC under approach_ttc_s
    # or a measured closing speed: the chair stands still while muted, then comes at 0.8 m/s from 5.0 s)
    from seeforme.live.activity import Activity, TaskAdapter
    from seeforme.live.controller import LiveController
    z = lambda t: 4.5 if t < 5.0 else 4.5 - 0.8 * (t - 5.0)
    def scene(t):
        go = t >= 5.0
        return [dict(tid=1, label="chair", bbox=box(0.5, h=min(0.95, 0.9 / z(t))), dist=z(t), rel=-0.8 if go else 0.0,
                     ttc=(z(t) - 0.6) / 0.8 if go else math.inf, path=0.3)]
    gate = AnnouncementGate(OfferAt([1.0, 4.0, 7.0]), GateConfig(background=False, objects_muted=True))
    cues = run(gate, scene, 8.0)
    # muted at 1.0 and 4.0 (static: not remembered), said by the gate itself as new once it comes (5.0), then closer
    assert [c.t for c in cues] == [5.0, 7.5], dict(gate.reasons)
    assert gate.reasons["muted"] == 2 and gate.reasons["new (deferred)"] == 1 and gate.stats["suppressed_muted"] == 2
    # a scheduler that offered it only while it stood still (muted) and not again: the gate says the cue it held itself
    # once it is no longer muted (fuzz seed 21570: the scheduler's last offer came 7 s before)
    gate = AnnouncementGate(OfferAt([1.0, 2.5]), GateConfig(background=False, objects_muted=True))
    cues = run(gate, scene, 8.0)
    assert [c.t for c in cues] == [5.0] and gate.reasons["new (deferred)"] == 1
    assert cues[0].distance_m == pytest.approx(z(5.0))
    # the adapter's read mode sets the field, and then does not filter itself
    gate = AnnouncementGate(NearestPolicy(), GateConfig())
    ad = TaskAdapter(gate=gate, policy=gate)
    ad.step(Activity.READING, 1.0)
    assert gate.gate_cfg.objects_muted is True and not ad.should_mute_objects
    ad.step(Activity.WALKING, 2.0)
    assert gate.gate_cfg.objects_muted is False
    # the live controller pinned to read mode: the chair is said once it comes at the user
    ctl = LiveController(OfferAt([1.0, 4.0, 7.0]), mode="read", setup="handheld")
    sc, said = Scene(), []
    for i in range(17):
        t = i * DT
        said += [c.t for c in ctl.policy.decide(sc.at(t, scene(t)), t, ChannelState())]
    assert said == [5.0, 7.5] and ctl.gate.gate_cfg.objects_muted


@pytest.mark.parametrize("depth_conf", [0.9, 0.25])
def test_a_farther_object_stepping_in_where_a_background_object_stood_is_not_background(depth_conf):
    # Major (fourth review, seed 22025): a background person at 4.4 m leaves; 0.6 s later another person appears where
    # they stood, 6.5 m away, and walks at the user. The boxes overlap, so the newcomer was adopted into the background
    # record and its only offer held; the release (against the old person's reference) came after the held cue expired.
    # A newcomer materially farther than the background object's reference is another object
    z = lambda t: 6.5 - 0.8 * max(0.0, t - 10.5)
    def scene(t):
        if t < 10.0:
            return [dict(tid=1, bbox=(0.35, 0.8 - 1.6 / 4.4 * 0.5, 0.35 + 0.5 / 4.4 * 0.5, 0.8), dist=4.4, depth_conf=depth_conf)]
        if t < 10.5:
            return []
        h, w = 1.6 / z(t) * 0.5, 0.5 / z(t) * 0.5
        return [dict(tid=2, bbox=(0.355, 0.8 - h, 0.355 + w, 0.8), dist=z(t), rel=-0.8, ttc=(z(t) - 0.6) / 0.8, path=0.3,
                     depth_conf=depth_conf)]
    gate = AnnouncementGate(OfferTrack({2: [10.5]}))
    cues = run(gate, scene, 13.0)
    assert [c.track_id for c in cues][:1] == [2] and cues[0].t <= 11.0, (dict(gate.reasons), gate.releases)
    assert not any(2 in b.track_ids for b in gate._bg)


def test_another_object_stepping_in_during_a_dropout_releases_the_background_record():
    # Major (fourth review): a background person at 3.0 m drops out for 0.3 s and a person 4.5 m away appears where
    # they stood, walking at the user (depth off). The IoU tracker matches the newcomer to the old track, so it kept the
    # 3.0 m background reference and was held until nearer than 2.55 m. A range materially farther than the reference
    # is not the object at its place: released (held again only once it rests)
    lin = lambda t: max(0.3, 4.5 - 1.5 * max(0.0, t - 8.3))
    objs = [dict(name="A", label="person", z=lambda t: 3.0, x=lambda t: 0.1, on=lambda t: 1.0 <= t < 8.0),
            dict(name="B", label="person", z=lin, x=lambda t: 0.15, on=lambda t: t >= 8.3)]
    tc = 8.3 + 3.9 / 1.5
    bare, _, ids = boxes_session(OfferAt([1.5, 8.5]), objs, T=tc, depth=False)
    assert bare[-1] == (8.5, "B") and ids["A"] == ids["B"]     # (the tracker kept one id for both)
    cues, gate, _ = boxes_session(OfferAt([1.5, 8.5]), objs, GateConfig(), T=tc, depth=False)
    assert [t for t, n in cues if n == "B"][:1] == [8.5], (cues, gate.releases)
    # (released by the range jump across the dropout, or by its range against the reference: either says "farther")
    assert any(r[2].startswith("background") and "farther" in r[2] for r in gate.releases), gate.releases


class OfferTrack(NearestPolicy):
    """A scheduler that offers given track ids at given decision times ({track id: [times]})."""

    def __init__(self, plan, **kw):
        super().__init__(**kw)
        self.plan = {tid: {round(x, 3) for x in ts} for tid, ts in plan.items()}

    def decide(self, tracks, t, ch=None):
        return super().decide([tr for tr in tracks if round(t, 3) in self.plan.get(tr.track_id, ())], t, ch)


@pytest.mark.parametrize("crosses", [False, True])
def test_a_different_object_of_the_label_does_not_inherit_the_memory(crosses):
    # Major (fourth review, seed 21603): car A, said at 3.3 m, leaves (driving off to the side, or just out of view);
    # car B appears 1.1 s later where A was said, 6.4 m away, coming at 3 m/s. B inherited A's memory (same label, same
    # sector, within switch_gap_s) and its first cue was held as a repeat. Memory passes only to a successor: where
    # the old track was last seen and not materially farther
    zb = lambda t: 6.4 - 3.0 * max(0.0, t - 8.5)
    def scene(t):
        if t < 7.5:
            cx = 0.5 if not crosses or t < 6.0 else 0.5 - 0.25 * (t - 6.0)
            return [dict(tid=1, label="car", bbox=box(cx, h=0.45, w=0.5), dist=3.3, path=0.3)]
        if t < 8.5:
            return []
        return [dict(tid=2, label="car", bbox=box(0.5, h=1.5 / zb(t) * 0.5, w=1.8 / zb(t) * 0.5), dist=zb(t), rel=-3.0,
                     ttc=(zb(t) - 0.6) / 3.0, path=0.3)]
    gate = AnnouncementGate(OfferTrack({1: [1.5], 2: [9.0]}), GateConfig(background=False))
    cues = run(gate, scene, 9.5)
    assert [(c.track_id, c.t) for c in cues] == [(1, 1.5), (2, 9.0)], dict(gate.reasons)
    # nor a person 0.6 m farther than where the one said at 3.8 m vanished 2 s before (seed 21635: half the `closer`
    # step farther is another object; nearer, it may be the same one that came closer unseen)
    gate = AnnouncementGate(OfferTrack({1: [1.0], 2: [8.5]}), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.48, h=0.42), dist=3.8)] if t < 6.5 else
               ([dict(tid=2, bbox=box(0.5, h=0.36), dist=4.4, rel=-1.25, path=0.6)] if t >= 8.5 else []), 9.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 1.0), (2, 8.5)], dict(gate.reasons)
    # a real id switch (the same car under a new id, at the same range, where it vanished) still inherits
    gate = AnnouncementGate(OfferTrack({1: [1.5], 2: [9.0]}), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1 if t < 7.5 else 2, label="car", bbox=box(0.5, h=0.45, w=0.5), dist=3.3, path=0.3)]
               if not 7.5 <= t < 8.5 else [], 9.5)
    assert [(c.track_id, c.t) for c in cues] == [(1, 1.5)] and gate.reasons["repeat"] == 1


def test_find_goal_flicker_does_not_wait_for_confirmation():
    # Major (fourth review, seed 21608): FIND person, depth off, heavy dropouts: the person is seen in 2-frame sightings
    # only (never min_hits at a decision); bare P-CACS says it 3.2 s before contact, the gate never did. A FIND goal
    # does not wait for the gate's confirmation (the tracker's own n_init still holds)
    def scene(t):
        return [dict(tid=int(t * 2), bbox=box(0.9, h=0.4, w=0.09), dist=4.0, depth_conf=0.0, hits=2)] if t in (2.5, 4.5) else []
    gate = AnnouncementGate(OfferAt([2.5]), GateConfig(background=False))
    gate.set_goal(["person"])
    assert [c.t for c in run(gate, scene, 5.0)] == [2.5]
    gate = AnnouncementGate(OfferAt([2.5]), GateConfig(background=False))       # not the goal (off the path): waits
    assert [c.t for c in run(gate, scene, 5.0)] == [4.5]       # for the second sighting (one object seen twice)


@pytest.mark.parametrize("nearer", [0.0, 0.6])
def test_a_static_object_back_after_a_dropout_is_not_said_again(nearer):
    # Minor (fourth review): `reacquired` said a static, already-said person in the path again after every dropout
    # longer than lost_s ("no farther than when last seen" holds half the time on noise). Only one that came
    # materially nearer while out of view is said again
    def scene(t):
        if 5.5 < t < 7.0:
            return []
        return [dict(tid=1 if t <= 5.5 else 2, bbox=box(0.5, h=0.6), dist=2.5 - (nearer if t >= 7.0 else 0.0))]
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cues = run(gate, scene, 12.0)
    assert [c.t for c in cues] == ([0.0, 7.0] if nearer else [0.0]), dict(gate.reasons)
    assert gate.reasons["reacquired"] == (1 if nearer else 0)


def test_a_second_person_walking_up_beside_one_already_said_is_new():
    # Fourth review fuzz (seed 21818): a person said at 2.4 m stands still; another walks up right next to them (boxes
    # overlapping, IoU 0.56, about the same range) and on toward the user. A box overlapping by IoU 0.3 at the same
    # range counted as the first person detected twice, so the newcomer inherited the memory (a repeat). One object
    # detected twice overlaps by dup_iou (0.6)
    def scene(t):
        q = dict(tid=1, bbox=(0.325, 0.2, 0.575, 0.8), dist=2.4)
        return [q] + ([dict(tid=2, bbox=(0.395, 0.2, 0.645, 0.8), dist=2.6, rel=-1.0, path=0.6)] if t >= 7.5 else [])
    gate = AnnouncementGate(OfferTrack({1: [1.0], 2: [8.0]}), GateConfig(background=False))
    assert [(c.track_id, c.t) for c in run(gate, scene, 9.0)] == [(1, 1.0), (2, 8.0)], dict(gate.reasons)


def test_a_background_object_out_of_view_as_the_warm_up_ends_is_judged_as_lost_when_it_returns():
    # Fourth review fuzz (seed 20113): a person static through the warm-up drops out of view just before it ends and
    # comes back 2 s later under a new track id, walking at the user; its background record had no track to measure
    # the dropout against, so it was held as background. The warm-up's track is the record's last one
    z = lambda t: 4.5 if t < 4.0 else 4.5 - 1.1 * (t - 4.0)
    def scene(t):
        if 2.0 < t < 4.0:
            return []
        h, w = 1.6 / z(t) * 0.5, 0.5 / z(t) * 0.5
        return [dict(tid=1 if t <= 2.0 else 2, bbox=(0.45, 0.8 - h, 0.45 + w, 0.8), dist=z(t), depth_conf=0.0,
                     rel=0.0 if t < 4.0 else -1.1, path=0.6)]
    gate = AnnouncementGate(OfferTrack({2: [4.5]}))
    cues = run(gate, scene, 5.0)
    assert [(c.track_id, c.t) for c in cues] == [(2, 4.5)] and any("lost from view" in r[2] for r in gate.releases)


def test_after_a_rebaseline_a_track_that_starts_to_approach_is_not_held_for_the_warm_up():
    # Fourth review fuzz (seed 22449): a person static for 6 s starts walking at the user (no measured depth) just as
    # the gate rebaselines (scene mode); the warm-up judges a track on its whole box history, mostly static, and held
    # the scheduler's one offer. A track whose box grows now (or that closes in by measured depth) is not held
    z = lambda t: 5.2 if t < 6.0 else 5.2 - 1.3 * (t - 6.0)
    def scene(t):
        h, w = 1.6 / z(t) * 0.5, 0.5 / z(t) * 0.5
        return [dict(tid=1, bbox=(0.45, 0.8 - h, 0.45 + w, 0.8), dist=z(t), depth_conf=0.0, path=0.6)]
    gate, sc, cues = AnnouncementGate(OfferTrack({1: [6.5]})), Scene(), []
    for i in range(15):
        t = i * DT
        if t == 6.0:
            gate.rebaseline()
        cues += gate.decide(sc.at(t, scene(t)), t, ChannelState())
    assert [c.t for c in cues] == [6.5], dict(gate.reasons)


def test_a_cue_said_again_by_the_gate_carries_its_distance_now():
    # Fourth review fuzz (seed 22894): a person said "about 6 meters" with depth was said again by the gate when 1.6 m
    # away, after the depth model dropped out: the old phrase was looked up with the track's *current* (low) depth
    # confidence, not found, and the stale "about 6 meters" was spoken. The phrase is found as the cue said it
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=False))
    cue = Cue(t=10.0, track_id=1, label="person", kind="object", message="Person, about 6 meters.", score=1.0,
              policy="nearest", distance_m=5.8)
    trk = Scene().at(13.0, [dict(tid=1, bbox=box(0.5, h=0.6, w=0.22), dist=1.6, depth_conf=0.25)])[0]
    near = gate._refreshed(cue, 0.0, trk, 13.0)
    assert near.message == "Person, very close." and near.distance_m == pytest.approx(gate._range_now(trk))
    far = Scene().at(13.0, [dict(tid=1, bbox=box(0.5, h=0.2, w=0.06), dist=5.0, depth_conf=0.25)])[0]
    assert gate._refreshed(cue, 0.0, far, 13.0).message == "Person."      # (a bbox guess that far: no distance)


# ---- fifth round (scripts/live_safety.py sweeps): tracker re-use, arrivals before a rebaseline, implausible successors --
def test_another_object_the_tracker_keeps_under_the_same_id_is_not_a_repeat():
    # Harness lost_background:380 (learned scheduler, depth off, background off): a person said at 4 m vanishes, and
    # 0.7 s later another person appears where they stood, 6 m away, and walks at the user. The IoU tracker kept the old
    # track id, so the memory of the first person (said 9.5 s before) held the newcomer's only offer as a repeat until
    # `closer` passed 2.5 s later. A range that jumps farther across a dropout is another object: the memory lets go
    zb = lambda t: max(0.3, 6.0 - 1.4 * max(0.0, t - 10.2))
    objs = [dict(name="A", label="person", z=lambda t: 4.0, x=lambda t: 0.1, on=lambda t: t < 9.5),
            dict(name="B", label="person", z=zb, x=lambda t: 0.1, on=lambda t: t >= 10.2)]
    tc = 10.2 + 5.4 / 1.4
    bare, _, ids = boxes_session(OfferAt([1.0, 10.5]), objs, T=tc, depth=False)
    assert (10.5, "B") in bare and ids["A"] == ids["B"]          # (one track id for both)
    for gcfg in (GateConfig(background=False), GateConfig()):
        cues, gate, _ = boxes_session(OfferAt([1.0, 10.5]), objs, gcfg, T=tc, depth=False)
        assert [t for t, n in cues if n == "B"][:1] == [10.5], (cues, dict(gate.reasons), gate.releases)
    # the same person back after the same dropout, at their place, is still a repeat
    objs = [dict(name="A", label="person", z=lambda t: 4.0, x=lambda t: 0.1, on=lambda t: not 9.5 <= t < 10.2)]
    cues, gate, _ = boxes_session(OfferAt([1.0, 10.5]), objs, GateConfig(background=False), T=12.0, depth=False)
    assert [t for t, _ in cues] == [1.0] and gate.reasons["repeat"] == 1


def test_a_person_who_walked_into_place_before_a_rebaseline_is_said_once():
    # Harness side_entry:194 (learned scheduler, controller): a person walks in from the left and stops 5.7 m ahead in
    # the path; the user stands still, the controller enters scene mode and rebaselines, and the person, static through
    # the new warm-up, became background: the scheduler's first offer (bare says it) was held. An object the gate saw
    # arrive (a person or an animal whose box moved into place) is said once before it is background; one that stood
    # there all along stays quiet
    def scene(t):
        cx = 0.2 + 0.32 * min(1.0, max(0.0, t - 2.0) / 2.5)
        walker = [dict(tid=1, bbox=box(cx, h=0.27, w=0.08), dist=5.7)] if t >= 2.0 else []
        return walker + [dict(tid=2, bbox=box(0.8, h=0.3, w=0.09), dist=5.0)]
    gate, sc, cues = AnnouncementGate(OfferTrack({1: [15.5], 2: [16.0]}), GateConfig()), Scene(), []
    for i in range(35):
        t = i * DT
        if t == 12.0:
            gate.rebaseline()                                   # (the walk-in is older than the tracker's box history)
        cues += gate.decide(sc.at(t, scene(t)), t, ChannelState())
    assert [(c.track_id, c.t) for c in cues] == [(1, 15.5)], (dict(gate.reasons), [(b.track_ids, b.say_once) for b in gate._bg])
    assert gate.last_verdict["reason"] == "background"          # the person who was there all along


def test_a_successor_nearer_than_it_could_have_come_is_a_different_object():
    # a chair said 6 m ahead vanishes; half a second later a chair appears 2 m ahead where it was. A chair does not move
    # itself: 4 m nearer in a second is another chair, new to the user (it inherited the memory: a repeat within
    # min_gap_s). A person 1 m nearer after the same gap may be the one said, come closer unseen: a successor
    gate = AnnouncementGate(OfferTrack({1: [4.5], 2: [5.0]}), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, label="chair", bbox=box(0.5, h=0.12), dist=6.0)] if t <= 4.5 else
               [dict(tid=2, label="chair", bbox=box(0.5, h=0.35), dist=2.0)], 6.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 4.5), (2, 5.0)], dict(gate.reasons)
    gate = AnnouncementGate(OfferTrack({1: [4.5], 2: [5.0]}), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.5, h=0.3), dist=4.0)] if t <= 4.5 else
               [dict(tid=2, bbox=box(0.5, h=0.4), dist=3.0)], 6.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 4.5)] and gate._seen[2].pred == (1,)


def test_a_successor_of_an_object_that_walked_away_since_it_was_said_is_new():
    # web gate successorOf: a person said 3 m ahead walks off to 6 m and vanishes; another person comes from there (a
    # new track at the range the first one vanished at). The first was said at 3 m: the newcomer at 6 m was held as a
    # repeat until it came materially closer than 3 m. One that walked off and vanished, and another coming from there,
    # are two objects
    zf = lambda t: 3.0 + 0.6 * min(5.0, max(0.0, t - 1.0))
    def scene(t):
        if t <= 6.0:
            return [dict(tid=1, bbox=box(0.5, h=min(0.95, 1.0 / zf(t))), dist=zf(t), rel=0.6 if 1.0 <= t < 6.0 else 0.0)]
        if t < 6.5:
            return []                                           # (out of view less than lost_s: not `reacquired`)
        z = 6.0 - 1.5 * (t - 6.5)
        return [dict(tid=2, bbox=box(0.5, h=min(0.95, 1.0 / z)), dist=z, rel=-1.5, path=0.6)]
    gate = AnnouncementGate(OfferTrack({1: [1.0], 2: [7.0]}), GateConfig(background=False))
    cues = run(gate, scene, 8.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 1.0), (2, 7.0)], dict(gate.reasons)
    assert gate._seen[2].pred == (1,)                           # (it does take over the first one's track)


def test_someone_stepping_into_the_path_during_the_warm_up_is_not_held_for_it():
    # Harness approach:1272 (learned scheduler, depth off): a person appears 3.6 m ahead in the walking path 2 s after
    # the camera started and walks at the user 1 s later (contact 1.4 s after that). The bare scheduler said it at once;
    # the gate held its only offer for the warm-up (static for a few samples) and said it 1 s later, 0.9 s before
    # contact. An object that came into view after the warm-up began is not part of the scene it learns
    z = lambda t: 3.6 if t < 3.0 else max(0.3, 3.6 - 2.3 * (t - 3.0))
    def scene(t):
        plant = dict(tid=1, label="potted plant", bbox=(0.85, 0.5, 0.95, 0.7), dist=4.0)
        person = [dict(tid=2, bbox=box(0.48, h=min(0.95, 1.5 / z(t))), dist=z(t), depth_conf=0.25, path=0.6)] if t >= 2.0 else []
        return [plant] + person
    gate = AnnouncementGate(OfferTrack({2: [2.5]}))
    cues = run(gate, scene, 4.0)
    assert [(c.track_id, c.t) for c in cues][:1] == [(2, 2.5)], dict(gate.reasons)   # (then again as it approaches)
    # a person in the path from the camera's first frame is still the warm-up's (held, background unless near: by the
    # box range at its nearest, BOX_SMALL x 7 m = 3.5 m, beyond bg_path_m)
    gate = AnnouncementGate(OfferTrack({2: [2.5]}))
    cues = run(gate, lambda t: [dict(tid=2, bbox=box(0.48, h=0.22), dist=7.0, depth_conf=0.25, path=0.6)], 4.0)
    assert cues == [] and gate.last_verdict["reason"] == "warmup"


def test_box_jitter_of_a_static_object_is_not_closing():
    # Harness read_scene:1415 / lost_background:1448 (depth off, box jitter of 0.8 % of the frame): the box of a static
    # sign or chair 1.6-2 m ahead grew 10 % between two single frames now and then, so the gate said it again by itself
    # as `near` / `closing` and the message held back the scheduler's warning about a car. Box growth now compares the
    # medians of the boxes around each end: 1 false closing in 1450 decisions (single frames: 25)
    import random
    from seeforme.geometry import CameraModel
    cam, gate = CameraModel(), AnnouncementGate(NearestPolicy(), GateConfig())
    false = n = 0
    for z, label, H, W, base_h in ((1.6, "chair", 0.9, 0.5, 0.0), (2.0, "text", 0.3, 0.8, 1.0)):
        base = cam.clip(cam.project_box(0.0, z, H, W, base_height=base_h))
        for seed in range(5):
            rng = random.Random(seed)
            trk = Track(track_id=1, label=label, kind="object", bbox=base, confidence=0.9, first_seen=0.0, last_seen=0.0)
            for i in range(300):
                trk.bbox = tuple(c + rng.gauss(0.0, 0.008) for c in base)
                trk.bbox_history.append((i / 10, trk.bbox))
                if i >= 10 and i % 5 == 0:
                    n += 1
                    false += gate._box_closing(trk, z) >= gate.gate_cfg.nodepth_approach_mps
    assert n == 580 and false <= 2, false
    # a real approach still shows: the same box growing as the user walks in at 0.5 m/s
    trk = Track(track_id=1, label="chair", kind="object", bbox=base, confidence=0.9, first_seen=0.0, last_seen=0.0)
    for i in range(15):
        zz = 2.0 - 0.05 * i
        trk.bbox = cam.clip(cam.project_box(0.0, zz, 0.9, 0.5))
        trk.bbox_history.append((i / 10, trk.bbox))
    assert gate._box_closing(trk, 1.3) >= 0.3


class BusyAware(OfferTrack):
    """Like P-CACS for a routine cue: offers nothing while it is shown a busy channel."""

    def decide(self, tracks, t, ch=None):
        return [] if ch is not None and ch.busy else super().decide(tracks, t, ch)


def test_a_message_the_gate_said_on_its_own_does_not_hold_back_the_scheduler():
    # Harness find:1017 (read mode): the gate said a chair cue it had held (muted) once that chair approached; while
    # that message played, the scheduler, shown the busy channel, did not offer the FIND-goal chair coming into the path,
    # which the controller without the gate said at once. The scheduler is now shown the channel as it would be without
    # the gate's own message (as the controller does for cane cues and replies), and its safety cue cuts that message
    za = lambda t: 4.5 if t < 3.0 else 4.5 - 1.0 * (t - 3.0)
    zh = lambda t: 5.5 - 0.9 * t
    def scene(t):
        return [dict(tid=1, label="chair", bbox=box(0.75, h=0.2), dist=za(t), rel=-1.0 if t >= 3.0 else 0.0,
                     ttc=(za(t) - 0.6) / 1.0 if t >= 3.0 else math.inf),
                dict(tid=2, label="chair", bbox=box(0.5, h=0.18), dist=zh(t), rel=-0.9, ttc=(zh(t) - 0.6) / 0.9, path=0.8)]
    gate = AnnouncementGate(BusyAware({1: [1.0], 2: [3.5]}), GateConfig(background=False, objects_muted=True))
    sc, said, busy_until, cur = Scene(), [], -1.0, None
    for i in range(10):
        t = i * DT
        ch = ChannelState(busy=t < busy_until, remaining_s=max(0.0, busy_until - t), current_track=cur if t < busy_until else None)
        for c in gate.decide(sc.at(t, scene(t)), t, ch):
            said.append((c.track_id, t, c.action))
            busy_until, cur = t + 2.4, c.track_id               # (a long message)
    # chair 1: muted at 1.0 (no TTC), said by the gate itself at 3.0 once it approaches (TTC 3.9 s); chair 2, offered
    # while that message plays, cuts it
    assert said == [(1, 3.0, "speak"), (2, 3.5, "interrupt")], (said, dict(gate.reasons))


def test_find_goal_another_unsaid_object_is_said_when_the_scheduler_repeats_one():
    # Harness find:2133 (learned scheduler, FIND "person"): two people in view; the scheduler named the one on the right
    # every 1.5 s (all held as repeats) and never the one 6 m away on the left, which the bare scheduler named at 19 s.
    # When a FIND-goal cue is held as a repeat while another object the goal names is in view and was never said, the
    # gate says that one instead
    def scene(t):
        return [dict(tid=1, bbox=box(0.62, h=0.45), dist=3.0), dict(tid=2, bbox=box(0.3, h=0.22), dist=6.0)]
    for goal in (["person"], None):
        gate = AnnouncementGate(OfferTrack({1: [1.0, 2.5, 4.0]}), GateConfig(background=False))
        if goal:
            gate.set_goal(goal)
        cues = run(gate, scene, 5.0)
        assert [(c.track_id, c.t) for c in cues] == ([(1, 1.0), (2, 2.5)] if goal else [(1, 1.0)]), dict(gate.reasons)


def test_a_second_car_whose_box_covers_the_first_is_not_a_duplicate_at_another_depth():
    # Harness twins:1375: car A said 5.4 m ahead; car B pulls in 0.8 m in front of it, its box over A's (IoU >= dup_iou),
    # then drives at the user. Within the `closer` step (1.1 m) the two boxes counted as one car detected twice, so B
    # inherited A's memory and its cue was held as a repeat. With measured depth on both, a duplicate box is at the same
    # depth (within max(release_m, release_frac x range)); without depth the coarser box range still decides, and B
    # takes A's memory, but in the walking path 0.83 m nearer than the car the user heard of is news either way (closing)
    from seeforme.geometry import CameraModel
    from seeforme.types import bbox_iou
    cam = CameraModel()
    ba, bb = cam.clip(cam.project_box(0.33, 5.37, 1.5, 1.8)), cam.clip(cam.project_box(0.06, 4.54, 1.5, 1.8))
    assert bbox_iou(ba, bb) >= GateConfig().dup_iou
    for conf, why in ((0.9, "new"), (0.25, "closing")):
        gate = AnnouncementGate(OfferTrack({1: [2.0], 2: [7.0]}), GateConfig(background=False))
        cues = run(gate, lambda t: [dict(tid=1, label="car", bbox=ba, dist=5.37, depth_conf=conf)]
                   + ([dict(tid=2, label="car", bbox=bb, dist=4.54, depth_conf=conf)] if t >= 5.5 else []), 7.5)
        assert [(c.track_id, c.t) for c in cues] == [(1, 2.0), (2, 7.0)], (conf, dict(gate.reasons))
        assert gate.reasons[why] == 1 + (why == "new")


def test_find_goal_an_unsaid_sign_comes_before_a_reminder_about_another():
    # Harness find:2709 (learned scheduler, FIND "exit"): the signs EXIT and EXIT 2 are both in view (the goal names
    # both); the scheduler named EXIT 2 when the goal was set and again as a reminder, never EXIT, which the bare
    # scheduler named. A goal cue that is only a reminder (or a repeat) while another object the goal names was never
    # said gives way to that one, in its own words
    def scene(t):
        return [dict(tid=1, kind="text", label="text", text="EXIT", bbox=box(0.35, h=0.08, w=0.2), dist=4.5),
                dict(tid=2, kind="text", label="text", text="EXIT 2", bbox=box(0.6, h=0.08, w=0.25), dist=3.0)]
    gate = AnnouncementGate(OfferTrack({2: [2.5, 11.0]}), GateConfig(background=False))
    cues = run(gate, scene, 2.0)
    gate.set_goal(["exit"])
    cues += run(gate, scene, 12.0, t0=2.5)
    assert [(c.track_id, c.t) for c in cues] == [(2, 2.5), (1, 11.0)], dict(gate.reasons)
    assert "EXIT" in cues[1].message and "EXIT 2" not in cues[1].message and cues[1].text == "EXIT"


# ---- second verification (gate_verify_r2): the scheduler's own view, FIND swaps, handovers, flickers ---------------
def _cam_box(x, z, label="person"):
    from seeforme.geometry import CameraModel
    from seeforme.geometry.camera import TYPICAL_SIZE_M
    H, W = TYPICAL_SIZE_M.get(label, (1.0, 0.5))
    return CameraModel().clip(CameraModel().project_box(x, z, H, W))


def test_the_gates_own_messages_leave_the_scheduler_as_it_would_be_alone():
    # Critical (static_path:70997 with P-CACS; find:71473, swap:91001 with the learned scheduler): the gate said cues
    # itself (a false release, a rescue) and stamped the track as announced, charged or refunded the speech budget,
    # and showed the scheduler a free channel where alone its held cue would still be playing. The scheduler then
    # decided on another state than alone and did not give the warning it gives alone (P-CACS's novelty decay kept the
    # bench unsaid until contact). The gate's own messages and its holds now leave the scheduler's tracks, budget and
    # channel as they would be without the gate
    z = lambda t: 5.6 if t < 12.4 else 5.6 - 0.37 * (t - 12.4)

    class Watch(OfferAt):
        def decide(self, tracks, t, ch=None):
            self.seen.append((round(t, 3), bool(ch is not None and ch.busy)))
            return super().decide(tracks, t, ch)
    inner = Watch([2.0, 10.0])
    inner.seen = []
    gate, sc = AnnouncementGate(inner, GateConfig(background=False)), Scene()
    for i in range(int(25.5 / DT) + 1):
        t = i * DT
        go = t >= 12.4
        gate.decide(sc.at(t, [dict(tid=1, label="car", bbox=box(0.46, h=min(0.95, 1.4 / z(t)), w=min(0.95, 1.7 / z(t))),
                                   dist=z(t), rel=-0.37 if go else 0.0, ttc=(z(t) - 0.6) / 0.37 if go else math.inf,
                                   path=0.6)]), t, ChannelState())
    trk = sc.tracks[1]
    assert gate.stats["rescued"] >= 1                                       # the gate did say the car itself ...
    assert trk.last_announced == 10.0 and trk.announce_count == 2           # ... but the track has only the scheduler's
    assert inner.budget == pytest.approx(inner.cfg.budget_capacity_s - 2 * 1.2)   # marks, and its budget is as alone
    seen = dict(inner.seen)
    assert seen[10.5] and seen[11.0] and not seen[11.5]     # its held cue (10.0 s, 1.2 s long) plays as it would alone


def test_find_goal_coming_closer_in_the_path_is_not_swapped_for_another_one():
    # Major (find:71797, learned scheduler, FIND "person", box depth): the scheduler named the goal person walking at
    # the user (4.8 m, in the path, 1.3 m nearer than said); the gate held it as a repeat and said instead another
    # person off to the left, who left view a moment later. The goal person was next said 1.4 s before contact. A goal
    # object in the path that came nearer by the `closing` step is news: no swap (and a swapped cue waits as held)
    zh = lambda t: 6.1 if t < 3.0 else 6.1 - 0.65 * (t - 3.0)
    def scene(t):
        return [dict(tid=1, bbox=_cam_box(0.0, zh(t)), dist=zh(t), depth_conf=0.25, path=0.53),
                dict(tid=2, bbox=_cam_box(-2.4, 4.7), dist=4.7, depth_conf=0.25, path=0.04)]
    gate = AnnouncementGate(OfferTrack({1: [1.0, 5.0]}), GateConfig(background=False))
    gate.set_goal(["person"])
    cues = run(gate, scene, 6.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 1.0), (1, 5.0)], dict(gate.reasons)
    assert gate.reasons["closing"] == 1


@pytest.mark.parametrize("gap_frames", [0, 1])
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("offers", ["every step", "once"])
def test_a_person_stepping_in_where_another_stood_without_a_gap_is_new(gap_frames, background, offers):
    # Major (bgref:90278, bgref:90998, the 144-scene handover grid; learned scheduler, no depth): person A, said at 3 m,
    # leaves; person B steps in where A stood 0.9 m farther in the same frame or the next one, then walks at the user at
    # 1.7 m/s. The IoU tracker keeps one id and no dropout of jump_gap_s shows the range jump, so B's cue was held as a
    # repeat of A (or as A's background) until B came 1 m nearer than 3 m, under 1 s before contact. A step farther
    # from one frame to the next that the object could not walk lets go of A's memory and background record; a cue held
    # as a repeat before the step showed is then said as new (the scheduler offered B once, at the switch)
    from seeforme.geometry import TrackDynamics
    from seeforme.geometry import CameraModel
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.types import Detection
    tracker = IoUTracker(PipelineConfig().tracker, dynamics=TrackDynamics(CameraModel()))
    inner = NearestPolicy() if offers == "every step" else OfferAt([0.5, 9.5 + 0.5 * gap_frames])   # (first step with B)
    gate = AnnouncementGate(inner, GateConfig(background=background))
    said = []
    for i in range(125):
        t = round(i / 10, 3)
        if i < 95:
            dets = [Detection("person", 0.9, _cam_box(0.1, 3.0))]
        elif i < 95 + gap_frames:
            dets = []
        else:
            dets = [Detection("person", 0.9, _cam_box(0.08, max(0.3, 3.9 - 1.7 * max(0.0, t - 9.8))))]
        tracks = tracker.update(dets, t)
        if i % 5 == 0:
            said += [c.t for c in gate.decide(tracks, t, ChannelState())]
    b = [x for x in said if x >= 9.5]
    assert b and b[0] <= 10.0, (said, dict(gate.reasons))           # contact at 11.7 s


def test_an_object_in_the_path_that_came_nearer_is_said_again_after_hold_moving_s():
    # Major (flicker:72656, learned scheduler, no depth, 2-frame sightings): a chair said at 4.2 m; 9 s later the user
    # walks at it, and the scheduler names it again at 3.75 m. Under flicker no box growth shows, and 0.45 m is under
    # the `closing` step (0.63 m) and the `closer` one (1 m): held as a static repeat for 30 s, it was said 0.7 s
    # before contact. An object in the walking path that is release_m nearer than when it was said is held only
    # hold_moving_s, like a moving one
    z = lambda t: 4.2 if t < 8.5 else 4.2 - 0.45 * (t - 8.5)
    def scene(t):
        return [dict(tid=1, label="chair", bbox=_cam_box(0.0, z(t), "chair"), dist=z(t), depth_conf=0.25, path=0.6)]
    gate = AnnouncementGate(OfferTrack({1: [0.5, 6.5, 9.5]}), GateConfig(background=False))
    cues = run(gate, scene, 10.0)
    assert [c.t for c in cues] == [0.5, 9.5] and gate.reasons["reminder"] == 1, dict(gate.reasons)


def test_a_slow_approach_in_the_path_is_said_again_on_a_range_step():
    # Major (read_scene:72879, learned scheduler, box depth): the user walks at a chair at 0.2 m/s; said at 1.9 m, the
    # scheduler names it again at 1.3 m, 3.5 s before contact. A closing speed of 0.2 m/s is under release_mps and
    # nodepth_approach_mps, so neither `closing` nor `near` held. A cue the scheduler offers about an object in the
    # walking path passes on the `closing` step alone (the gate's own rescue still needs a closing speed)
    z = lambda t: 2.0 - 0.2 * t
    def scene(t):
        return [dict(tid=1, label="chair", bbox=_cam_box(0.0, z(t), "chair"), dist=z(t), depth_conf=0.25, path=0.7)]
    gate = AnnouncementGate(OfferTrack({1: [0.5, 3.5]}), GateConfig(background=False))
    cues = run(gate, scene, 4.0)
    assert [c.t for c in cues] == [0.5, 3.5] and gate.reasons["closing"] == 1, dict(gate.reasons)


@pytest.mark.parametrize("background", [True, False])
def test_an_object_in_the_path_that_leaves_view_right_after_its_cue_was_held_is_said(background):
    # Major (rescue_switch:90860, 90676, 90572; learned scheduler, no depth): a person standing 4.9 m ahead in the path
    # (background, or said long ago) starts to walk at the user just before the scheduler names it again, and drops
    # out of view right after for 1.5 s. Nothing in view showed the approach, so the cue was held; the gate first said
    # the person back in view, 1.1 s before contact. A cue about an object in the path that the user did not hear about
    # for hold_moving_s, held as nothing new, is said at the next decision if the object leaves view meanwhile
    z = lambda t: 4.9 if t < 13.3 else 4.9 - 1.46 * (t - 13.3)
    def scene(t):
        if 13.6 <= t < 15.0:
            return []
        return [dict(tid=1 if t < 13.6 else 2, bbox=_cam_box(-0.2, z(t)), dist=z(t), depth_conf=0.25, path=0.55)]
    gate = AnnouncementGate(OfferTrack({1: [1.5, 13.5]}), GateConfig(background=background))
    cues = run(gate, scene, 15.5)
    first = [(c.track_id, c.t) for c in cues if c.t > 2.0][:1]
    assert first == [(1, 14.0)] and gate.reasons["lost (deferred)"] == 1, (first, dict(gate.reasons))
    # a static object in the path that leaves view is said at most once per hold_moving_s (a reacquired person)
    def flick(t):
        return [] if t % 3.0 >= 2.0 else [dict(tid=1 + int(t // 3.0), bbox=_cam_box(-0.2, 3.0), dist=3.0, depth_conf=0.25,
                                               path=0.55)]
    gate = AnnouncementGate(NearestPolicy(), GateConfig(background=background))
    cues = run(gate, flick, 60.0)
    assert len(cues) <= 1 + 60.0 / GateConfig().hold_moving_s, [c.t for c in cues]


def test_a_routine_cue_waits_while_the_gates_own_message_plays():
    # The live speakers drop a message they get while they speak (feedback/speech.py Speaker.say). While the gate's own
    # message plays (here a cue held for confirmation, said a step later), the scheduler is shown its own channel, which
    # is free, and offers a routine cue about another object: the gate passed it as "speak" onto the busy channel, where
    # it was lost although the gate recorded it as said. A safety cue still cuts the gate's message; a routine one now
    # waits for the channel
    def scene(t):
        return [dict(tid=1, bbox=box(0.95), dist=4.0, **({"hits": 1} if t == 0.0 else {})),
                dict(tid=2, bbox=box(0.08), dist=5.0)]
    gate = AnnouncementGate(OfferTrack({1: [0.0], 2: [1.0]}), GateConfig(background=False))
    sc, said, busy_until, cur = Scene(), [], -1.0, None
    for i in range(10):
        t = i * DT
        ch = ChannelState(busy=t < busy_until, remaining_s=max(0.0, busy_until - t), current_track=cur if t < busy_until else None)
        for c in gate.decide(sc.at(t, scene(t)), t, ch):
            assert c.action != "speak" or t >= busy_until            # never onto a busy channel
            said.append((c.track_id, t, c.action))
            busy_until, cur = t + 2.4, c.track_id
    assert said == [(1, 0.5, "speak"), (2, 3.0, "speak")], (said, dict(gate.reasons))


# ---- third verification (gate_verify_r3): the scheduler's view built from its own offers only; desk noise ------------
class LongOffers(OfferTrack):
    """OfferTrack whose cues hold the channel 2.4 s, offering only on a channel it is shown free (P-CACS's rule for a
    routine cue), recording what it was shown at every decision: (t, busy, remaining, current track)."""

    def __init__(self, plan, **kw):
        super().__init__(plan, **kw)
        self.shown = []

    def decide(self, tracks, t, ch=None):
        busy = bool(ch is not None and ch.busy)
        self.shown.append((round(t, 3), busy, round(ch.remaining_s, 3) if busy else 0.0, ch.current_track if busy else None))
        if busy:
            return []
        return [dataclasses.replace(c, duration_s=2.4) for c in super().decide(tracks, t, ch)]


def channel_session(policy, scene, t_end, gcfg=None):
    """A decision every 0.5 s on a speech channel (a "speak" holds it for the cue's duration_s, an interrupt cuts it, a
    "speak" onto a busy channel is dropped, as the live speaker does): the policy alone (gcfg None) or gated.
    Returns [(track id, t, action)] of what was said and the gate."""
    dec = policy if gcfg is None else AnnouncementGate(policy, gcfg)
    sc, said, busy_until, cur = Scene(), [], -1.0, None
    for i in range(int(round(t_end / DT)) + 1):
        t = i * DT
        busy = t < busy_until - 1e-9
        ch = ChannelState(busy=busy, remaining_s=max(0.0, busy_until - t), current_track=cur if busy else None)
        for c in dec.decide(sc.at(t, scene(t)), t, ch):
            if c.action == "speak" and busy:
                continue                                           # dropped
            said.append((c.track_id, t, c.action))
            if c.action != "haptic":
                busy_until, cur = t + c.duration_s, c.track_id
    return said, (dec if gcfg is not None else None)


def test_a_routine_cue_said_late_holds_the_schedulers_channel_only_from_when_it_was_offered():
    # Critical (gate_verify_r3, v3_find_disp:620507, learned scheduler): a routine cue the scheduler offered while the
    # gate's own message played waited for the channel and was said 1 s late; the gate then recorded it as the
    # scheduler's own message, so the scheduler was shown the real channel, busy with it, 1 s longer than alone. It then
    # decided on another channel than alone and never named a person walking at the user. The scheduler is now shown
    # only its own offers, each from when it was offered: what it is shown and what it offers are what they are alone
    def scene(t):
        return [dict(tid=1, bbox=box(0.95), dist=4.0, **({"hits": 1} if t == 0.0 else {})),   # (deferred: unconfirmed)
                dict(tid=2, bbox=box(0.08), dist=5.0), dict(tid=3, bbox=box(0.3, h=0.3), dist=6.0)]
    plan = {1: [0.0], 2: [2.5], 3: [5.0]}
    bare_pol, gated_pol = LongOffers(plan), LongOffers(plan)
    bare, _ = channel_session(bare_pol, scene, 8.0)
    said, gate = channel_session(gated_pol, scene, 8.0, GateConfig(background=False))
    assert [s[:2] for s in bare] == [(1, 0.0), (2, 2.5), (3, 5.0)]
    assert gated_pol.shown == bare_pol.shown, [(a, b) for a, b in zip(bare_pol.shown, gated_pol.shown) if a != b]
    # the gate's own message (track 1, deferred to 0.5 s) and the routine cues that waited for the channel
    assert said == [(1, 0.5, "speak"), (2, 3.0, "speak"), (3, 5.5, "speak")], (said, dict(gate.reasons))


@pytest.mark.parametrize("seed", range(3))
def test_a_shallow_lean_on_a_noisy_depth_map_is_not_an_approach(seed):
    # Desk noise (gate_verify_r3, desk:200514): a 0.32 m lean (shallower than the release step, 0.35 m) on depth noise
    # of 0.067 m released the operator once, and the gate said the user. The operator's release step now adds twice its
    # own depth noise at rest (estimated at the warm-up and at every re-capture)
    for make in (lambda: NearestPolicy(recent_block_s=2.8), lambda: PCACSPolicy(PipelineConfig().pcacs)):
        cues, _, gate = desk_session(make(), GateConfig(operator_filter=True), z0=0.8, noise=0.067, lean_m=0.32, T=60.0,
                                     seed=seed)
        assert cues == [], (seed, [c.t for c in cues], gate.releases)
        assert 0.02 < gate._operator.noise < 0.06


@pytest.mark.parametrize("depth", [True, False])
def test_an_arm_on_the_desk_user_is_the_user_but_a_visitor_is_not(depth):
    # Desk noise (gate_verify_r3, desk:202425 and desk:201690, the harness's "arm" variant): a hand or arm of the desk
    # user detected as a second, smaller person box over the user's (flickering) was an ordinary object: a reminder
    # every hold_static_s, a `lost` cue when it flickered out, a buzz. A box attached to the user's (its bottom cut by
    # the frame where the user's is, from its first sighting, overlapping it, not closing in, not farther than the user)
    # is the user. A visitor who walks up in front of the user never was attached: said as before
    user = _desk_user(0.8, 0.0, lambda t: True)
    arm = dict(name="arm", label="person", box=lambda t: (0.6, 0.45, 0.76, 0.98), z=lambda t: 0.5,
               on=lambda t: t >= 4.0, drop=lambda t: int(round(t * 10)) % 5 >= 2)   # (after the warm-up: not background)
    zv = lambda t: max(0.3, 3.5 - 0.6 * max(0.0, t - 61.0))
    visitor = dict(name="v", label="person", z=zv, x=lambda t: 0.05, on=lambda t: t >= 60.0)
    for make in (lambda: PCACSPolicy(PipelineConfig().pcacs), lambda: NearestPolicy(recent_block_s=2.8)):
        bare, _, _ = boxes_session(make(), [user, arm, visitor], T=66.0, depth=depth)
        cues, gate, _ = boxes_session(make(), [user, arm, visitor], GateConfig(operator_filter=True), T=66.0, depth=depth)
        assert [(t, n) for t, n in cues if n in ("user", "arm") and 8.0 <= t < 60.0] == [], (cues, gate.releases)
        tb = next((t for t, n in bare if n == "v"), None)          # (P-CACS without depth never names the visitor)
        tg = next((t for t, n in cues if n == "v"), None)
        assert tb is None or (tg is not None and tg <= tb + DT), (tb, tg, cues)


def test_a_near_object_said_a_moment_ago_that_the_user_walked_up_to_is_said_again():
    # Harness find:500033 (P-CACS, no depth model, FIND "bench"): two benches at one place ahead (one a step behind the
    # other: one object to the gate); the nearer was said at 7.5 s, the user walks at them, and the scheduler said the
    # other "very close" at 9.0 s, 1.7 s before contact. The gate held it as a repeat: within min_gap_s (2.5 s) of the
    # last mention, and by then the box filled the frame (no box range, no growth to measure). Offered by the scheduler,
    # a near object (under nodepth_near_m, or its box bottom low in the frame) in the path that is the `closing` step
    # nearer than when said is news within min_gap_s; far away the gap still holds
    def scene(z_said, z_now):
        def f(t):
            z = z_said if t < 2.0 else z_now
            return [dict(tid=1, label="bench", bbox=_cam_box(0.0, z, "bench"), dist=z, depth_conf=0.25, path=0.6)]
        return f
    for z_said, z_now, said in ((2.6, 1.8, [1.0, 3.0]), (6.0, 5.0, [1.0])):
        gate = AnnouncementGate(OfferAt([1.0, 3.0]), GateConfig(background=False))
        cues = run(gate, scene(z_said, z_now), 3.0)
        assert [c.t for c in cues] == said, (z_said, [c.t for c in cues], dict(gate.reasons))


def test_read_mode_never_mutes_an_object_coming_at_the_user():
    # Harness side_entry:71409 and read_scene:300131 (read mode, auto or pinned): a person walking at the user at
    # 0.55 m/s from 4 m (TTC 6.4 s, over approach_ttc_s) was muted; the controller without the gate (the adapter's
    # filter) and navigate mode warned at once. Read mode mutes the static scene around a user who reads: a measured
    # closing speed of release_mps, a vehicle or the FIND goal is never muted; a static chair still is
    z = lambda t: 4.6 - 0.55 * t
    for rel, said in ((-0.55, [1.0]), (0.0, [])):
        gate = AnnouncementGate(OfferAt([1.0]), GateConfig(background=False, objects_muted=True))
        cues = run(gate, lambda t: [dict(tid=1, bbox=box(0.5, h=min(0.95, 1.7 / z(t))), dist=z(t) if rel else 4.6,
                                         rel=rel, ttc=(z(t) - 0.6) / 0.55 if rel else math.inf, path=0.3)], 1.5)
        assert [c.t for c in cues] == said, (rel, dict(gate.reasons))
    gate = AnnouncementGate(OfferAt([1.0]), GateConfig(background=False, objects_muted=True))
    assert [c.t for c in run(gate, lambda t: [dict(tid=1, label="car", bbox=box(0.8, h=0.2, w=0.3), dist=9.0)], 1.5)] == [1.0]


# ---- fourth verification (gate_verify_r4): FIND swaps at motion onset, rebaselines, cut messages, depth switching ----
def on_channel(gate, script, t_end, events=None, speech=lambda c: 1.2):
    """Drive the gate on a speech channel (a message holds it for speech(cue) s, an interrupt cuts it) and return the
    cues it let through. events: {time: fn()} called before that decision (a rebaseline)."""
    sc, said, busy_until, cur = Scene(), [], -1.0, None
    for i in range(int(round(t_end / DT)) + 1):
        t = round(i * DT, 3)
        if events and t in events:
            events[t]()
        busy = t < busy_until - 1e-9
        ch = ChannelState(busy=busy, remaining_s=max(0.0, busy_until - t), current_track=cur if busy else None)
        for c in gate.decide(sc.at(t, script(t)), t, ch):
            said.append(c)
            c.duration_s = speech(c)
            busy_until, cur = t + c.duration_s, c.track_id
    return said


def test_find_goal_starting_to_come_at_the_user_is_not_swapped_for_another_one():
    # Critical (gate_verify_r4, v4_swap_fast:975020 and 15 more; v4_find_two_goals:900323): FIND "person"; the goal
    # person H stands 4.4 m away in the path (said at 3.5 s) and starts to walk at the user at 1.5 m/s at 9.15 s; the
    # scheduler offers H again at 9.5 s, as it starts. Its smoothed range was 0.5 m nearer than said (the `closing` step
    # is 0.66 m) and the tracker's closing speed lagged at 0.15 m/s, so the gate said instead another goal person D, off
    # the path and never said: D's message then held the channel and the closing rescue for H waited for it (H was next
    # named 1 s before contact). Half the closing step nearer is news enough: no swap, and the held cue about H is said
    # at the next decision as it closes
    zh = lambda t: 4.4 if t < 9.15 else 4.4 - 1.5 * (t - 9.15)
    def scene(t):
        h = dict(tid=1, bbox=_cam_box(0.0, zh(t)), dist=zh(t), path=0.6, rel=-0.15 if t >= 9.15 else 0.0)
        return [h] + ([dict(tid=2, bbox=_cam_box(-2.8, 5.4), dist=5.4, path=0.05)] if t >= 7.0 else [])
    gate = AnnouncementGate(OfferTrack({1: [3.5, 9.5]}), GateConfig(background=False))
    gate.set_goal(["person"])
    said = [(c.track_id, c.t) for c in on_channel(gate, scene, 11.0)]
    assert said[:2] == [(1, 3.5), (1, 10.0)], (said, dict(gate.reasons))
    # a goal person who stays where they were said is still swapped for the one the user never heard of (find:2133)
    gate = AnnouncementGate(OfferTrack({1: [3.5, 9.5]}), GateConfig(background=False))
    gate.set_goal(["person"])
    said = [(c.track_id, c.t) for c in on_channel(gate, lambda t: [dict(tid=1, bbox=_cam_box(0.0, 4.4), dist=4.4, path=0.6),
                                                                  dict(tid=2, bbox=_cam_box(-2.8, 5.4), dist=5.4, path=0.05)], 11.0)]
    assert said == [(1, 3.5), (2, 9.5)], (said, dict(gate.reasons))


@pytest.mark.parametrize("case", ["depth then box", "owed", "box only"])
def test_a_rebaseline_or_a_box_range_does_not_silence_a_child_in_the_path(case):
    # Critical (gate_verify_r4, v4_flip_size:970034, learned scheduler): a child (0.68 of an adult's size) stands 2.6 m
    # away in the walking path; auto setup's desk detection rebaselines the gate at 4.5 s; the depth turns off at 5.4 s
    # and the box, which assumes an adult, reads 3.9 m: beyond bg_path_m, so the new warm-up held the scheduler's
    # first cue at 6.0 s and the child, who walked at the user from 7.7 s, was first named 1.1 s before contact.
    # Measured depth seen in the warm-up is trusted over the box; a box range is taken at its nearest (BOX_SMALL: an
    # object half the typical size); and a rebaseline keeps the say-once cue the old warm-up found owed
    if case == "depth then box":     # (in view since 3.5 s: after the first warm-up; measured until 5.4 s)
        appear, offer, conf = 3.5, 6.0, lambda t: 0.9 if t < 5.4 else 0.25
        dist = lambda t: 2.6 if t < 5.4 else 3.9
    elif case == "owed":             # (the first warm-up measured it at 2.6 m; a box of a tiny dog reads 6.5 m from 4 s)
        appear, offer, conf = 0.0, 6.0, lambda t: 0.9 if t < 4.0 else 0.25
        dist = lambda t: 2.6 if t < 4.0 else 6.5
    else:                            # (no depth at all; the scheduler names it late, inside the first warm-up)
        appear, offer, conf = 0.0, 2.0, lambda t: 0.25
        dist = lambda t: 3.9
    def scene(t):
        return [dict(tid=1, bbox=_cam_box(0.0, 3.9), dist=dist(t), depth_conf=conf(t), path=0.6)] if t >= appear else []
    gate = AnnouncementGate(OfferTrack({1: [offer]}), GateConfig())
    cues = on_channel(gate, scene, offer + 0.5, events={4.5: gate.rebaseline} if offer > 4.5 else None)
    assert [(c.track_id, c.t) for c in cues] == [(1, offer)], dict(gate.reasons)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_still_desk_user_is_not_announced_when_the_depth_comes_back(seed):
    # Major (gate_verify_r4, v4_desk_still:900231, P-CACS): depth off for 8 s (run_live's 'd' key twice); the user's
    # cut-off box read 1.8 m, the depth 0.7 m when it came back, and the tracker's range rate across the step read as
    # closing at 0.6 m/s with TTC 0.1 s: the operator was released and "Person, right in front of you." said as critical.
    # A range rate that spans a change of range source is no measured rate (RATE_SETTLE_S)
    for make in (lambda: NearestPolicy(), lambda: PCACSPolicy(PipelineConfig().pcacs)):
        cues, _, gate = desk_session(make(), GateConfig(operator_filter=True), z0=0.65, noise=0.05, fidget=0.0, lean_m=0.0,
                                     T=40.0, seed=seed, depth_off=(12.4, 20.2))
        assert cues == [], (gate.last_reason, gate.releases)
        assert gate._operator is not None and not gate._operator.released
    # a real approach right after the depth came back is still released: the box and, once settled, the depth show it
    cues, _, gate = desk_session(NearestPolicy(), GateConfig(operator_filter=True), z0=2.0, noise=0.03, fidget=0.0,
                                 lean_m=0.0, T=26.0, seed=seed, depth_off=(12.4, 20.2), walk_at=21.0, walk_mps=0.7)
    assert gate._operator is None and [c for c in cues if c.t >= 21.0] and min(c.t for c in cues if c.t >= 21.0) <= 22.5


def test_a_message_cut_by_the_gates_own_interrupt_is_taken_back():
    # gate_verify_r4 (critical issue 3, the gate's own form): a message the gate let through is cut by the next one (a
    # critical cue about another object) after 0.5 s, before its distance phrase; the gate remembered it as said, so the
    # scheduler's next cue about that object was held as a repeat of words the user never heard. The gate notices its own
    # interrupt cutting the message on the real channel and takes the message back (heard_enough); a message heard
    # through its distance phrase stays said
    def scene(t):
        a = dict(tid=1, bbox=_cam_box(1.2, 3.0), dist=3.0, path=0.2)
        b = [dict(tid=2, bbox=_cam_box(0.0, 2.0), dist=2.0, rel=-1.5, ttc=0.9, path=0.9)] if t >= 1.5 else []
        return [a] + b
    for cut_at, again in ((1.5, True), (3.0, False)):                # (3.0 s: after A's message ended at 2.4 s)
        gate = AnnouncementGate(OfferTrack({1: [1.0, 5.0], 2: [cut_at]}), GateConfig(background=False))
        cues = on_channel(gate, lambda t: [x for x in scene(t) if x["tid"] == 1 or t >= cut_at], 5.0,
                          speech=lambda c: 1.4 if c.track_id == 1 else 1.2)
        said = [(c.track_id, c.t) for c in cues]
        assert (1, 5.0) in said if again else (1, 5.0) not in said, (cut_at, said, dict(gate.reasons))
        assert gate.reasons["cut (taken back)"] == int(again)
