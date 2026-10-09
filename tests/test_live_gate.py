"""AnnouncementGate (seeforme/live/gate.py): the live demo's memory of what was already said.

A worst-case fake scheduler offers a cue for the nearest track at every decision step (2 per second); the gate
must turn that into one announcement per object, survive tracker id switches, and still let approaching,
moving-to-another-side and critical objects through. No camera, no models."""
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
    # walking into it (1 m/s): every decision the TTC drops 0.5 s, then (TTC 0) the distance 0.5 m: each one passes;
    # once it stops getting worse it is held for critical_gap_s
    z = lambda t: max(0.3, 3.0 - t)
    gate = AnnouncementGate(NearestPolicy())
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=z(t), rel=-1.0, ttc=max(0.0, z(t) - 0.6), path=0.9)], 3.5)
    assert [c.t for c in cues] == [0.5, 1.0, 1.5, 2.0, 2.5] and gate.reasons["critical_gap"] == 2
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
    inner = NearestPolicy(recent_block_s=2.8)
    gate = AnnouncementGate(inner, GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=2.0, hits=1)] if t == 0.0 else [], 3.0)
    assert cues == [] and gate.stats["suppressed_unconfirmed"] == 1 and gate.reasons["unconfirmed (dropped)"] == 1
    assert inner.budget == pytest.approx(inner.cfg.budget_capacity_s)     # the unspoken cue's seconds were refunded
    # a real object is said one step later even though the scheduler (recent block) does not offer it again
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8), GateConfig(background=False))
    cues = run(gate, lambda t: [dict(tid=1, bbox=AHEAD, dist=2.0, **({"hits": 1} if t == 0.0 else {}))], 3.0)
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
    assert [c.t for c in cues] == [5.5, 13.5]                   # FIND goal: reminded after hold_goal_s, not 30 s


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
                 depth=True):
    """The user at a desk webcam through the real IoUTracker + TrackDynamics (10 fps, decisions every 0.5 s): a
    fidget of `fidget` (normalised box units), depth jitter sigma `noise`, and a lean of 0.25 m toward the screen at
    0.25 m/s every 20 s (held 2 s); from `walk_at` the user walks toward the camera at `walk_mps`. Returns the cues
    the gate let through and how many decisions saw the user's track as critical by measured TTC. `bare`: the
    scheduler's own cues, without the gate (the gate object is still returned, unused); `depth` False: no depth model."""
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
        lean = 0.25 * (min(1.0, ph - 5.0) if 5.0 <= ph < 8.0 else max(0.0, 9.0 - ph) if 8.0 <= ph < 9.0 else 0.0)
        z = z0 - lean + rng.gauss(0.0, noise)
        if walk_at is not None and t >= walk_at:
            z = max(0.3, z0 - walk_mps * (t - walk_at)) + rng.gauss(0.0, noise)
        j, grow = fidget * math.sin(0.7 * t), 0.16 * max(0.0, z0 - z)      # the box grows as the user comes closer
        bbox = (0.28 + j - grow, max(0.0, 0.12 - grow), 0.72 + j + grow, 0.98)
        tracks = tracker.update([Detection("person", 0.9, bbox, distance_m=z if depth else None,
                                           depth_conf=0.85 if depth else 0.0)], t)
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
    assert ts == [0.5, 3.0, 5.5, 8.0] and gate.reasons["interrupt"] == 3   # every min_gap_s (it moves)


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
    assert [c.t for c in cues] == [0.5, 10.0, 12.5] and gate.reasons["near"] >= 1
    assert gate.reasons["critical"] == 0 and all(c.distance_m > 0.6 for c in cues)


def test_in_path_object_present_at_startup_is_said_once_with_background_on():
    def scene(t):
        return [dict(tid=1, label="chair", bbox=AHEAD, dist=2.0),                     # in the path, 2 m
                dict(tid=2, label="chair", bbox=(0.80, 0.50, 0.95, 0.80), dist=4.0)]  # off to the right, 4 m
    gate = AnnouncementGate(NearestPolicy(recent_block_s=2.8))
    cues = run(gate, scene, 40.0)
    assert [(c.track_id, c.t) for c in cues] == [(1, 0.5)]      # at once (not after the warm-up), and never repeated
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
    assert ("text", "room 206 lab") not in gate._mem and [c.t for c in run(gate, sign, 6.0, t0=3.0)][:1] == [3.5]


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
