"""The cane (seeforme/live/cane.py) on synthetic metric depth maps: a pinhole camera at chest height (1.2-1.6 m),
pitched 0-50 degrees down, looking at a floor with boxes on it, with 2% depth noise. No model files, no camera.
Controller integration (said once, held, closer, still, merged with the scheduler) is at the end."""
import json
import math

import numpy as np
import pytest

from seeforme.live.cane import CANE_POLICY, CANE_TRACK_ID, Cane, analyze
from seeforme.live.controller import LiveController
from seeforme.live.summary import SessionLog
from seeforme.types import Cue, Track

HEIGHTS = (1.2, 1.4, 1.6)


def render(boxes=(), height=1.4, pitch=15.0, roll=0.0, w=160, h=120, hfov=70.0, noise=0.02, seed=0, scale=1.0):
    """Planar depth (m) seen by a pinhole camera `height` m above a floor, pitched down by `pitch` degrees. Boxes are
    (x0, x1, y0, y1, z0, z1) in metres: x right, y up from the floor, z forward. `scale` multiplies the whole map (a
    depth model whose metric scale is off)."""
    f = 0.5 * max(w, h) / math.tan(math.radians(hfov) / 2)
    xc, yc = np.meshgrid((np.arange(w) + 0.5 - w / 2) / f, (np.arange(h) + 0.5 - h / 2) / f)
    th, ro = math.radians(pitch), math.radians(roll)
    xr, yr = xc * math.cos(ro) - yc * math.sin(ro), xc * math.sin(ro) + yc * math.cos(ro)
    dx, dy, dz = xr, -(yr * math.cos(th) + math.sin(th)), math.cos(th) - yr * math.sin(th)   # world ray per unit depth
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(dy < -1e-6, height / -dy, np.inf)
        for (x0, x1, y0, y1, z0, z1) in boxes:
            lo, hi = np.full(xc.shape, -np.inf), np.full(xc.shape, np.inf)
            for o, d, a, b in ((0.0, dx, x0, x1), (height, dy, y0, y1), (0.0, dz, z0, z1)):
                ta, tb = (a - o) / d, (b - o) / d
                par = np.abs(d) < 1e-9
                inside = a <= o <= b
                lo = np.maximum(lo, np.where(par, -np.inf if inside else np.inf, np.minimum(ta, tb)))
                hi = np.minimum(hi, np.where(par, np.inf if inside else -np.inf, np.maximum(ta, tb)))
            T = np.where((hi >= lo) & (lo > 0), np.minimum(T, lo), T)
    T = np.where(np.isfinite(T), T, 30.0) * scale
    return (T * (1 + noise * np.random.default_rng(seed).standard_normal(T.shape))).astype(np.float32)


def wall(z=1.0):
    return (-3.0, 3.0, 0.0, 3.0, z, z + 0.2)


def ahead(fr):
    return fr.obstacles["ahead"]


# ---- one depth map ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("pitch", (0, 15, 35, 50))
def test_flat_floor_is_no_obstacle(height, pitch):
    fr = analyze(render(height=height, pitch=pitch, seed=int(height * 10) + pitch))
    assert fr.floor.source == "fit" and abs(fr.floor.height_m - height) < 0.05 and abs(fr.floor.pitch_deg - pitch) < 1.5
    assert all(o is None for o in fr.obstacles.values())
    assert fr.clear_m == {"left": 4.0, "ahead": 4.0, "right": 4.0} and fr.suggest() is None


def test_floor_with_roll_and_a_depth_scale_error():
    fr = analyze(render(height=1.4, pitch=20, roll=8))
    assert fr.floor.source == "fit" and abs(fr.floor.roll_deg) > 4 and all(o is None for o in fr.obstacles.values())
    # an outdoor model 4x too far up close: the camera cannot be 5.6 m up, so the map is rescaled to 1.4 m
    fr = analyze(render([wall(1.5)], height=1.4, pitch=30, scale=4.0))
    assert fr.floor.scale == pytest.approx(0.25, rel=0.05) and ahead(fr).distance_m == pytest.approx(1.5, abs=0.15)


@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("pitch", (0, 15, 35, 50))
def test_wall_one_metre_ahead(height, pitch):
    fr = analyze(render([wall(1.0)], height=height, pitch=pitch, seed=3))
    o = ahead(fr)
    assert o is not None and o.distance_m == pytest.approx(1.0, abs=0.12) and o.level == "body"
    assert fr.clear_m["ahead"] == pytest.approx(o.distance_m) and fr.suggest() is None   # blocked everywhere: no side


@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("pitch", (15, 35, 50))
def test_box_one_metre_ahead(height, pitch):
    """A 0.6 m box in the corridor (it is below the view of a level camera that close, as it would be in reality)."""
    fr = analyze(render([(-0.3, 0.3, 0.0, 0.6, 1.0, 1.4)], height=height, pitch=pitch, seed=5))
    o = ahead(fr)
    assert o is not None and o.distance_m == pytest.approx(1.0, abs=0.12) and o.level in ("body", "ground")
    assert fr.obstacles["left"] is None and fr.obstacles["right"] is None


@pytest.mark.parametrize("height", HEIGHTS)
def test_low_box_is_ground_level(height):
    fr = analyze(render([(-0.4, 0.4, 0.0, 0.3, 1.6, 2.0)], height=height, pitch=30, seed=7))
    assert ahead(fr) is not None and ahead(fr).level == "ground" and ahead(fr).distance_m == pytest.approx(1.6, abs=0.15)


@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("pitch", (15, 35))
def test_obstacle_on_the_left_suggests_right(height, pitch):
    # the left half of the corridor and the left side are blocked 1 m ahead; the right is open
    fr = analyze(render([(-1.5, -0.15, 0.0, 1.2, 1.0, 1.4)], height=height, pitch=pitch, seed=11))
    assert fr.obstacles["left"] is not None and ahead(fr) is not None and fr.obstacles["right"] is None
    assert fr.suggest() == "right" and fr.clear_m["right"] == 4.0
    # entirely beside the corridor: an obstacle on the left, nothing ahead, and no suggestion (never "go ahead")
    fr = analyze(render([(-1.5, -0.5, 0.0, 1.2, 1.0, 1.4)], height=height, pitch=pitch, seed=12))
    assert fr.obstacles["left"] is not None and ahead(fr) is None and fr.suggest() is None


@pytest.mark.parametrize("height", HEIGHTS)
@pytest.mark.parametrize("pitch", (0, 8))
def test_overhead_bar_over_a_clear_floor_is_head_level(height, pitch):
    fr = analyze(render([(-0.8, 0.8, 1.6, 1.75, 1.4, 1.5)], height=height, pitch=pitch, seed=13))
    o = ahead(fr)
    assert o is not None and o.level == "head" and o.distance_m == pytest.approx(1.4, abs=0.15)
    assert fr.suggest() is None                       # path choice is for what blocks the floor


def test_door_lintel_is_clearance_and_stop_when_close():
    fr = analyze(render([(-1.0, -0.45, 0, 2.1, 1.5, 1.6), (0.45, 1.0, 0, 2.1, 1.5, 1.6), (-1.0, 1.0, 2.05, 2.3, 1.5, 1.6)],
                        height=1.4, pitch=10, seed=17))
    assert ahead(fr) is None                          # walking through a doorway: jambs beside, lintel above 1.95 m
    fr = analyze(render([wall(0.5)], height=1.4, pitch=10))
    assert ahead(fr).distance_m < 0.6 and fr.suggest() == "stop"


def test_contract_shape():
    c = analyze(render([(-1.5, -0.15, 0.0, 1.2, 1.0, 1.4)], height=1.4, pitch=20)).as_contract()
    json.dumps(c)
    assert set(c) == {"clear_m", "obstacle", "suggest"} and set(c["clear_m"]) == {"left", "ahead", "right"}
    o = c["obstacle"]
    assert o["sector"] == "ahead" and o["level"] in ("ground", "body", "head") and 0 <= o["confidence"] <= 1
    assert len(o["bbox"]) == 4 and all(0 <= v <= 1 for v in o["bbox"]) and o["bbox"][0] < 0.5  # left part of the image
    assert analyze(np.full((60, 80), np.nan, np.float32)).floor is None


def test_server_returns_the_same_summary():
    pytest.importorskip("fastapi")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("seeforme_server", Path(__file__).resolve().parents[1] / "server" / "app.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    d = render([wall(1.2)], height=1.4, pitch=20)
    assert srv.cane_summary(d) == analyze(d).as_contract()
    assert srv.cane_summary(None) is None and srv.cane_summary(np.full((60, 80), np.nan, np.float32)) is None


# ---- over time ---------------------------------------------------------------------------------------------------
def test_needs_three_maps_over_half_a_second():
    cane = Cane()
    for i in range(3):                                # three maps in 0.2 s: not yet
        cane.update(render([wall(1.5)], seed=i), 0.1 * i)
    assert cane.propose(0.2) is None
    for i in range(3, 6):
        cane.update(render([wall(1.5)], seed=i), 0.1 * i)
    c = cane.propose(0.5)
    assert c is not None and c.message.startswith("Obstacle ahead") and c.track_id == CANE_TRACK_ID and c.policy == CANE_POLICY
    same = cane.frame
    assert cane.update(cane._last_map_ref, 0.6) is same    # the pipeline's map passed again between depth runs: ignored


def test_reuses_the_floor_when_a_wall_hides_it():
    cane = Cane()
    cane.update(render(height=1.4, pitch=5), 0.0)            # floor in view
    fr = cane.update(render([wall(0.9)], height=1.4, pitch=5), 0.2)
    assert fr.floor.source == "reused" and ahead(fr).distance_m == pytest.approx(0.9, abs=0.1)


# ---- controller --------------------------------------------------------------------------------------------------
class FakeScheduler:
    """Offers a cue about the nearest track at every decision (or nothing without tracks)."""
    name = "fake"
    decision_interval = 0.5
    supports_actions = True

    class _Cfg:
        verbosity = "balanced"
        units = "m"

    def __init__(self, severity=0.5):
        self.cfg = self._Cfg()
        self.severity = severity

    def reset(self):
        pass

    def last_observations(self):
        return []

    def set_goal(self, tags):
        pass

    def decide(self, tracks, t, ch=None):
        if not tracks:
            return []
        trk = min(tracks, key=lambda x: x.distance_m or 99)
        return [Cue(t, trk.track_id, trk.label, trk.kind, f"{trk.label.capitalize()}.", 1.0, self.name, distance_m=trk.distance_m,
                    severity=self.severity, ttc_s=trk.ttc_s if math.isfinite(trk.ttc_s) else None)]


def track(tid, label, bbox, dist, t, ttc=math.inf, path=0.0):
    tr = Track(tid, label, "object", bbox, 0.9, 0.0, t, confirmed=True)
    tr.hits, tr.distance_m, tr.depth_conf, tr.ttc_s, tr.path_prob = 30, dist, 0.9, ttc, path
    return tr


def drive(ctl, seconds, dist_fn, fps=10, depth_every=2, tracks_fn=None, frame_fn=None, boxes_fn=None, pitch=15, height=1.4):
    """run_live order: process (the policy decides every 0.5 s) then on_frame with the pipeline's latest depth map.
    dist_fn(t): distance to a wall ahead (None: no wall). Returns the cues the pipeline would act on."""
    out, next_dec, depth = [], 0.0, None
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (120, 160, 3)).astype(np.float64)
    for i in range(int(seconds * fps)):
        t = i / fps
        tracks = tracks_fn(t) if tracks_fn else []
        if t + 1e-9 >= next_dec:
            next_dec = t + 0.5
            cues = ctl.policy.decide(tracks, t, None)
            assert len(cues) <= 1                        # never two cues in one decision
            out += cues
        if i % depth_every == 0:
            d = None if boxes_fn else dist_fn(t)
            boxes = boxes_fn(t) if boxes_fn else ([wall(d)] if d is not None else [])
            depth = render(boxes, height=height, pitch=pitch, seed=i)
        frame = frame_fn(t, rng, base) if frame_fn else None
        ctl.on_frame(frame, t, tracks, depth=depth)
    return out


def cane_cues(cues):
    return [c for c in cues if c.policy == CANE_POLICY]


def test_said_once_held_and_said_again_when_closer():
    # the user walks up to 1.9 m from a wall, stands there (the camera still moves: no frames, activity unknown)
    # for 6 s, then walks on to 1.3 m
    def dist(t):
        return 3.0 - min(t, 1.1) if t < 8 else max(1.3, 1.9 - (t - 8))
    ctl = LiveController(FakeScheduler(), setup="handheld")
    said = cane_cues(drive(ctl, 12, dist))
    assert len(said) == 2, [(c.t, c.message) for c in said]
    assert said[0].t < 3.0 and said[0].message.startswith("Obstacle ahead")
    assert said[1].t >= 8.0 and said[1].distance_m < said[0].distance_m - 0.35


def test_walking_into_a_wall_warns_then_stops():
    ctl = LiveController(FakeScheduler(), setup="handheld")
    said = cane_cues(drive(ctl, 4.2, lambda t: max(0.3, 3.2 - 0.8 * t)))
    msgs = [(round(c.t, 1), c.action, c.message) for c in said]
    assert said[0].action == "speak" and said[-1].action == "haptic" and said[-1].message == "Stop.", msgs
    assert sum(c.action == "haptic" for c in said) == 1 and len(said) <= 4, msgs


def test_reminder_after_ten_seconds_of_walking():
    ctl = LiveController(FakeScheduler(), setup="handheld")
    said = cane_cues(drive(ctl, 14, lambda t: 1.8))
    assert len(said) == 2 and 10.0 <= said[1].t - said[0].t <= 11.0


def still_frame(t, rng, base):
    return np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)


def test_silent_while_stationary_unless_approaching():
    # standing still with a wall 1.5 m ahead from the start: nothing (static surroundings)
    ctl = LiveController(FakeScheduler(), setup="handheld")
    quiet = cane_cues(drive(ctl, 10, lambda t: 1.5 if t >= 3 else None, frame_fn=still_frame))
    assert ctl.activity.activity.value == "stationary" and quiet == []
    # something comes at the still user (a door swinging, a cart pushed): said
    ctl = LiveController(FakeScheduler(), setup="handheld")
    said = cane_cues(drive(ctl, 10, lambda t: None if t < 3 else (2.0 if t < 6 else max(1.0, 2.0 - 0.6 * (t - 6))),
                           frame_fn=still_frame))
    assert len(said) >= 1 and all(c.t >= 6.0 for c in said)


def test_desk_webcam_is_silent():
    """A person and a desk filling the view at 0.5 m in front of a laptop camera."""
    def boxes(t):
        return [(-0.8, 0.8, 0.0, 0.75, 0.3, 1.0), (-0.3, 0.3, 0.0, 1.5, 0.5, 0.8)]
    ana = analyze(render(boxes(0), height=1.0, pitch=0))
    assert ahead(ana) is not None and ahead(ana).distance_m < 0.7      # the cane alone sees it ...
    for setup, frames in (("desk", None), ("auto", still_frame)):        # ... the controller keeps it quiet
        ctl = LiveController(FakeScheduler(), setup=setup)
        cues = drive(ctl, 15, None, boxes_fn=boxes, frame_fn=frames, height=1.0, pitch=0)
        assert cane_cues(cues) == [] and ctl.desk and ctl.cane_state() == "off (desk)"


def test_off_without_depth_off_by_flag_and_muted_in_read_mode():
    ctl = LiveController(FakeScheduler(), setup="handheld")
    out, nd = [], 0.0
    for i in range(50):                                # no depth map at all
        t = i / 10
        if t + 1e-9 >= nd:
            nd = t + 0.5
            out += ctl.policy.decide([], t, None)
        ctl.on_frame(None, t, [])
    assert out == [] and ctl.cane_state() == "no depth"
    assert cane_cues(drive(LiveController(FakeScheduler(), setup="handheld", cane=False), 6, lambda t: 1.5)) == []
    read = cane_cues(drive(LiveController(FakeScheduler(), mode="read", setup="handheld"), 5, lambda t: max(0.3, 2.5 - 0.8 * t)))
    assert read and all(c.action == "haptic" for c in read)     # read mode: only the stop gets through


def test_an_obstacle_a_track_explains_is_left_to_the_scheduler():
    box = (-0.3, 0.3, 0.0, 1.7, 1.5, 1.8)               # a person-sized box 1.5 m ahead ...
    fr = analyze(render([box], height=1.4, pitch=15))
    person = lambda t: [track(1, "person", ahead(fr).bbox, 1.55, t)]   # ... the detector sees as a person
    ctl = LiveController(FakeScheduler(), setup="handheld")
    cues = drive(ctl, 6, None, boxes_fn=lambda t: [box], tracks_fn=person)
    assert cues and cane_cues(cues) == [] and ctl.cane.stats["explained"] > 0


def test_merged_with_a_scheduler_cue_without_double_speech(tmp_path):
    chair_box = (0.5, 0.9, 0.0, 0.9, 2.5, 2.9)
    log = SessionLog(str(tmp_path / "s.jsonl"))
    ctl = LiveController(FakeScheduler(severity=0.5), setup="handheld", log=log)
    chair = lambda t: [track(1, "chair", (0.75, 0.4, 0.95, 0.7), 2.7, t)] if t >= 1.0 else []   # same decision as the cane
    cues = drive(ctl, 6, None, boxes_fn=lambda t: [wall(1.6), chair_box], tracks_fn=chair)
    log.close()
    first_cane = next(i for i, c in enumerate(cues) if c.policy == CANE_POLICY)
    # the cane (an obstacle 1.6 m ahead, more urgent than a static chair) won its decision; the chair cue it displaced
    # (already recorded as said by the scheduler) came right after
    assert cues[first_cane + 1].label == "chair" and cues[first_cane + 1].t - cues[first_cane].t <= 0.51
    assert ctl.cane_stats["won"] >= 1 and ctl.cane_stats["displaced_said"] >= 1
    logged = [json.loads(line) for line in open(tmp_path / "s.jsonl")]
    assert any(r["kind"] == "cue" and r["policy"] == CANE_POLICY and r["label"] == "obstacle" for r in logged)


@pytest.mark.parametrize("label,ttc,path,goal", [("car", None, 0.0, None), ("person", 2.5, 0.0, None),
                                                  ("person", None, 0.9, None), ("door", None, 0.0, "door")])
def test_never_delays_a_safety_cue(label, ttc, path, goal):
    """A vehicle, an approaching or in-path object, a FIND goal: said exactly when the bare scheduler says it."""
    def objs(t):     # appears at the decision where the cane (a wall from 1.5 s, confirmed after 0.5 s) wants to speak
        return [track(5, label, (0.6, 0.3, 0.8, 0.8), 3.0, t, ttc=ttc if ttc is not None else math.inf, path=path)] if t >= 2.5 else []

    def first(cane):
        ctl = LiveController(FakeScheduler(severity=0.5), setup="handheld", gate=False, cane=cane,
                             goal_tags=[goal] if goal else None)
        cues = drive(ctl, 5, lambda t: 1.8 if t >= 1.5 else None, tracks_fn=objs)
        return next(c.t for c in cues if c.label == label), ctl.cane_stats["lost"]
    (with_cane, lost), (bare, _) = first(True), first(False)
    assert with_cane == bare == 2.5 and lost >= 1


def test_stop_delays_a_safety_cue_by_one_decision_at_most():
    walk = lambda t: max(0.3, 2.0 - 0.7 * t)
    t_stop = next(c.t for c in drive(LiveController(FakeScheduler(), setup="handheld", gate=False), 4, walk) if c.action == "haptic")
    ctl = LiveController(FakeScheduler(severity=0.5), setup="handheld", gate=False)
    car = lambda t: [track(9, "car", (0.6, 0.3, 0.9, 0.6), 6.0, t)] if t >= t_stop else []   # same decision as the stop
    cues = drive(ctl, 4, walk, tracks_fn=car)
    stop = next(c for c in cues if c.action == "haptic")
    car_t = next(c.t for c in cues if c.label == "car")
    assert stop.t == t_stop and ctl.cane_stats["won"] == 1
    assert car_t - t_stop == pytest.approx(0.5), [(c.t, c.label, c.action) for c in cues]   # the car: one decision later


def test_no_stop_for_what_the_user_did_not_walk_up_to():
    """A seat back 0.5 m ahead from the start (riding in a car), the user's own reach: no stop, no cue."""
    ctl = LiveController(FakeScheduler(), setup="handheld")
    assert cane_cues(drive(ctl, 12, lambda t: 0.5)) == [] and ctl.cane.stats["close_up"] > 0


def test_one_stop_per_obstacle():
    # walks into a wall and stays in front of it for 10 s: one stop; walks away and back to another: a second one
    def dist(t):
        if t < 3.5:
            return max(0.45, 2.6 - 0.8 * t)
        if t < 13:
            return 0.45
        if t < 16:
            return 3.0
        return max(0.45, 3.0 - 0.8 * (t - 16))
    cues = cane_cues(drive(LiveController(FakeScheduler(), setup="handheld"), 20, dist))
    stops = [c.t for c in cues if c.action == "haptic"]
    assert len(stops) == 2 and stops[0] < 3.5 and stops[1] > 16, [(c.t, c.message) for c in cues]


def test_close_up_over_a_table_is_silent():
    """The camera 0.7 m above a table top (seated, or leaning over an exhibit) with a box on it: not a walking view."""
    ctl = LiveController(FakeScheduler(), setup="handheld")
    cues = drive(ctl, 6, None, boxes_fn=lambda t: [(-0.2, 0.2, 0.0, 0.3, 0.9, 1.1)], height=0.7, pitch=40)
    assert cane_cues(cues) == [] and ctl.cane.frame.floor.height_m < 0.9 and not ctl.cane.walking_view


# ---- review fixes ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("height", (1.0, 1.4, 1.8))
@pytest.mark.parametrize("pitch", (55, 60, 62, 70, 80))
def test_a_floor_seen_steeply_is_never_a_wall(height, pitch):
    """Reviewer repro: a user looking down at the phone (pitch over 60 degrees): the floor fit was refused and the wall
    fallback read the empty floor as a wall 1.4 m ahead ("Obstacle ahead, about 1.5 meters.")."""
    fr = analyze(render(height=height, pitch=pitch, seed=pitch))
    assert fr.floor.source == "fit" and abs(fr.floor.pitch_deg - pitch) < 2.0
    assert all(o is None for o in fr.obstacles.values()) and fr.suggest() is None
    from seeforme.live.cane import CaneConfig, _grid, fit_wall
    c = CaneConfig()
    z, x, y, *_ = _grid(render(height=height, pitch=pitch, seed=pitch), c)
    assert fit_wall(z, x, y, c) is None                     # not a wall either, whatever falls back to it


@pytest.mark.parametrize("pitch", (60, 70))
@pytest.mark.parametrize("roll", (0, 10))
def test_no_cue_walking_over_a_flat_floor_looking_down(pitch, roll):
    cane = Cane()
    for i in range(20):
        t = 0.2 * i
        cane.update(render(height=1.4, pitch=pitch, roll=roll, seed=i), t)
        if i % 2:
            assert cane.propose(t, moving=True) is None


def test_stop_needs_a_steady_approach():
    """A noisy floor patch whose range jumps about (it was "seen farther" a moment ago) is not a walker closing on
    something: no stop. The same ranges in a steady approach: a stop."""
    from seeforme.live.cane import CaneFrame, Floor, Obstacle

    def feed(cane, ds):
        for i, d in enumerate(ds):
            t = 0.2 * i
            o = Obstacle("ahead", d, 0.0, "body", 0.8, (0.3, 0.5, 0.7, 1.0), 0.06, None, None)
            cane.sectors["ahead"].observe(t, o, cane.cfg)
            cane.frame, cane._t = CaneFrame({}, {"ahead": o}, Floor((0.0, 0.97, 0.26), 1.4, 1.0, "fit")), t
        return cane.propose(t, moving=True)
    jumpy = feed(Cane(), [0.9, 0.4, 0.85, 0.35, 0.9, 0.4, 0.8, 0.45])
    assert jumpy is None or jumpy.action != "haptic"
    steady = feed(Cane(), [1.6, 1.44, 1.28, 1.12, 0.96, 0.8, 0.64])
    assert steady is not None and steady.action == "haptic" and steady.message == "Stop."


@pytest.mark.parametrize("side", ("left", "right"))
def test_path_choice_reports_what_blocks_never_what_is_clear(side):
    """Reviewer: "Obstacle ahead, clearer on your left" implies a side path is clear (drop-offs are not seen)."""
    from seeforme.live.commands import says_clear
    x0, x1 = (-1.5, -0.15) if side == "left" else (0.15, 1.5)
    boxes = lambda t: [(x0, x1, 0.0, 1.2, max(0.9, 2.6 - 0.6 * t), max(0.9, 2.6 - 0.6 * t) + 0.4)]   # noqa: E731
    ctl = LiveController(FakeScheduler(), setup="handheld")
    said = cane_cues(drive(ctl, 5, None, boxes_fn=boxes))
    path = [c for c in said if getattr(c, "cane_path", None)]
    assert path and path[0].message.startswith("Obstacle ahead") and path[0].message.endswith(f"and on your {side}.")
    assert all(not says_clear(c.message) for c in said) and ctl.cane.stats["path"] >= 1


def test_note_puts_the_confirmed_obstacle_ahead_into_words():
    cane = Cane()
    assert cane.note(0.0) == ""
    for i in range(6):
        cane.update(render([wall(1.2)], seed=i), 0.1 * i)
    assert cane.note(0.5) == "The depth map shows an obstacle ahead, about 1 meter."
    assert cane.note(0.5, "ft").endswith("about 4 feet.") and cane.note(5.0) == ""          # stale: nothing
    head = Cane()
    for i in range(6):
        head.update(render([(-0.8, 0.8, 1.6, 1.75, 1.4, 1.5)], height=1.4, pitch=0, seed=i), 0.1 * i)
    assert head.note(0.5).startswith("The depth map shows something at head height ahead")


def test_server_suggests_stop_only_for_a_walking_client():
    """Reviewer repro: a seated user at a laptop (camera 1.0 m up, the desk 0.3 m away) got suggest "stop" from
    /api/depth. One map cannot tell; a client that knows the user walks says so (walking=true)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import importlib.util
    import warnings
    from pathlib import Path
    import cv2
    spec = importlib.util.spec_from_file_location("seeforme_server2", Path(__file__).resolve().parents[1] / "server" / "app.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

    class Stub:
        device = "cpu"

        def __init__(self, depth):
            self.depth = depth

        def __call__(self, img):
            return self.depth
    client = TestClient(srv.app)
    png = cv2.imencode(".png", np.zeros((120, 160, 3), np.uint8))[1].tobytes()

    def post(depth, **form):
        srv.STATE["depth"]["indoor"] = Stub(depth)
        r = client.post("/api/depth", files={"image": ("a.png", png, "image/png")}, data={"boxes": "[[0.4, 0.4, 0.6, 0.8]]", **form})
        assert r.status_code == 200
        return r.json()["cane"]
    desk = render([(-0.8, 0.8, 0.0, 0.75, 0.3, 1.0)], height=1.0, pitch=0)
    c = post(desk)
    assert c["obstacle"]["sector"] == "ahead" and c["obstacle"]["distance_m"] < 0.6 and c["suggest"] is None
    assert post(desk, walking="true")["suggest"] == "stop"
    assert post(render([(-1.5, -0.15, 0.0, 1.2, 1.0, 1.4)], height=1.4, pitch=20))["suggest"] == "right"   # path choice stays
    assert post(np.full((60, 80), np.nan, np.float32)) is None
