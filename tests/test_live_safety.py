"""scripts/live_safety.py: the live safety harness, and a small fast subset of its sweep (so a regression in the gate or
the controller shows up in pytest, not only in `make live-safety`).

The subset replays fixed scenes of the harness's families through the real IoUTracker, TrackDynamics and P-CACS, in
the gated configurations, and asserts what the full sweep's acceptance asserts: no severe row (the gated system silent,
or later than bare + 0.5 s and under 1.5 s before contact, where the bare scheduler warned with news at least 1.5 s
before contact), no cue from a still or shallow-leaning desk user after the warm-up, and a stranger or visitor walking
up to a desk webcam warned no later than bare + 0.5 s."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ls():
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = sys.modules.get("live_safety")
    if mod is None:
        spec = importlib.util.spec_from_file_location("live_safety", ROOT / "scripts" / "live_safety.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["live_safety"] = mod          # (its dataclasses look their module up)
        spec.loader.exec_module(mod)
    return mod


def test_scenes_are_seeded_and_cover_every_family(ls):
    for fam in ls.FAMILIES:
        a, b = ls.build_scene(fam, 7), ls.build_scene(fam, 7)
        assert a.params == b.params and a.T == b.T and a.depth == b.depth, fam
        assert any(o.hazard for o in a.objs) or fam in ("desk", "reacq"), fam
        fa, fb = ls.make_frames(a), ls.make_frames(b)
        assert len(fa) == len(fb) and fa[len(fa) // 2][1] == fb[len(fb) // 2][1], fam
    # the families the brief names, with the hazards' contact computed from the trajectory
    sc = ls.build_scene("approach", 3)
    h = sc.objs[0]
    assert h.hazard and h.contact is not None and h.z(h.contact) <= ls.CONTACT_M + 1e-9


def test_severity_rules(ls):
    Said, Obj, Scene = ls.Said, ls.Obj, ls.Scene
    h = Obj("H", "person", lambda t: max(0.25, 10.0 - t), hazard=True, t_lo=0.0, contact=9.4)
    sc = Scene("approach", 0, 12.0, [h])
    cue = lambda t, owner="H": Said(t, owner, "speak", "object", 0.5, None, "Person.")
    assert ls.judge(sc, h, [cue(3.0)], [cue(3.5)])["severe"] is False                  # one decision step: fine
    assert ls.judge(sc, h, [cue(3.0)], [cue(4.0)])["late"] is True                     # later than that: late ...
    assert ls.judge(sc, h, [cue(3.0)], [cue(4.0)])["severe"] is False                  # ... but 5.4 s before contact
    assert ls.judge(sc, h, [cue(3.0)], [cue(8.5)])["severe"] is True                   # 0.9 s before contact
    assert ls.judge(sc, h, [cue(3.0)], [cue(3.0, "other")])["severe"] is True          # silent about H
    assert ls.judge(sc, h, [cue(8.5)], [])["severe"] is False                          # bare itself too late to count
    # no news: the gated system said H 2 s before at about this range (it was static then: an object said at 7.1 m
    # that is now at 7.0 m is nothing new; one that came 1.5 m nearer is)
    st = Obj("H", "person", lambda t: 7.0 if t < 5.0 else max(0.25, 7.0 - (t - 5.0)), hazard=True, t_lo=4.0, contact=11.4)
    sc2 = Scene("approach", 0, 12.0, [st])
    assert ls.judge(sc2, st, [cue(4.5)], [cue(2.5)])["tb"] is None
    assert ls.judge(sc2, st, [cue(6.5)], [cue(2.5)])["tb"] == 6.5
    # read mode: a bare cue the read-mode filter drops is not a warning the gated controller owes
    assert ls.judge(sc, h, [cue(3.0)], [], read_mode=True)["tb"] is None
    assert ls.judge(sc, h, [Said(3.0, "H", "speak", "object", 0.5, 4.0, "Person.")], [], read_mode=True)["severe"]
    # every news cue counts, not only the first (gate_verify_r2): named once early, then silent while H closes in from
    # 9 m to 1.5 m, is severe; a gated cue within a step of each bare one is not
    j = ls.judge(sc, h, [cue(1.0), cue(7.5)], [cue(1.0)])
    assert j["tb"] == 1.0 and j["tg"] == 1.0 and j["severe"] and (j["every_tb"], j["every_tg"]) == (7.5, None)
    assert not ls.judge(sc, h, [cue(1.0), cue(7.5)], [cue(1.0), cue(8.0)])["severe"]
    # news by label and place: another person where H is, said at about H's range a moment ago, is what the user heard
    # (two people side by side are one thing to them); a person 2 m to the side is not
    for xo, known in ((0.4, True), (2.0, False)):
        o = Obj("O", "person", lambda t: 9.0, lambda t, xo=xo: xo)
        sc3 = Scene("approach", 0, 12.0, [h, o])
        assert (ls.judge(sc3, h, [cue(1.5)], [cue(1.0, "O")])["tb"] is None) == known


def test_shared_tracker_gives_each_configuration_its_own_run(ls):
    # the sweep tracks a scene once and gives every configuration its own copies of the tracks: the cues must be the
    # ones each configuration gets with a tracker of its own
    for sid in ("cross_switch:6", "desk:10"):
        fam, seed = sid.split(":")
        sc = ls.build_scene(fam, int(seed))
        sc.T = min(sc.T, 40.0)
        frames = ls.make_frames(sc)
        configs = ["bare", "gate", "ctl:auto"]
        together = ls.run_configs(sc, frames, "pcacs", configs)
        for c in configs:
            alone = ls.run_config(sc, frames, "pcacs", c)
            assert [(s.t, s.owner, s.message) for s in alone] == [(s.t, s.owner, s.message) for s in together[c]], (sid, c)


# scenes the sweep found severe while the gate was being fixed (each with the scheduler and configurations it failed in),
# and the desk cases of the critical lean-memory regression: a stranger walking up after the user leaned and left, a
# visitor walking up in front of the user, a deep lean and a shallow one
SUBSET = [
    ("side_entry:194", "rl", ("ctl:handheld",)),          # a person who walked in, then a rebaseline: background
    ("lost_background:380", "rl", ("gate_nobg",)),         # the tracker kept one id for two people: a repeat
    ("find:1017", "pcacs", ("ctl:read",)),                 # the gate's own message held the goal chair back
    ("flicker:1096", "rl", ("gate",)),                     # an arrival right before a rebaseline held for the warm-up
    ("approach:1272", "rl", ("gate",)),                    # someone stepping into the path during the warm-up
    ("read_scene:1415", "rl", ("gate",)),                  # box jitter said a sign again and held a car back
    ("lost_background:1448", "rl", ("gate_nobg",)),        # box jitter "closing", then min_gap_s
    ("find:2133", "rl", ("gate",)),                        # the scheduler repeating one goal person, never the other
    ("desk:1534", "pcacs", ("gate_op", "ctl:desk")),       # stranger after a 0.42 m lean
    ("desk:82", "pcacs", ("gate_op", "ctl:desk")),         # visitor in front of the user
    ("desk:430", "pcacs", ("gate_op",)),                   # one deep lean (0.56 m): at most one cue
    ("desk:814", "pcacs", ("gate_op",)),                   # a shallow lean (0.16 m): no cue
]


# the second verification's (gate_verify_r2) severe scenes, each with the scheduler and a configuration it failed in
SUBSET_R2 = [
    ("static_path:70997", "pcacs", ("ctl:handheld",)),    # the gate's own cue stamped the bench: P-CACS kept quiet
    ("find:71473", "rl", ("gate",)),                      # ... the learned scheduler, the FIND-goal bicycle
    ("swap:91001", "rl", ("gate",)),                      # ... no cue at all about the approaching person
    ("find:71797", "rl", ("gate",)),                      # FIND swap dropped the goal person walking at the user
    ("bgref:90278", "rl", ("gate",)),                     # handover: another person stepped in, a 0.11 s gap
    ("bgref:90998", "rl", ("gate_nobg",)),                # ... its cue held as a repeat before the step showed
    ("flicker:72656", "rl", ("gate",)),                   # 2-frame sightings, no depth: held as a static repeat
    ("flicker:72116", "rl", ("gate_nobg",)),              # 0.85 m nearer, under the closer step, no closing speed
    ("rescue_switch:90676", "rl", ("gate",)),             # starts to approach as it drops out of view
    ("rescue_switch:90572", "rl", ("ctl:desk",)),         # ... a chair, auto setup taken for a desk
    ("read_scene:72879", "rl", ("gate",)),                # a slow approach (0.2 m/s), no closing speed
]


def test_safety_subset_r2_has_no_severe_rows(ls):
    rows = []
    for sid, pol, configs in SUBSET_R2:
        fam, seed = sid.split(":")
        rows += ls.run_scene(fam, int(seed), [pol], configs=configs)["rows"]
    severe = [r for r in rows if r["severe"] and r["blame"] == "gate"]
    assert not severe, [(r["sid"], r["config"], r["tb"], r["tg"], r["every_tb"], r["every_tg"], r["contact"]) for r in severe]
    assert sum(r["tb"] is not None for r in rows) >= 8


@pytest.mark.parametrize("sid,pol,config", [("static_path:70997", "pcacs", "ctl:handheld"), ("find:71473", "rl", "gate"),
                                            ("swap:91001", "rl", "gate"), ("handover:3", "rl", "gate_nobg")])
def test_the_scheduler_offers_under_the_gate_what_it_offers_alone(ls, sid, pol, config):
    # Critical (gate_verify_r2): the gate's own messages stamped the tracks as announced, its holds refunded the speech
    # budget and freed the channel, so the scheduler decided on another state than alone and did not give the warning
    # it gave alone. Every offer the scheduler makes under the gate (passed or held) is now the one it makes alone
    from seeforme.config import PipelineConfig
    from seeforme.geometry import CameraModel, TrackDynamics
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.types import Detection
    fam, seed = sid.split(":")
    sc = ls.build_scene(fam, int(seed))
    frames = ls.make_frames(sc)
    runs = [ls._Run(pol, "bare"), ls._Run(pol, config)]
    offers = {}
    for r in runs:
        log, inner = offers.setdefault(r.config, []), r.pol.decide

        def watch(tracks, t, ch=None, *a, _log=log, _inner=inner, **kw):
            out = _inner(tracks, t, ch, *a, **kw)
            _log.append((round(t, 3), [(c.track_id, c.action, c.message) for c in out]))
            return out
        r.pol.decide = watch
    tracker = IoUTracker(PipelineConfig().tracker, dynamics=TrackDynamics(CameraModel()))
    for t, dets, named in frames:
        tracks = tracker.update([Detection(lb, 0.9, b, kind=k, text=tx, distance_m=d, depth_conf=c)
                                 for lb, b, k, tx, d, c in dets], t)
        for r in runs:
            r.step(sc, t, tracks, {})
    gate = runs[1].gate
    assert gate is not None and sum(1 for _, o in offers["bare"] if o) >= 2
    assert offers[config] == offers["bare"]


def test_safety_subset_has_no_severe_rows(ls):
    rows, desk = [], []
    for sid, pol, configs in SUBSET:
        fam, seed = sid.split(":")
        res = ls.run_scene(fam, int(seed), [pol], configs=configs)
        rows += res["rows"]
        desk += res["desk"]
    severe = [r for r in rows if r["severe"] and r["blame"] == "gate"]
    assert not severe, [(r["sid"], r["config"], r["tb"], r["tg"], r["contact"]) for r in severe]
    acc = ls.desk_acceptance(desk)
    assert not acc["bad"], acc["bad"]
    late_desk = [r for r in rows if r["family"] == "desk" and r["config"] in ls.OPERATOR_CONFIGS and r["late"]
                 and r["blame"] == "gate"]
    assert not late_desk, [(r["sid"], r["config"], r["tb"], r["tg"]) for r in late_desk]
    assert sum(r["tb"] is not None for r in rows) >= 6          # (the subset does exercise warnings with news)
