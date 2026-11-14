"""scripts/live_safety.py: the live safety harness, and a small fast subset of its sweep (so a regression in the gate or
the controller shows up in pytest, not only in `make live-safety`).

The subset replays fixed scenes of the harness's families through the real IoUTracker, TrackDynamics and P-CACS, in
the gated configurations, and asserts what the full sweep's acceptance asserts: no severe row (the gated system silent,
or later than bare + 0.5 s and under 1.5 s before contact, where the bare scheduler warned with news at least 1.5 s
before contact), no cue from a still or shallow-leaning desk user after the warm-up, and a stranger or visitor walking
up to a desk webcam warned no later than bare + 0.5 s. It also checks the scheduler's view: in every configuration (the
gate, the controller in every setup and mode, the cane on, spoken replies on the channel) the scheduler is shown and
offers at every decision exactly what it is shown and offers alone (ls.diverge), on one scene of every family and on the
scenes where it once diverged."""
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
        assert any(o.hazard for o in a.objs) or fam in ("desk", "reacq", "desk_still"), fam
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
    # news by label and place: another person where H is, said with its range at about H's range a moment ago, is what
    # the user heard (two people side by side are one thing to them); a person 2 m to the side is not
    ranged = lambda t, owner: Said(t, owner, "speak", "object", 0.5, None, "Person, about 9 meters.")
    for xo, known in ((0.4, True), (2.0, False)):
        o = Obj("O", "person", lambda t: 9.0, lambda t, xo=xo: xo)
        sc3 = Scene("approach", 0, 12.0, [h, o])
        assert (ls.judge(sc3, h, [cue(1.5)], [ranged(1.0, "O")])["tb"] is None) == known


def test_news_is_judged_at_hs_place_and_never_from_the_desk_user(ls):
    # (gate_verify_r3) a cue about another object of H's label makes a bare cue about H no news only at H's place:
    # H (8.5 m at 1.5 s) not nearer than it by max(0.35 m, 15 %), not farther by the `closer` step (max(1 m, 25 %)): one
    # 0.7 m in front of H counts, one 2.5 m in front (laterally where H is) does not; and never the desk user (naming the
    # user says nothing about a visitor walking up in front of them)
    Said, Obj, Scene = ls.Said, ls.Obj, ls.Scene
    h = Obj("H", "person", lambda t: max(0.25, 10.0 - t), hazard=True, t_lo=0.0, contact=9.4)
    cue = lambda t, owner="H": Said(t, owner, "speak", "object", 0.5, None, "Person.")
    ranged = lambda t, owner: Said(t, owner, "speak", "object", 0.5, None, "Person, about 9 meters.")
    for zo, role, known in ((8.6, "", True), (7.8, "", True), (6.0, "", False), (10.5, "", False), (8.6, "user", False)):
        o = Obj("O", "person", lambda t, zo=zo: zo, lambda t: 0.0, role=role)
        sc3 = Scene("approach", 0, 12.0, [h, o])
        assert (ls.judge(sc3, h, [cue(1.5)], [ranged(1.0, "O")])["tb"] is None) == known, (zo, role)
        # (gate_verify_r4, bgref:906) a message about another object that does not say how far says nothing about H's
        # range: "Person." about O is no warning about H when the bare cue says how far H is; when the bare cue does
        # not say it either, the user heard as much as it says
        assert ls.judge(sc3, h, [ranged(1.5, "H")], [cue(1.0, "O")])["tb"] == 1.5
        assert (ls.judge(sc3, h, [cue(1.5)], [cue(1.0, "O")])["tb"] is None) == known, (zo, role)


def test_a_message_cut_by_an_interrupt_counts_only_up_to_the_cut(ls):
    # (gate_verify_r3) a gated message an interrupt cut counts only up to the cut: cut before its first word ended, it
    # named nothing (no warning, and the bare cue that follows is news); cut after "Person" but before the distance, H
    # is known, its range not (the 8 s rule needs the range said)
    Said, Obj, Scene = ls.Said, ls.Obj, ls.Scene
    h = Obj("H", "person", lambda t: max(0.25, 10.0 - t), hazard=True, t_lo=0.0, contact=9.4)
    sc = Scene("approach", 0, 12.0, [h])
    st = Obj("H", "person", lambda t: 7.0 if t < 5.0 else max(0.25, 7.0 - (t - 5.0)), hazard=True, t_lo=4.0, contact=11.4)
    sc2 = Scene("approach", 0, 12.0, [st])
    cue = lambda t, owner="H": Said(t, owner, "speak", "object", 0.5, None, "Person.")
    full = Said(3.0, "H", "speak", "object", 0.5, None, "Person, about 7 meters.", end=4.4)
    cut0 = Said(3.0, "H", "speak", "object", 0.5, None, "Person, about 7 meters.", end=3.1)
    cut1 = Said(3.0, "H", "speak", "object", 0.5, None, "Person, about 7 meters.", end=3.5)
    assert ls.heard_label(full) and ls.heard_range(full) and not ls.heard_label(cut0)
    assert ls.heard_label(cut1) and not ls.heard_range(cut1)
    assert ls.judge(sc, h, [cue(3.0)], [cut0])["severe"] and not ls.judge(sc, h, [cue(3.0)], [cut1])["late"]
    assert ls.judge(sc, h, [cue(3.6)], [cut0])["tb"] == 3.6 and ls.judge(sc, h, [cue(3.6)], [cut1])["tb"] is None
    still = Obj("H", "person", lambda t: 7.0, hazard=True, t_lo=4.0)
    sc4 = Scene("approach", 0, 12.0, [still])
    assert ls.judge(sc4, still, [cue(5.5)], [full])["tb"] is None and ls.judge(sc4, still, [cue(5.5)], [cut1])["tb"] == 5.5


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


# the scheduler's view (gate_verify_r2 and r3): scenes where it once diverged, and one scene of every family; the
# configurations whose scheduler must be shown exactly what it is shown alone (ls.SAME_VIEW_CONFIGS: the gate's, and the
# controller's in navigate and find modes, with the cane on and off)
VIEW_SCENES = [("static_path:70997", "pcacs"), ("find:71473", "rl"), ("swap:91001", "rl"), ("handover:3", "rl"),
               ("v3_find_disp:620507", "rl"), ("find:400522", "pcacs"), ("find:400522", "rl")]


@pytest.mark.parametrize("sid,pol", VIEW_SCENES)
def test_the_scheduler_is_shown_and_offers_under_the_gate_what_it_does_alone(ls, sid, pol):
    # Critical (gate_verify_r2): the gate's own messages stamped the tracks as announced, its holds refunded the speech
    # budget and freed the channel. Critical (gate_verify_r3, v3_find_disp:620507): a routine cue said late was taken
    # for the scheduler's own message, so its channel was shown busy 1 s longer than alone, and the learned scheduler
    # never named a person walking at the user. At every decision the scheduler is now shown the channel it has alone
    # (busy, remaining time, current track) and makes the offers it makes alone, in every configuration that leaves its
    # settings as they are
    fam, seed = sid.split(":")
    sc = ls.build_scene(fam, int(seed))
    frames = ls.make_frames(sc)
    views = {}
    ls.run_configs(sc, frames, pol, ["bare"] + [c for c in ls.SAME_VIEW_CONFIGS if c in sc.configs], views=views)
    assert sum(1 for v in views["bare"] if v[4]) >= 2
    bad = ls.diverge(sc, frames, pol)
    assert not bad, bad


def test_the_scheduler_is_shown_what_it_is_shown_alone_in_every_family(ls):
    # the same property on one scene of every family (the waiting-cue families find_disp and displaced, and the cane
    # family with the cane on), alternating the schedulers
    bad = []
    for sid, pol in [(f"{fam}:{1000 + k}", ("pcacs", "rl")[k % 2]) for k, fam in enumerate(ls.FAMILIES)]:
        fam, seed = sid.split(":")
        sc = ls.build_scene(fam, int(seed))
        bad += ls.diverge(sc, ls.make_frames(sc), pol)
    assert not bad, bad


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


# the third verification's (gate_verify_r3) scenes and this round's: the waiting cue (critical), controller-only silence
# in quiet modes, desk noise (an arm on the user, a shallow lean on noisy depth), the cane on
SUBSET_R3 = [
    ("v3_find_disp:620507", "rl", ("gate", "ctl:find")),  # a routine cue said late kept the scheduler off the channel
    ("side_entry:71409", "rl", ("ctl:scene", "ctl:read")),  # the scene mode's quiet verbosity silenced the scheduler
    ("desk:202425", "rl", ("gate_op", "ctl:desk", "ctl:auto")),     # an arm detected as a person: reminders, lost
    ("desk:201690", "rl", ("gate_op", "ctl:desk", "ctl:auto")),     # ... a buzz about it
    ("desk:200514", "pcacs", ("gate_op", "ctl:desk", "ctl:auto")),  # a 0.32 m lean on 0.067 m depth noise
    ("desk:900054", "pcacs", ("gate_op", "ctl:desk", "ctl:auto")),  # a 0.34 m lean, a 1 cm reference error
    ("deskwalk:302868", "pcacs", ("ctl:desk",)),          # a stranger walking up held as the user in a re-warm-up
    ("find:500033", "pcacs", ("gate_nobg", "ctl:navigate")),   # a near bench's "very close" within min_gap_s
    ("find:511113", "pcacs", ("gate_nobg", "ctl:navigate")),   # a near rescue cut the FIND goal's cue before its name
    ("find:2913", "pcacs", ("gate", "gate_nobg")),        # the FIND goal sign walked to, offered again at 6.6 m: held
    ("cane:27", "pcacs", None),                           # still at the start (auto: a desk), then walking: the cane
    ("cane:61", "rl", None),
]


def test_safety_subset_r3_has_no_severe_rows_and_desk_runs_within_limits(ls):
    rows, desk = [], []
    for sid, pol, configs in SUBSET_R3:
        fam, seed = sid.split(":")
        res = ls.run_scene(fam, int(seed), [pol], configs=configs)
        rows += res["rows"]
        desk += res["desk"]
    severe = [r for r in rows if r["severe"]]                       # the gate's or the controller's
    assert not severe, [(r["sid"], r["config"], r["blame"], r["tb"], r["tg"], r["every_tb"], r["every_tg"]) for r in severe]
    late_desk = [r for r in rows if r["family"] in ls.DESK_FAMILIES and r["late_first"] and r["blame"] == "gate"]
    assert not late_desk, [(r["sid"], r["config"], r["tb"], r["tg"]) for r in late_desk]   # (stranger / visitor)
    assert not ls.desk_acceptance(desk)["bad"], ls.desk_acceptance(desk)["bad"]
    assert sum(r["tb"] is not None for r in rows) >= 3


def test_news_after_a_long_static_spell_and_at_motion_onset(ls):
    # (gate_verify_r4, minor issue 5) (b) a bare reminder about a static object the gated user heard of, however long
    # ago, is no news while it stays at that place (the gate holds such repeats for hold_static_s, 20-60 s; the 8 s
    # rule alone made 757 rows in v4_small_static severe 8-12 s after the last gated cue); (a) one that has started to
    # come at the user is news before it is any nearer (v4_swap_fast: the bare re-alert as the goal person started to
    # walk, while its range had moved under the 0.35 m / 15 % tolerance)
    Said, Obj, Scene = ls.Said, ls.Obj, ls.Scene
    st = Obj("H", "person", lambda t: 4.0 if t < 30.0 else max(0.25, 4.0 - 1.5 * (t - 30.0)), hazard=True, t_lo=0.0,
             contact=32.3)
    sc = Scene("small_static", 0, 40.0, [st])
    said = lambda t: Said(t, "H", "speak", "object", 0.5, None, "Person, about 4 meters.")
    assert ls.judge(sc, st, [said(20.0)], [said(2.0)])["tb"] is None            # static since: no news after 18 s
    assert ls.judge(sc, st, [said(30.3)], [said(2.0)])["tb"] == 30.3            # it started to walk at the user
    assert ls.judge(sc, st, [said(30.3)], [said(29.0)])["tb"] == 30.3           # ... even 1.3 s after the user heard it
    assert ls.judge(sc, st, [said(30.3)], [said(30.1)])["tb"] is None           # ... not once heard after the start
    # the user's own walk is no onset: a chair the user starts to walk at (the scene's walk) is no news by that alone
    walk = lambda t: max(0.0, t - 30.0)
    ch = Obj("H", "chair", lambda t: 4.0 - walk(t), hazard=True, t_lo=0.0, contact=33.4)
    sc2 = Scene("cane", 0, 40.0, [ch], walk=walk)
    assert ls.judge(sc2, ch, [said(30.3)], [said(27.0)])["tb"] is None


def test_replies_on_the_channel_cut_messages_and_are_blamed_against_the_same_controller(ls):
    # the +reply configurations (gate_verify_r4: run_live speak_reply on the real channel): a reply that cuts a gated cue
    # ends it (Said.end), the scheduler is still shown its own channel, and the blame is against the controller without
    # the gate with the same replies
    sc = ls.build_scene("v4_flip_size", 970079)
    frames = ls.make_frames(sc)
    runs = ls.run_configs(sc, frames, "rl", ["ctl:handheld+reply", "ctl:handheld+reply:nogate"])
    cut = [s for s in runs["ctl:handheld+reply"] if s.action != "haptic" and s.end < s.t + 1.2 - 1e-6]
    assert cut and cut[0].t == 2.5 and cut[0].end == 3.0, [(s.t, s.end, s.message) for s in runs["ctl:handheld+reply"]]
    assert not ls.diverge(sc, frames, "rl", ["ctl:handheld+reply", "ctl:navigate+reply", "ctl:auto+reply"])


# the fourth verification's (gate_verify_r4) scenes: the FIND swap at motion onset, a rebaseline's warm-up silencing a
# child in the path, a reply cutting a cue before its distance phrase, a still desk user when the depth comes back or
# while the user's box flickers
SUBSET_R4 = [
    ("v4_swap_fast:975020", "rl", ("gate", "ctl:find")),             # the swap held the warning about H behind D's cue
    ("v4_swap_fast:975019", "rl", ("gate_nobg",)),                    # ... a swap before any range step: H cuts it
    ("v4_find_two_goals:900323", "rl", ("ctl:find", "ctl:navigate")),   # (ext4.py's family: built from its params)
    ("v4_flip_size:970034", "rl", ("ctl:scene", "ctl:auto")),         # a child the box reads at 4 m, desk detection
    ("v4_flip_size:970079", "rl", ("ctl:handheld+reply",)),           # a reply cut "Chair, about 2.5 meters." early
    ("v4_desk_still:900231", "pcacs", ("gate_op", "ctl:desk", "ctl:auto")),   # depth back after 8 s off: no cue
    ("v4_desk_still:990122", "pcacs", ("gate_op", "ctl:desk", "ctl:auto")),   # the user's box flickering: no cue
    ("v4_desk_still:990043", "pcacs", ("gate_op", "ctl:desk", "ctl:auto")),   # ... with a depth flip
]


def test_safety_subset_r4_has_no_severe_rows_and_desk_runs_within_limits(ls):
    rows, desk = [], []
    for sid, pol, configs in SUBSET_R4:
        fam, seed = sid.rsplit(":", 1)
        if fam not in ls.BUILDERS and fam not in ls.ALIASES:
            continue                                  # (a family only the verifier's ext4.py builds)
        res = ls.run_scene(fam, int(seed), [pol], configs=configs, exact=True)
        rows += res["rows"]
        desk += res["desk"]
    severe = [r for r in rows if r["severe"]]
    assert not severe, [(r["sid"], r["config"], r["blame"], r["tb"], r["tg"], r["every_tb"], r["every_tg"]) for r in severe]
    assert not ls.desk_acceptance(desk)["bad"], ls.desk_acceptance(desk)["bad"]
    assert sum(d["user_cues"] for d in desk if d["config"] in ls.OPERATOR_CONFIGS) == 0
    assert sum(r["tb"] is not None for r in rows) >= 4
