"""LiveController: the gate, activity detector, mode adapter, summary and log composed as run_live.py uses them.
No camera, no audio, no model files: a fake scheduler and synthetic frames."""
import math

import numpy as np
import pytest

from seeforme.geometry import CameraModel
from seeforme.live.controller import LiveController
from seeforme.live.summary import SessionLog, scene_summary
from seeforme.types import Cue, Track

CAM = CameraModel()


class FakeScheduler:
    """Offers a cue about the nearest track on every decision (worst case for repetition)."""
    name = "fake"
    decision_interval = 0.5
    supports_actions = True

    class _Cfg:
        verbosity = "balanced"
        units = "m"

    def __init__(self, kind="object", severity=0.5):
        self.cfg = self._Cfg()
        self.kind, self.severity = kind, severity
        self.goal = []

    def reset(self):
        pass

    def last_observations(self):
        return []

    def set_goal(self, tags):
        self.goal = list(tags)

    def decide(self, tracks, t, ch=None):
        if not tracks:
            return []
        trk = min(tracks, key=lambda x: x.distance_m or 99)
        trk.last_announced = t; trk.announce_count += 1
        return [Cue(t, trk.track_id, trk.label, trk.kind, f"{trk.label.capitalize()}.", 1.0, self.name, text=trk.text,
                    distance_m=trk.distance_m, severity=self.severity)]


def person(tid, t, z=1.5, x=0.0, kind="object", text=None):
    b = CAM.project_box(x, z, 1.7, 0.5)
    tr = Track(tid, "person" if kind == "object" else "text", kind, b, 0.9, 0.0, t, text=text, confirmed=True)
    tr.hits, tr.distance_m, tr.depth_conf, tr.lateral_m = 30, z, 0.9, x
    tr.ttc_s, tr.path_prob = math.inf, 0.9
    for i in range(6):
        tr.dist_history.append((t - 1.2 + 0.2 * i, z, x))
    return tr


def still_frame(rng, base):
    return np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)


def run(ctl, seconds, fps=10, tracks_fn=None, frame_fn=None):
    """Drive the controller like run_live: frames every 1/fps, a decision every 0.5 s. Returns the spoken cues."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (240, 320, 3)).astype(np.float64)
    spoken, next_dec = [], 0.0
    for i in range(int(seconds * fps)):
        t = i / fps
        tracks = tracks_fn(t) if tracks_fn else []
        frame = frame_fn(t, rng, base) if frame_fn else still_frame(rng, base)
        if t + 1e-9 >= next_dec:
            next_dec = t + 0.5
            spoken += ctl.policy.decide(tracks, t, None)
        ctl.on_frame(frame, t, tracks)
    return spoken


def test_desk_webcam_user_is_not_repeated():
    """The reported bug: a static person in front of a desk webcam announced every few seconds."""
    bare = run(LiveController(FakeScheduler(), gate=False), 40, tracks_fn=lambda t: [person(1, t)])
    ctl = LiveController(FakeScheduler())
    gated = run(ctl, 40, tracks_fn=lambda t: [person(1, t)])
    assert len(bare) >= 60 and len(gated) <= 1
    assert ctl.desk and ctl.adapter.mode == "scene"            # still camera detected as a desk webcam


def test_new_person_arriving_at_the_desk_is_announced():
    def tracks(t):
        out = [person(1, t)]
        if t >= 12:
            out.append(person(2, t, z=max(1.2, 4.0 - 0.5 * (t - 12)), x=0.8))
        return out
    ctl = LiveController(FakeScheduler())
    spoken = run(ctl, 25, tracks_fn=tracks)
    assert any(c.track_id == 2 for c in spoken)


def test_track_id_switches_do_not_reannounce():
    ctl = LiveController(FakeScheduler(), setup="handheld")
    spoken = run(ctl, 30, tracks_fn=lambda t: [person(1 + int(t // 3), t, z=2.0)])
    assert len(spoken) <= 2


def test_pinned_read_mode_mutes_objects_but_not_text():
    ctl = LiveController(FakeScheduler(), mode="read", setup="handheld")
    assert ctl.adapter.mode == "read"
    assert run(ctl, 5, tracks_fn=lambda t: [person(1, t)]) == []
    ctl2 = LiveController(FakeScheduler(), mode="read", setup="handheld")
    spoken = run(ctl2, 5, tracks_fn=lambda t: [person(5, t, kind="text", text="EXIT")])
    assert spoken and spoken[0].kind == "text"


def test_critical_cue_passes_in_read_mode():
    ctl = LiveController(FakeScheduler(severity=1.0), mode="read", setup="handheld")
    assert run(ctl, 3, tracks_fn=lambda t: [person(1, t)])


def test_policy_switch_keeps_mode_and_goal():
    ctl = LiveController(FakeScheduler(), mode="scene", setup="desk", goal_tags=["door"])
    new = FakeScheduler()
    ctl.set_policy(new)
    assert ctl.adapter.mode == "scene" and new.goal == ["door"] and new.cfg.verbosity == "quiet"
    assert ctl.gate is not None and ctl.gate.gate_cfg.operator_filter


def test_mode_cycle_and_bad_arguments():
    ctl = LiveController(FakeScheduler())
    phrases = [ctl.cycle_mode() for _ in range(5)]
    assert phrases[0] == "Mode navigate." and phrases[-1].startswith("Automatic mode")
    with pytest.raises(ValueError):
        LiveController(FakeScheduler(), mode="dance")
    with pytest.raises(ValueError):
        LiveController(FakeScheduler(), setup="ceiling")


def test_scene_summary_and_session_log(tmp_path):
    tr = [person(1, 0.0, z=1.5), person(2, 0.0, z=4.0, x=-2.0), person(3, 0.0, z=6, x=1), person(4, 0.0, z=7, x=2)]
    s5 = scene_summary(tr + [person(5, 0.0, z=8, x=-1)], CAM)
    assert s5.endswith("2 more people.")
    s = scene_summary(tr, CAM)
    assert s.startswith("A person ahead, about 1.5 meters.") and "on your left" in s and s.endswith("1 more person.")
    assert "does not mean the way is clear" in scene_summary([], CAM)
    log = SessionLog(str(tmp_path / "s.jsonl"))
    ctl = LiveController(FakeScheduler(), setup="handheld", log=log)
    run(ctl, 3, tracks_fn=lambda t: [person(1, t)])
    log.close()
    kinds = [line.split('"kind": "')[1].split('"')[0] for line in open(tmp_path / "s.jsonl")]
    assert "cue" in kinds and "mode" in kinds


def test_scene_summary_says_one_object_once():
    """make live-check: "what's around me" said "A person slightly right, about 2.5 meters." twice: two overlapping
    tracks of one person. Same label, direction and distance with overlapping boxes is one object; two people side by
    side that read the same are counted; the "more" count does not count a duplicate either."""
    a = person(1, 0.0, z=2.5, x=0.6)
    b = person(2, 0.0, z=2.5, x=0.65)                    # the same person, a second track
    assert scene_summary([a, b], CAM) == "A person slightly right, about 2.5 meters."
    c = person(3, 0.0, z=2.5, x=1.05)                    # another person beside them, same phrase, no overlap
    s = scene_summary([a, b, c], CAM)
    assert s.startswith("2 people slightly right, about 2.5 meters.") and s.count("person") == 0, s
    far = [person(10 + i, 0.0, z=6 + i, x=-2.0 + 2.0 * i) for i in range(3)]
    s = scene_summary([a, b] + far + [person(20, 0.0, z=8, x=2.05)], CAM)    # 20 is a second track of far[2]
    assert s.count("slightly right, about 2.5 meters") == 1 and s.endswith("1 more person."), s
    t1 = person(30, 0.0, z=1.5, kind="text", text="EXIT")
    t2 = person(31, 0.0, z=1.5, x=0.02, kind="text", text="EXIT")
    assert scene_summary([t1, t2], CAM) == "A sign that says EXIT ahead, about 1.5 meters."


# ---- the audio channel: a cane cue or a spoken reply never holds a safety cue back --------------------------------
# The pipeline's channel (seeforme/eval/runner.SpeechChannel, as LivePipeline uses it): what plays holds it, a scheduler
# sees it busy, an interrupt cuts it. The user walks at a wall (the cane says "Obstacle ahead" at 3.0 s, severity 0.7)
# and something appears while that message, or a reply registered as run_live does (reply_cue), still plays.
from seeforme.cues.pcacs import ChannelState, PCACSPolicy                       # noqa: E402
from seeforme.eval.runner import SpeechChannel                                  # noqa: E402
from seeforme.live.cane import CANE_POLICY                                      # noqa: E402
from seeforme.live.commands import HELP_VOICE                                   # noqa: E402
from seeforme.live.controller import REPLY_TRACK_ID, reply_cue                  # noqa: E402
from tests.test_live_cane import FakeScheduler as NearestFirst, render, track, wall   # noqa: E402


class ChannelRule(NearestFirst):
    """P-CACS's channel rule (pcacs.py decide) for a non-critical cue: speak on a free channel; a busy channel holds it
    (it would cut in only with a critical cue or one scoring far above the message playing). The nearest track first."""

    def __init__(self, severity=0.5):
        super().__init__(severity)
        self.shown: list = []                    # (t, busy) of the channel each decision was shown

    def decide(self, tracks, t, ch=None):
        self.shown.append((t, bool(ch is not None and ch.busy)))
        return [] if ch is not None and ch.busy else super().decide(tracks, t, ch)


def drive_channel(policy, tracks_fn, cane=True, reply=None, gate=False, goal=None, seconds=8.0,
                  walk=lambda t: max(0.3, 3.5 - 0.6 * t), aid="none"):
    """Every cue the pipeline would act on: frames at 10 fps, a decision every 0.5 s, depth maps at 5 per s of a wall
    the user walks at (walk(t) metres; None: no depth). reply=(t, text): a reply put on the channel at t. The cane
    says "Obstacle ahead" at 3.0 s without an aid (2 m), at 2.0 s with a cane (2.5 m): aid "none" keeps the timings
    these channel tests were written for."""
    ctl = LiveController(policy, setup="handheld", gate=gate, cane=cane, goal_tags=goal, aid=aid)
    chan = SpeechChannel(1.2, 0.35)
    out, next_dec, depth = [], 0.0, None
    for i in range(int(seconds * 10)):
        t = i / 10
        if reply is not None and abs(t - reply[0]) < 1e-9:
            chan.speak(reply_cue(reply[1], t), interrupt=True)
        tracks = tracks_fn(t)
        if t + 1e-9 >= next_dec:
            next_dec = t + 0.5
            busy = not chan.free(t)
            cur = chan.current if busy else None
            ch = ChannelState(busy=busy, remaining_s=chan.remaining(t), current_utility=cur.utility if cur else 0.0,
                              current_severity=cur.severity if cur else 0.0, current_track=cur.track_id if cur else None)
            cues = ctl.policy.decide(tracks, t, ch)
            assert len(cues) <= 1
            for c in cues:
                out.append(c)
                if c.action != "haptic":
                    chan.speak(c, interrupt=c.action == "interrupt")
        if i % 2 == 0 and walk is not None:
            depth = render([wall(walk(t))], height=1.4, pitch=15, seed=i)
        ctl.on_frame(None, t, tracks, depth=depth)
    return out, ctl


def _first(cues, label):
    return next((c for c in cues if c.label == label), None)


IN_PATH_CHAIR = lambda appear: lambda t: [track(5, "chair", (0.80, 0.5, 0.95, 0.9), 1.5, t, path=0.8)] if t >= appear else []  # noqa: E731
GOAL_CHAIR = lambda appear: lambda t: [track(6, "chair", (0.05, 0.5, 0.2, 0.8), 3.0, t)] if t >= appear else []          # noqa: E731


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("aid,appear", [("none", 3.5), ("none", 4.0), ("cane", 2.5), ("cane", 3.0)])
@pytest.mark.parametrize("what", ["in-path chair", "FIND goal"])
def test_a_spoken_cane_cue_never_holds_a_safety_cue_back(what, aid, appear, gate):
    """Reviewer repro (critical), real P-CACS: the cane's "Obstacle ahead" (3.0 s) held the channel at severity 0.7,
    which P-CACS does not cut for a non-critical cue, so an in-path chair appearing at 3.5 s was said at 5.0 s. The
    same with the default cane aid, whose cue comes earlier (2.5 m, at 2.0 s)."""
    fn, goal = (IN_PATH_CHAIR(appear), None) if what == "in-path chair" else (GOAL_CHAIR(appear), ["chair"])
    with_cane, ctl = drive_channel(PCACSPolicy(), fn, cane=True, gate=gate, goal=goal, aid=aid)
    bare, _ = drive_channel(PCACSPolicy(), fn, cane=False, gate=gate, goal=goal, aid=aid)
    spoken_cane = [c for c in with_cane if c.policy == CANE_POLICY and c.action != "haptic" and c.t < appear]
    assert spoken_cane and appear - spoken_cane[-1].t <= 1.0          # the cane's message was still playing
    assert _first(bare, "chair").t == appear
    assert _first(with_cane, "chair").t == appear and _first(with_cane, "chair").action == "interrupt"
    assert ctl.cane_stats["cut_in"] >= 1


@pytest.mark.parametrize("label,ttc,speed", [("person", 2.5, 0.0), ("car", None, 0.0), ("person", None, -0.8)])
def test_an_approaching_object_or_a_vehicle_cuts_the_cane_cue(label, ttc, speed):
    """The same for an approaching person (TTC, or a measured closing speed) and a vehicle, with P-CACS's channel rule."""
    def fn(t):
        if t < 3.5:
            return []
        tr = track(5, label, (0.1, 0.3, 0.3, 0.8), 4.0, t, ttc=ttc if ttc is not None else math.inf)
        tr.rel_speed_mps = speed
        return [tr]
    with_cane, _ = drive_channel(ChannelRule(), fn, cane=True)
    bare, _ = drive_channel(ChannelRule(), fn, cane=False)
    assert _first(bare, label).t == _first(with_cane, label).t == 3.5


def test_a_routine_cue_waits_for_the_cane_cue():
    """While the cane speaks, a routine cue is not said over it: with nothing safety-relevant in view the scheduler sees
    the busy channel; with a vehicle in view it sees a free one, and its routine cue (already marked as said) waits for
    the channel and is said once the cane's message has ended."""
    person = lambda t: track(1, "person", (0.85, 0.3, 0.95, 0.8), 2.0, t) if t >= 3.5 else None   # noqa: E731
    for with_car in (False, True):
        pol = ChannelRule()
        car = lambda t: [track(9, "car", (0.0, 0.4, 0.1, 0.5), 9.0, t)] if with_car and t >= 3.5 else []   # noqa: E731
        cues, ctl = drive_channel(pol, lambda t: [p for p in [person(t)] if p] + car(t), cane=True,
                                  walk=lambda t: max(1.6, 3.5 - 0.6 * t))
        cane = next(c for c in cues if c.policy == CANE_POLICY)
        said = [c for c in cues if c.label == "person"]
        assert said and said[0].action == "speak" and said[0].t >= cane.t + 1.5, [(c.t, c.label, c.action) for c in cues]
        assert said[0].t <= 5.0 and len([c for c in said if c.t < 6.0]) == 1
        busy_at_35 = dict(pol.shown)[3.5]
        assert busy_at_35 is (not with_car)                         # shown free only with something safety-relevant
        assert (ctl.cane_stats["waited"] >= 1) is with_car


def test_a_cane_cue_cuts_a_reply_but_never_a_scheduler_message():
    cues, _ = drive_channel(NearestFirst(), lambda t: [], cane=True, reply=(2.9, "I said: Chair, about 2 meters."))
    first = next(c for c in cues if c.policy == CANE_POLICY)
    assert first.t == 3.0 and first.action == "interrupt" and first.interrupted_track == REPLY_TRACK_ID
    # a scheduler message playing: the cane waits for it
    talk = lambda t: [track(1, "person", (0.85, 0.3, 0.95, 0.8), 2.5, t)] if 2.9 <= t < 3.0 else []   # noqa: E731
    cues, _ = drive_channel(ChannelRule(), talk, cane=True)
    assert all(c.action != "interrupt" for c in cues if c.policy == CANE_POLICY)


@pytest.mark.parametrize("what", ["in-path chair", "FIND goal", "approaching person", "vehicle"])
def test_a_reply_never_holds_a_safety_cue_back(what):
    """Reviewer repro (major): the help reply (9.4 s of channel) registered at 3.0 s held an in-path chair appearing at
    3.5 s until 12.5 s. Now it is said at 3.5 s, cutting the reply, exactly as without the reply."""
    if what in ("in-path chair", "FIND goal"):
        fn, goal = (IN_PATH_CHAIR(3.5), None) if what == "in-path chair" else (GOAL_CHAIR(3.5), ["chair"])
        make = PCACSPolicy
    else:
        label, ttc = ("person", 2.5) if what == "approaching person" else ("car", math.inf)
        fn = lambda t: [track(5, label, (0.1, 0.3, 0.3, 0.8), 4.0, t, ttc=ttc)] if t >= 3.5 else []   # noqa: E731
        goal, make = None, ChannelRule
    for gate in (False, True):
        bare, _ = drive_channel(make(), fn, cane=False, gate=gate, goal=goal, walk=None)
        replied, ctl = drive_channel(make(), fn, cane=False, gate=gate, goal=goal, walk=None, reply=(3.0, HELP_VOICE))
        label = "chair" if "chair" in what or "goal" in what else fn(3.5)[0].label
        assert _first(bare, label).t == _first(replied, label).t == 3.5, (gate, [(c.t, c.action, c.message) for c in replied])
        assert _first(replied, label).action == "interrupt" and _first(replied, label).interrupted_track == REPLY_TRACK_ID


def test_a_routine_cue_waits_for_a_reply():
    """Nothing safety-relevant in view: the scheduler sees the reply's busy channel and speaks after it, as before."""
    pol = ChannelRule()
    still = lambda t: [track(1, "person", (0.85, 0.3, 0.95, 0.8), 2.0, t)] if t >= 3.5 else []   # noqa: E731
    text = "Paused. I'll still warn you about hazards."
    cues, ctl = drive_channel(pol, still, cane=False, walk=None, reply=(3.0, text))
    ends = 3.0 + SpeechChannel(1.2, 0.35).duration(reply_cue(text, 3.0))       # 5.45 s: the next decision is 5.5 s
    assert cues[0].label == "person" and cues[0].action == "speak" and cues[0].t == math.ceil(ends * 2) / 2
    assert all(busy for t, busy in pol.shown if 3.0 <= t < ends) and ctl.cane_stats["cut_in"] == 0


def test_camera_change_forgets_the_old_view():
    cues, ctl = drive_channel(NearestFirst(), lambda t: [], cane=True, seconds=4.0)
    assert ctl.cane.frame is not None and ctl.cane.floor is not None
    ctl._waiting.append((cues[0], 4.0))
    ctl.camera_changed()
    assert ctl.cane.frame is None and ctl.cane.floor is None and ctl.cane.memory == {} and ctl._waiting == []
    assert ctl.activity.activity.value == "unknown"


def test_through_a_reply_the_scheduler_sees_its_own_message_still_playing():
    """A reply that cut the scheduler's own message: the scheduler is shown that message as still playing (busy), as it
    would be without the reply, not a free channel; a free one made the learned scheduler skip a buzz on 00700."""
    in_path = lambda t: [track(1, "person", (0.4, 0.2, 0.6, 0.9), 2.0, t, path=0.8)]   # noqa: E731
    bare_pol, replied_pol = ChannelRule(), ChannelRule()
    bare, _ = drive_channel(bare_pol, in_path, cane=False, walk=None, seconds=4.0)
    replied, ctl = drive_channel(replied_pol, in_path, cane=False, walk=None, seconds=4.0, reply=(0.1, HELP_VOICE))
    assert bare_pol.shown == replied_pol.shown and bare_pol.shown[1] == (0.5, True)
    assert [c.t for c in bare] == [c.t for c in replied] == [0.0, 1.5, 3.0]
    assert replied[1].action == "interrupt" and replied[1].interrupted_track == REPLY_TRACK_ID
