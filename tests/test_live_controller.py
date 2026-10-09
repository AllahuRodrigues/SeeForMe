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
