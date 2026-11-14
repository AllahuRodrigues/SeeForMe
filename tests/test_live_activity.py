"""ActivityDetector (stationary / walking / turning / reading) and TaskAdapter (mode -> gate / verbosity settings).

Synthetic frames only: no camera, no models, a few seconds in total."""
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

try:
    from seeforme.live.activity import (Activity, ActivityConfig, ActivityDetector, MODE_SETTINGS, ModeSettings,
                                        TaskAdapter, measure_motion, gray_small)
except ImportError:  # seeforme/live/__init__.py also imports modules owned by other parts of the live layer
    import importlib.util, pathlib, sys
    _p = pathlib.Path(__file__).resolve().parents[1] / "seeforme" / "live" / "activity.py"
    _spec = importlib.util.spec_from_file_location("seeforme_live_activity", _p)
    _m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _m; _spec.loader.exec_module(_m)
    Activity, ActivityConfig, ActivityDetector = _m.Activity, _m.ActivityConfig, _m.ActivityDetector
    MODE_SETTINGS, ModeSettings, TaskAdapter = _m.MODE_SETTINGS, _m.ModeSettings, _m.TaskAdapter
    measure_motion, gray_small = _m.measure_motion, _m.gray_small

from seeforme.types import Cue, Track


# ---- synthetic scenes ------------------------------------------------------------------------------------------
def texture(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.uniform(30, 220, (h // 16, w // 16, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img += cv2.GaussianBlur(rng.normal(0, 25, (h, w, 3)).astype(np.float32), (0, 0), 1.5)
    for _ in range(25):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 120)), y + int(rng.integers(5, 60))),
                      tuple(float(v) for v in rng.uniform(0, 255, 3)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def desk_frame(base, t, rng, noise=2.0):
    """A desk webcam: static background, a person-sized blob swaying a few px and a moving hand, sensor noise."""
    img = base.astype(np.float32).copy()
    h, w = img.shape[:2]
    sway = 5 * np.sin(2 * np.pi * 0.25 * t)
    cv2.ellipse(img, (int(w / 2 + sway), int(h * 0.45)), (60, 80), 0, 0, 360, (90, 110, 150), -1)
    hx, hy = int(w * 0.3 + 80 * np.sin(2 * np.pi * 0.4 * t)), int(h * 0.75)
    cv2.ellipse(img, (hx, hy), (30, 40), 20, 0, 360, (120, 140, 190), -1)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run(det, frames_ts, **kw):
    return [(t, det.update(f, t, **kw)) for f, t in frames_ts]


def first_time(seq, activity):
    return next((t for t, a in seq if a == activity), None)


def static_seq(n, fps=10.0, seed=0):
    base = texture(seed=seed)
    rng = np.random.default_rng(seed)
    return [(desk_frame(base, i / fps, rng), i / fps) for i in range(n)]


def pan_seq(n, px_per_step, fps=10.0, w=320, h=240, seed=1):
    big = texture(h=h, w=w + abs(px_per_step) * n + 10, seed=seed)
    x0 = 0 if px_per_step > 0 else abs(px_per_step) * n
    return [(big[:, x0 + i * px_per_step: x0 + i * px_per_step + w].copy(), i / fps) for i in range(n)]


def zoom_seq(n, rate=1.04, fps=10.0, seed=2):
    big = texture(h=720, w=960, seed=seed)
    H, W = big.shape[:2]
    out = []
    for i in range(n):
        s = rate ** -i                               # the visible window shrinks: the scene grows (walking forward)
        ch, cw = int(H * s), int(W * s)
        y0, x0 = (H - ch) // 2, (W - cw) // 2
        out.append((cv2.resize(big[y0:y0 + ch, x0:x0 + cw], (320, 240), interpolation=cv2.INTER_AREA), i / fps))
    return out


def trk(tid, bbox, kind="object", label="chair", t=0.0, misses=0):
    return Track(track_id=tid, label=label, kind=kind, bbox=bbox, confidence=0.9, first_seen=0.0, last_seen=t,
                 misses=misses, text="EXIT" if kind == "text" else None)


# ---- measurements ----------------------------------------------------------------------------------------------
def test_measure_motion_sign_and_does_not_modify_inputs():
    a = gray_small(texture(h=240, w=400, seed=3)[:, :320])
    b = gray_small(texture(h=240, w=400, seed=3)[:, 8:328])       # window moved right: content moves left
    a0, b0 = a.copy(), b.copy()
    s = measure_motion(a, b, 0.2)
    assert s.dx < -0.1 and abs(s.dy) < 0.05                      # ~8 px of 320 per 0.2 s = -0.125 widths/s
    assert np.array_equal(a, a0) and np.array_equal(b, b0)       # cv2.phaseCorrelate works in place; we copy
    still = measure_motion(a, a + np.random.default_rng(0).normal(0, 1, a.shape).astype(np.float32), 0.2)
    assert still.flow < 0.3 and abs(still.dx) < 0.02


# ---- detector ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fps", [5.0, 10.0, 30.0])
def test_static_desk_is_stationary_within_2s(fps):
    det = ActivityDetector()
    seq = run(det, static_seq(int(4 * fps), fps=fps))
    t_still = first_time(seq, Activity.STATIONARY)
    assert t_still is not None and t_still <= 2.0
    assert all(a == Activity.STATIONARY for t, a in seq if t >= t_still)
    assert det.confidence > 0.9 and det.energy < 0.02 and det.motion < 0.02
    assert "stationary" in det.describe()


def test_horizontal_pan_is_turning_with_direction():
    det = ActivityDetector()
    seq = run(det, pan_seq(30, 12))                                # 12 px / 0.1 s on 320 px: 0.375 widths/s
    assert first_time(seq, Activity.TURNING) is not None and first_time(seq, Activity.TURNING) <= 2.0
    assert seq[-1][1] == Activity.TURNING
    assert det.shift[0] < -0.2 and "right" in det.describe()     # content moves left: the camera turns right
    det2 = ActivityDetector()
    run(det2, pan_seq(30, -12))
    assert det2.activity == Activity.TURNING and det2.shift[0] > 0.2 and "left" in det2.describe()


def test_zoom_in_is_walking_not_stationary():
    det = ActivityDetector()
    seq = run(det, zoom_seq(30))
    assert all(a != Activity.STATIONARY for _, a in seq)
    assert det.activity == Activity.WALKING
    assert det.state.forward > 0.1


def test_static_with_text_tracks_is_reading():
    det = ActivityDetector()
    frames = static_seq(50)
    seq = []
    for f, t in frames:
        tracks = [trk(1, (0.3, 0.3, 0.6, 0.4), kind="text", label="text", t=t), trk(2, (0.1, 0.5, 0.3, 0.9), t=t)]
        seq.append((t, det.update(f, t, tracks=tracks)))
    t_read = first_time(seq, Activity.READING)
    assert t_read is not None and t_read <= 3.0
    assert seq[-1][1] == Activity.READING
    # explicit text_present overrides the tracks; the text going away returns to STATIONARY after the dwell
    t0 = frames[-1][1]
    more = [(f, t0 + 0.1 * (i + 1)) for i, (f, _) in enumerate(static_seq(40, seed=0))]
    seq2 = [(t, det.update(f, t, tracks=[], text_present=False)) for f, t in more]
    assert seq2[-1][1] == Activity.STATIONARY


def test_text_while_walking_is_not_reading():
    det = ActivityDetector()
    seq = [(t, det.update(f, t, text_present=True)) for f, t in zoom_seq(30)]
    assert all(a != Activity.READING for _, a in seq)


def test_hysteresis_single_noisy_frame_does_not_flip():
    det = ActivityDetector()
    frames = static_seq(60)
    rng = np.random.default_rng(5)
    states = []
    for i, (f, t) in enumerate(frames):
        if i in (30, 45):                                          # a glitch: one frame of pure noise / a flash
            f = rng.integers(0, 255, f.shape, dtype=np.uint8) if i == 30 else np.full_like(f, 255)
        states.append((t, det.update(f, t)))
    settled = [a for t, a in states if t >= 2.0]
    assert settled and all(a == Activity.STATIONARY for a in settled)


def test_hysteresis_short_pan_does_not_switch_but_long_does():
    cfg = ActivityConfig()
    det = ActivityDetector(cfg)
    run(det, static_seq(30))
    assert det.activity == Activity.STATIONARY
    t0 = 3.0
    pan = pan_seq(40, 12)
    run(det, [(f, t0 + t) for f, t in pan[:4]])                   # 0.4 s of motion: shorter than the dwell
    assert det.activity == Activity.STATIONARY
    run(det, [(f, t0 + t) for f, t in pan[4:]])
    assert det.activity == Activity.TURNING


def test_frame_size_change_and_gray_frames():
    det = ActivityDetector()
    for i in range(10):
        det.update(texture(h=240, w=320, seed=0), i * 0.1)
    for i in range(10, 30):
        g = cv2.cvtColor(texture(h=480, w=640, seed=0), cv2.COLOR_BGR2GRAY)
        det.update(g, i * 0.1)
    assert det.activity == Activity.STATIONARY


def test_missing_frame_keeps_frame_source_and_time_reset():
    det = ActivityDetector()
    frames = static_seq(30)
    for i, (f, t) in enumerate(frames):
        det.update(None if i % 7 == 3 else f, t, tracks=[trk(9, (0.1 + 0.03 * i, 0.2, 0.3 + 0.03 * i, 0.5), t=t)])
    assert det.activity == Activity.STATIONARY and det.state.source == "frame"   # moving boxes ignored
    det.update(frames[0][0], 0.0)                                  # the video restarted
    assert det.activity == Activity.UNKNOWN and det.state.samples == 0


def test_track_based_fallback_without_frames():
    # static tracks: STATIONARY
    det = ActivityDetector()
    for i in range(30):
        t = i * 0.2
        det.update(None, t, tracks=[trk(1, (0.40, 0.2, 0.60, 0.9), label="person", t=t), trk(2, (0.1, 0.5, 0.2, 0.7), t=t)])
    assert det.activity == Activity.STATIONARY and det.state.source == "tracks"
    assert 0 < det.confidence <= 0.6                               # track-based evidence is discounted
    # every box drifts left consistently: TURNING
    det = ActivityDetector()
    for i in range(25):
        t, dx = i * 0.2, -0.06 * i                                 # 0.3 widths/s
        det.update(None, t, tracks=[trk(1, (0.5 + dx, 0.3, 0.6 + dx, 0.6), t=t), trk(2, (0.8 + dx, 0.4, 0.9 + dx, 0.8), t=t)])
    assert det.activity == Activity.TURNING
    # objects keep coming and going (new ids): WALKING
    det = ActivityDetector()
    for i in range(25):
        t = i * 0.2
        det.update(None, t, tracks=[trk(100 + i, (0.4, 0.4, 0.5, 0.6), t=t), trk(200 + i // 3, (0.1, 0.3, 0.2, 0.5), t=t)])
    assert det.activity == Activity.WALKING
    # remembered / missed tracks are not evidence; nothing visible keeps the state
    det = ActivityDetector()
    for i in range(10):
        det.update(None, i * 0.2, tracks=[trk(1, (0.4, 0.4, 0.5, 0.6), misses=2)])
    assert det.activity == Activity.UNKNOWN and det.state.samples == 0


def test_reset_and_as_dict():
    det = ActivityDetector()
    run(det, static_seq(25))
    assert det.activity == Activity.STATIONARY
    d = det.as_dict()
    assert d["activity"] == "stationary" and set(d) >= {"since", "motion", "shift", "energy", "confidence"}
    det.reset()
    assert det.activity == Activity.UNKNOWN and det.state.samples == 0


# ---- task adapter -----------------------------------------------------------------------------------------------
class FakeGate:
    """Duck-typed AnnouncementGate: set_mode_params validates against its config fields, like the real one."""

    def __init__(self, with_mute=False):
        fields = {"hold_static_s": 30.0, "background": True}
        if with_mute:
            fields["objects_muted"] = False
        self.gate_cfg = SimpleNamespace(**fields)
        self.calls, self.rebaselines = [], 0

    def set_mode_params(self, **kw):
        bad = [k for k in kw if not hasattr(self.gate_cfg, k)]
        if bad:
            raise ValueError(f"unknown {bad}")
        self.calls.append(kw)
        for k, v in kw.items():
            setattr(self.gate_cfg, k, v)

    def rebaseline(self):
        self.rebaselines += 1


def fake_policy():
    return SimpleNamespace(cfg=SimpleNamespace(verbosity="detailed"))


def test_adapter_modes_apply_gate_and_verbosity():
    gate, pol = FakeGate(), fake_policy()
    ad = TaskAdapter(gate=gate, policy=pol)
    assert ad.mode == "navigate" and gate.gate_cfg.hold_static_s == 20.0 and gate.gate_cfg.background is False
    assert pol.cfg.verbosity == "balanced"
    assert ad.step(Activity.WALKING, 1.0) == "navigate" and not ad.changed
    assert ad.step(Activity.STATIONARY, 2.0) == "scene" and ad.changed
    assert gate.gate_cfg.hold_static_s == 60.0 and gate.gate_cfg.background is True and pol.cfg.verbosity == "quiet"
    assert gate.rebaselines == 1                                   # what is static here becomes the background
    assert ad.step(Activity.READING, 3.0) == "read"
    assert gate.rebaselines == 1                                   # scene -> read: same place, no new baseline
    assert ad.should_mute_objects                                  # the fake gate has no objects_muted field
    assert ad.step(Activity.TURNING, 4.0) == "navigate" and not ad.should_mute_objects
    assert gate.gate_cfg.hold_static_s == 20.0 and pol.cfg.verbosity == "balanced"
    assert ad.step(Activity.STATIONARY, 5.0, has_goal=True) == "find"
    assert gate.gate_cfg.hold_static_s == 20.0 and gate.gate_cfg.background is False
    assert ad.step(Activity.UNKNOWN, 6.0) == "navigate"            # unknown is the conservative default
    assert [m for _, m, _ in ad.history] == ["navigate", "scene", "read", "navigate", "find", "navigate"]
    assert "navigate" in ad.describe()


def test_adapter_goal_fn_and_manual_override():
    goal = {"tags": []}
    gate, pol = FakeGate(), fake_policy()
    ad = TaskAdapter(gate=gate, policy=pol, goal_fn=lambda: bool(goal["tags"]))
    assert ad.step(Activity.STATIONARY, 1.0) == "scene"
    goal["tags"] = ["exit"]
    assert ad.step(Activity.STATIONARY, 2.0) == "find"
    assert ad.force_mode("read") == "read" and ad.mode == "read"   # applied at once
    assert ad.step(Activity.WALKING, 3.0, has_goal=False) == "read"
    assert "manual" in ad.describe()
    ad.force_mode("auto")
    assert ad.step(Activity.WALKING, 4.0, has_goal=False) == "navigate"
    seen = [ad.cycle_override() for _ in range(5)]
    assert seen == ["navigate", "scene", "read", "find", None]
    with pytest.raises(ValueError):
        ad.force_mode("dance")


def test_adapter_noop_without_gate_or_cfg():
    ad = TaskAdapter()                                             # gate=None, policy=None
    assert ad.step(Activity.STATIONARY, 1.0) == "scene" and ad.applied == {}
    ad = TaskAdapter(gate=None, policy=SimpleNamespace(name="threshold"))   # a policy without cfg
    assert ad.step(Activity.READING, 1.0) == "read" and ad.applied == {}
    ad.set_policy(SimpleNamespace(cfg=SimpleNamespace(verbosity="detailed")))
    assert ad.policy.cfg.verbosity == "quiet"                      # re-applied to a swapped-in scheduler
    # a gate without set_mode_params: only fields its config already has are set
    g = SimpleNamespace(gate_cfg=SimpleNamespace(hold_static_s=30.0))
    ad = TaskAdapter(gate=g)
    ad.step(Activity.STATIONARY, 1.0)
    assert g.gate_cfg.hold_static_s == 60.0 and not hasattr(g.gate_cfg, "background")


def _cue(kind="object", action="speak", severity=0.2, label="chair"):
    return Cue(t=0.0, track_id=1, label=label, kind=kind, message=label, score=1.0, policy="p", action=action,
               severity=severity)


def test_read_mode_filters_object_cues_unless_the_gate_mutes_them():
    ad = TaskAdapter(gate=FakeGate())
    cues = [_cue(), _cue(kind="text", label="text"), _cue(action="haptic"), _cue(severity=0.95)]
    assert ad.filter_cues(cues) == cues                            # navigate: unchanged
    ad.step(Activity.READING, 1.0)
    kept = ad.filter_cues(cues)
    assert [c.kind for c in kept] == ["text", "object", "object"] and cues[0] not in kept
    assert ad.filter_cues(None) == []
    gate = FakeGate(with_mute=True)
    ad = TaskAdapter(gate=gate)
    ad.step(Activity.READING, 1.0)
    assert gate.gate_cfg.objects_muted is True and not ad.should_mute_objects
    assert ad.filter_cues(cues) == cues
    ad.step(Activity.WALKING, 2.0)
    assert gate.gate_cfg.objects_muted is False


def test_adapter_with_the_real_gate():
    gate_mod = pytest.importorskip("seeforme.live.gate")
    from seeforme.cues.pcacs import PCACSPolicy
    pol = PCACSPolicy()
    gate = gate_mod.AnnouncementGate(pol)
    ad = TaskAdapter(gate=gate, policy=gate)                       # the gate forwards cfg to the scheduler
    ad.step(Activity.STATIONARY, 1.0)
    assert gate.gate_cfg.hold_static_s == 60.0 and gate.gate_cfg.background is True
    assert pol.cfg.verbosity == "quiet" and ad.applied.get("gate.rebaseline") is True
    ad.step(Activity.WALKING, 2.0)
    assert gate.gate_cfg.hold_static_s == 20.0 and gate.gate_cfg.background is False and pol.cfg.verbosity == "balanced"


def test_custom_mode_table():
    modes = dict(MODE_SETTINGS, scene=ModeSettings(hold_static_s=90.0, background=True, verbosity="quiet"))
    gate = FakeGate()
    ad = TaskAdapter(gate=gate, modes=modes)
    ad.step(Activity.STATIONARY, 1.0)
    assert gate.gate_cfg.hold_static_s == 90.0
    assert MODE_SETTINGS["scene"].hold_static_s == 60.0            # the default table is not modified


# ---- safety: leaving a still state, stalls, read mode ------------------------------------------------------------
def _settle_still(det, seconds=4.0, fps=10.0):
    seq = static_seq(int(seconds * fps), fps=fps)
    run(det, seq)
    assert det.activity == Activity.STATIONARY
    return seq[-1][1], seq[-1][0]


def test_slow_walk_after_standing_leaves_the_still_state():
    """Stood still, then walks slowly (zoom 2% per 0.1 s, flow under the walking threshold): must not stay STATIONARY."""
    det = ActivityDetector()
    t0, last = _settle_still(det)
    h, w = last.shape[:2]
    out, scale = [], 1.0
    for i in range(1, 41):
        scale *= 1.02
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        out.append((det.update(cv2.warpAffine(last, M, (w, h), borderMode=cv2.BORDER_REFLECT), t0 + 0.1 * i), t0 + 0.1 * i))
    left = next((t for a, t in out if a != Activity.STATIONARY), None)
    assert left is not None and left - t0 <= 2.5


def test_camera_stall_forgets_the_still_state():
    det = ActivityDetector()
    t0, _ = _settle_still(det)
    for i in range(1, 31):
        a = det.update(None, t0 + 0.1 * i)
    assert a != Activity.STATIONARY


def test_read_mode_keeps_approaching_objects_whatever_their_class():
    ad = TaskAdapter()
    ad.force_mode("read")
    mk = lambda label, sev, ttc: Cue(0.0, 1, label, "object", f"{label}.", 1.0, "x", severity=sev, ttc_s=ttc)
    kept = ad.filter_cues([mk("curb", 0.8, 1.2), mk("person", 0.5, 1.0), mk("chair", 0.7, None), mk("bench", 0.7, float("inf"))])
    assert [c.label for c in kept] == ["curb", "person"]


def test_adapter_clock_starts_at_the_first_step():
    ad = TaskAdapter()
    ad.step(Activity.WALKING, 1234.5)
    assert ad.since == 1234.5 and ad.history[0][0] == 1234.5
