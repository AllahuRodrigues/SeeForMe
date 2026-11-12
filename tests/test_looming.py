"""Image-scale time to contact (looming): under perspective, box height h = f*H/Z, so h/(dh/dt) = Z/(-dZ/dt).
Needs no metric depth and no estimate of the user's speed (CPSC 480 lecture 3, magnification inversely
proportional to depth)."""
import math

from seeforme.types import Track


def _track(heights_by_t):
    tr = Track(track_id=1, label="door", kind="object", bbox=(0.4, 0.4, 0.6, 0.6), confidence=0.9, first_seen=0.0, last_seen=0.0)
    for t, h in heights_by_t:
        tr.bbox_history.append((t, (0.45, 0.5 - h / 2, 0.55, 0.5 + h / 2)))
    return tr


def test_constant_approach_recovers_time_to_contact():
    # object 10 m away, user walks at 1.25 m/s: true time to contact at t=1.0 is (10 - 1.25)/1.25 = 7.0 s
    f_h = 0.5   # normalized focal * object height
    hs = [(t, f_h / (10.0 - 1.25 * t)) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    tau = _track(hs).looming_ttc()
    assert abs(tau - 7.0) / 7.0 < 0.10


def test_receding_or_static_object_has_no_contact():
    assert math.isinf(_track([(t, 0.2) for t in (0.0, 0.2, 0.4, 0.6)]).looming_ttc())
    assert math.isinf(_track([(t, 0.2 - 0.02 * t) for t in (0.0, 0.2, 0.4, 0.6)]).looming_ttc())


def test_too_little_history_is_unknown():
    assert math.isinf(_track([(0.0, 0.2), (0.2, 0.21)]).looming_ttc())


def test_only_the_recent_window_counts():
    old = [(t, 0.1) for t in (0.0, 0.2, 0.4, 0.6, 0.8)]            # static long ago
    new = [(t, 0.5 / (10.0 - 1.25 * (t - 3.0))) for t in (3.0, 3.2, 3.4, 3.6, 3.8, 4.0)]
    tau = _track(old + new).looming_ttc(window_s=1.2)
    assert abs(tau - 7.0) / 7.0 < 0.10
