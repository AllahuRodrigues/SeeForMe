from seeforme.geometry import CameraModel, TrackDynamics
from seeforme.types import Detection, Track


def test_projection_roundtrip_distance():
    cam = CameraModel()
    b = cam.project_box(0.0, 3.0, 0.9, 0.5)
    assert b is not None and 0 < b[0] < b[2] < 1 and 0 < b[1] < b[3] < 1
    # bbox-height inverse is within ~25% (pitch and the ground-contact point make it inexact)
    assert abs(cam.distance_from_bbox(b, "chair") - 3.0) / 3.0 < 0.3
    assert cam.project_box(0.0, 0.0, 0.9, 0.5) is None


def test_dynamics_ttc_and_path_probability():
    cam = CameraModel(); dyn = TrackDynamics(cam)
    tr = Track(1, "chair", "object", (0.4, 0.4, 0.6, 0.9), 0.9, 0, 0, confirmed=True)
    for i, z in enumerate([5.0, 4.5, 4.0, 3.5, 3.0, 2.5]):
        b = cam.project_box(0.1, z, 0.9, 0.5); tr.bbox = b
        dyn.update(tr, Detection("chair", 0.9, b, distance_m=z, depth_conf=0.9), i * 0.5)
    assert tr.rel_speed_mps < -0.7            # approaching at ~1 m/s
    assert 1.0 < tr.ttc_s < 3.5
    assert tr.path_prob > 0.6                 # straight ahead -> in the corridor
    off = Track(2, "chair", "object", (0.4, 0.4, 0.6, 0.9), 0.9, 0, 0, confirmed=True)
    for i, z in enumerate([5.0, 4.5, 4.0, 3.5, 3.0, 2.5]):
        b = cam.project_box(2.0, z, 0.9, 0.5); off.bbox = b
        dyn.update(off, Detection("chair", 0.9, b, distance_m=z, depth_conf=0.9), i * 0.5)
    assert off.path_prob < 0.25               # two metres to the side


def test_crossing_object_is_flagged():
    cam = CameraModel(); dyn = TrackDynamics(cam)
    tr = Track(3, "bicycle", "object", (0.4, 0.4, 0.6, 0.9), 0.9, 0, 0, confirmed=True)
    for i, (z, x) in enumerate([(3.0, 3.0), (2.9, 2.2), (2.8, 1.4), (2.7, 0.8), (2.6, 0.3)]):
        b = cam.project_box(x, z, 1.0, 0.6); tr.bbox = b
        dyn.update(tr, Detection("bicycle", 0.9, b, distance_m=z, depth_conf=0.9), i * 0.5)
    assert tr.lateral_speed_mps < -0.8 and tr.path_prob > 0.7
