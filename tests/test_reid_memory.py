"""M2: long-term appearance memory. An object that left the frame (beyond the tracker's short memory) is recognised
by its CLIP embedding when it comes back and keeps its 'already announced' state, so it is not announced as new."""
import numpy as np

from seeforme.config import TrackerConfig
from seeforme.perception.appearance import crop_boxes
from seeforme.tracking.reid import ReIDGallery
from seeforme.tracking.tracker import IoUTracker
from seeforme.types import Detection


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


A = _unit([1, 0, 0, 0])
A2 = _unit([0.97, 0.2, 0, 0])      # same object, slightly different view (cos ~ 0.98)
B = _unit([0, 1, 0, 0])            # a different object of the same class


def _det(emb, box=(0.4, 0.4, 0.6, 0.8), label="bench"):
    return Detection(label=label, confidence=0.9, bbox=box, emb=emb)


def _run(tracker, frames):
    out = []
    for t, dets in frames:
        out = tracker.update(dets, t)
    return out


def test_gallery_matches_same_object_not_different_one():
    g = ReIDGallery(horizon_s=120, threshold=0.85)
    g.add(label="bench", kind="object", emb=A, last_seen=10.0, last_announced=8.0, announce_count=1, track_id=3)
    assert g.match("bench", "object", B, t=30.0) is None
    m = g.match("bench", "object", A2, t=30.0)
    assert m is not None and m.track_id == 3 and m.last_announced == 8.0
    assert g.match("bench", "object", A2, t=31.0) is None           # consumed: one predecessor per object


def test_gallery_respects_label_and_horizon():
    g = ReIDGallery(horizon_s=60, threshold=0.85)
    g.add(label="bench", kind="object", emb=A, last_seen=10.0, last_announced=8.0, announce_count=1, track_id=3)
    assert g.match("chair", "object", A, t=20.0) is None
    assert g.match("bench", "object", A, t=80.0) is None             # older than the horizon


def _scene(emb_back):
    # object visible 0-3 s, gone 3-40 s (far beyond max_age and the 3 s ledger), back at 40 s
    frames = [(i * 0.2, [_det(A)]) for i in range(16)]
    frames += [(3.2 + i * 0.2, []) for i in range(20)]
    frames += [(40.0 + i * 0.2, [_det(emb_back)]) for i in range(5)]
    return frames


def _announce_first_track(tracker, frames, t_announce=1.0):
    for t, dets in frames:
        tracks = tracker.update(dets, t)
        if abs(t - t_announce) < 1e-6:
            for trk in tracks:
                trk.last_announced, trk.announce_count = t, 1
    return tracker.active_tracks()


def test_tracker_reidentifies_announced_object_after_long_absence():
    trk = IoUTracker(TrackerConfig(reid_horizon_s=120.0, reid_threshold=0.85))
    tracks = _announce_first_track(trk, _scene(A2))
    assert len(tracks) == 1
    t = tracks[0]
    assert t.reid_of is not None and t.last_announced == 1.0 and t.announce_count == 1


def test_different_object_is_new():
    trk = IoUTracker(TrackerConfig(reid_horizon_s=120.0, reid_threshold=0.85))
    t = _announce_first_track(trk, _scene(B))[0]
    assert t.reid_of is None and t.last_announced is None


def test_reid_off_by_default():
    trk = IoUTracker(TrackerConfig())
    t = _announce_first_track(trk, _scene(A))[0]
    assert t.reid_of is None and t.last_announced is None


def test_no_embedding_means_no_reid():
    trk = IoUTracker(TrackerConfig(reid_horizon_s=120.0))
    frames = [(t, [Detection(label=d.label, confidence=d.confidence, bbox=d.bbox)]) if d else (t, []) for t, ds in _scene(A) for d in (ds[0] if ds else None,)]
    t = _announce_first_track(trk, frames)[0]
    assert t.reid_of is None


def test_crop_boxes_clamps_and_pads():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crops = crop_boxes(frame, [(0.0, 0.0, 0.5, 0.5), (0.99, 0.99, 1.0, 1.0)])
    assert crops[0].shape[0] <= 100 and crops[0].shape[1] <= 200 and crops[0].shape[1] > 100   # padded, clamped
    assert min(crops[1].shape[:2]) >= 1 and crops[1].shape[2] == 3


def test_cache_carries_embeddings_into_detections():
    from tests.test_egoblind import synthetic_scenario
    sc, cache = synthetic_scenario()
    emb = np.random.default_rng(0).normal(size=(cache.n_detections, 8)).astype(np.float32)
    has = np.ones(cache.n_detections, dtype=bool); has[0] = False
    cache.attach_embeddings(emb, has)
    dets = [d for _, ds in cache.frames(sc.speak_events()) for d in ds]
    assert dets[0].emb is None and all(d.emb is not None for d in dets[1:])
    import pytest
    with pytest.raises(ValueError):
        cache.attach_embeddings(emb[:-1])


def test_tracker_logs_each_reid_match_for_audit():
    trk = IoUTracker(TrackerConfig(reid_horizon_s=120.0, reid_threshold=0.85))
    frames = [(t, [Detection(label=d.label, confidence=d.confidence, bbox=d.bbox, emb=d.emb, source_id=7)] if ds else [])
              for t, ds in _scene(A2) for d in (ds[0] if ds else None,)]
    _announce_first_track(trk, frames)
    assert len(trk.reid_log) == 1
    m = trk.reid_log[0]
    assert m["label"] == "bench" and m["old_sid"] == 7 and m["new_sid"] == 7
    assert 0.85 <= m["cos"] <= 1.0 and m["gap_s"] > 30
    trk.reset()
    assert trk.reid_log == []
