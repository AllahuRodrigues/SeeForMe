from seeforme.config import TrackerConfig
from seeforme.tracking.tracker import IoUTracker
from seeforme.types import Detection


def det(label, x, conf=0.9, w=0.2, kind="object", text=None, src=None):
    return Detection(label=label, confidence=conf, bbox=(x, 0.4, x + w, 0.7), kind=kind, text=text, source_id=src)


def test_track_persists_across_frames_and_confirms():
    tr = IoUTracker(TrackerConfig(n_init=2))
    assert tr.update([det("chair", 0.4)], 0.0) == []          # not confirmed yet
    tracks = tr.update([det("chair", 0.41)], 0.1)
    assert len(tracks) == 1 and tracks[0].track_id == 1 and tracks[0].hits == 2


def test_track_survives_misses_then_dies():
    tr = IoUTracker(TrackerConfig(n_init=1, max_age=2))
    tr.update([det("chair", 0.4)], 0.0)
    for i in range(2):
        assert tr.update([], 0.1 * (i + 1)) == []
    assert len(tr.tracks) == 1
    tr.update([], 0.5)
    assert len(tr.tracks) == 0


def test_two_objects_get_distinct_ids_and_source_votes():
    tr = IoUTracker(TrackerConfig(n_init=1))
    tracks = tr.update([det("chair", 0.1, src=1), det("person", 0.7, src=2)], 0.0)
    assert {t.source_id for t in tracks} == {1, 2}
    assert len({t.track_id for t in tracks}) == 2


def test_text_tracks_match_by_string_similarity():
    tr = IoUTracker(TrackerConfig(n_init=1))
    tr.update([det("text", 0.5, kind="text", text="EXIT")], 0.0)
    tracks = tr.update([det("text", 0.52, kind="text", text="EX1T")], 0.1)
    assert len(tracks) == 1 and tracks[0].hits == 2
