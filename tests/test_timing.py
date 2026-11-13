"""D1: onset-timing metrics in the units of EgoVoice and ProactEgoMem, on EgoBlind-style events."""
import pytest

from seeforme.eval.timing import TimedEvent, match_onsets, timing_metrics

EV = [TimedEvent(eid="a", sid=1, ref=10.0), TimedEvent(eid="b", sid=2, ref=20.0), TimedEvent(eid="c", sid=3, ref=30.0)]


def test_one_to_one_matching_by_source_and_window():
    cues = [(9.6, 1), (9.9, 1), (21.0, 2), (40.0, 3)]
    m = match_onsets(EV, cues, early=2.5, late=1.5)
    assert {e: round(d, 2) for e, d in m.items()} == {"a": -0.1, "b": 1.0}     # closest of the two cues about a; c too late


def test_cues_about_another_object_never_match():
    assert match_onsets(EV, [(10.0, 2)], early=2.5, late=1.5) == {}


def test_metrics():
    cues = [(9.9, 1), (21.0, 2), (40.0, 3), (55.0, None)]
    r = timing_metrics(EV, cues, early=2.5, late=1.5, taus=(0.5, 1.0), n_scored_cues=4)
    assert r["matched"] == 2 and r["recall"] == pytest.approx(2 / 3)
    assert r["start@0.5"] == pytest.approx(0.5) and r["start@1"] == pytest.approx(1.0)          # of matched
    assert r["start@0.5_all"] == pytest.approx(1 / 3)                                          # of all events
    assert r["mean_abs_delta"] == pytest.approx(0.55) and r["mean_signed_delta"] == pytest.approx(0.45)
    assert r["precision"] == pytest.approx(0.5) and r["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_no_events_or_cues():
    r = timing_metrics([], [], early=2.5, late=1.5)
    assert r["matched"] == 0 and r["recall"] is None
