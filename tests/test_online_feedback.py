"""M1/S1: one-sided feedback. Speaking may earn a tap; silence is labelled only when the user later asks."""
from seeforme.online.feedback import Feedback, FeedbackModel

from tests.test_egoblind import synthetic_scenario


def _scenario():
    sc, _ = synthetic_scenario()
    car = next(e for e in sc.speak_events() if e.label == "car")
    quiet = sc.quiet_events()[0]
    return sc, car, quiet


def test_useful_cue_earns_positive_tap():                                   # AC-1
    sc, car, _ = _scenario()
    fb = FeedbackModel(sc)
    f = fb.on_cue(car.ideal_s, source_id=car.obj_id)
    assert f == Feedback(t=car.ideal_s, label=1, kind="tap_useful", source_id=car.obj_id, t0=car.ideal_s, t1=car.ideal_s)


def test_repeat_cue_about_announced_event_is_not_useful():
    sc, car, _ = _scenario()
    fb = FeedbackModel(sc)
    assert fb.on_cue(car.ideal_s, source_id=car.obj_id).label == 1
    again = fb.on_cue(car.ideal_s + 0.2, source_id=car.obj_id)
    assert again is not None and again.label == -1


def test_off_event_cue_near_question_is_negative_elsewhere_unlabelled():   # AC-2
    sc, car, _ = _scenario()
    fb = FeedbackModel(sc)
    near = fb.on_cue(car.detail["t_q"] + 3.0, source_id=None)
    assert near is not None and near.label == -1 and near.kind == "tap_not_useful"
    assert fb.on_cue(29.5, source_id=None) is None                              # > 6 s from every labelled moment


def test_missed_event_becomes_positive_when_user_asks():                   # AC-3
    sc, car, _ = _scenario()
    fb = FeedbackModel(sc)
    assert fb.advance(car.detail["t_q"] - 0.1) == []
    out = fb.advance(car.detail["t_q"])
    missed = [f for f in out if f.kind == "missed_question"]
    assert len(missed) == 1
    m = missed[0]
    assert m.label == 1 and m.source_id == car.obj_id and m.t0 == car.earliest_useful_s and m.t1 == car.detail["t_q"]
    later = fb.advance(car.detail["t_q"] + 5)
    assert all(f.source_id != car.obj_id for f in later)                         # each question fires once


def test_announced_event_produces_no_missed_label():
    sc, car, _ = _scenario()
    fb = FeedbackModel(sc)
    fb.on_cue(car.ideal_s, source_id=car.obj_id)
    out = fb.advance(car.detail["t_q"])
    assert not [f for f in out if f.kind == "missed_question"]


def test_quiet_window_yields_negative_for_all_candidates():                # AC-4
    sc, _, quiet = _scenario()
    fb = FeedbackModel(sc)
    out = fb.advance(quiet.detail["t_q"])
    neg = [f for f in out if f.kind == "quiet_window"]
    assert len(neg) == 1 and neg[0].label == -1 and neg[0].source_id is None
    assert (neg[0].t0, neg[0].t1) == (quiet.earliest_useful_s, quiet.latest_safe_s)


def test_silence_alone_gives_no_label():                                   # AC-5
    sc, car, _ = _scenario()
    fb = FeedbackModel(sc)
    assert fb.advance(car.earliest_useful_s - 0.5) == []


def test_tap_dropout_is_seeded():                                          # AC-6
    sc, car, _ = _scenario()
    a = [FeedbackModel(sc, p_tap=0.5, seed=7).on_cue(car.ideal_s, source_id=car.obj_id) for _ in range(5)]
    b = [FeedbackModel(sc, p_tap=0.5, seed=7).on_cue(car.ideal_s, source_id=car.obj_id) for _ in range(5)]
    assert a == b
    dropped = sum(FeedbackModel(sc, p_tap=0.5, seed=s).on_cue(car.ideal_s, source_id=car.obj_id) is None for s in range(200))
    assert 60 < dropped < 140
    assert FeedbackModel(sc, p_tap=0.0).advance(car.detail["t_q"])            # questions are never dropped
