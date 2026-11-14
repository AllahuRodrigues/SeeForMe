"""Model pre-labels in the review store: kept apart from human verdicts, validated on import, comparable afterwards."""
import pytest

from seeforme.egoblind.review import (disagreements, latest_verdicts, make_verdict, model_verdict_records, packet_frames,
                                      rater_kind)

EVENTS = [{"sample_id": "SFM-001", "key": "7:1", "claim": "Car, ahead."},
          {"sample_id": "SFM-002", "key": "8:9001", "claim": ""}]
GOOD = [{"sample_id": "SFM-001", "usefulness": "proactive_warn", "claim": "correct", "evidence": "yes", "confidence": "high", "note": "close"},
        {"sample_id": "SFM-002", "usefulness": "silent"}]


def test_model_labels_become_model_records_under_their_own_reviewer():
    recs = model_verdict_records(GOOD, EVENTS, reviewer="m1")
    assert [r["key"] for r in recs] == ["7:1", "8:9001"]
    assert all(r["verification"] == "model" and r["reviewer"] == "m1" for r in recs)
    assert recs[0]["claim"] == "correct" and "claim" not in recs[1]


def test_model_records_never_count_as_human_verdicts():
    recs = model_verdict_records(GOOD, EVENTS, reviewer="m1")
    assert latest_verdicts(recs, reviewer="m1") == {}
    assert set(latest_verdicts(recs, reviewer="m1", human_only=False)) == {"7:1", "8:9001"}


def test_a_reviewer_id_with_human_verdicts_cannot_receive_model_labels():
    human = [make_verdict("7:1", "silent", "ar", claim="correct")]
    with pytest.raises(ValueError, match="human"):
        model_verdict_records(GOOD, EVENTS, reviewer="ar", existing=human)


def test_rater_kind_reports_human_model_or_nothing_and_rejects_a_mix():
    human, model = make_verdict("7:1", "silent", "ar"), make_verdict("7:1", "silent", "m1", verification="model")
    assert rater_kind([human, model], "ar") == "human" and rater_kind([human, model], "m1") == "model"
    assert rater_kind([human], "nobody") is None
    with pytest.raises(ValueError):
        rater_kind([human, make_verdict("8:1", "silent", "ar", verification="model")], "ar")


@pytest.mark.parametrize("bad, msg", [
    ([{"sample_id": "SFM-009", "usefulness": "silent"}], "unknown"),
    ([GOOD[1], GOOD[1]], "twice"),
    ([{"sample_id": "SFM-001", "usefulness": "proactive_warn"}], "sentence"),
    ([{"sample_id": "SFM-002", "usefulness": "silent", "claim": "correct"}], "no sentence"),
    ([{"sample_id": "SFM-002", "usefulness": "wrong_object"}], "usefulness"),
    ([{"sample_id": "SFM-001", "usefulness": "silent", "claim": "maybe"}], "claim"),
])
def test_import_rejects_labels_that_do_not_fit_the_queue(bad, msg):
    with pytest.raises(ValueError, match=msg):
        model_verdict_records(bad, EVENTS, reviewer="m1")


def test_disagreements_name_what_differs_and_skip_agreement():
    a = {"1:1": make_verdict("1:1", "proactive_warn", "m1", verification="model", claim="correct"),
         "2:1": make_verdict("2:1", "silent", "m1", verification="model", claim="wrong"),
         "3:1": make_verdict("3:1", "on_request", "m1", verification="model"),
         "4:1": make_verdict("4:1", "uncertain", "m1", verification="model"),
         "5:1": make_verdict("5:1", "silent", "m1", verification="model")}
    b = {"1:1": make_verdict("1:1", "silent", "ar", claim="correct"),
         "2:1": make_verdict("2:1", "silent", "ar", claim="correct"),
         "3:1": make_verdict("3:1", "silent", "ar"),
         "4:1": make_verdict("4:1", "silent", "ar"),
         "5:1": make_verdict("5:1", "silent", "ar"),
         "6:1": make_verdict("6:1", "silent", "ar")}
    d = {x["key"]: x["differs"] for x in disagreements(a, b)}
    assert d == {"1:1": ["speak_or_not", "category"], "2:1": ["claim"], "3:1": ["category"], "4:1": ["one_unsure", "category"]}


def test_packet_frames_end_at_the_decision_and_spread_backwards():
    times = [round(0.75 * i, 2) for i in range(10)]
    assert packet_frames(times, t_q=5.0, n=4, gap=2) == [0, 2, 4, 6]       # nothing after 5.0 s; the last tile is the decision
    assert packet_frames(times[:3], t_q=9.0, n=4, gap=2) == [0, 1, 2]
    assert packet_frames(times, t_q=-1.0) == []
