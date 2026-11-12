"""M3: blind human review of the weak EgoBlind events (issue #5)."""
import pytest

from seeforme.egoblind.review import (VERDICTS, blind_row, cohen_kappa, event_key, latest_verdicts, merge_verdicts,
                                      make_verdict, sample_second_rater)

ROW = {"video": "egoblind_2", "video_id": "2", "obj_id": "1", "t_q": "6.99", "window": "5.6-7.5", "question": "Is it safe to cross?",
       "labels": "car,truck", "human_verdict": "", "human_note": "", "bin:pcacs": "spoken_timely", "max_score:pcacs": "1.27"}


def test_blind_row_hides_every_model_column():
    b = blind_row(ROW)
    assert not any(k.startswith(("bin:", "max_score:")) for k in b)
    assert b["question"] == ROW["question"] and b["key"] == event_key(ROW) == "2:1"


def test_verdict_validation_and_provenance():
    v = make_verdict("2:1", "speak", reviewer="ar", window=(5.0, 7.5), note="car approaching")
    assert v["verification"] == "human" and v["verdict"] == "speak" and v["window"] == [5.0, 7.5]
    with pytest.raises(ValueError):
        make_verdict("2:1", "maybe", reviewer="ar")
    with pytest.raises(ValueError):
        make_verdict("2:1", "speak", reviewer="")
    with pytest.raises(ValueError):
        make_verdict("2:1", "speak", reviewer="ar", window=(8.0, 7.0))
    assert set(VERDICTS) == {"speak", "silent", "wrong_object", "unclear"}


def test_latest_verdict_per_reviewer_wins_and_merge_fills_csv_columns():
    vs = [make_verdict("2:1", "silent", reviewer="ar"), make_verdict("2:1", "speak", reviewer="ar", note="changed mind"),
          make_verdict("2:1", "silent", reviewer="r2")]
    latest = latest_verdicts(vs, reviewer="ar")
    assert latest["2:1"]["verdict"] == "speak"
    rows = merge_verdicts([dict(ROW)], latest)
    assert rows[0]["human_verdict"] == "speak" and rows[0]["human_note"] == "changed mind"


def test_model_prelabels_never_count_as_human():
    v = make_verdict("2:1", "speak", reviewer="model:claude", verification="model")
    assert latest_verdicts([v], reviewer="model:claude", human_only=True) == {}


def test_cohen_kappa():
    a = {"1": "speak", "2": "speak", "3": "silent", "4": "silent"}
    assert cohen_kappa(a, dict(a)) == pytest.approx(1.0)
    b = {"1": "speak", "2": "silent", "3": "speak", "4": "silent"}
    assert cohen_kappa(a, b) == pytest.approx(0.0)
    assert cohen_kappa(a, {}) is None


def test_second_rater_sample_is_seeded_and_sized():
    keys = [str(i) for i in range(100)]
    s1, s2 = sample_second_rater(keys, 0.2, seed=0), sample_second_rater(keys, 0.2, seed=0)
    assert s1 == s2 and len(s1) == 20 and set(s1) <= set(keys)
