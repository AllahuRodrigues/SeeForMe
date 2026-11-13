"""M3: blind human review of the weak EgoBlind events (issue #5)."""
import pytest

from seeforme.egoblind.review import (CLAIMS, USEFULNESS, VERDICTS, agreement, blind_row, cohen_kappa, consensus_targets, event_key,
                                      latest_verdicts, merge_verdicts, make_verdict, proactive_target, sample_second_rater, two_axis)

ROW = {"video": "egoblind_2", "video_id": "2", "obj_id": "1", "t_q": "6.99", "window": "5.6-7.5", "question": "Is it safe to cross?",
       "labels": "car,truck", "human_verdict": "", "human_note": "", "bin:pcacs": "spoken_timely", "max_score:pcacs": "1.27"}


def test_blind_row_hides_every_model_column():
    b = blind_row(ROW)
    assert not any(k.startswith(("bin:", "max_score:")) for k in b)
    assert b["question"] == ROW["question"] and b["key"] == event_key(ROW) == "2:1"


def test_verdict_validation_and_provenance():
    v = make_verdict("2:1", "proactive_warn", reviewer="ar", window=(5.0, 7.5), note="car approaching")
    assert v["verification"] == "human" and v["verdict"] == "proactive_warn" and v["window"] == [5.0, 7.5]
    with pytest.raises(ValueError):
        make_verdict("2:1", "maybe", reviewer="ar")
    with pytest.raises(ValueError):
        make_verdict("2:1", "speak", reviewer="")
    with pytest.raises(ValueError):
        make_verdict("2:1", "speak", reviewer="ar", window=(8.0, 7.0))
    assert set(VERDICTS) == {"proactive_warn", "proactive_clear", "on_request", "silent", "uncertain", "wrong_object"}


def test_latest_verdict_per_reviewer_wins_and_merge_fills_csv_columns():
    vs = [make_verdict("2:1", "silent", reviewer="ar"), make_verdict("2:1", "speak", reviewer="ar", note="changed mind"),
          make_verdict("2:1", "silent", reviewer="r2")]
    latest = latest_verdicts(vs, reviewer="ar")
    assert latest["2:1"]["verdict"] == "proactive_warn"
    rows = merge_verdicts([dict(ROW)], latest)
    assert rows[0]["human_verdict"] == "proactive_warn" and rows[0]["human_note"] == "changed mind"


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


def test_derived_answer_is_hidden_from_the_reviewer():
    row = dict(ROW, **{"derived:label": "silent"})
    assert "derived:label" not in blind_row(row)


def test_audit_score_confusion_and_rates():
    from seeforme.egoblind.review import audit_score
    derived = {"a": "speak", "b": "speak", "c": "silent", "d": "silent", "e": "silent"}
    verdicts = {"a": "speak", "b": "silent", "c": "silent", "d": "speak", "e": "unclear"}
    s = audit_score(derived, verdicts)
    assert s["n"] == 5 and s["n_decided"] == 4
    assert s["speak_confirmed"] == 0.5 and s["silent_confirmed"] == 0.5     # of decided, per derived class
    assert s["confusion"]["silent"]["proactive_warn"] == 1 and s["uncertain"] == 1


def test_confirm_clear_counts_as_not_a_warning_in_the_audit():
    from seeforme.egoblind.review import audit_score
    derived = {"a": "speak", "b": "silent", "c": "silent"}
    s = audit_score(derived, {"a": "speak", "b": "confirm_clear", "c": "silent"})
    assert s["speak_confirmed"] == 1.0 and s["silent_confirmed"] == 1.0 and s["confirm_clear_on_quiet"] == 0.5


def test_legacy_verdicts_map_onto_rubric_v3():
    from seeforme.egoblind.review import canonical
    assert canonical("speak") == "proactive_warn" and canonical("confirm_clear") == "proactive_clear"
    assert canonical("unclear") == "uncertain" and canonical("silent") == "silent"


def test_uncertain_is_never_scored_as_silent():
    from seeforme.egoblind.review import audit_score
    s = audit_score({"a": "silent", "b": "silent"}, {"a": "uncertain", "b": "on_request"})
    assert s["n_decided"] == 1 and s["silent_confirmed"] == 1.0          # on_request is not a proactive warning


def test_decision_time_frames_stop_at_the_question():
    from seeforme.egoblind.review import decision_time_frames
    assert decision_time_frames([4.0, 5.0, 6.0, 7.0, 8.0], t_q=6.5) == [0, 1, 2]


# rubric v4: usefulness and claim correctness are separate labels
def test_usefulness_and_claim_are_separate_labels():
    v = make_verdict("2:1", "proactive_warn", reviewer="ar", claim="wrong", evidence="yes", confidence="high")
    assert two_axis(v) == ("proactive_warn", "wrong")
    assert "wrong_object" not in USEFULNESS and set(CLAIMS) == {"correct", "partially_correct", "wrong", "not_assessable"}
    with pytest.raises(ValueError):
        make_verdict("2:1", "silent", reviewer="ar", claim="maybe")
    with pytest.raises(ValueError):
        make_verdict("2:1", "silent", reviewer="ar", evidence="often")
    with pytest.raises(ValueError):
        make_verdict("2:1", "silent", reviewer="ar", confidence="very")


def test_legacy_wrong_object_splits_into_useful_but_wrong_claim():
    assert two_axis(make_verdict("2:1", "wrong_object", reviewer="ar")) == ("proactive_warn", "wrong")
    assert two_axis(make_verdict("2:1", "speak", reviewer="ar")) == ("proactive_warn", None)


def test_uncertain_is_never_a_negative():
    assert proactive_target("proactive_warn") == 1 and proactive_target("proactive_clear") == 1
    assert proactive_target("on_request") == 0 and proactive_target("silent") == 0
    assert proactive_target("uncertain") is None


def test_consensus_masks_disagreement_and_uncertainty():
    mk = lambda k, v, r, **kw: make_verdict(k, v, reviewer=r, **kw)
    a = {"1": mk("1", "proactive_warn", "r1", claim="correct"), "2": mk("2", "silent", "r1"), "3": mk("3", "proactive_warn", "r1"),
         "4": mk("4", "uncertain", "r1"), "5": mk("5", "on_request", "r1"), "6": mk("6", "silent", "r1")}
    b = {"1": mk("1", "proactive_clear", "r2", claim="correct"), "2": mk("2", "on_request", "r2"), "3": mk("3", "silent", "r2"),
         "4": mk("4", "silent", "r2"), "5": mk("5", "on_request", "r2")}
    c = consensus_targets(a, b)
    assert c["1"] == {"target": 1, "status": "agree"}              # both proactive (warn vs clear differ in content, not in whether)
    assert c["2"] == {"target": 0, "status": "agree"}              # both not-proactive
    assert c["3"] == {"target": None, "status": "disagree"}
    assert c["4"] == {"target": None, "status": "uncertain"}       # one rater unsure: masked, not negative
    assert c["5"]["target"] == 0
    assert c["6"] == {"target": None, "status": "single"}          # one rater only: not ground truth yet


def test_agreement_reports_both_axes_and_what_was_excluded():
    mk = lambda k, v, r, **kw: make_verdict(k, v, reviewer=r, **kw)
    a = {str(i): mk(str(i), "proactive_warn" if i < 5 else "silent", "r1", claim="correct" if i % 2 else "wrong") for i in range(10)}
    b = {str(i): mk(str(i), "proactive_warn" if i < 5 else "silent", "r2", claim="correct" if i % 2 else "wrong") for i in range(10)}
    b["9"] = mk("9", "uncertain", "r2", claim="not_assessable")
    r = agreement(a, b, n_boot=200, seed=0)
    assert r["n_overlap"] == 10 and r["binary"]["n"] == 9 and r["binary"]["excluded_uncertain"] == 1
    assert r["binary"]["kappa"] == pytest.approx(1.0)
    assert r["usefulness"]["kappa"] < 1.0                          # the uncertain item disagrees on the 5-way axis
    assert r["claim"]["n"] == 10 and 0.0 < r["claim"]["raw"] < 1.0
    lo, hi = r["binary"]["kappa_ci"]
    assert lo <= r["binary"]["kappa"] <= hi
