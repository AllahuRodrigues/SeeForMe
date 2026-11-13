"""Event-failure funnel: every should-speak event is attributed to the first pipeline stage that lost it."""
from seeforme.eval.funnel import STAGES, EventTrace, classify, funnel_counts


def tr(**kw):
    base = dict(window=(5.0, 9.0), det=[], track=[], cand=[], cand_legal=[], cue=[])
    base.update(kw)
    return EventTrace(**base)


def test_stage_order():
    assert STAGES == ("perception", "tracking", "candidate", "scheduler", "timing", "delivered")


def test_never_detected_is_a_perception_miss():
    assert classify(tr()) == "perception"


def test_detected_but_never_tracked():
    assert classify(tr(det=[6.0])) == "tracking"


def test_tracked_but_never_a_candidate():
    assert classify(tr(det=[6.0], track=[6.2])) == "candidate"


def test_candidate_but_never_spoken_is_scheduler():
    assert classify(tr(det=[6.0], track=[6.2], cand=[6.5], cand_legal=[6.5])) == "scheduler"


def test_spoken_outside_window_is_timing():
    assert classify(tr(det=[6.0], track=[6.2], cand=[6.5], cand_legal=[6.5], cue=[9.5])) == "timing"


def test_spoken_inside_window_is_delivered():
    assert classify(tr(det=[6.0], track=[6.2], cand=[6.5], cand_legal=[6.5], cue=[7.0])) == "delivered"


def test_only_in_window_evidence_counts_for_stages_before_scheduling():
    # seen only after the window closed: the pipeline could not have helped in time
    assert classify(tr(det=[9.6], track=[9.8], cand=[9.9], cand_legal=[9.9], cue=[9.9])) == "perception"


def test_candidate_never_legal_is_a_scheduler_miss_with_blocking():
    t = tr(det=[6.0], track=[6.2], cand=[6.5], cand_legal=[])
    assert classify(t) == "scheduler" and t.blocked


def test_funnel_counts_are_monotone():
    traces = [tr(), tr(det=[6.0]), tr(det=[6.0], track=[6.2], cand=[6.5], cand_legal=[6.5], cue=[7.0])]
    f = funnel_counts(traces)
    assert f["N"] == 3 and f["P"] == 2 and f["T"] == 1 and f["C"] == 1 and f["S"] == 1 and f["D"] == 1
    assert f["N"] >= f["P"] >= f["T"] >= f["C"] >= f["S"] >= f["D"]
