"""Ranking diagnostics: does a score order moments inside one recording, or only tell recordings apart?"""
import math

import pytest

from seeforme.eval.ranking import null_summary, within_group_auc


def test_score_that_only_separates_recordings_has_no_within_group_auc():
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    g = ["a", "a", "a", "b", "b", "b"]
    r = within_group_auc(y, s, g)
    assert r["groups"] == 0 and r["rows"] == 0 and math.isnan(r["auc"])


def test_perfect_order_inside_each_recording_despite_offsets():
    y = [1, 0, 0, 1, 0, 0]
    s = [0.30, 0.20, 0.10, 0.95, 0.90, 0.85]    # pooled order is imperfect, each recording's order is perfect
    g = ["a", "a", "a", "b", "b", "b"]
    r = within_group_auc(y, s, g)
    assert r["auc"] == pytest.approx(1.0) and r["groups"] == 2 and r["rows"] == 6


def test_mean_is_weighted_by_rows():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.8, 0.2, 0.1, 0.1, 0.9]          # recording a: 1.0 on 4 rows; recording b: 0.0 on 2 rows
    g = ["a", "a", "a", "a", "b", "b"]
    r = within_group_auc(y, s, g)
    assert r["auc"] == pytest.approx(4 / 6) and r["per_group"] == {"a": 1.0, "b": 0.0}


def test_single_class_recordings_are_skipped_and_counted():
    y = [1, 0, 1, 1]
    s = [0.9, 0.1, 0.5, 0.4]
    g = ["a", "a", "b", "b"]
    r = within_group_auc(y, s, g)
    assert r["groups"] == 1 and r["skipped"] == 1 and r["auc"] == pytest.approx(1.0)


def test_ties_count_half():
    r = within_group_auc([1, 0], [0.5, 0.5], ["a", "a"])
    assert r["auc"] == pytest.approx(0.5)


def test_null_summary_counts_the_observed_value_itself():
    r = null_summary(0.70, [0.50] * 99)
    assert r["p"] == pytest.approx(1 / 100) and r["mean"] == pytest.approx(0.50) and r["n"] == 99


def test_null_summary_p_is_the_share_at_or_above_observed():
    null = [0.40] * 90 + [0.75] * 9
    r = null_summary(0.70, null)
    assert r["p"] == pytest.approx(10 / 100) and r["lo"] == pytest.approx(0.40) and r["hi"] == pytest.approx(0.75)


def test_null_summary_rejects_an_empty_null():
    with pytest.raises(ValueError):
        null_summary(0.7, [])
