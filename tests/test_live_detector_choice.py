"""The decision logic behind results/live/detector_choice.md: a reversed rule or a flipped bootstrap sign must fail here."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
eld = pytest.importorskip("eval_live_detector")

M = ["yolo11n.pt", "yolo11s.pt"]


def test_recommend_picks_the_more_accurate_model_within_budget():
    f1 = {0.25: {M[0]: 0.86, M[1]: 0.90}, 0.35: {M[0]: 0.85, M[1]: 0.89}}
    assert eld.recommend(M, f1, {M[0]: 51.0, M[1]: 54.0}, 120.0, M[0])["model"] == M[1]


def test_recommend_falls_back_when_over_budget_or_when_thresholds_disagree():
    f1 = {0.25: {M[0]: 0.86, M[1]: 0.90}, 0.35: {M[0]: 0.85, M[1]: 0.89}}
    assert eld.recommend(M, f1, {M[0]: 51.0, M[1]: 150.0}, 120.0, M[0])["model"] == M[0]
    mixed = {0.25: {M[0]: 0.86, M[1]: 0.90}, 0.35: {M[0]: 0.90, M[1]: 0.89}}
    r = eld.recommend(M, mixed, {M[0]: 51.0, M[1]: 54.0}, 120.0, M[0])
    assert r["model"] == M[0] and r["more_accurate"] is None
    tie = {0.25: {M[0]: 0.88, M[1]: 0.88}}
    assert eld.recommend(M, tie, {M[0]: 51.0, M[1]: 54.0}, 120.0, M[0])["more_accurate"] is None


def test_greedy_matches_is_one_to_one():
    a = [(0, 0, 1, 1)]
    b = [(0, 0, 1, 1), (0, 0, 0.98, 1)]
    assert eld.greedy_matches(a, b, 0.5) == 1 and eld.greedy_matches(b, b, 0.5) == 2


def test_paired_bootstrap_sign_is_b_minus_a():
    ca = {f"s{i}": Counter(tp=5, fp=1, fn=5) for i in range(8)}
    cb = {f"s{i}": Counter(tp=8, fp=1, fn=2) for i in range(8)}
    out = eld.paired_bootstrap(ca, cb, n_boot=200)
    assert out["recall"]["diff"] > 0 and out["recall"]["ci95"][0] > 0


def test_persistence_skips_pairs_across_clips():
    box = (0.4, 0.2, 0.6, 0.9)
    frames = {"m": [[("person", 0.9, box)], [], [("person", 0.9, box)]]}
    across = eld.persistence(frames, [0, 1, 1], "m", 0.25, classes=("person",))["person"]
    same = eld.persistence(frames, [0, 0, 0], "m", 0.25, classes=("person",))["person"]
    assert across["transitions"] == 1 and same["transitions"] == 2      # the 0 -> 1 pair crosses clips and is skipped
