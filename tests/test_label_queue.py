"""Stratified two-rater label queue of decision moments (issue #5, rubric v4)."""
import pytest

from seeforme.egoblind.label_queue import DEV_SPLITS, QUOTAS, build_queue, stratum_of


def pool(n_per=40, strata=("door", "stairs", "pole", "scene", "covered_other", "quiet", "unasked"), split="train"):
    return [{"key": f"{s}:{i}", "stratum": s, "group": f"{s}-rec{i}", "source_split": split} for s in strata for i in range(n_per)]


def test_stratum_from_label_and_concepts():
    assert stratum_of("door", []) == "door"
    assert stratum_of("unknown", ["door"]) == "door"            # the object is often only in the concepts
    assert stratum_of("unknown", ["stairs"]) == "stairs"
    assert stratum_of("fence", []) == "pole" and stratum_of("unknown", ["pole"]) == "pole"
    assert stratum_of("unknown", []) == "scene"
    assert stratum_of("car", []) == "covered_other"
    assert stratum_of("none", [], should_speak=False, reason="user_confirmed_clear") == "quiet"


def test_quotas_are_met_when_the_pool_is_large_enough():
    picked, rep = build_queue(pool(), QUOTAS, seed=7)
    assert len(picked) == sum(QUOTAS.values()) == 160
    for s, q in QUOTAS.items():
        assert rep["picked"][s] == q and rep["shortfall"][s] == 0
    assert len({p["key"] for p in picked}) == len(picked)
    assert [p["sample_id"] for p in picked] == [f"SFM-{i + 1:03d}" for i in range(len(picked))]


def test_shortages_are_filled_from_other_strata_and_reported():
    p = [x for x in pool() if not (x["stratum"] == "stairs" and int(x["key"].split(":")[1]) >= 8)]   # only 8 stairs
    picked, rep = build_queue(p, QUOTAS, seed=7)
    assert len(picked) == 160
    assert rep["picked"]["stairs"] == 8 and rep["shortfall"]["stairs"] == QUOTAS["stairs"] - 8
    assert sum(rep["filled"].values()) == QUOTAS["stairs"] - 8
    assert "stairs" not in rep["filled"]


def test_seeded_and_order_does_not_reveal_the_stratum():
    a, _ = build_queue(pool(), QUOTAS, seed=7)
    b, _ = build_queue(list(reversed(pool())), QUOTAS, seed=7)
    assert [x["key"] for x in a] == [x["key"] for x in b]            # same pool, same seed -> same queue
    c, _ = build_queue(pool(), QUOTAS, seed=8)
    assert [x["key"] for x in a] != [x["key"] for x in c]
    head = [x["stratum"] for x in a[:24]]
    assert len(set(head)) > 2                                        # shuffled, not blocks of one stratum


def test_only_development_splits_can_enter_the_queue():
    assert DEV_SPLITS == ("train", "finetune")
    for bad in ("select", "holdout", "test", "campus_final"):
        with pytest.raises(ValueError):
            build_queue(pool(split=bad), QUOTAS, seed=7)


def test_one_recording_cannot_dominate():
    p = pool()
    for x in p:
        if x["stratum"] == "covered_other":
            x["group"] = "one-long-recording"
    picked, rep = build_queue(p, QUOTAS, seed=7, max_per_group=2)
    assert sum(x["group"] == "one-long-recording" for x in picked) == 2
    assert len(picked) == 160                                        # the gap is filled from elsewhere
