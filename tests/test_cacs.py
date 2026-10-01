from seeforme.config import CACSConfig, BaselineConfig
from seeforme.cues import CACSPolicy, ConfidenceThresholdPolicy, ConfidenceCooldownPolicy
from seeforme.types import Track


def mk(tid, label, bbox, conf, kind="object", text=None, t=0.0):
    tr = Track(track_id=tid, label=label, kind=kind, bbox=bbox, confidence=conf, first_seen=t, last_seen=t, text=text, confirmed=True)
    tr.bbox_history.append((t, bbox))
    return tr


def test_proposal_example_chair_beats_recently_announced_poster():
    """Section 4.4: chair ahead (0.83, never announced) vs poster (0.95, announced 2 s ago)."""
    chair = mk(1, "chair", (0.35, 0.45, 0.65, 0.95), 0.83)
    poster = mk(2, "poster", (0.6, 0.1, 0.9, 0.35), 0.95)
    poster.last_announced = 8.0; poster.last_announced_bbox = poster.bbox; poster.announce_count = 1
    pol = CACSPolicy(CACSConfig())
    cues = pol.select([poster, chair], t=10.0)
    assert cues and cues[0].label == "chair"
    # a confidence-only baseline picks the poster
    base = ConfidenceThresholdPolicy(BaselineConfig(confidence_threshold=0.5))
    assert base.select([poster, chair], 10.0)[0].label == "poster"


def test_low_confidence_is_suppressed_and_uncertain_is_hedged():
    cfg = CACSConfig(min_confidence=0.35, sure_confidence=0.6, threshold=0.0)
    pol = CACSPolicy(cfg)
    assert pol.select([mk(1, "chair", (0.3, 0.4, 0.7, 0.95), 0.2)], 0.0) == []
    cues = pol.select([mk(2, "chair", (0.3, 0.4, 0.7, 0.95), 0.5)], 0.0)
    assert cues and cues[0].hedged and cues[0].message.startswith("possibly")


def test_duplicate_penalty_silences_repeat():
    pol = CACSPolicy(CACSConfig())
    chair = mk(1, "chair", (0.35, 0.45, 0.65, 0.95), 0.9)
    assert pol.select([chair], 0.0)
    assert pol.select([chair], 0.5) == []      # same object, half a second later


def test_important_sign_outranks_decorative_sign():
    pol = CACSPolicy(CACSConfig())
    exit_sign = mk(1, "text", (0.4, 0.1, 0.7, 0.22), 0.7, kind="text", text="EXIT")
    ad = mk(2, "text", (0.1, 0.1, 0.4, 0.22), 0.9, kind="text", text="SALE 50% OFF")
    cues = pol.select([ad, exit_sign], 0.0)
    assert cues and cues[0].text == "EXIT"


def test_cooldown_baseline_suppresses_same_label():
    pol = ConfidenceCooldownPolicy(BaselineConfig(confidence_threshold=0.5, cooldown=5.0))
    a = mk(1, "chair", (0.3, 0.4, 0.6, 0.9), 0.9)
    assert pol.select([a], 0.0)
    assert pol.select([a], 1.0) == []
    assert pol.select([a], 6.0)
