"""CARE: strict speech ledger, utility-maximising scheduler with explicit silence, finite-family risk selection."""
import math
import random

import pytest

from seeforme.eval.risk_control import hoeffding_upper, select_config
from seeforme.events.schema import CommunicationDecision, SceneEvent
from seeforme.scheduling.care import CareWeights, choose_action, utility
from seeforme.scheduling.speech_ledger import SpeechLedger


# ------------------------------------------------------------------ Guarantee 1: bounded speech expenditure
def test_ledger_bound_holds_for_adversarial_sequences():
    rng = random.Random(0)
    for trial in range(200):
        C, r = rng.uniform(1, 8), rng.uniform(0.02, 0.3)
        led = SpeechLedger(capacity_s=C, refill_per_s=r)
        t, accepted = 0.0, []
        for _ in range(300):
            t += rng.expovariate(rng.uniform(0.5, 10))          # bursts and lulls
            d = rng.uniform(0.2, 4.0)
            if led.try_spend(d, t):
                accepted.append((t, d))
        for i in range(len(accepted)):                           # every interval (s, t] between accepted cues
            for j in range(i, len(accepted)):
                s = accepted[i][0] - 1e-9
                spent = sum(d for (ti, d) in accepted[i:j + 1])
                assert spent <= C + r * (accepted[j][0] - s) + 1e-6


def test_ledger_refuses_what_it_cannot_pay_and_is_monotone_in_time():
    led = SpeechLedger(capacity_s=3.0, refill_per_s=0.1)
    assert led.try_spend(2.5, 0.0) and not led.try_spend(1.0, 0.1)
    assert led.try_spend(1.0, 10.0)                              # refilled 1 s over 10 s
    with pytest.raises(ValueError):
        led.try_spend(0.5, 5.0)                                  # time must not go backwards
    with pytest.raises(ValueError):
        SpeechLedger(capacity_s=0, refill_per_s=0.1)


# ------------------------------------------------------------------ CARE choice
def ev(eid, p_help, dur=1.5, **kw):
    base = dict(event_id=eid, t=10.0, category="object", track_ids=(eid,), p_helpful=p_help, duration_s=dur)
    base.update(kw)
    return SceneEvent(**base)


W = CareWeights(cost_wrong=2.0, cost_repeat=1.0, cost_speech_per_s=0.1)


def test_silence_wins_when_no_event_has_positive_utility():
    led = SpeechLedger(capacity_s=10, refill_per_s=1)
    d = choose_action([ev("a", 0.05), ev("b", 0.1)], t=10.0, ledger=led, busy=False, w=W)
    assert d.action == "SILENCE" and d.reason_code == "no_positive_utility"


def test_highest_utility_event_is_spoken_and_paid_for():
    led = SpeechLedger(capacity_s=10, refill_per_s=1)
    d = choose_action([ev("a", 0.4), ev("b", 0.9), ev("c", 0.95, p_wrong=0.5)], t=10.0, ledger=led, busy=False, w=W)
    assert d.action == "SPEAK" and d.event_id == "b"
    assert led.balance(10.0) == pytest.approx(10 - 1.5)


def test_ineligible_events_are_skipped_with_reasons():
    led = SpeechLedger(capacity_s=1.0, refill_per_s=0.0001)
    evs = [ev("x", 0.99, expired=True), ev("y", 0.99, announced=True), ev("z", 0.99, grounded=False), ev("w", 0.99, dur=2.0)]
    d = choose_action(evs, t=10.0, ledger=led, busy=False, w=W)
    assert d.action == "SILENCE" and d.reason_code == "no_eligible_event"
    assert d.skipped == {"x": "expired", "y": "already_announced", "z": "ungrounded", "w": "budget"}


def test_busy_channel_means_silence_without_spending():
    led = SpeechLedger(capacity_s=10, refill_per_s=1)
    d = choose_action([ev("a", 0.99)], t=10.0, ledger=led, busy=True, w=W)
    assert d.action == "SILENCE" and d.reason_code == "channel_busy" and led.balance(10.0) == pytest.approx(10.0)


def test_utility_formula():
    e = ev("a", 0.8, dur=2.0, value=1.5, p_wrong=0.1, p_redundant=0.2)
    assert utility(e, W) == pytest.approx(1.5 * 0.8 - 2.0 * 0.1 - 1.0 * 0.2 - 0.1 * 2.0)


def test_scene_event_has_no_ground_truth_fields():
    assert not any("source" in f or "ground" in f.lower() and f != "grounded" for f in SceneEvent.__dataclass_fields__)
    assert isinstance(CommunicationDecision(action="SILENCE", reason_code="x"), CommunicationDecision)


# ------------------------------------------------------------------ risk control (finite family, Hoeffding + union bound)
def test_hoeffding_upper_bound_matches_formula():
    assert hoeffding_upper(0.05, n=34, k=10, delta=0.05) == pytest.approx(0.05 + math.sqrt(math.log(10 / 0.05) / 68))


def test_select_maximises_recall_subject_to_certified_risk():
    configs = {"loud": {"recall": 0.30, "risk": 0.20}, "mid": {"recall": 0.22, "risk": 0.02}, "quiet": {"recall": 0.10, "risk": 0.0}}
    best, info = select_config(configs, n=400, alpha=0.15, delta=0.05)
    assert best == "mid" and info["mid"]["upper"] <= 0.15 and info["loud"]["upper"] > 0.15
    none, _ = select_config(configs, n=34, alpha=0.10, delta=0.05)
    assert none is None                                          # 34 recordings cannot certify 0.10
