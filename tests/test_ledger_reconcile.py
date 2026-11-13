"""Speech ledger against real playback: reserve an estimate, settle with what was actually played."""
import random

import pytest

from seeforme.scheduling.speech_ledger import SpeechLedger


def test_an_interrupted_cue_gets_the_unplayed_part_back():
    led = SpeechLedger(6.0, 0.0)
    rid = led.reserve(3.0, t=0.0)
    assert rid is not None and led.balance(0.0) == pytest.approx(3.0) and led.reserved_open == pytest.approx(3.0)
    assert led.settle(rid, played_s=1.0, t=1.0) == pytest.approx(2.0)          # barge-in after one second
    assert led.balance(1.0) == pytest.approx(5.0) and led.reserved_open == 0.0


def test_failed_speech_costs_nothing_and_a_refund_never_exceeds_capacity():
    led = SpeechLedger(4.0, 1.0)
    rid = led.reserve(2.0, t=0.0)
    led.settle(rid, played_s=0.0, t=10.0)                                      # the voice never started; the bucket had refilled
    assert led.balance(10.0) == pytest.approx(4.0)


def test_speech_that_ran_long_is_charged_and_blocks_the_next_cue_until_repaid():
    led = SpeechLedger(3.0, 0.5)
    rid = led.reserve(3.0, t=0.0)
    assert led.settle(rid, played_s=4.0, t=4.0) == pytest.approx(-1.0)         # one second over the estimate
    assert led.balance(4.0) == pytest.approx(1.0)                              # 2.0 refilled, 1.0 overrun charged
    led2 = SpeechLedger(3.0, 0.0)
    r2 = led2.reserve(3.0, t=0.0)
    led2.settle(r2, played_s=4.5, t=4.5)
    assert led2.balance(4.5) == pytest.approx(-1.5) and led2.reserve(0.1, t=5.0) is None   # in debt: silent


def test_a_reservation_settles_once():
    led = SpeechLedger(6.0, 0.0)
    rid = led.reserve(1.0, t=0.0)
    led.settle(rid, 1.0, t=1.0)
    with pytest.raises(ValueError):
        led.settle(rid, 1.0, t=2.0)
    with pytest.raises(ValueError):
        led.settle(999, 1.0, t=2.0)
    with pytest.raises(ValueError):
        led.settle(led.reserve(1.0, t=2.0), played_s=-0.1, t=3.0)


def test_no_budget_builds_up_while_the_session_is_paused():
    led = SpeechLedger(6.0, 1.0)
    led.settle(led.reserve(6.0, t=0.0), 6.0, t=6.0)                            # 6 s refilled while it played
    led.settle(led.reserve(6.0, t=6.0), 6.0, t=6.0)                            # bucket now empty at t = 6
    led.pause(t=6.0)
    assert led.reserve(0.5, t=30.0) is None and not led.can_spend(0.5, 30.0)   # paused: nothing is accepted
    led.resume(t=60.0)
    assert led.balance(60.0) == pytest.approx(0.0)                             # 54 s of pause earned nothing
    assert led.balance(62.0) == pytest.approx(2.0)


def test_session_reset_abandons_whatever_was_still_playing():
    led = SpeechLedger(6.0, 0.0)
    led.reserve(2.0, t=0.0); led.reserve(1.0, t=0.5)
    assert led.abandon_all(t=1.0) == pytest.approx(3.0)
    assert led.balance(1.0) == pytest.approx(6.0) and led.reserved_open == 0.0


def test_try_spend_is_reserve_and_settle_at_the_estimate():
    a, b = SpeechLedger(5.0, 0.2), SpeechLedger(5.0, 0.2)
    assert a.try_spend(2.0, 1.0)
    b.settle(b.reserve(2.0, 1.0), 2.0, 1.0)
    assert a.balance(3.0) == pytest.approx(b.balance(3.0))


def test_played_seconds_are_bounded_on_one_channel_even_with_wrong_estimates():
    """For every interval, played seconds of cues started in it are at most C + r * (running time) + the largest
    single overrun: the estimate is settled before the next cue is accepted, so debt never compounds."""
    for seed in range(20):
        rng = random.Random(seed)
        cap, r = 4.0, 0.08
        led = SpeechLedger(cap, r)
        t, cues, worst = 0.0, [], 0.0
        for _ in range(300):
            t += rng.uniform(0.0, 6.0)
            est = rng.uniform(0.6, 3.0)
            rid = led.reserve(est, t)
            if rid is None:
                continue
            played = max(0.0, est + rng.choice([-1.0, -0.4, 0.0, 0.0, 0.3, 0.9]) * rng.random())   # cut short, exact or long
            worst = max(worst, played - est)
            led.settle(rid, played, t + played)
            cues.append((t, played))
            t += played
        assert cues
        for i in range(len(cues)):
            tot = 0.0
            for j in range(i, len(cues)):
                tot += cues[j][1]
                assert tot <= cap + r * (cues[j][0] + cues[j][1] - cues[i][0]) + worst + 1e-6
