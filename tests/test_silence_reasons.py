"""Every legal speaking opportunity a policy passes up gets the exact reason its own decision rule used."""
from seeforme.cues.pcacs import ChannelState, PCACSConfig, PCACSPolicy
from seeforme.online.policy import OnlinePolicy
from tests.test_egoblind import synthetic_scenario
from tests.test_online_policy import _tracks_at

REASONS = {"spoken", "outranked", "below_threshold", "budget", "recently_announced", "guard", "channel_busy", "low_confidence"}


def _tracks():
    sc, _ = synthetic_scenario()
    return _tracks_at(sc, 6)


def test_pcacs_below_threshold_when_threshold_is_unreachable():
    pol = PCACSPolicy(PCACSConfig(threshold_speak=99.0))
    pol.reset()
    tr = _tracks()
    assert pol.decide(tr, 6.0, ChannelState()) == []
    assert set(pol.last_reasons.values()) <= REASONS and "below_threshold" in pol.last_reasons.values()


def test_pcacs_reasons_mark_the_spoken_track():
    pol = PCACSPolicy(PCACSConfig(threshold_speak=-99.0, use_budget=False))
    pol.reset()
    cues = pol.decide(_tracks(), 6.0, ChannelState())
    assert cues and pol.last_reasons[cues[0].track_id] == "spoken"


def test_online_reasons():
    tr = _tracks()
    hi = OnlinePolicy(mode="greedy", tau0=0.999, budget=False); hi.reset()
    assert hi.decide(tr, 6.0, ChannelState()) == [] and set(hi.last_reasons.values()) == {"below_threshold"}
    busy = OnlinePolicy(mode="greedy", tau0=0.0, budget=False); busy.reset()
    busy.decide(tr, 6.0, ChannelState(busy=True, remaining_s=1.0))
    assert set(busy.last_reasons.values()) == {"channel_busy"}
    lo = OnlinePolicy(mode="greedy", tau0=0.0, budget=False); lo.reset()
    cues = lo.decide(tr, 6.0, ChannelState())
    assert lo.last_reasons[cues[0].track_id] == "spoken"
    assert all(v in ("spoken", "outranked") for v in lo.last_reasons.values())
