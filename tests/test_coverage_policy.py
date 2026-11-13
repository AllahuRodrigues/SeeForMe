"""Coverage-first diagnostic policy: every confirmed candidate is announced once, under the strict speech ledger."""
from seeforme.cues.pcacs import ChannelState
from seeforme.eval.funnel import FunnelRecorder, classify
from seeforme.scheduling.coverage_policy import CoverageFirstPolicy
from tests.test_egoblind import synthetic_scenario
from tests.test_online_policy import _tracks_at

REASONS = {"spoken", "unconfirmed", "already_announced", "channel_busy", "budget", "queued", "low_confidence"}


def _tracks():
    sc, _ = synthetic_scenario()
    return _tracks_at(sc, 6)            # two confident tracks: a chair and a car


def test_a_candidate_must_be_seen_twice_before_it_is_announced():
    pol = CoverageFirstPolicy(confirm_n=2, refill_per_s=None); pol.reset()
    tr = _tracks()
    assert pol.decide(tr, 6.0, ChannelState()) == []
    assert set(pol.last_reasons.values()) == {"unconfirmed"}
    cues = pol.decide(tr, 6.5, ChannelState())
    assert len(cues) == 1 and pol.last_reasons[cues[0].track_id] == "spoken"
    assert set(pol.last_reasons.values()) <= REASONS


def test_every_confirmed_candidate_is_announced_exactly_once():
    pol = CoverageFirstPolicy(confirm_n=1, refill_per_s=None); pol.reset()
    tr, spoken, t = _tracks(), [], 6.0
    for _ in range(20):
        spoken += [c.track_id for c in pol.decide(tr, t, ChannelState())]
        t += 0.5
    assert sorted(spoken) == sorted(trk.track_id for trk in tr)      # no learned threshold suppresses anything
    assert set(pol.last_reasons.values()) == {"already_announced"}


def test_a_busy_channel_delays_but_does_not_lose_a_candidate():
    pol = CoverageFirstPolicy(confirm_n=1, refill_per_s=None); pol.reset()
    tr = _tracks()
    assert pol.decide(tr, 6.0, ChannelState(busy=True, remaining_s=1.0)) == []
    assert set(pol.last_reasons.values()) == {"channel_busy"}
    assert len(pol.decide(tr, 6.5, ChannelState())) == 1


def _cue_lengths():
    pol = CoverageFirstPolicy(confirm_n=1, refill_per_s=None); pol.reset()
    tr = _tracks()
    return [c.duration_s for k in range(4) for c in pol.decide(tr, 6.0 + 0.5 * k, ChannelState())]


def test_the_ledger_is_the_only_limit_and_is_logged():
    cap = max(_cue_lengths())                                         # room for one cue, never for both
    pol = CoverageFirstPolicy(confirm_n=1, capacity_s=cap, refill_per_s=0.0); pol.reset()
    tr, spent, t = _tracks(), 0.0, 6.0
    for _ in range(10):
        cues = pol.decide(tr, t, ChannelState())
        spent += sum(c.duration_s for c in cues)
        t += 0.5
    assert 0.0 < spent <= cap + 1e-9
    assert "budget" in pol.last_reasons.values()                     # the other cue did not fit and says so
    assert pol.stats["budget_blocked"] >= 1 and pol.stats["announced"] == 1


def test_a_candidate_that_disappears_is_dropped_as_stale():
    pol = CoverageFirstPolicy(confirm_n=1, refill_per_s=None, stale_s=2.0); pol.reset()
    tr = _tracks()
    pol.decide(tr, 6.0, ChannelState(busy=True, remaining_s=5.0))     # both queued, nothing said
    assert pol.decide([], 9.0, ChannelState()) == []
    assert pol.stats["stale"] == 2


def test_one_ledger_runs_over_the_whole_stream():
    cap = max(_cue_lengths())
    pol = CoverageFirstPolicy(confirm_n=1, capacity_s=cap, refill_per_s=0.0); pol.reset()
    tr = _tracks()
    first = [c for k in range(4) for c in pol.decide(tr, 6.0 + 0.5 * k, ChannelState())]
    pol.reset()                                                       # next clip: the queue is new, the budget is not
    for trk in tr:
        trk.last_announced = None
    second = [c for k in range(4) for c in pol.decide(tr, 0.5 * k, ChannelState())]
    assert len(first) == 1 and second == []                           # a per-clip refill would speak again here


def test_recorder_feeds_the_funnel_for_any_policy():
    from seeforme.egoblind import run_on_egoblind
    sc, _ = synthetic_scenario()
    pol = CoverageFirstPolicy(confirm_n=1, refill_per_s=None)
    rec = FunnelRecorder(pol)
    run_on_egoblind(pol, sc, None, seed=0, on_decision=rec)
    det = {}
    for t, dets in sc.frames():
        for d in dets:
            if d.source_id is not None:
                det.setdefault(d.source_id, []).append(t)
    e = sc.speak_events()[0]
    tr = rec.trace(e, det.get(e.obj_id, []))
    assert tr.cand and tr.cue and classify(tr) in ("delivered", "timing")
    assert all(r in REASONS for _, r in tr.reasons)
