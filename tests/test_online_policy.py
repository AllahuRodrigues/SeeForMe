"""M1/S3: OnlinePolicy (greedy / apple / thompson / full_info), budget controller, stream runner."""
import random

from seeforme.cues.pcacs import ChannelState
from seeforme.online.policy import BudgetController, OnlinePolicy
from seeforme.online.stream import run_stream

from tests.test_egoblind import synthetic_scenario


def _car(sc):
    return next(e for e in sc.speak_events() if e.label == "car")


def _timely(log, ev):
    return any(c.source_id == ev.obj_id and ev.in_window(c.t) for c in log.cues)


def test_budget_controller_holds_rate():                                    # AC-9
    bc = BudgetController.make(target_per_min=2.0, tau0=0.5)
    rng = random.Random(0)
    spoken, dt, T = 0, 0.5, 0.0
    for step in range(4000):
        p = rng.random()
        s = p >= bc.tau
        bc = bc.step(s, dt)
        T += dt
        if step >= 2000:
            spoken += s
    rate = spoken / (2000 * dt / 60.0)
    assert 1.5 <= rate <= 2.5


def test_greedy_learns_from_missed_questions():
    sc, _ = synthetic_scenario()
    car = _car(sc)
    pol = OnlinePolicy(mode="greedy", tau0=0.6, budget=False, seed=0)
    logs = run_stream(pol, [sc] * 8)
    assert not _timely(logs[0][1], car)                                     # silent at first: nothing learnt yet
    assert _timely(logs[-1][1], car)                                        # the user's question taught it
    assert pol.n_labels > 0


def test_full_info_learns_at_least_as_fast():                               # AC-10
    sc, _ = synthetic_scenario()
    car = _car(sc)
    def first_timely(mode):
        pol = OnlinePolicy(mode=mode, tau0=0.6, budget=False, seed=0, oracle_scenario=None)
        for i, (_, log) in enumerate(run_stream(pol, [sc] * 8)):
            if _timely(log, car):
                return i
        return 99
    assert first_timely("full_info") <= first_timely("greedy")


def test_weights_persist_across_videos():
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.6, budget=False)
    run_stream(pol, [sc])
    w = pol.state.w.copy()
    pol.reset()
    assert (pol.state.w == w).all()


def test_no_speech_on_busy_channel():                                       # AC-11
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.0, budget=False)
    pol.reset()
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.config import PipelineConfig
    tr = IoUTracker(PipelineConfig().tracker)
    tracks = []
    for t, dets in sc.frames():
        tracks = tr.update(dets, t)
        if t > 6:
            break
    assert pol.decide(tracks, 6.0, ChannelState(busy=True, remaining_s=1.0)) == []
    assert pol.decide(tracks, 6.0, ChannelState()) != []


def test_apple_tasting_explores_below_threshold():
    sc, _ = synthetic_scenario()
    quiet = OnlinePolicy(mode="greedy", tau0=0.99, budget=False, seed=1)
    explore = OnlinePolicy(mode="apple", tau0=0.99, budget=False, eps=1.0, target_per_min=60.0, seed=1)
    n_greedy = sum(len(log.cues) for _, log in run_stream(quiet, [sc]))
    n_apple = sum(len(log.cues) for _, log in run_stream(explore, [sc]))
    assert n_greedy == 0 and n_apple > 0


def test_stream_is_reproducible():                                          # AC-12
    sc, _ = synthetic_scenario()
    a = [[(c.t, c.source_id) for c in log.cues] for _, log in run_stream(OnlinePolicy(mode="thompson", seed=3), [sc] * 3)]
    b = [[(c.t, c.source_id) for c in log.cues] for _, log in run_stream(OnlinePolicy(mode="thompson", seed=3), [sc] * 3)]
    assert a == b


def test_protected_splits_need_final_flag():                                # AC-13
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for split in ("holdout", "test", "train,holdout"):
        r = subprocess.run([sys.executable, "scripts/run_online.py", "--stream", split], cwd=root, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "--final" in r.stderr


def test_apple_exploration_spends_budget_share():
    pol = OnlinePolicy(mode="apple", eps=0.25, target_per_min=2.0)
    p_explore = 0.25 * 2.0 * pol.decision_interval / 60.0
    assert abs(p_explore - 0.25 * 2.0 / 120.0) < 1e-12        # 0.25 of 2 cues/min at 2 decisions/s


def _tracks_at(sc, t_stop):
    from seeforme.tracking.tracker import IoUTracker
    from seeforme.config import PipelineConfig
    tr = IoUTracker(PipelineConfig().tracker)
    tracks = []
    for t, dets in sc.frames():
        tracks = tr.update(dets, t)
        if t > t_stop:
            break
    return tracks


def test_no_implicit_negatives_by_default():
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.999, budget=False, lookback_s=5.0)
    pol.reset()
    pol.decide(_tracks_at(sc, 6), 6.0, ChannelState())
    pol.decide([], 20.0, ChannelState())
    pol.reset()
    assert pol.n_labels == 0


def test_unasked_candidates_become_weak_negatives_when_they_expire():
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.999, budget=False, lookback_s=5.0, implicit_neg_w=0.1)
    pol.reset()
    pol.decide(_tracks_at(sc, 6), 6.0, ChannelState())
    n_seen = len(pol._hist)
    assert n_seen > 0 and pol.n_labels == 0
    pol.decide([], 20.0, ChannelState())                                     # 14 s later: nobody asked
    assert pol.n_labels == n_seen and pol.n_pos == 0


def test_candidates_covered_by_a_question_are_not_implicitly_negated():
    from seeforme.online.feedback import Feedback
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.999, budget=False, lookback_s=5.0, implicit_neg_w=0.1)
    pol.reset()
    pol.decide(_tracks_at(sc, 6), 6.0, ChannelState())
    n_seen = len(pol._hist)
    pol.learn([Feedback(t=7.0, label=1, kind="missed_question", source_id=None, t0=0.0, t1=6.0)])
    assert pol.n_pos == n_seen
    pol.reset()                                                              # end of video flushes the rest
    assert pol.n_labels == n_seen


def test_tapped_cue_is_not_implicitly_negated():
    from seeforme.online.feedback import Feedback
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.0, budget=False, lookback_s=5.0, implicit_neg_w=0.1)
    pol.reset()
    assert pol.decide(_tracks_at(sc, 6), 6.0, ChannelState()) != []
    n_seen = len(pol._hist)
    pol.learn([Feedback(t=6.0, label=1, kind="tap_useful", source_id=None, t0=6.0, t1=6.0)])
    assert pol.n_labels == 1
    pol.reset()
    assert pol.n_labels == 1 + (n_seen - 1)                                  # the spoken candidate is not negated


# ---------------------------------------------------------------- causal full-information reference (algo item 1)

def _full_info_at(sc, delay, t=6.0):
    pol = OnlinePolicy(mode="full_info", tau0=0.999, budget=False, full_info_delay=delay)
    pol.begin_video(sc, None)
    pol.reset()
    pol.decide(_tracks_at(sc, t), t, ChannelState())
    return pol


def test_full_info_default_learns_immediately():
    sc, _ = synthetic_scenario()
    pol = _full_info_at(sc, None)
    assert pol.n_labels > 0                                                  # zero-delay reference, unchanged default


def test_causal_full_info_holds_labels_until_video_end():
    sc, _ = synthetic_scenario()
    pol = _full_info_at(sc, "video")
    w0 = pol.state.w.copy()
    assert pol.n_labels == 0 and pol.n_pending > 0
    pol.decide(_tracks_at(sc, 8), 8.0, ChannelState())
    assert pol.n_labels == 0 and (pol.state.w == w0).all()                   # still nothing applied mid-video
    pending = pol.n_pending
    pol.reset()                                                              # video end flushes the buffer
    assert pol.n_pending == 0 and pol.n_labels == pending and not (pol.state.w == w0).all()


def test_full_info_fixed_delay_releases_after_d_seconds():
    sc, _ = synthetic_scenario()
    pol = _full_info_at(sc, 2.0)
    assert pol.n_labels == 0 and pol.n_pending > 0
    first = pol.n_pending
    pol.decide([], 7.5, ChannelState())                                      # 1.5 s later: not yet
    assert pol.n_labels == 0
    pol.decide([], 8.0, ChannelState())                                      # 2 s later: released
    assert pol.n_labels == first and pol.n_pending == 0


def test_full_info_delay_validation():
    import pytest
    for bad in (-1.0, "soon"):
        with pytest.raises(ValueError):
            OnlinePolicy(mode="full_info", full_info_delay=bad)


def test_causal_full_info_equals_greedy_within_the_first_video():
    # With labels held to video end, full info gets only the one-sided signals inside video 1, exactly like greedy.
    sc, _ = synthetic_scenario()
    a = run_stream(OnlinePolicy(mode="full_info", tau0=0.6, budget=False, full_info_delay="video"), [sc])
    b = run_stream(OnlinePolicy(mode="greedy", tau0=0.6, budget=False), [sc])
    assert [(c.t, c.source_id) for c in a[0][1].cues] == [(c.t, c.source_id) for c in b[0][1].cues]


def test_label_counts_by_kind_are_logged():
    from seeforme.online.feedback import Feedback
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.999, budget=False, lookback_s=5.0)
    pol.reset()
    pol.decide(_tracks_at(sc, 6), 6.0, ChannelState())
    n_seen = len(pol._hist)
    pol.learn([Feedback(t=7.0, label=1, kind="missed_question", source_id=None, t0=0.0, t1=6.0)])
    d = pol.to_dict()
    assert d["n_pos"] == n_seen and d["label_counts"] == {"missed_question": n_seen}


def test_stream_passes_p_ask():
    sc, _ = synthetic_scenario()
    pol = OnlinePolicy(mode="greedy", tau0=0.999, budget=False)
    run_stream(pol, [sc], p_ask=0.0)
    assert pol.n_labels == 0                                                 # never spoke, never asked: no labels
