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
