"""Issue 6: the learned (IL) scheduler re-announces the same live track.

Cause: SchedulerPolicy's only per-track guard is `recent_block_s` (about 2.8 s). The P-CACS rule also multiplies its
score by a novelty term that decays over `reannounce_tau_s` (10 s), but the learned model does not reproduce that,
so a track it liked once becomes legal again 2.8 s later and is spoken again at the first decision after the block.
The opt-in `reannounce_guard_s` blocks speaking the same track again within that many seconds unless it has come
at least `reannounce_move_m` closer since the last announcement.
"""
import torch

from seeforme.cues.pcacs import ChannelState, PCACSConfig
from seeforme.ml.scheduler_model import ACTION_IDX, SchedulerModel, SchedulerPolicy
from seeforme.types import Track


def _always_speak_model() -> SchedulerModel:
    m = SchedulerModel(K=8, epochs=0)
    with torch.no_grad():
        for p in m.net.action.parameters():
            p.zero_()
        m.net.action.bias[ACTION_IDX["speak"]] = 20.0          # P(speak) ~ 1 for every candidate
    return m


def _track(dist: float) -> Track:
    tr = Track(track_id=7, label="chair", kind="object", bbox=(0.4, 0.4, 0.6, 0.8), confidence=0.9, first_seen=0.0, last_seen=0.0,
               confirmed=True)
    tr.distance_m, tr.depth_conf, tr.path_prob = dist, 0.8, 0.8
    return tr


def _spoken_times(pol, dists, dt=0.5):
    tr, out = _track(dists[0]), []
    pol.reset()
    for k, d in enumerate(dists):
        t = k * dt
        tr.distance_m, tr.last_seen = d, t
        cues = pol.decide([tr], t, ChannelState())
        if any(c.action in ("speak", "interrupt") for c in cues):     # haptic pulses are not announcements
            out.append(t)
    return out


def test_default_policy_reannounces_after_recent_block():
    # documents the cause: without the guard, a liked track is spoken again as soon as recent_block_s has passed
    pol = SchedulerPolicy(_always_speak_model(), PCACSConfig(), threshold=0.5, mode="imitation")
    times = _spoken_times(pol, [5.0] * 20)                      # 10 s, object static
    assert len(times) >= 3 and times[1] - times[0] < pol.cfg.recent_block_s + 0.6


def test_guard_blocks_same_track_reannouncement():
    pol = SchedulerPolicy(_always_speak_model(), PCACSConfig(), threshold=0.5, mode="imitation", reannounce_guard_s=10.0)
    times = _spoken_times(pol, [5.0] * 20)
    assert times == [0.0]


def test_guard_allows_reannouncement_after_horizon():
    pol = SchedulerPolicy(_always_speak_model(), PCACSConfig(), threshold=0.5, mode="imitation", reannounce_guard_s=4.0)
    times = _spoken_times(pol, [5.0] * 20)
    assert times[0] == 0.0 and len(times) >= 2 and times[1] >= 4.0


def test_guard_allows_reannouncement_when_object_approaches():
    pol = SchedulerPolicy(_always_speak_model(), PCACSConfig(), threshold=0.5, mode="imitation", reannounce_guard_s=10.0)
    dists = [9.0 - 0.5 * k for k in range(14)]                  # 1 m/s closer; 3 m after 3 s
    times = _spoken_times(pol, dists)
    assert len(times) >= 2 and 3.0 <= times[1] - times[0] < 10.0


def test_guard_also_applies_in_rl_mode():
    pol = SchedulerPolicy(_always_speak_model(), PCACSConfig(), mode="rl", reannounce_guard_s=10.0)
    with torch.no_grad():
        pol.model.net.silence_bias.fill_(-20.0)
    assert _spoken_times(pol, [5.0] * 20) == [0.0]


def test_policy_name_suffix_turns_the_guard_on():
    import sys
    from pathlib import Path
    import pytest
    root = Path(__file__).resolve().parents[1]
    if not (root / "models" / "egoblind" / "scheduler" / "scheduler_il.pt").exists():
        pytest.skip("no trained EgoBlind scheduler")
    sys.path.insert(0, str(root / "scripts"))
    import egoblind_common as ec
    from seeforme.config import PipelineConfig
    from tests.test_egoblind import synthetic_scenario
    sc, _ = synthetic_scenario()
    cfg = PipelineConfig()
    assert ec.make_egoblind_policy("egoblind_learned:il", cfg, sc).reannounce_guard_s == 0.0
    p = ec.make_egoblind_policy("egoblind_learned:il+guard10", cfg, sc)
    assert p.reannounce_guard_s == 10.0 and p.mode == "imitation" and p.name == "egoblind_learned:il+guard10"


def test_repeat_counts_separate_same_track_from_same_event():
    # repeat_gt counts a second cue about the same weak event even when it names a different physical object:
    # EgoBlind links every detection of the event's label inside its window to that event (cache.frames)
    from seeforme.egoblind.metrics import repeat_counts
    from seeforme.types import Cue
    def cue(t, tid, sid, label="person"):
        return Cue(t=t, track_id=tid, label=label, kind="object", message="", score=1.0, policy="x", source_id=sid)
    spoken = [cue(0.0, 1, 5), cue(1.2, 2, 5), cue(3.0, 1, 5), cue(30.0, 3, None, "car"), cue(100.0, 4, 6, "car")]
    r = repeat_counts(spoken, label_window=60.0)
    assert r == {"repeat_gt": 2, "repeat_track": 1, "repeat_label": 2}
