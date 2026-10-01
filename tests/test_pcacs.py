import math
from seeforme.config import PipelineConfig
from seeforme.cues.compose import compose, speech_seconds
from seeforme.cues.pcacs import ChannelState, PCACSConfig, PCACSPolicy
from seeforme.geometry import CameraModel
from seeforme.types import Track


def mk(tid, label, x, z, conf=0.9, kind="object", text=None, t=0.0, approach=1.0, depth_conf=0.9):
    cam = CameraModel()
    b = cam.project_box(x, z, 0.9 if label != "person" else 1.7, 0.5)
    tr = Track(tid, label, kind, b, conf, t, t, text=text, confirmed=True)
    tr.distance_m, tr.depth_conf, tr.lateral_m = z, depth_conf, x
    tr.rel_speed_mps = -approach
    tr.ttc_s = (z - 0.6) / approach if approach > 0.1 else math.inf
    for i in range(5):
        tr.dist_history.append((t - 2 + 0.4 * i, z + (4 - i) * 0.4 * approach, x))
    tr.path_prob = 0.9 if abs(x) < 0.6 else 0.05
    return tr


def test_chair_in_path_beats_confident_poster_and_far_person():
    pol = PCACSPolicy(PCACSConfig())
    chair = mk(1, "chair", 0.1, 2.5, conf=0.81)
    poster = mk(2, "poster", 1.5, 2.0, conf=0.99)
    person = mk(3, "person", 1.8, 7.0, conf=0.99)
    cues = pol.decide([poster, person, chair], 10.0, ChannelState())
    assert cues and cues[0].label == "chair" and cues[0].action == "speak"
    assert "meter" in cues[0].message


def test_silence_when_nothing_is_actionable():
    pol = PCACSPolicy(PCACSConfig())
    poster = mk(2, "poster", 1.5, 2.0, conf=0.99)
    person = mk(3, "person", 2.0, 8.0, conf=0.99)
    assert pol.decide([poster, person], 5.0, ChannelState()) == []


def test_interrupt_only_for_more_severe_candidate():
    pol = PCACSPolicy(PCACSConfig())
    stairs = mk(1, "stairs", 0.0, 3.0, conf=0.9)
    busy_low = ChannelState(busy=True, remaining_s=1.5, current_utility=1.0, current_severity=0.45, current_track=9)
    cues = pol.decide([stairs], 3.0, busy_low)
    assert cues and cues[0].action == "interrupt" and cues[0].interrupted_track == 9
    pol.reset()
    chair = mk(2, "chair", 0.0, 4.0, conf=0.9)
    busy_high = ChannelState(busy=True, remaining_s=1.5, current_utility=3.0, current_severity=1.0, current_track=9)
    assert pol.decide([chair], 3.0, busy_high) == []


def test_haptic_for_imminent_hazard_while_channel_busy():
    pol = PCACSPolicy(PCACSConfig())
    chair = mk(1, "chair", 0.0, 1.3, conf=0.9)          # ttc ~0.7 s
    busy = ChannelState(busy=True, remaining_s=2.0, current_utility=5.0, current_severity=1.0, current_track=9)
    cues = pol.decide([chair], 3.0, busy)
    assert cues and cues[0].action == "haptic" and cues[0].message == ""


def test_budget_limits_non_urgent_speech():
    cfg = PCACSConfig(budget_capacity_s=2.0, budget_refill=0.0, urgent_bypass=0.99)
    pol = PCACSPolicy(cfg)
    signs = [mk(i, "text", 0.2, 4.0 + i, conf=0.9, kind="text", text=f"ROOM 20{i}") for i in range(1, 4)]
    spoken = []
    for k in range(8):
        tr = signs[k % 3]
        c = pol.decide([tr], k * 3.0, ChannelState())
        spoken += c
    assert 1 <= len(spoken) <= 2               # a 2 s budget pays for one message, refill is off


def test_goal_makes_off_path_landmark_relevant():
    pol = PCACSPolicy(PCACSConfig(), goal_tags=["elevator"])
    elev = mk(1, "elevator", -1.8, 5.0, conf=0.85)
    elev.path_prob = 0.02
    cues = pol.decide([elev], 2.0, ChannelState())
    assert cues and cues[0].label == "elevator"
    pol2 = PCACSPolicy(PCACSConfig())
    assert pol2.decide([mk(1, "elevator", -1.8, 5.0, conf=0.85)], 2.0, ChannelState()) == []


def test_compose_hedges_uncertain_depth():
    assert compose(["object", "distance"], "backpack", distance_m=1.4, depth_conf=0.3) == "Backpack, very close."
    assert compose(["object", "distance"], "backpack", distance_m=1.4, depth_conf=0.9) == "Backpack, about 1.5 meters."
    assert speech_seconds("Stairs down, about 3 meters. Handrail left.") > 2.0
