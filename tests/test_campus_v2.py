"""CampusBench v2 mechanics: occlusion, camera pans, external speech, unknown obstacles, the memory ledger,
reaction-time scaling, the four new scene kinds and seed stability of the original kinds."""
import json
from dataclasses import replace

from seeforme.campus import CAMPUS_KINDS, CAMPUS_KINDS_V1, CampusScenario, make_campus_scenario
from seeforme.campus.metrics import evaluate_run, score
from seeforme.campus.oracle import CampusOracle
from seeforme.campus.runner import RunLog, distractor_cue, run_on_campus
from seeforme.campus.world import EVENT_SPECS, WorldObject, default_event_type
from seeforme.config import PipelineConfig, TrackerConfig
from seeforme.cues.pcacs import PCACSPolicy
from seeforme.cues.sign_vocab import taxonomy_rows, translate
from seeforme.cues.text_importance import normalise_text
from seeforme.tracking.tracker import IoUTracker
from seeforme.types import Detection


def test_new_kinds_build_annotate_and_roundtrip(tmp_path):
    for kind in ("contention", "occlusion", "wayfinding_airport", "ood_aftermath"):
        assert kind in CAMPUS_KINDS and kind not in CAMPUS_KINDS_V1
        sc = make_campus_scenario(kind, 7)
        ev = sc.events()
        assert len(ev) == len(sc.objects)
        for e in ev:
            if e.should_speak:
                assert e.earliest_useful_s <= e.ideal_s <= e.latest_safe_s and e.ideal_cue
        p = tmp_path / f"{kind}.json"; sc.to_json(str(p))
        sc2 = CampusScenario.from_json(str(p))
        assert [e.as_dict() for e in sc2.events()] == [e.as_dict() for e in ev]
        assert sc2.distractor_speech == sc.distractor_speech and sc2.yaw_keyframes == sc.yaw_keyframes
    assert make_campus_scenario("contention", 1).distractor_speech and len(make_campus_scenario("contention", 1).distractor_speech) == 2
    assert make_campus_scenario("ood_aftermath", 2).speak_events() and any(e.event_type == "unknown_obstacle" for e in make_campus_scenario("ood_aftermath", 2).speak_events())


def test_v1_kinds_unchanged_by_v2():
    sc = make_campus_scenario("hallway", 2001)
    assert sc.distractor_speech == [] and sc.yaw_keyframes == [] and sc.reaction_scale == 1.0
    assert sc.pan_deg(10.0) == 0.0


def test_occlusion_and_pan_hide_objects():
    sc = make_campus_scenario("hallway", 5)
    o = next(x for x in sc.objects if x.label == "chair")
    t_vis = next(t / 10 for t in range(0, 300) if o.visible(t / 10))
    o.occluded_windows = [(t_vis, t_vis + 2.0)]
    assert not o.visible(t_vis + 1.0) and o.occluded(t_vis + 1.0)
    sc.yaw_keyframes = [(0.0, 0.0), (0.5, 60.0), (3.0, 60.0), (3.5, 0.0)]
    assert sc.pan_deg(1.5) == 60.0 and sc.pan_deg(0.25) == 30.0 and sc.pan_deg(9.0) == 0.0
    ahead = WorldObject(obj_id=99, label="box", keyframes=[(0.0, 0.0, 3.0)]); ahead._scene = sc
    assert ahead.bbox_at(0.0) is not None and ahead.bbox_at(1.5) is None      # 60 degrees of pan takes it out of a 70 degree FOV


def test_reaction_scale_moves_the_deadline():
    base = make_campus_scenario("stairs", 3)
    slow = make_campus_scenario("stairs", 3); slow.reaction_scale = 2.0
    e0 = next(e for e in base.speak_events() if e.label == "stairs")
    e1 = next(e for e in slow.speak_events() if e.label == "stairs")
    assert e1.latest_safe_s < e0.latest_safe_s and e1.t_contact == e0.t_contact


def test_external_speech_occupies_channel_and_can_be_interrupted():
    cfg = PipelineConfig()
    sc = make_campus_scenario("contention", 4)
    log = run_on_campus(CampusOracle(sc), sc, cfg)
    assert len(log.distractors) == 2 and all(c.policy == "external" for c in log.distractors)
    assert any(a <= 8.0 <= b or a <= 9.0 <= b for a, b, c in log.channel_log if c.policy == "external")
    m = evaluate_run(log, sc)
    assert m.n_interrupts >= 1 and all(c.track_id != -1 for c in log.cues)
    # a run with no distractors at all is allowed too
    log2 = run_on_campus(PCACSPolicy(cfg.pcacs), sc, cfg, distractors=[])
    assert log2.distractors == []


def test_external_speech_is_not_charged_to_the_policy():
    """Distractor speech occupies the channel but is not the policy's airtime: it leaves score() and audio_occupancy
    unchanged (so cutting it short earns no bonus) and still counts when it blocks a critical warning."""
    cfg = PipelineConfig()
    sc = make_campus_scenario("contention", 4)
    log = run_on_campus(CampusOracle(sc), sc, cfg)
    assert any(c.policy == "external" for _, _, c in log.channel_log)
    own = replace(log, channel_log=[x for x in log.channel_log if x[2].policy != "external"])
    cut = replace(log, channel_log=[[a, a + 0.1 if c.policy == "external" else b, c] for a, b, c in log.channel_log])
    assert score(log, sc) == score(own, sc) == score(cut, sc)
    assert evaluate_run(log, sc).audio_occupancy == evaluate_run(own, sc).audio_occupancy == evaluate_run(cut, sc).audio_occupancy
    # a silent run while a low-severity external message plays across a critical deadline: blocked, but no airtime charged
    sc = make_campus_scenario("stairs", 5)
    e = next(e for e in sc.speak_events() if e.severity >= 0.7)
    blocked = RunLog(scenario=sc.name, policy="silent")
    blocked.channel_log = [[e.latest_safe_s - 1.0, e.latest_safe_s + 1.0, distractor_cue(e.latest_safe_s - 1.0, 2.0, 0.3, cfg.speech)]]
    m = evaluate_run(blocked, sc)
    assert m.critical_blocking_rate > 0 and m.audio_occupancy == 0.0
    assert score(blocked, sc) == score(RunLog(scenario=sc.name, policy="silent"), sc)


def test_unknown_obstacle_taxonomy_and_hedged_phrase():
    assert default_event_type("debris") == "unknown_obstacle" and EVENT_SPECS["unknown_obstacle"].severity > 0
    assert default_event_type("poster") == "decor" and default_event_type("chair") == "hazard_static"
    from seeforme.cues.compose import compose
    assert compose(["object", "distance"], "unknown object", distance_m=2.0, depth_conf=0.9, hedged=True).startswith("Possibly an obstacle")


def test_memory_ledger_keeps_decays_and_reassociates():
    cfg = TrackerConfig(max_age=2, n_init=1, memory_ttl_s=3.0, memory_decay=0.5)
    tr = IoUTracker(cfg)
    d = Detection("chair", 0.9, (0.4, 0.5, 0.6, 0.9), distance_m=4.0, depth_conf=0.9)
    tr.update([d], 0.0); tr.update([d], 0.1)
    for i in range(2, 8):                        # six empty frames: beyond max_age, inside the TTL
        active = tr.update([], i * 0.1)
    assert len(active) == 1 and active[0].remembered and active[0].confidence < 0.9
    # the chair reappears closer and bigger: re-associated instead of duplicated
    active = tr.update([Detection("chair", 0.9, (0.3, 0.45, 0.7, 0.95), distance_m=2.5, depth_conf=0.9)], 0.8)
    assert len(active) == 1 and not active[0].remembered and active[0].track_id == 1
    # without the ledger the track is dropped after max_age misses
    tr2 = IoUTracker(TrackerConfig(max_age=2, n_init=1))
    tr2.update([d], 0.0)
    for i in range(1, 5):
        active = tr2.update([], i * 0.1)
    assert active == []
    # a remembered track expires after the TTL
    tr3 = IoUTracker(cfg); tr3.update([d], 0.0)
    for i in range(1, 40):
        active = tr3.update([], i * 0.1)
    assert active == []


def test_memory_helps_on_long_occlusion():
    """With the ledger, the reference scheduler recovers every hidden hazard and P-CACS recovers more of them;
    the price (false alarms when the object was moved while hidden) is measured, not hidden."""
    cfg = PipelineConfig()
    mem = PipelineConfig(); mem.tracker = replace(mem.tracker, memory_ttl_s=12.0, memory_decay=0.9)
    rec = {"frame": 0.0, "mem": 0.0, "oracle": 0.0, "oracle+mem": 0.0}
    for seed in range(1, 17):
        sc = make_campus_scenario("occlusion", seed)
        if "'moved': True" in sc.description:
            continue
        rec["frame"] += evaluate_run(run_on_campus(PCACSPolicy(cfg.pcacs), sc, cfg), sc).action_weighted_recall
        rec["mem"] += evaluate_run(run_on_campus(PCACSPolicy(mem.pcacs), sc, mem), sc).action_weighted_recall
        rec["oracle"] += evaluate_run(run_on_campus(CampusOracle(sc), sc, cfg), sc).action_weighted_recall
        rec["oracle+mem"] += evaluate_run(run_on_campus(CampusOracle(sc), sc, mem), sc).action_weighted_recall
    assert rec["oracle+mem"] > rec["oracle"] and rec["mem"] >= rec["frame"]


def test_sign_vocab_and_accent_normalisation():
    assert translate("exit", "pt") == "SAÍDA" and translate("exit", "xx") == "EXIT" and translate("nonsense") == "NONSENSE"
    assert normalise_text("SAÍDA DE EMERGÊNCIA") == "saida de emergencia"
    rows = taxonomy_rows()
    assert len(rows) > 150 and {r["language"] for r in rows} == {"en", "pt", "es", "fr"} and any(r["label"] == 0 for r in rows)
