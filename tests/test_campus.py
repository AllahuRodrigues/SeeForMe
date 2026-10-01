import json
from seeforme.campus import CampusScenario, make_campus_scenario, CAMPUS_KINDS, CampusDetectorSimulator
from seeforme.campus.metrics import evaluate_run, score
from seeforme.campus.oracle import CampusOracle
from seeforme.campus.runner import run_on_campus
from seeforme.config import PipelineConfig
from seeforme.cues import ConfidenceThresholdPolicy
from seeforme.cues.pcacs import PCACSPolicy


def test_every_kind_builds_and_annotates():
    for kind in CAMPUS_KINDS:
        sc = make_campus_scenario(kind, 11)
        ev = sc.events()
        assert len(ev) == len(sc.objects)
        for e in ev:
            if e.should_speak:
                assert e.earliest_useful_s <= e.ideal_s <= e.latest_safe_s
                assert e.ideal_cue and "object" in e.minimum_information
    assert not make_campus_scenario("negative", 11).speak_events()


def test_stairs_window_is_before_contact_and_has_action_slot():
    sc = make_campus_scenario("stairs", 5)
    st = [e for e in sc.speak_events() if e.label == "stairs"]
    assert st and all(e.latest_safe_s < e.t_contact and "action" in e.minimum_information for e in st)
    assert "Stairs" in st[0].ideal_cue and "Handrail" in st[0].ideal_cue


def test_json_roundtrip(tmp_path):
    sc = make_campus_scenario("crossing", 7)
    p = tmp_path / "s.json"; sc.to_json(str(p))
    sc2 = CampusScenario.from_json(str(p))
    assert [e.as_dict() for e in sc2.events()] == [e.as_dict() for e in sc.events()]
    d = json.loads(p.read_text())
    assert "events" in d and d["events"][0].keys() >= {"should_speak", "earliest_useful_s", "ideal_cue", "minimum_information"}


def test_depth_simulator_adds_distances():
    sc = make_campus_scenario("hallway", 2)
    dets = [d for _, ds in CampusDetectorSimulator(sc).frames() for d in ds]
    with_depth = [d for d in dets if d.distance_m is not None]
    assert len(with_depth) == len(dets) and all(0 < d.depth_conf <= 1 for d in dets)


def test_oracle_scores_best_and_pcacs_beats_confidence_baseline():
    cfg = PipelineConfig()
    tot = {"oracle": 0.0, "pcacs": 0.0, "confidence_threshold": 0.0}
    for kind in ("hallway", "stairs", "temporary_hazard", "negative"):
        sc = make_campus_scenario(kind, 21)
        tot["oracle"] += score(run_on_campus(CampusOracle(sc), sc, cfg), sc)
        tot["pcacs"] += score(run_on_campus(PCACSPolicy(cfg.pcacs, goal_tags=sc.goal_tags), sc, cfg), sc)
        tot["confidence_threshold"] += score(run_on_campus(ConfidenceThresholdPolicy(cfg.baseline), sc, cfg), sc)
    assert tot["oracle"] >= tot["pcacs"] > tot["confidence_threshold"]


def test_metrics_fields():
    cfg = PipelineConfig()
    sc = make_campus_scenario("negative", 3)
    m = evaluate_run(run_on_campus(PCACSPolicy(cfg.pcacs), sc, cfg), sc)
    assert m.n_events == 0 and m.audio_occupancy < 0.5 and 0 <= m.silence_precision <= 1
