from seeforme.config import PipelineConfig
from seeforme.cues import CACSPolicy, ConfidenceThresholdPolicy
from seeforme.eval import aggregate, evaluate_cues, run_policy_on_scenario
from seeforme.sim.scenario import Scenario, SceneObject, make_benchmark, make_scenario
from seeforme.types import Cue


def test_metrics_on_handmade_cues():
    obj = SceneObject(obj_id=1, label="chair", t_appear=0, t_disappear=10, important_windows=[(2.0, 8.0)])
    decor = SceneObject(obj_id=2, label="poster", t_appear=0, t_disappear=10)
    sc = Scenario(name="unit", duration=60.0, fps=10, objects=[obj, decor])
    cues = [Cue(t=3.0, track_id=1, label="chair", kind="object", message="chair", score=1, policy="p", source_id=1),
            Cue(t=4.0, track_id=1, label="chair", kind="object", message="chair", score=1, policy="p", source_id=1),
            Cue(t=5.0, track_id=2, label="poster", kind="object", message="poster", score=1, policy="p", source_id=2),
            Cue(t=6.0, track_id=9, label="cup", kind="object", message="cup", score=1, policy="p", source_id=None)]
    m = evaluate_cues(cues, sc, "p")
    assert m.important_cue_recall == 1.0
    assert m.cue_precision == 0.25            # only the first chair cue is useful
    assert m.duplicate_rate == 0.25           # second chair cue is a repeat
    assert m.n_false_positive_cues == 1
    assert abs(m.mean_critical_delay - 1.0) < 1e-9
    assert m.audio_load_per_min == 4.0


def test_scenario_json_roundtrip(tmp_path):
    sc = make_scenario("office", 3)
    p = tmp_path / "s.json"; sc.to_json(str(p))
    sc2 = Scenario.from_json(str(p))
    assert sc2.name == sc.name and len(sc2.objects) == len(sc.objects)
    assert sc2.important_events() == sc.important_events()


def test_cacs_talks_less_than_confidence_baseline():
    cfg = PipelineConfig()
    scs = make_benchmark(n_per_kind=1, seed0=7)
    rows = []
    for sc in scs:
        for pol in [ConfidenceThresholdPolicy(cfg.baseline), CACSPolicy(cfg.cacs)]:
            rows.append(evaluate_cues(run_policy_on_scenario(pol, sc, cfg), sc, pol.name))
    a = aggregate(rows)
    assert a["cacs"]["audio_load_per_min"] < a["confidence_threshold"]["audio_load_per_min"]
    assert a["cacs"]["duplicate_rate"] < a["confidence_threshold"]["duplicate_rate"]
