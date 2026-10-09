"""EgoBlind integration: weak labels from questions, perception cache replay, oracle / P-CACS runs, trainer plumbing.
No model downloads, no videos: the cache is synthetic."""
import numpy as np

from seeforme.campus.metrics import evaluate_run, score
from seeforme.campus.oracle import CampusOracle
from seeforme.config import PipelineConfig
from seeforme.cues import ConfidenceThresholdPolicy
from seeforme.cues.pcacs import PCACSPolicy
from seeforme.egoblind import (EgoBlindScenario, EgoNoise, PerceptionCache, Question, build_scenario, egoblind_extras, events_for_video,
                               match_concepts, normalize_type, refine_with_cache, run_on_egoblind, weak_label)
from seeforme.egoblind.cache import synthetic_cache
from seeforme.egoblind.data import answer_polarity, question_kind
from seeforme.geometry import CameraModel
from seeforme.ml.scheduler_model import SchedulerDataset, SchedulerModel, collect, N_IN


def q(qid, t, qtype, question, answers, vid=1):
    raw = {"safety": "safety warnings", "navigation": "navigation", "information": "information reading"}[qtype]
    return Question(qid, vid, "train", t, qtype, raw, question, answers)


def synthetic_scenario(with_cache=True):
    qs = [q("v_1_1", 10.0, "safety", "Is it safe to cross?", ["No, a car is coming."]),
          q("v_1_2", 20.0, "safety", "Is there any obstacle ahead?", ["No."]),
          q("v_1_3", 25.0, "information", "What am I holding?", ["A cup."]),
          q("v_1_4", 1.0, "navigation", "Where is the exit?", ["Go straight."])]
    cam = CameraModel()
    z = lambda t: max(0.8, 12.0 - (t - 4.0) * 1.9)
    tracks = [dict(label="car", t0=4.0, t1=10.5, bbox=lambda t: cam.clip(cam.project_box(0.0, z(t), 1.5, 1.8)), dist=z, conf=0.9),
              dict(label="chair", t0=0.0, t1=30.0, bbox=(0.80, 0.55, 0.95, 0.80), dist=5.0, conf=0.6)]
    cache = synthetic_cache(tracks, duration=30.0)
    return build_scenario(1, qs, "train", cache=cache if with_cache else None), cache


def test_type_normalisation_and_matching():
    assert normalize_type("local navigation and positioning") == "navigation" == normalize_type("navigation")
    assert normalize_type("Information reading") == "information"
    coco, other = match_concepts("No, there are cars crossing the road ahead; find pedestrians to cross the zebra crossing")
    assert coco[:2] == ["car", "person"] and "zebra" not in coco and "crossing" in other and "road" in other
    coco, other = match_concepts("a pedestrian pushing a luggage cart six or seven steps ahead")
    assert "person" in coco and "suitcase" in coco and "cart" in other and "stairs" not in other
    assert match_concepts("hot dog stand")[0] == [] and "bicycle" in match_concepts("shared bicycles blocking the road")[0]


def test_weak_label_polarity():
    assert question_kind("Is it safe to cross the road?") == "safety_q" and question_kind("Is there any obstacle ahead?") == "presence_q"
    assert answer_polarity("No, a car is coming") == "no" and answer_polarity("I don't know.") == "unknown" and answer_polarity("Yes.") == "yes"
    assert weak_label(q("a", 5, "safety", "Is it safe to cross?", ["Yes."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Is there any obstacle ahead?", ["No."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Is there a clear path to walk through?", ["Yes, the path is clear.", "Yes."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Any obstacles on the floor?", ["No, the floor is clear of obstacles."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Any obstacle between me and the train?", ["No, there are no obstacles between you and the train entrance."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Is there any obstacle ahead?", ["No.", "There is a pedestrian six paces ahead."])).reason == "ambiguous_clear"
    w = weak_label(q("a", 5, "safety", "Is the path clear?", ["Yes, the path ahead is mostly clear, but there are people walking in the area."]))
    assert w.status == "present" and w.labels == ["person"]
    assert weak_label(q("a", 5, "safety", "Is it safe to board?", ["Yes, it is safe."])).status == "clear"
    w = weak_label(q("a", 5, "safety", "Is it safe to cross now?", ["No, a car is coming."]))
    assert w.status == "present" and w.labels == ["car"]
    assert weak_label(q("a", 5, "safety", "Is there anything I might hit?", ["Yes."])).reason == "present_unnamed"
    assert weak_label(q("a", 5, "safety", "Any traffic nearby?", ["I don't know.", "Yes, a tricycle on your right."])).labels[:1] == ["bicycle"]
    assert answer_polarity("No, it is not safe to step forward right now.") == "no"
    assert answer_polarity("Unsafe.") == "no" and answer_polarity("Not safe.") == "no"
    assert weak_label(q("a", 5, "safety", "Is it safe for me to step forward?",
                        ["No, it is not safe to step forward right now.", "Unsafe.", "No."])).status == "present"
    w = weak_label(q("a", 5, "safety", "Is it safe for me to cross the road now?",
                     ["There are cars coming. It is not safe to cross the street now."]))
    assert w.status == "present" and w.labels[:1] == ["car"]
    assert weak_label(q("a", 5, "navigation", "Where is the exit?", ["Go straight."])).status == "skip"
    assert weak_label(q("a", 5, "information", "What is this?", ["A car."])).status == "skip"


def test_events_for_video_windows():
    sc, _ = synthetic_scenario(with_cache=False)
    speak, quiet = sc.speak_events(), sc.quiet_events()
    assert len(speak) == 1 and len(quiet) == 1 and len(sc.queries) == 2
    e = speak[0]
    assert e.label == "car" and e.event_type == "vehicle" and e.severity == 1.0 and e.detail["coverable"]
    assert (e.earliest_useful_s, e.ideal_s, e.latest_safe_s) == (4.0, 8.0, 10.5) and not e.sudden
    assert "motion" in e.minimum_information and e.interrupt_current_speech
    assert (quiet[0].earliest_useful_s, quiet[0].latest_safe_s) == (16.0, 20.5) and not quiet[0].should_speak
    assert sc.kind == "egoblind_safety" and sc.duration == 28.0 and sc.question_times == [1.0, 10.0, 20.0, 25.0]
    sudden = events_for_video([q("s", 0.5, "safety", "Is it safe?", ["No, a bus is coming."])])[0]
    assert sudden[0].sudden and sudden[0].label == "bus"


def test_cache_roundtrip_and_source_ids(tmp_path):
    sc, cache = synthetic_scenario()
    p = tmp_path / "1.npz"; cache.save(p)
    c2 = PerceptionCache.load(p)
    assert c2.n_frames == cache.n_frames and c2.n_detections == cache.n_detections and c2.labels == cache.labels
    frames = c2.frames(sc.speak_events())
    car = {round(t, 2): [d for d in dets if d.label == "car"] for t, dets in frames}
    assert all(d.source_id == 1 for d in car[6.0]) and all(d.distance_m is not None for d in car[6.0])
    assert all(d.source_id is None for t, dets in frames for d in dets if d.label == "chair")
    none_dets = [d for t, dets in c2.frames(sc.speak_events(), noise=EgoNoise(p_miss=1.0)) for d in dets]
    assert none_dets == []


def test_refine_with_cache_uses_first_detection():
    sc, cache = synthetic_scenario()
    e = sc.speak_events()[0]
    assert e.detail["detected"] is True and e.detail["first_seen_s"] == 4.0
    assert e.earliest_useful_s == 4.0 and e.ideal_s == 8.0 and e.latest_safe_s == 10.5
    assert e.distance_at_ideal_m is not None and 3.0 < e.distance_at_ideal_m < 5.5 and "Car" in e.ideal_cue
    assert sc.duration == 30.0 and sc.stats["detected"] == 1


def test_json_roundtrip(tmp_path):
    sc, cache = synthetic_scenario()
    p = tmp_path / "1.json"; sc.to_json(p)
    sc2 = EgoBlindScenario.from_json(p)
    assert [e.as_dict() for e in sc2.events()] == [e.as_dict() for e in sc.events()] and sc2.queries == sc.queries


def test_oracle_pcacs_and_baseline_replay():
    sc, cache = synthetic_scenario()
    cfg = PipelineConfig()
    o_log = run_on_egoblind(CampusOracle(sc), sc, cfg)
    o = evaluate_run(o_log, sc)
    assert o.action_weighted_recall == 1.0 and o.n_spoken == 1 and egoblind_extras(o_log, sc)["quiet_violation_rate"] == 0.0
    ex = egoblind_extras(o_log, sc)
    assert ex["anticipation_lead_s"] > 1.5 and ex["coverable_recall"] == 1.0 and ex["detected_recall"] == 1.0
    p_log = run_on_egoblind(PCACSPolicy(cfg.pcacs), sc, cfg)
    p = evaluate_run(p_log, sc)
    assert 0.0 <= p.audio_occupancy <= 1.0 and p.n_events == 1
    b = evaluate_run(run_on_egoblind(ConfidenceThresholdPolicy(cfg.baseline), sc, cfg), sc)
    assert score(o_log, sc) >= score(p_log, sc) and b.cues_per_min > p.cues_per_min


def test_collect_and_fit_on_egoblind_replay():
    sc, cache = synthetic_scenario()
    cfg = PipelineConfig()
    data = collect(lambda s: PCACSPolicy(cfg.pcacs), [sc], cfg, K=8, run=run_on_egoblind)
    data.extend(collect(lambda s: CampusOracle(s), [sc], cfg, K=8, run=run_on_egoblind))
    assert len(data) > 20 and all(len(w[-1]) == N_IN for w in data.windows) and 1 in data.action   # some "speak" labels
    m = SchedulerModel(K=8, epochs=2).fit(data)
    la, lu, ls = m.forward(data.windows[:5])
    assert la.shape == (5, 4) and len(m.history) == 2


def test_false_alarms_count_only_near_questions():
    from seeforme.campus.runner import RunLog
    from seeforme.types import Cue
    qs = [q("v_2_1", 10.0, "safety", "Is it safe to cross?", ["No, a car is coming."])]
    sc = build_scenario(2, qs, "train")
    sc.duration = 40.0
    assert sc.counts_fp(12.0) and sc.counts_fp(15.9) and not sc.counts_fp(30.0)
    syn, _ = synthetic_scenario(with_cache=False)       # questions at 1 (navigation, no event), 10 (car), 20 (quiet), 25 (information)
    assert syn.labelled_times == [10.0, 20.0] and syn.counts_fp(14.0) and syn.counts_fp(22.0) and not syn.counts_fp(27.0) and not syn.counts_fp(2.0)
    far = RunLog(scenario=sc.name, policy="p"); far.cues = [Cue(t=30.0, track_id=5, label="chair", kind="object", message="Chair.", score=1.0, policy="p", duration_s=1.2)]
    far.channel_log = [[30.0, 31.2, far.cues[0]]]
    near = RunLog(scenario=sc.name, policy="p"); near.cues = [Cue(t=12.0, track_id=5, label="chair", kind="object", message="Chair.", score=1.0, policy="p", duration_s=1.2)]
    near.channel_log = [[12.0, 13.2, near.cues[0]]]
    assert score(far, sc) > score(near, sc)          # the same off-event cue is a false alarm near the question, unknown far from it
    from seeforme.campus import make_campus_scenario
    csc = make_campus_scenario("hallway", 1)
    assert not hasattr(csc, "counts_fp")              # CampusBench counts every off-event cue


def test_question_kind_decides_what_yes_means():
    """'Is the path clear?' and 'is it safe?': yes = clear. 'Is it dangerous?' and 'is there anything?': yes = something there."""
    assert question_kind("Is the path ahead of me clear?") == "clear_q"
    assert question_kind("Is there a risk of tripping around the counter?") == "hazard_q"
    assert question_kind("Is there any danger, is the road clear?") == "hazard_q"      # first strong cue decides
    assert question_kind("Is the elevator door open?") == "other"
    assert question_kind("Are there any open flames nearby?") == "presence_q"
    assert weak_label(q("a", 5, "safety", "Is the path ahead of me clear?", ["Yes."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Is it clear to cross the street now?", ["Yes."])).status == "clear"
    w = weak_label(q("a", 5, "safety", "Is the path clear to the door?", ["No."]))
    assert w.status == "present" and w.reason == "present_unnamed" and "door" not in w.concepts   # the door is the destination
    w = weak_label(q("a", 5, "safety", "Is the counter clear of knives?", ["No."]))
    assert w.status == "present" and w.labels == ["knife"]
    w = weak_label(q("a", 5, "safety", "Is the path ahead clear or are there obstacles?", ["No, there is a parked bicycle ahead."]))
    assert w.status == "present" and "bicycle" in w.labels
    assert weak_label(q("a", 5, "safety", "Are there any sharp or dangerous objects near my hands?", ["No."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Is there a risk of tripping here?", ["No."])).status == "clear"
    assert weak_label(q("a", 5, "safety", "Is it dangerous to walk here?", ["Yes."])).status == "present"
    # a plain 'yes' to a clear / safety question that names a new object is ambiguous, not a hazard
    assert weak_label(q("a", 5, "safety", "Is the path clear?", ["Yes.", "There is a bench on your left."])).reason == "ambiguous_clear"


def test_refine_with_cache_is_idempotent_across_caches():
    """Refining with a second cache must not keep the first cache's window or detection."""
    sc, cache = synthetic_scenario()
    ev = next(e for e in sc.speak_events() if e.label == "car")
    refined = (ev.earliest_useful_s, ev.detail["detected"])
    base = tuple(ev.detail["base"]["window"])
    empty = synthetic_cache([dict(label="chair", t0=0.0, t1=30.0, bbox=(0.8, 0.55, 0.95, 0.8), dist=5.0, conf=0.6)], duration=30.0)
    refine_with_cache(sc, empty)
    assert ev.detail["detected"] is False and (ev.earliest_useful_s, ev.ideal_s, ev.latest_safe_s) == base
    refine_with_cache(sc, cache)
    assert (ev.earliest_useful_s, ev.detail["detected"]) == refined


def test_open_vocabulary_cache_maps_aliases():
    """A dog event is decor for COCO but an 'animal' for the open-vocabulary detector; detections map to the event."""
    sc = build_scenario(2, [q("v_2_1", 10.0, "safety", "Is there anything I should watch out for?", ["Yes, a dog is in front of you."], vid=2)], "train")
    ev = sc.speak_events()[0]
    assert ev.detail["labels"] == ["dog"] and not ev.detail["coverable"]
    ow = synthetic_cache([dict(label="animal", t0=4.0, t1=11.0, bbox=(0.4, 0.5, 0.6, 0.9), dist=3.0, conf=0.8)], duration=20.0,
                         meta={"vocab": ["animal", "person", "door"]})
    refine_with_cache(sc, ow)
    assert ev.detail["coverable"] and ev.detail["detected"] and ev.detail["det_labels"] == ["animal"]
    frames = ow.frames(sc.speak_events())
    assert any(d.source_id == ev.obj_id for _, dets in frames for d in dets)


def test_70_20_10_split_is_by_source_recording():
    """No source recording may have clips in two of train / finetune / select / holdout or in the official test, and the
    shares are 70 / 15 / 5 / 10
    (needs data/egoblind/meta from the downloader)."""
    import sys
    from pathlib import Path
    import pytest
    root = Path(__file__).resolve().parents[1]
    if not (root / "data" / "egoblind" / "meta" / "video_source.csv").exists():
        pytest.skip("EgoBlind metadata not downloaded")
    sys.path.insert(0, str(root / "scripts"))
    import egoblind_common as ec
    assign, src = ec.train_pool_assignment(), ec.source_recordings()
    seen = {}
    for vid, split in assign.items():
        seen.setdefault(src.get(vid, f"clip-{vid}"), set()).add(split)
    assert all(len(s) == 1 for s in seen.values())
    # overlapping rows in video_source.csv are one recording; clips without a row train only; clips sharing a recording
    # with the official test are excluded from every split
    assert len({assign[v] for v in (665, 666, 667, 668, 669) if v in assign}) <= 1
    assert len({assign[v] for v in (715, 716, 717, 719, 720, 728) if v in assign}) <= 1
    assert all(assign[v] == "train" for v in assign if v not in src)
    test_recs = {src[v] for v in ec.load_egoblind_questions("test") if v in src}
    assert all(assign[v] == "excluded" for v in assign if src.get(v) in test_recs) and assign.get(794) == "excluded"
    n = len(assign)
    sizes = {s: sum(1 for v in assign.values() if v == s) / n for s in ("train", "finetune", "select", "holdout")}
    assert sum(1 for v in assign.values() if v == "excluded") <= 8
    # 70 / 15 / 5 / 10 up to one recording of overshoot per held-out split
    assert abs(sizes["holdout"] - 0.10) < 0.03 and abs(sizes["select"] - 0.05) < 0.02 and abs(sizes["finetune"] - 0.15) < 0.03
    assert 0.64 < sizes["train"] < 0.72
    assert not set(ec.load_egoblind_questions("test")) & set(assign)


def test_clock_directions_are_not_clocks():
    assert match_concepts("In front of you at two o'clock on your right.") == ([], [])
    assert match_concepts("A car at 2 o'clock, a wall clock above the door")[0] == ["car", "clock"]
