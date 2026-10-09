"""EgoBlind (Xiao et al., NeurIPS 2025 D&B track) as a real-world, weakly labelled benchmark and
training source for the cue scheduler.

    data.py       CSV loading, question-type normalisation, object keyword matching, weak events
    scenario.py   EgoBlindScenario: duck-types CampusScenario for the runner, metrics, oracle and trainers
    cache.py      perception cache: YOLO + metric depth run once per video, replayed many times
    runner.py     closed-loop replay of a scheduler over a cached video (same channel semantics as CampusBench)
    metrics.py    EgoBlind-specific extras (quiet-window false alarms, anticipation lead, coverage)
"""
from .data import Question, load_questions, group_by_video, normalize_type, match_concepts, weak_label, events_for_video
from .scenario import EgoBlindScenario, build_scenario, refine_with_cache
from .cache import PerceptionCache, build_cache, EgoNoise
from .runner import run_on_egoblind, make_runner
from .metrics import egoblind_extras, aggregate_extras

__all__ = ["Question", "load_questions", "group_by_video", "normalize_type", "match_concepts", "weak_label", "events_for_video",
           "EgoBlindScenario", "build_scenario", "refine_with_cache", "PerceptionCache", "build_cache", "EgoNoise",
           "run_on_egoblind", "make_runner", "egoblind_extras", "aggregate_extras"]
