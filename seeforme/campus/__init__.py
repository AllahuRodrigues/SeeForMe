"""CampusBench: a metric egocentric benchmark for predictive cue scheduling.

Objects live in metres relative to a walking user; bounding boxes come from a pinhole
camera; a noisy detector *and* a noisy depth sensor produce the observations. Ground
truth is no longer a binary "important" flag but a communication window per event:
earliest useful time, ideal time, latest safe time, severity, the minimum information
a user needs, and the ideal cue text.
"""
from .world import WorldObject, CampusScenario, EventAnnotation, EVENT_SPECS
from .scenes import make_campus_scenario, make_campus_benchmark, CAMPUS_KINDS, CAMPUS_KINDS_V1, OOD_KINDS
from .simulator import CampusDetectorSimulator

__all__ = ["WorldObject", "CampusScenario", "EventAnnotation", "EVENT_SPECS", "make_campus_scenario",
           "make_campus_benchmark", "CAMPUS_KINDS", "CAMPUS_KINDS_V1", "OOD_KINDS", "CampusDetectorSimulator"]
