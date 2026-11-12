"""Live-demo layer on top of any scheduler (used by scripts/run_live.py only; benchmarks never import it).

    AnnouncementGate   remembers what was said about each object and direction, so a static object is not repeated,
                       a track-id switch does not re-announce, and approaching or critical objects still get through
    ActivityDetector   stationary / walking / turning / reading, from camera motion and visible text
    TaskAdapter        turns the activity (and a FIND goal) into a mode and the gate / verbosity settings for it
    Cane               class-agnostic obstacles from the metric depth map (walls, doors, poles, low and head-height
                       things the detector cannot name): what a white cane and a guide dog would tell the user
"""
from .gate import AnnouncementGate, GateConfig
from .activity import Activity, ActivityConfig, ActivityDetector, TaskAdapter
from .cane import Cane, CaneConfig, CaneFrame, analyze as analyze_depth

__all__ = ["AnnouncementGate", "GateConfig", "Activity", "ActivityConfig", "ActivityDetector", "TaskAdapter", "Cane",
           "CaneConfig", "CaneFrame", "analyze_depth"]
