"""Live-demo layer on top of any scheduler (used by scripts/run_live.py only; benchmarks never import it).

    AnnouncementGate   remembers what was said about each object and direction, so a static object is not repeated,
                       a track-id switch does not re-announce, and approaching or critical objects still get through
    ActivityDetector   stationary / walking / turning / reading, from camera motion and visible text
    TaskAdapter        turns the activity (and a FIND goal) into a mode and the gate / verbosity settings for it
"""
from .gate import AnnouncementGate, GateConfig
from .activity import Activity, ActivityConfig, ActivityDetector, TaskAdapter

__all__ = ["AnnouncementGate", "GateConfig", "Activity", "ActivityConfig", "ActivityDetector", "TaskAdapter"]
