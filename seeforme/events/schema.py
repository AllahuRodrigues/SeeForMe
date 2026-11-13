"""Canonical event contract shared by perception, the usefulness model, the scheduler and every runtime.

A SceneEvent carries only what a deployed system can know at decision time: no benchmark event ids, no question
times, no annotations. Evaluation adapters map events to ground truth outside this contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class SceneEvent:
    event_id: str
    t: float                                   # decision time (s)
    category: str                              # e.g. "object", "stairs", "door", "scene_change"
    track_ids: Tuple = ()
    p_helpful: float = 0.0                     # P(proactively helpful)       (usefulness head)
    p_wrong: float = 0.0                       # P(statement wrong/misleading) (claim-reliability head)
    p_redundant: float = 0.0                   # P(redundant given what was said)
    value: float = 1.0                         # value of delivering it now
    duration_s: float = 1.5                    # expected speech duration
    deadline_t: Optional[float] = None         # latest useful time
    expired: bool = False
    announced: bool = False
    grounded: bool = True                      # the claim is supported by current observations
    text: str = ""
    model_version: str = ""
    extra: Dict = field(default_factory=dict)


@dataclass
class CommunicationDecision:
    action: str                                # "SILENCE" | "SPEAK" | "INTERRUPT" | "HAPTIC"
    reason_code: str
    event_id: Optional[str] = None
    utility: float = 0.0
    text: str = ""
    skipped: Dict[str, str] = field(default_factory=dict)
