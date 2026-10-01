"""Common interface for cue-selection policies (baselines and CACS)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import Cue, Track


class CuePolicy(ABC):
    name: str = "policy"
    decision_interval: float = 0.5

    @abstractmethod
    def select(self, tracks: List[Track], t: float, scene_ctx=None) -> List[Cue]:
        """Return the cues to speak at decision time t, most important first."""

    def reset(self) -> None:  # pragma: no cover
        pass

    def last_observations(self):
        """Feature-level view of the last decision, for the debug overlay."""
        return []
