"""Long-term re-identification gallery: what the assistant still remembers after an object leaves the view.

The tracker's ledger (memory_ttl_s, ~3 s) covers short occlusions. This gallery covers minutes: when a confirmed track
with an appearance embedding dies, its (label, embedding, announced state) is kept for `horizon_s`. A new track of
the same label whose embedding has cosine >= `threshold` with a gallery entry is the same object coming back; it
inherits the announced state, so the scheduler treats it as known rather than new. Each entry is consumed by its
first match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class MemoryEntry:
    label: str
    kind: str
    emb: np.ndarray
    last_seen: float
    last_announced: Optional[float]
    announce_count: int
    track_id: int
    source_id: Optional[int] = None     # ground-truth object id when known (evaluation only)


class ReIDGallery:
    def __init__(self, horizon_s: float = 180.0, threshold: float = 0.85, max_entries: int = 256):
        self.horizon_s, self.threshold, self.max_entries = horizon_s, threshold, max_entries
        self._entries: List[MemoryEntry] = []
        self.last_score: Optional[float] = None    # cosine of the most recent successful match

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, label: str, kind: str, emb: np.ndarray, last_seen: float, last_announced: Optional[float],
            announce_count: int, track_id: int, source_id: Optional[int] = None) -> None:
        e = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(e))
        if n <= 1e-6:
            return
        self._entries.append(MemoryEntry(label, kind, e / n, last_seen, last_announced, announce_count, track_id, source_id))
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

    def match(self, label: str, kind: str, emb: np.ndarray, t: float) -> Optional[MemoryEntry]:
        """Best entry of the same label and kind seen within the horizon with cosine >= threshold; consumed on match."""
        self._entries = [e for e in self._entries if t - e.last_seen <= self.horizon_s]
        q = np.asarray(emb, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n <= 1e-6:
            return None
        q = q / n
        best, best_i, best_s = None, -1, self.threshold
        for i, e in enumerate(self._entries):
            if e.label != label or e.kind != kind:
                continue
            s = float(e.emb @ q)
            if s >= best_s:
                best, best_i, best_s = e, i, s
        if best is not None:
            self._entries = self._entries[:best_i] + self._entries[best_i + 1:]
            self.last_score = best_s
        return best

    def clear(self) -> None:
        self._entries = []
