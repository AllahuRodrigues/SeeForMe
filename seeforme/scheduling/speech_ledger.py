"""Strict speech budget: a token bucket whose tokens are seconds of speech.

Tokens refill at `refill_per_s` up to `capacity_s`; a cue is accepted only if its reserved duration can be paid.
Guarantee (bounded expenditure): for every interval (s, t], the reserved duration of accepted cues is at most
capacity_s + refill_per_s * (t - s). Proof: the balance at s is at most C, it gains at most r (t - s), and spending
never makes it negative. This bounds reserved duration of cues *started* in the interval; playback occupancy and
interruptions are tracked by the audio channel. There is no hidden urgent bypass.
"""
from __future__ import annotations


class SpeechLedger:
    def __init__(self, capacity_s: float, refill_per_s: float, start_full: bool = True):
        if capacity_s <= 0 or refill_per_s < 0:
            raise ValueError("capacity must be positive and refill non-negative")
        self.capacity_s, self.refill_per_s = float(capacity_s), float(refill_per_s)
        self._bal = self.capacity_s if start_full else 0.0
        self._t = None

    def _advance(self, t: float) -> None:
        if self._t is not None:
            if t < self._t - 1e-9:
                raise ValueError(f"time went backwards: {t} < {self._t}")
            self._bal = min(self.capacity_s, self._bal + self.refill_per_s * (t - self._t))
        self._t = t if self._t is None else max(self._t, t)

    def balance(self, t: float) -> float:
        self._advance(t)
        return self._bal

    def can_spend(self, d: float, t: float) -> bool:
        return self.balance(t) + 1e-12 >= d

    def try_spend(self, d: float, t: float) -> bool:
        if d < 0:
            raise ValueError("duration must be non-negative")
        if not self.can_spend(d, t):
            return False
        self._bal -= d
        return True

    def reset(self) -> None:
        self._bal, self._t = self.capacity_s, None
