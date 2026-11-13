"""Strict speech budget: a token bucket whose tokens are seconds of speech.

Tokens refill at `refill_per_s` up to `capacity_s`; a cue is accepted only if its reserved duration can be paid.
Guarantee (bounded expenditure): for every interval (s, t], the reserved duration of accepted cues is at most
capacity_s + refill_per_s * (t - s). Proof: the balance at s is at most C, it gains at most r (t - s), and spending
never makes it negative. This bounds reserved duration of cues *started* in the interval. There is no hidden urgent
bypass.

Against real playback the reserved duration is only an estimate. `reserve` holds the estimate and `settle` replaces it
with the seconds that were actually played: speech that was interrupted, cancelled or never started gets the unplayed
part back (never above capacity), and speech that ran long is charged the overrun, which can leave a debt that keeps
the assistant silent until the refill repays it. With one audio channel, each cue is settled before the next is
accepted, so played seconds over any interval are at most capacity_s + refill_per_s * (running time) + the largest
single overrun. While the session is paused nothing refills and nothing is accepted.
"""
from __future__ import annotations


class SpeechLedger:
    def __init__(self, capacity_s: float, refill_per_s: float, start_full: bool = True):
        if capacity_s <= 0 or refill_per_s < 0:
            raise ValueError("capacity must be positive and refill non-negative")
        self.capacity_s, self.refill_per_s = float(capacity_s), float(refill_per_s)
        self._bal = self.capacity_s if start_full else 0.0
        self._t = None
        self._open = {}                 # reservation id -> reserved seconds, until playback is settled
        self._next_id = 1
        self._paused = False

    def _advance(self, t: float) -> None:
        if self._t is not None:
            if t < self._t - 1e-9:
                raise ValueError(f"time went backwards: {t} < {self._t}")
            if not self._paused:
                self._bal = min(self.capacity_s, self._bal + self.refill_per_s * (t - self._t))
        self._t = t if self._t is None else max(self._t, t)

    def balance(self, t: float) -> float:
        self._advance(t)
        return self._bal

    def can_spend(self, d: float, t: float) -> bool:
        return self.balance(t) + 1e-12 >= d and not self._paused

    def try_spend(self, d: float, t: float) -> bool:
        if d < 0:
            raise ValueError("duration must be non-negative")
        if not self.can_spend(d, t):
            return False
        self._bal -= d
        return True

    # ------------------------------------------------------------------ real playback
    @property
    def reserved_open(self) -> float:
        """Seconds reserved for cues whose playback has not been settled yet."""
        return float(sum(self._open.values()))

    def reserve(self, d: float, t: float):
        """Hold the estimated duration of a cue that is about to be spoken. Returns a reservation id, or None."""
        if not self.try_spend(d, t):
            return None
        rid, self._next_id = self._next_id, self._next_id + 1
        self._open[rid] = float(d)
        return rid

    def settle(self, rid, played_s: float, t: float) -> float:
        """Replace a reservation with the seconds actually played. Returns the adjustment (positive is a refund)."""
        if rid not in self._open:
            raise ValueError(f"unknown or already settled reservation {rid!r}")
        if played_s < 0:
            raise ValueError("played seconds must be non-negative")
        self._advance(t)
        adj = self._open.pop(rid) - float(played_s)
        self._bal = min(self.capacity_s, self._bal + adj) if adj >= 0 else self._bal + adj
        return adj

    def abandon_all(self, t: float) -> float:
        """Session reset or audio failure: nothing still open was heard. Returns the seconds given back."""
        return float(sum(self.settle(rid, 0.0, t) for rid in list(self._open)))

    def pause(self, t: float) -> None:
        self._advance(t)
        self._paused = True

    def resume(self, t: float) -> None:
        self._advance(t)                 # the paused stretch earns nothing
        self._paused = False

    def reset(self) -> None:
        self._bal, self._t = self.capacity_s, None
        self._open, self._paused = {}, False
