"""Simulated one-sided user feedback over a scenario with weak events (EgoBlind or CampusBench-like).

Label sources, in the order a real user would produce them:

    tap_useful        the assistant spoke about an event inside its window, for the first time      +1
    tap_not_useful    it spoke near a labelled moment (within `near_s` of a question) and the cue    -1
                      was not a timely first mention of an event
    missed_question   the user asked at t_q about an event the assistant never announced in time:    +1
                      candidates of that event seen in [earliest_useful_s, t_q] should have been spoken
    quiet_window      the user asked and the answer was "nothing there": every candidate seen in the  -1
                      quiet window should have stayed silent

Cues far from every labelled moment and silence get no label. Taps are dropped with probability 1 - p_tap
(users do not always tap). By default every question in the data is asked (p_ask = 1). With p_ask < 1 each question
(speak or quiet) is asked with probability p_ask, so a miss can go unreported: the feedback is then truly one-sided
on the scored region too. Taps and questions use separate seeded generators, so p_ask leaves the tap stream unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Set

NEAR_S = 6.0


@dataclass(frozen=True)
class Feedback:
    t: float                      # when the feedback arrives
    label: int                    # +1 should have spoken, -1 should have stayed silent
    kind: str                     # tap_useful | tap_not_useful | missed_question | quiet_window
    source_id: Optional[int]      # event the label is about; None = every candidate in [t0, t1]
    t0: float                     # time span of the decisions the label applies to
    t1: float


class FeedbackModel:
    def __init__(self, scenario, p_tap: float = 1.0, seed: int = 0, near_s: float = NEAR_S, p_ask: float = 1.0):
        if not 0.0 <= float(p_ask) <= 1.0:
            raise ValueError(f"p_ask must be in [0, 1], got {p_ask!r}")
        self.p_tap = float(p_tap)
        self.p_ask = float(p_ask)
        self._ask_rng = random.Random(seed * 7919 + 17)
        self.near_s = float(near_s)
        self._rng = random.Random(seed)
        self._events = {e.obj_id: e for e in scenario.speak_events()}
        self._quiet = list(scenario.quiet_events())
        self._labelled = sorted({e.detail["t_q"] for e in scenario.events() if "t_q" in e.detail})
        # questions in time order: (t_q, kind, event)
        qs = [(e.detail["t_q"], "speak", e) for e in self._events.values() if "t_q" in e.detail]
        qs += [(e.detail["t_q"], "quiet", e) for e in self._quiet if "t_q" in e.detail]
        self._questions = sorted(qs, key=lambda x: (x[0], x[2].obj_id))
        self._next_q = 0
        self._announced: Set[int] = set()

    # ------------------------------------------------------------------ speaking
    def on_cue(self, t: float, source_id: Optional[int]) -> Optional[Feedback]:
        """Feedback for a cue spoken at t about `source_id` (None when it matches no event)."""
        e = self._events.get(source_id) if source_id is not None else None
        if e is not None and e.in_window(t) and source_id not in self._announced:
            self._announced.add(source_id)
            return self._tap(Feedback(t, 1, "tap_useful", source_id, t, t))
        if e is not None and e.in_window(t):
            return self._tap(Feedback(t, -1, "tap_not_useful", source_id, t, t))   # repeat of an announced event
        if any(abs(t - q) <= self.near_s for q in self._labelled):
            return self._tap(Feedback(t, -1, "tap_not_useful", source_id, t, t))
        return None

    def _tap(self, f: Feedback) -> Optional[Feedback]:
        if self.p_tap >= 1.0:
            return f
        return f if self._rng.random() < self.p_tap else None

    # ------------------------------------------------------------------ questions
    def advance(self, t: float) -> List[Feedback]:
        """Feedback from every question asked up to and including time t (each question fires once)."""
        out: List[Feedback] = []
        while self._next_q < len(self._questions) and self._questions[self._next_q][0] <= t + 1e-9:
            t_q, kind, e = self._questions[self._next_q]
            self._next_q += 1
            if self.p_ask < 1.0 and self._ask_rng.random() >= self.p_ask:
                continue                                  # the user did not ask this time
            if kind == "speak":
                if e.obj_id in self._announced or e.earliest_useful_s is None:
                    continue
                out.append(Feedback(t_q, 1, "missed_question", e.obj_id, float(e.earliest_useful_s), float(t_q)))
            else:
                out.append(Feedback(t_q, -1, "quiet_window", None, float(e.earliest_useful_s), float(e.latest_safe_s)))
        return out

    @property
    def announced(self) -> Set[int]:
        return set(self._announced)
