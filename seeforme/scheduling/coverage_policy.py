"""Coverage-first diagnostic policy: announce every confirmed candidate once, as soon as the channel and the ledger allow.

This is a measuring instrument, not a product policy. It removes the learned usefulness score and its silence
threshold, so the replay shows how many events the current candidates, the audio channel and the speech budget can
deliver at all, and what that costs in unnecessary speech. It must never be a default in the live app.

A candidate is a track above the detector-confidence floor. It is confirmed after `confirm_n` decisions, queued once,
and announced first-in first-out when the channel is free and the ledger can pay for the cue. A queued candidate
that has not been observed for `stale_s` is dropped. Every candidate gets a reason at every decision.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from ..cues.pcacs import ChannelState
from ..online.policy import OnlinePolicy
from .speech_ledger import SpeechLedger


class CoverageFirstPolicy(OnlinePolicy):
    def __init__(self, capacity_s: float = 6.0, refill_per_s: Optional[float] = 4.0 / 60.0, confirm_n: int = 2,
                 stale_s: float = 2.0, name: Optional[str] = None, **kw):
        super().__init__(mode="greedy", budget=False, lr=0.0, name=name or "coverage_first", **kw)
        self.ledger = None if refill_per_s is None else SpeechLedger(capacity_s, refill_per_s)   # None: no budget at all
        self.confirm_n, self.stale_s = int(confirm_n), float(stale_s)
        self.stats: Counter = Counter()
        self._clock_offset, self._video_t = 0.0, 0.0     # one ledger over the whole stream, as in CarePolicy
        self._new_video()

    def _new_video(self) -> None:
        self._seen: Dict[int, int] = {}
        self._last_seen: Dict[int, float] = {}
        self._queue: List[int] = []
        self._announced: set = set()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_clock_offset"):
            self._clock_offset += self._video_t + self.decision_interval
            self._video_t = 0.0
            self._new_video()

    def learn(self, feedback) -> None:      # nothing is learned
        return None

    def decide(self, tracks, t: float, ch: Optional[ChannelState] = None, scene_ctx=None) -> List:
        c = self.cfg
        ch = ch or ChannelState()
        self._video_t = max(self._video_t, t)
        self.last_action = "silence"
        allobs = self._features(tracks, t, ch)
        self.last_reasons = {o.track.track_id: "low_confidence" for o in allobs if o.track.confidence < c.min_confidence}
        obs = [o for o in allobs if o.track.confidence >= c.min_confidence]
        self._last_obs = obs
        live = {o.track.track_id: o for o in obs}
        for tid, o in live.items():
            self._seen[tid] = self._seen.get(tid, 0) + 1
            self._last_seen[tid] = t
            o.score = 1.0
            if tid in self._announced:
                self.last_reasons[tid] = "already_announced"
            elif tid in self._queue:
                self.last_reasons[tid] = "channel_busy" if ch.busy else "queued"
            elif self._seen[tid] >= self.confirm_n:
                self._queue.append(tid); self.stats["confirmed"] += 1
                self.last_reasons[tid] = "channel_busy" if ch.busy else "queued"
            else:
                self.last_reasons[tid] = "unconfirmed"
        keep = []
        for tid in self._queue:                                   # forget what is no longer in view
            if tid not in live and t - self._last_seen.get(tid, t) > self.stale_s:
                self.stats["stale"] += 1
            else:
                keep.append(tid)
        self._queue = keep
        if ch.busy:
            return []
        for tid in list(self._queue):
            o = live.get(tid)
            if o is None:
                continue
            if self.ledger is not None and not self.ledger.try_spend(float(o.duration), self._clock_offset + t):
                self.last_reasons[tid] = "budget"; self.stats["budget_blocked"] += 1
                continue                                          # a shorter cue further back may still fit
            self._queue.remove(tid); self._announced.add(tid); self.stats["announced"] += 1
            self.last_action = "speak"
            self.last_reasons[tid] = "spoken"
            return [self._make_cue(o, t, "speak", ch)]
        return []

    def to_dict(self):
        return {"confirm_n": self.confirm_n, "stale_s": self.stale_s, "stats": dict(self.stats),
                "ledger": None if self.ledger is None else {"capacity_s": self.ledger.capacity_s, "refill_per_s": self.ledger.refill_per_s}}
