"""Replay a time-ordered stream of videos through one policy, delivering one-sided feedback as it happens.

Progressive validation: every decision is made before the feedback it causes is seen, so the logs of a learning
policy are an honest online evaluation. Frozen policies (P-CACS, the October GRU) run through the same loop and
simply ignore the feedback.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from ..campus.runner import RunLog
from ..config import PipelineConfig
from ..egoblind.runner import run_on_egoblind
from .feedback import FeedbackModel


def run_stream(policy, scenarios: Sequence, cfg: Optional[PipelineConfig] = None, p_tap: float = 1.0, seed: int = 0,
               noise=None, on_video: Optional[Callable] = None) -> List[Tuple[object, RunLog]]:
    out: List[Tuple[object, RunLog]] = []
    learns = hasattr(policy, "learn")
    for k, sc in enumerate(scenarios):
        fb = FeedbackModel(sc, p_tap=p_tap, seed=seed * 100_003 + k)
        if hasattr(policy, "begin_video"):
            policy.begin_video(sc, fb)

        def hook(t, tracks, obs, cue, ch, due, _fb=fb):
            got = []
            if cue is not None and cue.action != "haptic":
                f = _fb.on_cue(t, cue.source_id)
                if f is not None:
                    got.append(f)
            got.extend(_fb.advance(t))
            if learns and got:
                policy.learn(got)

        log = run_on_egoblind(policy, sc, cfg, seed=seed, on_decision=hook, noise=noise)
        tail = fb.advance(float("inf"))           # questions after the last processed frame
        if learns and tail:
            policy.learn(tail)
        out.append((sc, log))
        if on_video is not None:
            on_video(k, sc, log)
    return out
