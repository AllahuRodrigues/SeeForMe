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
               noise=None, on_video: Optional[Callable] = None, p_ask: float = 1.0,
               decision_hook: Optional[Callable] = None) -> List[Tuple[object, RunLog]]:
    """`decision_hook(scenario)` may return an extra on_decision callback for that video (diagnostics such as the
    failure funnel); it runs before the feedback of the same decision is delivered."""
    out: List[Tuple[object, RunLog]] = []
    learns = hasattr(policy, "learn")
    for k, sc in enumerate(scenarios):
        fb = FeedbackModel(sc, p_tap=p_tap, seed=seed * 100_003 + k, p_ask=p_ask)
        if hasattr(policy, "begin_video"):
            policy.begin_video(sc, fb)

        extra = decision_hook(sc) if decision_hook is not None else None

        def hook(t, tracks, obs, cue, ch, due, _fb=fb, _extra=extra):
            if _extra is not None:
                _extra(t, tracks, obs, cue, ch, due)
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
