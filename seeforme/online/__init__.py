"""Online learning of when to speak from one-sided feedback (M1).

The assistant only learns about a moment when it speaks (the user may tap useful / not useful) or when the user later
has to ask (the question reveals a missed cue, or confirms that nothing was there). Silence is never labelled directly:
the apple-tasting / partial-monitoring setting.
"""
from .feedback import Feedback, FeedbackModel

__all__ = ["Feedback", "FeedbackModel"]
