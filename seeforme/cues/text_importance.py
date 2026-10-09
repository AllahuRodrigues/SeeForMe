"""Text / sign importance signal T_i.

Two implementations share one interface:
  * KeywordTextImportance: hand-written rules (always available).
  * LearnedTextImportance: supervised character n-gram classifier trained in
    ml/text_classifier.py; falls back to the keyword rules when no model is found.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

CRITICAL_PATTERNS = [
    r"\bexit\b", r"\bstop\b", r"\bdanger\b", r"\bcaution\b", r"\bwarning\b",
    r"\bemergency\b", r"\bfire\b", r"\bwet\s*floor\b", r"\bdo\s*not\s*enter\b",
    r"\bkeep\s*out\b", r"\bstairs?\b", r"\belevator\b", r"\brestroom\b", r"\btoilet\b",
    r"\bmen\b", r"\bwomen\b", r"\bpush\b", r"\bpull\b", r"\bentrance\b", r"\bopen\b",
    r"\bclosed\b", r"\bout\s*of\s*order\b", r"\bno\s*entry\b", r"\bone\s*way\b",
    r"\bbus\s*stop\b", r"\bplatform\b", r"\bgate\s*\w+\b", r"\broom\s*\d+", r"\b\d{3,4}[a-z]?\b",
    r"\bfloor\s*\d", r"\blevel\s*\d", r"\bcross(ing)?\b", r"\bwalk\b", r"\bdont\s*walk\b",
]
DECORATIVE_PATTERNS = [
    r"\bsale\b", r"\b%\s*off\b", r"\bwelcome\b", r"\bmenu\b", r"\bcoffee\b", r"\bspecial\b",
    r"\bfollow\s*us\b", r"\bwww\.", r"\.com\b", r"\bevent\b", r"\bconcert\b", r"\bclub\b",
    r"\bhappy\b", r"\bbirthday\b", r"\bfree\b", r"\bnew\b", r"\bbuy\b", r"\bjoin\b",
]


def normalise_text(s: str) -> str:
    """Lower-case, strip accents (SAÍDA -> saida) and punctuation, collapse whitespace."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9%$.\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class KeywordTextImportance:
    name = "keyword"

    def score(self, text: Optional[str]) -> float:
        if not text:
            return 0.0
        s = normalise_text(text)
        if not s:
            return 0.0
        for p in CRITICAL_PATTERNS:
            if re.search(p, s):
                return 1.0
        for p in DECORATIVE_PATTERNS:
            if re.search(p, s):
                return 0.05
        # short all-caps strings tend to be signage; long strings tend to be prose/ads
        n_words = len(s.split())
        if n_words <= 2:
            return 0.35
        if n_words <= 5:
            return 0.2
        return 0.1


class LearnedTextImportance:
    name = "learned"

    def __init__(self, model_path: str = "models/text_importance.joblib"):
        self.model = None
        self.fallback = KeywordTextImportance()
        p = Path(model_path)
        if p.exists():
            try:
                import joblib
                self.model = joblib.load(p)
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, text: Optional[str]) -> float:
        if not text:
            return 0.0
        if self.model is None:
            return self.fallback.score(text)
        s = normalise_text(text)
        if not s:
            return 0.0
        return float(self.model.predict_proba([s])[0, 1])
