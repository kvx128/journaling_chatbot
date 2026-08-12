from __future__ import annotations

import re
from dataclasses import dataclass

from shared.extraction.finance import extract_finance
from shared.extraction.mood import extract_mood_signal
from shared.models.enums import IntentEnum


@dataclass
class RouteDecision:
    intent: IntentEnum
    confidence: float
    matched_signal: str


SMALLTALK_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|good morning|good night|sup)\b",
    re.IGNORECASE
)


def route(text: str) -> RouteDecision:
    if not text or not text.strip():
        return RouteDecision(IntentEnum.UNKNOWN, 0.0, "Empty input")

    finance_res = extract_finance(text)
    if finance_res.confidence >= 0.50 and finance_res.intent in (IntentEnum.FINANCE_LOG, IntentEnum.FINANCE_QUERY):
        return RouteDecision(
            finance_res.intent,
            finance_res.confidence,
            finance_res.explanation or "Finance intent detected",
        )

    has_mood, mood_kws, score = extract_mood_signal(text)
    if has_mood:
        return RouteDecision(
            IntentEnum.MOOD_CHECKIN,
            0.8,
            f"Mood signals detected: {', '.join(mood_kws)}" + (f" (Score: {score})" if score else ""),
        )

    if len(text) < 30 and SMALLTALK_PATTERNS.search(text.strip()):
        return RouteDecision(IntentEnum.SMALLTALK, 0.9, "Matches smalltalk patterns")

    if len(text.split()) > 4:
        return RouteDecision(IntentEnum.JOURNAL_FREE, 0.6, "Reasonably long text with no other signals")

    return RouteDecision(IntentEnum.UNKNOWN, 0.0, "Too short or ambiguous")
