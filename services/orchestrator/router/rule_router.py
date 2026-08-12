from __future__ import annotations

import re
from dataclasses import dataclass, field

from shared.extraction.amount import extract_amount
from shared.extraction.finance import extract_finance
from shared.extraction.mood import extract_mood_signal
from shared.models.enums import IntentEnum


@dataclass
class RouteDecision:
    intent: IntentEnum
    confidence: float
    matched_signal: str
    # Additional intents this message also carries. A journaling message very often
    # holds both a spend and a feeling ("dropped 2k on dinner, felt awful after"),
    # and the cross-domain correlation engine needs both halves recorded.
    secondary_intents: list[IntentEnum] = field(default_factory=list)


SMALLTALK_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|bye|good morning|good night|sup)\b",
    re.IGNORECASE
)


def route(text: str) -> RouteDecision:
    if not text or not text.strip():
        return RouteDecision(IntentEnum.UNKNOWN, 0.0, "Empty input")

    has_mood, mood_kws, score = extract_mood_signal(text)

    finance_res = extract_finance(text)
    if finance_res.confidence >= 0.50 and finance_res.intent in (IntentEnum.FINANCE_LOG, IntentEnum.FINANCE_QUERY):
        secondary = []
        signal = finance_res.explanation or "Finance intent detected"
        # Only FINANCE_LOG carries a real event worth pairing with a mood reading;
        # a FINANCE_QUERY is just a lookup.
        if has_mood and finance_res.intent is IntentEnum.FINANCE_LOG:
            secondary.append(IntentEnum.MOOD_CHECKIN)
            signal += f" | also mood: {', '.join(mood_kws)}"
        return RouteDecision(
            finance_res.intent,
            finance_res.confidence,
            signal,
            secondary_intents=secondary,
        )

    if has_mood:
        secondary = []
        signal = f"Mood signals detected: {', '.join(mood_kws)}" + (f" (Score: {score})" if score else "")
        # Mood-led message that still names an amount ("felt awful, blew 2000 on
        # takeout") — queue the finance side too rather than losing the spend.
        amt = extract_amount(text)
        if amt is not None:
            secondary.append(IntentEnum.FINANCE_LOG)
            signal += f" | also amount: {amt.amount_minor}"
        return RouteDecision(
            IntentEnum.MOOD_CHECKIN,
            0.8,
            signal,
            secondary_intents=secondary,
        )

    if len(text) < 30 and SMALLTALK_PATTERNS.search(text.strip()):
        return RouteDecision(IntentEnum.SMALLTALK, 0.9, "Matches smalltalk patterns")

    if len(text.split()) > 4:
        # Free journaling with a spend buried in it still deserves a transaction row.
        secondary = []
        signal = "Reasonably long text with no other signals"
        amt = extract_amount(text)
        if amt is not None:
            secondary.append(IntentEnum.FINANCE_LOG)
            signal = f"Journal text carrying an amount: {amt.amount_minor}"
        return RouteDecision(
            IntentEnum.JOURNAL_FREE,
            0.6,
            signal,
            secondary_intents=secondary,
        )

    return RouteDecision(IntentEnum.UNKNOWN, 0.0, "Too short or ambiguous")
