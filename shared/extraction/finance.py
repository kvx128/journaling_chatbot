from __future__ import annotations

import re
from datetime import date
from shared.extraction.amount import extract_amount
from shared.extraction.category import extract_category
from shared.extraction.date_extract import extract_date
from shared.extraction.mood import extract_mood_signal
from shared.models.enums import Category, Direction, IntentEnum
from shared.models.schemas import ExtractionResult

SPEND_VERBS = [
    "spent", "spend", "spends", "spending", "paid", "bought", "cost",
    "charged", "debited", "sent", "gave", "used", "bill", "ordered", "purchased"
]

FINANCE_QUERY_PATTERNS = [
    r"\b(how much|total|summary|show|list|get|check)\b.*\b(spend|spent|spends|spending|expense|expenses|cost|transactions)\b",
    r"\b(spend|spent|spends|spending|expense|expenses|cost)\b.*\b(how much|total|summary)\b",
]


def extract_finance(text: str, reference_date: date | None = None) -> ExtractionResult:
    if not text or not text.strip():
        return ExtractionResult(
            intent=IntentEnum.UNKNOWN,
            confidence=0.0,
            raw_text=text or "",
            explanation="Empty input text",
        )

    text_lower = text.lower()

    is_query = any(re.search(pat, text_lower) for pat in FINANCE_QUERY_PATTERNS)
    if is_query:
        category, cat_kw = extract_category(text)
        occurred_on, date_explicit = extract_date(text, reference_date)
        return ExtractionResult(
            intent=IntentEnum.FINANCE_QUERY,
            category=category,
            occurred_on=occurred_on if date_explicit else None,
            confidence=0.85 if category or date_explicit else 0.70,
            raw_text=text,
            explanation=f"Detected financial query. Category: {category}",
        )

    amt_match = extract_amount(text)
    category, cat_kw = extract_category(text)
    occurred_on, date_explicit = extract_date(text, reference_date)
    has_mood, mood_kws, mood_score = extract_mood_signal(text)

    has_spend_verb = any(re.search(r"\b" + verb + r"\b", text_lower) for verb in SPEND_VERBS)
    amount_minor = amt_match.amount_minor if amt_match else None

    if amount_minor is not None and category is not None and has_spend_verb:
        confidence = 0.92
    elif amount_minor is not None and category is not None:
        confidence = 0.85
    elif amount_minor is not None and has_spend_verb:
        confidence = 0.80
    elif amount_minor is not None:
        confidence = amt_match.confidence if amt_match else 0.60
    elif category is not None and has_spend_verb:
        confidence = 0.40
    else:
        confidence = 0.10

    if amount_minor is not None:
        intent = IntentEnum.FINANCE_LOG
    elif category is not None and has_spend_verb:
        intent = IntentEnum.FINANCE_LOG
    elif has_mood:
        intent = IntentEnum.MOOD_CHECKIN
    else:
        intent = IntentEnum.UNKNOWN

    explanation_parts = []
    if amount_minor:
        explanation_parts.append(f"Amount: {amount_minor} minor units")
    if category:
        explanation_parts.append(f"Category: {category}")
    if date_explicit:
        explanation_parts.append(f"Date: {occurred_on}")
    if has_mood:
        explanation_parts.append(f"Mood signal detected: {mood_kws}")

    return ExtractionResult(
        intent=intent,
        amount_minor=amount_minor,
        category=category or (Category.OTHER if intent == IntentEnum.FINANCE_LOG else None),
        occurred_on=occurred_on,
        direction=Direction.DEBIT,
        mood_score=mood_score,
        confidence=round(confidence, 2),
        raw_text=text,
        explanation="; ".join(explanation_parts) if explanation_parts else "Low confidence extraction",
    )
