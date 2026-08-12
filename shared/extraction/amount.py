from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AmountMatch:
    amount_minor: int
    raw_match: str
    confidence: float
    start: int
    end: int


NON_AMOUNT_SUFFIXES = {
    "th", "st", "nd", "rd", "am", "pm", "hour", "hours", "hr", "hrs",
    "min", "mins", "minute", "minutes", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "kg", "g", "gm", "km", "m",
    "percent", "%", "star", "stars", "rating", "ratings"
}


def extract_amount(text: str) -> AmountMatch | None:
    if not text:
        return None

    matches: list[AmountMatch] = []

    pattern_explicit = re.compile(
        r"(?i)(?:"
        r"(?:(?:₹|rs\.?|inr)\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?\s*(k|kilo)?)"
        r"|"
        r"(?:(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?\s*(k|kilo)?\s*(?:rs\.?|rupees?|inr))"
        r"|"
        r"(?:(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?\s*(k|kilo))"
        r")"
    )

    for m in pattern_explicit.finditer(text):
        groups = m.groups()
        if groups[0] is not None:
            num_str, dec_str, k_sfx = groups[0], groups[1], groups[2]
            conf = 0.95
        elif groups[3] is not None:
            num_str, dec_str, k_sfx = groups[3], groups[4], groups[5]
            conf = 0.95
        elif groups[6] is not None:
            num_str, dec_str, k_sfx = groups[6], groups[7], groups[8]
            conf = 0.90
        else:
            continue

        amount_minor = _to_minor_units(num_str, dec_str, k_sfx)
        if amount_minor > 0:
            matches.append(
                AmountMatch(
                    amount_minor=amount_minor,
                    raw_match=m.group(0),
                    confidence=conf,
                    start=m.start(),
                    end=m.end(),
                )
            )

    if matches:
        matches.sort(key=lambda x: (x.confidence, x.amount_minor), reverse=True)
        return matches[0]

    pattern_plain = re.compile(r"(?i)\b(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?\b")

    for m in pattern_plain.finditer(text):
        start, end = m.start(), m.end()
        num_str = m.group(1)
        dec_str = m.group(2)

        trailing_text = text[end:].strip().split()
        if trailing_text and trailing_text[0].lower() in NON_AMOUNT_SUFFIXES:
            continue

        leading_text = text[:start].strip().split()
        if leading_text and leading_text[-1].lower() in {"rated", "rating", "stage", "phase", "version", "top"}:
            continue

        amount_minor = _to_minor_units(num_str, dec_str, None)
        if 1990 <= amount_minor // 100 <= 2035 and dec_str is None:
            conf = 0.35
        else:
            conf = 0.65

        if amount_minor > 0:
            matches.append(
                AmountMatch(
                    amount_minor=amount_minor,
                    raw_match=m.group(0),
                    confidence=conf,
                    start=start,
                    end=end,
                )
            )

    if matches:
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[0]

    return None


def _to_minor_units(num_str: str, dec_str: str | None, k_sfx: str | None) -> int:
    num_clean = num_str.replace(",", "")
    val = float(num_clean)
    if dec_str:
        val += float(f"0.{dec_str}")
    if k_sfx:
        val *= 1000.0
    return int(round(val * 100))
