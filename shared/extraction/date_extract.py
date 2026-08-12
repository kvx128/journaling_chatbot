from __future__ import annotations

import re
from datetime import date, datetime, timedelta
import dateparser

RELATIVE_DATE_PATTERNS = [
    (re.compile(r"\b(today|tonight|this morning)\b", re.I), 0),
    (re.compile(r"\byesterday\b", re.I), -1),
    (re.compile(r"\bday before yesterday\b", re.I), -2),
]

LAST_NEXT_WEEKDAY_PATTERN = re.compile(
    r"\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b",
    re.I,
)

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}


def extract_date(text: str, reference_date: date | None = None) -> tuple[date, bool]:
    if reference_date is None:
        reference_date = date.today()

    if not text:
        return reference_date, False

    text_lower = text.lower()

    # 1. Quick relative phrase matches
    for pattern, offset in RELATIVE_DATE_PATTERNS:
        if pattern.search(text_lower):
            return reference_date + timedelta(days=offset), True

    # 2. Hand-coded specific logic for last/next weekday to override dateparser's "None" behavior
    match = LAST_NEXT_WEEKDAY_PATTERN.search(text_lower)
    if match:
        direction = match.group(1).lower()
        weekday_str = match.group(2).lower()
        target_weekday = WEEKDAYS[weekday_str]

        if direction == "last":
            days_since = (reference_date.weekday() - target_weekday) % 7
            if days_since == 0:
                days_since = 7
            return reference_date - timedelta(days=days_since), True
        else:
            days_until = (target_weekday - reference_date.weekday()) % 7
            if days_until == 0:
                days_until = 7
            return reference_date + timedelta(days=days_until), True

    # 3. Fallback to general dateparser processing
    parsed_dt = dateparser.parse(
        text,
        settings={
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": datetime.combine(reference_date, datetime.min.time()),
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )

    if parsed_dt is not None:
        parsed_d = parsed_dt.date()
        date_tokens = [
            "last", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june", "july", "august", "september",
            "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec", "ago"
        ]
        has_date_token = any(token in text_lower for token in date_tokens) or bool(
            re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
        )

        if has_date_token:
            return parsed_d, True

    return reference_date, False
