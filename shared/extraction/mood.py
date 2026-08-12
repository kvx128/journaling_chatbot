from __future__ import annotations

import re

MOOD_KEYWORDS = {
    "happy", "great", "good", "awesome", "excited", "relaxed", "joyful", "energetic",
    "calm", "peaceful", "grateful", "productive", "fantastic", "amazing", "blessed",
    "feeling", "felt", "tired", "drained", "exhausted", "sad", "anxious", "stressed",
    "depressed", "upset", "angry", "frustrated", "overwhelmed", "low", "hopeless",
    "lonely", "sick", "unmotivated", "burnt out", "burnout", "miserable"
}

MOOD_SCORE_PATTERN = re.compile(
    r"(?i)\b(?:mood|feeling|energy|rated?|score)?\s*:?\s*([1-5])\s*(?:/|out of)\s*5\b|\b(?:mood|score)\s*:?\s*([1-5])\b"
)


def extract_mood_signal(text: str) -> tuple[bool, list[str], int | None]:
    if not text:
        return False, [], None

    text_lower = text.lower()
    matched_kws: list[str] = []

    for kw in MOOD_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            matched_kws.append(kw)

    score: int | None = None
    m = MOOD_SCORE_PATTERN.search(text)
    if m:
        val_str = m.group(1) or m.group(2)
        if val_str:
            score = int(val_str)

    has_signal = len(matched_kws) > 0 or score is not None
    return has_signal, matched_kws, score
