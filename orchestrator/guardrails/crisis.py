from __future__ import annotations

import re
from dataclasses import dataclass


CRISIS_RESPONSE = (
    "It sounds like you're going through a very difficult time. You don't have to face this alone. "
    "Please reach out to the KIRAN helpline at 1800-599-0019 or Tele-MANAS at 14416. "
    "There are people who want to support you."
)

IDIOM_PATTERNS = [
    re.compile(r"\bkilling me\b", re.IGNORECASE),
    re.compile(r"\bdying to\b", re.IGNORECASE),
    re.compile(r"\bkill for\b", re.IGNORECASE),
    re.compile(r"\bkill time\b", re.IGNORECASE),
    re.compile(r"\bdead tired\b", re.IGNORECASE),
]

CRISIS_PATTERNS = [
    re.compile(r"\bkill myself\b", re.IGNORECASE),
    re.compile(r"\bend it all\b", re.IGNORECASE),
    re.compile(r"\bnot worth living\b", re.IGNORECASE),
    re.compile(r"\bwant to die\b", re.IGNORECASE),
    re.compile(r"\bbetter off dead\b", re.IGNORECASE),
    re.compile(r"\bcan'?t go on\b", re.IGNORECASE),
    re.compile(r"\bno reason to live\b", re.IGNORECASE),
    re.compile(r"\bhurt myself\b", re.IGNORECASE),
    re.compile(r"\bend my life\b", re.IGNORECASE),
    re.compile(r"\bsuicide\b", re.IGNORECASE),
]


@dataclass
class CrisisResult:
    matched: bool
    matched_phrase: str | None


def scan_for_crisis(text: str) -> CrisisResult:
    if not text:
        return CrisisResult(False, None)

    sanitized_text = text
    for idiom in IDIOM_PATTERNS:
        sanitized_text = idiom.sub("", sanitized_text)

    for pattern in CRISIS_PATTERNS:
        match = pattern.search(sanitized_text)
        if match:
            return CrisisResult(True, match.group(0))

    return CrisisResult(False, None)
