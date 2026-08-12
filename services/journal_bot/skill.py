from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from shared.extraction.mood import extract_mood_signal
from shared.repositories.journal_repo import JournalRepository
from shared.repositories.mood_repo import MoodRepository


def handle_mood_checkin(db: Session, user_id: int, text: str) -> tuple[str, dict[str, Any] | None]:
    repo = MoodRepository(db)
    has_mood, kws, score = extract_mood_signal(text)

    final_score = score if score is not None else 3

    mood_entry = repo.create(
        user_id=user_id,
        self_report=final_score,
        note=text
    )

    reply = "I've noted down your mood. Hope you have a good day!"
    structured = {
        "mood_entry_id": mood_entry.id,
        "self_report": mood_entry.self_report,
    }
    return reply, structured


def handle_journal_free(db: Session, user_id: int, text: str) -> tuple[str, dict[str, Any] | None]:
    repo = JournalRepository(db)
    entry = repo.create(user_id=user_id, body=text)

    reply = "I've saved that to your journal."
    structured = {
        "journal_entry_id": entry.id
    }
    return reply, structured
