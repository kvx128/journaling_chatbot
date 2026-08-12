from __future__ import annotations

from sqlalchemy.orm import Session

from shared.extraction.mood import extract_mood_signal
from shared.models.orm import MoodEntry, User
from shared.models.schemas import MoodCheckinCreate
from shared.repositories.mood_repo import MoodRepository


def log_mood(db: Session, user: User, checkin: MoodCheckinCreate) -> str:
    repo = MoodRepository(db)
    repo.create(
        user_id=user.id,
        self_report=checkin.self_report,
        sleep_hours=checkin.sleep_hours,
        energy=checkin.energy,
        social_contact=checkin.social_contact,
        note=checkin.note,
    )

    if checkin.self_report <= 2:
        return "I've noted that down. Take it easy today."
    elif checkin.self_report == 3:
        return "Logged your mood."
    else:
        return "Glad you're having a good day! Logged."


def log_mood_from_chat_text(db: Session, user: User, text: str) -> tuple[str, MoodEntry]:
    _, _, score = extract_mood_signal(text)

    final_score = score if score is not None else 3

    repo = MoodRepository(db)
    entry = repo.create(
        user_id=user.id,
        self_report=final_score,
        note=text,
    )

    if score is not None:
        if final_score <= 2:
            reply = "I've logged your mood. Take it easy today."
        elif final_score == 3:
            reply = "Logged your mood."
        else:
            reply = "Glad you're having a good day! Logged."
    else:
        reply = "I've saved this as a mood check-in with a neutral score (3). Next time, try including a number like '4/5' to rate your mood!"

    return reply, entry
