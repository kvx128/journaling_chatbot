from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from shared.extraction.mood import extract_mood_signal
from shared.repositories.journal_repo import JournalRepository
from shared.repositories.mood_repo import MoodRepository
from services.journal_bot.model_client import ModelServerClient


def handle_mood_checkin(db: Session, user_id: int, text: str) -> tuple[str, dict[str, Any] | None]:
    repo = MoodRepository(db)

    model_client = ModelServerClient()
    model_res = model_client.infer_mood(text)

    if model_res is not None:
        valence = model_res["valence"]
        arousal = model_res["arousal"]
        emotion_tags = model_res.get("emotion_tags")

        # Scale valence (-1.0 to 1.0) to self_report (1 to 5)
        raw_score = round(((valence + 1) / 2) * 4) + 1
        final_score = max(1, min(5, int(raw_score)))

        mood_entry = repo.create(
            user_id=user_id,
            self_report=final_score,
            note=text,
            valence=valence,
            arousal=arousal,
            emotion_tags=emotion_tags
        )

        if valence < -0.15:
            reply = "That sounds like a rough one. Noted."
        elif valence > 0.15:
            reply = "Glad to hear it. Noted."
        else:
            reply = "Noted, thanks for checking in."

        structured = {
            "mood_entry_id": mood_entry.id,
            "self_report": mood_entry.self_report,
            "valence": valence,
            "arousal": arousal,
            "emotion_tags": emotion_tags
        }
        return reply, structured

    # Fallback to existing logic
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

    model_client = ModelServerClient()
    model_res = model_client.infer_mood(text)

    if model_res is not None:
        valence = model_res["valence"]
        arousal = model_res["arousal"]
        emotion_tags = model_res.get("emotion_tags")

        entry = repo.create(
            user_id=user_id,
            body=text,
            valence=valence,
            arousal=arousal,
            emotion_tags=emotion_tags,
        )

        reply = "I've saved that to your journal."
        structured = {
            "journal_entry_id": entry.id,
            "valence": valence,
            "arousal": arousal,
            "emotion_tags": emotion_tags,
        }
        return reply, structured

    # Model unavailable/malformed output — still save the text, just without a
    # mood score. Leaves valence/arousal/emotion_tags null rather than guessing.
    entry = repo.create(user_id=user_id, body=text)

    reply = "I've saved that to your journal."
    structured = {
        "journal_entry_id": entry.id
    }
    return reply, structured
