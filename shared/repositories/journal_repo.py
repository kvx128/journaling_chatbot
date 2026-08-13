from __future__ import annotations

import json
from sqlalchemy.orm import Session

from shared.models.orm import JournalEntry
from shared.repositories.base import BaseRepository


class JournalRepository(BaseRepository[JournalEntry]):
    def create(
        self,
        user_id: int,
        body: str,
        valence: float | None = None,
        arousal: float | None = None,
        emotion_tags: list[str] | None = None,
    ) -> JournalEntry:
        entry = JournalEntry(
            user_id=user_id,
            body=body,
            valence=valence,
            arousal=arousal,
            emotion_tags=json.dumps(emotion_tags) if emotion_tags is not None else None,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry
