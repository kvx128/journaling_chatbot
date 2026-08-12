from __future__ import annotations

from sqlalchemy.orm import Session

from shared.models.orm import JournalEntry
from shared.repositories.base import BaseRepository


class JournalRepository(BaseRepository[JournalEntry]):
    def create(self, user_id: int, body: str) -> JournalEntry:
        entry = JournalEntry(user_id=user_id, body=body)
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry
