from __future__ import annotations

from datetime import date, datetime
from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.models.orm import MoodEntry
from shared.repositories.base import BaseRepository


class MoodRepository(BaseRepository[MoodEntry]):
    def create(
        self,
        user_id: int,
        self_report: int,
        sleep_hours: float | None = None,
        energy: int | None = None,
        social_contact: bool | None = None,
        note: str | None = None,
        recorded_at: datetime | None = None,
    ) -> MoodEntry:
        entry = MoodEntry(
            user_id=user_id,
            self_report=self_report,
            sleep_hours=sleep_hours,
            energy=energy,
            social_contact=social_contact,
            note=note,
        )
        if recorded_at is not None:
            entry.recorded_at = recorded_at

        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def list_for_user(
        self,
        user_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
    ) -> list[MoodEntry]:
        stmt = select(MoodEntry).where(MoodEntry.user_id == user_id)

        if start_date is not None:
            start_dt = datetime.combine(start_date, datetime.min.time())
            stmt = stmt.where(MoodEntry.recorded_at >= start_dt)
        if end_date is not None:
            end_dt = datetime.combine(end_date, datetime.max.time())
            stmt = stmt.where(MoodEntry.recorded_at <= end_dt)

        stmt = stmt.order_by(MoodEntry.recorded_at.desc(), MoodEntry.id.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())
