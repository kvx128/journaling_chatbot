from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.models.orm import User
from shared.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def get_by_handle(self, handle: str) -> User | None:
        stmt = select(User).where(User.handle == handle)
        return self.db.scalar(stmt)

    def get_or_create(self, handle: str = "me", tz: str = "Asia/Kolkata", currency: str = "INR") -> User:
        user = self.get_by_handle(handle)
        if user is None:
            user = User(handle=handle, tz=tz, currency=currency)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user
