from __future__ import annotations

from typing import Generic, TypeVar
from sqlalchemy.orm import Session

from shared.models.base import Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    def __init__(self, db: Session) -> None:
        self.db = db
