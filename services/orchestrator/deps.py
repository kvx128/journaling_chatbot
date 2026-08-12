from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from shared.core.config import get_settings
from shared.core.db import get_db
from shared.models.orm import User
from shared.repositories.user_repo import UserRepository


def get_api_key(x_api_key: str | None = Header(None)) -> str:
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return x_api_key


def get_current_user(db: Session = Depends(get_db)) -> User:
    settings = get_settings()
    repo = UserRepository(db)
    user = repo.get_or_create(handle=settings.default_user_handle)
    return user


def guarded(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _: str = Depends(get_api_key),
) -> tuple[Session, User]:
    return db, user
