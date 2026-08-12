from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from shared.core.db import get_db
from shared.models.enums import Category
from shared.models.schemas import CategoriesResponse

router = APIRouter(tags=["open"])


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def readiness_check(db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "db": "reachable"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unreachable",
        )


@router.get("/taxonomy/categories", response_model=CategoriesResponse)
def list_categories() -> CategoriesResponse:
    return CategoriesResponse(categories=[c.value for c in Category])
