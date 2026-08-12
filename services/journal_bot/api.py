from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.journal_bot.skill import handle_journal_free, handle_mood_checkin
from shared.core.db import get_db
from shared.guardrails.crisis import CRISIS_RESPONSE, scan_for_crisis
from shared.models.schemas import MoodCheckinCreate, MoodCheckinRead
from shared.repositories.mood_repo import MoodRepository

router = APIRouter()


class HandleRequest(BaseModel):
    user_id: int
    text: str
    task: str


class HandleResponse(BaseModel):
    reply: str
    structured_data: dict[str, Any] | None = None


@router.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/handle", response_model=HandleResponse)
def handle_endpoint(req: HandleRequest, db: Session = Depends(get_db)) -> HandleResponse:
    if req.task == "MOOD_CHECKIN":
        reply, structured = handle_mood_checkin(db, req.user_id, req.text)
    elif req.task == "JOURNAL_FREE":
        reply, structured = handle_journal_free(db, req.user_id, req.text)
    else:
        reply, structured = "I don't know how to handle that task.", None

    return HandleResponse(reply=reply, structured_data=structured)


@router.post("/mood/checkin", response_model=MoodCheckinRead)
def create_mood_checkin_endpoint(payload: dict[str, Any], db: Session = Depends(get_db)) -> MoodCheckinRead:
    user_id = payload.pop("user_id")
    create_schema = MoodCheckinCreate(**payload)

    if create_schema.note:
        crisis_res = scan_for_crisis(create_schema.note)
        if crisis_res.matched:
            raise HTTPException(status_code=400, detail=CRISIS_RESPONSE)

    repo = MoodRepository(db)
    entry = repo.create(
        user_id=user_id,
        self_report=create_schema.self_report,
        sleep_hours=create_schema.sleep_hours,
        energy=create_schema.energy,
        social_contact=create_schema.social_contact,
        note=create_schema.note,
    )
    return MoodCheckinRead.model_validate(entry)
