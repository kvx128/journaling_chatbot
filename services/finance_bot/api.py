from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from services.finance_bot.skill import answer_finance_query, get_summary, log_transaction_from_text
from shared.core.db import get_db
from shared.extraction.finance import extract_finance
from shared.models.enums import Category, DateRangeEnum
from shared.models.schemas import ExtractionResult, FinanceSummaryResponse, TransactionCreate, TransactionRead
from shared.repositories.transaction_repo import TransactionRepository

router = APIRouter()


class HandleRequest(BaseModel):
    user_id: int
    text: str
    task: str


class HandleResponse(BaseModel):
    reply: str
    structured_data: dict[str, Any] | None = None


class ExtractRequest(BaseModel):
    text: str


@router.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/extract", response_model=ExtractionResult)
def extract_endpoint(req: ExtractRequest) -> ExtractionResult:
    return extract_finance(req.text)


@router.post("/handle", response_model=HandleResponse)
def handle_endpoint(req: HandleRequest, db: Session = Depends(get_db)) -> HandleResponse:
    if req.task == "FINANCE_LOG":
        reply, structured = log_transaction_from_text(db, req.user_id, req.text)
    elif req.task == "FINANCE_QUERY":
        reply, structured = answer_finance_query(db, req.user_id, req.text)
    else:
        reply, structured = "I don't know how to handle that task.", None

    return HandleResponse(reply=reply, structured_data=structured)


@router.post("/transactions", response_model=TransactionRead)
def create_transaction_endpoint(payload: dict[str, Any], db: Session = Depends(get_db)) -> TransactionRead:
    user_id = payload.pop("user_id")
    create_schema = TransactionCreate(**payload)

    repo = TransactionRepository(db)
    txn = repo.create(
        user_id=user_id,
        amount_minor=create_schema.amount_minor,
        category=create_schema.category,
        direction=create_schema.direction,
        merchant=create_schema.merchant,
        occurred_on=create_schema.occurred_on,
        payment_method=create_schema.payment_method,
        source=create_schema.source,
        confirmed=create_schema.confirmed,
    )
    return TransactionRead.model_validate(txn)


@router.get("/summary", response_model=FinanceSummaryResponse)
def get_summary_endpoint(
    user_id: int,
    category: str | None = None,
    date_range: DateRangeEnum = DateRangeEnum.THIS_MONTH,
    db: Session = Depends(get_db)
) -> FinanceSummaryResponse:
    cat_enum = Category(category) if category else None
    return get_summary(db, user_id, date_range, cat_enum)
