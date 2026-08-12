from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.orchestrator.deps import guarded
from shared.guardrails.crisis import CRISIS_RESPONSE, scan_for_crisis
from services.orchestrator.router.rule_router import route
from services.orchestrator.clients.finance_client import FinanceClient
from services.orchestrator.clients.journal_client import JournalClient
from shared.models.enums import DateRangeEnum, IntentEnum
from shared.models.orm import User
from shared.models.schemas import (
    ChatRequest,
    ChatResponse,
    FinanceSummaryResponse,
    MoodCheckinCreate,
    MoodCheckinRead,
    TransactionCreate,
    TransactionRead,
)

router = APIRouter(dependencies=[Depends(guarded)])


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, deps: tuple[Session, User] = Depends(guarded)) -> ChatResponse:
    db, user = deps

    crisis_res = scan_for_crisis(request.message)
    if crisis_res.matched:
        return ChatResponse(
            reply=CRISIS_RESPONSE,
            intent=IntentEnum.UNKNOWN,
            crisis_flagged=True,
        )

    decision = route(request.message)
    structured_data = None
    reply = "I'm not sure how to handle that."

    try:
        if decision.intent in (IntentEnum.FINANCE_LOG, IntentEnum.FINANCE_QUERY):
            fc = FinanceClient()
            task = "FINANCE_LOG" if decision.intent == IntentEnum.FINANCE_LOG else "FINANCE_QUERY"
            res = fc.handle(user.id, request.message, task)
            reply = res.get("reply", "")
            structured_data = res.get("structured_data")

        elif decision.intent in (IntentEnum.MOOD_CHECKIN, IntentEnum.JOURNAL_FREE):
            jc = JournalClient()
            task = "MOOD_CHECKIN" if decision.intent == IntentEnum.MOOD_CHECKIN else "JOURNAL_FREE"
            res = jc.handle(user.id, request.message, task)
            reply = res.get("reply", "")
            structured_data = res.get("structured_data")

        elif decision.intent == IntentEnum.SMALLTALK:
            reply = "Hello! I'm here to help you log your expenses and mood. What's on your mind?"

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            raise HTTPException(status_code=400, detail=e.response.json().get("detail", "Bad Request"))
        raise HTTPException(status_code=502, detail="Bot service error")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Bot service unavailable")

    return ChatResponse(
        reply=reply,
        intent=decision.intent,
        structured_data=structured_data,
        crisis_flagged=False,
    )


@router.post("/finance/transactions", response_model=TransactionRead)
def create_transaction_endpoint(payload: TransactionCreate, deps: tuple[Session, User] = Depends(guarded)) -> TransactionRead:
    fc = FinanceClient()
    try:
        data = payload.model_dump(mode="json")
        data["user_id"] = deps[1].id
        res = fc.create_transaction(data)
        return TransactionRead(**res)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@router.get("/finance/summary", response_model=FinanceSummaryResponse)
def get_finance_summary_endpoint(
    category: str | None = None,
    date_range: DateRangeEnum = DateRangeEnum.THIS_MONTH,
    deps: tuple[Session, User] = Depends(guarded)
) -> FinanceSummaryResponse:
    fc = FinanceClient()
    try:
        params = {"user_id": deps[1].id, "date_range": date_range.value}
        if category:
            params["category"] = category
        res = fc.get_summary(params)
        return FinanceSummaryResponse(**res)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@router.post("/mood/checkin", response_model=MoodCheckinRead)
def create_mood_checkin_endpoint(payload: MoodCheckinCreate, deps: tuple[Session, User] = Depends(guarded)) -> MoodCheckinRead:
    jc = JournalClient()
    try:
        data = payload.model_dump(mode="json")
        data["user_id"] = deps[1].id
        res = jc.create_mood_checkin(data)
        return MoodCheckinRead(**res)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            raise HTTPException(status_code=400, detail=e.response.json().get("detail", "Bad Request"))
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
