from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from orchestrator.deps import guarded
from orchestrator.guardrails.crisis import CRISIS_RESPONSE, scan_for_crisis
from orchestrator.router.rule_router import route
from orchestrator.skills import finance_skill, mood_skill
from shared.models.enums import Category, DateRangeEnum, IntentEnum
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
from shared.repositories.journal_repo import JournalRepository
from shared.repositories.mood_repo import MoodRepository

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

    if decision.intent == IntentEnum.FINANCE_LOG:
        reply, txn = finance_skill.log_transaction_from_text(db, user, request.message)
        if txn:
            structured_data = {
                "transaction_id": txn.id,
                "amount_minor": txn.amount_minor,
                "category": txn.category.value,
            }

    elif decision.intent == IntentEnum.FINANCE_QUERY:
        reply = finance_skill.answer_finance_query(db, user, request.message)

    elif decision.intent == IntentEnum.MOOD_CHECKIN:
        reply, entry = mood_skill.log_mood_from_chat_text(db, user, request.message)
        structured_data = {"mood_entry_id": entry.id, "self_report": entry.self_report}

    elif decision.intent == IntentEnum.JOURNAL_FREE:
        repo = JournalRepository(db)
        entry = repo.create(user.id, request.message)
        reply = "Journal entry saved."
        structured_data = {"journal_entry_id": entry.id}

    elif decision.intent == IntentEnum.SMALLTALK:
        reply = "Hello! I'm here to help you log your expenses and mood. What's on your mind?"

    else:
        reply = "I didn't quite catch that. You can log an expense, check your spending, or log your mood."

    return ChatResponse(
        reply=reply,
        intent=decision.intent,
        structured_data=structured_data,
        crisis_flagged=False,
    )


@router.post("/finance/transactions", response_model=TransactionRead)
def create_transaction(txn: TransactionCreate, deps: tuple[Session, User] = Depends(guarded)):
    db, user = deps
    return finance_skill.create_transaction_manual(db, user, txn)


@router.get("/finance/summary", response_model=FinanceSummaryResponse)
def get_finance_summary(
    category: Category | None = None,
    date_range: DateRangeEnum = DateRangeEnum.THIS_MONTH,
    deps: tuple[Session, User] = Depends(guarded),
):
    db, user = deps
    return finance_skill.get_summary(db, user, date_range, category)


@router.post("/mood/checkin", response_model=MoodCheckinRead)
def create_mood_checkin(checkin: MoodCheckinCreate, deps: tuple[Session, User] = Depends(guarded)):
    db, user = deps

    if checkin.note:
        crisis_res = scan_for_crisis(checkin.note)
        if crisis_res.matched:
            raise HTTPException(status_code=400, detail=CRISIS_RESPONSE)

    repo = MoodRepository(db)
    return repo.create(
        user_id=user.id,
        self_report=checkin.self_report,
        sleep_hours=checkin.sleep_hours,
        energy=checkin.energy,
        social_contact=checkin.social_contact,
        note=checkin.note,
    )
