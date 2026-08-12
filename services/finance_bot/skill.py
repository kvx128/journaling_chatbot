from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from shared.extraction.category import extract_category
from shared.extraction.finance import extract_finance
from shared.models.enums import Category, DateRangeEnum, TxnSource
from shared.models.schemas import FinanceSummaryResponse
from shared.repositories.transaction_repo import TransactionRepository


def log_transaction_from_text(db: Session, user_id: int, text: str) -> tuple[str, dict[str, Any] | None]:
    ext_res = extract_finance(text)
    if not ext_res.amount_minor:
        return "I think this is a finance log, but I couldn't find an exact amount.", None

    confirmed = ext_res.confidence >= 0.85
    repo = TransactionRepository(db)

    txn = repo.create(
        user_id=user_id,
        amount_minor=ext_res.amount_minor,
        category=ext_res.category or Category.OTHER,
        direction=ext_res.direction,
        merchant=ext_res.merchant,
        occurred_on=ext_res.occurred_on,
        payment_method=ext_res.payment_method,
        source=TxnSource.CHAT,
        raw_text=text,
        extraction_confidence=ext_res.confidence,
        confirmed=confirmed,
    )

    amt = txn.amount_minor / 100
    reply = f"Logged a debit of {amt:.2f} for {txn.category.value}."
    if not confirmed:
        reply += " (Confidence was a bit low, marked as unconfirmed.)"

    structured = {
        "transaction_id": txn.id,
        "amount_minor": txn.amount_minor,
        "category": txn.category.value,
    }
    return reply, structured


def _get_dates_for_range(date_range: DateRangeEnum) -> tuple[date, date]:
    today = date.today()
    if date_range == DateRangeEnum.TODAY:
        return today, today
    elif date_range == DateRangeEnum.THIS_WEEK:
        start = today - timedelta(days=today.weekday())
        return start, today
    elif date_range == DateRangeEnum.THIS_MONTH:
        start = today.replace(day=1)
        return start, today
    elif date_range == DateRangeEnum.LAST_30D:
        start = today - timedelta(days=30)
        return start, today
    return today, today


def get_summary(db: Session, user_id: int, date_range: DateRangeEnum, category: Category | None = None) -> FinanceSummaryResponse:
    start_date, end_date = _get_dates_for_range(date_range)
    repo = TransactionRepository(db)
    summary_data = repo.summarize(user_id, start_date, end_date, category)

    return FinanceSummaryResponse(
        date_range=date_range,
        start_date=start_date,
        end_date=end_date,
        total_debit_minor=summary_data["total_debit_minor"],
        total_credit_minor=summary_data["total_credit_minor"],
        net_minor=summary_data["net_minor"],
        transaction_count=summary_data["transaction_count"],
        by_category=summary_data["by_category"],
    )


def answer_finance_query(db: Session, user_id: int, text: str) -> tuple[str, dict[str, Any] | None]:
    text_lower = text.lower()

    date_range = DateRangeEnum.THIS_MONTH
    if re.search(r"\b(today)\b", text_lower):
        date_range = DateRangeEnum.TODAY
    elif re.search(r"\b(this week)\b", text_lower):
        date_range = DateRangeEnum.THIS_WEEK
    elif re.search(r"\b(last 30 days|30 days)\b", text_lower):
        date_range = DateRangeEnum.LAST_30D

    category, _ = extract_category(text)
    summary = get_summary(db, user_id, date_range, category)

    amt = summary.total_debit_minor / 100
    period_str = date_range.value.replace("_", " ").lower()
    cat_str = f" on {category.value}" if category else ""

    reply = f"You've spent {amt:.2f}{cat_str} {period_str}."

    if summary.transaction_count > 0 and not category and len(summary.by_category) > 0:
        top_cat = sorted(summary.by_category, key=lambda x: x.total_minor, reverse=True)[0]
        top_amt = top_cat.total_minor / 100
        reply += f" Your biggest expense category is {top_cat.category.value} ({top_amt:.2f})."

    return reply, None
