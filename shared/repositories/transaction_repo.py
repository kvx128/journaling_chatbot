from __future__ import annotations

from datetime import date
from typing import Any
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from shared.models.enums import Category, Direction, TxnSource
from shared.models.orm import Transaction
from shared.repositories.base import BaseRepository


class TransactionRepository(BaseRepository[Transaction]):
    def create(
        self,
        user_id: int,
        amount_minor: int,
        category: Category = Category.OTHER,
        direction: Direction = Direction.DEBIT,
        merchant: str | None = None,
        occurred_on: date | None = None,
        payment_method: str | None = None,
        source: TxnSource = TxnSource.CHAT,
        raw_text: str | None = None,
        extraction_confidence: float | None = None,
        confirmed: bool = True,
    ) -> Transaction:
        if occurred_on is None:
            occurred_on = date.today()

        txn = Transaction(
            user_id=user_id,
            amount_minor=amount_minor,
            category=category,
            direction=direction,
            merchant=merchant,
            occurred_on=occurred_on,
            payment_method=payment_method,
            source=source,
            raw_text=raw_text,
            extraction_confidence=extraction_confidence,
            confirmed=confirmed,
        )
        self.db.add(txn)
        self.db.commit()
        self.db.refresh(txn)
        return txn

    def list_for_user(
        self,
        user_id: int,
        category: Category | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 100,
    ) -> list[Transaction]:
        stmt = select(Transaction).where(Transaction.user_id == user_id)

        if category is not None:
            stmt = stmt.where(Transaction.category == category)
        if start_date is not None:
            stmt = stmt.where(Transaction.occurred_on >= start_date)
        if end_date is not None:
            stmt = stmt.where(Transaction.occurred_on <= end_date)

        stmt = stmt.order_by(Transaction.occurred_on.desc(), Transaction.id.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def summarize(
        self,
        user_id: int,
        start_date: date,
        end_date: date,
        category: Category | None = None,
    ) -> dict[str, Any]:
        """Aggregate SUM/COUNT via actual SQL aggregation, grouped by category."""
        debit_case = case((Transaction.direction == Direction.DEBIT, Transaction.amount_minor), else_=0)
        credit_case = case((Transaction.direction == Direction.CREDIT, Transaction.amount_minor), else_=0)

        tot_stmt = select(
            func.coalesce(func.sum(debit_case), 0).label("total_debit"),
            func.coalesce(func.sum(credit_case), 0).label("total_credit"),
            func.count(Transaction.id).label("txn_count"),
        ).where(
            Transaction.user_id == user_id,
            Transaction.occurred_on >= start_date,
            Transaction.occurred_on <= end_date,
        )

        if category is not None:
            tot_stmt = tot_stmt.where(Transaction.category == category)

        tot_row = self.db.execute(tot_stmt).one()
        total_debit = int(tot_row.total_debit)
        total_credit = int(tot_row.total_credit)
        txn_count = int(tot_row.txn_count)

        cat_stmt = select(
            Transaction.category,
            func.coalesce(func.sum(debit_case), 0).label("cat_total"),
            func.count(Transaction.id).label("cat_count"),
        ).where(
            Transaction.user_id == user_id,
            Transaction.occurred_on >= start_date,
            Transaction.occurred_on <= end_date,
        )

        if category is not None:
            cat_stmt = cat_stmt.where(Transaction.category == category)

        cat_stmt = cat_stmt.group_by(Transaction.category)
        cat_rows = self.db.execute(cat_stmt).all()

        by_category = [
            {
                "category": row.category,
                "total_minor": int(row.cat_total),
                "count": int(row.cat_count),
            }
            for row in cat_rows
        ]

        net_minor = total_credit - total_debit

        return {
            "total_debit_minor": total_debit,
            "total_credit_minor": total_credit,
            "net_minor": net_minor,
            "transaction_count": txn_count,
            "by_category": by_category,
        }
