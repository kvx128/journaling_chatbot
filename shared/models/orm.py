from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.models.base import Base
from shared.models.enums import BudgetPeriod, Category, Direction, TxnSource


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    handle: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    tz: Mapped[str] = mapped_column(String(64), nullable=False, server_default="Asia/Kolkata")
    currency: Mapped[str] = mapped_column(String(10), nullable=False, server_default="INR")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    transactions: Mapped[list[Transaction]] = relationship(back_populates="user", cascade="all, delete-orphan")
    budgets: Mapped[list[Budget]] = relationship(back_populates="user", cascade="all, delete-orphan")
    mood_entries: Mapped[list[MoodEntry]] = relationship(back_populates="user", cascade="all, delete-orphan")
    journal_entries: Mapped[list[JournalEntry]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    amount_minor: Mapped[int] = mapped_column(Integer, nullable=False)
    direction: Mapped[Direction] = mapped_column(
        Enum(Direction, native_enum=False), nullable=False, server_default=Direction.DEBIT.value
    )
    category: Mapped[Category] = mapped_column(
        Enum(Category, native_enum=False), nullable=False, server_default=Category.OTHER.value
    )
    merchant: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    occurred_on: Mapped[date] = mapped_column(Date, nullable=False)
    payment_method: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source: Mapped[TxnSource] = mapped_column(
        Enum(TxnSource, native_enum=False), nullable=False, server_default=TxnSource.CHAT.value
    )
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extraction_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user: Mapped[User] = relationship(back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_user_occurred", "user_id", "occurred_on"),
    )


class Budget(Base):
    __tablename__ = "budgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    category: Mapped[Category] = mapped_column(Enum(Category, native_enum=False), nullable=False)
    period: Mapped[BudgetPeriod] = mapped_column(
        Enum(BudgetPeriod, native_enum=False), nullable=False, server_default=BudgetPeriod.MONTHLY.value
    )
    limit_minor: Mapped[int] = mapped_column(Integer, nullable=False)
    active_from: Mapped[date] = mapped_column(Date, nullable=False)
    active_to: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    user: Mapped[User] = relationship(back_populates="budgets")


class MoodEntry(Base):
    __tablename__ = "mood_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    self_report: Mapped[int] = mapped_column(Integer, nullable=False)
    sleep_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    energy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    social_contact: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user: Mapped[User] = relationship(back_populates="mood_entries")

    __table_args__ = (
        CheckConstraint("self_report >= 1 AND self_report <= 5", name="chk_self_report_range"),
        CheckConstraint("energy IS NULL OR (energy >= 1 AND energy <= 5)", name="chk_energy_range"),
        Index("ix_mood_entries_user_recorded", "user_id", "recorded_at"),
    )


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user: Mapped[User] = relationship(back_populates="journal_entries")
