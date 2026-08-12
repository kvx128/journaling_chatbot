from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from shared.models.enums import (
    BudgetPeriod,
    Category,
    DateRangeEnum,
    Direction,
    IntentEnum,
    TxnSource,
)


class ExtractionResult(BaseModel):
    intent: IntentEnum
    amount_minor: Optional[int] = None
    category: Optional[Category] = None
    occurred_on: Optional[date] = None
    merchant: Optional[str] = None
    payment_method: Optional[str] = None
    direction: Direction = Direction.DEBIT
    mood_score: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    explanation: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    intent: IntentEnum
    structured_data: Optional[dict[str, Any]] = None
    crisis_flagged: bool = False


class TransactionCreate(BaseModel):
    amount_minor: int = Field(gt=0, description="Amount in minor units (e.g., paise)")
    category: Category = Category.OTHER
    direction: Direction = Direction.DEBIT
    merchant: Optional[str] = None
    occurred_on: Optional[date] = None
    payment_method: Optional[str] = None
    source: TxnSource = TxnSource.API
    raw_text: Optional[str] = None
    confirmed: bool = True


class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    amount_minor: int
    direction: Direction
    category: Category
    merchant: Optional[str] = None
    occurred_on: date
    payment_method: Optional[str] = None
    source: TxnSource
    raw_text: Optional[str] = None
    extraction_confidence: Optional[float] = None
    confirmed: bool
    created_at: datetime


class MoodCheckinCreate(BaseModel):
    self_report: int = Field(ge=1, le=5, description="Self reported mood rating 1 to 5")
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    energy: Optional[int] = Field(default=None, ge=1, le=5)
    social_contact: Optional[bool] = None
    note: Optional[str] = None


class MoodCheckinRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    self_report: int
    sleep_hours: Optional[float] = None
    energy: Optional[int] = None
    social_contact: Optional[bool] = None
    note: Optional[str] = None
    recorded_at: datetime


class CategorySummaryItem(BaseModel):
    category: Category
    total_minor: int
    count: int


class FinanceSummaryResponse(BaseModel):
    date_range: DateRangeEnum
    start_date: date
    end_date: date
    total_debit_minor: int
    total_credit_minor: int
    net_minor: int
    transaction_count: int
    by_category: list[CategorySummaryItem]


class CategoriesResponse(BaseModel):
    categories: list[str]
