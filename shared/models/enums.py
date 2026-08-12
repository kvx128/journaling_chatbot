from __future__ import annotations

from enum import Enum


class Category(str, Enum):
    GROCERIES = "GROCERIES"
    DINING = "DINING"
    FOOD_DELIVERY = "FOOD_DELIVERY"
    TRANSPORT = "TRANSPORT"
    FUEL = "FUEL"
    RENT = "RENT"
    UTILITIES = "UTILITIES"
    MOBILE_INTERNET = "MOBILE_INTERNET"
    SUBSCRIPTIONS = "SUBSCRIPTIONS"
    SHOPPING = "SHOPPING"
    HEALTH = "HEALTH"
    FITNESS = "FITNESS"
    EDUCATION = "EDUCATION"
    ENTERTAINMENT = "ENTERTAINMENT"
    TRAVEL = "TRAVEL"
    GIFTS = "GIFTS"
    INSURANCE = "INSURANCE"
    SAVINGS_INVESTMENT = "SAVINGS_INVESTMENT"
    OTHER = "OTHER"


class Direction(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class TxnSource(str, Enum):
    CHAT = "chat"
    API = "api"


class BudgetPeriod(str, Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DateRangeEnum(str, Enum):
    TODAY = "TODAY"
    THIS_WEEK = "THIS_WEEK"
    THIS_MONTH = "THIS_MONTH"
    LAST_30D = "LAST_30D"


class IntentEnum(str, Enum):
    FINANCE_LOG = "FINANCE_LOG"
    FINANCE_QUERY = "FINANCE_QUERY"
    MOOD_CHECKIN = "MOOD_CHECKIN"
    JOURNAL_FREE = "JOURNAL_FREE"
    SMALLTALK = "SMALLTALK"
    UNKNOWN = "UNKNOWN"
