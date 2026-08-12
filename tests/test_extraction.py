from __future__ import annotations

from datetime import date, timedelta

from shared.extraction.amount import extract_amount
from shared.extraction.category import extract_category
from shared.extraction.date_extract import extract_date
from shared.extraction.finance import extract_finance
from shared.extraction.mood import extract_mood_signal
from shared.models.enums import Category, IntentEnum


def test_plain_extraction():
    res = extract_finance("450 groceries")
    assert res.amount_minor == 45000
    assert res.category == Category.GROCERIES
    assert res.intent == IntentEnum.FINANCE_LOG


def test_verbose_extraction_high_confidence():
    res = extract_finance("I spent about 450 rupees on groceries today")
    assert res.amount_minor == 45000
    assert res.category == Category.GROCERIES
    assert res.confidence >= 0.85
    assert res.intent == IntentEnum.FINANCE_LOG


def test_buried_amount():
    res = extract_finance("the bill came to about 1.2k after the discount")
    assert res.amount_minor == 120000
    assert res.intent == IntentEnum.FINANCE_LOG


def test_date_extraction_no_explicit_date():
    today = date.today()
    resolved_date, explicit = extract_date("just bought some stuff")
    assert resolved_date == today
    assert not explicit


def test_date_extraction_explicit_past_date():
    today = date.today()
    days_since_tuesday = (today.weekday() - 1) % 7
    if days_since_tuesday == 0:
        days_since_tuesday = 7
    expected_tuesday = today - timedelta(days=days_since_tuesday)

    resolved_date, explicit = extract_date("last Tuesday")
    assert explicit
    assert resolved_date == expected_tuesday


def test_ambiguous_no_amount_text():
    res = extract_finance("I went to the store but didn't find anything")
    assert res.amount_minor is None


def test_mixed_finance_and_mood():
    text = "Feeling really drained today, spent 500 on zomato to cope"
    res = extract_finance(text)
    assert res.amount_minor == 50000
    assert res.category == Category.FOOD_DELIVERY
    assert res.intent == IntentEnum.FINANCE_LOG

    has_mood, kws, score = extract_mood_signal(text)
    assert has_mood
    assert "drained" in kws


def test_pure_mood_not_finance():
    text = "feeling really drained today"
    res = extract_finance(text)
    assert res.intent == IntentEnum.MOOD_CHECKIN
    assert res.amount_minor is None


def test_pure_smalltalk_not_finance():
    text = "hey how's it going"
    res = extract_finance(text)
    assert res.intent == IntentEnum.UNKNOWN
    assert res.amount_minor is None


def test_extract_amount_rs_suffix():
    match = extract_amount("450rs")
    assert match is not None
    assert match.amount_minor == 45000


def test_extract_amount_decimal():
    match = extract_amount("1,200.50")
    assert match is not None
    assert match.amount_minor == 120050


def test_extract_amount_decimal_k():
    match = extract_amount("4.5k")
    assert match is not None
    assert match.amount_minor == 450000


def test_extract_category_subscriptions():
    cat, _ = extract_category("netflix subscription renewal")
    assert cat == Category.SUBSCRIPTIONS


def test_finance_query_intent():
    res = extract_finance("how much did I spend on food delivery this month?")
    assert res.intent == IntentEnum.FINANCE_QUERY
    assert res.category == Category.FOOD_DELIVERY
