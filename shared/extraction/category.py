from __future__ import annotations

import re
from shared.models.enums import Category

CATEGORY_KEYWORD_MAP: dict[Category, list[str]] = {
    Category.FOOD_DELIVERY: [
        "swiggy", "zomato", "eatclub", "magicpin", "food delivery", "food order", "ordered food", "order food"
    ],
    Category.DINING: [
        "restaurant", "cafe", "dining", "diner", "mcdonalds", "mcdonald's", "kfc", "dominos", "domino's",
        "pizza", "burger", "starbucks", "eatery", "dhaba", "pub", "bar", "bistro", "coffee", "lunch", "dinner", "breakfast"
    ],
    Category.GROCERIES: [
        "blinkit", "zepto", "instamart", "bigbasket", "supermarket", "grocery", "groceries", "milk",
        "vegetables", "fruits", "vegetable", "fruit", "kirana", "dmart", "veggies", "ration"
    ],
    Category.TRANSPORT: [
        "uber", "ola", "rapido", "cab", "taxi", "auto", "metro", "bus", "train", "fare", "rickshaw",
        "train ticket", "flight ticket", "toll", "parking", "commute"
    ],
    Category.FUEL: [
        "petrol", "diesel", "fuel", "cng", "hpcl", "bpcl", "iocl", "indian oil", "shell", "gas station"
    ],
    Category.RENT: [
        "house rent", "pg rent", "flat rent", "room rent", "rent", "maintenance fee", "landlord"
    ],
    Category.UTILITIES: [
        "electricity", "water bill", "gas bill", "cylinder", "broadband", "wifi bill", "electricity bill",
        "power bill", "utility", "utility bill"
    ],
    Category.MOBILE_INTERNET: [
        "mobile recharge", "airtel", "jio", "vi", "vodafone", "internet recharge", "phone recharge", "data pack"
    ],
    Category.SUBSCRIPTIONS: [
        "netflix", "spotify", "prime", "youtube", "disney", "hotstar", "apple", "icloud", "chatgpt", "subscription",
        "patreon", "medium"
    ],
    Category.SHOPPING: [
        "amazon", "flipkart", "myntra", "ajio", "meesho", "clothes", "shoes", "mall", "shopping", "electronics",
        "zudio", "zara", "h&m", "apparel"
    ],
    Category.HEALTH: [
        "pharmacy", "medicine", "doctor", "hospital", "clinic", "lab", "apollo", "pharmeasy", "1mg", "medical",
        "chemist", "pills", "health checkup"
    ],
    Category.FITNESS: [
        "gym", "cult", "cult.fit", "fitness", "workout", "trainer", "protein", "whey", "gym membership"
    ],
    Category.EDUCATION: [
        "course", "tuition", "books", "school", "college", "udemy", "coursera", "exam fee", "fee", "textbook"
    ],
    Category.ENTERTAINMENT: [
        "movie", "cinema", "pvr", "inox", "concert", "event", "game", "gaming", "show", "play", "amusement"
    ],
    Category.TRAVEL: [
        "hotel", "resort", "flight", "trip", "vacation", "airbnb", "makemytrip", "mmt", "goibibo", "booking.com", "stay"
    ],
    Category.GIFTS: [
        "gift", "present", "flowers", "birthday gift", "anniversary gift", "gifting"
    ],
    Category.INSURANCE: [
        "insurance", "premium", "lic", "health insurance", "car insurance", "term insurance"
    ],
    Category.SAVINGS_INVESTMENT: [
        "sip", "mutual fund", "zerodha", "groww", "stocks", "investment", "savings", "fd", "gold", "crypto"
    ],
}


def extract_category(text: str) -> tuple[Category | None, str | None]:
    if not text:
        return None, None

    text_lower = text.lower()
    matches: list[tuple[Category, str, int]] = []

    for cat, keywords in CATEGORY_KEYWORD_MAP.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                matches.append((cat, kw, len(kw)))

    if matches:
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[0][0], matches[0][1]

    return None, None
