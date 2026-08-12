from typing import List, Optional
from pydantic import BaseModel, Field
import logging

from shared.models.enums import IntentEnum, Category


class RouterExample(BaseModel):
    text: str
    intent: IntentEnum


class RouterBatch(BaseModel):
    examples: List[RouterExample]


class FinanceExample(BaseModel):
    text: str
    amount_minor: Optional[int]
    category: Optional[Category]
    merchant: Optional[str]


class FinanceBatch(BaseModel):
    examples: List[FinanceExample]


class JournalExample(BaseModel):
    text: str
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=-1.0, le=1.0)
    emotion_tags: List[str]


class JournalBatch(BaseModel):
    examples: List[JournalExample]


def validate_batch(task: str, raw: dict) -> List[BaseModel]:
    valid_items = []

    if "examples" not in raw or not isinstance(raw["examples"], list):
        logging.error("Raw batch missing 'examples' list.")
        return valid_items

    for item in raw["examples"]:
        try:
            if task == "router":
                valid_items.append(RouterExample(**item))
            elif task == "finance":
                valid_items.append(FinanceExample(**item))
            elif task == "journal":
                valid_items.append(JournalExample(**item))
            else:
                logging.error(f"Unknown task type: {task}")
                break
        except Exception as e:
            logging.warning(f"Validation failed for item: {item}. Error: {e}")
            continue

    return valid_items
