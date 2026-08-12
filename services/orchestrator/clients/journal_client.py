from __future__ import annotations

import os
from typing import Any
import httpx


class JournalClient:
    def __init__(self):
        self.base_url = os.getenv("JOURNAL_BOT_URL", "http://127.0.0.1:8002")
        self.client = httpx.Client(base_url=self.base_url, timeout=5.0)

    def is_ready(self) -> bool:
        try:
            res = self.client.get("/ready")
            return res.status_code == 200
        except Exception:
            return False

    def handle(self, user_id: int, text: str, task: str) -> dict[str, Any]:
        res = self.client.post("/handle", json={"user_id": user_id, "text": text, "task": task})
        res.raise_for_status()
        return res.json()

    def create_mood_checkin(self, payload: dict[str, Any]) -> dict[str, Any]:
        res = self.client.post("/mood/checkin", json=payload)
        res.raise_for_status()
        return res.json()
