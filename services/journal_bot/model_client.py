from __future__ import annotations

import os
import httpx


class ModelServerClient:
    def __init__(self):
        self.base_url = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:8003")
        self.client = httpx.Client(base_url=self.base_url, timeout=15.0)

    def infer_mood(self, text: str) -> dict | None:
        try:
            res = self.client.post("/infer/journal", json={"text": text})
            res.raise_for_status()
            data = res.json()

            if data.get("_parse_error") or data.get("valence") is None or data.get("arousal") is None:
                return None

            return data
        except (httpx.RequestError, httpx.HTTPStatusError):
            return None
