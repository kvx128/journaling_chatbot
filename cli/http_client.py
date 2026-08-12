from __future__ import annotations

import os
import httpx


class CLIConnectionError(Exception):
    pass


class JournalAPIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or os.getenv("JOURNAL_API_URL", "http://127.0.0.1:8000")
        self.api_key = api_key or os.getenv("JOURNAL_API_KEY", "dev-local-key")
        self.headers = {"X-API-Key": self.api_key}
        self.client = httpx.Client(base_url=self.base_url, headers=self.headers, timeout=10.0)

    def _handle_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.NetworkError, httpx.ConnectTimeout) as e:
            raise CLIConnectionError(
                f"Could not connect to the API server at {self.base_url}. Is the server running? Try: journal serve"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                try:
                    data = e.response.json()
                    detail = data.get("detail", str(e))
                    raise CLIConnectionError(f"API Error: {detail}") from e
                except Exception:
                    pass
            raise

    def health(self) -> dict:
        resp = self._handle_request("GET", "/health")
        return resp.json()

    def chat(self, message: str) -> dict:
        resp = self._handle_request("POST", "/chat", json={"message": message})
        return resp.json()

    def create_transaction(self, payload: dict) -> dict:
        resp = self._handle_request("POST", "/finance/transactions", json=payload)
        return resp.json()

    def get_summary(self, category: str | None = None, date_range: str = "THIS_MONTH") -> dict:
        params = {"date_range": date_range}
        if category:
            params["category"] = category
        resp = self._handle_request("GET", "/finance/summary", params=params)
        return resp.json()

    def mood_checkin(self, payload: dict) -> dict:
        resp = self._handle_request("POST", "/mood/checkin", json=payload)
        return resp.json()

    def get_categories(self) -> list[str]:
        resp = self._handle_request("GET", "/taxonomy/categories")
        data = resp.json()
        return data.get("categories", [])
