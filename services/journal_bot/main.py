from __future__ import annotations

from fastapi import FastAPI
from services.journal_bot.api import router


def create_app() -> FastAPI:
    app_instance = FastAPI(title="Journal Bot Internal API")
    app_instance.include_router(router)
    return app_instance


app = create_app()
