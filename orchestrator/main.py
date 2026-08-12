from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.core.logging import setup_logging
from orchestrator.api import guarded, open as open_api


def create_app() -> FastAPI:
    setup_logging()

    app_instance = FastAPI(
        title="Journaling Chatbot",
        version="0.1.0",
        description="Phase 1 orchestrator",
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_instance.include_router(open_api.router)
    app_instance.include_router(guarded.router)

    return app_instance


app = create_app()
