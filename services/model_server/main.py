from fastapi import FastAPI
from services.model_server.api import router

def create_app() -> FastAPI:
    app_instance = FastAPI(title="Model Server")
    app_instance.include_router(router)
    return app_instance

app = create_app()
