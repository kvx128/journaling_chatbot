from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.model_server import inference

router = APIRouter()

class InferRequest(BaseModel):
    text: str

@router.get("/ready")
def get_ready():
    if inference.is_ready():
        return {"status": "ok"}
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Model failed to load: {inference._load_error}"
        )

@router.post("/infer/journal")
def post_infer_journal(req: InferRequest):
    return inference.infer_mood(req.text)
