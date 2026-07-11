# backend/app/main.py
from fastapi import FastAPI, Depends, HTTPException, Header
from app.config import settings
from app.schemas import AnswerRequest, AnswerResponse
app = FastAPI(title="Multimodal RAG Trust Layer")

def require_token(authorization: str = Header(default="")):
    if settings.backend_token and authorization != f"Bearer {settings.backend_token}":
        raise HTTPException(status_code=401, detail="unauthorized")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/answer", response_model=AnswerResponse, dependencies=[Depends(require_token)])
def answer(req: AnswerRequest):
    from app.generate.answer import answer_question
    return answer_question(req, context="")   # context wired in later phases
