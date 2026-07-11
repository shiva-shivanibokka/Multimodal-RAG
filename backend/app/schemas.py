# backend/app/schemas.py
from pydantic import BaseModel
class AnswerRequest(BaseModel):
    question: str
    provider: str            # openai | groq | gemini | anthropic
    model: str
    api_key: str
    session_id: str | None = None
    retrieval_mode: str = "hybrid"   # hybrid | cross_modal | caption_baseline | dense
    verified: bool = True
class Citation(BaseModel):
    page: int
    bbox: list[float]
    snippet: str
class Claim(BaseModel):
    text: str
    supported: bool
    score: float
    citations: list[Citation] = []
class AnswerResponse(BaseModel):
    answer: str
    refused: bool
    claims: list[Claim] = []
    citations: list[Citation] = []
