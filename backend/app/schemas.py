# backend/app/schemas.py
from typing import Literal

from pydantic import BaseModel, Field
class AnswerRequest(BaseModel):
    # Task 5: cap request body string fields so a malicious/broken client
    # can't send an arbitrarily huge payload (memory/CPU cost before we ever
    # reject it). Limits are generous for real use: 4000 chars covers a long
    # multi-paragraph question, 200/500 comfortably exceed any real
    # provider model id / API key length.
    question: str = Field(max_length=4000)
    provider: Literal["openai", "groq", "gemini", "anthropic"]
    model: str = Field(max_length=200)
    api_key: str = Field(max_length=500)
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
