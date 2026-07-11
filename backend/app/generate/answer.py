# backend/app/generate/answer.py
from app.schemas import AnswerRequest, AnswerResponse
from app.generate.providers import generate
def answer_question(req: AnswerRequest, context: str, images=None) -> AnswerResponse:
    system = ("Answer ONLY using the provided context. If the context does not "
              "contain the answer, reply exactly: NOT_IN_DOCUMENTS.")
    user = f"Context:\n{context}\n\nQuestion: {req.question}"
    text = generate(req.provider, req.model, req.api_key,
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}], images=images)
    refused = text.strip() == "NOT_IN_DOCUMENTS"
    return AnswerResponse(answer="" if refused else text, refused=refused)
