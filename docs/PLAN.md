# Multimodal RAG — "Trust Layer" for Scanned Enterprise Documents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A multimodal document-QA system that answers *only* what your scanned documents support — verifying every claim against source evidence, refusing when grounding is absent, and proving each answer with pixel-level citations.

**Architecture:** Two tiers. A headless **FastAPI + local-ML backend** on a free CPU Hugging Face Docker Space handles ingestion (OCR, layout, tables, CLIP + text embeddings), hybrid retrieval, an NLI faithfulness gate, and BYOK generation. A **Next.js frontend on Vercel** (shadcn/ui + Tailwind) handles upload, chat, the visual-citation viewer, the evaluation dashboard, and BYOK key entry. The two communicate over HTTPS; the user's API key is forwarded per-request and never stored.

**Tech Stack:** Backend — Python 3.11, FastAPI, PyMuPDF, docTR (OCR), img2table, sentence-transformers (bge-small text embed, clip-ViT-B-32 image embed, bge-reranker-base, nli-deberta-v3-base), faiss-cpu, rank-bm25. Frontend — Next.js (App Router, TypeScript), shadcn/ui, Tailwind. Providers — OpenAI, Groq, Gemini (OpenAI-compatible), Anthropic.

## Global Constraints

- **No GPU anywhere.** Free CPU HF Space (2 vCPU / 16GB RAM). No ZeroGPU (requires HF Pro — unavailable). Every model must run on CPU.
- **No paid infra.** No Supabase, Render, Fly.io, or hosted vector DB. Vector index is in-process FAISS on Space local disk.
- **BYOK only for generation.** Ingestion + retrieval + verification are 100% local/free. The user's provider API key is entered in the UI, sent per-request in a header, and **never persisted** server-side.
- **One embedding library.** All transformer models (text embed, CLIP, reranker, NLI) load through `sentence-transformers` to minimize the dependency surface and RAM footprint.
- **Provenance is mandatory.** Every text chunk, table cell, and image region carries `(page_number, bbox)` from ingestion onward — citations depend on it.
- **Backend is headless.** No Gradio, no server-rendered UI. FastAPI returns JSON only.
- **Backend auth:** a shared `BACKEND_TOKEN` bearer header gates all endpoints so the Space isn't an open proxy; generation additionally requires a BYOK key.
- **TDD, DRY, YAGNI, frequent commits.** Every non-trivial unit ships with a runnable test.

---

## Repository Structure

```
Multimodal-RAG/
├── docs/PLAN.md                         # this file
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app, routes, auth dependency
│   │   ├── config.py                    # settings, model ids, thresholds
│   │   ├── schemas.py                   # Pydantic request/response models
│   │   ├── ingest/
│   │   │   ├── loader.py                # PDF/image → page images + text layer (PyMuPDF)
│   │   │   ├── ocr.py                   # docTR OCR on scanned pages → words + bbox
│   │   │   ├── tables.py                # img2table → structured tables + cell bbox
│   │   │   └── chunk.py                 # blocks → chunks w/ (page, bbox) provenance
│   │   ├── index/
│   │   │   ├── embedders.py             # bge text + clip image (sentence-transformers)
│   │   │   ├── store.py                 # FAISS text index + FAISS image index + BM25
│   │   │   └── rerank.py                # bge-reranker cross-encoder (optional)
│   │   ├── retrieve/
│   │   │   └── hybrid.py                # dense+BM25 fusion, cross-modal, mode toggle
│   │   ├── verify/
│   │   │   └── nli.py                   # claim splitting + NLI faithfulness gate
│   │   ├── generate/
│   │   │   ├── providers.py             # BYOK adapter: openai-compat + anthropic
│   │   │   └── answer.py                # prompt build, generate, citation mapping
│   │   └── session.py                   # per-session index/doc store (in-process)
│   ├── eval/
│   │   ├── gold/enterprise_docs.json    # ~40 Q/A with source pages
│   │   ├── metrics.py                   # recall@k, MRR, citation acc, faithfulness, refusal
│   │   └── run_eval.py                  # runs suite → report.json
│   ├── tests/                           # pytest, mirrors app/ structure
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md                        # HF Space card (sdk: docker)
├── frontend/
│   ├── app/
│   │   ├── page.tsx                     # chat + upload + citation viewer
│   │   ├── eval/page.tsx                # evaluation dashboard
│   │   └── api/                         # thin route handlers proxying to backend
│   ├── components/                      # shadcn/ui + custom (CitationViewer, KeyForm...)
│   ├── lib/backend.ts                   # typed backend client
│   └── ...                              # standard Next.js scaffold
└── README.md                            # portfolio-facing top-level readme
```

**Session model (ponytail):** documents and their FAISS indexes are held per-session in an in-process dict keyed by a session id, persisted to Space disk. `# ponytail: single-process in-memory sessions; move to a shared store only if multi-replica scaling is ever needed.`

---

## Phase 0 — End-to-End Skeleton (prove the loop)

**Outcome:** Type a question in the Next.js UI with a BYOK key, and get a real LLM answer through the full frontend→backend→provider pipe, using a hardcoded context. No retrieval yet.

### Task 0.1: Backend app + auth + health

**Files:**
- Create: `backend/app/main.py`, `backend/app/config.py`, `backend/app/schemas.py`
- Create: `backend/requirements.txt`, `backend/tests/test_health.py`

**Interfaces:**
- Produces: `GET /health → {"status": "ok"}`; a `require_token` FastAPI dependency reading `Authorization: Bearer <BACKEND_TOKEN>`.

- [ ] **Step 1: Write the failing test**
```python
# backend/tests/test_health.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_protected_requires_token(monkeypatch):
    monkeypatch.setenv("BACKEND_TOKEN", "secret")
    r = client.post("/answer", json={"question": "hi", "provider": "groq", "model": "x", "api_key": "k"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**
Run: `cd backend && pytest tests/test_health.py -v`
Expected: FAIL (app not importable / route missing)

- [ ] **Step 3: Write minimal implementation**
```python
# backend/app/config.py
import os
class Settings:
    backend_token = os.getenv("BACKEND_TOKEN", "")
    text_model = "BAAI/bge-small-en-v1.5"
    clip_model = "clip-ViT-B-32"
    reranker_model = "BAAI/bge-reranker-base"
    nli_model = "cross-encoder/nli-deberta-v3-base"
    faithfulness_threshold = 0.5   # ponytail: tunable knob, not magic
    retrieval_min_score = 0.25     # below this → refuse
settings = Settings()
```
```python
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
```
```python
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
```
```
# backend/requirements.txt  (Phase 0 subset — grows per phase)
fastapi
uvicorn[standard]
pydantic>=2
httpx
pytest
```

- [ ] **Step 4: Run test to verify it passes**
Run: `cd backend && pytest tests/test_health.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add backend/ && git commit -m "feat(backend): FastAPI skeleton with health + token auth"
```

### Task 0.2: BYOK provider adapter

**Files:**
- Create: `backend/app/generate/providers.py`, `backend/app/generate/answer.py`
- Create: `backend/tests/test_providers.py`

**Interfaces:**
- Produces: `generate(provider, model, api_key, messages, images=None) -> str`. `messages` is OpenAI-style `[{"role","content"}]`; `images` is a list of base64 PNGs attached to the last user turn.
- Consumes: `AnswerRequest` from 0.1.

- [ ] **Step 1: Write the failing test** (adapter selects the right backend; no network — inject a fake transport)
```python
# backend/tests/test_providers.py
from app.generate.providers import route_provider
def test_route_openai_compatible():
    assert route_provider("groq")["kind"] == "openai_compat"
    assert route_provider("gemini")["kind"] == "openai_compat"
    assert route_provider("openai")["kind"] == "openai_compat"
def test_route_anthropic():
    assert route_provider("anthropic")["kind"] == "anthropic"
def test_unknown_provider_raises():
    import pytest
    with pytest.raises(ValueError):
        route_provider("cohere")
```

- [ ] **Step 2: Run to verify it fails**
Run: `cd backend && pytest tests/test_providers.py -v` → FAIL (module missing)

- [ ] **Step 3: Write minimal implementation**
```python
# backend/app/generate/providers.py
import httpx
_OPENAI_COMPAT = {
    "openai": "https://api.openai.com/v1",
    "groq":   "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}
def route_provider(provider: str) -> dict:
    if provider in _OPENAI_COMPAT:
        return {"kind": "openai_compat", "base_url": _OPENAI_COMPAT[provider]}
    if provider == "anthropic":
        return {"kind": "anthropic", "base_url": "https://api.anthropic.com/v1"}
    raise ValueError(f"unsupported provider: {provider}")

def generate(provider, model, api_key, messages, images=None, timeout=120) -> str:
    cfg = route_provider(provider)
    if cfg["kind"] == "openai_compat":
        payload = {"model": model, "messages": _attach_images_openai(messages, images)}
        r = httpx.post(f"{cfg['base_url']}/chat/completions",
                       headers={"Authorization": f"Bearer {api_key}"},
                       json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    # anthropic
    payload = {"model": model, "max_tokens": 1024,
               "messages": _attach_images_anthropic(messages, images)}
    r = httpx.post(f"{cfg['base_url']}/messages",
                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                   json=payload, timeout=timeout)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])

def _attach_images_openai(messages, images):
    if not images: return messages
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    parts = [{"type": "text", "text": last["content"]}]
    parts += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]
    last["content"] = parts
    return msgs

def _attach_images_anthropic(messages, images):
    if not images: return messages
    msgs = [dict(m) for m in messages]
    last = msgs[-1]
    parts = [{"type": "text", "text": last["content"]}]
    parts += [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}} for b in images]
    last["content"] = parts
    return msgs
```
```python
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
```

- [ ] **Step 4: Run to verify it passes**
Run: `cd backend && pytest tests/test_providers.py -v` → PASS

- [ ] **Step 5: Commit**
```bash
git add backend/app/generate backend/tests/test_providers.py && git commit -m "feat(backend): BYOK provider adapter (openai-compat + anthropic, vision-ready)"
```

### Task 0.3: Next.js shell + settings + chat wired to `/answer`

**Files:**
- Create: `frontend/` via `npx create-next-app@latest` (TypeScript, App Router, Tailwind)
- Create: `frontend/lib/backend.ts`, `frontend/components/KeyForm.tsx`, `frontend/app/page.tsx`

**Interfaces:**
- Consumes: backend `POST /answer` (typed by `AnswerResponse`).
- Produces: a settings panel (provider dropdown, model input, password key field held in React state / `sessionStorage` only) + a chat box that posts to the backend and renders the answer or the refusal state.

- [ ] **Step 1: Scaffold**
```bash
cd frontend && npx create-next-app@latest . --ts --tailwind --app --eslint --use-npm --yes
npx shadcn@latest init -y && npx shadcn@latest add button input textarea card select badge
```

- [ ] **Step 2: Typed backend client**
```typescript
// frontend/lib/backend.ts
export type Citation = { page: number; bbox: number[]; snippet: string };
export type Claim = { text: string; supported: boolean; score: number; citations: Citation[] };
export type AnswerResponse = { answer: string; refused: boolean; claims: Claim[]; citations: Citation[] };

export async function askBackend(body: {
  question: string; provider: string; model: string; api_key: string;
  session_id?: string; retrieval_mode?: string; verified?: boolean;
}): Promise<AnswerResponse> {
  const r = await fetch("/api/answer", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`backend ${r.status}`);
  return r.json();
}
```

- [ ] **Step 3: Proxy route (keeps BACKEND_TOKEN server-side, forwards BYOK key)**
```typescript
// frontend/app/api/answer/route.ts
export async function POST(req: Request) {
  const body = await req.text();
  const r = await fetch(`${process.env.BACKEND_URL}/answer`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${process.env.BACKEND_TOKEN}` },
    body,
  });
  return new Response(await r.text(), { status: r.status, headers: { "content-type": "application/json" } });
}
```

- [ ] **Step 4: Minimal chat page** — provider `<Select>` (openai/groq/gemini/anthropic), model `<Input>`, key `<Input type="password">` (sessionStorage), question `<Textarea>`, submit → `askBackend` → render `answer` or a "Not in your documents" badge when `refused`.

- [ ] **Step 5: Manual verify + commit**
Run backend (`uvicorn app.main:app --reload`) + frontend (`npm run dev`), set `BACKEND_URL`, ask a question with a real free Groq/Gemini key → confirm an answer renders end-to-end.
```bash
git add frontend/ && git commit -m "feat(frontend): Next.js shell with BYOK settings + chat wired to backend"
```

**Phase 0 gate:** full pipe works on a hardcoded context. Stop, review, then proceed.

---

## Phase 1 — Ingestion of Scanned Documents (with provenance)

**Outcome:** Upload a scanned/mixed PDF; get back structured chunks — text, tables, and figure regions — each tagged with `(page, bbox)`.

### Task 1.1: Page loader (PyMuPDF) → page images + native text layer + bbox
- **Files:** `backend/app/ingest/loader.py`, `backend/tests/test_loader.py`
- **Produces:** `load_document(bytes) -> list[Page]` where `Page = {index:int, image_png:bytes, width, height, text_blocks: list[{text, bbox}]}`. Born-digital pages yield text blocks directly; empty text layer marks the page for OCR.
- **Test:** a 1-page synthetic PDF with known text returns one page with a text block whose bbox is within page bounds.

### Task 1.2: OCR fallback (docTR) → words + bbox
- **Files:** `backend/app/ingest/ocr.py`, `backend/tests/test_ocr.py`
- **Produces:** `ocr_page(image_png) -> list[{text, bbox}]`. Called only when a page's native text layer is empty (scanned).
- **Test:** render a known phrase to an image, OCR it, assert the phrase (case-insensitive, allowing minor OCR noise) appears with a plausible bbox. `# ponytail: assert on token overlap, not exact string — OCR is never pixel-perfect.`
- **requirements.txt +=** `python-doctr[torch]`, `pymupdf`

### Task 1.3: Table extraction (img2table) → structured tables + cell bbox
- **Files:** `backend/app/ingest/tables.py`, `backend/tests/test_tables.py`
- **Produces:** `extract_tables(image_png) -> list[Table]` where `Table = {bbox, cells: list[{row, col, text, bbox}], markdown: str, df_json: str}`. `df_json` powers deterministic numeric answers in Phase 4.
- **Test:** an image of a 2×2 numeric table returns one table with 4 cells and a markdown round-trip; the parsed DataFrame sums correctly.
- **requirements.txt +=** `img2table`

### Task 1.4: Chunker → retrieval units with provenance
- **Files:** `backend/app/ingest/chunk.py`, `backend/tests/test_chunk.py`
- **Produces:** `chunk_pages(pages, tables) -> list[Chunk]` where `Chunk = {id, kind: "text"|"table"|"figure", text, page, bbox, table_df_json?}`. Text chunks are ~500-token windows respecting block boundaries; each table is its own chunk; each non-text image region is a `figure` chunk (image bytes referenced by id for CLIP in Phase 3).
- **Test:** given 2 pages + 1 table, returns chunks covering all text with monotonic ids and every chunk carrying a valid `(page, bbox)`.

### Task 1.5: `/ingest` endpoint + session store
- **Files:** modify `backend/app/main.py`; create `backend/app/session.py`; `backend/tests/test_ingest_endpoint.py`
- **Produces:** `POST /ingest (multipart file) -> {session_id, n_pages, n_chunks}`; `get_session(session_id)` returning stored chunks + page images.
- **Test:** post a small PDF → 200 with `n_chunks > 0`; `get_session` returns them.

**Phase 1 gate:** upload a real scanned sample → inspect chunks have correct provenance.

---

## Phase 2 — Text Hybrid Retrieval + Grounded Answers

**Outcome:** Ask about an uploaded doc and get a grounded answer with **page-level text citations**. (Images still ignored.)

### Task 2.1: Embedders (bge text) — `backend/app/index/embedders.py`
- **Produces:** `embed_texts(list[str]) -> np.ndarray` via `sentence-transformers` `bge-small-en-v1.5` (normalized). Lazy-load the model once. Test: two similar sentences score higher cosine than two unrelated ones.
- **requirements.txt +=** `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `numpy`

### Task 2.2: Store — FAISS text index + BM25 — `backend/app/index/store.py`
- **Produces:** `Index.add(chunks)`, `Index.dense(query, k)`, `Index.bm25(query, k)`, persisted per session. Test: add 5 chunks, query exact phrase → correct chunk is top-1 in both dense and bm25.

### Task 2.3: Hybrid fusion + optional rerank — `backend/app/retrieve/hybrid.py`, `backend/app/index/rerank.py`
- **Produces:** `retrieve(session, query, mode, k) -> list[ScoredChunk]` doing reciprocal-rank fusion of dense+bm25, then optional `bge-reranker` cross-encoder. `mode="dense"` skips bm25. Test: RRF ranks a chunk hit by both retrievers above one hit by a single retriever; reranker reorders a known pair.

### Task 2.4: Wire retrieval into `/answer` — modify `backend/app/generate/answer.py`, `main.py`
- **Produces:** `/answer` now: retrieve → if top score `< retrieval_min_score` return `refused=True` (no LLM call); else build context with `[page N]` tags, generate, and attach `citations` from the retrieved chunks' `(page, bbox)`. Test: question answerable from an ingested doc returns a non-refused answer with ≥1 citation whose page matches the source.

**Phase 2 gate:** end-to-end cited answer on text. First deployable-quality milestone.

---

## Phase 3 — Multimodal: Cross-Modal + Caption Baseline (benchmarked)

**Outcome:** Figure/image questions retrieve the right region via **two** paths, switchable and comparable.

### Task 3.1: CLIP embedder — extend `embedders.py`
- **Produces:** `embed_images(list[png]) -> np.ndarray` and `embed_query_clip(str) -> np.ndarray` via `clip-ViT-B-32` (shared space). Test: an image of a cat scores higher against "a cat" than "a bar chart".

### Task 3.2: Image index (cross-modal path) — extend `store.py`
- **Produces:** a second FAISS index over `figure` chunk images; `Index.cross_modal(query, k)` embeds the text query with CLIP and searches images directly. Test: query text retrieves the semantically matching figure chunk.

### Task 3.3: Caption baseline path — extend ingestion/index
- **Produces:** each `figure` chunk also gets `caption_text` = OCR of its region (Phase 1.2 reused, no new model — ponytail: dropped BLIP), embedded into the **text** index. `mode="caption_baseline"` retrieves figures via this text vector. Test: same query retrieves the same figure via the caption path.

### Task 3.4: Mode toggle end-to-end — extend `hybrid.py`, `answer.py`
- **Produces:** `retrieval_mode ∈ {hybrid, cross_modal, caption_baseline, dense}` selects the path; retrieved figure images are passed to the VLM via `generate(..., images=[...])`. Frontend gets a mode selector. Test: each mode returns figure citations for an image question.

**Phase 3 gate:** both retrieval architectures demonstrably work; ready to be measured in Phase 5.

---

## Phase 4 — The Trust Layer (the differentiators)

**Outcome:** Faithfulness firewall, calibrated refusal, deterministic table facts, visual bbox highlights.

### Task 4.1: Claim splitting + NLI faithfulness gate — `backend/app/verify/nli.py`
- **Produces:** `verify_claims(answer, retrieved) -> list[Claim]`. Split the answer into sentences; for each, run `nli-deberta-v3-base` (entailment) against the concatenated retrieved evidence; `supported = entailment_score >= faithfulness_threshold`. Attach the best-matching chunk's citation. Test: a sentence entailed by evidence → supported; an invented sentence → unsupported.

### Task 4.2: Wire verification + refusal into `/answer` — modify `answer.py`
- **Produces:** when `req.verified`, run `verify_claims`; if **no** claim is supported, override to `refused=True`. Response carries per-claim `supported`/`score`/`citations`. Test: an answer with all-unsupported claims flips to refused; a grounded answer keeps its claims marked supported.

### Task 4.3: Deterministic table answers — modify `answer.py`, use `tables.df_json`
- **Produces:** if the top retrieved chunk is a `table` and the question is numeric-aggregate (sum/avg/count/min/max over a column — simple keyword+column match), compute the answer from the DataFrame and return it with the table as citation, bypassing LLM arithmetic. Test: "total of the Amount column" over a known table returns the exact computed sum. `# ponytail: keyword-matched aggregates only; NL-to-pandas is a later upgrade if needed.`

### Task 4.4: Visual citation viewer — `frontend/components/CitationViewer.tsx`
- **Produces:** renders the cited page image (served by a backend `GET /page/{session}/{n}.png`) with absolute-positioned highlight boxes from `bbox`; unsupported claims render red with their score; refusal renders an explicit "Not supported by your documents" panel showing the weak evidence. Test (manual + a component render test): boxes land on the right regions; refusal state renders.

**Phase 4 gate:** the killer demo — out-of-corpus question refuses with proof; a table question is exactly right; claims show red/green with highlights.

---

## Phase 5 — Evaluation Harness (the resume centerpiece)

**Outcome:** Reproducible benchmark comparing retrieval architectures and the trust layer, surfaced in a dashboard.

### Task 5.1: Gold set — `backend/eval/gold/enterprise_docs.json`
- **Produces:** ~40 items `{question, answer, source_pages:[...], answerable:bool}` over a small fixed scanned-doc corpus (include ~8 deliberately out-of-corpus questions to score refusal). No code test; a schema-validation test ensures well-formedness.

### Task 5.2: Metrics — `backend/eval/metrics.py`
- **Produces:** `recall_at_k`, `mrr`, `citation_accuracy` (cited page ∈ source_pages), `faithfulness_rate` (NLI-supported claims / total), `refusal_accuracy` (correct refuse on unanswerable). Unit test each on a tiny hand-checked example.

### Task 5.3: Runner — `backend/eval/run_eval.py`
- **Produces:** ingests the corpus once, runs every gold question under each `retrieval_mode` and `verified ∈ {true,false}`, writes `eval/report.json` (metrics per configuration). Test: runner produces a report with all configurations populated on a 3-question smoke subset.

### Task 5.4: Dashboard — `frontend/app/eval/page.tsx`
- **Produces:** reads `report.json`, renders comparison tables/bar charts: cross-modal vs caption-baseline (recall@k, MRR), hybrid vs dense, verified vs raw (faithfulness, refusal accuracy). Follow the `dataviz` skill for the charts. Manual verify.

**Phase 5 gate:** real numbers that tell the "trust layer beats a raw LLM" story.

---

## Phase 6 — Deploy + Portfolio Polish

**Outcome:** Live backend Space + live Vercel frontend + recruiter-facing README.

### Task 6.1: Backend Dockerfile + HF Space
- **Files:** `backend/Dockerfile`, `backend/README.md` (`sdk: docker`, `app_port: 7860`)
- **Produces:** container that pre-downloads all models at build time (warm cold-starts), runs `uvicorn app.main:app --host 0.0.0.0 --port 7860`. Set `BACKEND_TOKEN` as a Space secret. Verify: `/health` returns ok on the public Space URL.

### Task 6.2: Frontend on Vercel
- **Produces:** Vercel project with env `BACKEND_URL` (the Space URL) + `BACKEND_TOKEN`. Preview deploy per PR, production on main. Verify: end-to-end ask on the live URL with a free BYOK key.

### Task 6.3: Top-level README + polish
- **Produces:** `README.md` with the thesis, architecture diagram, the four differentiators, benchmark table screenshot, live demo links, and a "how it works" section. Follow the `readme-writer-skill`. Add sample scanned docs so recruiters can try it instantly.

**Phase 6 gate:** two live URLs, a README that sells it, sample docs preloaded.

---

## Self-Review Notes

- **Spec coverage:** domain (scanned enterprise docs → Phase 1 OCR/tables), both retrieval architectures (Phase 3), BYOK 4-provider (Task 0.2), trust-layer differentiators (Phase 4), evaluation (Phase 5), Vercel + HF split deploy (Phase 6), free CPU / no-ZeroGPU / no-paid-infra (Global Constraints) — all mapped.
- **Deferred detail (intentional, not placeholder):** Phases 2–6 are specified at task + interface + test-target level; step-level TDD code (like Phase 0's) is generated per phase at execution time, because each phase's concrete signatures depend on what the prior phase actually produced on CPU. This avoids a stale mega-plan — a deliberate call, flagged here.
- **Type consistency:** `AnswerRequest/AnswerResponse/Claim/Citation` (Task 0.1) are the contract used unchanged through Phases 2–5; `Chunk`/`Table`/`Page` shapes are defined in Phase 1 and consumed as-is later.
