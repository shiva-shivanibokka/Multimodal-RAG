---
title: Multimodal RAG Trust Layer
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Multimodal RAG — Trust Layer Backend

> **Primary deployment is Google Cloud Run** — see [`../DEPLOY.md`](../DEPLOY.md)
> for the actual runbook. The Hugging Face Space card below is retained only
> as an optional alternative deploy target for anyone with an HF Pro
> subscription (HF's free tier is now static-only and can't run a Docker
> Space); it is not what's live today.

Headless FastAPI + local-ML backend for the Multimodal RAG "Trust Layer"
project: OCR/layout/table ingestion, hybrid + cross-modal retrieval, an NLI
faithfulness gate, and BYOK generation. Designed to run on a free CPU tier —
no GPU, no paid infra required. See the top-level repo README for the full
project writeup and architecture.

Returns JSON only (no UI here). The companion Next.js frontend is deployed
separately on Vercel.

## Secrets (Space Settings → Repository secrets)

- **`BACKEND_TOKEN`** — a shared bearer token gating every endpoint
  (`Authorization: Bearer <token>`) so this Space isn't an open proxy. Set it
  in the Space's **Settings → Variables and secrets**, never commit it. The
  frontend sends the same value from its own server-side env (see below) —
  it is never exposed to the browser.

No other secrets are required by this Space. In particular, LLM provider API
keys are **not** a Space secret — see BYOK below.

## BYOK (bring your own key)

Generation (the `/answer` endpoint's LLM call) is BYOK: the caller supplies
`provider` + `model` + `api_key` (OpenAI, Groq, Gemini, or Anthropic) in the
request body. That key is forwarded to the provider for that single request
and is **never persisted** — not to disk, not to a database, not logged.
Ingestion, retrieval, and verification (OCR, embeddings, reranking, the NLI
faithfulness gate) are 100% local to this Space and need no key at all.

## Free-CPU constraints

- 2 vCPU / 16GB RAM, no GPU, no ZeroGPU. Every model (bge-small text embed,
  clip-ViT-B-32, bge-reranker-base, nli-deberta-v3-base, docTR OCR) runs on
  CPU via the `sentence-transformers` CPU wheel of `torch`.
- All four models + OCR weights are downloaded and baked into the Docker
  image at **build time** (see `Dockerfile`), so the first request after a
  cold start doesn't pay a multi-GB download — only local CPU inference
  latency, which is still noticeably slower than a GPU deploy.
- Sessions (ingested documents + their FAISS indexes) are held in-process in
  memory (`app/session.py`) and do not survive a Space restart.

## Connecting the Vercel frontend

The frontend never calls this Space directly from the browser — it proxies
through its own Next.js API routes so `BACKEND_TOKEN` stays server-side.
Configure on the Vercel project:

- `BACKEND_URL` — this Space's public URL, e.g.
  `https://<user>-<space-name>.hf.space`
- `BACKEND_TOKEN` — the same value set as this Space's secret above

The BYOK provider key entered in the frontend UI is held client-side
(`sessionStorage`) and sent only in the body of each `/answer` request; it
passes through the Vercel proxy but is never stored by either tier.

## Endpoints

`GET /health` · `POST /ingest` (multipart file) · `POST /answer` ·
`GET /page/{session_id}/{page_index}` · `GET /eval/report` — all except
`/health` require the `Authorization: Bearer <BACKEND_TOKEN>` header.
