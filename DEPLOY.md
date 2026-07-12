# Deploy runbook

Two independent deploys, connected over HTTPS:

- **Backend** — FastAPI + local ML, on **Google Cloud Run** (free tier).
- **Frontend** — Next.js, on **Vercel**.

Nothing is shared except two env vars (`BACKEND_URL`, `BACKEND_TOKEN`). Deploy
the backend first — the frontend needs its URL.

**Why Cloud Run and not a Hugging Face Space:** HF's free Spaces tier is now
static-only — running a Docker/Gradio Space (which this backend needs) requires
an HF Pro subscription (~$9/mo). Google Cloud Run's free tier covers this
project's needs (scale-to-zero, pay only for active request time) at zero
ongoing cost, so that's what's actually deployed. `backend/README.md` (the HF
Space card) is kept in the repo as an optional alternative if you do have HF
Pro, but it is not the primary path below.

---

## Part A — Backend on Google Cloud Run

No local Docker install is required — Cloud Build builds the image in the
cloud from `backend/`.

### 1. Enable the required APIs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 2. Create an Artifact Registry repo (one-time)

```bash
gcloud artifacts repositories create mrag --repository-format=docker --location=us-central1
```

### 3. Build the image with Cloud Build

```bash
gcloud builds submit backend --tag us-central1-docker.pkg.dev/PROJECT/mrag/backend:v2 --machine-type=e2-highcpu-8 --timeout=2400
```

Replace `PROJECT` with your GCP project id. The Dockerfile bakes all four ML
models (bge text embedder, CLIP, the bge reranker, the NLI model) plus docTR's
OCR weights into the image at **build time**, so the first real request
doesn't pay a multi-gigabyte download. That makes the build itself slow —
expect roughly **8–15 minutes** on `e2-highcpu-8` while it installs CPU-only
PyTorch/torchvision (pinned to 0.28.0) and downloads every model. The
`--timeout=2400` gives it headroom.

### 4. Deploy to Cloud Run

```bash
gcloud run deploy multimodal-rag-backend \
  --image us-central1-docker.pkg.dev/PROJECT/mrag/backend:v2 \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 3 \
  --concurrency 4 \
  --set-env-vars BACKEND_TOKEN=<secret>
```

- `PROJECT` — your GCP project id (same as step 3).
- `<secret>` — any long random string you generate yourself (e.g.
  `openssl rand -hex 32`). This is **not** an LLM provider key — it's a
  shared secret that gates every backend endpoint so the service can't be
  used as an open proxy by strangers who find the URL. Keep it handy — the
  frontend needs the exact same string in Part B.
- `--allow-unauthenticated` makes the HTTP endpoint public; auth is handled
  at the application layer by `BACKEND_TOKEN`, not Cloud Run IAM.
- 8Gi memory / 2 CPU is needed to hold four CPU transformer models plus OCR
  in memory at once; `--concurrency 4` and `--max-instances 3` keep the
  free-tier request/CPU budget in check.
- The container honors Cloud Run's injected `$PORT` (8080), falling back to
  7860 for local dev — no code change needed between environments.

### 5. Cold starts (scale-to-zero)

Free-tier Cloud Run scales to zero when idle, so the **first request after a
period of inactivity cold-starts** — expect roughly **60–90 seconds** while
the container starts and lazy-loads models into memory before it can answer.
Subsequent requests are fast until the instance idles out again. Sessions are
also held in-memory (`app/session.py`), so a cold start (or any instance
restart) drops any documents you'd ingested — re-upload after one.

### 6. Get the public URL and verify

`gcloud run deploy` prints the service URL on success, e.g.:

```
https://multimodal-rag-backend-1061434430143.us-central1.run.app
```

Verify it's live:

```bash
curl https://multimodal-rag-backend-1061434430143.us-central1.run.app/health
# -> {"status":"ok"}
```

`/health` is intentionally unauthenticated; every other endpoint requires
`Authorization: Bearer <BACKEND_TOKEN>`.

To redeploy after backend changes, re-run steps 3–4 with a new image tag.

---

## Part B — Frontend on Vercel

This is a standard Next.js App Router app — it needs **zero** Vercel
config beyond pointing at the right subdirectory and setting two env vars.
No `vercel.json` is required or included; adding one would just be
config for values that never change per-environment.

1. [vercel.com/new](https://vercel.com/new) → import this GitHub repo, or
   link the `frontend/` directory locally with `vercel link`.
2. **Root Directory**: set to `frontend` in the Vercel project settings (the
   repo root is the monorepo, not the Next.js app — this is the one setting
   that isn't automatic, and it's required for `git push`-triggered deploys
   to work at all).
3. Framework preset: Vercel auto-detects **Next.js** once the root
   directory is set — leave build/output settings on their defaults.
4. **Environment Variables** — add both, for the Production (and Preview,
   if you want preview deploys to also talk to the backend) environment:

   | Name            | Value                                              |
   |-----------------|-----------------------------------------------------|
   | `BACKEND_URL`   | your Cloud Run service URL, e.g. `https://multimodal-rag-backend-1061434430143.us-central1.run.app` |
   | `BACKEND_TOKEN` | the exact same string you set with `--set-env-vars BACKEND_TOKEN=...` in Part A.4 |

   Both are server-only — they're read in Next.js route handlers
   (`frontend/app/api/*/route.ts`) and never sent to the browser. See
   `frontend/.env.example` for the same two names used in local dev.
5. Deploy: with Root Directory set, either push to the connected branch
   (git auto-deploy) or run it directly:

   ```bash
   cd frontend
   vercel deploy --prod
   ```

   Either path gives you a `*.vercel.app` production URL, e.g.
   `https://multimodal-rag-plum.vercel.app`.

---

## Part C — Connect + smoke test

1. Confirm both env vars are set on the Vercel project (Settings →
   Environment Variables) and match the Cloud Run `BACKEND_TOKEN` exactly —
   a mismatch here is the most common cause of every backend call returning
   401.
2. If you changed either value after the first deploy, trigger a redeploy
   (Vercel → Deployments → ⋯ → Redeploy) — env var changes don't apply to
   already-built deployments.
3. Open the Vercel URL (e.g.
   [multimodal-rag-plum.vercel.app](https://multimodal-rag-plum.vercel.app)).
   Upload a small sample PDF or scanned image (the upload panel on `/`).
4. Paste a free API key from [Groq](https://console.groq.com/keys) or
   [Google AI Studio (Gemini)](https://aistudio.google.com/apikey) into
   the settings panel — it's held in `sessionStorage` in your browser only,
   forwarded per-request, never stored by either tier.
5. Ask a question about the document you uploaded. You should get a cited
   answer (or an honest refusal, with the weak evidence shown, if the
   question isn't actually answerable from the document — that's the
   trust layer working as intended, not a bug). Verified end-to-end: the
   frontend's `/api/answer` proxy reaching the Cloud Run backend returns a
   clean refusal with no API key supplied, since the refusal check happens
   before any LLM call is made.
6. If the backend has scaled to zero, the first call above will take
   roughly 60–90 seconds (cold start) before responding — this is expected
   on the free tier, not a failure.
7. Visit `/eval` to see the benchmark dashboard. Until you run the
   benchmark (see `BENCHMARK.md`) and commit `backend/eval/report.json`,
   it shows clearly-labeled illustrative sample data, not real numbers.

No secrets are stored in this repo. `BACKEND_TOKEN` and every provider API
key live only in the Cloud Run service's env vars / Vercel env vars / the
browser's `sessionStorage`, respectively.
