# Deploy runbook

Two independent deploys, connected over HTTPS:

- **Backend** — FastAPI + local ML, on a free-CPU Hugging Face **Docker Space**.
- **Frontend** — Next.js, on **Vercel**.

Nothing is shared except two env vars (`BACKEND_URL`, `BACKEND_TOKEN`). Deploy
the backend first — the frontend needs its URL.

---

## Part A — Backend on Hugging Face Spaces

### 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Pick an owner + name (e.g. `multimodal-rag-trust-layer`).
3. **SDK: Docker**, template "Blank". Visibility: your choice (Public is
   fine — no secrets live in the repo).
4. Hardware: the default free **CPU basic** tier (2 vCPU / 16GB RAM). No
   GPU, no ZeroGPU — the whole point of this project is that it runs
   without one.

### 2. Push `backend/` to the Space

The Space's git repo needs the *contents* of `backend/` at its root (it
looks for `Dockerfile` and `README.md` at the top level), not the whole
monorepo. Cleanest way from this repo, without restructuring it: push the
`backend/` subtree as a separate ref using `git subtree`.

```bash
# one-time: add the Space as a remote
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>

# push the backend/ subdirectory as the Space's root
git subtree push --prefix backend space main
```

(If the Space's default branch isn't `main`, push to whatever it is —
check the Space's "Files" tab.) Re-run the same `git subtree push` command
after future backend changes to redeploy.

Alternative if you'd rather not deal with `git subtree`: clone the empty
Space repo separately, copy `backend/`'s contents into it by hand, commit,
and push. Slightly more manual but harder to get wrong.

### 3. Set the `BACKEND_TOKEN` secret

In the Space: **Settings → Variables and secrets → New secret**.

- Name: `BACKEND_TOKEN`
- Value: any long random string you generate yourself (e.g.
  `openssl rand -hex 32`). This is **not** an LLM provider key — it's a
  shared secret that gates every backend endpoint so the Space can't be
  used as an open proxy by strangers who find the URL.

Keep this value handy — the frontend needs the exact same string in
Part B.

### 4. Wait for the build

The Dockerfile bakes all four ML models (bge text embedder, CLIP, the
bge reranker, the NLI model) plus docTR's OCR weights into the image at
**build time**, so the first real request doesn't pay a multi-gigabyte
download. That means the **first build is slow** — expect 15–30+ minutes
on the free tier while it installs CPU-only PyTorch and downloads every
model. Watch progress under the Space's "Logs" tab. Subsequent pushes are
faster (Docker layer caching) unless `requirements.txt` changed.

Free-CPU reality check: no GPU means inference (OCR, embedding, NLI,
generation-adjacent work) is slower than a typical hosted demo — expect an
`/ingest` or `/answer` call to take several seconds to tens of seconds
depending on document size, not sub-second. Sessions are also held
in-memory (`app/session.py`), so a Space restart (e.g. after a rebuild, or
the free tier's idle-sleep) drops any documents you'd ingested — re-upload
after a cold start.

### 5. Get the public URL and verify

Once the Space shows "Running", its public URL is:

```
https://<your-username>-<space-name>.hf.space
```

(shown at the top of the Space page). Verify it's live:

```bash
curl https://<your-username>-<space-name>.hf.space/health
# -> {"status":"ok"}
```

`/health` is intentionally unauthenticated; every other endpoint requires
`Authorization: Bearer <BACKEND_TOKEN>`.

---

## Part B — Frontend on Vercel

This is a standard Next.js App Router app — it needs **zero** Vercel
config beyond pointing at the right subdirectory and setting two env vars.
No `vercel.json` is required or included; adding one would just be
config for values that never change per-environment.

1. [vercel.com/new](https://vercel.com/new) → import this GitHub repo.
2. **Root Directory**: set to `frontend` (the repo root is the monorepo,
   not the Next.js app — this is the one setting that isn't automatic).
3. Framework preset: Vercel auto-detects **Next.js** once the root
   directory is set — leave build/output settings on their defaults.
4. **Environment Variables** — add both, for the Production (and Preview,
   if you want preview deploys to also talk to the backend) environment:

   | Name            | Value                                              |
   |-----------------|-----------------------------------------------------|
   | `BACKEND_URL`   | your Space's URL, e.g. `https://<user>-<space>.hf.space` |
   | `BACKEND_TOKEN` | the exact same string you set as the Space secret in Part A.3 |

   Both are server-only — they're read in Next.js route handlers
   (`frontend/app/api/*/route.ts`) and never sent to the browser. See
   `frontend/.env.example` for the same two names used in local dev.
5. Click **Deploy**. Vercel builds and gives you a `*.vercel.app` URL.

---

## Part C — Connect + smoke test

1. Confirm both env vars are set on the Vercel project (Settings →
   Environment Variables) and match the Space secret exactly — a mismatch
   here is the most common cause of every backend call returning 401.
2. If you changed either value after the first deploy, trigger a redeploy
   (Vercel → Deployments → ⋯ → Redeploy) — env var changes don't apply to
   already-built deployments.
3. Open the Vercel URL. Upload a small sample PDF or scanned image (the
   upload panel on `/`).
4. Paste a free API key from [Groq](https://console.groq.com/keys) or
   [Google AI Studio (Gemini)](https://aistudio.google.com/apikey) into
   the settings panel — it's held in `sessionStorage` in your browser only,
   forwarded per-request, never stored by either tier.
5. Ask a question about the document you uploaded. You should get a cited
   answer (or an honest refusal, with the weak evidence shown, if the
   question isn't actually answerable from the document — that's the
   trust layer working as intended, not a bug).
6. Visit `/eval` to see the benchmark dashboard. Until you run the
   benchmark (see `BENCHMARK.md`) and commit `backend/eval/report.json`,
   it shows clearly-labeled illustrative sample data, not real numbers.

No secrets are stored in this repo. `BACKEND_TOKEN` and every provider API
key live only in Hugging Face Space secrets / Vercel env vars / the
browser's `sessionStorage`, respectively.
