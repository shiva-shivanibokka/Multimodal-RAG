# Benchmark runbook

`backend/eval/run_eval.py` produces `backend/eval/report.json` — the file
the `/eval` dashboard reads (via the backend's `GET /eval/report`, with a
labeled sample-data fallback if it's missing). This doc is how to actually
generate it.

## 1. Get the eval corpus

The gold set (`backend/eval/gold/enterprise_docs.json`) is committed and
schema-tested, but the underlying page images
(`backend/eval/corpus/*.png`) are **gitignored** — DocVQA is research-use
licensed, so the repo doesn't redistribute them. Fetch them once:

```bash
cd backend
pip install datasets   # not a base dependency; only needed for this fetch
python scripts/fetch_sample_docs.py --n 40 --out eval/corpus
```

This streams 40 unique pages from `lmms-lab/DocVQA` (HuggingFace, no
auth required) and writes `eval/corpus/*.png` + `eval/corpus/manifest.json`.
It's deterministic (stable `docId`s), so re-running reproduces the same 40
pages. See `backend/eval/corpus/README.md` for details.

## 2. Run the benchmark

```bash
cd backend
python eval/run_eval.py --out eval/report.json
```

**Windows / OneDrive note:** if this repo lives under a long OneDrive path
(as it does by default — `C:\Users\...\OneDrive\Desktop\...`), point the
Hugging Face / torch caches at a short local path instead of letting them
default into your profile dir, to dodge Windows `MAX_PATH` issues with the
deeply-nested cache filenames these libraries create:

```bash
HF_HOME=C:/mrag/.cache TORCH_HOME=C:/mrag/.cache python eval/run_eval.py --out eval/report.json
```

(Same convention already used by `backend/tests/test_nli.py` and
`test_rerank.py` — reuse the same cache dir so models aren't downloaded
twice.)

### What this does

- Ingests all 40 corpus pages into one combined in-memory session (OCR →
  table extraction → chunking — the same pipeline `/ingest` runs).
- Runs every gold-set question through all four retrieval modes (`dense`,
  `hybrid`, `cross_modal`, `caption_baseline`) and computes recall@1,
  recall@5, MRR, citation accuracy, and refusal accuracy per mode.
- **No API key required** for any of the above — it's pure ingestion +
  retrieval, never calls an LLM.

### How long it takes

Expect roughly **10–20 minutes** on a typical CPU, dominated by OCR and
table extraction over 40 scanned pages, not the retrieval scoring itself
(which is fast once embeddings exist). One known inefficiency: table
extraction (`img2table`) runs its own internal OCR pass independent of the
page-level docTR OCR, so a page with a table gets OCR'd twice — this is a
real redundancy in the current pipeline, not a benchmark artifact, and
would be the first thing to optimize if this needed to run faster
(`# ponytail: redundant OCR pass on table pages; a shared OCR result would
cut ingestion time` — see `backend/app/ingest/tables.py`).

Use `--limit N` to smoke-test the runner on a handful of questions instead
of the full gold set before committing to a full run:

```bash
python eval/run_eval.py --out eval/report.smoke.json --limit 5
```

## 3. Optional: the faithfulness path

Add `--api-key` (plus `--provider`/`--model`) to also run real generation
+ NLI verification through the exact same `answer_question` code path the
`/answer` endpoint uses, over every *answerable* gold question in one
retrieval mode:

```bash
python eval/run_eval.py --out eval/report.json \
  --api-key "$GROQ_API_KEY" --provider groq --model llama-3.1-8b-instant \
  --faithfulness-mode hybrid
```

This is the only part of the benchmark that costs anything (a free-tier
Groq or Gemini key is enough) and is the only part that calls an external
API — everything in step 2 stays fully local. Without `--api-key`, the
report's `faithfulness` field is `null` and the dashboard shows an honest
"run with `--api-key` to populate" note instead of a number.

## 4. Commit the report

`backend/eval/report.json` is **not** gitignored — it's a committed
artifact, not scratch output (only `backend/eval/report.smoke.json`, the
`--limit`-based smoke run, is ignored). After a real run:

```bash
git add backend/eval/report.json
git commit -m "chore(eval): update benchmark report"
```

so the deployed `/eval` dashboard (both local and on Vercel, once the
backend Space serves it via `GET /eval/report`) shows real measured
numbers instead of the illustrative sample fixture
(`frontend/app/eval/report.sample.json`).
