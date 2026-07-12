# Remediation plan (NEEDS-PLAN findings)

Not applied — reviewable task list only. Ordered by severity (HIGH → MED → LOW). Each task: exact file, exact change, verification.

## HIGH

### 1. Model singleton double-init race
**Files:** `backend/app/index/embedders.py`, `backend/app/index/rerank.py`, `backend/app/verify/nli.py`, `backend/app/ingest/ocr.py`, `backend/app/ingest/tables.py`, `backend/app/main.py`

**Change:** Add an `@app.on_event("startup")` (or FastAPI `lifespan`) hook in `main.py` that imports and calls each module's `_get_*()` once, sequentially, before the app starts serving:
```python
@app.on_event("startup")
def _warm_models():
    from app.index.embedders import _get_model, _get_clip_model
    from app.index.rerank import _get_model as _get_reranker
    from app.verify.nli import _get_model as _get_nli
    from app.ingest.ocr import _get_predictor
    from app.ingest.tables import _get_ocr
    _get_model(); _get_clip_model(); _get_reranker(); _get_nli(); _get_predictor(); _get_ocr()
```
This makes cold-start loading happen once, serially, before Cloud Run routes any traffic to the container (`concurrency=4` only matters once `/health` starts passing). Leaves the lazy singletons themselves unchanged (still safe for single-threaded reuse after warm-up). If a startup hook is rejected (e.g. slows liveness probe unacceptably), fall back to `threading.Lock` double-checked locking in each `_get_*`.

**Verify:** Add `backend/tests/test_startup_warm.py` that monkeypatches each module's loader to a call-counting stub, invokes the startup hook (or triggers app startup via `TestClient` context manager), and asserts each loader was called exactly once. For the concurrency claim specifically: a manual smoke test hitting `/answer` with 4 parallel requests immediately after container start, watching that memory doesn't spike past a single model's footprint x5.

### 2. `get_index` concurrent build race
**File:** `backend/app/session.py`

**Change:** Add a per-session lock. Store `_locks: dict[str, threading.Lock]` alongside `_sessions` (or a single `threading.Lock` guarding a lock-creation step), and wrap the check-and-build in `get_index`:
```python
import threading
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

def get_index(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        return None
    _sessions.move_to_end(session_id)
    with _locks_guard:
        lock = _locks.setdefault(session_id, threading.Lock())
    with lock:
        if "index" not in session:
            from app.index.store import Index
            idx = Index()
            idx.add(session["chunks"], pages=session["pages"])
            session["index"] = idx
    return session["index"]
```
Clean up `_locks[session_id]` when a session is evicted in `create_session`'s `popitem` path, to avoid unbounded growth.

**Verify:** A test that spawns 2 threads calling `get_index(same_session_id)` concurrently against a session whose `Index.add` is monkeypatched to a call-counting stub with a small `time.sleep` to widen the race window; assert `add` was called exactly once.

### 3. `/ingest` page-count amplification (DoS)
**Files:** `backend/app/main.py`, `backend/app/ingest/loader.py`

**Change:** In `main.py`'s `ingest()`, right after `pages = load_document(data)` (currently line ~52), add a page-count cap:
```python
MAX_PAGES = 100  # add to app/config.py as settings.max_pages instead of a local const
if len(pages) > MAX_PAGES:
    raise HTTPException(status_code=413, detail=f"document has too many pages (max {MAX_PAGES})")
```
Prefer adding `max_pages = 100` to `Settings` in `backend/app/config.py` alongside `max_upload_bytes`, and reference `settings.max_pages` here, matching the existing pattern.

**Verify:** `backend/tests/test_ingest_page_cap.py` — build a >100-page in-memory PDF (or monkeypatch `load_document` to return a 101-entry list) and POST to `/ingest`; assert 413 and that OCR/table extraction (mock and assert not called) never runs.

### 4. Redundant OCR
**Files:** `backend/app/index/store.py` (`Index.add`), `backend/app/main.py` (ingest loop)

**Change (store.py):** `Index.add`'s page loop (`store.py:87-88`) currently always calls `ocr_page(p["image_png"])`. Pages already OCR'd in `/ingest` (`needs_ocr=True` pages) have their words in `p["text_blocks"]`... but note `text_blocks` from `loader.py` is native-text-layer format (`{"text","bbox"}` per block, not per word) — check whether `ocr_page`'s word-level output is actually required downstream (caption index just joins `w["text"]` into a string, `store.py:89`) before reusing `text_blocks` directly. If page already has non-empty `text_blocks` (native text or already OCR'd), build `caption_text` by joining `text_blocks` text instead of re-running `ocr_page`; only call `ocr_page` for pages where `text_blocks` is still empty (shouldn't happen post-`/ingest`, but keep as a fallback for direct `Index.add` callers/tests).

**Change (main.py):** Gate `extract_tables(page["image_png"])` (`main.py:62`) so it isn't unconditionally run on every page. Options: (a) only run it on pages likely to contain a table (e.g. skip pages with `needs_ocr=False` and a high native-text-block count that suggests prose, tune with the eval set), or (b) accept the cost is inherent to table detection but at minimum stop double-OCRing by having `extract_tables` accept pre-computed OCR text/words when available instead of always invoking its own `DocTR` pass. (b) is the safer minimal fix; (a) needs eval validation.

**Verify:** Monkeypatch `ocr_page` (and `img2table`'s `DocTR.__call__` or `_get_ocr`) with call counters in a test that ingests a scanned page; assert `ocr_page` is called at most once per page across the whole `/ingest` → `create_session` → first `get_index` flow.

## MEDIUM

### 5. `/answer` request-size cap
**File:** `backend/app/schemas.py`

**Change:**
```python
from pydantic import BaseModel, Field

class AnswerRequest(BaseModel):
    question: str = Field(max_length=4000)
    provider: Literal["openai", "groq", "gemini", "anthropic"]
    model: str = Field(max_length=200)
    api_key: str = Field(max_length=500)
    ...
```
Pick limits generous enough for real use (a long multi-paragraph question, longest real model id, longest real provider key format) — check `test_answer.py`/`test_health.py` fixtures fit within them before committing to a number.

**Verify:** POST `/answer` with a `question` over the cap → 422; existing fixtures (`test_answer.py:60`, `test_health.py:14`) still pass unchanged.

### 6. Malformed provider response → 500
**File:** `backend/app/generate/providers.py`

**Change:** Wrap the response-shape access in `generate()` (`:58` and `:65`) in a `try/except (KeyError, IndexError, TypeError)` and raise `ProviderError(502, "provider returned an unexpected response")`:
```python
try:
    return r.json()["choices"][0]["message"]["content"]
except (KeyError, IndexError, TypeError) as exc:
    raise ProviderError(502, "provider returned an unexpected response") from None
```
(mirror for the anthropic branch). Also guard `r.json()` itself against non-JSON bodies the same way.

**Verify:** `backend/tests/test_providers.py` — add a test that monkeypatches `httpx.post` to return a 200 with body `{"unexpected": "shape"}`; assert `generate(...)` raises `ProviderError` with `status_code == 502` instead of an uncaught exception.

### 7. Hybrid grounding gate ignores lexical evidence
**File:** `backend/app/generate/answer.py`

**Change:** In `_grounding_top` (`:84-91`), for `mode == "hybrid"`, don't gate on `index.dense` alone. Options, in order of minimal-diff:
- (a) Gate on `max(dense_top1_score, normalized_bm25_top1_score)` — needs a BM25-score normalization since raw BM25 isn't 0-1 scaled (module docstring at `:10-14` explains why raw fusion/rerank scores aren't currently used for the gate). A simple normalization: divide by the max BM25 score across the session's own corpus, computed once at index-build time.
- (b) Gate on whether `retrieve()`'s top result (already RRF-fused across dense+BM25, reranked) exceeds a *separately calibrated* threshold in that score space, instead of trying to reuse `retrieval_min_score` (which is calibrated for 0-1 cosine). Requires a second threshold constant in `config.py` and eval-set tuning.
(a) is the smaller, more surgical change; (b) is architecturally cleaner but needs threshold recalibration against the eval harness (`backend/eval/run_eval.py`) before shipping either way.

**Verify:** A test/eval case with a short exact-keyword query (e.g. a distinctive proper noun or SKU present verbatim in a synthetic doc) where dense cosine alone would sit below `retrieval_min_score` but BM25 top-1 is a strong hit; assert `answer_question` no longer refuses. Also re-run `backend/eval/run_eval.py` and confirm `refusal_accuracy` for hybrid mode doesn't regress.

### 8. `_dedupe_columns` can re-collide
**File:** `backend/app/ingest/tables.py`

**Change:** Track the accumulated *output* set, not just original names:
```python
def _dedupe_columns(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    used: set[str] = set()
    deduped = []
    for name in names:
        candidate = name
        while candidate in used:
            seen[name] = seen.get(name, 0) + 1
            candidate = f"{name}.{seen[name]}"
        used.add(candidate)
        deduped.append(candidate)
    return deduped
```

**Verify:** Unit test in `backend/tests/test_tables.py` (new or existing): `_dedupe_columns(["A", "A", "A.1"]) == ["A", "A.1", "A.2"]`, plus the existing simple case `["A", "A"] == ["A", "A.1"]` still holds. Also feed the collision case through `_rows_to_dataframe` end-to-end and assert no exception.

### 9. Dropdown not keyboard-operable
**File:** `frontend/components/Dropdown.tsx`

**Change:** Add keyboard handling on the `<ul role="listbox">` (or track a `highlightedIndex` state): `ArrowDown`/`ArrowUp` move a roving highlight (`tabIndex={-1}` on each `<li>`, `tabIndex={0}` + `.focus()` on the highlighted one, or manage highlight via `aria-activedescendant` on the listbox instead of moving DOM focus), `Enter`/`Space` selects the highlighted option and closes, `Escape` closes without selecting (already handled on the trigger, extend to the list), `Home`/`End` jump to first/last. Standard combobox/listbox pattern (WAI-ARIA APG "Listbox").

**Verify:** Manual keyboard-only pass: Tab to trigger → Enter opens → Arrow keys move highlight → Enter selects → focus returns to trigger. Optionally a Playwright/Testing-Library test simulating `keyDown` events and asserting `onChange` fires with the expected value.

### 10. Proxy routes have no fetch timeout
**Files:** `frontend/app/api/answer/route.ts`, `frontend/app/api/eval-report/route.ts`, `frontend/app/api/ingest/route.ts`, `frontend/app/api/page/[session]/[page]/route.ts`

**Change:** Add `signal: AbortSignal.timeout(N)` to each `fetch(...)` call (pick `N` per route — `/answer` needs the longest budget since it includes NLI verification, e.g. 60000ms; `/page` and `/eval-report` can be short, e.g. 10000ms; `/ingest` needs a large budget for OCR, e.g. 120000ms). Catch the resulting `TimeoutError`/`DOMException` (name `"TimeoutError"`) alongside the existing generic `catch` and map it to a 504:
```ts
} catch (err) {
  if (err instanceof Error && err.name === "TimeoutError") {
    return Response.json({ error: "backend timed out" }, { status: 504 });
  }
  return Response.json({ error: "backend unavailable" }, { status: 502 });
}
```
(`page/[session]/[page]/route.ts` returns a raw `Response`, not `Response.json`, and already has its own catch — mirror the same pattern there.)

**Verify:** Point `BACKEND_URL` at a route/server that never responds (e.g. a local listener that accepts the connection and hangs) and confirm the proxy returns 504 within the configured timeout instead of hanging to the platform limit.

## LOW

### 11. Pin backend deps
**File:** `backend/requirements.txt`

**Change:** Either pin every line to the version currently installed in the working `.venv` (`pip freeze` filtered to direct deps), or generate a proper lockfile (`pip-compile` via `pip-tools`, or switch to `uv pip compile`). Keep the existing `torch`/`torchvision` pin scheme and comments intact.

**Verify:** Fresh `pip install -r requirements.txt` into a clean venv reproduces the same resolved versions as the current working environment (`pip freeze` diff is empty for direct deps); fast test suite still passes.

### 12. sessionStorage BYOK key is write-only
**File:** `frontend/app/page.tsx`

**Change:** Decide one of two directions and implement it:
- **Delete the write** (simplest — matches the "key never stored" messaging in the UI, `page.tsx:190,209`) — remove the `sessionStorage.setItem` call in `handleApiKeyChange` (`:104-108`) entirely, keep only `setApiKey(value)`.
- **Or make it real tab-persistence** — restore on mount via a lazy `useState` initializer: `useState(() => (typeof window !== "undefined" ? sessionStorage.getItem("byok_api_key") ?? "" : ""))` for the `apiKey` state (`:82`), keeping the existing write.

**Verify:** If deleting: confirm `sessionStorage` is never touched (grep the built bundle or DevTools Application tab shows no `byok_api_key` key after typing). If restoring: refresh the tab mid-session and confirm the API key field is repopulated; open a new tab and confirm it is NOT (sessionStorage is tab-scoped, not shared).

### 13. `/ingest` echoes raw parse exception
**File:** `backend/app/main.py`

**Change:** Log the exception server-side and return a generic client message:
```python
import logging
logger = logging.getLogger(__name__)
...
except Exception as exc:
    logger.exception("failed to parse uploaded document")
    raise HTTPException(status_code=400, detail="could not parse document") from exc
```

**Verify:** Existing ingest-failure tests (if any) still assert a 400; confirm the response body no longer contains the raw exception string while the server log does.

### 14. Refresh stale model IDs
**File:** `frontend/app/page.tsx` (`MODELS.anthropic`, `:37-41`)

**Change:** Add current-generation Claude model entries. Verify exact model ID strings against Anthropic's live model list (`https://docs.claude.com/en/docs/about-claude/models` or the `/v1/models` API) before committing — do not guess IDs.

**Verify:** Each added `value` round-trips a real (non-error) response from the Anthropic API with a valid key, confirmed manually or via a one-off smoke script.

### 15. Add `lib/bbox.ts` test
**File:** `frontend/lib/bbox.test.mts` (new)

**Change:** Add a `node --test` file covering `bboxToOverlayRect` (`frontend/lib/bbox.ts`): a basic scale case, an identity case (`natural == rendered`), and the `natural.width === 0` / `natural.height === 0` guard (asserts `scaleX`/`scaleY` become `0` rather than throwing/`Infinity`/`NaN`).
```ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { bboxToOverlayRect } from "./bbox.ts";

test("scales bbox by rendered/natural ratio", () => {
  const rect = bboxToOverlayRect([10, 20, 30, 40], { width: 100, height: 200 }, { width: 200, height: 400 });
  assert.deepEqual(rect, { left: 20, top: 40, width: 40, height: 40 });
});

test("guards divide-by-zero when natural size is 0", () => {
  const rect = bboxToOverlayRect([10, 20, 30, 40], { width: 0, height: 0 }, { width: 200, height: 400 });
  assert.deepEqual(rect, { left: 0, top: 0, width: 0, height: 0 });
});
```

**Verify:** `npm test` (`node --test lib/**/*.test.mts`) picks up and passes the new file.
