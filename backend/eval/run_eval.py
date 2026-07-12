#!/usr/bin/env python
# backend/eval/run_eval.py
"""Task 5.3: benchmark runner -- ingests the whole DocVQA eval corpus into
ONE combined session, runs every gold question under each retrieval mode,
computes metrics with eval/metrics.py, and writes eval/report.json.

No API key is required for the core report: steps 1-4 below are pure
ingestion + retrieval (Phases 0-3), never call an LLM. The optional
faithfulness path (guarded by --api-key) additionally runs real generation
+ NLI verification via app.generate.answer.answer_question, reusing the
exact same production code path the /answer route uses.

Combined-session page remapping: every corpus doc is a single scanned page
(see backend/eval/corpus/manifest.json). ``ingest_corpus`` assigns each doc
a PAGE INDEX in the combined session (0..n_docs-1, one per doc, in sorted
filename order) and returns a ``doc_to_page`` map. The gold set's
``source_pages: [0]`` is per-doc (page 0 of that doc's own single-page PDF);
in the combined session, the doc's true page is ``doc_to_page[source_doc]``,
computed in ``evaluate_mode`` below -- this is the remap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `app.*` and `eval.*` importable both when run directly
# (`python eval/run_eval.py`, no package context) and when imported by
# pytest (pytest.ini already puts backend/ on sys.path, so this is a no-op).
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.ingest.chunk import chunk_pages  # noqa: E402
from app.ingest.loader import load_document  # noqa: E402
from app.ingest.ocr import ocr_page  # noqa: E402
from app.ingest.tables import extract_tables  # noqa: E402
from app.retrieve.hybrid import retrieve  # noqa: E402
from app.session import create_session, get_index  # noqa: E402
from eval import metrics  # noqa: E402

MODES = ["dense", "hybrid", "cross_modal", "caption_baseline"]
_RERANK_MODES = {"dense", "hybrid"}  # figure modes never rerank, see hybrid.py


def ingest_corpus(corpus_dir: Path) -> tuple[str, dict[str, int]]:
    """Ingest every ``*.png`` in ``corpus_dir`` into ONE combined session,
    reusing the exact load -> OCR(needs_ocr) -> tables -> chunk ->
    create_session orchestration main.py's /ingest route runs per-document,
    but accumulating pages/chunks/tables across the whole corpus so page
    indices are unique session-wide (0..n_docs-1).

    Returns ``(session_id, doc_to_page)`` where ``doc_to_page`` maps each
    image filename (e.g. "14465.png", matching gold's ``source_doc``) to
    its combined-session page index.
    """
    files = sorted(corpus_dir.glob("*.png"))
    all_pages: list[dict] = []
    tables_by_page: dict[int, list[dict]] = {}
    doc_to_page: dict[str, int] = {}

    for combined_index, f in enumerate(files):
        pages = load_document(f.read_bytes())
        page = pages[0]  # each corpus doc is a single scanned page
        page["index"] = combined_index
        if page["needs_ocr"]:
            page["text_blocks"] = ocr_page(page["image_png"])
        tables = extract_tables(page["image_png"])
        if tables:
            tables_by_page[combined_index] = tables
        all_pages.append(page)
        doc_to_page[f.name] = combined_index

    chunks = chunk_pages(all_pages, tables_by_page)
    session_id = create_session(all_pages, chunks)
    return session_id, doc_to_page


def _dedup_pages(results: list[dict]) -> list[int]:
    """Ordered, de-duplicated list of page numbers from retrieve() results."""
    pages: list[int] = []
    seen: set[int] = set()
    for r in results:
        p = r["chunk"]["page"]
        if p not in seen:
            seen.add(p)
            pages.append(p)
    return pages


def _grounding_score(index, mode: str, question: str) -> float:
    """Delegate to the exact runtime gate so the eval's refusal metric always
    matches /answer behavior (incl. Task 7's hybrid max(dense, normalized_bm25)
    rule). Do NOT re-implement the gate here -- that caused eval/runtime drift."""
    from app.generate.answer import _grounding_score as _runtime_grounding_score

    return _runtime_grounding_score(index, mode, question)


def evaluate_mode(index, gold_items: list[dict], doc_to_page: dict[str, int], mode: str) -> list[dict]:
    """Run every gold item's question through ``mode`` and collect one
    per-item record. No aggregation here -- see ``aggregate``."""
    records = []
    for item in gold_items:
        source_pages: list[int] = []
        if item["answerable"] and item["source_doc"] in doc_to_page:
            source_pages = [doc_to_page[item["source_doc"]]]

        results = retrieve(index, item["question"], mode=mode, k=5, use_rerank=mode in _RERANK_MODES)
        retrieved_pages = _dedup_pages(results)
        cited_pages = [retrieved_pages[0]] if retrieved_pages else []
        score = _grounding_score(index, mode, item["question"])

        records.append(
            {
                "id": item["id"],
                "retrieved_pages": retrieved_pages,
                "cited_pages": cited_pages,
                "refused": score < settings.retrieval_min_score,
                "source_pages": source_pages,
                "answerable": item["answerable"],
            }
        )
    return records


def aggregate(records: list[dict]) -> dict:
    """Pure per-mode aggregation over ``evaluate_mode``'s records -- no
    models, no I/O. Recall/MRR/citation_accuracy are computed over
    ``source_pages`` (empty for unanswerable items, so metrics.py's own
    skip-if-no-sources rule naturally restricts them to answerable items);
    refusal_accuracy uses every record. Unit-tested directly in
    tests/test_run_eval.py with synthetic records."""
    retrieved = [r["retrieved_pages"] for r in records]
    cited = [r["cited_pages"] for r in records]
    sources = [r["source_pages"] for r in records]
    refused = [r["refused"] for r in records]
    answerable = [r["answerable"] for r in records]
    return {
        "recall_at_1": metrics.recall_at_k(retrieved, sources, 1),
        "recall_at_5": metrics.recall_at_k(retrieved, sources, 5),
        "mrr": metrics.mrr(retrieved, sources),
        "citation_accuracy": metrics.citation_accuracy(cited, sources),
        "refusal_accuracy": metrics.refusal_accuracy(refused, answerable),
    }


def run_faithfulness(session_id: str, gold_items: list[dict], provider: str, model: str, api_key: str, mode: str) -> dict:
    """Real generation + NLI faithfulness path -- only runs when the caller
    supplied an API key. Reuses answer_question (the production /answer
    handler) unchanged so this measures the same code path users hit."""
    from app.generate.answer import answer_question
    from app.schemas import AnswerRequest

    answerable_items = [g for g in gold_items if g["answerable"]]
    supported_flags: list[list[bool]] = []
    refused_flags: list[bool] = []
    for item in answerable_items:
        req = AnswerRequest(
            question=item["question"],
            provider=provider,
            model=model,
            api_key=api_key,
            session_id=session_id,
            retrieval_mode=mode,
            verified=True,
        )
        resp = answer_question(req)
        supported_flags.append([c.supported for c in resp.claims])
        refused_flags.append(resp.refused)

    return {
        "mode": mode,
        "n_items": len(answerable_items),
        "faithfulness_rate": metrics.faithfulness_rate(supported_flags),
        "generation_refusal_accuracy": metrics.refusal_accuracy(refused_flags, [True] * len(refused_flags)),
    }


def _print_summary(report: dict) -> None:
    print(
        f"\nDataset: {report['dataset']}  docs={report['n_docs']}  "
        f"answerable={report['n_answerable']}  ood={report['n_ood']}"
    )
    header = f"{'mode':<18}{'recall@1':>10}{'recall@5':>10}{'mrr':>8}{'citation':>10}{'refusal':>10}"
    print(header)
    print("-" * len(header))
    for mode, m in report["modes"].items():
        print(
            f"{mode:<18}{m['recall_at_1']:>10.3f}{m['recall_at_5']:>10.3f}"
            f"{m['mrr']:>8.3f}{m['citation_accuracy']:>10.3f}{m['refusal_accuracy']:>10.3f}"
        )
    if report.get("faithfulness"):
        f = report["faithfulness"]
        print(
            f"\nFaithfulness (mode={f['mode']}, n={f['n_items']}): "
            f"rate={f['faithfulness_rate']:.3f} "
            f"generation_refusal_accuracy={f['generation_refusal_accuracy']:.3f}"
        )
    elif report.get("note"):
        print(f"\n{report['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocVQA retrieval + trust-layer eval runner (Task 5.3)")
    parser.add_argument("--corpus", default=str(Path(__file__).parent / "corpus"), help="directory of page PNGs")
    parser.add_argument("--gold", default=str(Path(__file__).parent / "gold" / "enterprise_docs.json"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "report.json"))
    parser.add_argument("--limit", type=int, default=None, help="cap number of gold items evaluated (smoke testing)")
    parser.add_argument("--timestamp", default=None, help="optional value for report.generated_at; omit for a static/no-clock report")
    parser.add_argument("--api-key", default=None, help="BYOK provider key; enables the optional faithfulness path")
    parser.add_argument("--provider", default="openai", help="openai | groq | gemini | anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--faithfulness-mode", default="hybrid", choices=MODES)
    args = parser.parse_args()

    gold_items = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.limit is not None:
        gold_items = gold_items[: args.limit]

    print(f"Ingesting corpus from {args.corpus} into one combined session...")
    session_id, doc_to_page = ingest_corpus(Path(args.corpus))
    index = get_index(session_id)
    print(f"Ingested {len(doc_to_page)} docs into session {session_id}; evaluating {len(gold_items)} gold items.")

    n_answerable = sum(1 for g in gold_items if g["answerable"])
    n_ood = len(gold_items) - n_answerable

    mode_metrics = {}
    for mode in MODES:
        records = evaluate_mode(index, gold_items, doc_to_page, mode)
        mode_metrics[mode] = aggregate(records)

    report = {
        "dataset": "DocVQA (lmms-lab)",
        "n_docs": len(doc_to_page),
        "n_answerable": n_answerable,
        "n_ood": n_ood,
        "generated_at": args.timestamp,
        "modes": mode_metrics,
    }

    if args.api_key:
        print(f"Running faithfulness path (mode={args.faithfulness_mode}, provider={args.provider})...")
        report["faithfulness"] = run_faithfulness(
            session_id, gold_items, args.provider, args.model, args.api_key, args.faithfulness_mode
        )
    else:
        report["faithfulness"] = None
        report["note"] = "faithfulness requires a key -- run with --api-key to populate"

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
