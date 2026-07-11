#!/usr/bin/env python
"""Fetch N sample DocVQA pages into backend/eval/corpus/ for the Phase 1
integration/demo corpus and the Phase 5 eval gold set.

STANDALONE utility: not imported by the app, not run by pytest (network +
a large-ish dependency, `datasets`, don't belong in the hermetic test suite).

Source: HuggingFace `datasets`, `lmms-lab/DocVQA` (config "DocVQA", split
"validation") -- an openly-downloadable re-upload of the DocVQA benchmark,
no auth/token required. Validated against the real dataset (2026-07-11);
actual example fields are:
    questionId (str), question (str), question_types (list[str]),
    image (PIL.Image), docId (int), ucsf_document_id (str),
    ucsf_document_page_no (str), answers (list[str]), data_split (str)
https://huggingface.co/datasets/lmms-lab/DocVQA

The validation split is ~6 parquet shards with embedded images (multi-GB
total) -- `load_dataset(..., split="validation[:n]")` would still need to
resolve/download whole shards before slicing. Instead this uses
`streaming=True`, which reads the remote parquet via HTTP range requests
row-by-row and never materializes the full split on disk, so the download
is naturally bounded by how many rows we actually iterate.

Multiple questions share the same page (docId), so reaching `n` *unique*
pages requires scanning more than `n` questions -- all Q/A pairs seen along
the way are kept in the manifest (extra grounded questions for free), and
the stream is cut as soon as `n` unique pages are collected.

Usage:
    pip install datasets   # not in the base test deps; only needed to run this
    HF_HOME=... HF_DATASETS_CACHE=...  # keep the HF cache off OneDrive
    python scripts/fetch_sample_docs.py --n 40 --out eval/corpus

Saves into <out>/:
    <doc_id>.png     -- one image per unique document page (<= n files)
    manifest.json    -- list of {"question", "answers", "doc_id", "image_file"}
"""
import argparse
import json
from pathlib import Path

DATASET_NAME = "lmms-lab/DocVQA"
DATASET_CONFIG = "DocVQA"
DATASET_SPLIT = "validation"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=10, help="number of unique document pages to fetch (default: 10)")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "eval" / "corpus"),
        help="output directory (default: backend/eval/corpus)",
    )
    return parser.parse_args(argv)


def fetch(n: int, out_dir: str) -> None:
    from datasets import load_dataset  # deferred: keep `--help` usable without the dep installed

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)

    manifest = []
    saved_pages: set[str] = set()
    for example in ds:
        doc_id = str(example["docId"])
        if doc_id not in saved_pages:
            if len(saved_pages) >= n:
                break  # hit the page cap -- stop pulling from the stream
            example["image"].save(out_path / f"{doc_id}.png")
            saved_pages.add(doc_id)
            print(f"saved page {doc_id}.png ({len(saved_pages)}/{n})")
        manifest.append(
            {
                "question": example["question"],
                "answers": example.get("answers", []),
                "doc_id": doc_id,
                "image_file": f"{doc_id}.png",
            }
        )

    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fetched {len(saved_pages)} pages, {len(manifest)} Q/A pairs -> {out_path}")


def main() -> None:
    args = parse_args()
    fetch(args.n, args.out)


if __name__ == "__main__":
    main()
