#!/usr/bin/env python
"""Fetch N sample DocVQA pages into backend/eval/corpus/ for the Phase 1
integration/demo corpus and the Phase 5 eval gold set.

STANDALONE utility: not imported by the app, not run by pytest (network +
a large-ish dependency, `datasets`, don't belong in the hermetic test suite).

Source: HuggingFace `datasets`, `lmms-lab/DocVQA` (config "DocVQA", default
split "validation") -- an openly-downloadable re-upload of the DocVQA
benchmark, no auth/token required.
https://huggingface.co/datasets/lmms-lab/DocVQA

Usage:
    pip install datasets   # not in the base test deps; only needed to run this
    python scripts/fetch_sample_docs.py --n 10 --out eval/corpus

For each of the first N examples this saves:
    <out>/<i>_<question_id>.png   -- the document page image
    <out>/<i>_<question_id>.json  -- {"doc_id", "question_id", "question", "answers"}
"""
import argparse
import json
from pathlib import Path

DATASET_NAME = "lmms-lab/DocVQA"
DATASET_CONFIG = "DocVQA"
DATASET_SPLIT = "validation"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=10, help="number of pages to fetch (default: 10)")
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

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=f"{DATASET_SPLIT}[:{n}]")
    for i, example in enumerate(ds):
        stem = f"{i:03d}_{example['questionId']}"
        example["image"].save(out_path / f"{stem}.png")
        meta = {
            "doc_id": example.get("docId") or example.get("ucsf_document_id"),
            "question_id": example["questionId"],
            "question": example["question"],
            "answers": example.get("answers", []),
        }
        (out_path / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {stem}")


def main() -> None:
    args = parse_args()
    fetch(args.n, args.out)


if __name__ == "__main__":
    main()
