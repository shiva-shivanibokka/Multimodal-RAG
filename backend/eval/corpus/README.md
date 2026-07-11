# Eval corpus (DocVQA sample)

`manifest.json` in this directory is committed (small, text-only: 40 pages /
151 question-answer pairs, ~30KB). The page images (`*.png`, ~26MB for 40
pages) are **gitignored** -- DocVQA is research-use licensed, so the images
aren't redistributed in this repo. Regenerate them locally:

```bash
pip install datasets
HF_HOME=/path/to/cache HF_DATASETS_CACHE=/path/to/cache \
  python backend/scripts/fetch_sample_docs.py --n 40 --out backend/eval/corpus
```

This re-downloads the same 40 pages from `lmms-lab/DocVQA` (HuggingFace,
validation split, streamed -- see the script's docstring for the exact
field names and how the download is bounded) and regenerates
`manifest.json` identically (docIds are stable across runs).

`backend/eval/gold/enterprise_docs.json` (the evaluation gold set) was
derived from this manifest and is committed independently of the corpus
images, so gold-set schema tests (`backend/tests/test_gold_set.py`) don't
need the corpus or network access.
