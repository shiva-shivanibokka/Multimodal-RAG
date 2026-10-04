# Third-party data in `eval_sop/` — licence notes

These are notes, not legal advice. Nothing has been removed. The repository owner
should decide whether each item may stay in a public repo.

## DocVQA (Mathew, Karatzas & Jawahar, WACV 2021)

- **Source.** The QA pairs come from `lmms-lab/DocVQA` on Hugging Face (validation
  split), a re-upload of the official DocVQA release.
- **Original terms.** The official release is distributed by the RRC portal
  (rrc.cvc.uab.es) under its own terms of use, and the page images come from the
  UCSF Industry Documents Library. This repo's `backend/eval/corpus/README.md`
  describes the data as "research-use licensed", which is why the page images are
  gitignored.
- **Not verified.** We did not verify the licence field of the `lmms-lab/DocVQA`
  dataset card. The Hugging Face API returned no card data when we queried it.
  Please check the terms before redistribution.
- **Committed files with DocVQA-derived text:**
  - `data/gold_all151.json` and `data/ambiguity_labels.csv`. Both contain DocVQA
    questions and answers, which already appear in
    `backend/eval/corpus/manifest.json` on `main`.
  - `results/generation_llama3.2_latest_T0.7.jsonl`. It contains DocVQA questions
    and gold answers, plus model replies that can quote OCR text of DocVQA pages.
- **Not committed.** The page images and the OCR cache (`eval_sop/cache/`) are
  gitignored.
- **No DocVQA text.** `results/retrieval_*.json`, `gate_scores.json` and
  `risk_coverage_curves.csv` hold only ids, ranks and scores.

## RAGTruth (Niu et al., ACL 2024)

- **Source.** The data comes from `wandb/RAGTruth-processed`, test split. That
  dataset card's README states "released under the MIT License", although its
  `license` metadata field is empty.
- **Upstream licence.** The upstream GitHub repo `ParticleMedia/RAGTruth` reports
  the MIT licence via the GitHub API.
- **Committed file with RAGTruth text:** `results/nli_ragtruth_claims.csv`. It
  holds sentences from RAGTruth model outputs, with the human labels we derived.
- **No RAGTruth text.** `results/nli_ragtruth_responses.csv` and
  `nli_ragtruth_summary.json` hold only ids, scores and labels.
- **Not committed.** The parquet itself is gitignored (`eval_sop/cache/ragtruth/`).

## Author-constructed data

- `data/ood_questions.json` and `data/ood_hard_for_review.csv`:
  - 8 questions are copied from this repo's original gold set.
  - 30 questions were constructed by the evaluator (Claude) to name corpus entities.
- The ambiguity flags in `data/ambiguity_labels.csv` were assigned by the
  evaluator (Claude) and have not been human-verified.
