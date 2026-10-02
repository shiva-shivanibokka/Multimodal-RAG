# Multimodal-RAG: SOP evaluation (branch `sop-eval`)

This is an independent re-evaluation of the retrieval, refusal and faithfulness
claims made in `README.md` and `backend/eval/report.json`. Everything below was
measured on this machine. Nothing is estimated. Raw outputs are in
`eval_sop/results/` and inputs are in `eval_sop/data/`.

**Status.** The non-LLM parts are done: reproduction, retrieval ablations,
refusal as selective prediction, and validation of the NLI gate on RAGTruth.
**Generation has NOT been run.** End-to-end answer ANLS/EM, end-to-end refusal
and the NLI flag rate on this system's own answers all need a local LLM slot.
The scripts are ready (see "Prepared, not run"). No headline here depends on
future human labels.

## 1. Setup

| item | value |
|---|---|
| Corpus | 40 single-page scans from DocVQA (`lmms-lab/DocVQA`, validation split), exactly the pages listed in `backend/eval/corpus/manifest.json` (gitignored PNGs copied from the original checkout) |
| Ingestion | the production pipeline, via `backend/eval/run_eval.py::ingest_corpus`: docTR OCR, img2table, then chunking. This yields **50 chunks (42 text, 8 table)** in one combined session. It is cached in `eval_sop/cache/ingested.pkl` (gitignored because it holds OCR text of research-licensed images). |
| Models (HF snapshot) | bge-small-en-v1.5 `5c38ec7`, bge-reranker-base `2cfc18c`, clip-ViT-B-32 `327ab67`, nli-deberta-v3-base `6c749ce` |
| Libraries | Python 3.12.3, torch 2.13.0+cpu, sentence-transformers 5.6.0, transformers 5.13.1, faiss 1.14.3, python-doctr 1.0.1, numpy 2.5.1, pandas 3.0.3. All from the pre-existing dev venv `C:\mrag\.venv`, which sits outside the repo. Nothing was installed. |
| Hardware | Windows 11 laptop, CPU only for torch. The run was capped at 2 threads and CrossEncoder batch size 8 to stay polite (speed only). |
| Determinism | Retrieval and NLI are deterministic. The only randomness is in the bootstrap (seed 0, 10,000 resamples for retrieval and 2,000 for AUROC/RAGTruth), the CV folds (seeds 0–4) and the RAGTruth sample (`random_state=0`). |
| Bootstrap | percentile 95% CI over questions. Differences use a paired bootstrap. RAGTruth claim-level CIs use a cluster bootstrap over responses. |

**Gold sets** (`eval_sop/data/`)
- `gold_all151.json` contains **all 151** DocVQA QA pairs in the manifest. The original gold set used only 38: 30 answerable plus 8 OOD.
- **44 questions are flagged as ambiguous**, with the reason for each in `ambiguity_labels.csv`:
  - 13 are flagged automatically because the identical question is asked of two or more different documents (for example "What is the name of the company?" is asked of 10 docs).
  - 31 are flagged manually by Claude because they are deictic ("this document", "the logo", "the first rectangle") and name nothing that picks out one of the 40 pages.
  - The flags were assigned **before** any retrieval was run on the 151 questions. They are a single annotator's judgement (Claude) and have not been human-verified.
  - The remaining 107 make up the "filtered" set.
- `ood_questions.json` contains two kinds of out-of-corpus questions:
  - 8 *trivial* OOD questions from the original repo (Falcon 9, FIFA, ...).
  - **30 *hard*, near-domain OOD questions, author-constructed by Claude.** Each names an entity that appears in the corpus (ITC, CIGFIL, Taco Bell, the Missouri Food Donation Program, ...) but asks for a fact that a keyword search over the docTR OCR suggests is absent. "Unanswerable" holds by construction plus that OCR check. It is not human-verified (OCR can miss text).

**Exact reproduction commands** (from the worktree root, with `C:\mrag\.venv`):
```
set OMP_NUM_THREADS=2 & set MKL_NUM_THREADS=2
python eval_sop/00_ingest.py          # runs production ingestion once, caches it
python eval_sop/01_build_gold.py      # 151-item gold set + ambiguity flags
python eval_sop/make_ood.py           # 8 trivial + 30 author-constructed hard OOD
python eval_sop/02_retrieval.py       # reproduction + ablations + gate scores
python eval_sop/03_refusal.py         # AUROC / fixed threshold / CV threshold / risk-coverage
python eval_sop/repro_findings.py     # deterministic reproductions of code findings
# RAGTruth (MIT): download test parquet to eval_sop/cache/ragtruth/test.parquet (URL in script docstring)
python eval_sop/06_nli_ragtruth.py --n 450 --seed 0
python eval_sop/07_plot.py            # any python with matplotlib
```

## 2. Results

### 2.0 Reproduction of the committed report
`02_retrieval.py` re-ran the original `evaluate_mode`/`aggregate`, and the result
matches `backend/eval/report.json` **exactly** (every diff is 0.0; see
`results/reproduce_report.json`). The prior review's points check out as follows:
- **Refusal accuracy 0.789 = 30/38, the never-refuse baseline. Confirmed.** For dense, hybrid and caption modes the gate refused **0 of 38** questions, including all 8 trivial OOD. The lowest bge gate score on *any* question is 0.42, which is above the 0.25 threshold. For CLIP, 0.711 = 27/38: the gate refused 3 answerable questions and 0 OOD.
- **Citation accuracy is recall@1. Confirmed.** In `run_eval.py`, `cited_pages = [retrieved_pages[0]]`, so the two columns are identical in every mode.
- **Faithfulness was never measured. Confirmed** (`"faithfulness": null`).

### 2.1 Retrieval (n = 107 unambiguous; strict = the DocVQA source page)
Notes on the metrics:
- "R@5" means at least one hit among the **top-5 chunks**. These are de-duplicated pages, so it covers 4.6 pages on average for the text modes and 5 for CLIP.
- With 40 pages, random ranking gives R@1 = 0.025.

| config | R@1 [95% CI] | R@5 [95% CI] | MRR [95% CI] |
|---|---|---|---|
| **hybrid + rerank (production default)** | **0.832 [0.757, 0.897]** | 0.953 [0.907, 0.991] | 0.883 [0.829, 0.932] |
| hybrid, no rerank | 0.813 [0.738, 0.879] | 0.935 [0.888, 0.972] | 0.856 [0.795, 0.912] |
| dense + rerank | 0.813 [0.738, 0.888] | 0.944 [0.897, 0.981] | 0.864 [0.807, 0.918] |
| dense, no rerank | 0.701 [0.607, 0.785] | 0.860 [0.794, 0.925] | 0.772 [0.700, 0.839] |
| **BM25 only (simple baseline)** | 0.794 [0.720, 0.869] | 0.925 [0.869, 0.972] | 0.850 [0.787, 0.905] |
| caption_baseline (bge on page OCR) | 0.692 [0.598, 0.776] | 0.879 [0.813, 0.935] | 0.768 [0.698, 0.836] |
| cross_modal (CLIP text to page image) | 0.290 [0.206, 0.374] | 0.551 [0.458, 0.645] | 0.387 [0.307, 0.469] |

On the same 107 questions, the lenient criterion counts a retrieved page as relevant if its OCR contains a 4+ character gold answer. Under it, hybrid+rerank gets R@1 0.888 [0.822, 0.944] and MRR 0.928.

The **unfiltered 151** questions score much lower. Hybrid+rerank gets R@1 **0.623 [0.543, 0.702]**, R@5 0.801 and MRR 0.689. BM25 gets R@1 0.603, caption_baseline 0.556 and CLIP 0.252. On the 44 ambiguous questions alone, hybrid+rerank gets R@1 0.114. The full tables are in `results/retrieval_summary.json`.

**Paired differences (filtered, strict R@1):**

| comparison | ΔR@1 [95% CI] | bootstrap p |
|---|---|---|
| dense: rerank − no rerank | +0.112 [0.028, 0.206] | 0.015 |
| hybrid: rerank − no rerank | +0.019 [−0.065, 0.103] | 0.73 |
| no rerank: hybrid − dense | +0.112 [0.056, 0.178] | <0.001 |
| rerank: hybrid − dense | +0.019 [0.000, 0.047] | 0.26 |
| hybrid+rerank − BM25 only | +0.037 [−0.028, 0.103] | 0.33 |
| hybrid+rerank − caption_baseline | +0.140 [0.047, 0.234] | 0.005 |
| caption_baseline − CLIP | +0.402 [0.280, 0.523] | <0.001 |

Answer ANLS/EM needs generation, so it is not reported yet (§5). The closest
non-LLM proxy is answer-string coverage: a 4+ character gold answer appears in
the top-5 retrieved pages for **0.972** of the filtered questions (hybrid+rerank, lenient R@5).

### 2.2 Refusal as selective prediction (no LLM)
- Positives: 107 unambiguous answerable questions.
- Negatives: 8 trivial OOD + 30 hard OOD (author-constructed).
- AUROC = P(answerable question scores higher than OOD question).
- The production rule is "answer iff gate ≥ 0.25". The "CV threshold" row instead picks the cutoff on held-out folds by maximising balanced accuracy (5-fold stratified × seeds 0–4, mean ± std).

| signal | AUROC vs trivial OOD | AUROC vs hard OOD | AUROC vs all OOD | acc @0.25 (always-answer baseline) | AURC (random-order AURC) |
|---|---|---|---|---|---|
| dense gate (bge cosine) | 0.911 [0.811, 0.981] | **0.302 [0.198, 0.404]** | 0.430 [0.321, 0.546] | 0.738 (0.738), refuses 0/145 | 0.412 (0.400) |
| hybrid gate (production default) | identical to dense (see note) | 0.302 | 0.430 | 0.738 (0.738) | 0.411 (0.386) |
| caption_baseline gate | 0.916 | 0.281 [0.184, 0.378] | 0.414 | 0.738 (0.738) | 0.447 (0.490) |
| CLIP gate | 0.831 | 0.294 [0.185, 0.415] | 0.407 | **0.621** (0.738): refuses 26% of answerable and 29% of OOD | 0.688 (0.786) |
| BM25-normalised only | 0.916 | 0.445 [0.328, 0.563] | 0.544 | – | 0.286 (0.386) |
| bge-reranker top-1 score (*not shipped*) | 0.895 | **0.632 [0.538, 0.724]** | 0.687 [0.601, 0.772] | – | **0.184** (0.386) |

Notes:
- **The fixed 0.25 threshold never refuses** for any bge-based mode. Answerable questions score 0.465–0.825 and trivial OOD 0.42–0.61. The threshold is below the whole score range, so "refusal accuracy" is just the class prior.
- **Hard OOD questions score *higher* than real questions** (median dense gate 0.706 vs 0.639), so every embedding-similarity gate falls below chance (AUROC ≈ 0.3). The hard questions deliberately name corpus entities. This shows the gate measures topical similarity, not answerability.
- **The hybrid gate never uses its BM25 term.** `bm25_normalized_top1` peaks at 0.046 across all 189 questions, while dense is always ≥ 0.42, so `max(dense, bm25_norm)` equals dense on 189/189 questions. The "Task 7" lexical rescue is inert at this corpus scale.
- Even with a cutoff learned on held-out data, balanced accuracy against all OOD is only 0.518 ± 0.010 (dense) and 0.674 ± 0.005 (reranker score). The plain accuracy of a learned cutoff, 0.717 ± 0.004 for dense, is *below* the always-answer baseline of 0.738.
- Risk–coverage curves are in `results/risk_coverage.png` and `risk_coverage_curves.csv`. A question counts as correct iff it is answerable and the mode's top-1 page is the gold page.

### 2.3 NLI faithfulness gate validated against human labels (RAGTruth)
- Data: `wandb/RAGTruth-processed`, test split, task_type = QA. A seeded sample of **450 responses** split by the product's `split_claims` into **2,924 claims**. Parquet sha256 `2fc4fb70…3bbd`.
- Labels are human span annotations from RAGTruth. A claim counts as hallucinated iff it overlaps an annotated span. Prevalence is 7.0% of claims and 15.8% of responses.
- The gate is the production `verify_claims` (nli-deberta-v3-base, threshold 0.5), with the context windowed the way the product chunks it.
- Positive = hallucinated.

| unit / decision rule | precision | recall | balanced acc | AUROC | flag rate |
|---|---|---|---|---|---|
| claim: flagged if P(entail) < 0.5 | 0.084 [0.054, 0.119] | 0.941 [0.874, 0.985] | 0.585 [0.548, 0.611] | 0.737 [0.651, 0.805] | **0.784** |
| response: ≥1 claim flagged | 0.169 [0.133, 0.205] | 0.986 [0.954, 1.0] | 0.538 [0.517, 0.556] | 0.671 [0.607, 0.732] | 0.922 |
| response: all claims flagged (= `/answer` refusal rule) | 0.207 [0.154, 0.263] | 0.606 [0.492, 0.721] | 0.585 [0.522, 0.650] | 0.671 | 0.462 |
| baseline: always flag (claims / responses) | 0.070 / 0.158 | 1.0 | 0.5 | 0.5 | 1.0 |

As used in production, the gate would **refuse 46% of RAGTruth QA answers, and
79% of those refusals hit answers that human annotators found fully faithful.**
It also misses 39% of hallucinated answers. It ranks claims better than chance
(AUROC 0.74), but the 0.5 threshold sits in the wrong place for this data.
Plausible causes have not been tested:
- deberta's 512-token truncation of ~230-word contexts plus the claim;
- hedging or meta sentences ("Based on the passages…");
- NLI-style entailment being stricter than RAGTruth's "unsupported" definition.

### 2.4 Code-level findings (reproduced, not fixed): `results/repro_findings.json`
- **F1 (sentinel).** `answer.py` treats the reply as a refusal only if `text.strip() == "NOT_IN_DOCUMENTS"`. With a trailing period (`"NOT_IN_DOCUMENTS."`), lower case, or the sentinel inside a sentence, the reply is not treated as a refusal. It flows on to NLI as if it were an answer. An earlier, discarded partial generation run (see §4) produced one such reply from llama3.2.
- **F2 (table shortcut).** On a synthetic attendance table modelled on DocVQA doc 4751, `try_table_answer` answers "How many meetings has Y.C. Deveshwar attended?" with "The count of the No. of meetings attended column is 3." The true value is 2, yet the shortcut sets `supported=True, score=1.0` and skips NLI.
  - On the real corpus the shortcut fired for **0 of 189** questions, so it does not affect any number above.

## 3. What the numbers support, and what they don't

**Supported**
- On 107 unambiguous DocVQA questions over a 40-page / 50-chunk corpus, hybrid retrieval with cross-encoder reranking puts the source page first 83% of the time [76%, 90%].
- Reranking helps dense retrieval (+11 pts R@1, CI excludes 0).
- Text-based retrieval beats CLIP page-image retrieval by a wide margin on these text-dense scans.
- The committed refusal accuracy is the class prior. The retrieval gate cannot separate in-corpus questions from near-domain unanswerable ones.
- The production NLI gate, checked against human labels on 450 RAGTruth QA responses, has low precision (0.08 per claim).

**Not supported**
- Any claim that the system "refuses calibratedly" or "prevents hallucinations". The gate refuses nothing at 0.25, and the NLI firewall flags most faithful claims.
- "Hybrid beats BM25" or "reranking helps hybrid": the CIs include 0.
- The README's ranking "caption_baseline is best (recall@5 0.80)": on 107 or 151 questions, hybrid+rerank is higher (e.g. R@1 +0.14 [0.05, 0.23] on the filtered set).
- Any answer-quality (ANLS/EM) or end-to-end faithfulness number: generation was not run.
- Generalisation beyond 40 pages. The corpus is tiny (50 chunks), so these are easy-retrieval numbers.

## 4. Threats to validity
- **Tiny corpus**, one page per document and 50 chunks. Retrieval at this scale is easy, so R@1 will drop with realistic corpus sizes.
- **Ambiguity filter.** A single annotator (Claude) assigned it before retrieval was run. Removing the 44 flagged questions raises R@1 from 0.62 to 0.83, so the unfiltered numbers should be reported alongside. The criterion is inspectable in `data/ambiguity_labels.csv`.
- **Single gold page.** DocVQA gives one source page, but the same answer can appear on another page (e.g. ITC brands across annual-report pages). The lenient metric bounds this effect.
- **Hard OOD questions** were written by the same party that evaluated them. They deliberately name corpus entities, so AUROC vs hard OOD depends on how the set was constructed. A keyword-over-OCR check suggests they are unanswerable, but this is not human-verified. Thirty items give wide CIs.
- **Reranker score cache.** The eval-side sqlite cache can change a reranker float by ~1e-6 versus scoring in a different batch (padding), which could flip an exact tie only. The reproduction (§2.0) was run before the cache existed and matched exactly.
- **RAGTruth vs this system.** RAGTruth responses come from other LLMs over web passages, not from this system's prompts over OCR text. It validates the gate component, not this system's end-to-end faithfulness. The span-to-sentence mapping is ours.
- **Bootstrap p-values** are approximate, with no multiple-comparison correction across the 8 paired tests.
- **Discarded partial runs.** A first run of 02/04/06 was killed (machine overload). Its partial outputs were moved to `eval_sop/cache/stale_partial_runs/` (gitignored) and **not used**. That includes 42 generation records from llama3.2 3B on a private Ollama instance.

## 5. Prepared, not run (needs an Ollama slot; no paid API is needed)
- `python eval_sop/04_generation.py --model llama3.2:latest --seeds 0 1 2 --temperature 0.7 --ollama-url http://localhost:11434/v1`
  - Runs the production `answer_question` path (gate, hybrid+rerank, table shortcut, LLM, NLI firewall) on the 107 unambiguous + 38 OOD questions, plus a closed-book baseline on the 107.
  - That is 252 local calls per seed, 756 in total.
  - A localhost-only guard aborts any non-local call. The script logs the path taken by each answer, raw output, claims and token usage.
  - Model provenance to record: ollama digest `a80c4f17acd5` (llama3.2 3B Q4).
- Then `python eval_sop/05_analyze_generation.py results/generation_llama3.2_latest_T0.7.jsonl`.
  - It reports ANLS/EM/contains vs DocVQA gold (refused = 0), mean ± std over seeds and bootstrap CIs.
  - It also reports end-to-end refusal on trivial and hard OOD vs the always-answer baseline, the claim flag rate, the share of answers refused by the firewall, the table-shortcut count and the closed-book baseline.
  - The claims CSV it writes is optional, and no headline depends on it.
- If a paid model is wanted later for more representative answer quality, the same run is about 145 × 3 calls × ~1.75k prompt tokens ≈ **0.76M input + ~25k output tokens**. That cost is on the order of $1 for a small-tier model and a few dollars for a frontier-tier model at typical list prices. Check the provider's current price sheet. Nothing was called.

## 6. SOP-ready sentences (strictly true given these numbers)
1. "I built a multimodal document-QA system. On 107 unambiguous DocVQA questions over a 40-page corpus, its hybrid BM25+dense retrieval with cross-encoder reranking places the source page first 83% of the time (95% CI 76–90%). That beats CLIP page-image retrieval by 52 points but is statistically indistinguishable from a plain BM25 baseline."
2. "When I re-evaluated my own system, I found that its reported 0.79 refusal accuracy equalled the never-refuse baseline. On 30 near-domain unanswerable questions, its similarity-based abstention gate scored below chance (AUROC 0.30). A reranker-score signal did better (AUROC 0.63)."
3. "Validating my NLI faithfulness gate against human hallucination labels on 450 RAGTruth QA responses showed high recall (0.94) but very low precision (0.08) at the shipped threshold. That result redirected my work toward calibrating verification rather than adding it."

## 7. Change log (this branch)
Product code (`backend/`, `frontend/`) is **unchanged**. The README is unchanged.

| commit | what | why | evidence | preserved |
|---|---|---|---|---|
| 98399fd | Added `eval_sop/` scripts, `data/gold_all151.json`, `ambiguity_labels.csv`, `ood_questions.json` | the eval plan | §1 | — |
| e18b486 | Eval-side only: 2-thread/batch-8 limits, sqlite reranker cache, checkpoint/resume, `--ollama-url` with a localhost guard, partial-seed filtering in 05, OOD relabelled "author-constructed" (question text verified identical), synthetic table repro | shared laptop and a killed first run; coordinator rules | `eval_sop/common.py`, `02_retrieval.py`, `06_nli_ragtruth.py` | all product rationale comments untouched |
| f94c9a1 | Retrieval/refusal/repro raw results | §2.0–2.2, 2.4 | `eval_sop/results/*` | — |
| next commit | RAGTruth NLI-gate validation raw results | §2.3 | `results/nli_ragtruth_*` | — |
| following commit | RESULTS.md | deliverable | this file | — |

`04_generation.py` monkeypatches `providers._OPENAI_COMPAT["openai"]` and `providers._post` **in-process only**, so a local model can be used without editing the product.

## 8. Proposed, not done (need the owner's decision)
1. Gate threshold, `config.py` `retrieval_min_score = 0.25`: it is below every observed bge score. Consider gating on the reranker score instead, with a threshold tuned on held-out data, and calibrate per mode. CLIP scores sit on a different scale (0.22–0.38).
2. Sentinel, `answer.py` (F1): normalise the reply (strip, strip trailing punctuation, compare case-insensitively) before comparing to `NOT_IN_DOCUMENTS`. This needs a failing test first, which `repro_findings.py` F1 provides.
3. Table shortcut, `table_answer.py` (F2): do not map "how many …" to a column count when the question names a row entity. Alternatively, route the shortcut answer through NLI instead of hard-coding `score=1.0`.
4. NLI gate: recalibrate the threshold, and handle the 512-token truncation by windowing evidence at the model's token limit rather than at 500 words.
5. `run_eval.py`: report citation accuracy from the generated answer's citations, not `retrieved_pages[0]`, and add ANLS/EM.
6. **Proposed README corrections** (not applied):
   - Replace "~0.79 refusal accuracy" with "refusal accuracy equals the never-refuse baseline (0.79 = 30/38); the 0.25 gate does not refuse any question".
   - Replace "caption_baseline … recall@5 0.80" with the 107/151-question numbers above.
   - State that faithfulness has not been measured end to end, and give the RAGTruth gate numbers.
   - Remove "calibrated refusal" wording until there is evidence for it.
7. Housekeeping for the user, which I could not do:
   - A private `ollama serve` that I started on port 11435 (PID 104868, started 2026-10-01 18:34) is still running. My stop request was denied by the permission system, so please stop it yourself.
   - The venv `C:\mrag\.venv` is outside the repo.
