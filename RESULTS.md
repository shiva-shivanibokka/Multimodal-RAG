# Multimodal-RAG: SOP evaluation (branch `sop-eval`)

This is an independent re-evaluation of the retrieval, refusal and faithfulness
claims made in `README.md` and `backend/eval/report.json`. Everything below was
measured on this machine. Nothing is estimated. Raw outputs are in
`eval_sop/results/` and inputs are in `eval_sop/data/`.

**Status.** All parts have been run:
- reproduction of the committed report;
- retrieval ablations;
- refusal as selective prediction;
- validation of the NLI gate on RAGTruth;
- end-to-end generation with a **local** llama3.2 3B through Ollama, using 3 seeds (§2.5).

No paid API was called. No headline depends on future human labels.

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
  - The flags were assigned **before** any retrieval was run on the 151 questions. They are a single annotator's judgement (an LLM, Claude) and have not been human-verified.
  - The independent review found that several dropped questions are in fact uniquely answerable in this corpus (it named mq-103, 104, 107, 109, 113, 137 and 007). The filter is therefore too aggressive. Section 2.1 reports all 151 questions alongside the filtered set, plus a sensitivity analysis.
  - The remaining 107 make up the "filtered" set.
- `ood_questions.json` contains two kinds of out-of-corpus questions:
  - 8 *trivial* OOD questions from the original repo (Falcon 9, FIFA, ...).
  - **30 *hard*, near-domain OOD questions, constructed by the evaluator (Claude) to name corpus entities.** Each names an entity that appears in the corpus (ITC, CIGFIL, Taco Bell, the Missouri Food Donation Program, ...) but asks for a fact that a keyword search over the docTR OCR suggests is absent. "Unanswerable" holds by construction plus that OCR check. It is not human-verified (OCR can miss text).

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
python eval_sop/08_sensitivity.py     # retrieval sensitivity to the ambiguity filter
```

## 2. Results

### 2.0 Reproduction of the committed report
`02_retrieval.py` re-ran the original `evaluate_mode`/`aggregate`, and the result
matches `backend/eval/report.json` **exactly** (every diff is 0.0; see
`results/reproduce_report.json`). The prior review's points check out as follows:
- **Refusal accuracy 0.789 = 30/38, the never-refuse baseline. Confirmed.** For dense, hybrid and caption modes the gate refused **0 of 38** questions, including all 8 trivial OOD. The lowest bge gate score on *any* question is 0.42, which is above the 0.25 threshold. For CLIP, 0.711 = 27/38: the gate refused 3 answerable questions and 0 OOD.
- **Citation accuracy is recall@1. Confirmed.** In `run_eval.py`, `cited_pages = [retrieved_pages[0]]`, so the two columns are identical in every mode.
- **Faithfulness was never measured. Confirmed** (`"faithfulness": null`).

### 2.1 Retrieval (strict = the DocVQA source page)

**Headline, reported both ways.** Production hybrid+rerank R@1 is:
- **0.623 [0.543, 0.702] on all 151 questions**;
- **0.832 [0.757, 0.897] on the 107** kept by the single-annotator (LLM) ambiguity filter.

The table below uses the filtered 107.
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

**Sensitivity to the ambiguity filter** (`08_sensitivity.py` → `results/retrieval_sensitivity.json`). Each row is hybrid+rerank R@1.

| subset | n | R@1 [95% CI] | Δ vs BM25 | Δ vs CLIP |
|---|---|---|---|---|
| all 151 (no filter) | 151 | 0.623 [0.543, 0.702] | +0.020 [−0.040, 0.079] | +0.371 [0.272, 0.470] |
| filtered (Claude) | 107 | 0.832 [0.757, 0.897] | +0.037 [−0.028, 0.103] | +0.542 [0.430, 0.654] |
| filtered + the 7 items the reviewer named | 114 | 0.825 [0.754, 0.895] | +0.061 [−0.009, 0.140] | +0.544 [0.430, 0.649] |
| filtered + 22 items restored by an explicit rule\* | 129 | 0.721 [0.643, 0.798] | +0.039 [−0.023, 0.109] | +0.457 [0.349, 0.558] |
| reviewer's 12-item restoration (*quoted, not reproduced*: the 12 ids were not provided) | 119 | 0.790 [0.714, 0.857] | +0.050 [−0.017, 0.126] | – |

\*The rule restores a dropped item if a gold answer of 4+ normalised characters occurs in the OCR of exactly one corpus page and that page is the source page. Short generic answers ("bird", "1971") can pass it trivially, so it is an over-restoration bound. Two of the reviewer's items fail it: mq-007 and mq-109 have answers that appear on two pages.

**Across every subset, hybrid+rerank vs BM25 is not significant (CI includes 0), and the gap to CLIP is 37–54 points.**

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

Answer-string coverage is a non-LLM proxy for answer quality: a 4+ character gold
answer appears in the top-5 retrieved pages for **0.972** of the filtered questions
(hybrid+rerank, lenient R@5). Generated-answer ANLS/EM is in §2.5.

### 2.2 Refusal as selective prediction (no LLM)

**Core finding:** the shipped 0.25 threshold refused **0 of 145** questions (107 answerable + 38 OOD) in the dense, hybrid and caption modes. The committed "refusal accuracy" is therefore the class prior.

The gate does separate answerable questions from obviously off-topic ones: AUROC is **0.91** against the 8 trivial OOD questions. It does not separate them from the 30 hard OOD questions (AUROC **0.30**). Those were constructed by the evaluator (Claude) to name corpus entities, so this number depends on how the set was built.

- Positives: 107 unambiguous answerable questions.
- Negatives: 8 trivial OOD (from the repo) + 30 hard OOD (constructed by the evaluator (Claude) to name corpus entities).
- AUROC = P(answerable question scores higher than OOD question).
- The production rule is "answer iff gate ≥ 0.25". The "CV threshold" row instead picks the cutoff on held-out folds by maximising balanced accuracy (5-fold stratified × seeds 0–4, mean ± std).

| signal | AUROC vs trivial OOD | AUROC vs hard OOD | AUROC vs all OOD | acc @0.25 (always-answer baseline) | AURC (random-order AURC) |
|---|---|---|---|---|---|
| dense gate (bge cosine) | **0.911 [0.811, 0.981]** | **0.302 [0.198, 0.404]** | 0.430 [0.321, 0.546] | 0.738 (0.738), refuses 0/145 | 0.412 (0.400) |
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
- **F1 (sentinel).** `answer.py` treats the reply as a refusal only if `text.strip() == "NOT_IN_DOCUMENTS"`. With a trailing period (`"NOT_IN_DOCUMENTS."`), lower case, or the sentinel inside a sentence, the reply is not treated as a refusal. It flows on to NLI as if it were an answer. This occurred once in 42 replies of an earlier, discarded partial llama3.2 run (see §4).
- **F2 (table shortcut).** On a synthetic attendance table modelled on DocVQA doc 4751, `try_table_answer` answers "How many meetings has Y.C. Deveshwar attended?" with "The count of the No. of meetings attended column is 3." The true value is 2, yet the shortcut sets `supported=True, score=1.0` and skips NLI.
  - On the real corpus the shortcut fired for **0 of 189** questions, so it does not affect any number above.

### 2.5 End-to-end generation (production `answer_question`, local llama3.2 3B)

**Setup**
- Model: Ollama `llama3.2:latest`, a 3.2B Q4_K_M build, digest `a80c4f17acd5`.
- Server: the shared server on :11434, with server context 4096. The largest prompt was 3,336 tokens, so nothing was truncated.
- Sampling: temperature 0.7, seeds 0, 1 and 2, one request at a time. The model was unloaded afterwards (`keep_alive: 0`, `/api/ps` empty).
- Pipeline: the full production path (gate, then hybrid+rerank, table shortcut, LLM, NLI firewall). It is wired to Ollama by an in-process monkeypatch only.
- Data: 107 unambiguous answerable + 8 trivial OOD + 30 hard OOD questions per seed, plus a closed-book baseline on the 107. That is 756 calls, with 0 errors.
- Statistics: mean ± std across the 3 seeds; [95% CI] is a bootstrap over questions of each question's seed-averaged value.
- Raw output: `results/generation_llama3.2_latest_T0.7.jsonl`. Summary: `..._summary.json`.

**Answer quality vs the DocVQA gold answers** (n = 107; a refused answer scores 0)

| metric | RAG (production) | closed-book baseline (same model, no context) |
|---|---|---|
| ANLS | 0.093 ± 0.019 [0.059, 0.130] | 0.005 ± 0.007 [0.000, 0.012] |
| exact match | 0.059 ± 0.022 [0.031, 0.093] | 0.000 ± 0.000 |
| gold answer contained in reply (lenient) | 0.377 ± 0.035 [0.302, 0.452] | 0.006 ± 0.004 [0.000, 0.019] |
| answered (not refused) | 0.573 ± 0.012 [0.498, 0.648] | - |
| top-level citations include the gold page | 0.333 ± 0.032 [0.262, 0.408] | - |

How to read these numbers:
- ANLS and EM are low partly because the model answers in full sentences while DocVQA gold answers are short spans. The prompt does not ask for a short answer, and we did not change it.
- The "contains" metric is the fairer read of the same answers.
- Over answered items only (pooled over seeds, n = 184): contains = 0.658, ANLS = 0.162, citation includes the gold page = 0.582.

**Refusal outcomes**

| question set | refusal rate | correct-decision accuracy | always-answer baseline |
|---|---|---|---|
| trivial OOD (n=8) | 0.750 ± 0.000 [0.375, 1.0] | | |
| hard OOD (n=30) | 0.711 ± 0.096 [0.589, 0.833] | | |
| answerable (n=107) | 0.427 ± 0.012 (wrongly refused) | | |
| all 145 | | **0.611 ± 0.018 [0.547, 0.676]** | **0.738** |

- **No** question was refused by the retrieval gate.
- The LLM's exact `NOT_IN_DOCUMENTS` reply caused 6 OOD refusals in total.
- Every other refusal came from the NLI firewall.

**Decision accuracy: plain vs balanced, lenient vs strict.** These come from `decision_metrics` in the summary JSON. Each cell is a point estimate with a cluster-bootstrap CI over the 145 questions (3 seeds each). Two definitions are used:
- **Lenient:** an answerable question is "correct" if it was not refused, even when the answer is wrong or a literal sentinel reply. An OOD question is correct if refused.
- **Strict:** an answerable question is correct only if it was answered, the reply is not a `NOT_IN_DOCUMENTS` variant, and the reply contains the gold answer. An OOD question is correct if refused or if the reply is a sentinel variant.

"Firewall off" is a post-hoc counterfactual built from the logged raw drafts. It shows every LLM draft except the exact sentinel; no new calls were made.

| policy | plain acc (lenient) | balanced acc (lenient) | plain acc (strict) | balanced acc (strict) |
|---|---|---|---|---|
| never refuse (baseline) | 0.738 | 0.500 | – | – |
| **system as shipped** | 0.611 [0.547, 0.676] | **0.646 [0.576, 0.714]** | 0.531 [0.460, 0.600] | 0.671 [0.630, 0.712] |
| firewall off (counterfactual) | 0.752 [0.680, 0.816] | 0.526 [0.505, 0.551] | 0.582 [0.517, 0.646] | 0.578 [0.512, 0.643] |

How to read the table:
- **The mix is 74% answerable.** Under plain accuracy, the shipped system (0.611) is below never refusing (0.738).
- **Balanced accuracy reverses that.** The shipped system scores 0.646, against 0.500 for never refusing.
- **Under the strict definition, the firewall trades answer coverage for abstention.**
  - Answerable questions correct: 0.377 shipped vs 0.586 with the firewall off.
  - OOD questions correct: 0.965 shipped vs 0.570 with the firewall off.
  - Strict balanced accuracy is higher with the firewall (0.671 vs 0.578), but strict plain accuracy is lower (0.531 vs 0.582).
- Neither metric alone is "the" answer, so both are reported.

**The NLI firewall on this system's own answers**
- Claim flag rate (unsupported): **0.507 [0.431, 0.582]**, pooled over 613 claims from 145 questions with a cluster CI. Per seed it is 0.482 ± 0.018 on answerable questions and 0.609 ± 0.075 on OOD.
- The firewall fully refused 49.6% ± 2.8% of the LLM-drafted answers.
- A label-free check against the DocVQA gold answers covers 321 answerable drafts from 107 questions. CIs come from a cluster bootstrap over questions. The firewall refused:
  - **35.6% [26.0%, 45.4%]** of drafts *containing the gold answer* (n = 188);
  - 52.6% [41.5%, 63.3%] of drafts lacking it (n = 133).
  - The gap is **17.0 points [2.8, 31.1]**: selective, but weakly so, consistent with §2.3.
- "Containing the gold answer" is the lenient contains-match, not human-judged correctness.

**F1 has real impact.** 31 of the 435 RAG replies were a sentinel variant that was not refused: 3 on answerable questions and 28 on OOD questions. Of these, 30 were exactly `"NOT_IN_DOCUMENTS."` (with a period) and 1 was `"Not_IN_DOCUMENTS."`.
- `answer.py` does not treat these replies as refusals.
- The NLI gate then scores the single claim as *entailed*, with P = 0.748 or 0.743.
- So the reply is returned as a supported, cited answer.

If the sentinel were matched leniently (post-hoc analysis only; the product is unchanged):
- hard-OOD refusal would be 0.956 ± 0.063;
- trivial-OOD refusal would be 1.0;
- overall plain decision accuracy would be 0.669 ± 0.020 (vs 0.738 for never refusing on plain accuracy). The strict columns above already count sentinel replies on OOD as refusals.

The table shortcut fired 0 times.

## 3. What the numbers support, and what they don't

**Supported**
- Over a 40-page / 50-chunk corpus, hybrid retrieval with cross-encoder reranking puts the source page first:
  - 62% [54%, 70%] of the time on all 151 DocVQA questions;
  - 83% [76%, 90%] on the 107 an LLM annotator judged unambiguous;
  - 72%–83% under the filter-sensitivity variants.
- Reranking helps dense retrieval (+11 pts R@1, CI excludes 0).
- Text-based retrieval beats CLIP page-image retrieval by a wide margin on these text-dense scans.
- The committed refusal accuracy is the class prior: the 0.25 threshold refused 0/145.
- The retrieval gate separates trivial OOD questions well (AUROC 0.91). It does not separate near-domain questions that the evaluator constructed to name corpus entities (AUROC 0.30).
- The production NLI gate, checked against human labels on 450 RAGTruth QA responses, has low precision (0.08 per claim).
- With a local 3B model, retrieval-augmented answering beats the closed-book baseline by a wide margin: gold answer contained 0.377 vs 0.006.
- End-to-end, the answer/refuse decision has balanced accuracy 0.646 [0.576, 0.714] (never refusing: 0.50). Its plain accuracy, 0.611, is below never refusing (0.738) because the mix is 74% answerable.
- The NLI firewall refuses 36% [26%, 45%] of drafts containing the gold answer, vs 53% of drafts lacking it.

**Not supported**
- Any claim that the system "refuses calibratedly" or "prevents hallucinations". The gate refuses nothing at 0.25, and the NLI firewall flags most faithful claims.
- "Hybrid beats BM25" or "reranking helps hybrid": the CIs include 0.
- The README's ranking "caption_baseline is best (recall@5 0.80)": on 107 or 151 questions, hybrid+rerank is higher (e.g. R@1 +0.14 [0.05, 0.23] on the filtered set).
- Any answer-quality claim about a frontier model. Only a 3B local model was run, at temperature 0.7.
- "Faithfulness rate" in the README's sense. The NLI "supported" rate (about 0.49) is a property of the gate, which §2.3 shows is poorly calibrated, not a human-validated faithfulness rate.
- Generalisation beyond 40 pages. The corpus is tiny (50 chunks), so these are easy-retrieval numbers.

## 4. Threats to validity
- **Tiny corpus**, one page per document and 50 chunks. Retrieval at this scale is easy, so R@1 will drop with realistic corpus sizes.
- **Ambiguity filter.** A single LLM annotator (Claude) assigned it before retrieval was run, and the review found it over-aggressive. Removing the 44 flagged questions raises R@1 from 0.62 to 0.83, and plausible restorations give 0.72–0.83 (§2.1). Always quote the all-151 number alongside. The criterion is inspectable in `data/ambiguity_labels.csv`.
- **Single gold page.** DocVQA gives one source page, but the same answer can appear on another page (e.g. ITC brands across annual-report pages). The lenient metric bounds this effect.
- **Hard OOD questions** were written by the same party that evaluated them. They deliberately name corpus entities, so AUROC vs hard OOD depends on how the set was constructed. A keyword-over-OCR check suggests they are unanswerable, but this is not human-verified. Thirty items give wide CIs.
- **Reranker score cache.** The eval-side sqlite cache can change a reranker float by ~1e-6 versus scoring in a different batch (padding), which could flip an exact tie only. The reproduction (§2.0) was run before the cache existed and matched exactly.
- **RAGTruth vs this system.** RAGTruth responses come from other LLMs over web passages, not from this system's prompts over OCR text. It validates the gate component, not this system's end-to-end faithfulness. The span-to-sentence mapping is ours.
- **Bootstrap p-values** are approximate, with no multiple-comparison correction across the 8 paired tests.
- **Generation model.** A 3B Q4 local model at temperature 0.7 with the product's unmodified prompt. Answer quality, sentinel formatting and firewall rates are model-dependent. Server context was 4096 tokens; the largest prompt was 3,336.
- **"Contains" metric** is lenient. A short gold answer such as "2" can match a long reply by accident, so it is an upper bound, not EM. ANLS and EM are reported alongside.
- **Discarded partial runs.** A first run of 02/04/06 was killed (machine overload). Its partial outputs were moved to `eval_sop/cache/stale_partial_runs/` (gitignored) and **not used**. That includes 42 generation records from llama3.2 3B on a private Ollama instance.

## 5. Generation run (done; kept here for reproducibility)
- `python eval_sop/04_generation.py --model llama3.2:latest --seeds 0 1 2 --temperature 0.7 --ollama-url http://localhost:11434/v1`
  - Runs the production `answer_question` path (gate, hybrid+rerank, table shortcut, LLM, NLI firewall) on the 107 unambiguous + 38 OOD questions, plus a closed-book baseline on the 107.
  - That is 252 local calls per seed, 756 in total.
  - A localhost-only guard aborts any non-local call. The script logs the path taken by each answer, raw output, claims and token usage.
  - Model provenance to record: ollama digest `a80c4f17acd5` (llama3.2 3B Q4).
- Then `python eval_sop/05_analyze_generation.py results/generation_llama3.2_latest_T0.7.jsonl`.
  - It reports ANLS/EM/contains vs DocVQA gold (refused = 0), mean ± std over seeds and bootstrap CIs.
  - It also reports end-to-end refusal on trivial and hard OOD vs the always-answer baseline, the claim flag rate, the share of answers refused by the firewall, the table-shortcut count and the closed-book baseline.
  - It can also write a claims CSV with `--export-claims`. This is optional, was not run, and no headline depends on it.
- If a paid model is wanted later for more representative answer quality, the same run is about 145 × 3 calls × ~1.75k prompt tokens ≈ **0.76M input + ~25k output tokens**. That cost is on the order of $1 for a small-tier model and a few dollars for a frontier-tier model at typical list prices. Check the provider's current price sheet. Nothing was called.

## 6. SOP-ready sentences (strictly true given these numbers)
1. "I built a multimodal document-QA system. Over a 40-page DocVQA corpus, its hybrid BM25+dense retrieval with cross-encoder reranking ranks the source page first for 62% of all 151 questions (95% CI 54–70%), and 83% (76–90%) of the 107 questions an LLM annotator judged unambiguous. That is 37–54 points above CLIP page-image retrieval, but not statistically distinguishable from a plain BM25 baseline in any subset."
2. "When I re-evaluated my own system, I found that its reported 0.79 refusal accuracy equalled the never-refuse baseline: the shipped similarity threshold refused 0 of 145 questions. The underlying score separated obviously off-topic questions well (AUROC 0.91), but not 30 near-domain unanswerable questions that I constructed to name entities in the corpus (AUROC 0.30)."
3. "Validating my NLI faithfulness gate against human hallucination labels on 450 RAGTruth QA responses showed high recall (0.94) but very low precision (0.08) at the shipped threshold."
4. "In an end-to-end test with a local 3B model, retrieval raised the share of replies containing the gold answer from 0.6% (closed-book) to 38%. The NLI firewall refused 36% (95% CI 26–45%) of drafts containing the gold answer versus 53% of drafts lacking it. The system's answer/refuse decisions reached a balanced accuracy of 0.65, against 0.50 for never refusing."

## 7. Change log (this branch)
Product code (`backend/`, `frontend/`) is **unchanged**. The README is unchanged.

| commit | what | why | evidence | preserved |
|---|---|---|---|---|
| 98399fd | Added `eval_sop/` scripts, `data/gold_all151.json`, `ambiguity_labels.csv`, `ood_questions.json` | the eval plan | §1 | — |
| e18b486 | Eval-side only: 2-thread/batch-8 limits, sqlite reranker cache, checkpoint/resume, `--ollama-url` with a localhost guard, partial-seed filtering in 05, OOD relabelled "author-constructed" (question text verified identical), synthetic table repro | shared laptop and a killed first run; coordinator rules | `eval_sop/common.py`, `02_retrieval.py`, `06_nli_ragtruth.py` | all product rationale comments untouched |
| f94c9a1 | Retrieval/refusal/repro raw results | §2.0–2.2, 2.4 | `eval_sop/results/*` | — |
| 0f813b3 | RAGTruth NLI-gate validation raw results | §2.3 | `results/nli_ragtruth_*` | — |
| 3edd16c, 23f63bb | RESULTS.md, plus a wording fix (CLIP gap is 54 pts, not 52) | deliverable | this file | — |
| 78a18e3 | `05_analyze_generation.py`: e2e decision CIs, pooled claim flag rate with cluster bootstrap, post-hoc sentinel-normalised analysis, claims export made opt-in | needed for the CIs requested for §2.5 | `eval_sop/05_analyze_generation.py` | product unchanged; the original metrics are kept |
| b92bc09 | generation raw results + §2.5 | coordinator granted the Ollama slot | `results/generation_*`, `04_generation.log` | — |
| 6637983 | change-log hashes | bookkeeping | this file | — |
| 02a3d4c | `05`: plain + balanced and strict decision accuracy, firewall-off counterfactual, firewall refusal by draft correctness, all with cluster bootstrap over questions | review items 1 and 4 | `decision_metrics` in `results/generation_..._summary.json`; reviewer's ≈0.65 balanced and 36% [26, 45] / gap [3, 31] reproduced | existing summary keys byte-identical (checked) |
| 1b2b7d5 | `04`: `--num-ctx-check` default 8192 → 4096 and actually enforced (it was never read) | review item 6 | `04_generation.py` arg parser; recorded max prompt+completion 3,633 < 4096 | recorded results unchanged (no rerun) |
| 86a936d | `08_sensitivity.py` + `results/retrieval_sensitivity.json` | review item 3 | §2.1 table | — |
| b0cee09 | `eval_sop/data/LICENSES.md` | review item 7 | DocVQA / RAGTruth notes | nothing deleted |
| this commit | RESULTS.md review fixes: all-151 R@1 next to 83%, sensitivity, refusal core finding + trivial AUROC, hard-OOD label, decision table, firewall CIs, sentinel count 30+1 and scores 0.748/0.743, SOP sentences 1–4 rewritten (sentence 3's untrue "redirected my work" clause removed), §8.7 housekeeping removed | review items 1–7 | sections above | numbers unchanged except where marked |

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
