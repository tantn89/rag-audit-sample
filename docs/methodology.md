# Methodology

How the audit was run, what it measures and where each number comes from. The eval set and the
grader have their own pages: [eval-set-design.md](eval-set-design.md) and
[grader-design.md](grader-design.md).

## System under test

| Item | Value | Where it is set |
|---|---|---|
| App | AnythingLLM 1.20.0, image pinned by digest, unmodified | [`compose.yaml`](../compose.yaml) |
| Corpus | 16 FOMC 2025 PDFs (8 statements, 8 minutes), 181 pages, SHA-256 pinned | [`corpus/corpus-manifest.json`](../corpus/corpus-manifest.json) |
| Chat model | gpt-4.1-mini, OpenAI-compatible, through the local `llm-api-guard` proxy | `compose.yaml` |
| Chat mode | `query` (documents only), a fresh session per question, streaming off | [`rag_audit/audit_configs.py`](../rag_audit/audit_configs.py) |
| Configurations | baseline, fix and two retrieval candidates, each a settings-only change | `rag_audit/audit_configs.py` |

The app's code is never modified. Every configuration is something a client could set in the
AnythingLLM UI or API.

## Configurations

| Name | Change from baseline | How it was measured |
|---|---|---|
| `baseline` | none: topN 4, threshold 0.25, default search, default prompt, local MiniLM embedder, 1000-char chunks | 3 trials × 67 questions, plus a `/vector-search` top-10 probe |
| `candidate-rerank` | `vectorSearchMode: rerank` | 1-trial chat pilot (context doc recall@4 0.842) |
| `candidate-openai-embedder` | text-embedding-3-small embedder **and** about 6k-char chunks (`EMBEDDING_MODEL_MAX_CHUNK_LENGTH=8191`, which the app also uses as the chunk size) | `/vector-search` probe; kept for the record, confounded by chunk size |
| `candidate-openai-embedder-1000` | text-embedding-3-small embedder only, 1000-char chunks | `/vector-search` probe, directly comparable with the fix |
| `fix` | `vectorSearchMode: rerank` | 3 trials × 67 questions, plus a `/vector-search` probe |

The `/vector-search` endpoint honours the workspace search mode, so the fix's probe measures the
reranked order. Retrieval figures in the report all come from this probe over the 56 answerable
questions.

## Trials, timeouts and the gate

- **3 trials per question per configuration** (201 per configuration). The model runs at the app's
  default temperature, so answers vary between trials; repeated trials separate real changes from
  noise.
- **Timeouts are not quality failures.** Each call has a timeout and at most 2 billed attempts. A
  trial that still times out is saved as `status: timeout`, excluded from the pass rate and counted
  separately. `run --retry-timeouts` gives such trials one bounded extra pass; the timed-out
  attempts stay in the trace. The final numbers use 201 of 201 trials in both configurations.
- **regressgate** compares two configurations: a permutation test on the pass rate (10,000
  shuffles) and a Fisher exact test per question, Tarone-corrected. Its report is
  [`results/compare-baseline-vs-fix.md`](../results/compare-baseline-vs-fix.md).

## Disclosures

- **The grader was refined after reading the baseline traces.** The refinements were: context
  support counts only chunks from the expected documents; an answer whose lead declines is not a
  pass; partial multi-hop recall is a retrieval miss; four case fixes (alternate phrasings that
  were correct but not accepted, and one case that needed a stricter fact). One sentence-splitter
  fix ("U.S." ended a sentence) was found in the fix traces. Every refinement applies identically
  to both configurations, and both were re-graded from the saved traces with
  `python -m rag_audit reproduce`.
- **The LLM judge is weak.** It agreed with manual labels on 11 of 18 spot-checked trials (61%)
  for each of three prompt versions ([`results/judge-spot-check.jsonl`](../results/judge-spot-check.jsonl)).
  It only adds the `unsupported_claim` flag; pass and fail never depend on it.
- **Latency is indicative.** The provider stalled during the fix run (calls finished in 1–3 s or
  hung for minutes), so end-to-end tails are confounded. The report separates the app's own time
  (retrieval and reranking) from the LLM time recorded by the app.
- **Costs are estimates** at OpenAI list prices for gpt-4.1-mini ($0.40 / $1.60 per million prompt /
  completion tokens) and text-embedding-3-small ($0.02 per million). The provider's own prices
  were not verified. Token counts come from the proxy log.

## Where every number comes from

| Number | Source | Rebuilt by |
|---|---|---|
| Pass rates, failure labels, recall, latency, tokens | `results/<config>/summary.json`, `trial-grades.jsonl` | `python -m rag_audit reproduce` (baseline, fix), `python -m rag_audit report <config> --from-bundle` (candidates) |
| regressgate verdict and per-question changes | `results/compare-baseline-vs-fix.{md,json}` | `python -m rag_audit reproduce` |
| Judge agreement | `results/compare-baseline-vs-fix.json` (`judge_agreement`) | `python -m rag_audit reproduce` |
| Cover-page share of context, latency split, charts | `report/report-numbers.json`, `report/figures/` | `python report/build-report-figures.py` |
| Trace excerpts quoted in the report | `results/<config>/traces.jsonl.gz` | read directly (gzip JSON lines) |
