# Methodology

How the audit was run, what it measures and where each number comes from. The eval set and the
grader have their own pages: [eval-set-design.md](eval-set-design.md) and
[grader-design.md](grader-design.md).

## System under test

| Item | Value | Where it is set |
|---|---|---|
| App | AnythingLLM 1.20.0, image pinned by digest, unmodified | [`compose.yaml`](../compose.yaml) |
| Corpus | 16 FOMC 2025 PDFs (8 statements, 8 minutes), 181 pages, SHA-256 pinned | [`corpus/corpus-manifest.json`](../corpus/corpus-manifest.json) |
| Chat model | `gpt-4.1-mini`, the configured model preference of an OpenAI-compatible endpoint, reached through a local allowlisting proxy | `compose.yaml` |
| Chat mode | `query` per request (documents only), a fresh session per question, streaming off | [`rag_audit/audit_configs.py`](../rag_audit/audit_configs.py) |
| Configurations | baseline, fix and three candidates, each a settings-only change | `rag_audit/audit_configs.py` |

The app's code is never modified. Every configuration is something a client could set in the
AnythingLLM UI or API.

- **Model name.** `gpt-4.1-mini` is the model the app and the judge asked for. The endpoint's
  responses named the same model; that was not verified independently.
- **Chat mode.** The workspace's stored `chatMode` stays at the app default, `automatic` (it is
  recorded in every trace). Every request sent `mode: "query"`, which overrides it for that call.
- **Live re-runs.** The proxy used here (`llm-api-guard`) is the author's own and is not published.
  To re-run live you need an OpenAI-compatible endpoint, ideally behind your own allowlisting
  proxy. Point `GENERIC_OPEN_AI_BASE_PATH` in `compose.yaml` at it, or set `LLM_GUARD_DIR` for
  `scripts/start-guard.ps1`.

## Configurations

| Name | Change from baseline | How it was measured |
|---|---|---|
| `baseline` | none: topN 4, threshold 0.25, default search, default prompt, local MiniLM embedder, 1000-char chunks | 3 trials × 67 questions, plus a `/vector-search` top-10 probe |
| `candidate-rerank` | `vectorSearchMode: rerank` | 1-trial chat pilot on the same 67 questions (context doc recall@4 0.842) |
| `candidate-openai-embedder` | text-embedding-3-small embedder **and** about 6k-char chunks (`EMBEDDING_MODEL_MAX_CHUNK_LENGTH=8191`, which the app also uses as the chunk size) | `/vector-search` probe (recall@4 0.750); two changes at once |
| `fix` | `vectorSearchMode: rerank` | 3 trials × 67 questions, plus a `/vector-search` probe |
| `candidate-openai-embedder-1000` | text-embedding-3-small embedder only, 1000-char chunks | `/vector-search` probe (recall@4 0.792), measured after the fix run |

The `/vector-search` endpoint honours the workspace search mode, so the fix's probe measures the
reranked order. Retrieval figures in the report come from this probe over the 56 answerable
questions. The pilot's 0.842 comes from the sources of its `/chat` calls, which are the same top 4.

## How the fix was chosen

This is the order in which things ran (UTC times from the traces, 2026-11-23):

1. **07:02** Baseline, 3 trials × 67 questions.
2. **07:20** Reranker pilot: 1 trial on all 67 questions. Context doc recall@4 was **0.842**,
   against 0.562 for the baseline.
3. **07:25** Embedder probe at 8191-char chunks: recall@4 **0.750**. This candidate changed two
   things at once, the embedder and the chunk size.
4. Choice: the reranker. It had the higher recall at the same topN and prompt size, needs no
   re-ingest and no API spend, and keeps the corpus on the machine.
5. **07:28** Fix probe, then **07:29–07:50** the fix run, 3 trials × 67 questions.
6. **08:06** Embedder re-measured at the baseline's 1000-char chunks, so that only the embedder
   changes: recall@4 **0.792**. This came after the fix run. It confirms the choice (0.842 against
   0.792) but was not part of it.

**Limitation: there is no held-out set.** The fix was selected and measured on the same 67
questions, and the pilot was a 1-trial run on those questions. The before/after gain is
therefore in-sample.

## Trials, timeouts and the gate

- **3 trials per question per configuration** (201 per configuration). The model runs at the app's
  default temperature, so answers vary between trials. Repeated trials separate real changes from
  sampling noise.
- **Timeouts are not quality failures.** One `run` pass makes at most 2 billed attempts per trial
  (`--max-attempts 2`). It retries timeouts, 5xx errors and connection failures after a 10 s
  backoff. A trial whose attempts all time out is saved as `status: timeout` and excluded from the
  pass rate.
  - The fix run hit a provider stall, so trials still saved as timeouts got one explicit
    `run --retry-timeouts` pass, which adds up to 2 more billed attempts.
  - Some fix trials therefore carry **4 billed attempts**, and baseline trials up to 3. In all, 9
    baseline and 69 fix trials were retried, with 12 and 105 timed-out attempts.
  - Every attempt stays in the trace. The final numbers use 201 of 201 trials in both
    configurations.
- **regressgate** compares two configurations. It runs a permutation test on the pass rate (10,000
  shuffles) and a Fisher exact test per question, Tarone-corrected. Its report is
  [`results/compare-baseline-vs-fix.md`](../results/compare-baseline-vs-fix.md).
  - The permutation p-value has a floor of 1/10,001, so it is reported as **p < 0.001 (10,000
    shuffles)**, not as a measured 0.0001.
  - The test shuffles trials, so it rules out run-to-run noise for these questions. It says
    nothing about other questions.
  - The per-question tests cannot flag any question at 3 vs 3 trials. The smallest possible
    one-sided Fisher p is 0.05 (3/3 to 0/3), above the corrected threshold of 0.025. Per-question
    drops are therefore named, not tested.
- **Question-level statistics** ([`rag_audit/question_level_stats.py`](../rag_audit/question_level_stats.py))
  count each question once. `compare` writes them to `results/compare-baseline-vs-fix.json`
  under `question_level`.
  - An exact one-sided sign test on the questions whose pass rate changed.
  - A seeded percentile bootstrap (10,000 resamples of the 67 questions, seed 20251210) of the
    mean per-question gain.

## Disclosures

### Post-baseline changes to the grader and the cases

The grader and the case file were changed after reading the baseline traces, in two rounds. Every
change applies to both configurations, and both were re-graded from the saved traces with
`python -m rag_audit reproduce`. No answer was re-generated.

**Round 1, before the fix run.**
- Grader:
  - context support counts only chunks from the expected documents;
  - an answer whose lead declines is not a pass;
  - partial multi-hop recall is labelled a retrieval miss.
- A sentence-splitter fix ("U.S." ended a sentence), found in the fix traces. Its effect on
  baseline grades was not measured separately.
- Case changes:
  - `num-jan-iorb`, `num-sep-iorb`, `num-oct-iorb`, `num-dec-iorb`: the minutes were added as
    expected documents, because they also state the rate on reserve balances.
  - `fact-media-contact`: "(202) 452-2955" is accepted.
  - `fact-vice-chair`: any set of minutes is accepted as evidence, since every set lists the Vice
    Chair.
  - `mh-upper-bound-jul-oct`: the answer must also state the July upper bound, 4.5.

  The case file was first committed after these edits, so their before-state is not in git.

**Round 2, after the pre-release review.**

Grader rules:
- **Multi-hop support from every meeting.** When a case lists several expected-document groups,
  each group's chunks must hold at least one of the case's context facts. Pooled chunks no longer
  count. This changed no grade: every multi-hop trial that passed already had support from each
  meeting.
- **Refusals that go on to assert content.** A refusal passes only if it declines in the lead and
  states none of the case's forbidden specifics. It must also not guess anywhere in the answer
  (for example "it can be inferred", "was likely around", "best estimate").
- **List markers.** Numbered-list markers ("3. On December 10") are stripped from the answer
  before facts are matched.
- **Relabel.** A correct but unsupported answer is labelled `retrieval_miss` rather than
  `citation_mismatch` when an expected document was never retrieved.
- **Refusal phrasing.** Softened phrasings count as declining: "not directly provided", "do not
  explicitly state", "not specifically mentioned".
- **Zero denominators.** "1/0" is left as written instead of raising an error. This changed no
  grade.

Case changes, every one listed:

Phrase terms replace terms that matched by accident ("hold*" matched "holdings", "may" the modal
verb, "half" in "first half", and bare 3, 5 and 11 matched any number):
- `fact-oct-dissenters`: "half" became phrases such as "half a percentage"; "hold*" became phrases
  such as "hold the target range".
- `mh-goolsbee-oct-dec`: "hold*" became the same hold phrases.
- `mh-kashkari-alternate`: "may" became "may 2025", "may 6", "may 7", "may meeting", "in may" and
  "may fomc".
- `mh-miran-dissents`: "3" and "three" became count phrases ("three times", "three 2025",
  "three policy", …); "half" became phrases.
- `mh-cut-count`: "3" and "three" became "three times", "three cuts", "thrice" and similar.
- `svm-dec-vote-split`: "3" and "three" became "three members", "three voted", "three
  dissent*"; "half" became phrases.
- `mh-upper-bound-jul-oct`: "half" became phrases.
- `num-mar-treasury-cap`: "25" and "5" became "25 billion" and "5 billion".
- `mh-treasury-cap-path`: "25" and "5" became "25 billion" and "5 billion", in both
  `must_include` and `ctx_terms`.
- `num-jul-srf-peak`: "11" became "11 billion" ("11.1" kept).

`must_not_include` now names, for each out-of-corpus question, the invented specifics a fabricated
answer would state. Before, 9 of the 10 refusal cases had none.
- `ref-jan-2026-decision`: "minutes focus", "minutes emphasiz*", and "decided to …" / "… the
  target range" decision phrases.
- `ref-ecb-deposit-rate`, `ref-rbi-repo`: "percent", "basis point*".
- `ref-boe-nov-2025`: "percent", "bank rate", "voted", "majority".
- `ref-boj-level`: "0.75" and "percent" added to "0.5".
- `ref-mar-2024-vote`: "unanimous*", "all members", and "nine" to "twelve members".
- `ref-powell-successor`: the names Hassett, Warsh, Waller, Bowman, Jefferson, Miran and Williams.
- `ref-cpi-mar-2026`: "during 2026", "in early 2026", "cpi inflation was", "cpi rose", "cpi
  increased".
- `ref-sep-2026-dots`: "percent", "is contained in".
- `ref-bitcoin-dec-2025`: any "$" amount, "dollar*", "trading at", "traded at".
- `date-sep-minutes-published`: "published on" and "released on" added to "2026" and "11/23".

**Effect of round 2, trial by trial** (diff of `results/*/trial-grades.jsonl` against the
previous commit):
- **Baseline: 3 passes became fails.**
  - `ref-jan-2026-decision#3` describes the contents of "January 2026 meeting minutes".
  - `ref-cpi-mar-2026#2` says inflation stayed above target "during 2026".
  - `ref-sep-2026-dots#2` says the June 2026 projections "are contained in" the June 2025 minutes.

  Baseline refusal fell from 93.3% to 83.3%, and the overall rate from 46.3% to 44.8%.
- **Fix: no pass or fail changed.** Its 62.7% stands. 4 failure labels moved from
  `citation_mismatch` to `retrieval_miss`:
  - `date-jan-statement-release` ×3;
  - `mh-cut-count#2`, which says "two times" and had matched "3" only through its list marker.
- **Labels only:**
  - baseline `fact-media-contact` ×3 and pilot `date-jan-statement-release#1` moved to
    `retrieval_miss`;
  - `ref-sep-2026-dots#2` in the fix run is now detected as declining ("not directly provided"),
    but still fails, because it then guesses a figure.

The gain therefore grew from +16.4 to +17.9 pp. That is because the stricter refusal rule caught
fabrications only in baseline answers. One fix trial (`mh-cut-count#2`) would now be sent to the
judge but has no verdict, because no new LLM calls were made. It carries no `unsupported_claim`
flag.

### The judge, latency, cost and the index

- **The LLM judge is weak.** It agreed with manual labels on 11 of 18 spot-checked trials (61%) for
  each of three prompt versions ([`results/judge-spot-check.jsonl`](../results/judge-spot-check.jsonl)).
  - Answering "unsupported" every time would agree on 13 of 18 (72%).
  - The check is in-sample: baseline trials only, the same ones used while iterating on the prompt.
  - One v3 verdict says `supported: true` while listing an unsupported claim; it is counted as
    returned.
  - The judge only adds the `unsupported_claim` flag; pass and fail never depend on it.
- **Latency is indicative.** The provider stalled during the fix run: calls either finished in
  1–3 s or hung for minutes, so end-to-end tails are confounded. The report separates the app's own
  time (retrieval and reranking) from the LLM time the app records.
- **Cost is derived from the logs in `results/`, at OpenAI list prices.** gpt-4.1-mini costs
  $0.40 / $1.60 per million prompt / completion tokens. The provider's own prices were not
  verified.
  - Chat: $0.0880 (baseline) + $0.0327 (pilot) + $0.0996 (fix) = $0.2203, over 469 successful
    calls.
  - Judge: $0.1037 (baseline) + $0.0658 (fix) = $0.1695, over 328 calls, all prompt versions.
  - Total: **≈$0.39 logged**. Not metered: 117 timed-out attempts, whose usage is unknown although
    the provider may have billed them, and the embedding calls.
  - Rebuilt by `report/build-report-figures.py` into `report/report-numbers.json`
    (`audit_cost_logged`).
- **The index was rebuilt between runs.** The documents were re-embedded with the native embedder
  after the embedder test, so baseline and fix traces carry different chunk ids for the same text.
  Retrieval was checked to match the baseline's rankings on 67 of 67 questions after the rebuild.

## Where every number comes from

| Number | Source | Rebuilt by |
|---|---|---|
| Pass rates, failure labels, recall, latency, tokens | `results/<config>/summary.json`, `trial-grades.jsonl` | `python -m rag_audit reproduce` (all configurations) |
| regressgate verdict and per-question changes | `results/compare-baseline-vs-fix.{md,json}` | `python -m rag_audit reproduce` |
| Sign test and bootstrap CI | `results/compare-baseline-vs-fix.json` (`question_level`) | `python -m rag_audit reproduce` |
| Judge agreement and the constant baseline | `results/compare-baseline-vs-fix.json` (`judge_agreement`) | `python -m rag_audit reproduce` |
| Cover-page share of context, latency split, logged cost, charts | `report/report-numbers.json`, `report/figures/` | `python report/build-report-figures.py` |
| Trace excerpts quoted in the report | `results/<config>/traces.jsonl.gz` | read directly (gzip JSON lines) |
