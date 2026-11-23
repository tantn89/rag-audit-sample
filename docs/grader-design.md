# Grader design

Code: [`rag_audit/grading.py`](../rag_audit/grading.py) and
[`rag_audit/text_matching.py`](../rag_audit/text_matching.py); tests in [`tests/`](../tests/).

## Principle: deterministic first

Every pass or fail is decided by string and number checks that give the same result on every run.
An LLM is used only to label *why* a failed answer went wrong, never to decide whether it passed.

## Normalisation

- Fractions and percentages become decimals: "4-1/4", "4- 1/4" (as the PDF text extracts it),
  "4¼", "4.25%" and "¼ point" all compare equal by value.
- Ranges are detected in any form ("3½ to 3¾", "3.50%-3.75%").
- Text terms match on word boundaries, so "december 1" does not match "december 10". Terms that
  are common words or small numbers are written as phrases ("5 billion", "three times", "may
  2025", "half a percentage") so they cannot match by accident ("holdings", the modal "may",
  "first half", a stray "3").
- Numbered-list markers ("3. On December 10") are stripped from answers before facts are matched.
- "1/0" and other zero denominators are left as written.
- The `<document_metadata>` header that AnythingLLM prepends to each chunk is stripped before the
  context is checked.

## Pass rules

**Answerable question.** All four must hold:
1. every `must_include` group and every range is in the answer;
2. no `must_not_include` term appears;
3. the first two sentences do not decline (unless `allow_caveat` is set); the sentence splitter is
   abbreviation-safe ("U.S.", "Mr.");
4. the facts are also present in the retrieved chunks **from the expected documents**, so the answer
   can be verified from the app's own citations. A matching number from another meeting's chunk does
   not count. A multi-hop question lists one expected-document group per meeting, and the chunks
   of **every** group must hold at least one of its facts, so one meeting's evidence cannot
   stand in for another's.

**Refusal question.** All three must hold:
1. the first two sentences decline (including softened forms such as "not directly provided");
2. no `must_not_include` term appears; each refusal case lists the invented specifics a
   fabricated answer would state, such as a rate, a name or "2026 minutes" content;
3. the answer does not go on to guess anywhere ("it can be inferred", "was likely around", "best
   estimate"). Declining first and inventing afterwards is still an answer.

## Failure taxonomy

One primary label per failed trial, assigned in pipeline order:

| Label | Meaning |
|---|---|
| `format_tool_error` | error, empty answer or non-text response |
| `missing_refusal` | an out-of-corpus question was answered |
| `citation_mismatch` | the answer is right and every expected document was retrieved, but its chunks do not support the answer |
| `retrieval_miss` | an expected document (for multi-hop, any of them) never reached the model, whether or not the answer happened to be right |
| `wrong_chunk_first` | the expected documents arrived, but not the chunk with the facts |
| `unsupported_claim` | the context had the facts and the judge says the answer asserts something else |
| `answer_incomplete` | the context had the facts and the answer left them out or declined |

Flags can be added on top: `unsupported_claim` (judge verdict on any failed, non-declining answer),
`refused`, `forbidden_term` and `refused_then_asserted` (a refusal case that declined in the
lead, then stated invented specifics or guessed). The report's failure chart counts primary labels plus the
`unsupported_claim` flag. Timeouts get `latency_timeout` and are excluded from quality numbers.

## Retrieval metrics

- **recall@k**: the share of expected-document groups with at least one chunk in the top k of the
  `/vector-search` probe (k = 1, 4, 10), averaged over the 56 answerable questions. Recall@4 is what
  the model sees at the default topN.
- **MRR@10**: the mean reciprocal rank of the first chunk from any expected document.
- **Context doc recall**: the same group recall over the sources the `/chat` call actually used.

## The groundedness judge

- [`rag_audit/groundedness_judge.py`](../rag_audit/groundedness_judge.py): the same guarded
  gpt-4.1-mini at temperature 0, given the question, the answer and the retrieved chunks labelled by
  meeting, and asked whether each claim is supported by a cited chunk from the meeting the answer
  names.
- It runs only on failed answerable trials that did not decline, and on refusal questions that were
  answered (71 verdicts for the baseline, 56 for the fix). Verdicts are cached in
  `results/<config>/judge-verdicts.jsonl`. The cache key hashes everything the judge saw (question,
  answer, cited chunk titles and text) plus the prompt version, so re-grading costs nothing and a
  changed trace never reuses a stale verdict.
- **Agreement:** a manual spot check of every 4th baseline verdict (18 trials, labels fixed before
  seeing later prompt versions) agreed with the judge on 11 of 18 (61%) for each of three prompt
  versions.
  - Always answering "unsupported" would agree on 13 of 18 (72%), so the judge adds no measurable
    information.
  - The check is in-sample: baseline only, and the same trials were used while iterating on the
    prompt.
  - Most disagreements are the judge accepting a fact that is true of a different meeting.
  - The `unsupported_claim` counts are indicative only. A stronger judge is item 5 of the report's
    fix plan.
- `judge <config> --budget-usd N` stops before the estimated total spend (chat plus judge) would
  pass N. It exits nonzero when the budget or a failed call leaves any verdict missing, and
  `full-rerun` then stops without publishing results.

## Tests

`python -m pytest -q` runs 98 tests and 32 subtests:
- normalisation, ranges, list markers and zero denominators;
- refusal and speculation detection, and the abbreviation-safe lead;
- the published phrase terms against incidental matches;
- recall@k, failure labels, multi-hop support per meeting and the caveat rule;
- case validation and the corpus manifest pins;
- the runner paths: judge budget and exit codes, `full-rerun` not publishing, `reproduce` with
  retrieval-only bundles, re-embed settings, the judge cache key and the question-level
  statistics.
