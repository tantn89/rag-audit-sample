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
- Text terms match on word boundaries, so "december 1" does not match "december 10".
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
   not count.

**Refusal question.** The first two sentences decline and no forbidden value appears.

## Failure taxonomy

One primary label per failed trial, assigned in pipeline order:

| Label | Meaning |
|---|---|
| `format_tool_error` | error, empty answer or non-text response |
| `missing_refusal` | an out-of-corpus question was answered |
| `citation_mismatch` | the answer is right, but no retrieved chunk from the expected documents supports it |
| `retrieval_miss` | an expected document (for multi-hop, any of them) never reached the model |
| `wrong_chunk_first` | the expected documents arrived, but not the chunk with the facts |
| `unsupported_claim` | the context had the facts and the judge says the answer asserts something else |
| `answer_incomplete` | the context had the facts and the answer left them out or declined |

Flags can be added on top: `unsupported_claim` (judge verdict on any failed, non-declining answer),
`refused` and `forbidden_term`. The report's failure chart counts primary labels plus the
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
  `results/<config>/judge-verdicts.jsonl`, keyed by the answer text, so re-grading costs nothing.
- **Agreement:** a manual spot check of every 4th baseline verdict (18 trials, labels fixed before
  seeing later prompt versions) agreed with the judge on 11 of 18 (61%) for each of three prompt
  versions. Most disagreements are the judge accepting a fact that is true of a different meeting.
  The `unsupported_claim` counts are indicative only; a stronger judge is item 5 of the report's
  fix plan.

## Tests

`python -m pytest -q` runs 46 tests and 32 subtests: normalisation, ranges, refusal detection, the
abbreviation-safe lead, recall@k, failure labels, the caveat rule, case validation and the corpus
manifest pins.
