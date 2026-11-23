# Eval set design

[`evals/cases.jsonl`](../evals/cases.jsonl) holds 67 questions, one JSON object per line. The
schema is documented in [`rag_audit/eval_cases.py`](../rag_audit/eval_cases.py) and checked on load.

## Composition

| Category | Questions | What it probes | Why it matters for this corpus |
|---|---|---|---|
| `fact` | 16 | target ranges, dissenters, absent member, officers, next-meeting date, media contact | eight meetings use nearly identical language, so the meeting named in the question must drive retrieval |
| `numeric` | 16 | IORB, ON RRP and primary credit rates, balance-sheet caps, PCE and unemployment figures, reserves, SRF take-up | numbers are written as "4-1/4", "4¼" or "4.25 percent"; a grader must compare values, not strings |
| `multi_hop` | 11 | number and total of cuts in 2025, dissent histories, IORB and cap paths, unemployment trend | needs chunks from several meetings at once; topN 4 caps what is possible |
| `statement_vs_minutes` | 7 | facts only in the minutes: dissent reasons, vote splits, framework review | the short statement ranks high but lacks the answer |
| `date_sensitive` | 7 | release dates, "latest decision", "right now", minutes publication date, effective dates | the app injects today's date and an upload timestamp into the context |
| `refusal` | 10 | January 2026 decision, other central banks, a March 2024 vote, a Powell successor, 2026 data, Bitcoin | the corpus cannot answer them; the app must decline |

11 questions must be refused: the 10 refusal questions and one date-sensitive question (the
minutes' publication date is not stated in the minutes themselves).

## Fields per question

- `must_include`: groups of alternatives; every group needs one match ("1/4" or "25 basis"). Small
  numbers and common words are written as phrases ("5 billion", "three times", "may 2025").
- `ranges`: rate ranges as `[low, high]`, matched in any spelling ("3½ to 3¾", "3.50%-3.75%").
- `must_not_include`: values that fail the answer on sight, such as "2026" or the upload date.
  Every must-refuse question lists the invented specifics a fabricated answer would give (a rate,
  a vote count, a name, a dollar amount, "2026 minutes" content).
- `ctx_terms`: optional; the facts the retrieved chunks must hold when they differ from the
  answer's wording. Defaults to `must_include` and `ranges`.
- `expected_docs`: groups of documents that hold the answer. A multi-hop question has one group
  per meeting it needs; recall@k is the share of groups retrieved within k.
- `allow_caveat`: set on 2 "now/latest" questions, where opening with "as of the latest documents…"
  is the right behaviour.
- `reference_answer` and `evidence` (`doc`, `page`, `quote`): the human-readable answer key.

## Evidence check

Every evidence quote was taken from the PDF text and is verified on its stated page:

```powershell
python -m rag_audit validate-cases --corpus-dir data\corpus
```

All quotes are found (`data\corpus` is filled by `python scripts\download-corpus.py`).

## How the set was built, and its limits

- Questions were written by the auditor from the documents, not taken from real user logs. In a
  client engagement the set starts from the client's real questions and expected answers.
- Categories are sized for the failure modes the corpus invites (many similar meetings, numbers in
  several spellings, dates), not for user-traffic frequency. The overall pass rate is a mean per
  question across categories, so read it next to the per-category table.
- Every case change made after reading the baseline traces is listed by case id in
  [methodology.md](methodology.md#post-baseline-changes-to-the-grader-and-the-cases).
- The same 67 questions were used to choose the fix and to measure it; there is no held-out set.
