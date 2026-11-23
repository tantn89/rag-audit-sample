"""Grade one trial deterministically and label why it failed.

Pass rules
- Answerable case: every `must_include` group and `ranges` pair is in the answer (numbered-list
  markers are ignored), nothing from `must_not_include` is, and the retrieved chunks (the app's
  citations) from the expected documents contain the case's context facts, so a reader could
  verify the answer from what the app showed. When `expected_docs` has several groups (one per
  meeting a multi-hop question needs), the chunks of every group must also hold at least one of
  those facts: one meeting's evidence cannot stand in for the others.
- Refusal case: the answer declines within its first two sentences, states nothing from
  `must_not_include` (the invented specifics a fabricated answer would give) and does not go on
  to guess (SPECULATION_PATTERNS, anywhere in the answer).

Failure taxonomy (one primary label per failed trial, in pipeline order, plus flags):
  format_tool_error   the app returned an error, an empty answer or a non-text response
  missing_refusal     an out-of-corpus question was answered instead of declined
  citation_mismatch   the answer is right and every expected document was retrieved, but the
                      retrieved chunks do not support it
  retrieval_miss      an expected document (for multi-hop: any of them) is absent from the
                      retrieved chunks, whether or not the answer happened to be right
  wrong_chunk_first   the expected documents were retrieved, but not the chunk with the facts
  unsupported_claim   the context had the facts; the answer asserts something else (judge)
  answer_incomplete   the context had the facts; the answer left them out or refused
An answer whose first two sentences decline is not counted as correct unless the case sets
`allow_caveat` ("what is the range right now?" should add that later moves are unknown). A judge verdict
of "unsupported" on a retrieval failure is kept as the `unsupported_claim` flag.
Latency timeouts are not quality failures: `status="timeout"` trials are reported apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rag_audit.eval_cases import EvalCase
from rag_audit.text_matching import (
    group_in,
    is_refusal,
    normalize,
    range_in,
    speculation_in,
    strip_chunk_metadata,
    strip_list_markers,
    term_in,
)

FAILURE_LABELS = ("format_tool_error", "missing_refusal", "citation_mismatch", "retrieval_miss",
                  "wrong_chunk_first", "unsupported_claim", "answer_incomplete")


@dataclass
class TrialGrade:
    case_id: str
    trial: int
    status: str  # ok | timeout | error
    passed: bool | None  # None for timeouts: excluded from quality numbers
    failure: str | None = None
    flags: list[str] = field(default_factory=list)
    refused: bool = False
    facts_ok: bool = False
    missing: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    context_supported: bool = False
    doc_recall: float | None = None
    needs_judge: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def source_doc(source: dict) -> str:
    return (source.get("title") or "").strip()


def ranked_sources(trace: dict) -> list[dict]:
    return sorted(trace.get("sources") or [], key=lambda s: s.get("score") or 0, reverse=True)


def doc_group_recall(groups: list[list[str]], docs: list[str]) -> float | None:
    """Share of expected-document groups with at least one document among `docs`."""
    if not groups:
        return None
    found = set(docs)
    return sum(any(d in found for d in g) for g in groups) / len(groups)


def missing_facts(norm: str, groups: list[list[str]], ranges: list[list[float]]) -> list[str]:
    missing = [" | ".join(g) for g in groups if not group_in(norm, g)]
    missing += [f"{lo}-{hi}" for lo, hi in ranges if not range_in(norm, lo, hi)]
    return missing


def _context(sources: list[dict], docs: set[str]) -> str:
    return normalize(" ".join(strip_chunk_metadata(s.get("text")) for s in sources
                              if source_doc(s) in docs))


def context_support(case: EvalCase, sources: list[dict]) -> bool:
    """The case's context facts are in the chunks from its expected documents, and, for a
    question that needs several meetings, each meeting's chunks hold at least one of them."""
    wanted = {d for group in case.expected_docs for d in group}
    if missing_facts(_context(sources, wanted), case.context_terms, case.context_ranges):
        return False
    if len(case.expected_docs) < 2:
        return True
    for group in case.expected_docs:
        ctx = _context(sources, set(group))
        if not (any(group_in(ctx, g) for g in case.context_terms)
                or any(range_in(ctx, lo, hi) for lo, hi in case.context_ranges)):
            return False
    return True


def grade_trial(case: EvalCase, trace: dict, judge: dict | None = None) -> TrialGrade:
    """`judge` is the cached groundedness verdict for this trial, if one was requested."""
    g = TrialGrade(case.id, int(trace.get("trial", 0)), trace.get("status", "ok"), None)
    if g.status == "timeout":
        g.failure = "latency_timeout"
        return g
    answer = trace.get("answer") or ""
    if g.status != "ok" or trace.get("error") or not answer.strip() \
            or trace.get("response_type") not in (None, "textResponse"):
        g.status, g.passed, g.failure = "error", False, "format_tool_error"
        return g

    norm = normalize(strip_list_markers(answer))
    sources = ranked_sources(trace)
    g.refused = is_refusal(answer)
    g.forbidden_hits = [t for t in case.must_not_include if term_in(norm, t)]
    g.doc_recall = doc_group_recall(case.expected_docs, [source_doc(s) for s in sources])

    if case.must_refuse:
        # Declining in the lead is not enough: going on to assert invented specifics or to guess
        # an estimate is still an answer to a question the documents cannot answer.
        g.forbidden_hits += [f"speculation: {s}" for s in speculation_in(answer)]
        g.passed = g.refused and not g.forbidden_hits
        if not g.passed:
            g.failure = "missing_refusal"
            g.needs_judge = not g.refused
            if g.refused:
                g.flags.append("refused_then_asserted")
        return g

    g.missing = missing_facts(norm, case.must_include, case.ranges)
    # An answer that opens by declining is not a correct answer, even if a matching number
    # appears later (e.g. "the May minutes do not give March's figure; May's was 2.6 percent").
    g.facts_ok = not g.missing and not g.forbidden_hits and (not g.refused or case.allow_caveat)
    # Support must come from the expected documents: the same name or number in another
    # meeting's chunk (an attendance list, an older range) does not back the answer.
    g.context_supported = context_support(case, sources)
    g.passed = g.facts_ok and g.context_supported
    if g.passed:
        return g
    if g.facts_ok:
        # A right answer without support is a citation problem only when every expected
        # document reached the model; otherwise retrieval is the cause.
        g.failure = "citation_mismatch" if (g.doc_recall or 0) >= 1 else "retrieval_miss"
        return g

    g.needs_judge = not g.refused
    unsupported = bool(judge) and judge.get("supported") is False
    if unsupported:
        g.flags.append("unsupported_claim")
    if g.refused:
        g.flags.append("refused")
    if g.forbidden_hits:
        g.flags.append("forbidden_term")
    if (g.doc_recall or 0) < 1:  # an expected document never reached the model
        g.failure = "retrieval_miss"
    elif not g.context_supported:
        g.failure = "wrong_chunk_first"
    else:
        g.failure = "unsupported_claim" if unsupported else "answer_incomplete"
    return g


def recall_at_k(ranked_docs: list[str], groups: list[list[str]], k: int) -> float | None:
    """recall@k over expected-document groups for one ranked retrieval list."""
    return doc_group_recall(groups, ranked_docs[:k])


def first_hit_rank(ranked_docs: list[str], groups: list[list[str]]) -> int | None:
    """1-based rank of the first chunk from any expected document (None when absent)."""
    wanted = {d for g in groups for d in g}
    return next((i for i, d in enumerate(ranked_docs, start=1) if d in wanted), None)
