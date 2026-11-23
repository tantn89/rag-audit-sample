"""The eval set (evals/cases.jsonl): one question per line with deterministic grading fields.

    {"id": "fact-sep-range", "category": "fact",
     "question": "What target range did the FOMC set at its September 2025 meeting?",
     "must_include": [["1/4", "25 basis"]],      # groups: every group needs one alternative
     "ranges": [[4, 4.25]],                     # "4 to 4-1/4 percent" in any spelling
     "must_not_include": ["2026"],              # any hit fails the answer
     "ctx_terms": [["4.25"]],                   # what the retrieved chunks must contain
     "must_refuse": false,                      # out-of-corpus questions set this to true
     "allow_caveat": false,                     # true: "...; later moves cannot be confirmed" is fine
     "expected_docs": [["fomc-statement-20250917.pdf", "fomc-minutes-20250917.pdf"]],
     "reference_answer": "...", "evidence": [{"doc": "...", "page": 1, "quote": "..."}]}

`expected_docs` is a list of groups as well: a multi-hop question needs one document from
each group, and recall@k is the share of groups retrieved within k.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("fact", "numeric", "multi_hop", "statement_vs_minutes", "date_sensitive", "refusal")
DEFAULT_CASES = Path(__file__).resolve().parent.parent / "evals" / "cases.jsonl"


class CaseError(ValueError):
    """A case file line that does not match the schema; the message names the line."""


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    question: str
    must_include: list[list[str]] = field(default_factory=list)
    ranges: list[list[float]] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    ctx_terms: list[list[str]] | None = None
    must_refuse: bool = False
    allow_caveat: bool = False  # a declining phrase in the lead does not void a correct answer
    expected_docs: list[list[str]] = field(default_factory=list)
    reference_answer: str = ""
    evidence: list[dict] = field(default_factory=list)

    @property
    def context_terms(self) -> list[list[str]]:
        """Facts the retrieved context must hold for the answer to count as cited."""
        return self.must_include if self.ctx_terms is None else self.ctx_terms

    @property
    def context_ranges(self) -> list[list[float]]:
        return self.ranges if self.ctx_terms is None else []


def _check(case: EvalCase, where: str) -> None:
    if case.category not in CATEGORIES:
        raise CaseError(f"{where}: unknown category {case.category!r}")
    if not case.question.strip():
        raise CaseError(f"{where}: empty question")
    if case.must_refuse:
        if case.must_include or case.ranges:
            raise CaseError(f"{where}: a refusal case cannot also require facts")
    elif not (case.must_include or case.ranges):
        raise CaseError(f"{where}: an answerable case needs must_include or ranges")
    elif not case.expected_docs:
        raise CaseError(f"{where}: an answerable case needs expected_docs")
    for group in case.must_include + (case.ctx_terms or []) + case.expected_docs:
        if not isinstance(group, list) or not group or not all(isinstance(x, str) for x in group):
            raise CaseError(f"{where}: groups must be non-empty lists of strings")
    for pair in case.ranges:
        if len(pair) != 2 or not all(isinstance(x, (int, float)) for x in pair):
            raise CaseError(f"{where}: ranges must be [low, high] numbers")


def load_cases(path: Path = DEFAULT_CASES) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name} line {lineno}"
        try:
            case = EvalCase(**json.loads(line))
        except (ValueError, TypeError) as err:
            raise CaseError(f"{where}: {err}") from None
        _check(case, where)
        if case.id in seen:
            raise CaseError(f"{where}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise CaseError(f"{path}: no cases")
    return cases
