"""Trial grading, failure labels and retrieval metrics."""

import json

import pytest

from rag_audit.eval_cases import CaseError, EvalCase, load_cases
from rag_audit.grading import first_hit_rank, grade_trial, recall_at_k

SEP_ST, SEP_MIN, JUL_ST = ("fomc-statement-20250917.pdf", "fomc-minutes-20250917.pdf",
                           "fomc-statement-20250730.pdf")
SEP_CHUNK = ("<document_metadata>\nsourceDocument: fomc-statement-20250917.pdf\npublished: "
             "11/23/2026\n</document_metadata>\n\ndecided to lower the target range for the federal "
             "funds rate by 1/4 percentage point to 4 to\n4-  1/4 percent.")

ANSWERABLE = EvalCase(id="sep", category="fact", question="Sep range?", ranges=[[4, 4.25]],
                      must_not_include=["2026"], expected_docs=[[SEP_ST, SEP_MIN]])
REFUSAL = EvalCase(id="ecb", category="refusal", question="ECB rate?", must_refuse=True,
                   must_not_include=["2.0"])


def trace(answer, sources=(), **extra):
    return {"trial": 1, "status": "ok", "answer": answer, "response_type": "textResponse",
            "sources": [{"title": d, "score": s, "text": t} for d, s, t in sources], **extra}


def test_correct_cited_answer_passes():
    g = grade_trial(ANSWERABLE, trace("The range is 4% to 4.25%.", [(SEP_ST, 0.7, SEP_CHUNK)]))
    assert g.passed and g.failure is None and g.doc_recall == 1.0


def test_correct_answer_without_supporting_chunk_is_citation_mismatch():
    g = grade_trial(ANSWERABLE, trace("The range is 4 to 4-1/4 percent.", [(JUL_ST, 0.7, "held")]))
    assert not g.passed and g.failure == "citation_mismatch"


def test_missing_expected_document_is_retrieval_miss():
    g = grade_trial(ANSWERABLE, trace("They cut by a quarter point; the level is not specified.",
                                      [(JUL_ST, 0.8, "maintain 4-1/4 to 4-1/2 percent")]))
    assert g.failure == "retrieval_miss" and "refused" in g.flags and not g.needs_judge


def test_facts_in_another_meetings_chunk_do_not_count_as_support():
    case = EvalCase(id="absent", category="fact", question="Who was absent in July?",
                    must_include=[["kugler"]], expected_docs=[[JUL_ST]])
    other = ("fomc-minutes-20250319.pdf", 0.5, "Attendance: Adriana D. Kugler, Alberto G. Musalem")
    g = grade_trial(case, trace("The documents do not say who was absent.", [other]))
    assert not g.context_supported and g.failure == "retrieval_miss"


def test_right_document_wrong_chunk():
    g = grade_trial(ANSWERABLE, trace("The range was 4.25 to 4.5 percent.",
                                      [(SEP_MIN, 0.8, "participants discussed risks")]))
    assert g.failure == "wrong_chunk_first" and g.needs_judge


def test_context_had_facts_but_answer_wrong_uses_judge_verdict():
    t = trace("The range was 4.25 to 4.5 percent.", [(SEP_ST, 0.8, SEP_CHUNK)])
    assert grade_trial(ANSWERABLE, t).failure == "answer_incomplete"
    judged = grade_trial(ANSWERABLE, t, {"supported": False, "unsupported_claims": ["4.25-4.5"]})
    assert judged.failure == "unsupported_claim"


def test_forbidden_ingestion_date_fails_an_otherwise_correct_answer():
    g = grade_trial(ANSWERABLE, trace("As of 11/23/2026 the range is 4 to 4.25 percent.",
                                      [(SEP_ST, 0.8, SEP_CHUNK)]))
    assert not g.passed and g.forbidden_hits == ["2026"]


def test_refusal_case():
    ok = grade_trial(REFUSAL, trace("The documents do not specify the ECB's deposit rate."))
    assert ok.passed
    answered = grade_trial(REFUSAL, trace("The ECB set its deposit facility rate at 2.0 percent."))
    assert answered.failure == "missing_refusal" and answered.needs_judge
    hedged = grade_trial(REFUSAL, trace("The documents do not say. It was likely 2.0 percent."))
    assert hedged.failure == "missing_refusal" and not hedged.needs_judge


def test_answer_that_opens_by_declining_is_not_correct_unless_caveats_are_allowed():
    case = EvalCase(id="pce", category="numeric", question="Core PCE in March?",
                    must_include=[["2.6"]], expected_docs=[[SEP_MIN]])
    chunk = [(SEP_MIN, 0.8, "core PCE price inflation was 2.6 percent in March")]
    declined = "The minutes do not give March's figure. May's core PCE was 2.6 percent."
    assert grade_trial(case, trace(declined, chunk)).facts_ok is False
    caveated = EvalCase(**{**case.__dict__, "allow_caveat": True})
    assert grade_trial(caveated, trace(declined, chunk)).passed


def test_partial_multi_hop_retrieval_is_a_retrieval_miss():
    case = EvalCase(id="miran", category="multi_hop", question="How many Miran dissents?",
                    must_include=[["three"]], ctx_terms=[["miran"]],
                    expected_docs=[[SEP_ST], ["fomc-statement-20251210.pdf"]])
    g = grade_trial(case, trace("He dissented twice.", [(SEP_ST, 0.6, "Stephen I. Miran")]))
    assert g.context_supported and g.doc_recall == 0.5 and g.failure == "retrieval_miss"


def test_timeouts_are_excluded_and_errors_are_format_failures():
    timeout = grade_trial(ANSWERABLE, {"trial": 2, "status": "timeout"})
    assert timeout.passed is None and timeout.failure == "latency_timeout"
    empty = grade_trial(ANSWERABLE, trace("  "))
    assert empty.passed is False and empty.failure == "format_tool_error"


def test_recall_at_k_over_document_groups():
    ranked = [JUL_ST, SEP_MIN, "x.pdf", SEP_ST]
    groups = [[SEP_ST, SEP_MIN], [JUL_ST]]
    assert recall_at_k(ranked, groups, 1) == 0.5
    assert recall_at_k(ranked, groups, 2) == 1.0
    assert recall_at_k(ranked, [[SEP_ST]], 3) == 0.0
    assert recall_at_k(ranked, [], 3) is None
    assert first_hit_rank(ranked, [[SEP_ST]]) == 4
    assert first_hit_rank(ranked, [["absent.pdf"]]) is None


def test_case_loader_rejects_invalid_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "a", "category": "fact", "question": "q?"}) + "\n")
    with pytest.raises(CaseError, match="needs must_include"):
        load_cases(path)
    path.write_text(json.dumps({"id": "a", "category": "refusal", "question": "q?",
                                "must_refuse": True, "must_include": [["x"]]}) + "\n")
    with pytest.raises(CaseError, match="cannot also require"):
        load_cases(path)


def test_published_eval_set_loads():
    cases = load_cases()
    assert len(cases) >= 60
    assert sum(c.must_refuse for c in cases) >= 8
    assert {c.category for c in cases} >= {"fact", "numeric", "multi_hop", "refusal",
                                           "statement_vs_minutes", "date_sensitive"}
