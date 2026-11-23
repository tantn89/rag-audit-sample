"""Runner paths that spend money or publish results: judge budget and exit codes, full-rerun
publishing, reproduce with retrieval-only bundles, re-embed settings, the judge cache key and
the question-level statistics."""

import argparse
import json
import math

import pytest

import rag_audit.__main__ as cli
import rag_audit.groundedness_judge as judge_mod
from rag_audit.groundedness_judge import affordable_calls, verdict_key
from rag_audit.question_level_stats import bootstrap_ci, question_level, sign_test

# A wrong answer to a published case: grading sends it to the judge.
WRONG = {"case_id": "fact-sep-range", "trial": 1, "status": "ok", "response_type": "textResponse",
         "question": "What target range for the federal funds rate did the FOMC set at its "
                     "September 2025 meeting?",
         "answer": "It held the range at 4.25 to 4.5 percent.",
         "sources": [{"title": "fomc-statement-20250730.pdf", "score": 0.7, "text": "held"}]}


@pytest.fixture
def isolated_results(tmp_path, monkeypatch):
    """Judge verdicts go to a temp results dir; traces come from WRONG only."""
    monkeypatch.setattr(judge_mod, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "load_traces", lambda config, from_bundle=False: ([dict(WRONG)], {}))
    return tmp_path


def _ok_verdict(trace):
    return {"supported": False, "unsupported_claims": ["4.25-4.5"], "judge_version": 3,
            "model": "test", "prompt_tokens": 1000, "completion_tokens": 100}


def test_judge_saves_verdicts_and_exits_zero(isolated_results):
    assert cli.judge_missing("baseline", _ok_verdict, budget_usd=3, chat_spent_usd=0) == 0
    rows = (isolated_results / "baseline" / "judge-verdicts.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["key"] == verdict_key(WRONG)
    # cached now: a second run has nothing to do and makes no call
    assert cli.judge_missing("baseline", lambda t: pytest.fail("called"), budget_usd=3,
                             chat_spent_usd=0) == 0


def test_judge_exits_nonzero_when_a_verdict_is_missing(isolated_results):
    def failing(trace):
        raise RuntimeError("HTTP 502")
    assert cli.judge_missing("baseline", failing, budget_usd=3,
                             chat_spent_usd=0) == cli.EXIT_INCOMPLETE


def test_judge_respects_the_budget(isolated_results):
    code = cli.judge_missing("baseline", lambda t: pytest.fail("over budget"), budget_usd=1.0,
                             chat_spent_usd=0.9999)
    assert code == cli.EXIT_BUDGET
    assert not (isolated_results / "baseline" / "judge-verdicts.jsonl").exists()


def test_affordable_calls():
    assert affordable_calls(10, spent_usd=2.99, per_call_usd=0.002, budget_usd=3.0) == 5
    assert affordable_calls(10, spent_usd=3.5, per_call_usd=0.002, budget_usd=3.0) == 0
    assert affordable_calls(4, spent_usd=0, per_call_usd=0.002, budget_usd=3.0) == 4


def _rerun_args(**extra):
    return argparse.Namespace(configs=["baseline", "fix"], trials=3, timeout=60, max_attempts=2,
                              concurrency=1, budget_usd=3.0, **extra)


def test_full_rerun_passes_the_budget_to_the_judge_and_does_not_publish_on_failure(monkeypatch):
    seen, published = [], []
    for name in ("cmd_apply_config", "cmd_retrieval", "cmd_run", "cmd_pack"):
        monkeypatch.setattr(cli, name, lambda ns: 0)

    def judge(ns):
        seen.append(ns.budget_usd)
        return cli.EXIT_INCOMPLETE
    monkeypatch.setattr(cli, "cmd_judge", judge)
    monkeypatch.setattr(cli, "report", lambda *a, **k: published.append(a))
    monkeypatch.setattr(cli, "compare", lambda *a: published.append(a))
    assert cli.cmd_full_rerun(_rerun_args()) == cli.EXIT_INCOMPLETE
    assert seen == [3.0] and published == []


def test_reproduce_compares_chat_configs_only(monkeypatch):
    compared = []
    monkeypatch.setattr(cli, "report", lambda name, from_bundle: (
        {"config": name} if "embedder" in name else {"config": name, "pass_rate": 0.5}))
    monkeypatch.setattr(cli, "compare", lambda base, head: compared.append((base, head)) or 0)
    ns = argparse.Namespace(configs=["baseline", "candidate-openai-embedder-1000"])
    assert cli.cmd_reproduce(ns) == 0 and compared == []
    ns = argparse.Namespace(configs=list(cli.REPRODUCE_ORDER))
    assert cli.cmd_reproduce(ns) == 0 and compared == [("baseline", "fix")]


class FakeClient:
    """In-memory AnythingLLM workspace: delete and recreate reset settings to app defaults."""

    DEFAULTS = {"slug": "fomc-2025", "topN": 4, "vectorSearchMode": "default"}

    def __init__(self, settings):
        self.ws = {**self.DEFAULTS, **settings, "documents": [{"docpath": "b.json"},
                                                              {"docpath": "a.json"}]}

    def workspace(self, slug):
        return self.ws

    def delete_workspace(self, slug):
        self.ws = None

    def ensure_workspace(self, slug):
        self.ws = {**self.DEFAULTS, "documents": []}
        return self.ws

    def update_workspace(self, slug, settings):
        self.ws.update(settings)
        return self.ws

    def update_embeddings(self, slug, adds, deletes):
        self.ws["documents"] = [{"docpath": p} for p in adds]
        return {"workspace": self.ws}


def test_reembed_restores_the_workspace_settings(tmp_path):
    client = FakeClient({"vectorSearchMode": "rerank", "topN": 4, "similarityThreshold": 0.25})
    (tmp_path / "old.json").write_text("{}")
    assert cli.reembed(client, tmp_path) == 0
    assert client.ws["vectorSearchMode"] == "rerank" and client.ws["similarityThreshold"] == 0.25
    assert sorted(d["docpath"] for d in client.ws["documents"]) == ["a.json", "b.json"]
    assert not list(tmp_path.glob("*.json"))


def test_verdict_key_covers_every_judge_input():
    base = verdict_key(WRONG)
    assert base.startswith("fact-sep-range#1#") and verdict_key(dict(WRONG)) == base
    changed = [
        {**WRONG, "question": "Another question?"},
        {**WRONG, "answer": "Another answer."},
        {**WRONG, "sources": [{**WRONG["sources"][0], "text": "other evidence"}]},
        {**WRONG, "sources": [{**WRONG["sources"][0], "title": "fomc-minutes-20250917.pdf"}]},
    ]
    assert all(verdict_key(t) != base for t in changed)
    assert verdict_key(WRONG, version=2) != base


def test_sign_test_is_exact():
    st = sign_test([1.0] * 12 + [-0.5] + [0.0] * 54)
    assert (st["up"], st["down"], st["unchanged"]) == (12, 1, 54)
    assert math.isclose(st["p_one_sided"], 14 / 2 ** 13)
    assert sign_test([0.0, 0.0])["p_one_sided"] == 1.0


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    deltas = [1.0, 0.0, 0.0, 1 / 3, -2 / 3, 0.0, 1.0, 0.0]
    a, b = bootstrap_ci(deltas, resamples=2000), bootstrap_ci(deltas, resamples=2000)
    assert a == b and a["low"] <= a["mean"] <= a["high"]
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_question_level_uses_questions_in_both_runs():
    base = {"q1": [False] * 3, "q2": [True, True, False], "q3": [True] * 3}
    head = {"q1": [True] * 3, "q2": [False] * 3, "q4": [True] * 3}
    ql = question_level(base, head)
    assert ql["questions"] == 2 and ql["dropped"] == ["q2"]
    assert (ql["sign_test"]["up"], ql["sign_test"]["down"]) == (1, 1)
