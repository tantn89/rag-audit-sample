"""Turn stored traces into graded results: summary JSON, regressgate report, judge agreement."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from rag_audit.audit_configs import get_config
from rag_audit.eval_cases import CATEGORIES, EvalCase
from rag_audit.grading import TrialGrade, first_hit_rank, grade_trial, recall_at_k
from rag_audit.groundedness_judge import answer_key, judge_cost, load_verdicts
from rag_audit.trace_store import RESULTS, trace_cost


def grade_all(cases: list[EvalCase], chats: list[dict], config: str) -> list[tuple[dict, TrialGrade]]:
    by_id = {c.id: c for c in cases}
    verdicts = load_verdicts(config)
    graded = []
    for t in sorted(chats, key=lambda t: (t["case_id"], t["trial"])):
        case = by_id.get(t["case_id"])
        if case is None:
            continue  # trace for a case that was removed from the eval set
        verdict = verdicts.get(answer_key(case.id, t["trial"], t.get("answer") or ""))
        graded.append((t, grade_trial(case, t, verdict)))
    return graded


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, round(q * (len(s) - 1)))], 2)


def retrieval_metrics(cases: list[EvalCase], retrieval: dict[str, dict]) -> dict:
    answerable = [c for c in cases if c.expected_docs and c.id in retrieval]
    out: dict = {"cases": len(answerable)}
    ranked = {c.id: [r["doc"] for r in retrieval[c.id]["results"]] for c in answerable}
    for k in (1, 4, 10):
        out[f"recall@{k}"] = round(statistics.fmean(
            recall_at_k(ranked[c.id], c.expected_docs, k) for c in answerable), 3) if answerable else None
    ranks = [first_hit_rank(ranked[c.id], c.expected_docs) for c in answerable]
    out["mrr@10"] = round(statistics.fmean(1 / r if r else 0 for r in ranks), 3) if ranks else None
    return out


def summarize(config: str, cases: list[EvalCase], chats: list[dict], retrieval: dict[str, dict],
              trials: int) -> tuple[dict, list[tuple[dict, TrialGrade]]]:
    graded = grade_all(cases, chats, config)
    quality = [(t, g) for t, g in graded if g.status != "timeout"]
    per_case: dict[str, list[TrialGrade]] = defaultdict(list)
    for _, g in quality:
        per_case[g.case_id].append(g)
    cat_of = {c.id: c.category for c in cases}

    by_cat = {}
    for cat in CATEGORIES:
        ids = [cid for cid in per_case if cat_of[cid] == cat]
        rates = [statistics.fmean(g.passed for g in per_case[cid]) for cid in ids]
        by_cat[cat] = {"cases": len(ids), "trials": sum(len(per_case[i]) for i in ids),
                       "pass_rate": round(statistics.fmean(rates), 3) if rates else None}

    primary = Counter(g.failure for _, g in quality if not g.passed)
    any_label = Counter(primary)
    for _, g in quality:
        any_label.update(f for f in g.flags if f != g.failure and f in ("unsupported_claim",))
    answerable = [g for _, g in quality if g.doc_recall is not None]
    ok = [t for t, _ in graded if t.get("status") == "ok"]
    lat = [t["latency_ms"] / 1000 for t in ok]
    m = [t.get("metrics") or {} for t in ok]
    verdicts = load_verdicts(config)
    first = ok[0] if ok else (chats[0] if chats else {})
    return {
        "config": config, "description": get_config(config).description,
        "embedder": get_config(config).embedder,
        "workspace_settings": first.get("workspace_settings"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(cases), "trials_per_case": trials, "trials_planned": len(cases) * trials,
        "trials_saved": len(graded), "trials_graded": len(quality),
        "trials_timed_out": len(graded) - len(quality),
        "pass_rate": round(statistics.fmean(statistics.fmean(g.passed for g in gs)
                                            for gs in per_case.values()), 3) if per_case else None,
        "cases_always_pass": sum(all(g.passed for g in gs) for gs in per_case.values()),
        "cases_flaky": sum(0 < sum(g.passed for g in gs) < len(gs) for gs in per_case.values()),
        "cases_always_fail": sum(not any(g.passed for g in gs) for gs in per_case.values()),
        "by_category": by_cat,
        "failures_primary": dict(primary.most_common()),
        "failures_including_flags": dict(any_label.most_common()),
        "retrieval": {
            "vector_search_top10": retrieval_metrics(cases, retrieval),
            "chat_context_doc_recall": round(statistics.fmean(g.doc_recall for g in answerable), 3)
            if answerable else None,
        },
        "latency_s": {"ok_trials": len(lat), "mean": round(statistics.fmean(lat), 2) if lat else None,
                      "p50": _quantile(lat, 0.5), "p90": _quantile(lat, 0.9),
                      "p95": _quantile(lat, 0.95), "max": _quantile(lat, 1.0),
                      "over_30s": sum(x > 30 for x in lat),
                      "retried_trials": sum(len(t.get("attempts") or []) > 1 for t, _ in graded),
                      "timed_out_attempts": sum(a.get("timeout", False) for t, _ in graded
                                                for a in t.get("attempts") or []),
                      "slowest_timed_out_attempt_s": max(
                          (a["latency_ms"] / 1000 for t, _ in graded for a in t.get("attempts") or []
                           if a.get("timeout")), default=None),
                      "billed_attempts": sum(len(t.get("attempts") or []) for t, _ in graded)},
        "tokens": {"prompt": sum(int(x.get("prompt_tokens") or 0) for x in m),
                   "completion": sum(int(x.get("completion_tokens") or 0) for x in m)},
        "est_cost_usd": {"chat": round(sum(trace_cost(t) for t in ok), 4),
                         "judge": round(judge_cost(config), 4)},
        "judge": {"verdicts": len(verdicts),
                  "unsupported": sum(not v["supported"] for v in verdicts.values())},
        "per_case": [{"id": cid, "category": cat_of[cid], "passed": sum(g.passed for g in gs),
                      "trials": len(gs), "failures": [g.failure for g in gs if not g.passed]}
                     for cid, gs in sorted(per_case.items())],
    }, graded


def regressgate_report(config: str, cases: list[EvalCase], graded: list[tuple[dict, TrialGrade]],
                       trials: int):
    """A regressgate RunReport (schema v1); timed-out trials are left out of quality."""
    from regressgate import RunReport, __version__
    from regressgate.run_report import CaseResult, CheckResult, Trial

    must_refuse = {c.id: c.must_refuse for c in cases}
    rows: dict[str, list] = defaultdict(list)
    for t, g in graded:
        if g.status == "timeout":
            continue
        if g.status != "ok":
            case_checks = {"app_response": CheckResult(passed=False, detail=g.failure or "")}
        elif must_refuse[g.case_id]:
            case_checks = {"refusal": CheckResult(passed=bool(g.passed))}
        else:
            case_checks = {
                "key_facts": CheckResult(passed=g.facts_ok, detail="; ".join(g.missing)[:200]),
                "context_support": CheckResult(passed=g.context_supported)}
        m = t.get("metrics") or {}
        rows[g.case_id].append(Trial(
            passed=bool(g.passed), checks=case_checks, latency_s=(t.get("latency_ms") or 0) / 1000,
            cost_usd=round(trace_cost(t), 6), input_tokens=m.get("prompt_tokens"),
            output_tokens=m.get("completion_tokens"), output=t.get("answer"),
            error=None if g.status == "ok" else (t.get("error") or g.failure),
            metadata={"failure": g.failure, "flags": g.flags}))
    return RunReport(
        regressgate_version=__version__, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        target=f"anythingllm-1.20.0:{config}", repeats=trials,
        metadata={"config": config, "settings": get_config(config).workspace},
        cases=[CaseResult(id=c.id, tags=[c.category], trials=rows[c.id]) for c in cases if rows[c.id]])


def judge_agreement() -> dict:
    """Manual spot check: results/judge-spot-check.jsonl rows {config, key, human_supported},
    scored against every judge prompt version that has verdicts for those trials."""
    from rag_audit.groundedness_judge import JUDGE_PROMPTS
    path = RESULTS / "judge-spot-check.jsonl"
    if not path.exists():
        return {"checked": 0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    out = {}
    for version in sorted(JUDGE_PROMPTS):
        pairs = []
        for r in rows:
            v = load_verdicts(r["config"], version).get(r["key"])
            if v is not None:
                pairs.append((v["supported"], r["human_supported"], r["key"]))
        agree = sum(a == b for a, b, _ in pairs)
        out[f"v{version}"] = {"checked": len(pairs), "agree": agree,
                              "agreement_rate": round(agree / len(pairs), 3) if pairs else None,
                              "disagreements": [k for a, b, k in pairs if a != b]}
    return out
