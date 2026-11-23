"""Drive the app: retrieval probes via /vector-search (no LLM spend) and chat trials via /chat.

Chat trials are resumable (a trial with a saved trace is skipped), bounded per pass (one `run`
makes at most `max_attempts` billed calls per trial, retrying only timeouts, 5xx errors and
connection failures after a 10 s then 30 s backoff), and budgeted (the run stops submitting work
when spent + projected cost would exceed the budget). A trial whose attempts all timed out is
saved as a timeout; each explicit `run --retry-timeouts` pass re-runs it, adding up to
`max_attempts` more billed calls, so after one such pass a trial can carry 2 x `max_attempts`
attempts in its trace (the published runs: at most 4). A
timed-out call may still be billed by the provider, but its token usage is unknown: the cost
estimate charges it the running average and the published cost leaves it unmetered.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

from rag_audit.anythingllm_client import AnythingLLMClient, ApiError, ApiTimeout, settings_snapshot
from rag_audit.audit_configs import CHAT_MODE, SLUG
from rag_audit.eval_cases import EvalCase
from rag_audit.trace_store import (
    TRACES,
    chat_trace_path,
    read_json,
    retrieval_trace_path,
    trace_cost,
    write_json,
)

BACKOFF_S = (10, 30)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def probe_retrieval(client: AnythingLLMClient, config: str, cases: list[EvalCase],
                    top_n: int = 10) -> int:
    """Rank chunks for every question with the workspace's embedder; deterministic and free."""
    for case in cases:
        results = client.vector_search(SLUG, case.question, top_n)
        write_json(retrieval_trace_path(config, case.id), {
            "case_id": case.id, "question": case.question, "top_n": top_n, "asked_at": _now(),
            "results": [{"rank": i, "id": r.get("id"), "doc": (r.get("metadata") or {}).get("title"),
                         "score": r.get("score"), "text": r.get("text")}
                        for i, r in enumerate(results, start=1)],
        })
    return len(cases)


def _is_retryable(err: ApiError) -> bool:
    text = str(err)
    return isinstance(err, ApiTimeout) or "HTTP 5" in text or "is the stack running" in text


def ask(client: AnythingLLMClient, config: str, case: EvalCase, trial: int, settings: dict,
        timeout: float, max_attempts: int) -> dict:
    attempts: list[dict] = []
    trace = {"config": config, "case_id": case.id, "trial": trial, "question": case.question,
             "mode": CHAT_MODE, "workspace_settings": settings, "asked_at": _now(),
             "timeout_s": timeout, "max_attempts": max_attempts}
    for attempt in range(1, max_attempts + 1):
        session_id = f"audit-{uuid.uuid4().hex[:12]}"  # fresh session: no chat history
        started = time.perf_counter()
        try:
            res = client.chat(SLUG, case.question, CHAT_MODE, session_id, timeout=timeout)
        except ApiError as err:
            elapsed = round((time.perf_counter() - started) * 1000)
            attempts.append({"attempt": attempt, "latency_ms": elapsed, "error": str(err)[:300],
                             "timeout": isinstance(err, ApiTimeout)})
            if attempt < max_attempts and _is_retryable(err):
                time.sleep(BACKOFF_S[min(attempt, len(BACKOFF_S)) - 1])
                continue
            status = "timeout" if all(a["timeout"] for a in attempts) else "error"
            return {**trace, "status": status, "attempts": attempts, "error": str(err)[:300]}
        latency_ms = round((time.perf_counter() - started) * 1000)
        attempts.append({"attempt": attempt, "latency_ms": latency_ms, "error": None, "timeout": False})
        return {**trace, "status": "ok", "attempts": attempts, "session_id": session_id,
                "latency_ms": latency_ms, "answer": res.get("textResponse"),
                "response_type": res.get("type"), "error": res.get("error"),
                "sources": [{"id": s.get("id"), "title": s.get("title"), "score": s.get("score"),
                             "text": s.get("text") or s.get("chunk")}
                            for s in res.get("sources") or []],
                "metrics": res.get("metrics"), "chat_id": res.get("id")}
    raise AssertionError("unreachable")


def spent_usd() -> tuple[float, int]:
    """Estimated spend and billed-call count across every saved chat trace (all configs)."""
    total, calls = 0.0, 0
    for path in TRACES.glob("*/chat/*/trial-*.json"):
        t = read_json(path)
        total += trace_cost(t)
        calls += len(t.get("attempts") or [])
    return total, calls


def _needs_run(config: str, case_id: str, trial: int, retry_timeouts: bool) -> bool:
    path = chat_trace_path(config, case_id, trial)
    if not path.exists():
        return True
    return retry_timeouts and read_json(path).get("status") == "timeout"


def run_trials(client: AnythingLLMClient, config: str, cases: list[EvalCase], trials: int, *,
               timeout: float, max_attempts: int, concurrency: int, budget_usd: float,
               extra_spent_usd: float = 0.0, retry_timeouts: bool = False, log=print) -> dict:
    """`retry_timeouts` re-runs trials saved as timeouts (an explicit, one-off pass: each pass
    adds at most `max_attempts` billed calls per trial); earlier attempts stay in the trace."""
    ws = client.workspace(SLUG)
    if not ws:
        raise ApiError(f"workspace {SLUG!r} does not exist")
    settings = settings_snapshot(ws)
    todo = [(c, t) for c in cases for t in range(1, trials + 1)
            if _needs_run(config, c.id, t, retry_timeouts)]
    spent, calls = spent_usd()
    spent += extra_spent_usd
    log(f"[{config}] {len(todo)} trials to run; estimated spend so far ${spent:.4f} ({calls} calls)")
    lock = threading.Lock()
    stats = {"done": 0, "ok": 0, "timeout": 0, "error": 0, "cost": 0.0, "stopped_for_budget": False}

    def job(case: EvalCase, trial: int) -> tuple[dict, int]:
        path = chat_trace_path(config, case.id, trial)
        earlier = read_json(path).get("attempts", []) if path.exists() else []
        trace = ask(client, config, case, trial, settings, timeout, max_attempts)
        new_timeouts = sum(a["timeout"] for a in trace["attempts"])
        if earlier:  # keep the timed-out history: it is part of the latency record
            merged = earlier + trace["attempts"]
            for i, a in enumerate(merged, start=1):
                a["attempt"] = i
            trace["attempts"] = merged
        write_json(path, trace)
        return trace, new_timeouts

    pending: set = set()
    queue = list(todo)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while queue or pending:
            while queue and len(pending) < concurrency:
                with lock:
                    avg = stats["cost"] / stats["ok"] if stats["ok"] else 0.001
                    projected = spent + stats["cost"] + avg * (len(queue) + len(pending))
                if projected > budget_usd:
                    stats["stopped_for_budget"] = True
                    log(f"[{config}] stop: projected ${projected:.3f} > budget ${budget_usd:.2f}")
                    queue.clear()
                    break
                pending.add(pool.submit(job, *queue.pop(0)))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                trace, lost = fut.result()
                with lock:
                    stats["done"] += 1
                    stats[trace["status"]] += 1
                    # unknown usage on timed-out attempts: charge the running average (conservative)
                    avg = stats["cost"] / stats["ok"] if stats["ok"] else 0.001
                    stats["cost"] += trace_cost(trace) + lost * avg
                if stats["done"] % 10 == 0 or trace["status"] != "ok":
                    log(f"[{config}] {stats['done']}/{len(todo)} {trace['case_id']}#{trace['trial']} "
                        f"{trace['status']} {trace.get('latency_ms', '-')} ms; "
                        f"run cost ${stats['cost']:.4f}")
    return stats
