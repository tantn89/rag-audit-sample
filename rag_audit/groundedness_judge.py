"""LLM-judge groundedness check, used only where deterministic grading cannot decide.

The grader asks for a verdict only on trials whose answer failed its key-fact checks without
refusing, and on out-of-corpus questions that were answered instead of declined. The judge says
whether every factual claim in the answer is supported by the retrieved chunks. Verdicts are
cached in results/<config>/judge-verdicts.jsonl (committed), keyed by case, trial and a hash of
everything the judge saw (question, answer, cited chunk titles and text) and the prompt version,
so recomputing the numbers never calls the LLM again and a changed trace never reuses a stale
verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

from rag_audit.audit_configs import CHAT_MODEL
from rag_audit.text_matching import strip_chunk_metadata
from rag_audit.trace_store import RESULTS, ROOT, usage_cost

GUARD_BASE = os.environ.get("GUARD_BASE_URL", "http://127.0.0.1:18431/v1")
GUARD_TOKEN_FILE = ROOT / ".run" / "guard-token"

_JUDGE_BASE = (
    "You are a strict fact-checker for a retrieval-augmented assistant. You get a question, the "
    "context chunks the assistant retrieved, and its answer. Decide whether every factual claim "
    "in the answer (numbers, dates, names, decisions, votes) is stated in or directly implied by "
    "the context. Statements that the context lacks the information are not claims. "
)
# v2 adds attribution: the manual spot check of v1 found it accepted facts that are true of a
# different meeting than the one the answer names (6 of 7 disagreements).
_JUDGE_ATTRIBUTION = (
    "Each chunk is labelled with its source file; the file name holds the meeting date (for "
    "example fomc-minutes-20250730.pdf is the July 29-30, 2025 meeting). A claim about a specific "
    "meeting or document is supported only by a chunk from that meeting's files; the same fact "
    "from another meeting does not support it. A generalisation such as 'never', 'at any "
    "meeting' or 'only once' is supported only if the chunks cover every case. "
)
_JUDGE_REPLY = 'Reply with JSON only: {"supported": true or false, "unsupported_claims": ["..."]}'
# v3: v2 did not fix attribution (11 of 18 agreed, the same as v1), so the judge must now cite,
# per claim, the chunk that states it for the meeting the claim is about; chunks are labelled in
# plain words.
_JUDGE_PER_CLAIM = (
    "You are a strict fact-checker for a retrieval-augmented assistant answering questions about "
    "FOMC meetings. Each context chunk is labelled with the meeting and document it comes from. "
    "List every factual claim in the answer (numbers, dates, names, decisions, votes, counts). For "
    "each, name the meeting or document the claim is about and the number of a chunk FROM THAT "
    "SAME meeting that states it, or null. A fact stated only for a different meeting does not "
    "count. 'Never', 'at any meeting', 'only once' and counts need chunks covering every case. "
    "Saying the context lacks information is not a claim. supported is false if any claim has "
    'chunk null. Reply with JSON only: {"claims": [{"claim": "...", "about": "...", "chunk": 1}], '
    '"supported": true or false, "unsupported_claims": ["..."]}'
)
JUDGE_PROMPTS = {1: _JUDGE_BASE + _JUDGE_REPLY, 2: _JUDGE_BASE + _JUDGE_ATTRIBUTION + _JUDGE_REPLY,
                 3: _JUDGE_PER_CLAIM}
JUDGE_VERSION = 3
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


def chunk_label(title: str | None) -> str:
    """'fomc-minutes-20250730.pdf' -> 'minutes of the FOMC meeting ending July 30, 2025'."""
    m = re.fullmatch(r"fomc-(statement|minutes)-(\d{4})(\d{2})(\d{2})\.pdf", title or "")
    if not m:
        return title or "unknown source"
    kind = "post-meeting statement" if m[1] == "statement" else "minutes"
    return f"{kind} of the FOMC meeting ending {_MONTHS[int(m[3]) - 1]} {int(m[4])}, {m[2]}"


def verdict_key(trace: dict, version: int = JUDGE_VERSION) -> str:
    """Cache key of one verdict: readable case#trial plus a hash of every judge input."""
    judged = {"judge_version": version, "question": trace.get("question") or "",
              "answer": trace.get("answer") or "",
              "sources": [[s.get("title") or "", s.get("text") or ""]
                          for s in trace.get("sources") or []]}
    digest = hashlib.sha256(json.dumps(judged, ensure_ascii=False, sort_keys=True)
                            .encode("utf-8")).hexdigest()[:16]
    return f"{trace.get('case_id')}#{trace.get('trial')}#{digest}"


def verdicts_path(config: str) -> Path:
    return RESULTS / config / "judge-verdicts.jsonl"


def _rows(config: str) -> list[dict]:
    path = verdicts_path(config)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_verdicts(config: str, version: int = JUDGE_VERSION) -> dict[str, dict]:
    """Verdicts of one judge prompt version (rows without a version are v1)."""
    return {r["key"]: r for r in _rows(config) if r.get("judge_version", 1) == version}


def judge_cost(config: str) -> float:
    """Every judge call ever made for this config, all prompt versions."""
    return sum(usage_cost(r.get("prompt_tokens", 0), r.get("completion_tokens", 0))
               for r in _rows(config))


DEFAULT_JUDGE_CALL_USD = 0.002  # used until a verdict has been logged (well above a real call)


def judge_call_estimate(configs) -> float:
    """Mean logged cost of one judge call across `configs` (a conservative default before any)."""
    rows = [r for c in configs for r in _rows(c)]
    if not rows:
        return DEFAULT_JUDGE_CALL_USD
    return sum(usage_cost(r.get("prompt_tokens", 0), r.get("completion_tokens", 0))
               for r in rows) / len(rows)


def affordable_calls(wanted: int, spent_usd: float, per_call_usd: float, budget_usd: float) -> int:
    """How many of `wanted` judge calls fit in the budget after `spent_usd`."""
    if per_call_usd <= 0:
        return wanted
    room = budget_usd - spent_usd
    # the small epsilon keeps float error from losing a call that fits exactly ($0.01 / $0.002)
    return max(0, min(wanted, int(room / per_call_usd + 1e-9)))


def _guard_token() -> str:
    token = os.environ.get("GUARD_LOCAL_TOKEN", "").strip()
    if not token and GUARD_TOKEN_FILE.exists():
        token = GUARD_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("no guard token: start the guard (scripts/start-guard.ps1)")
    return token


def _parse(content: str) -> dict:
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        raise ValueError(f"judge reply is not JSON: {content[:200]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data.get("supported"), bool):
        raise ValueError(f"judge reply lacks a boolean 'supported': {data}")
    claims = data.get("unsupported_claims") or []
    return {"supported": data["supported"], "unsupported_claims": [str(c) for c in claims][:10]}


def judge_trace(trace: dict, timeout: float = 180) -> dict:
    context = "\n\n".join(f"[{i}] ({chunk_label(s.get('title'))}) "
                          f"{strip_chunk_metadata(s.get('text')).strip()}"
                          for i, s in enumerate(trace.get("sources") or [], start=1))
    user = (f"Question: {trace['question']}\n\nContext chunks:\n{context or '(none)'}\n\n"
            f"Answer to check:\n{trace.get('answer')}")
    body = {"model": CHAT_MODEL, "temperature": 0, "max_tokens": 700,
            "messages": [{"role": "system", "content": JUDGE_PROMPTS[JUDGE_VERSION]},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{GUARD_BASE}/chat/completions", method="POST",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {_guard_token()}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res = json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"judge call failed: HTTP {err.code} "
                           f"{err.read().decode('utf-8', 'replace')[:200]}") from err
    verdict = _parse(res["choices"][0]["message"]["content"])
    usage = res.get("usage") or {}
    return {**verdict, "judge_version": JUDGE_VERSION, "model": res.get("model", CHAT_MODEL),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0)}


_APPEND_LOCK = threading.Lock()


def append_verdict(config: str, row: dict) -> None:
    path = verdicts_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK, path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
