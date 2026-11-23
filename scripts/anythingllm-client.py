"""Tiny AnythingLLM developer-API CLI: create a workspace, ingest files, ask and save a trace.

Stdlib only. The API key is read from $ANYTHINGLLM_API_KEY or .run/anythingllm-api-key
(git-ignored) and is never written into traces. The HTTP client lives in rag_audit/.

Usage:
    python scripts/anythingllm-client.py workspace --slug fomc-2025
    python scripts/anythingllm-client.py ingest --slug fomc-2025 data/corpus/fomc-statement-20250917.pdf
    python scripts/anythingllm-client.py ask --slug fomc-2025 "What target range did the FOMC set in September 2025?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_audit.anythingllm_client import (  # noqa: E402
    DEFAULT_BASE,
    AnythingLLMClient,
    ApiError,
    load_api_key,
    settings_snapshot,
)

VALID_MODES = ("query", "chat")  # "automatic" enables tool calling, which is out of audit scope


def cmd_workspace(client: AnythingLLMClient, args) -> int:
    ws = client.ensure_workspace(args.slug)
    print(json.dumps(settings_snapshot(ws), indent=2))
    return 0


def cmd_ingest(client: AnythingLLMClient, args) -> int:
    client.ensure_workspace(args.slug)
    for name in args.files:
        path = Path(name)
        if not path.is_file():
            print(f"skip (not a file): {path}", file=sys.stderr)
            return 1
        started = time.perf_counter()
        res = client.post_file("/v1/document/upload", path, {"addToWorkspaces": args.slug})
        if not res.get("success"):
            raise ApiError(f"upload failed for {path.name}: {res.get('error')}")
        docs = res.get("documents") or []
        words = sum(d.get("wordCount") or 0 for d in docs)
        tokens = sum(d.get("token_count_estimate") or 0 for d in docs)
        print(f"ingested {path.name}: {len(docs)} doc(s), {words} words, ~{tokens} tokens, "
              f"{time.perf_counter() - started:.1f}s")
    return 0


def cmd_ask(client: AnythingLLMClient, args) -> int:
    ws = client.workspace(args.slug)
    if not ws:
        raise ApiError(f"workspace {args.slug!r} does not exist")
    session_id = args.session or f"audit-{uuid.uuid4().hex[:12]}"  # fresh session = no chat history
    started = time.perf_counter()
    res = client.chat(args.slug, args.question, args.mode, session_id)
    latency_ms = round((time.perf_counter() - started) * 1000)

    trace = {
        "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": args.question,
        "mode": args.mode,
        "session_id": session_id,
        "workspace_settings": settings_snapshot(ws),
        "latency_ms": latency_ms,
        "answer": res.get("textResponse"),
        "response_type": res.get("type"),
        "error": res.get("error"),
        "sources": [
            {"title": s.get("title"), "score": s.get("score"), "chunk_source": s.get("chunkSource"),
             "text": s.get("text") or s.get("chunk")}
            for s in res.get("sources") or []
        ],
        "metrics": res.get("metrics"),  # token usage and timing, when the provider reports them
        "chat_id": res.get("id"),
    }
    out = Path(args.out) if args.out else ROOT / "traces" / f"{session_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"answer ({latency_ms} ms, {len(trace['sources'])} sources): {trace['answer']}")
    print(f"trace saved: {out}")
    return 0 if not trace["error"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="AnythingLLM developer-API client for the audit")
    ap.add_argument("--base-url", default=os.environ.get("ANYTHINGLLM_BASE_URL", DEFAULT_BASE))
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("workspace", help="create the workspace if missing and print its settings")
    w.add_argument("--slug", required=True)

    i = sub.add_parser("ingest", help="upload files and embed them into the workspace")
    i.add_argument("--slug", required=True)
    i.add_argument("files", nargs="+")

    a = sub.add_parser("ask", help="ask one question and save the trace as JSON")
    a.add_argument("--slug", required=True)
    a.add_argument("--mode", choices=VALID_MODES, default="query")
    a.add_argument("--session", help="reuse a session id (default: a fresh one per question)")
    a.add_argument("--out", help="trace path (default: traces/<session>.json)")
    a.add_argument("question")

    args = ap.parse_args()
    handlers = {"workspace": cmd_workspace, "ingest": cmd_ingest, "ask": cmd_ask}
    try:
        client = AnythingLLMClient(args.base_url, load_api_key())
        return handlers[args.cmd](client, args)
    except ApiError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
