"""Tiny AnythingLLM developer-API client: create a workspace, ingest files, ask and save a trace.

Stdlib only. The API key is read from $ANYTHINGLLM_API_KEY or .run/anythingllm-api-key
(git-ignored) and is never written into traces.

Usage:
    python scripts/anythingllm-client.py workspace --slug fomc-2025
    python scripts/anythingllm-client.py ingest --slug fomc-2025 data/corpus/fomc-statement-20250917.pdf
    python scripts/anythingllm-client.py ask --slug fomc-2025 "What target range did the FOMC set in September 2025?"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "http://127.0.0.1:3001/api"
KEY_FILE = ROOT / ".run" / "anythingllm-api-key"
VALID_MODES = ("query", "chat")  # "automatic" enables tool calling, which is out of audit scope
WORKSPACE_SETTING_KEYS = ("slug", "chatMode", "topN", "similarityThreshold", "openAiTemp",
                          "openAiHistory", "vectorSearchMode", "chatProvider", "chatModel",
                          "queryRefusalResponse", "openAiPrompt")


class ApiError(RuntimeError):
    pass


class AnythingLLMClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            raise ApiError(f"{method} {path} -> HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise ApiError(f"{method} {path} -> {err.reason} (is the stack running?)") from err
        try:
            return json.loads(raw) if raw else {}
        except ValueError as err:
            raise ApiError(f"{method} {path} -> non-JSON response: {raw[:200]!r}") from err

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    def post_json(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, json.dumps(payload).encode("utf-8"), "application/json")

    def post_file(self, path: str, file_path: Path, fields: dict[str, str]) -> dict:
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in fields.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                         f"{value}\r\n".encode("utf-8"))
        ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                     f'filename="{file_path.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode("utf-8"))
        parts.append(file_path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        return self._request("POST", path, b"".join(parts), f"multipart/form-data; boundary={boundary}")

    # --- workspace helpers -------------------------------------------------------------

    def workspace(self, slug: str) -> dict | None:
        data = self.get(f"/v1/workspace/{slug}").get("workspace")
        if isinstance(data, list):  # the API returns a one-element list
            data = data[0] if data else None
        return data or None

    def ensure_workspace(self, slug: str) -> dict:
        ws = self.workspace(slug)
        if ws:
            return ws
        created = self.post_json("/v1/workspace/new", {"name": slug})
        ws = created.get("workspace") or {}
        if ws.get("slug") != slug:
            raise ApiError(f"workspace created with slug {ws.get('slug')!r}, expected {slug!r}; "
                           "use that slug or rename it in the app")
        return ws


def load_api_key() -> str:
    key = os.environ.get("ANYTHINGLLM_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        sys.exit(f"No API key: set ANYTHINGLLM_API_KEY or create {KEY_FILE}")
    return key


def settings_snapshot(ws: dict | None) -> dict:
    return {k: ws.get(k) for k in WORKSPACE_SETTING_KEYS} if ws else {}


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
    payload = {"message": args.question, "mode": args.mode, "sessionId": session_id}
    started = time.perf_counter()
    res = client.post_json(f"/v1/workspace/{args.slug}/chat", payload)
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
    client = AnythingLLMClient(args.base_url, load_api_key())
    handlers = {"workspace": cmd_workspace, "ingest": cmd_ingest, "ask": cmd_ask}
    try:
        return handlers[args.cmd](client, args)
    except ApiError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
