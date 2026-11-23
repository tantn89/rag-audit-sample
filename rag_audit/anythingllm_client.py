"""Stdlib HTTP client for the AnythingLLM developer API (the audited app's public interface).

The API key comes from $ANYTHINGLLM_API_KEY or .run/anythingllm-api-key (git-ignored) and is
never written into traces or reports.
"""

from __future__ import annotations

import json
import mimetypes
import os
import socket
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "http://127.0.0.1:3001/api"
KEY_FILE = ROOT / ".run" / "anythingllm-api-key"
WORKSPACE_SETTING_KEYS = ("slug", "chatMode", "topN", "similarityThreshold", "openAiTemp",
                          "openAiHistory", "vectorSearchMode", "chatProvider", "chatModel",
                          "queryRefusalResponse", "openAiPrompt")


class ApiError(RuntimeError):
    """The app answered with an error, a non-JSON body, or could not be reached."""


class ApiTimeout(ApiError):
    """No response within the client timeout (the app may still finish, and bill, the call)."""


class AnythingLLMClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: bytes | None = None,
                 content_type: str | None = None, timeout: float | None = None,
                 expect_json: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            raise ApiError(f"{method} {path} -> HTTP {err.code}: {detail}") from err
        except (TimeoutError, socket.timeout) as err:
            raise ApiTimeout(f"{method} {path} -> no response within {timeout or self.timeout}s") from err
        except urllib.error.URLError as err:
            if isinstance(err.reason, (TimeoutError, socket.timeout)):
                raise ApiTimeout(f"{method} {path} -> no response within "
                                 f"{timeout or self.timeout}s") from err
            raise ApiError(f"{method} {path} -> {err.reason} (is the stack running?)") from err
        if not expect_json:
            return {}
        try:
            return json.loads(raw) if raw else {}
        except ValueError as err:
            raise ApiError(f"{method} {path} -> non-JSON response: {raw[:200]!r}") from err

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    def post_json(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        return self._request("POST", path, json.dumps(payload).encode("utf-8"),
                             "application/json", timeout)

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

    def update_workspace(self, slug: str, settings: dict) -> dict:
        res = self.post_json(f"/v1/workspace/{slug}/update", settings)
        ws = res.get("workspace")
        if not ws:
            raise ApiError(f"workspace update failed: {res.get('message') or res}")
        return ws

    def chat(self, slug: str, message: str, mode: str, session_id: str,
             timeout: float | None = None) -> dict:
        payload = {"message": message, "mode": mode, "sessionId": session_id}
        return self.post_json(f"/v1/workspace/{slug}/chat", payload, timeout)

    def delete_workspace(self, slug: str) -> None:
        self._request("DELETE", f"/v1/workspace/{slug}", expect_json=False)  # replies "OK"

    def update_embeddings(self, slug: str, adds: list[str], deletes: list[str]) -> dict:
        return self.post_json(f"/v1/workspace/{slug}/update-embeddings",
                              {"adds": adds, "deletes": deletes}, timeout=1800)

    def vector_search(self, slug: str, query: str, top_n: int) -> list[dict]:
        res = self.post_json(f"/v1/workspace/{slug}/vector-search", {"query": query, "topN": top_n})
        return res.get("results") or []


def load_api_key() -> str:
    key = os.environ.get("ANYTHINGLLM_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ApiError(f"No API key: set ANYTHINGLLM_API_KEY or create {KEY_FILE}")
    return key


def settings_snapshot(ws: dict | None) -> dict:
    return {k: ws.get(k) for k in WORKSPACE_SETTING_KEYS} if ws else {}


def make_client(timeout: float = 300) -> AnythingLLMClient:
    base = os.environ.get("ANYTHINGLLM_BASE_URL", DEFAULT_BASE)
    return AnythingLLMClient(base, load_api_key(), timeout)
