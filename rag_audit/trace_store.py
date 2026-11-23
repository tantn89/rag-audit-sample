"""Where traces live and what they cost.

Raw traces: traces/<config>/chat/<case>/trial-<n>.json and traces/<config>/retrieval/<case>.json
(git-ignored). `pack` copies them into results/<config>/traces.jsonl.gz, which is committed so the
published numbers can be recomputed from a clone without calling the app or the LLM.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "traces"
RESULTS = ROOT / "results"

# Estimate only: OpenAI list prices for gpt-4.1-mini (USD per 1M tokens). The provider's own
# price was not published; the guard log is the source of truth for token counts.
PRICE_PER_M = {"prompt": 0.40, "completion": 1.60}
EXCERPT_CHARS = 300  # retrieval-probe chunk text kept in bundles (chat traces keep full text)


def chat_trace_path(config: str, case_id: str, trial: int) -> Path:
    return TRACES / config / "chat" / case_id / f"trial-{trial}.json"


def retrieval_trace_path(config: str, case_id: str) -> Path:
    return TRACES / config / "retrieval" / f"{case_id}.json"


def bundle_path(config: str) -> Path:
    return RESULTS / config / "traces.jsonl.gz"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a killed run never leaves a half-written trace behind


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def usage_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_PER_M["prompt"] + completion_tokens * PRICE_PER_M["completion"]) / 1e6


def trace_cost(trace: dict) -> float:
    m = trace.get("metrics") or {}
    return usage_cost(int(m.get("prompt_tokens") or 0), int(m.get("completion_tokens") or 0))


def load_raw(config: str) -> tuple[list[dict], dict[str, dict]]:
    """(chat traces, retrieval traces by case id) from traces/<config>/."""
    base = TRACES / config
    chats = [read_json(p) for p in sorted((base / "chat").glob("*/trial-*.json"))]
    retrieval = {p.stem: read_json(p) for p in sorted((base / "retrieval").glob("*.json"))}
    return chats, retrieval


def pack(config: str) -> Path:
    chats, retrieval = load_raw(config)
    if not chats and not retrieval:
        raise SystemExit(f"no traces under {TRACES / config}; run the config first")
    out = bundle_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip bytes identical for identical traces (no churn in git)
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for t in chats:
            gz.write((json.dumps({"kind": "chat", **t}, ensure_ascii=False) + "\n").encode())
        for case_id, r in retrieval.items():
            # recall@k needs only ranks and documents; keep a short excerpt of each chunk
            short = [{**x, "text": (x.get("text") or "")[:EXCERPT_CHARS]} for x in r["results"]]
            gz.write((json.dumps({"kind": "retrieval", "case_id": case_id, **r, "results": short},
                                 ensure_ascii=False) + "\n").encode())
    return out


def load_bundle(config: str) -> tuple[list[dict], dict[str, dict]]:
    path = bundle_path(config)
    if not path.exists():
        raise SystemExit(f"{path} not found; run `python -m rag_audit pack {config}` first")
    chats, retrieval = [], {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            kind = row.pop("kind")
            if kind == "chat":
                chats.append(row)
            else:
                retrieval[row["case_id"]] = row
    return chats, retrieval


def load_traces(config: str, from_bundle: bool = False) -> tuple[list[dict], dict[str, dict]]:
    if not from_bundle and (TRACES / config).is_dir():
        return load_raw(config)
    return load_bundle(config)
