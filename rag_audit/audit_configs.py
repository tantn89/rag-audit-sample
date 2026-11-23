"""Named app configurations under test. Each is a settings-only change a client could apply in
the AnythingLLM UI or API; the app's code is never modified."""

from __future__ import annotations

from dataclasses import dataclass, field

SLUG = "fomc-2025"
CHAT_MODE = "query"  # documents only; "automatic" would enable tool calling (out of scope)
CHAT_MODEL = "gpt-4.1-mini"

# AnythingLLM 1.20.0's default workspace prompt, verbatim (from the workspace settings).
DEFAULT_PROMPT = (
    "Given the following conversation, relevant context, and a follow up question, reply with an "
    "answer to the current question the user is asking. The current date and time is {datetime}. "
    "Return only your response to the question given the above information following the users "
    "instructions as needed."
)


@dataclass(frozen=True)
class AuditConfig:
    name: str
    description: str
    workspace: dict = field(default_factory=dict)  # POST /v1/workspace/:slug/update payload
    embedder: str = "native (all-MiniLM-L6-v2, 1000-char chunks, 20 overlap)"


BASELINE = AuditConfig(
    name="baseline",
    description="AnythingLLM 1.20.0 defaults: topN 4, similarity threshold 0.25, default vector "
                "search, default prompt, built-in local embedder.",
    workspace={"topN": 4, "similarityThreshold": 0.25, "vectorSearchMode": "default",
               "openAiPrompt": DEFAULT_PROMPT},
)

# Fix candidates, each one setting away from the baseline. The reranker was piloted with 1 chat
# trial on all 67 questions (context recall from /chat sources), the embedder with the
# /vector-search probe. Both endpoints honour the workspace search mode.
CANDIDATE_RERANK = AuditConfig(
    name="candidate-rerank",
    description="Baseline plus vectorSearchMode=rerank (AnythingLLM's built-in local reranker "
                "re-scores up to 50 nearest chunks and keeps the top 4).",
    workspace={**BASELINE.workspace, "vectorSearchMode": "rerank"},
)
CANDIDATE_EMBEDDER = AuditConfig(
    name="candidate-openai-embedder",
    description="Baseline workspace settings with the embedder switched to text-embedding-3-small "
                "(OpenAI-compatible, through the guard), documents re-embedded. Measured with "
                "EMBEDDING_MODEL_MAX_CHUNK_LENGTH=8191, which AnythingLLM also uses as the chunk "
                "size, so chunks grew from ~900 to ~6200 chars: two changes, not one. Larger chunks "
                "favour document-level recall, so the reranker's lead over it is conservative.",
    workspace=dict(BASELINE.workspace),
    embedder="text-embedding-3-small via llm-api-guard (chunks up to 8191 chars, 20 overlap)",
)
# The same embedder swap with the chunk size held at the baseline's 1000 characters, so only the
# embedder changes. Retrieval-only (/vector-search), directly comparable with the fix's probe.
# Measured after the fix run, as a check on the choice, not as part of it.
CANDIDATE_EMBEDDER_1000 = AuditConfig(
    name="candidate-openai-embedder-1000",
    description="Baseline workspace settings with the embedder switched to text-embedding-3-small "
                "(OpenAI-compatible, through the guard), documents re-embedded with "
                "EMBEDDING_MODEL_MAX_CHUNK_LENGTH=1000, the baseline chunk size: one change.",
    workspace=dict(BASELINE.workspace),
    embedder="text-embedding-3-small via llm-api-guard (1000-char chunks, 20 overlap)",
)

# The chosen fix: in the 1-trial pilot on the same 67 questions the reranker raised context doc
# recall@4 from 0.56 to 0.84, against 0.75 for the embedder probe at 8191-char chunks (two
# changes at once), with no re-ingest, no API spend and no corpus leaving the machine. The
# embedder re-measured at 1000-char chunks after the fix run reached 0.79. One workspace setting
# changes; everything else equals the baseline.
FIX = AuditConfig(
    name="fix",
    description="Baseline plus vectorSearchMode=rerank (one workspace setting).",
    workspace=dict(CANDIDATE_RERANK.workspace),
)

CONFIGS = {c.name: c for c in (BASELINE, CANDIDATE_RERANK, CANDIDATE_EMBEDDER,
                                CANDIDATE_EMBEDDER_1000, FIX)}


def get_config(name: str) -> AuditConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise SystemExit(f"unknown config {name!r}; known: {', '.join(CONFIGS)}") from None
