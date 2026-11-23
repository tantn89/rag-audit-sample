# rag-audit-sample

**Sample audit on an open-source project.** This is not a client engagement. It shows what a
"RAG/LLM answer audit" delivers, run end to end on a well-known open-source RAG app and a public
corpus, with every number reproducible from this repo.

- **Audited app:** [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) 1.20.0 (MIT),
  run unmodified in Docker. Only its settings are changed.
- **Corpus:** the 2025 FOMC post-meeting statements and meeting minutes (16 PDFs) from the Board of
  Governors of the Federal Reserve System. They are public domain; source: federalreserve.gov. See
  [corpus/corpus-manifest.json](corpus/corpus-manifest.json) for URLs and SHA-256 pins.
- **LLM access:** chat goes through a local allowlisting proxy (`llm-api-guard`) to an
  OpenAI-compatible provider. The app container only holds a per-launch local token.

Status: setup spike done. The eval set, runner, findings and report are in progress.

## Layout

| Path | What it is |
|------|------------|
| `compose.yaml` | AnythingLLM pinned to `1.20.0@sha256:…`, loopback port 3001, local embedder |
| `scripts/start-guard.ps1` / `stop-guard.ps1` | Start or stop the local guard; the token goes to `.run/` (git-ignored) |
| `scripts/start-stack.ps1` / `stop-stack.ps1` | Guard plus app up or down |
| `scripts/download-corpus.py` | Downloads the corpus and fails on any SHA-256 mismatch |
| `scripts/anythingllm-client.py` | Creates the workspace, ingests files, asks one question and saves a JSON trace |

## Quick start (Windows, Docker Desktop)

```powershell
python scripts\download-corpus.py
scripts\start-stack.ps1   # first run also creates an app API key in .run\ (git-ignored)
python scripts\anythingllm-client.py workspace --slug fomc-2025
python scripts\anythingllm-client.py ingest --slug fomc-2025 (Get-ChildItem data\corpus\*.pdf).FullName
python scripts\anythingllm-client.py ask --slug fomc-2025 "What target range did the FOMC set in September 2025?"
scripts\stop-stack.ps1
```

Requirements: Docker Desktop, Python 3.11+ (stdlib only), and the
`llm-api-guard` proxy with access to an OpenAI-compatible provider. To use a different
OpenAI-compatible endpoint, edit `GENERIC_OPEN_AI_BASE_PATH` in `compose.yaml`.

## Licences

The code in this repo is under the MIT licence. AnythingLLM is MIT-licensed by Mintplex Labs.
The corpus documents are US Federal Reserve Board publications in the public domain.
