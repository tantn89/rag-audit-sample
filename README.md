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

Status: eval set, runner, grader, baseline and one before/after fix are done (numbers in
[`results/`](results/)). The written report is in progress.

## Layout

| Path | What it is |
|------|------------|
| `compose.yaml` | AnythingLLM pinned to `1.20.0@sha256:…`, loopback port 3001, local embedder |
| `scripts/start-guard.ps1` / `stop-guard.ps1` | Start or stop the local guard; the token goes to `.run/` (git-ignored) |
| `scripts/start-stack.ps1` / `stop-stack.ps1` | Guard plus app up or down (`-Embedder openai` adds `compose.embedder-openai.yaml`) |
| `scripts/download-corpus.py` | Downloads the corpus and fails on any SHA-256 mismatch |
| `scripts/anythingllm-client.py` | Creates the workspace, ingests files, asks one question and saves a JSON trace |
| `evals/cases.jsonl` | The eval set: 67 questions with key facts, refusal flags, expected source documents and page-cited evidence |
| `rag_audit/` | Runner, deterministic grader, groundedness judge and reports (`python -m rag_audit --help`) |
| `results/<config>/` | Per-config summary, per-trial grades, regressgate report, judge verdicts and packed traces |
| `results/compare-*.md` | regressgate before/after verdict |
| `results/candidate-*/` | The two fix candidates compared before choosing the fix (reranker pilot, embedder swap) |

## Reproduce the numbers

From committed traces, with no app and no LLM calls:

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m rag_audit reproduce          # re-grades baseline and fix, re-runs the gate
.venv\Scripts\regressgate compare results\baseline\regressgate-report.json results\fix\regressgate-report.json
.venv\Scripts\python -m pytest -q
```

Full live re-run (stack running, see Quick start; spends about $0.3 at list prices). Move or
delete `traces\` first; otherwise saved trials are skipped and the run resumes:

```powershell
.venv\Scripts\python -m rag_audit full-rerun --configs baseline fix --budget-usd 3
```

## How answers are graded

- **Deterministic first.** Each case lists key facts (`must_include`, numbers compared by
  value after normalising "4-1/4", "4¼", "4.25%"), rate ranges, forbidden terms, and the
  documents that hold the answer. A pass needs the facts in the answer *and* in the chunks the
  app retrieved from those documents, so the answer is verifiable from its citations.
- **Refusals.** Out-of-corpus questions pass only when the answer declines in its first two
  sentences and states none of the forbidden values.
- **LLM judge only where needed.** Failed answers that did not refuse get a groundedness verdict
  from the same guarded model (cached in `results/<config>/judge-verdicts.jsonl`). It labels a
  failure as an unsupported claim; it never changes pass or fail. A manual spot check of 18
  judged trials (`results/judge-spot-check.jsonl`) agreed with the judge on 11 (61%) for each of
  three prompt versions: gpt-4.1-mini tends to accept a fact that is true of a different meeting
  than the one the answer names. Treat the unsupported-claim label as indicative only.
- **Repeated trials.** 3 trials per case per config. regressgate compares configs with a
  permutation test and per-case Fisher tests. Latency is reported apart from quality: a trial
  that times out after its retry is excluded from the pass rate and counted separately;
  `run --retry-timeouts` gives such trials one more bounded pass, keeping the timed-out attempts
  in the trace.

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
