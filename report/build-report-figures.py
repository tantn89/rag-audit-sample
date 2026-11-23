"""Build the report's charts and derived numbers from the committed results only.

    python report/build-report-figures.py

Reads results/<config>/summary.json, results/compare-baseline-vs-fix.json and the packed traces
(results/<config>/traces.jsonl.gz). Writes report/figures/*.png and report/report-numbers.json.
No app, no LLM and no network: every figure in the report can be rebuilt from a clone.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from rag_audit.groundedness_judge import judge_cost, verdicts_path  # noqa: E402
from rag_audit.trace_store import RESULTS, load_bundle, trace_cost  # noqa: E402

CHAT_CONFIGS = ("baseline", "candidate-rerank", "fix")  # every config that made chat calls
JUDGED_CONFIGS = ("baseline", "fix")

FIGURES = ROOT / "report" / "figures"
NUMBERS = ROOT / "report" / "report-numbers.json"

CATEGORIES = ["fact", "numeric", "multi_hop", "statement_vs_minutes", "date_sensitive", "refusal"]
CATEGORY_LABELS = ["Fact", "Numeric", "Multi-hop", "Statement vs\nminutes", "Date-\nsensitive",
                   "Refusal"]
FAILURES = ["retrieval_miss", "wrong_chunk_first", "citation_mismatch", "missing_refusal",
            "answer_incomplete", "unsupported_claim"]
FAILURE_LABELS = ["Retrieval miss", "Wrong chunk first", "Citation mismatch", "Missing refusal",
                  "Answer incomplete", "Unsupported claim"]
SHORT_CHUNK_CHARS = 150  # a chunk body this short is a cover page or a footer, not content

GREY, BLUE, SKY, INK = "#94a3b8", "#0369a1", "#38bdf8", "#0f172a"


def load_summary(config: str) -> dict:
    path = RESULTS / config / "summary.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m rag_audit reproduce` first")
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_body(text: str | None) -> str:
    """Chunk text without AnythingLLM's <document_metadata> header (what the content really is)."""
    return re.sub(r"(?s)^.*?</document_metadata>", "", text or "").strip()


def context_stats(config: str) -> dict:
    """Short-chunk share and the latency split (app minus LLM time) from the packed chat traces."""
    chats, _ = load_bundle(config)
    ok = [t for t in chats if t.get("status") == "ok"]
    slots = [s for t in ok for s in (t.get("sources") or [])]
    short = sum(len(chunk_body(s.get("text"))) < SHORT_CHUNK_CHARS for s in slots)
    trials_with_short = sum(any(len(chunk_body(s.get("text"))) < SHORT_CHUNK_CHARS
                                for s in (t.get("sources") or [])) for t in ok)
    non_llm = sorted(t["latency_ms"] / 1000 - float(t["metrics"]["duration"]) for t in ok
                     if t.get("latency_ms") and (t.get("metrics") or {}).get("duration") is not None)
    llm = sorted(float(t["metrics"]["duration"]) for t in ok
                 if (t.get("metrics") or {}).get("duration") is not None)

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 2)

    return {
        "ok_trials": len(ok),
        "context_slots": len(slots),
        "short_chunk_slots": short,
        "short_chunk_share": round(short / len(slots), 3) if slots else None,
        "trials_with_short_chunk": trials_with_short,
        "retrieval_and_app_s": {"p50": pct(non_llm, 50), "p90": pct(non_llm, 90)},
        "llm_s": {"p50": pct(llm, 50), "p90": pct(llm, 90), "max": round(max(llm), 2) if llm else None},
    }


def logged_cost() -> dict:
    """Spend that the committed results record, at list prices, and what they cannot meter.

    Chat: token usage of every successful chat trace in the bundles. Judge: token usage of every
    verdict row (all prompt versions). A timed-out attempt has no usage (the client gave up), and
    embedding calls are not traced, so both are counted but not priced.
    """
    chat = {c: 0.0 for c in CHAT_CONFIGS}
    chat_calls = timed_out = max_attempts = 0
    for config in CHAT_CONFIGS:
        chats, _ = load_bundle(config)
        for t in chats:
            attempts = t.get("attempts") or []
            max_attempts = max(max_attempts, len(attempts))
            timed_out += sum(bool(a.get("timeout")) for a in attempts)
            if t.get("status") == "ok":
                chat[config] += trace_cost(t)
                chat_calls += 1
    judge = {c: judge_cost(c) for c in JUDGED_CONFIGS}
    judge_calls = sum(len(verdicts_path(c).read_text(encoding="utf-8").splitlines())
                      for c in JUDGED_CONFIGS)
    total = sum(chat.values()) + sum(judge.values())
    return {"chat_usd": {k: round(v, 4) for k, v in chat.items()},
            "judge_usd": {k: round(v, 4) for k, v in judge.items()},
            "total_usd": round(total, 3), "metered_chat_calls": chat_calls,
            "metered_judge_calls": judge_calls, "unmetered_timed_out_attempts": timed_out,
            "max_billed_attempts_per_trial": max_attempts,
            "not_metered": "timed-out chat attempts and embedding calls"}


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def label_bars(ax, bars, fmt: str) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)


def chart_pass_rate(base: dict, fix: dict) -> Path:
    labels = CATEGORY_LABELS + ["All\n(per case)"]
    b = [base["by_category"][c]["pass_rate"] * 100 for c in CATEGORIES] + [base["pass_rate"] * 100]
    f = [fix["by_category"][c]["pass_rate"] * 100 for c in CATEGORIES] + [fix["pass_rate"] * 100]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    w = 0.38
    label_bars(ax, ax.bar([i - w / 2 for i in x], b, w, color=GREY, label="Baseline (defaults)"), "{:.0f}")
    label_bars(ax, ax.bar([i + w / 2 for i in x], f, w, color=BLUE, label="Fix (rerank)"), "{:.0f}")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Pass rate, %", color=INK, fontsize=9)
    ax.set_title("Pass rate by category, 67 questions x 3 trials per config", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    out = FIGURES / "pass-rate-by-category.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_failures(base: dict, fix: dict) -> Path:
    b = [base["failures_including_flags"].get(k, 0) for k in FAILURES]
    f = [fix["failures_including_flags"].get(k, 0) for k in FAILURES]
    y = list(range(len(FAILURES)))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    h = 0.38
    bb = ax.barh([i + h / 2 for i in y], b, h, color=GREY, label="Baseline")
    fb = ax.barh([i - h / 2 for i in y], f, h, color=BLUE, label="Fix (rerank)")
    for bars in (bb, fb):
        for bar in bars:
            ax.annotate(f"{bar.get_width():.0f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, FAILURE_LABELS)
    ax.set_xlabel("Failed trials (of 201); unsupported claim is a judge flag, not a primary label",
                  color=INK, fontsize=8)
    ax.set_title("Failure types, baseline vs fix", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGURES / "failure-types.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_recall(rows: list[tuple[str, dict, str]]) -> Path:
    ks = ["recall@1", "recall@4", "recall@10", "mrr@10"]
    labels = ["Recall@1", "Recall@4\n(what the LLM sees)", "Recall@10", "MRR@10"]
    fig, ax = plt.subplots(figsize=(7.2, 2.7), dpi=200)
    w = 0.8 / len(rows)
    for j, (name, metrics, colour) in enumerate(rows):
        xs = [i - 0.4 + w / 2 + j * w for i in range(len(ks))]
        label_bars(ax, ax.bar(xs, [metrics[k] for k in ks], w, color=colour, label=name), "{:.2f}")
    ax.set_xticks(range(len(ks)), labels)
    ax.set_ylim(0, 1.32)  # headroom so the legend sits above the bars
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Expected document found", color=INK, fontsize=9)
    ax.set_title("Retrieval on the 56 answerable questions (/vector-search, top 10, 1000-char chunks)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    style(ax)
    fig.tight_layout()
    out = FIGURES / "retrieval-recall-at-k.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base, fix = load_summary("baseline"), load_summary("fix")
    emb = load_summary("candidate-openai-embedder-1000")
    emb_8191 = load_summary("candidate-openai-embedder")
    gate = json.loads((RESULTS / "compare-baseline-vs-fix.json").read_text(encoding="utf-8"))

    recall = {
        "baseline": base["retrieval"]["vector_search_top10"],
        "candidate-openai-embedder-1000": emb["retrieval"]["vector_search_top10"],
        "fix": fix["retrieval"]["vector_search_top10"],
        "candidate-openai-embedder (8191-char chunks, confounded)":
            emb_8191["retrieval"]["vector_search_top10"],
    }
    figures = [
        chart_pass_rate(base, fix),
        chart_failures(base, fix),
        chart_recall([("Baseline (local MiniLM)", recall["baseline"], GREY),
                      ("Embedder swap (text-embedding-3-small)",
                       recall["candidate-openai-embedder-1000"], SKY),
                      ("Fix (local reranker)", recall["fix"], BLUE)]),
    ]
    numbers = {
        "source": "built by report/build-report-figures.py from results/",
        "pass_rate": {"baseline": base["pass_rate"], "fix": fix["pass_rate"]},
        "pass_rate_by_category": {c: {"baseline": base["by_category"][c]["pass_rate"],
                                      "fix": fix["by_category"][c]["pass_rate"]} for c in CATEGORIES},
        "regressgate": {k: gate[k] for k in ("verdict", "base_rate", "head_rate", "delta",
                                             "p_worse", "p_better", "permutations",
                                             "cases_compared")},
        "question_level": gate["question_level"],
        "judge_agreement": {v: {k: gate["judge_agreement"][v][k] for k in ("checked", "agree")}
                            for v in gate["judge_agreement"]},
        "judge_needed_without_verdict": {"baseline": base["judge"]["needed_without_verdict"],
                                         "fix": fix["judge"]["needed_without_verdict"]},
        "audit_cost_logged": logged_cost(),
        "retrieval_vector_search_top10": recall,
        "context": {"baseline": context_stats("baseline"), "fix": context_stats("fix")},
        "prompt_tokens": {"baseline": base["tokens"]["prompt"], "fix": fix["tokens"]["prompt"]},
        "chat_cost_usd_list_price": {"baseline": base["est_cost_usd"]["chat"],
                                     "fix": fix["est_cost_usd"]["chat"]},
    }
    NUMBERS.write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8", newline="\n")
    for path in figures:
        print(f"wrote {path.relative_to(ROOT)}")
    print(f"wrote {NUMBERS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
