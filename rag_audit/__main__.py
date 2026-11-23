"""Audit command line. Reproduce the published numbers from committed traces (no app, no LLM):

    python -m rag_audit reproduce

Full re-run against a running stack (scripts/start-stack.ps1), one config at a time:

    python -m rag_audit full-rerun --configs baseline fix --budget-usd 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_audit.anythingllm_client import ApiError, make_client, settings_snapshot
from rag_audit.audit_configs import CONFIGS, SLUG, get_config
from rag_audit.eval_cases import load_cases
from rag_audit.trace_store import RESULTS, ROOT, load_traces, pack, write_json


def cmd_validate(args) -> int:
    cases = load_cases()
    print(f"{len(cases)} cases OK")
    if args.corpus_dir:
        from rag_audit.corpus_evidence_check import check_evidence
        problems = check_evidence(cases, Path(args.corpus_dir))
        for p in problems:
            print("  " + p)
        print(f"evidence quotes: {'all found' if not problems else f'{len(problems)} problems'}")
        return 1 if problems else 0
    return 0


def cmd_apply_config(args) -> int:
    cfg = get_config(args.config)
    ws = make_client().update_workspace(SLUG, cfg.workspace)
    snap = settings_snapshot(ws)
    wrong = {k: (v, snap.get(k)) for k, v in cfg.workspace.items() if snap.get(k) != v}
    if wrong:
        print(f"settings not applied: {wrong}", file=sys.stderr)
        return 1
    print(json.dumps(snap, indent=2))
    return 0


def cmd_reembed(args) -> int:
    """Rebuild the workspace's vectors with the embedder the app is running now.

    The documents stay parsed in storage/documents; only their vectors are dropped (workspace
    delete), the app's per-document vector cache is cleared so nothing from the old embedder is
    reused, and the same documents are embedded again. This is what the app's UI does after an
    embedder change ("reset vector database").
    """
    client = make_client()
    ws = client.workspace(SLUG)
    if not ws or not ws.get("documents"):
        print(f"workspace {SLUG!r} has no documents to re-embed", file=sys.stderr)
        return 1
    docpaths = sorted(d["docpath"] for d in ws["documents"])
    client.delete_workspace(SLUG)
    cache = ROOT / "storage" / "vector-cache"
    removed = 0
    for path in cache.glob("*.json"):
        path.unlink()
        removed += 1
    client.ensure_workspace(SLUG)
    res = client.update_embeddings(SLUG, docpaths, [])
    count = len((res.get("workspace") or {}).get("documents") or [])
    print(f"re-embedded {count}/{len(docpaths)} documents (cleared {removed} cached vector files)")
    return 0 if count == len(docpaths) else 1


def cmd_retrieval(args) -> int:
    from rag_audit.trial_runner import probe_retrieval
    n = probe_retrieval(make_client(), args.config, load_cases(), args.top_n)
    print(f"[{args.config}] retrieval probed for {n} cases")
    return 0


def cmd_run(args) -> int:
    from rag_audit.groundedness_judge import judge_cost
    from rag_audit.trial_runner import run_trials
    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    judged = sum(judge_cost(c) for c in CONFIGS)
    stats = run_trials(make_client(), args.config, cases, args.trials, timeout=args.timeout,
                       max_attempts=args.max_attempts, concurrency=args.concurrency,
                       budget_usd=args.budget_usd, extra_spent_usd=judged,
                       retry_timeouts=getattr(args, "retry_timeouts", False))
    print(json.dumps(stats))
    return 3 if stats["stopped_for_budget"] else 0


def cmd_judge(args) -> int:
    from rag_audit.groundedness_judge import answer_key, append_verdict, judge_trace, load_verdicts
    from rag_audit.results_report import grade_all
    chats, _ = load_traces(args.config)
    done = load_verdicts(args.config)
    todo = [t for t, g in grade_all(load_cases(), chats, args.config) if g.needs_judge
            and answer_key(t["case_id"], t["trial"], t.get("answer") or "") not in done]
    print(f"[{args.config}] {len(todo)} trials need a groundedness verdict")

    def one(t: dict) -> bool:  # one call per trial; a failure is left for the next run
        key = answer_key(t["case_id"], t["trial"], t.get("answer") or "")
        try:
            verdict = judge_trace(t)
        except (RuntimeError, ValueError, OSError) as err:
            print(f"  {key}: judge failed ({err}); left for the next run", file=sys.stderr)
            return False
        append_verdict(args.config, {"key": key, **verdict})
        return True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        done_ok = sum(pool.map(one, todo))
    print(f"[{args.config}] {done_ok}/{len(todo)} verdicts saved")
    return 0


def report(config: str, from_bundle: bool) -> dict:
    from rag_audit.results_report import regressgate_report, summarize
    from rag_audit.results_report import retrieval_metrics
    cases = load_cases()
    chats, retrieval = load_traces(config, from_bundle)
    if not chats:  # retrieval-only candidate (probed with /vector-search, never chatted)
        summary = {"config": config, "description": get_config(config).description,
                   "embedder": get_config(config).embedder,
                   "retrieval": {"vector_search_top10": retrieval_metrics(cases, retrieval)}}
        write_json(RESULTS / config / "summary.json", summary)
        print(f"[{config}] retrieval only: {summary['retrieval']['vector_search_top10']}")
        return summary
    trials = max((t["trial"] for t in chats), default=0)
    summary, graded = summarize(config, cases, chats, retrieval, trials)
    write_json(RESULTS / config / "summary.json", summary)
    rows = [{"case_id": g.case_id, "trial": g.trial, "status": g.status, "passed": g.passed,
             "failure": g.failure, "flags": g.flags, "missing": g.missing,
             "doc_recall": g.doc_recall, "latency_ms": t.get("latency_ms")} for t, g in graded]
    path = RESULTS / config / "trial-grades.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    regressgate_report(config, cases, graded, trials).save(RESULTS / config / "regressgate-report.json")
    print(f"[{config}] pass rate {summary['pass_rate']} over {summary['trials_graded']} graded trials; "
          f"failures {summary['failures_primary']}")
    return summary


def cmd_report(args) -> int:
    report(args.config, args.from_bundle)
    return 0


def cmd_pack(args) -> int:
    print(f"packed {pack(args.config)}")
    return 0


def compare(base: str, head: str) -> int:
    from regressgate import GateConfig, RunReport, compare_reports
    from regressgate.markdown_report import render_gate

    from rag_audit.results_report import judge_agreement
    result = compare_reports(RunReport.load(RESULTS / base / "regressgate-report.json"),
                             RunReport.load(RESULTS / head / "regressgate-report.json"),
                             GateConfig(alpha=0.05))
    text = render_gate(result)
    (RESULTS / f"compare-{base}-vs-{head}.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    write_json(RESULTS / f"compare-{base}-vs-{head}.json", {
        "verdict": result.verdict, "base_rate": result.base_rate, "head_rate": result.head_rate,
        "delta": result.delta, "p_worse": result.p_worse, "p_better": result.p_better,
        "cases_compared": result.cases_compared, "judge_agreement": judge_agreement()})
    print(text)
    return 0


def cmd_compare(args) -> int:
    return compare(args.base, args.head)


def cmd_reproduce(args) -> int:
    for name in args.configs:
        report(name, from_bundle=True)
    return compare(args.configs[0], args.configs[-1]) if len(args.configs) > 1 else 0


def cmd_full_rerun(args) -> int:
    for name in args.configs:
        ns = argparse.Namespace(config=name, top_n=10, trials=args.trials, only=None,
                                timeout=args.timeout, max_attempts=args.max_attempts,
                                concurrency=args.concurrency, budget_usd=args.budget_usd)
        for step in (cmd_apply_config, cmd_retrieval, cmd_run, cmd_judge, cmd_pack):
            code = step(ns)
            if code:
                print(f"[{name}] {step.__name__} exited {code}; stopping", file=sys.stderr)
                return code
        report(name, from_bundle=False)
    return compare(args.configs[0], args.configs[-1]) if len(args.configs) > 1 else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag_audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate-cases", help="check the eval set (and quotes, with --corpus-dir)")
    v.add_argument("--corpus-dir", help=f"e.g. {ROOT / 'data' / 'corpus'}")
    sub.add_parser("reembed", help="drop and rebuild the workspace vectors with the current "
                                   "embedder").set_defaults(fn=cmd_reembed)
    for name, fn, hlp in (("apply-config", cmd_apply_config, "set the workspace settings"),
                          ("retrieval", cmd_retrieval, "probe /vector-search for every case"),
                          ("judge", cmd_judge, "groundedness verdicts where grading needs one"),
                          ("pack", cmd_pack, "bundle raw traces into results/<config>/"),
                          ("report", cmd_report, "grade traces and write results/<config>/")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("config", choices=sorted(CONFIGS))
        p.set_defaults(fn=fn)
        if name == "retrieval":
            p.add_argument("--top-n", type=int, default=10)
        if name == "report":
            p.add_argument("--from-bundle", action="store_true")
    r = sub.add_parser("run", help="run chat trials (resumable, budgeted)")
    r.add_argument("config", choices=sorted(CONFIGS))
    r.add_argument("--only", nargs="+", help="case ids")
    r.add_argument("--retry-timeouts", action="store_true",
                   help="one extra pass over trials saved as timeouts (bounded by --max-attempts)")
    f = sub.add_parser("full-rerun", help="every step, live, for each config")
    for p in (r, f):
        p.add_argument("--trials", type=int, default=3)
        p.add_argument("--timeout", type=float, default=180, help="per-call seconds")
        p.add_argument("--max-attempts", type=int, default=2, help="billed calls per trial, max")
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--budget-usd", type=float, default=3.0)
    f.add_argument("--configs", nargs="+", default=["baseline", "fix"], choices=sorted(CONFIGS))
    c = sub.add_parser("compare", help="regressgate verdict for two reported configs")
    c.add_argument("base")
    c.add_argument("head")
    rp = sub.add_parser("reproduce", help="recompute every number from committed traces")
    rp.add_argument("--configs", nargs="+", default=["baseline", "fix"], choices=sorted(CONFIGS))
    v.set_defaults(fn=cmd_validate)
    r.set_defaults(fn=cmd_run)
    f.set_defaults(fn=cmd_full_rerun)
    c.set_defaults(fn=cmd_compare)
    rp.set_defaults(fn=cmd_reproduce)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except ApiError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
