"""Check that every evidence quote in the eval set appears on its stated page of the corpus PDF.

Needs pypdf (`pip install -r requirements-dev.txt`) and the downloaded corpus in data/corpus/.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_audit.eval_cases import EvalCase


def _squash(text: str) -> str:
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def check_evidence(cases: list[EvalCase], corpus_dir: Path) -> list[str]:
    from pypdf import PdfReader

    pages: dict[str, list[str]] = {}
    problems = []
    for case in cases:
        for ev in case.evidence:
            doc, page, quote = ev.get("doc"), ev.get("page"), ev.get("quote", "")
            if doc not in pages:
                path = corpus_dir / doc
                if not path.exists():
                    problems.append(f"{case.id}: {doc} not found in {corpus_dir}")
                    pages[doc] = []
                    continue
                pages[doc] = [_squash(p.extract_text() or "") for p in PdfReader(str(path)).pages]
            texts = pages[doc]
            if not texts:
                continue
            if not isinstance(page, int) or not 1 <= page <= len(texts):
                problems.append(f"{case.id}: {doc} has no page {page}")
            elif _squash(quote) not in texts[page - 1]:
                problems.append(f"{case.id}: quote not on {doc} p{page}: {quote[:60]!r}")
    return problems
