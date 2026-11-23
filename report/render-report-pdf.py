"""Render report/rag-audit-report.md to report/rag-audit-report.pdf.

    python report/render-report-pdf.py

Markdown -> HTML with the `markdown` package, then HTML -> PDF with headless Microsoft Edge
(`--print-to-pdf`), so no LaTeX or browser-automation dependency is needed. Set EDGE_PATH to use
another Chromium-based browser. Run report/build-report-figures.py first for the charts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from pypdf import PdfReader

REPORT_DIR = Path(__file__).resolve().parent
SOURCE = REPORT_DIR / "rag-audit-report.md"
OUTPUT = REPORT_DIR / "rag-audit-report.pdf"
TITLE = "RAG answer audit: AnythingLLM 1.20.0 on the FOMC 2025 corpus (sample)"
PAGE_RANGE = (3, 5)  # the deliverable is a 3-5 page report

CSS = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm;
        @bottom-right { content: "Page " counter(page) " of " counter(pages);
                        font: 7.5pt 'Segoe UI', Arial, sans-serif; color: #64748b; }
        @bottom-left { content: "Sample audit on an open-source project. Not a client engagement.";
                       font: 7.5pt 'Segoe UI', Arial, sans-serif; color: #64748b; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 9.4pt/1.42 'Segoe UI', Arial, sans-serif; color: #0f172a; margin: 0; }
.cover { border-top: 6px solid #0369a1; background: #f8fafc; padding: 14px 18px 12px;
         margin-bottom: 10px; }
.cover .eyebrow { color: #0369a1; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
                  font-size: 8.5pt; margin: 0 0 4px; }
.cover h1 { font-size: 18pt; line-height: 1.2; margin: 0 0 8px; }
.cover .label { display: inline-block; background: #e0f2fe; border: 1px solid #bae6fd;
                color: #075985; border-radius: 6px; padding: 3px 8px; margin: 0 0 6px;
                font-weight: 600; }
.cover .meta { color: #475569; margin: 0; font-size: 8.5pt; }
h2 { color: #0369a1; font-size: 12pt; margin: 12px 0 4px; break-after: avoid; }
p { margin: 4px 0 6px; }
p:has(+ table) { break-after: avoid; }  /* keep a table's lead-in line with the table */
ul { margin: 3px 0 6px; padding-left: 18px; }
li { margin: 1px 0; }
table { border-collapse: collapse; width: 100%; margin: 5px 0 8px; font-size: 8.2pt;
        break-inside: avoid; }  /* every table is short; keep each on one page */
tr { break-inside: avoid; }
th { background: #f1f5f9; text-align: left; font-weight: 600; }
th, td { border: 1px solid #e2e8f0; padding: 2.5px 5px; vertical-align: top; }
code { font: 8.2pt Consolas, 'Courier New', monospace; background: #f1f5f9; padding: 0 2px;
       border-radius: 3px; }
pre { background: #f1f5f9; padding: 6px 8px; border-radius: 4px; white-space: pre-wrap;
      break-inside: avoid; }
pre code { padding: 0; }
img { display: block; width: 100%; max-width: 165mm; margin: 4px auto 6px; break-inside: avoid; }
strong { color: #0f172a; }
"""


def find_browser() -> str:
    candidates = [os.environ.get("EDGE_PATH"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  shutil.which("msedge"), shutil.which("microsoft-edge")]
    for path in candidates:
        if path and Path(path).exists():
            return path
    raise SystemExit("Microsoft Edge not found; set EDGE_PATH to a Chromium-based browser.")


def build_html() -> str:
    body = markdown.markdown(SOURCE.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    # <base> makes the relative figure paths resolve while the HTML sits in a temp directory
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<base href="{REPORT_DIR.as_uri()}/"><title>{TITLE}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main() -> int:
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found")
    browser = find_browser()
    with tempfile.TemporaryDirectory(prefix="rag-audit-report-") as tmp:
        html = Path(tmp) / "rag-audit-report.html"
        html.write_text(build_html(), encoding="utf-8")
        # A throwaway profile keeps this run apart from any open browser window and exits cleanly.
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-pdf-header-footer", f"--user-data-dir={Path(tmp) / 'profile'}",
               f"--print-to-pdf={OUTPUT}", html.as_uri()]
        try:
            subprocess.run(cmd, check=True, timeout=120, capture_output=True)
        except subprocess.CalledProcessError as err:
            raise SystemExit(f"browser exited {err.returncode}: {err.stderr.decode(errors='replace')[:500]}")
        except subprocess.TimeoutExpired:
            raise SystemExit("browser did not finish printing within 120 s") from None
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        raise SystemExit("no PDF was written")
    pages = len(PdfReader(OUTPUT).pages)
    print(f"wrote {OUTPUT.relative_to(REPORT_DIR.parent)} ({pages} pages)")
    if not PAGE_RANGE[0] <= pages <= PAGE_RANGE[1]:
        print(f"page count {pages} is outside {PAGE_RANGE[0]}-{PAGE_RANGE[1]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
