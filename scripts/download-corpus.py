"""Download the audit corpus and verify every file against its SHA-256 pin.

The corpus is the 2025 FOMC meeting statements and minutes published by the Federal Reserve
Board (public domain; cite the Board as the source). URLs and pins live in
corpus/corpus-manifest.json. Files land in data/corpus/ (git-ignored).

Usage:
    python scripts/download-corpus.py               # download + verify, fail on any mismatch
    python scripts/download-corpus.py --write-pins  # maintainer only: record pins for new entries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus" / "corpus-manifest.json"
DEST = ROOT / "data" / "corpus"
USER_AGENT = "rag-audit-sample/0.1 (corpus download)"
ALLOWED_PREFIX = "https://www.federalreserve.gov/"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, target: Path, attempts: int = 3) -> None:
    """Download to a temp file in the target folder, then move into place."""
    if not url.startswith(ALLOWED_PREFIX):
        raise ValueError(f"refusing non-allowlisted URL: {url}")
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, tempfile.NamedTemporaryFile(
                dir=target.parent, delete=False, suffix=".part"
            ) as tmp:
                while chunk := resp.read(1 << 16):
                    tmp.write(chunk)
            Path(tmp.name).replace(target)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last_err = err
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-pins", action="store_true",
                    help="record sha256/bytes for entries that have no pin yet")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    failures = []
    for entry in manifest["files"]:
        target = DEST / entry["name"]
        pinned = entry.get("sha256")
        if not (target.exists() and pinned and sha256_of(target) == pinned):
            fetch(entry["url"], target)
        digest = sha256_of(target)
        if pinned is None:
            if not args.write_pins:
                failures.append(f"{entry['name']}: no pin in manifest")
                continue
            entry["sha256"], entry["bytes"] = digest, target.stat().st_size
            print(f"pinned  {entry['name']} {digest}")
        elif digest != pinned:
            failures.append(f"{entry['name']}: sha256 {digest} != pinned {pinned}")
        else:
            print(f"ok      {entry['name']}")
    if args.write_pins:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print(f"{len(manifest['files'])} files verified in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
