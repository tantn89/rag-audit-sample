"""The corpus manifest must stay fully pinned and limited to the Federal Reserve Board site."""

import json
import re
import unittest
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "corpus" / "corpus-manifest.json"


class CorpusManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_every_file_has_a_sha256_pin_and_size(self):
        for entry in self.manifest["files"]:
            with self.subTest(entry["name"]):
                self.assertRegex(entry.get("sha256") or "", r"^[0-9a-f]{64}$")
                self.assertGreater(entry.get("bytes") or 0, 0)

    def test_urls_are_https_federalreserve_only(self):
        for entry in self.manifest["files"]:
            with self.subTest(entry["name"]):
                self.assertTrue(entry["url"].startswith("https://www.federalreserve.gov/"))

    def test_names_are_unique_and_safe(self):
        names = [e["name"] for e in self.manifest["files"]]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertRegex(name, r"^[a-z0-9-]+\.pdf$")

    def test_licence_is_recorded(self):
        self.assertTrue(re.search(r"public domain", self.manifest["license"], re.I))


if __name__ == "__main__":
    unittest.main()
