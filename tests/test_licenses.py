import json
import tempfile
import unittest
from pathlib import Path

from app.production import _write_licenses


class LicenseCreditTests(unittest.TestCase):
    def test_shipped_commons_still_is_credited_even_without_a_content_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = Path(tmp, "assets")
            assets.mkdir()
            record = {"shot_id": "1.1", "accepted": True, "path": "a_commons.png", "attempts": [
                {"path": "a_flux.png", "technical": {"ok": True}},
                {"path": "a_commons.png", "technical": {"ok": True}, "semantic": {"error": "reviewer down"},
                 "provenance": {"title": "Leaf", "author": "Jane", "license": "CC BY 4.0"}}]}
            (assets / "shot_1_1.json").write_text(json.dumps(record), encoding="utf-8")
            _write_licenses(Path(tmp))
            credits = json.loads(Path(tmp, "licenses.json").read_text(encoding="utf-8"))["assets"]
            self.assertEqual([c["author"] for c in credits], ["Jane"])

    def test_generated_still_needs_no_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = Path(tmp, "assets")
            assets.mkdir()
            record = {"shot_id": "1.1", "accepted": True, "path": "a_flux.png", "attempts": [{"path": "a_flux.png"}]}
            (assets / "shot_1_1.json").write_text(json.dumps(record), encoding="utf-8")
            _write_licenses(Path(tmp))
            self.assertEqual(json.loads(Path(tmp, "licenses.json").read_text(encoding="utf-8"))["assets"], [])


if __name__ == "__main__":
    unittest.main()
