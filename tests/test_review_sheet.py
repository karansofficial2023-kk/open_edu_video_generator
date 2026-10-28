import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location("review_sheet", Path(__file__).resolve().parents[1] / "tools" / "review_sheet.py")
review_sheet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review_sheet)


def _contract(tmp: Path) -> Path:
    data = {"title": "Demo", "scenes": [{"scene_number": 1, "scene_title": "Cells", "narration": "x", "shots": [
        {"shot_id": "1.1", "narration": "Stop imbibing.", "subtitle": "Stop imbibing.", "asset_path": "", "visual_type": "realistic_image"},
        {"shot_id": "1.2", "narration": "The cell has a nucleus.", "subtitle": "The cell has a nucleus.", "asset_path": "", "visual_type": "realistic_image"},
        {"shot_id": "1.3", "narration": "Both are acidic.", "subtitle": "Both are acidic.", "asset_path": "", "visual_type": "realistic_image"}]}]}
    path = tmp / "Demo_contract.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ReviewSheetTests(unittest.TestCase):
    def test_edits_and_removals_go_to_a_copy_and_the_sign_off_names_the_reviewer(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            contract = _contract(tmp)
            sheet = review_sheet.export(contract)
            rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig", newline="")))
            self.assertEqual([r["shot_id"] for r in rows], ["1.1", "1.2", "1.3"])
            rows[0].update(decision="edit", new_narration="Water moves into the cell.")
            rows[2].update(decision="remove")
            with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=review_sheet.COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            reviewed, sign = review_sheet.apply(contract, "Dr Rao")
            out = json.loads(reviewed.read_text(encoding="utf-8"))
            shots = out["scenes"][0]["shots"]
            self.assertEqual([s["shot_id"] for s in shots], ["1.1", "1.2"])
            self.assertEqual(shots[0]["narration"], "Water moves into the cell.")
            self.assertEqual(shots[0]["subtitle"], "Water moves into the cell.")
            self.assertEqual(sign["changes"], {"edited": 1, "picture": 0, "removed": 1, "unchanged": 1})
            self.assertEqual(sign["reviewer"], "Dr Rao")
            original = json.loads(contract.read_text(encoding="utf-8"))
            self.assertEqual(len(original["scenes"][0]["shots"]), 3)           # the source contract is never changed

    def test_a_bad_sheet_writes_nothing(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            contract = _contract(tmp)
            sheet = review_sheet.export(contract)
            rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig", newline="")))
            rows[1].update(decision="edit", new_narration="")                  # edit without text
            with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=review_sheet.COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(SystemExit):
                review_sheet.apply(contract, "Dr Rao")
            self.assertFalse((tmp / "Demo_contract.reviewed.json").exists())


if __name__ == "__main__":
    unittest.main()
