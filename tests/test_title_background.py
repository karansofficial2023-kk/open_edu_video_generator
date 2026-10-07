import json
import tempfile
import unittest
from pathlib import Path

from app.assets import StillProducer
from app.config import AppConfig
from app.contract import read_contract_json


def board():
    shot = lambda sid, vt, **kw: {"shot_id": sid, "heading": "h", "narration": "Plants make food using light energy.", "duration": 4,
                                  "visual_type": vt, "media_type": "photo", "image_prompt": "a leaf", "labels": [], **kw}
    return read_contract_json({"schema_version": "2.0", "title": "T", "language": {"bcp47": "en-IN"},
                               "scenes": [{"scene_number": 1, "title": "S", "shots": [shot("1.1", "title_card"), shot("1.2", "realistic_image")]}]})


class TitleBackgroundTest(unittest.TestCase):
    def jobs(self, generate):
        config = AppConfig()
        config.production.generate_title_backgrounds = generate
        with tempfile.TemporaryDirectory() as folder:
            return [job.key for job in StillProducer(board(), Path(folder), config)._jobs()]

    def test_title_cards_are_generated_by_default(self):
        self.assertEqual(self.jobs(True), ["1_1", "1_2"])

    def test_gradient_titles_leave_only_real_photos_to_generate(self):
        self.assertEqual(self.jobs(False), ["1_2"])


if __name__ == "__main__":
    unittest.main()
