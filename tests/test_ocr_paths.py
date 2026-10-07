import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import ocr


class OcrPathTest(unittest.TestCase):
    @unittest.skipIf(ocr._get_engine() is None, "rapidocr is not installed")
    def test_lettering_is_found_even_in_a_folder_with_a_tamil_name(self):
        folder = Path(tempfile.mkdtemp()) / "ஒளிச்சேர்க்கை"
        folder.mkdir()
        try:
            image = Image.new("RGB", (1344, 768), (230, 235, 240))
            draw = ImageDraw.Draw(image)
            draw.text((200, 300), "BEAKER 100 ML", fill=(0, 0, 0), font=ImageFont.load_default(size=90))
            path = folder / "still.png"
            image.save(path)
            words = ocr.stray_text(path)
            self.assertIsNotNone(words, "OCR must not silently switch itself off for a non-ASCII folder name")
            self.assertTrue(ocr.has_stray_text(words), words)
        finally:
            shutil.rmtree(folder.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
