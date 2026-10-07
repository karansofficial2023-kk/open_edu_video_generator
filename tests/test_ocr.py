import unittest

from app.ocr import has_stray_text


class StrayTextTest(unittest.TestCase):
    def test_noticeable_lettering(self):
        self.assertTrue(has_stray_text(["NGLL"]))
        self.assertTrue(has_stray_text(["100", "90"]))
        self.assertTrue(has_stray_text(["ab", "cd"]))

    def test_nothing_or_a_lone_short_mark(self):
        self.assertFalse(has_stray_text([]))
        self.assertFalse(has_stray_text(None))
        self.assertFalse(has_stray_text(["00"]))


if __name__ == "__main__":
    unittest.main()
