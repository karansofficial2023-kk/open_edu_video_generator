import unittest

from app.labels import _flag, _num


class ModelOutputParsingTest(unittest.TestCase):
    def test_text_booleans_are_read_by_their_words(self):
        self.assertFalse(_flag("false"))
        self.assertFalse(_flag("No"))
        self.assertTrue(_flag("true"))
        self.assertTrue(_flag(True))
        self.assertFalse(_flag(None))

    def test_unusable_numbers_become_zero_instead_of_crashing(self):
        self.assertEqual(_num("0.8"), 0.8)
        self.assertEqual(_num("high"), 0.0)
        self.assertEqual(_num(None), 0.0)


if __name__ == "__main__":
    unittest.main()
