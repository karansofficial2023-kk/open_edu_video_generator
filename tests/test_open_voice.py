import unittest

from app import tts_edge, tts_indic
from app.config import AppConfig


class NumberTest(unittest.TestCase):
    def test_telugu_and_kannada_use_the_number_library(self):
        self.assertEqual(tts_indic.expand_numbers("6 వోల్ట్", "te"), "ఆరు వోల్ట్")
        self.assertEqual(tts_indic.expand_numbers("12", "kn"), "ಹನ್ನೆರಡು")

    def test_tamil_uses_exact_words_then_digit_by_digit(self):
        self.assertEqual(tts_indic.expand_numbers("6", "ta"), "ஆறு")
        self.assertEqual(tts_indic.expand_numbers("20", "ta"), "இருபது")
        self.assertEqual(tts_indic.expand_numbers("25", "ta"), "இரண்டு ஐந்து")           # no sandhi rules: digit by digit
        self.assertEqual(tts_indic.expand_numbers("2.5", "ta"), "இரண்டு புள்ளி ஐந்து")

    def test_text_without_digits_is_untouched(self):
        self.assertEqual(tts_indic.expand_numbers("ஒளிச்சேர்க்கை", "ta"), "ஒளிச்சேர்க்கை")


class PronunciationTest(unittest.TestCase):
    def test_longest_term_wins_and_prepare_collapses_space(self):
        table = {"CO2": "கார்பன் டை ஆக்சைடு", "CO": "கார்பன் மோனாக்சைடு"}
        self.assertEqual(tts_indic.apply_pronunciations("CO2 and CO", table), "கார்பன் டை ஆக்சைடு and கார்பன் மோனாக்சைடு")
        self.assertEqual(tts_indic.prepare_text("  CO2   6 ", "ta", table), "கார்பன் டை ஆக்சைடு ஆறு")
        self.assertEqual(tts_indic.apply_pronunciations("x", None), "x")


class CaptionTest(unittest.TestCase):
    def test_words_are_ordered_inside_the_audible_range(self):
        captions = tts_indic.distribute_captions("one two three, four five.", duration=5.0, lead=0.2, tail=0.3)
        self.assertEqual([c.text for c in captions], ["one", "two", "three,", "four", "five."])
        self.assertAlmostEqual(captions[0].start, 0.2, places=2)
        self.assertLessEqual(captions[-1].end, 4.7 + 1e-6)
        for earlier, later in zip(captions, captions[1:]):
            self.assertLessEqual(earlier.end, later.start + 1e-6)
            self.assertLess(earlier.start, earlier.end)

    def test_longer_words_get_more_time_and_phrase_marks_a_pause(self):
        captions = tts_indic.distribute_captions("a incredibly b", duration=6.0)
        self.assertGreater(captions[1].end - captions[1].start, captions[0].end - captions[0].start)
        paused = tts_indic.distribute_captions("ab, cd", duration=4.0)
        self.assertGreater(paused[1].start - paused[0].end, 0.1)

    def test_empty_text(self):
        self.assertEqual(tts_indic.distribute_captions("   ", 3.0), [])


class ProviderTest(unittest.TestCase):
    def config(self, provider):
        config = AppConfig()
        config.production.tts_provider = provider
        config.production.open_tts.python = "Z:/missing/python.exe"
        return config

    def test_english_and_edge_mode_never_use_the_open_voice(self):
        self.assertEqual(tts_edge.chosen_provider("en", self.config("open")), "edge")
        self.assertEqual(tts_edge.chosen_provider("ta", self.config("edge")), "edge")

    def test_auto_falls_back_but_strict_open_mode_refuses(self):
        self.assertEqual(tts_edge.chosen_provider("ta", self.config("auto")), "edge")
        with self.assertRaises(ValueError):
            tts_edge.chosen_provider("ta", self.config("open"))


if __name__ == "__main__":
    unittest.main()
