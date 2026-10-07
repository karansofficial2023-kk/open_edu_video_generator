import unittest

from app.tts_indic import _speech_weight, distribute_captions


class CaptionWeightTests(unittest.TestCase):
    def test_combining_marks_count_as_spoken_length(self):
        self.assertGreater(_speech_weight("ஒளிச்சேர்க்கை", "ta", None), 8)

    def test_unreadable_latin_term_is_nearly_silent_but_mapped_term_is_spoken(self):
        self.assertEqual(_speech_weight("ATP", "ta", None), 1.0)
        self.assertGreater(_speech_weight("CO2", "ta", {"CO2": "கார்பன் டை ஆக்சைடு"}), 8)

    def test_digits_are_weighted_as_their_spoken_word(self):
        self.assertGreater(_speech_weight("6", "ta", None), 1)

    def test_timings_cover_the_audio_without_overlap(self):
        caps = distribute_captions("ஒளிச்சேர்க்கை ATP ஆகும்.", 4.0, 0.2, 0.2, "ta")
        self.assertTrue(3.0 < caps[-1].end <= 3.8)
        self.assertTrue(all(a.end <= b.start + 1e-6 for a, b in zip(caps, caps[1:])))
        self.assertLess(caps[1].end - caps[1].start, caps[0].end - caps[0].start)


if __name__ == "__main__":
    unittest.main()
