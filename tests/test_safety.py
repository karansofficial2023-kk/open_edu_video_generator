import unittest
from unittest import mock

from app import safety
from app.config import AppConfig


class SafetyTests(unittest.TestCase):
    def ask(self, answer):
        return mock.patch("app.safety.vision.ask", return_value=answer)

    def test_negative_prompt_always_carries_the_safety_terms(self):
        for base in ("", "blurry, watermark"):
            text = safety.with_safety_negative(base)
            for term in ("nude", "nsfw", "bare skin", "underwear"):
                self.assertIn(term, text)

    def test_clean_image_is_allowed(self):
        with self.ask({"adult_content": False, "body_visible": False, "clothed_only": True, "graphic": False}):
            self.assertIsNone(safety.unsafe_reason("x.png", AppConfig()))

    def test_covered_person_is_allowed(self):
        with self.ask({"adult_content": False, "body_visible": True, "clothed_only": True, "graphic": False}):
            self.assertIsNone(safety.unsafe_reason("x.png", AppConfig()))

    def test_adult_content_bare_body_and_blood_are_blocked(self):
        for answer in ({"adult_content": True}, {"adult_content": False, "body_visible": True, "clothed_only": False},
                       {"adult_content": False, "body_visible": False, "graphic": "true"}):
            with self.ask(answer):
                self.assertTrue(safety.unsafe_reason("x.png", AppConfig()).startswith("unsafe"))

    def test_organs_inside_a_body_outline_are_unsafe_on_their_own(self):
        # regression: glowing organs drawn inside a translucent hips-and-thighs outline passed the old rule ("not fully clothed" was false)
        with self.ask({"adult_content": False, "body_visible": False, "clothed_only": True, "graphic": False, "body_outline": True}):
            self.assertIn("body outline", safety.unsafe_reason("x.png", AppConfig()))

    def test_one_unsafe_answer_out_of_two_rejects_the_picture(self):
        # the vision model is not deterministic: the second look catches what the first one missed
        clean = {"adult_content": False, "body_visible": False, "clothed_only": True, "graphic": False}
        with mock.patch("app.safety.vision.ask", side_effect=[clean, {**clean, "graphic": True}]):
            self.assertIn("blood", safety.unsafe_reason("x.png", AppConfig(), passes=2))
        with mock.patch("app.safety.vision.ask", side_effect=[clean, clean]):
            self.assertIsNone(safety.unsafe_reason("x.png", AppConfig(), passes=2))

    def test_regression_phrases_that_must_never_get_a_generated_picture(self):
        from app.assets import BODY_WORDS, EXTERNAL_GENITAL
        for text in ("The urethra is the passage that transports semen out of the body through the penis.",
                     "A menstrual period typically lasts between two to five days.", "The scrotum holds the testes.", "a sanitary pad"):
            self.assertTrue(EXTERNAL_GENITAL.search(text), text)
        for text in ("The prostate gland adds fluid to the semen.", "The uterus is a muscular organ.", "Cells divide."):
            self.assertFalse(EXTERNAL_GENITAL.search(text), text)
        cleaned = BODY_WORDS.sub("", "glowing uterus and ovaries inside a human body outline of a woman's torso")
        for word in ("torso", "outline", "human body", "woman"):
            self.assertNotIn(word, cleaned)
        self.assertIn("uterus and ovaries", cleaned)

    def test_fails_closed_when_the_model_is_down_or_answers_nonsense(self):
        with mock.patch("app.safety.vision.ask", side_effect=RuntimeError("offline")):
            self.assertIn("could not run", safety.unsafe_reason("x.png", AppConfig()))
        with self.ask({"hello": "world"}):
            self.assertIn("no usable answer", safety.unsafe_reason("x.png", AppConfig()))


if __name__ == "__main__":
    unittest.main()
