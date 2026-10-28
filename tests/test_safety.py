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

    def test_fails_closed_when_the_model_is_down_or_answers_nonsense(self):
        with mock.patch("app.safety.vision.ask", side_effect=RuntimeError("offline")):
            self.assertIn("could not run", safety.unsafe_reason("x.png", AppConfig()))
        with self.ask({"hello": "world"}):
            self.assertIn("no usable answer", safety.unsafe_reason("x.png", AppConfig()))


if __name__ == "__main__":
    unittest.main()
