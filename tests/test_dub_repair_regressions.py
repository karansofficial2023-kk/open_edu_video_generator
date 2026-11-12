import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.translation_qa import repair_protected_content, _numbers_outside_formulas
from app.audio_replacement import _fit_segment


class RepairRegressionTests(unittest.TestCase):
    def repair(self, source, narration):
        segment = SimpleNamespace(source_narration=source, narration=narration,
                                  shot_id="1.1", segment_number=1)
        board = SimpleNamespace(language={"bcp47": "hi-IN"},
                                all_segments=lambda: [(None, segment)])
        repairs, issues = repair_protected_content(board)
        return segment.narration, repairs, issues

    def test_missing_operator_is_repaired_when_numbers_match(self):
        text, repairs, issues = self.repair("43 minus 35 is equal to 8", "43 से 35 8 के बराबर है।")
        self.assertIn("घटाने", text)
        self.assertTrue(repairs)
        self.assertFalse(issues)

    def test_inconsistent_source_is_never_rebuilt(self):
        original = "35 minus 23 10 के बराबर है।"
        text, _, issues = self.repair("35 minus 23 is equal to 10", original)
        self.assertEqual(text, original)
        self.assertTrue(any(i['kind'] == 'source_math_inconsistent' for i in issues))

    def test_conjunction_does_not_merge_values(self):
        self.assertEqual(_numbers_outside_formulas("twenty and five"), {"20": 1, "5": 1})
        self.assertEqual(_numbers_outside_formulas("twenty-five"), {"25": 1})
        self.assertEqual(_numbers_outside_formulas("ये दोनों बल"), {"2": 1})
        self.assertEqual(_numbers_outside_formulas("on both sides"), {"2": 1})
        self.assertEqual(_numbers_outside_formulas("आओ, हमें पता लगाने दो."), {})

    def test_electric_current_terminology(self):
        text, repairs, issues = self.repair("An electric current flows.", "इलेक्ट्रिक धारा बहती है।")
        self.assertEqual(text, "विद्युत धारा बहती है।")
        self.assertTrue(repairs)
        self.assertFalse(issues)

    def test_motor_hand_rule_terms(self):
        text, repairs, issues = self.repair("The middle finger points along arm AB.",
                                           "मध्य अंगूठी हाथ AB के साथ इंगित करती है।")
        self.assertIn("मध्यमा उंगली", text)
        self.assertIn("भुजा AB", text)
        self.assertTrue(repairs)
        self.assertFalse(issues)

    def test_short_hindi_speech_slows_without_overrunning_slot(self):
        config = SimpleNamespace(ffmpeg_path="ffmpeg")
        with patch("app.audio_replacement.audio_duration", return_value=6.0), \
             patch("app.audio_replacement._tool_path", return_value="ffmpeg"), \
             patch("app.audio_replacement._run") as run:
            speed = _fit_segment(Path("input.wav"), Path("output.wav"), 12.0, config, 10.0)
        self.assertEqual(speed, 0.85)
        self.assertIn("atempo=0.850000", " ".join(run.call_args.args[0]))

    def test_long_hindi_speech_still_fits_video_slot(self):
        config = SimpleNamespace(ffmpeg_path="ffmpeg")
        with patch("app.audio_replacement.audio_duration", return_value=8.0), \
             patch("app.audio_replacement._tool_path", return_value="ffmpeg"), \
             patch("app.audio_replacement._run"):
            speed = _fit_segment(Path("input.wav"), Path("output.wav"), 6.0, config, 12.0)
        self.assertAlmostEqual(speed, 8.0 / 6.0)


if __name__ == '__main__':
    unittest.main()
