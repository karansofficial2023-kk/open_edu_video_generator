import unittest

from app.assets import apply_checklist, is_filler_narration, is_screen_interaction, strip_ui_clause


class ScreenInteractionTest(unittest.TestCase):
    def test_detects_software_operation(self):
        self.assertTrue(is_screen_interaction("Now drag and drop the highlighted piece onto the board"))
        self.assertTrue(is_screen_interaction("Tap the button in this simulation"))

    def test_ordinary_science_sentences_are_not_screen_operations(self):
        for line in ("Tap water contains dissolved minerals.", "The image forms on the screen in this optics experiment.",
                     "A button mushroom has a small round cap.", "Move the slider of the rheostat to change the resistance.",
                     "The slider crank mechanism converts rotation into linear motion."):
            self.assertFalse(is_screen_interaction(line) and "rheostat" not in line and "slider crank" not in line, line)
        self.assertFalse(is_screen_interaction("Tap water contains dissolved minerals."))
        self.assertFalse(is_screen_interaction("The image forms on the screen in this optics experiment."))
        self.assertFalse(is_screen_interaction("A button mushroom has a small round cap."))

    def test_real_screen_operations_are_still_found(self):
        for line in ("Click the button to start.", "Tap on the answer you think is right.", "Drag the labels into their boxes.",
                     "Press the start button.", "Let's take the quiz.", "On the screen, you can click each part."):
            self.assertTrue(is_screen_interaction(line), line)

    def test_content_bearing_lets_sentences_are_not_filler(self):
        self.assertFalse(is_filler_narration("Let's look at a simple circuit with a battery."))
        self.assertFalse(is_filler_narration("Let's see how silver ions move."))
        self.assertTrue(is_filler_narration("Let's recap."))
        self.assertTrue(is_filler_narration("Let's begin."))

    def test_ignores_physical_descriptions(self):
        self.assertFalse(is_screen_interaction("A drop of water falls on the leaf"))
        self.assertFalse(is_screen_interaction("Fossils show how species changed over time"))

    def test_hindi_filler(self):
        for line in ("धन्यवाद!", "सीखते रहिए, फिर मिलेंगे।", "चलिए शुरू करते हैं।"):
            self.assertTrue(is_filler_narration(line), line)
        self.assertFalse(is_filler_narration("पौधे सूर्य के प्रकाश से भोजन बनाते हैं।"))

    def test_filler_narration(self):
        for line in ("Let's answer the given question.", "Keep learning - we believe in you.", "Thank you for watching!", "So, let's recap."):
            self.assertTrue(is_filler_narration(line), line)
        for line in ("Let's see how silver atoms deposit onto the keychain because the current drives silver ions to the cathode in the cell.",
                     "Silver is deposited on the keychain.", "The mass of silver increases over time."):
            self.assertFalse(is_filler_narration(line), line)

    def test_strip_ui_clause(self):
        self.assertEqual(strip_ui_clause("Let's drag the labels into their boxes: the positive terminal is the anode."),
                         "the positive terminal is the anode.")
        self.assertEqual(strip_ui_clause("Click the button. Silver ions gain electrons at the cathode."),
                         "Silver ions gain electrons at the cathode.")
        self.assertEqual(strip_ui_clause("Silver ions gain electrons."), "Silver ions gain electrons.")

    def test_checklist_overrides_a_lenient_verdict_when_most_items_are_missing(self):
        verdict = {"matches": True, "score": 1.0, "contains_text": False, "missing": [], "issues": []}
        checks = [{"item": "beaker", "visible": True}, {"item": "battery", "visible": False}, {"item": "wire", "visible": False}]
        result = apply_checklist(dict(verdict), checks)
        self.assertFalse(result["matches"])
        self.assertEqual(result["score"], 0.33)
        self.assertEqual(result["missing"], ["battery", "wire"])
        self.assertTrue(apply_checklist(dict(verdict), [{"item": "beaker", "visible": True}])["matches"])
        self.assertEqual(apply_checklist(dict(verdict), None), verdict)

    def test_one_missing_item_lowers_the_score_but_keeps_a_good_picture(self):
        verdict = {"matches": True, "score": 0.9, "contains_text": False, "missing": [], "issues": []}
        two = apply_checklist(dict(verdict), [{"item": "leaf", "visible": True}, {"item": "droplet", "visible": False}])
        self.assertTrue(two["matches"])
        self.assertEqual(two["score"], 0.75)
        three = apply_checklist(dict(verdict), [{"item": "a", "visible": True}, {"item": "b", "visible": True}, {"item": "c", "visible": False}])
        self.assertTrue(three["matches"])
        nothing = apply_checklist(dict(verdict), [{"item": "a", "visible": False}, {"item": "b", "visible": False}])
        self.assertFalse(nothing["matches"])

    def test_text_booleans_from_the_model_are_read_correctly(self):
        verdict = {"matches": True, "score": 0.9, "contains_text": False, "missing": [], "issues": []}
        result = apply_checklist(dict(verdict), [{"item": "a", "visible": "false"}, {"item": "b", "visible": "false"}, {"item": "c", "visible": "true"}])
        self.assertFalse(result["matches"], "the string 'false' must not count as visible")
        from app.assets import _as_float, _is_true
        self.assertFalse(_is_true("false"))
        self.assertTrue(_is_true("yes"))
        self.assertEqual(_as_float("0.7"), 0.7)
        self.assertEqual(_as_float("high"), 0.0)
        self.assertEqual(_as_float(None), 0.0)


if __name__ == "__main__":
    unittest.main()
