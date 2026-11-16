import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.audio_replacement import _coverage
from app.contract_precheck import (precheck_contract_dict, precheck_shot, review_item_is_advisory)
from app.spoken_forms import hindi_number, hindi_only_text, is_formula_token, split_spans, spell_formula
from app.translation_qa import _formula_tokens, repair_protected_content
from app.tts_piper import _pronunciations_without_formulas


def langs(text):
    return [(span.lang, span.text) for span in split_spans(text)]


class StrictFormulaRuleTests(unittest.TestCase):
    def test_formula_is_spoken_in_english_letters_and_numbers(self):
        self.assertEqual(spell_formula("H2SO4"), "H two S O four")
        self.assertEqual(langs("सल्फ्यूरिक अम्ल H2SO4 प्रबल है।"),
                         [("hi", "सल्फ्यूरिक अम्ल"), ("en", "H two S O four"), ("hi", "प्रबल है।")])

    def test_full_equation_is_one_english_span(self):
        spans = langs("2NaOH + H2SO4 → Na2SO4 + 2H2O अभिक्रिया है।")
        self.assertEqual(spans[0][0], "en")
        self.assertIn("plus", spans[0][1])
        self.assertIn("gives", spans[0][1])
        self.assertEqual(spans[-1][0], "hi")

    def test_labels_and_acronyms_are_english(self):
        self.assertEqual(langs("भुजा AB और AC मोटर"),
                         [("hi", "भुजा"), ("en", "A B"), ("hi", "और"), ("en", "A C"), ("hi", "मोटर")])

    def test_digits_become_hindi_words_outside_formulas(self):
        self.assertEqual(hindi_number("25"), "पच्चीस")
        self.assertEqual(hindi_number("1500"), "एक हज़ार पाँच सौ")
        self.assertEqual(langs("5 cm दूरी पर 25% है"), [("hi", "पाँच सेंटीमीटर दूरी पर पच्चीस प्रतिशत है")])

    def test_formulas_are_never_replaced_by_hindi_pronunciations(self):
        kept = _pronunciations_without_formulas({"NaOH": "सोडियम हाइड्रॉक्साइड", "अम्ल": "एसिड"})
        self.assertNotIn("NaOH", kept)
        self.assertIn("अम्ल", kept)

    def test_english_words_that_look_like_elements_are_not_formulas(self):
        for word in ("His", "SO", "NO", "As", "In", "He"):
            self.assertFalse(is_formula_token(word), word)
        self.assertEqual([token for token, _s, _e in _formula_tokens("His SO is NO but HCl and H2O")], ["HCl", "H2O"])

    def test_hindi_asr_comparison_ignores_the_english_spans(self):
        self.assertEqual(hindi_only_text("अम्ल H2SO4 है"), "अम्ल है")
        self.assertGreater(_coverage("अम्ल है", "अम्ल कुछ अजीब शब्द है"), 0.9)

    def test_devanagari_spelled_formula_is_restored_to_latin(self):
        segment = SimpleNamespace(source_narration="Sulphuric acid H2SO4 is strong.",
                                  narration="सल्फ्यूरिक अम्ल एच टू एस ओ फोर प्रबल है।", shot_id="1.1", segment_number=1)
        board = SimpleNamespace(language={"bcp47": "hi-IN"}, all_segments=lambda: [(None, segment)])
        _repairs, issues = repair_protected_content(board)
        self.assertIn("H2SO4", segment.narration)
        self.assertEqual(issues, [])


class ContractPrecheckTests(unittest.TestCase):
    def test_foreign_script_and_function_words_are_removed(self):
        result = precheck_shot("25.1", "hexagonal form, the球 we can conclude", "षट्कोणीय रूप में, the हम निष्कर्ष निकालते हैं", "hi-IN")
        self.assertNotIn("球", result["source"])
        self.assertNotIn(" the ", result["narration"])
        self.assertTrue(any(i["kind"] == "source_asr_garbled" for i in result["issues"]))

    def test_known_mistranslations_are_repaired_from_source_terms(self):
        result = precheck_shot("23.1", "The middle finger shows the direction.", "मध्य अंगूठी दिशा दिखाती है।", "hi-IN")
        self.assertIn("मध्यमा उंगली", result["narration"])
        result = precheck_shot("15.1", "Arm AB touches the slip ring S2.", "हाथ AB स्लीप अंगूठी S2 को छूता है।", "hi-IN")
        self.assertIn("भुजा AB", result["narration"])
        self.assertIn("स्लिप रिंग", result["narration"])

    def test_numbering_cue_is_spoken(self):
        self.assertEqual(precheck_shot("33.1", "2.", "2.", "hi-IN")["narration"], "दूसरा।")

    def test_contract_dict_is_repaired_and_timing_nudged(self):
        data = {"language": {"bcp47": "hi-IN"}, "scenes": [{"narration": "", "shots": [
            {"shot_id": "1.1", "source_narration": "Hello.", "narration": "नमस्ते।", "source_start": 1.0, "source_end": 1.0},
            {"shot_id": "2.1", "source_narration": "Bye.", "narration": "अलविदा।", "source_start": 1.0, "source_end": 3.0}]}]}
        report = precheck_contract_dict(data)
        shots = data["scenes"][0]["shots"]
        self.assertGreater(shots[0]["source_end"], shots[0]["source_start"])
        self.assertGreater(shots[1]["source_start"], shots[0]["source_start"])
        self.assertEqual(report["issue_count"], 0)

    def test_advisory_only_for_low_risk_review_items(self):
        low = {"where": "scene 11.1 source_asr", "problem": "English ASR needs source-audio verification at 1-2s; uncertain words: Now,"}
        risky = {"where": "scene 15.1 source_asr", "problem": "English ASR needs source-audio verification at 1-2s; uncertain words: Arm"}
        self.assertIsNotNone(review_item_is_advisory(low, False))
        self.assertIsNone(review_item_is_advisory(risky, False))
        self.assertIsNone(review_item_is_advisory(low, True))


class ChemistryNotationTests(unittest.TestCase):
    def test_bonds_groups_and_suffixes_are_spoken_as_chemistry(self):
        self.assertEqual(langs("समूह C=O है"), [("hi", "समूह"), ("en", "C double bond O"), ("hi", "है")])
        self.assertEqual(langs("C≡N"), [("en", "C triple bond N")])
        self.assertEqual(langs("R-OH"), [("en", "R dash O H")])
        self.assertEqual(langs("समूह -COOH"), [("hi", "समूह"), ("en", "dash C O O H")])
        self.assertEqual(langs("प्रत्यय -one"), [("hi", "प्रत्यय"), ("en", "suffix own")])

    def test_physics_formulas_are_not_turned_into_bonds(self):
        self.assertEqual(langs("F=BIL"), [("en", "F, equals, B I L")])

    def test_iupac_suffix_swapped_for_carbonyl_group_is_flagged(self):
        from app.translation_qa import check_translation_segment
        issues = check_translation_segment("16.1", "Ketones are named by replacing E by O-N-E.",
                                           "कीटोन को E की जगह C=O से नामित किया जाता है।", "hi-IN")
        self.assertTrue(any(i["kind"] == "iupac_suffix_replaced_by_group" for i in issues))
        self.assertFalse(any(i["kind"] == "iupac_suffix_replaced_by_group" for i in check_translation_segment(
            "16.1", "Ketones are named by replacing E by O-N-E.", "कीटोन को E की जगह O-N-E से नामित किया जाता है।", "hi-IN")))

    def test_audit_confusing_suffix_with_carbonyl_is_advisory(self):
        item = {"where": "scene 16.1 semantic_audit", "source": "replacing the E of an alkane by O -N -E.",
                "translation": "...", "problem": "semantic audit: should be the chemical designation C=O"}
        self.assertIsNotNone(review_item_is_advisory(item, True))


class V21RegressionTests(unittest.TestCase):
    """Defects found by reviewing the real two-video run."""

    def test_split_formulas_are_rejoined_but_spelled_endings_are_not(self):
        from app.contract_precheck import _tidy_spacing
        self.assertEqual(_tidy_spacing("we get R -C -O -N -H2 which"), "we get RCONH2 which")
        self.assertEqual(_tidy_spacing("compound R-C-O-C-L is"), "compound RCOCl is")
        self.assertEqual(_tidy_spacing("replacing E by O-N-E"), "replacing E by O-N-E")

    def test_known_mistranslations_from_the_real_run_are_repaired(self):
        cases = [("These are called alkanals.", "इन्हें अलकनंदा कहा जाता है।", "ऐल्केनल"),
                 ("if the hydroxyl group is replaced", "यदि हाइड्रोक्लोरिक समूह बदला जाए", "हाइड्रॉक्सिल"),
                 ("Fullerenes is another allotrope of carbon.", "फुलेरीन कार्बन का एक अन्य समस्थिति है।", "अपररूप"),
                 ("For example, ethane nitrile.", "उदाहरण के लिए, एथेन नाइट्राइड।", "नाइट्राइल")]
        for source, hindi, expected in cases:
            self.assertIn(expected, precheck_shot("1.1", source, hindi, "hi-IN")["narration"], source)

    def test_double_bond_spoken_by_the_teacher_becomes_the_bond_formula(self):
        result = precheck_shot("8.1", "The C double bond O group is called carbonyl.", "C डबल बॉन्ड O समूह कार्बोनिल कहलाता है।", "hi-IN")
        self.assertTrue(result["narration"].startswith("C=O"))
        self.assertEqual(langs(result["narration"])[0], ("en", "C double bond O"))

    def test_hindi_oblique_numbers_count_as_numbers(self):
        from app.translation_qa import check_translation_segment
        issues = check_translation_segment("18.1", "Out of these four, it shares three electrons.",
                                           "इन चारों में से, यह तीन इलेक्ट्रॉन साझा करता है।", "hi-IN")
        self.assertFalse([i for i in issues if i["kind"] == "number_mismatch"])

    def test_cue_check_tolerates_words_placed_slightly_outside_the_cue(self):
        import json, tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from app.audio_replacement import _cue_speech_checks
        with tempfile.TemporaryDirectory() as folder:
            transcript = Path(folder) / "t.json"
            transcript.write_text(json.dumps({"segments": [{"words": [
                {"word": "कार्बोक्सिलिक", "start": 10.0, "end": 10.5}, {"word": "अम्ल", "start": 10.6, "end": 10.9}]}]}), encoding="utf-8")
            segment = SimpleNamespace(shot_id="1.1", narration="कार्बोक्सिलिक अम्ल")
            checks = _cue_speech_checks(transcript, [{"segment_ids": ["1.1"], "start_seconds": 10.8, "slot_seconds": 1.4}], [segment])
        self.assertEqual(checks[0]["status"], "passed")
        self.assertGreater(checks[0]["similarity"], 0.9)


class VoiceMatchTests(unittest.TestCase):
    def test_pitch_ratio_is_clamped_and_ignores_small_gaps(self):
        from app import voice_match as vm
        self.assertEqual(vm.pitch_ratio(110.0, 109.0), 1.0)                  # within 2%: leave the voice alone
        self.assertAlmostEqual(vm.pitch_ratio(120.0, 100.0), 1.2)             # clamped to +20%
        self.assertAlmostEqual(vm.pitch_ratio(50.0, 100.0), 1 / 1.2)          # clamped to -20%
        self.assertEqual(vm.pitch_ratio(0.0, 100.0), 1.0)                     # unknown pitch: no change

    def test_gain_is_limited_and_filter_is_built(self):
        from app import voice_match as vm
        self.assertEqual(vm.gain_db(1.0, 0.1), vm.MAX_GAIN_DB)
        self.assertEqual(vm.gain_db(0.0, 0.1), 0.0)
        self.assertEqual(vm.match_filter(1.0, 22050, 0.1), "")
        chain = vm.match_filter(0.95, 22050, -2.0)
        self.assertIn("asetrate=", chain)
        self.assertIn("volume=-2.00dB", chain)

    def test_median_pitch_of_a_pure_tone(self):
        import numpy as np
        from app import voice_match as vm
        rate = 22050
        tone = (0.3 * np.sin(2 * np.pi * 120.0 * np.arange(rate) / rate)).astype(np.float32)
        self.assertAlmostEqual(vm.median_f0(tone, rate), 120.0, delta=4.0)


class NarratorGenderTests(unittest.TestCase):
    def test_folder_default_gender_applies_unless_the_video_has_its_own(self):
        import json, tempfile
        from pathlib import Path
        from app.voice_profile import _override_gender
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "Lesson.mp4"
            self.assertEqual(_override_gender(video), ("", None))
            (Path(folder) / "default.voice.json").write_text(json.dumps({"gender": "male"}), encoding="utf-8")
            self.assertEqual(_override_gender(video)[0], "male")
            (Path(folder) / "Lesson.voice.json").write_text(json.dumps({"gender": "female"}), encoding="utf-8")
            self.assertEqual(_override_gender(video)[0], "female")


if __name__ == "__main__":
    unittest.main()


class MissingContentCheck(unittest.TestCase):
    def test_dropped_second_sentence_is_flagged(self):
        from app.contract_precheck import precheck_shot
        en = ("It is a mixture of sodium salt of phenolphthalein, methyl red, thymol blue, methanol, propanol, etc. "
              "Olfactory indicators are substances that indicate the nature of a medium by a change in odor.")
        hi = "यह फेनोल्फ्थालीन, मिथाइल रेड, थाइमोल ब्लू, मीथेनॉल, प्रोपेनॉल आदि के सोडियम नमक का मिश्रण है।"
        kinds = [i["kind"] for i in precheck_shot("61.1", en, hi, "hi")["issues"]]
        self.assertIn("hindi_missing_content", kinds)

    def test_complete_translation_is_not_flagged(self):
        from app.contract_precheck import precheck_shot
        en = "Indicators are found abundantly around us. They change colour in acids and bases."
        hi = "हमारे आस-पास सूचक भरपूर मिलते हैं। वे अम्ल और क्षार में रंग बदलते हैं।"
        kinds = [i["kind"] for i in precheck_shot("1.1", en, hi, "hi")["issues"]]
        self.assertNotIn("hindi_missing_content", kinds)
