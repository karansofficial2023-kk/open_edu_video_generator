import unittest

from app import video_text_translation as v


def line(text, x0, y0, x1, y1, conf=0.99):
    return v.Line(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], conf)


class BlockGroupingTests(unittest.TestCase):
    def test_wrapped_bullet_is_one_block_and_two_bullets_are_two_blocks(self):
        # the Alloys summary slide: bullet 1 is three wrapped lines, bullet 2 starts after paragraph spacing
        lines = [line("Steel: It is much harder than iron,", 102, 274, 621, 308),
                 line("used for making railway lines and", 105, 315, 598, 347),
                 line("bridges", 104, 349, 224, 389),
                 line("Brass: It is stronger than copper,", 133, 416, 610, 451),
                 line("used to manufacture nuts, bolts,", 105, 454, 581, 489),
                 line("utensils and coins", 107, 495, 370, 524)]
        blocks = v.group_blocks(lines)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0].text.startswith("Steel:"))
        self.assertTrue(blocks[1].text.startswith("Brass:"))
        self.assertEqual([len(b.lines) for b in blocks], [3, 3])

    def test_separate_labels_stay_separate(self):
        blocks = v.group_blocks([line("Aircrafts", 183, 50, 296, 86), line("Pressure cookers", 589, 58, 815, 92)])
        self.assertEqual(len(blocks), 2)

    def test_a_label_colon_line_starts_a_new_block_even_when_close(self):
        blocks = v.group_blocks([line("Alloys are mixtures", 100, 100, 400, 132), line("Steel: strong", 100, 134, 300, 166)])
        self.assertEqual(len(blocks), 2)


class TextRepairTests(unittest.TestCase):
    def test_ocr_joined_words_are_split_but_real_words_are_not_touched(self):
        self.assertEqual(v.fix_spacing("applications in thereal world"), "applications in the real world")
        self.assertEqual(v.fix_spacing("Aluminium is lightweight"), "Aluminium is lightweight")

    def test_lesson_vocabulary_helps_repair(self):
        v.set_vocabulary("railway lines bridges".split())
        try:
            self.assertEqual(v.fix_spacing("making railwaylines"), "making railway lines")
        finally:
            v.set_vocabulary([])

    def test_bullets_and_whitespace_are_removed_from_line_text(self):
        self.assertEqual(v.clean_line_text("O Steel: strong"), "Steel: strong")
        self.assertEqual(v.clean_line_text("● Alloys"), "Alloys")

    def test_bullet_read_as_a_letter_is_cut_off_the_ocr_box(self):
        def fake_ocr(_frame):
            return [([[102, 274], [621, 274], [621, 308], [102, 308]], "O Steel: It is much harder than iron,", 0.99)], None
        (cut,) = v._ocr_lines(fake_ocr, None, 0.8)
        self.assertEqual(cut.text, "Steel: It is much harder than iron,")
        self.assertGreater(cut.x0, 102 + 10)           # the circle itself is no longer inside the box


class ValidationTests(unittest.TestCase):
    def test_oblique_hindi_forms_are_accepted_for_materials(self):
        self.assertTrue(v._translation_valid("Brass: It is stronger than copper", "पीतल: यह तांबे से अधिक मजबूत है"))
        self.assertFalse(v._translation_valid("Brass shines", "चांदी चमकती है"))        # silver swapped for brass

    def test_wording_is_harmonised(self):
        self.assertEqual(v._safe_translation("Steel is much harder than iron", "स्टील लोहे से बहुत कठिन है"),
                         "इस्पात लोहे से बहुत कठोर है")
        self.assertEqual(v._safe_translation("Silvery shine", "चांदी का चमक"), "चांदी जैसी चमक")
        self.assertEqual(v._safe_translation("It is hard to see", "देखना कठिन है"), "देखना कठिन है")   # real difficulty stays

    def test_numbers_and_formulas_must_survive(self):
        self.assertTrue(v._protected_tokens_ok("Steel has 0.5 percent carbon", "इस्पात में 0.5 प्रतिशत कार्बन है"))
        self.assertFalse(v._protected_tokens_ok("Steel has 0.5 percent carbon", "इस्पात में थोड़ा कार्बन है"))
        self.assertTrue(v._protected_tokens_ok("Water is H2O", "जल H2O है"))
        self.assertFalse(v._protected_tokens_ok("Water is H2O", "जल एच टू ओ है"))

    def test_blocks_with_units_or_operators_stay_english(self):
        self.assertEqual(v.classify_block("Steel: used for railway lines and bridges", []), "translate")
        self.assertNotEqual(v.classify_block("Speed = 5 cm per second", []), "translate")
        self.assertNotEqual(v.classify_block("12345", []), "translate")


class IsolationTests(unittest.TestCase):
    """The on-screen translation must never run in the other launchers."""

    ENV = ("VIDEO_TEXT_TRANSLATION", "VIDEO_TEXT_TRANSLATION_AUTO", "VIDEO_TEXT_OCR_EVERY", "VIDEO_TEXT_ENGINE")

    def cleared(self, **extra):
        from unittest import mock
        import os
        env = {k: v for k, v in os.environ.items() if k not in self.ENV}
        env.update(extra)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_both_shipped_yaml_files_leave_it_off_by_default(self):
        from pathlib import Path
        from app.config import load_config
        root = Path(__file__).resolve().parents[1]
        with self.cleared():
            for name in ("config.realistic.12gb.yaml", "config.hindi.piper.production.yaml"):
                config = load_config(root / name)
                self.assertFalse(v.is_enabled(config), name)
                self.assertFalse(v.is_auto(config), name)
                self.assertEqual(v.engine_name(config), "madlad")
                self.assertEqual(v.ocr_every(config), 3)

    def test_environment_wins_over_yaml_and_yaml_decides_when_unset(self):
        from app.config import AppConfig
        config = AppConfig()
        with self.cleared():
            self.assertFalse(v.is_enabled(config))
            config.production.video_text_translation = True
            self.assertTrue(v.is_enabled(config))                 # YAML true, variable not set
        with self.cleared(VIDEO_TEXT_TRANSLATION="0"):
            self.assertFalse(v.is_enabled(config))                # explicit 0 beats YAML true
        config.production.video_text_translation = False
        with self.cleared(VIDEO_TEXT_TRANSLATION="1"):
            self.assertTrue(v.is_enabled(config))                 # the visual bat turns it on without touching YAML
        with self.cleared(VIDEO_TEXT_TRANSLATION="maybe"):
            self.assertFalse(v.is_enabled(config))                # junk values are ignored

    def test_engine_and_scan_interval_settings(self):
        from app.config import AppConfig
        config = AppConfig()
        with self.cleared(VIDEO_TEXT_ENGINE="glossary", VIDEO_TEXT_OCR_EVERY="5"):
            self.assertEqual(v.engine_name(config), "glossary")
            self.assertEqual(v.ocr_every(config), 5)
        with self.cleared(VIDEO_TEXT_ENGINE="nonsense"):
            self.assertEqual(v.engine_name(config), "madlad")
        with self.cleared():
            config.production.video_text_translation_engine = "qwen"
            self.assertEqual(v.engine_name(config), "qwen")

    def test_glossary_only_engine_never_calls_a_model(self):
        from unittest import mock
        from app.config import AppConfig
        config = AppConfig()
        with self.cleared(VIDEO_TEXT_ENGINE="glossary"),                 mock.patch.object(v, "madlad_translate", side_effect=AssertionError("model used")),                 mock.patch.object(v, "qwen_translate", side_effect=AssertionError("model used")):
            out, engine = v.translate_texts(["Steel", "Something unlisted here"], config)
        self.assertEqual(out[0], "इस्पात")
        self.assertIsNone(out[1])
        self.assertEqual(engine, "glossary_only")


class PlanGuardTests(unittest.TestCase):
    """Defects found in the real Acid-base plan (105 blocks)."""

    def test_back_translation_overlap_flags_a_changed_meaning(self):
        self.assertLess(v.content_overlap("Examples of olfactory indicators", "Example of cell structure"), v.MIN_BACK_OVERLAP)
        self.assertLess(v.content_overlap("Red cabbage solution", "Carrot Solution"), v.MIN_BACK_OVERLAP)
        self.assertGreaterEqual(v.content_overlap("Almost all acids are sour to taste", "Almost all acids taste sour."), v.MIN_BACK_OVERLAP)

    def test_taste_words_are_never_flipped(self):
        self.assertEqual(v._safe_translation("Almost all acids are sour to taste", "लगभग सभी अम्ल स्वाद में मीठे होते हैं"),
                         "लगभग सभी अम्ल स्वाद में खट्टे होते हैं")
        self.assertEqual(v._safe_translation("All of them are sour to taste", "सभी के स्वाद के लिए कड़वा है"),
                         "सभी के स्वाद के लिए खट्टा है")

    def test_ocr_confusions_and_abbreviations_are_repaired_before_translation(self):
        self.assertEqual(v.apply_ocr_fixes("dil.HCI"), "dilute HCl")
        self.assertEqual(v.apply_ocr_fixes("Dilute HCI (acidic)"), "Dilute HCl (acidic)")

    def test_reviewed_glossary_file_is_loaded(self):
        self.assertEqual(v._GLOSSARY["red cabbage"], "लाल पत्तागोभी")
        self.assertEqual(v._GLOSSARY["clove oil"], "लौंग का तेल")
        self.assertEqual(v._GLOSSARY["ph scale"], "pH पैमाना")

    def test_product_lettering_and_symbols_stay_as_they_are(self):
        self.assertEqual(v.classify_block("VANILLA ESSENCE", []), "all_caps_label")
        self.assertEqual(v.classify_block("Summary", []), "translate")             # a caption, not shouting
        self.assertTrue(v._protected_tokens_ok("pH meter", "pH मीटर"))
        self.assertFalse(v._protected_tokens_ok("pH meter", "पीएच मीटर"))           # the symbol must stay
        self.assertFalse(v._protected_tokens_ok("Dilute HCl", "हल्का एचसीएल"))

    def test_ocr_variants_of_one_label_are_merged_into_one(self):
        hits = {"clove oil": {"english": "Clove oil", "times": [1, 2, 3, 4], "reason": "translate"},
                "cloveoi": {"english": "Cloveoi", "times": [5], "reason": "translate"},
                "lemon juice": {"english": "Lemon juice", "times": [1], "reason": "translate"}}
        merged = v._merge_near_duplicates(hits)
        self.assertEqual(sorted(merged), ["clove oil", "lemon juice"])
        self.assertEqual(sorted(merged["clove oil"]["times"]), [1, 2, 3, 4, 5])

    def test_a_block_must_stay_on_screen_long_enough(self):
        self.assertEqual(v.MIN_SAMPLES, 3)        # 3 samples x 0.25 s = 0.75 s: animation fragments are dropped


if __name__ == "__main__":
    unittest.main()
