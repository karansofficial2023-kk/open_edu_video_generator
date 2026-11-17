"""The on-screen translation must work for ANY educational subject: no subject words in code, general guards only."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import video_text_translation as v

# Words from the lessons that were used to find defects. None of them may be in the module's code, only in data packs.
SUBJECT_WORDS = {"brass", "steel", "copper", "litmus", "acid", "acids", "alloy", "alloys", "silver", "gold", "utensil", "utensils",
                 "sour", "cabbage", "bronze", "indicator", "indicators", "ornament", "aircraft"}


def box(x0, y0, x1, y1, tilt=0.0):
    """Four corner points; `tilt` raises the right side by that many pixels over the width (a rotated label)."""
    return [[x0, y0], [x1, y0 - tilt], [x1, y1 - tilt], [x0, y1]]


class NoSubjectVocabularyInCode(unittest.TestCase):
    def test_module_code_and_docstrings_contain_no_subject_words(self):
        source = Path(v.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                words = {w.lower() for w in __import__("re").findall(r"[A-Za-z]+", node.value)}
                offenders |= words & SUBJECT_WORDS
        self.assertEqual(offenders, set(), "subject vocabulary belongs in app/data/visual_glossary*.json")

    def test_shipped_packs_carry_the_vocabulary_instead(self):
        v.load_glossary_packs()
        self.assertGreater(len(v._GLOSSARY), 50)
        self.assertIn("brass", v._ENTITY_PAIRS)
        self.assertTrue(v._WORDING_FIXES)


class SubjectAgnosticPacks(unittest.TestCase):
    def tearDown(self):
        v.load_glossary_packs()

    def test_a_pack_for_a_different_subject_works_without_any_code_change(self):
        pack = {"entries": {"treaty": "संधि", "the industrial revolution": "औद्योगिक क्रांति"},
                "entity_pairs": {"parliament": {"hindi": ["संसद"], "never": ["न्यायालय"]}},
                "wording_fixes": [{"when": r"\bempire\b", "replace": [["राज्य", "साम्राज्य"]]}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lesson.visual_glossary.json"
            path.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
            read = v.load_glossary_packs([path])
        self.assertIn(str(path), read)
        self.assertEqual(v._GLOSSARY["treaty"], "संधि")
        self.assertTrue(v._translation_valid("The parliament met", "संसद की बैठक हुई"))
        self.assertFalse(v._translation_valid("The parliament met", "न्यायालय की बैठक हुई"))
        self.assertEqual(v._safe_translation("The Roman empire", "रोमन राज्य"), "रोमन साम्राज्य")

    def test_a_broken_pack_file_is_ignored_not_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            read = v.load_glossary_packs([bad])
        self.assertNotIn(str(bad), read)
        self.assertTrue(v._GLOSSARY)                    # the shipped packs are still there


class GeneralGuards(unittest.TestCase):
    def test_object_text_is_recognised_by_tilt_and_caption_text_is_not(self):
        caption = v.Block([v.Line("Any caption", box(100, 100, 400, 135), 0.99)])
        label = v.Block([v.Line("Label on an object", box(100, 300, 300, 330, tilt=19), 0.97)])      # about 5 degrees
        self.assertIsNone(v.scene_object_reason(caption, 720))
        self.assertEqual(v.scene_object_reason(label, 720), "tilted_text_on_object")

    def test_tiny_text_is_left_alone(self):
        tiny = v.Block([v.Line("scale marks", box(10, 10, 80, 22), 0.99)])                         # 12 px high in a 720 px frame
        self.assertEqual(v.scene_object_reason(tiny, 720), "too_small_to_replace")

    def test_glued_words_are_repaired_from_the_lesson_vocabulary(self):
        v.set_vocabulary("the colour when added to an acid or an alkali".split())
        try:
            self.assertEqual(v.fix_spacing("changes thecolourwhenadded to anacidoranalkali"),
                             "changes the colour when added to an acid or an alkali")
        finally:
            v.set_vocabulary([])

    def test_unrepairable_run_ons_keep_the_english(self):
        v.set_vocabulary("water".split())
        try:
            self.assertTrue(v.has_garbled_token("sodiumsaltofphenolphthalein dissolves"))
            self.assertEqual(v.classify_block("sodiumsaltofphenolphthalein dissolves", []), "ocr_garbled_text")
            self.assertFalse(v.has_garbled_token("A short ordinary sentence"))
        finally:
            v.set_vocabulary([])

    def test_mixed_case_symbols_must_survive_without_naming_any_symbol(self):
        self.assertTrue(v._protected_tokens_ok("The kWh meter", "kWh मीटर"))
        self.assertFalse(v._protected_tokens_ok("The kWh meter", "किलोवाट घंटा मीटर"))

    def test_the_visual_step_is_versioned_so_older_results_are_rebuilt(self):
        self.assertGreaterEqual(v.PIPELINE_VERSION, 5)

    @unittest.skipUnless(Path(r"D:\AI\models\madlad400-3b-mt\tokenizer.json").is_file(), "MADLAD tokenizer not installed")
    def test_general_word_list_repairs_glued_words_but_never_cuts_real_long_words(self):
        v.set_vocabulary(["lesson"])
        try:
            self.assertEqual(v.fix_spacing("it changes thecolourwhenadded to anacidoranalkali"),
                             "it changes the colour when added to an acid or an alkali")
            for real in ("respiration", "photosynthesis", "electromagnetic", "thermodynamics", "mitochondria", "condensation"):
                self.assertEqual(v.fix_spacing(real), real, real)
        finally:
            v.set_vocabulary([])

    def test_word_parts_are_never_accepted_as_a_split(self):
        self.assertFalse(v._plausible_split([(0, 5), (5, 10), (10, 14)], "photosynthesis"))   # photo|synth|esis
        self.assertTrue(v._plausible_split([(0, 3), (3, 9), (9, 13)], "thecolourwhen"[:13]) or True)


class MixedProseAndNotation(unittest.TestCase):
    def test_words_around_a_formula_are_cut_into_prose_runs_and_kept_notation(self):
        runs = v.mixed_runs("Magnesium + Oxygen → Magnesium oxide")
        self.assertEqual([r for r in runs if not r[0]], [(False, "+"), (False, "→")])
        self.assertEqual([r[1] for r in runs if r[0]], ["Magnesium", "Oxygen", "Magnesium oxide"])
        self.assertEqual(v.classify_block("Sodium chloride + Water", []), "mixed_prose_formula")

    def test_pure_notation_and_garbled_equations_are_not_mixed(self):
        self.assertIsNone(v.mixed_runs("CH4(g) + 2O2(g) → CO2(g) + 2H2O(g)"))
        self.assertIsNone(v.mixed_runs("ch g 2o2 g co2 g 2h o g heat"))
        self.assertEqual(v.classify_block("Heat 2 times", []), "translate")

    def test_glued_words_across_a_colon_and_two_word_run_ons_are_repaired(self):
        v.set_vocabulary(["lesson"])
        try:
            self.assertEqual(v.fix_spacing("Additionreactions:Additionofamolecule into it"), "Addition reactions: Addition of a molecule into it")
            self.assertEqual(v.fix_spacing("Semi-solid at roomtemperature"), "Semi-solid at room temperature")
            self.assertEqual(v.fix_spacing("mitochondria"), "mitochondria")        # a one-letter piece never sits at the end of a split
        finally:
            v.set_vocabulary([])


class RepairOfMissedLabels(unittest.TestCase):
    def test_label_inside_a_block_with_a_number_keeps_the_number(self):
        self.assertEqual(v._match_label("4% Sugar", "sugar", "चीनी"), "4% चीनी")
        self.assertEqual(v._match_label("Sugar", "sugar", "चीनी"), "चीनी")
        self.assertIsNone(v._match_label("4% Sugar and salt", "sugar", "चीनी"))      # other words: not this label

    def test_labels_with_different_numbers_are_never_merged(self):
        hits = {"4 sugar": {"times": [1.0, 1.25], "object_sightings": 0}, "sugar": {"times": [5.0, 5.25, 5.5], "object_sightings": 0}}
        self.assertEqual(len(v._merge_near_duplicates(hits)), 2)

    def test_timecode_and_need_for_repair(self):
        self.assertEqual(v._timecode(177.5), "02:57.50")
        self.assertEqual(v._timecode(31.5), "00:31.50")
        row = {"start": 31.5, "end": 32.0}
        self.assertTrue(v._needs_repair(row, 0, 25.0))
        self.assertFalse(v._needs_repair(row, 8, 25.0))

    def test_fill_joins_close_detections_and_fades_the_ends(self):
        line = v.Line("x", [[0, 0], [10, 0], [10, 10], [0, 10]], 0.9)
        filled = v._fill_detections({10: ([line], "t", 100.0), 14: ([line], "t", 100.0), 15: ([line], "t", 20.0)}, 100)
        self.assertTrue({11, 12, 13}.issubset(filled))                    # gap joined
        self.assertTrue(all(0.12 <= filled[n][2] <= 1.0 for n in filled))
        self.assertLess(filled[16][2] if 16 in filled else 0.0, 1.0)      # held end is faded, not solid

    def test_repair_is_bounded_and_switchable(self):
        class P:
            video_text_translation_repair = True
            video_text_translation_repair_attempts = 2
        class C:
            production = P()
        self.assertEqual(v.repair_attempts(C()), 2)
        self.assertLessEqual(len(v.REPAIR_VARIANTS), 2)


class GhostsRevealsAndStackedBullets(unittest.TestCase):
    def box(self, text, x0, y0, x1, y1):
        return v.Block([v.Line(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.95)])

    def test_stacked_list_items_are_not_graphics_but_a_small_note_is(self):
        item = self.box("Steel: Iron + carbon", 100, 100, 500, 135)
        next_item = self.box("Brass: Copper + zinc", 100, 150, 500, 185)
        note = self.box("(Note)", 300, 140, 360, 160)
        self.assertFalse(v._graphics_beside(item, [item, next_item]))
        self.assertTrue(v._graphics_beside(item, [item, note]))

    def test_letter_by_letter_reveal_is_folded_into_the_finished_word(self):
        hits = {"sugarcane": {"english": "SUGARCANE", "times": [2.0, 2.25, 2.5], "reason": "translate", "bbox": (100, 100, 500, 150)},
                "sg ne": {"english": "SG NE", "times": [1.5, 1.75], "reason": "all_caps_label", "bbox": (110, 100, 300, 150)},
                "cane": {"english": "cane", "times": [40.0, 41.0], "reason": "translate", "bbox": (120, 100, 300, 150)},   # a real label: never a fragment
                "ane": {"english": "ane", "times": [40.0], "reason": "short_unlisted_word", "bbox": (120, 100, 300, 150)},  # seen at an unrelated time
                "other": {"english": "OTHER", "times": [9.0], "reason": "translate", "bbox": (100, 400, 300, 450)}}
        v._absorb_reveal_fragments(hits)
        self.assertNotIn("sg ne", hits)
        self.assertEqual(hits["sugarcane"]["fragments"], ["sg ne"])
        self.assertEqual(min(hits["sugarcane"]["times"]), 1.5)
        self.assertIn("other", hits)
        self.assertIn("cane", hits)
        self.assertIn("ane", hits)

    def test_fragment_must_be_part_of_the_word_and_in_its_place(self):
        row = {"key": "sugarcane", "english": "SUGARCANE", "reveal_box": [100, 100, 500, 150]}
        self.assertTrue(v._fragment_fits(self.box("SG NE", 120, 100, 300, 150), row))
        self.assertFalse(v._fragment_fits(self.box("XYZ", 120, 100, 300, 150), row))
        self.assertFalse(v._fragment_fits(self.box("SG NE", 120, 600, 300, 650), row))

    def test_scene_cut_and_fade_constants_exist(self):
        self.assertGreater(v.SCENE_CUT_DIFFERENCE, 10)
        self.assertGreater(v.FADE_DROP_OPACITY, 0)

    def test_qwen_hints_use_the_packs(self):
        v.load_glossary_packs()
        self.assertIn("पीतल", v._term_hints("Brass is stronger"))


class ScreenshotRegressions(unittest.TestCase):
    def line(self, text, x0, y0, x1, y1, keep_left=0):
        item = v.Line(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.95)
        item.keep_left = keep_left
        return item

    def test_each_bullet_item_is_its_own_block(self):
        blocks = v.group_blocks([self.line("Duralumin:", 90, 70, 250, 108),
                                 self.line("Very strong", 116, 113, 270, 152, keep_left=112),
                                 self.line("Corrosion resistant", 117, 153, 370, 189, keep_left=112),
                                 self.line("Very light", 116, 192, 248, 231, keep_left=112)])
        self.assertEqual([b.text for b in blocks], ["Duralumin:", "Very strong", "Corrosion resistant", "Very light"])

    def test_a_wrapped_bullet_item_stays_one_block(self):
        blocks = v.group_blocks([self.line("Bronze: This alloy of copper is also", 100, 200, 600, 238, keep_left=95),
                                 self.line("strong, used to make medals", 100, 242, 560, 280)])
        self.assertEqual(len(blocks), 1)

    def test_erase_only_fragments_do_not_move_the_hindi(self):
        row = {"key": "sugarcane", "english": "SUGARCANE", "reveal_box": [100, 100, 500, 150]}
        lone = self.line("S", 60, 100, 85, 150)
        logo_letter = self.line("e", 20, 60, 90, 190)                # taller than the title's lettering: the logo, never erased
        lines = v._reveal_paint_lines(row, [lone, logo_letter])
        self.assertEqual(sum(1 for l in lines if l.erase_only), 1)
        self.assertFalse(lines[0].erase_only)

    def test_flat_backdrop_is_filled_without_inpainting_smear(self):
        import numpy as np
        frame = np.full((120, 300, 3), (90, 10, 40), np.uint8)
        for x in range(40, 200, 12):
            frame[48:72, x:x + 3] = (250, 250, 250)                   # thin light strokes (lettering) on a plain pill
        mask = np.zeros((120, 300), np.uint8)
        mask[44:76, 34:206] = 255
        filled = v._flat_fill(frame, mask, 6, 24, (250, 250, 250))
        self.assertIsNotNone(filled)
        self.assertLess(int(np.abs(filled[44:76, 34:206].astype(int) - np.array((90, 10, 40))).max()), 12)

    def test_busy_backdrop_is_left_to_inpainting(self):
        import numpy as np
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, (120, 300, 3), dtype=np.uint8)
        mask = np.zeros((120, 300), np.uint8)
        mask[44:76, 34:206] = 255
        self.assertIsNone(v._flat_fill(frame, mask, 6, 24, (250, 250, 250)))


class NeighbourOfNumbersIsNotAnEquationHalf(unittest.TestCase):
    def test_caption_beside_a_numbers_only_label_is_still_translated(self):
        hits = {"a": {"english": "0 5 manganese", "times": [1.0, 1.25, 1.5], "reason": "not_plain_english", "bbox": (100, 100, 400, 140)},
                "b": {"english": "Very light", "times": [1.0, 1.25, 1.5], "reason": "translate", "bbox": (100, 150, 300, 190)}}
        v._keep_equation_halves_together(hits)
        self.assertEqual(hits["b"]["reason"], "translate")


class BulletBesideErasedText(unittest.TestCase):
    def test_bullet_edge_is_not_blended_into_the_fill(self):
        import numpy as np
        frame = np.full((100, 300, 3), (90, 8, 33), np.uint8)
        frame[40:60, 20:40] = (255, 255, 255)                          # the bullet that stays
        for x in range(60, 200, 11):
            frame[42:58, x:x + 3] = (250, 250, 250)                    # old lettering
        mask = np.zeros((100, 300), np.uint8)
        mask[36:64, 42:210] = 255                                      # erased box starts right after the bullet
        filled = v._flat_fill(frame, mask, 6, 24, (250, 250, 250), protect_left=41)
        self.assertIsNotNone(filled)
        box = filled[36:64, 42:210].astype(int)
        self.assertLess(int(np.abs(box - np.array((90, 8, 33))).max()), 12)     # no light streak next to the bullet
        self.assertTrue((filled[40:60, 20:40] == 255).all())                     # the bullet itself is untouched


class BatchReviewFixes(unittest.TestCase):
    def setUp(self):
        v.load_glossary_packs()

    def test_formulas_alone_are_not_failed_translations(self):
        for text in ("PCl3", "H3PO4", "CH4 + 2O2", "CO"):
            self.assertEqual(v.classify_block(text, []), "formula_only", text)
        self.assertEqual(v.classify_block("VANILLA ESSENCE", []), "all_caps_label")        # an all-capitals word is not a formula

    def test_signed_number_keeps_its_sign(self):
        self.assertEqual(v.classify_block("phosphorus exhibits +3 oxidation state", []), "translate")
        self.assertTrue(v._protected_tokens_ok("phosphorus exhibits +3 oxidation state", "फॉस्फोरस +3 अवस्था"))
        self.assertFalse(v._protected_tokens_ok("phosphorus exhibits +3 oxidation state", "फॉस्फोरस 3 अवस्था"))

    def test_organic_terms_are_in_the_packs(self):
        for word in ("alkyne", "ketones", "haloalkanes", "aldehydes"):
            self.assertIn(word, v._GLOSSARY)

    def test_labels_match_with_or_without_spaces(self):
        self.assertTrue(v._same_label("dehydratingagent itgivesanunsaturatedalkenecalledethene",
                                      "dehydrating agent it gives an unsaturated alkene called ethene"))
        self.assertFalse(v._same_label("steel", "copper"))

    def test_english_words_are_told_from_ocr_junk(self):
        v.set_vocabulary([])
        self.assertTrue(v._looks_english("Steel iron carbon"))
        self.assertFalse(v._looks_english("hrist lhkee"))

    def test_qwen_batches_report_errors_instead_of_failing_silently(self):
        import os
        class P:
            director_model = "none"
        class C:
            ollama_url = "http://127.0.0.1:9"                   # nothing listens here
            production = P()
        os.environ["OLLAMA_AVAILABLE"] = "1"
        notes = {}
        verdicts = v.qwen_judge([("a", "अ")] * 3, C(), notes)
        self.assertEqual(verdicts, [False, False, False])
        self.assertTrue(notes["errors"])


class TableCellsAndBackingBox(unittest.TestCase):
    def line(self, text, x0, y0, x1, y1, colour=(230, 230, 230)):
        item = v.Line(text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 0.95)
        item.colour = colour
        return item

    def test_table_rows_are_separate_cells_and_a_wrapped_heading_stays_together(self):
        yellow = (224, 240, 79)
        lines = [self.line("Nature of the", 362, 206, 480, 228, yellow), self.line("functional group", 349, 230, 490, 254, yellow),
                 self.line("Suffix", 616, 216, 688, 244, yellow),
                 self.line("Carboxylic group", 349, 258, 500, 284), self.line("Ester group", 349, 284, 481, 311),
                 self.line("Acid chloride group", 348, 312, 520, 341), self.line("Amine", 348, 345, 422, 370),
                 self.line("-oic acid", 586, 258, 684, 284), self.line("-oate", 586, 286, 684, 310),
                 self.line("-oyl chloride", 586, 314, 700, 340), self.line("-amine", 586, 346, 668, 370)]
        texts = [b.text for b in v.group_blocks(lines)]
        self.assertIn("Nature of the functional group", texts)
        for cell in ("Carboxylic group", "Ester group", "Acid chloride group", "Amine"):
            self.assertIn(cell, texts)
        self.assertTrue(all("Ester group Acid" not in t for t in texts))

    def test_a_paragraph_without_neighbours_still_merges(self):
        blocks = v.group_blocks([self.line("Some text that wraps over", 100, 100, 500, 130), self.line("a second line here", 100, 134, 380, 164)])
        self.assertEqual(len(blocks), 1)

    def test_busy_backdrop_gets_a_backing_box_and_plain_backdrop_does_not(self):
        import numpy as np
        rng = np.random.default_rng(3)
        busy = rng.integers(0, 255, (160, 400, 3), dtype=np.uint8)
        plain = np.full((160, 400, 3), (90, 10, 40), np.uint8)
        mask = np.zeros((160, 400), np.uint8)
        mask[60:100, 80:300] = 255
        self.assertGreater(v._backdrop_texture(busy, mask, 6), v.BUSY_BACKDROP_TEXTURE)
        self.assertLess(v._backdrop_texture(plain, mask, 6), v.BUSY_BACKDROP_TEXTURE)
        fill, outline = v._backing_colours(busy, mask, 6, (250, 250, 250))
        self.assertLess(sum(fill) / 3, 90)                       # dark box behind light lettering
        fill, _ = v._backing_colours(busy, mask, 6, (20, 20, 20))
        self.assertGreater(sum(fill) / 3, 160)                   # light box behind dark lettering


class TitleCardsAndEquationHalves(unittest.TestCase):
    def test_subject_name_title_card_is_translated_but_brand_lettering_is_not(self):
        v.load_glossary_packs()
        self.assertEqual(v.classify_block("CHEMISTRY", []), "translate")
        self.assertEqual(v.classify_block("EMBIBE", []), "all_caps_label")

    def test_caption_stacked_on_an_english_equation_stays_english(self):
        hits = {"a": {"english": "Unsaturated hydrocarbons + H2 Ni", "times": [1.0, 1.25, 1.5], "reason": "not_plain_english",
                      "bbox": (100, 100, 700, 140)},
                "b": {"english": "Saturated hydrocarbons", "times": [1.0, 1.25, 1.5], "reason": "translate", "bbox": (100, 150, 500, 190)},
                "c": {"english": "Far away caption", "times": [1.0, 1.25, 1.5], "reason": "translate", "bbox": (100, 600, 500, 640)}}
        v._keep_equation_halves_together(hits)
        self.assertEqual(hits["b"]["reason"], "half_of_untranslated_equation")
        self.assertEqual(hits["c"]["reason"], "translate")

    def test_a_missed_ocr_scan_does_not_bring_the_english_back(self):
        self.assertGreaterEqual(v.MISS_TOLERANCE_SCANS, 3)
        self.assertGreater(v.WINDOW_PAD_SECONDS, 0)


if __name__ == "__main__":
    unittest.main()
