"""Second teacher review (6 Oct): text-free product pictures, one board per equation, automatic screen checks, no frozen long shots."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from app.config import AppConfig
from app.schema import Scene, Segment, Shot, Storyboard


def _seg(number, narration, template="process", formula=None, labels=None, steps=None, explain=None):
    shot = Shot(template=template, heading="H", steps=steps or (["A b c", "D e f"] if template == "process" else []))
    return Segment(segment_number=number, narration=narration, shot_id=f"1.{number}", formula_lines=formula or [],
                   labels=labels or [], explain_steps=explain or [], shot=shot)


class TextFreePictureTests(unittest.TestCase):
    def test_products_are_asked_for_unprinted_and_documents_are_dropped(self):
        from app.assets import text_free_prompt
        prompt = text_free_prompt("A medical textbook page with question 176. A doctor holds a pill bottle on a desk.")
        self.assertNotIn("textbook", prompt)
        self.assertIn("unbranded", prompt)
        self.assertIn("no label", prompt)
        self.assertEqual(text_free_prompt("A sunflower field at noon."), "A sunflower field at noon.")      # nothing to change
        self.assertNotIn("unbranded", text_free_prompt("A signal transformer on a pole."))                  # 'sign' inside a word

    def test_the_reviewers_text_claim_needs_ocr_support(self):
        from app import assets, ocr
        calls = []

        def fake_stray(path, min_conf=0.85, min_height=0.035):
            calls.append(min_height)
            return []                        # OCR (normal and sensitive pass) sees no words at all

        original_stray, original_qa = ocr.stray_text, assets.semantic_qa
        ocr.stray_text = fake_stray
        assets.semantic_qa = lambda *a, **k: {"matches": True, "score": 1.0, "contains_text": True, "missing": [], "issues": []}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                board = Storyboard(title="T", scenes=[Scene(scene_number=1, title="S", narration="n",
                                                            segments=[_seg(1, "Tablets on a table.", "photo")])])
                producer = assets.StillProducer(board, Path(tmp), AppConfig())
                job = assets.Job("1_1", board.scenes[0].segments[0], board.scenes[0].segments[0], "S")
                image = Path(tmp, "a.png")
                Image.new("RGB", (64, 36), "white").save(image)
                record = {"prompt": "p", "attempts": [{"path": str(image), "technical": {"ok": True}}], "accepted": False}
                producer._save = lambda *a, **k: None
                producer._review(job, record)
        finally:
            ocr.stray_text, assets.semantic_qa = original_stray, original_qa
        self.assertTrue(record["accepted"])
        self.assertTrue(record["attempts"][-1]["semantic"].get("text_overruled_by_ocr"))
        self.assertIn(0.02, calls)                                   # the sensitive pass ran before overruling


class OneBoardPerEquationTests(unittest.TestCase):
    def test_a_repeated_equation_with_cards_between_is_one_board(self):
        from app.storyboard_cleanup import merge_repeated_formulas
        rows = [_seg(1, "The question gives f.", "formula", ["f(x) = x + 1/x"], explain=["Function"]),
                _seg(2, "We need its minimum."),
                _seg(3, "Let the number be x.", "formula", [r"f(x) = x + \frac{1}{x}"], explain=["Variable"]),
                _seg(4, "Its derivative follows.", "formula", ["f'(x) = 1 - 1/x^2"]),
                _seg(5, "A photo of a wire.", "photo"),
                _seg(6, "Again the derivative.", "formula", ["f'(x) = 1 - 1/x^2"])]
        board = Storyboard(title="T", subject="Physics", scenes=[Scene(scene_number=1, title="S", narration="n", segments=rows)])
        self.assertEqual(merge_repeated_formulas(board), 2)
        shots = board.scenes[0].segments
        self.assertEqual([s.shot_id for s in shots], ["1.1", "1.4", "1.5", "1.6"])     # a photograph ends the run
        self.assertEqual(shots[0].narration, "The question gives f. We need its minimum. Let the number be x.")
        self.assertEqual(shots[0].explain_steps, ["Function", "Variable"])

    def test_in_mathematics_a_planned_photo_between_repeats_does_not_break_the_board(self):
        from app.storyboard_cleanup import merge_repeated_formulas
        rows = [_seg(1, "The question gives f.", "formula", ["f(x) = x + 1/x"]),
                _seg(2, "We are asked to find the value of x.", "photo"),
                _seg(3, "Therefore f is x plus one over x.", "formula", [r"f(x) = x + \frac{1}{x}"])]
        board = Storyboard(title="T", subject="Mathematics", scenes=[Scene(scene_number=1, title="S", narration="n", segments=rows)])
        self.assertEqual(merge_repeated_formulas(board), 2)
        self.assertEqual(len(board.scenes[0].segments), 1)


class ScreenCheckTests(unittest.TestCase):
    def test_blank_and_repeated_screens_and_fragment_cards_are_found(self):
        from app.qa import blank_screen, fragment_card, near_identical
        blank = Image.new("RGB", (1920, 1080), (246, 244, 238))
        ImageDraw.Draw(blank).text((96, 48), "Heading only", fill=(20, 20, 20))
        ImageDraw.Draw(blank).rounded_rectangle((110, 214, 1810, 716), radius=34, fill="white", outline=(220, 220, 220))
        full = blank.copy()
        from PIL import ImageFont
        ImageDraw.Draw(full).text((600, 400), "f(x) = x + 1/x", fill=(20, 90, 60), font=ImageFont.load_default(size=72))
        self.assertTrue(blank_screen(np.asarray(blank)))
        self.assertFalse(blank_screen(np.asarray(full)))
        self.assertTrue(near_identical(np.asarray(full), np.asarray(full)))
        self.assertFalse(near_identical(np.asarray(full), np.asarray(blank)))
        self.assertTrue(fragment_card("Weak acids are classified as weak electrolytes due to their low", "Weak acids are classified as weak electrolytes due to their low molar conductivities."))
        self.assertTrue(fragment_card("What", "What draws a pollinator?"))
        self.assertFalse(fragment_card("Pollinator", "What draws a pollinator?"))
        self.assertFalse(fragment_card("We are asked to find the value of x.", "We are asked to find the value of x."))


class NeverFrozenTests(unittest.TestCase):
    def renderer(self, tmp):
        from app.production_renderer import ShotRenderer
        config = AppConfig()
        config.output_resolution.width, config.output_resolution.height = 640, 360
        return ShotRenderer(config, Path(tmp))

    def test_a_long_equation_board_drifts_and_its_notes_advance(self):
        from app.production_renderer import ShotPlan
        with tempfile.TemporaryDirectory() as tmp:
            r = self.renderer(tmp)
            seg = Segment(segment_number=1, narration="x", formula_lines=["a = b"], explain_steps=["one", "two", "three"],
                          shot=Shot(template="formula", heading="H"))
            plan = ShotPlan(segment=seg, scene_title="s", duration=30.0)
            self.assertTrue(r.drifting(plan))
            early, late = r.base_frame(plan, 2.0, 30.0), r.base_frame(plan, 28.0, 30.0)
            self.assertIsNotNone(ImageChops.difference(early.convert("RGB"), late.convert("RGB")).getbbox())
            self.assertEqual([r.note_index(plan, t, 1) for t in (1.0, 12.0, 25.0)], [0, 1, 2])

    def test_short_boards_are_unchanged(self):
        from app.production_renderer import ShotPlan
        with tempfile.TemporaryDirectory() as tmp:
            r = self.renderer(tmp)
            seg = Segment(segment_number=1, narration="x", formula_lines=["a = b"], explain_steps=["one"],
                          shot=Shot(template="formula", heading="H"))
            plan = ShotPlan(segment=seg, scene_title="s", duration=8.0)
            self.assertFalse(r.drifting(plan))
            self.assertIsNone(ImageChops.difference(r.base_frame(plan, 3.0, 8.0).convert("RGB"),
                                                    r.base_frame(plan, 7.0, 8.0).convert("RGB")).getbbox())

    def test_slow_push_starts_exactly_on_the_picture(self):
        from app.production_renderer import slow_push
        image = Image.radial_gradient("L").convert("RGB").resize((320, 180))
        self.assertIs(slow_push(image, 0.0, 1.06), image)
        self.assertIsNotNone(ImageChops.difference(image, slow_push(image, 1.0, 1.06)).getbbox())


if __name__ == "__main__":
    unittest.main()


class TeacherReview3Tests(unittest.TestCase):
    def test_an_equation_that_ends_one_scene_and_opens_the_next_is_written_once(self):
        from app.storyboard_cleanup import merge_repeated_formulas
        first = Scene(scene_number=1, title="A", narration="n", segments=[_seg(1, "So R equals rho L over A.", "formula", ["R = \rho L / A"])])
        second = Scene(scene_number=2, title="B", narration="n", segments=[_seg(1, "Using R equals rho L over A again.", "formula", ["R = \rho L/A"]),
                                                                           _seg(2, "Now the length doubles.", "formula", ["L' = 2L"])])
        board = Storyboard(title="T", subject="Physics", scenes=[first, second])
        self.assertEqual(merge_repeated_formulas(board), 1)
        self.assertEqual(len(board.scenes[0].segments), 1)
        self.assertEqual([s.formula_lines for s in board.scenes[1].segments], [["L' = 2L"]])

    def test_a_picture_of_a_graph_is_never_generated(self):
        from app.assets import text_free_prompt
        self.assertEqual(text_free_prompt("A graph of molar conductivity against concentration with a curve approaching the y-axis."), "")
        self.assertEqual(text_free_prompt("A copper wire stretched between two clamps."), "A copper wire stretched between two clamps.")

    def test_a_chemical_species_inside_a_latex_line_gets_real_subscripts(self):
        from app.formulas import normalize_formula
        self.assertIn(r"\mathrm{CH_{3}COOH}", normalize_formula(r"\Lambda^\circ(CH3COOH) = \Lambda^\circ(H^+)"))
        self.assertEqual(normalize_formula(r"\Lambda_m^\circ(weak)"), r"\Lambda_m^\circ(weak)")       # a word is not a species
