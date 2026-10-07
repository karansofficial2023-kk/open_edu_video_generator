import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.config import AppConfig
from app.production_renderer import ShotPlan, ShotRenderer
from app.schema import Segment, Shot


class FormulaScreenTests(unittest.TestCase):
    """The equation screen is a notebook-style card: numbered steps, the current one highlighted, the explanation in a pill."""

    def frame(self, lines, notes, t):
        config = AppConfig()
        config.output_resolution.width, config.output_resolution.height = 1920, 1080
        with tempfile.TemporaryDirectory() as tmp:
            renderer = ShotRenderer(config, Path(tmp))
            seg = Segment(segment_number=1, narration="x", formula_lines=lines, explain_steps=notes,
                          shot=Shot(template="formula", heading="Second Derivative Test"))
            return renderer._formula_frame(ShotPlan(segment=seg, scene_title="s"), t, 12.0)

    def test_screen_has_a_white_card_and_the_equation_is_drawn_on_it(self):
        image = self.frame(["f'(x) = 1 - 1/x^2", "x^2 = 1 => x = 1"], ["Derivative.", "Solve."], 4.0)
        self.assertEqual(image.size, (1920, 1080))
        self.assertEqual(image.getpixel((1700, 290)), (255, 255, 255))                       # inside the card, away from the equations
        card = image.crop((220, 240, 1700, 700)).convert("L")
        self.assertGreater(sum(1 for p in card.getdata() if p < 120), 3000)                  # ink of the equations is on the card

    def test_numbered_steps_only_for_derivations(self):
        many = self.frame(["a = b", "c = d", "e = f"], [], 7.0)
        one = self.frame(["a = b"], [], 1.0)
        # the step badge of line 1 sits at the left of the card in a derivation, not in a single equation
        badge = lambda im: sum(1 for p in im.crop((170, 250, 260, 400)).convert("L").getdata() if p < 235)
        self.assertGreater(badge(many), 300)
        self.assertLess(badge(one), 50)

    def test_the_first_equation_is_visible_straight_away(self):
        early = self.frame(["a = b"], ["Note."], 0.5)
        card = early.crop((220, 240, 1700, 700)).convert("L")
        self.assertGreater(sum(1 for p in card.getdata() if p < 120), 1500)


if __name__ == "__main__":
    unittest.main()
