import unittest
from pathlib import Path

from app import figures
from app.config import load_config

CONFIG = load_config(Path(__file__).resolve().parents[1] / "config.yaml")


class FigureTests(unittest.TestCase):
    def test_concept_chosen_from_narration(self):
        self.assertEqual(figures.concept_for("The tangent to a circle is perpendicular to the radius at the point of contact.", "", "maths"), "tangent_circle")
        self.assertEqual(figures.concept_for("A convex lens forms a real, inverted image when the object is beyond the focus.", "", "physics"), "lens_rays")
        self.assertIsNone(figures.concept_for("Today we will talk about the history of the school.", "", "maths"))

    def test_frame_is_full_size_and_builds_up(self):
        text = "The tangent to a circle is perpendicular to the radius."
        early = figures.frame((1920, 1080), CONFIG, "tangent_circle", "Tangents", 0.1, text, True)
        late = figures.frame((1920, 1080), CONFIG, "tangent_circle", "Tangents", 6.0, text, True)
        self.assertEqual(early.size, (1920, 1080))
        self.assertEqual(late.size, (1920, 1080))
        self.assertNotEqual(early.tobytes(), late.tobytes())

    def test_dark_board_frame(self):
        img = figures.frame((1920, 1080), CONFIG, "ohm", "Ohm's law", 6.0, "Current is proportional to the potential difference.", False)
        self.assertEqual(img.size, (1920, 1080))


if __name__ == "__main__":
    unittest.main()
