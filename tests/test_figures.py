import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / ".runtime-deps")]

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

    def test_sci_figures_and_occupancy(self):
        from app import figures_sci, labels, tts_indic
        # Chemistry figures choose from narration
        self.assertEqual(figures.concept_for("During the electrolysis of copper sulphate, ions migrate to the cathode.", "", "chemistry"), "electrolysis_cell")
        self.assertEqual(figures.concept_for("In this titration, the indicator reaches its equivalence point with a sharp colour change.", "", "chemistry"), "titration")
        # Occupancy grid
        grid = labels.OccupancyGrid((1920, 1080))
        cells = grid.register("test", (100, 100, 300, 300))
        self.assertIn("top_left", cells)
        # Pronunciations
        spoken = tts_indic.apply_pronunciations("Reaction of CO2 and H2O", {"CO2": "carbon dioxide", "H2O": "water"})
        self.assertEqual(spoken, "Reaction of carbon dioxide and water")

    def test_electroplating_scientific_figures(self):
        # 1. Wiring & Electrodes (Points 1, 2, 3)
        self.assertEqual(
            figures.concept_for("The red wire is connected to the positive terminal of the battery, while the blue wire is linked to the negative terminal through a switch.", "", "chemistry"),
            "electrolysis_cell"
        )
        self.assertEqual(
            figures.concept_for("The terminal connected to the battery's positive side is called the anode, while the one connected to the negative side is known as the cathode.", "", "chemistry"),
            "electrolysis_cell"
        )

        # 2. Outermost shell with empty electrons (Point 7)
        self.assertEqual(
            figures.concept_for("Notice that the silver ion has an empty spot in its outermost shell. It attracts an electron into this position.", "", "chemistry"),
            "silver_ion_shell"
        )

        # 3. Dual redox reaction balance (Point 6)
        self.assertEqual(
            figures.concept_for("Since reduction occurs at the cathode and oxidation takes place at the anode, the net reaction is a redox reaction.", "", "chemistry"),
            "redox_reaction_board"
        )

        # 4. Verify rendering on dark teacher board
        cell_img = figures.frame((1920, 1080), CONFIG, "electrolysis_cell", "Electroplating Setup", 2.0, "The red wire is connected to positive terminal, blue wire to negative terminal through switch.", False)
        self.assertEqual(cell_img.size, (1920, 1080))

        shell_img = figures.frame((1920, 1080), CONFIG, "silver_ion_shell", "Silver Ion Shell", 2.0, "empty spot in outermost shell", False)
        self.assertEqual(shell_img.size, (1920, 1080))

        redox_img = figures.frame((1920, 1080), CONFIG, "redox_reaction_board", "Redox Reactions", 2.0, "net reaction is a redox reaction", False)
        self.assertEqual(redox_img.size, (1920, 1080))

    def test_combustion_and_equivalence_ratio_figures(self):
        # 1. Equivalence ratio spectrum analyzer matching
        self.assertEqual(
            figures.concept_for("The equivalence ratio phi compares the actual fuel-to-air ratio with the stoichiometric ratio.", "", "engineering"),
            "equivalence_ratio_dial"
        )
        self.assertEqual(
            figures.concept_for("In a lean mixture, excess air leads to lower peak temperatures and high fuel economy.", "", "thermodynamics"),
            "equivalence_ratio_dial"
        )
        self.assertEqual(
            figures.concept_for("A rich mixture contains excess fuel that cannot completely burn, producing soot.", "", "general"),
            "equivalence_ratio_dial"
        )

        # 2. Combustion chamber cylinder dynamics matching
        self.assertEqual(
            figures.concept_for("Inside the internal combustion engine cylinder, the spark plug ignites the compressed mixture.", "", "engineering"),
            "combustion_chamber"
        )
        self.assertEqual(
            figures.concept_for("A high-pressure flame front propagates rapidly across the combustion chamber, forcing the piston down.", "", "thermodynamics"),
            "combustion_chamber"
        )

        # 3. Verify high quality rendering at 1080p
        dial_img = figures.frame((1920, 1080), CONFIG, "equivalence_ratio_dial", "Equivalence Ratio", 2.5, "lean mixture excess air", False)
        self.assertEqual(dial_img.size, (1920, 1080))

        chamber_img = figures.frame((1920, 1080), CONFIG, "combustion_chamber", "Combustion Dynamics", 2.5, "flame front power stroke piston", False)
        self.assertEqual(chamber_img.size, (1920, 1080))




    def test_four_stroke_cycle_and_air_fuel_ratio_gauge(self):
        # 1. 4-Stroke Engine Cycle matching
        self.assertEqual(
            figures.concept_for("In a four-stroke internal combustion engine, the intake stroke pulls the air-fuel mixture into the cylinder.", "", "engineering"),
            "four_stroke_cycle"
        )
        self.assertEqual(
            figures.concept_for("During the compression stroke, both valves are closed as the piston moves upward.", "", "thermodynamics"),
            "four_stroke_cycle"
        )
        self.assertEqual(
            figures.concept_for("The engine cycle consists of intake, compression, power stroke, and exhaust stroke.", "", "physics"),
            "four_stroke_cycle"
        )

        # 2. Air-Fuel Ratio (AFR) & Spark Plug Diagnostics matching
        self.assertEqual(
            figures.concept_for("The air-fuel ratio of stoichiometric 14.7:1 is ideal for gasoline combustion.", "", "engineering"),
            "air_fuel_ratio_gauge"
        )
        self.assertEqual(
            figures.concept_for("A lean mixture causes the spark plug condition to show blistered white overheated deposits.", "", "thermodynamics"),
            "air_fuel_ratio_gauge"
        )
        self.assertEqual(
            figures.concept_for("A rich mixture with excess fuel causes the spark plug tip to become carbon fouled black.", "", "general"),
            "air_fuel_ratio_gauge"
        )

        # 3. Verify high quality rendering at 1080p
        stroke_img = figures.frame((1920, 1080), CONFIG, "four_stroke_cycle", "Four-Stroke Cycle", 2.2, "spark plug fires in power stroke", False)
        self.assertEqual(stroke_img.size, (1920, 1080))

        afr_img = figures.frame((1920, 1080), CONFIG, "air_fuel_ratio_gauge", "Air-Fuel Ratio", 4.0, "stoichiometric 14.7:1 ideal ratio", False)
        self.assertEqual(afr_img.size, (1920, 1080))


if __name__ == "__main__":
    unittest.main()
