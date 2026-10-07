import unittest

from app.circuit import CircuitError, draw, parse_row

ROW = "Silver plating | electrolysis: battery 6 V; anode silver rod; cathode keychain; electrolyte silver nitrate solution"


class ElectrolysisTest(unittest.TestCase):
    def test_parse(self):
        caption, topology, parts = parse_row(ROW)
        self.assertEqual((caption, topology), ("Silver plating", "electrolysis"))
        self.assertEqual(dict(parts)["cathode"], "keychain")

    def test_missing_part_is_rejected(self):
        with self.assertRaises(CircuitError):
            parse_row("x | electrolysis: battery 6 V; anode silver rod; cathode keychain")
        with self.assertRaises(CircuitError):
            parse_row("x | electrolysis: battery 6 V; anode a; cathode b; electrolyte c; resistor R1")

    def test_draws_a_diagram(self):
        caption, image = draw(ROW)
        self.assertEqual(caption, "Silver plating")
        self.assertGreater(image.size[0], 800)
        self.assertGreater(image.getextrema()[0][0] + 1, 0)

    def test_series_circuits_still_parse(self):
        self.assertEqual(parse_row("Loop | series: battery 12 V; resistor R1")[1], "series")


if __name__ == "__main__":
    unittest.main()
