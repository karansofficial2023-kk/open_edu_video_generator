import unittest

from app.charts import _to_python, parse_function, parse_rows


class ChartRowTest(unittest.TestCase):
    def test_scientific_and_indic_units_are_kept(self):
        rows = parse_rows(["Resistance: 10 Ω", "Acceleration: 9.8 m/s²", "Temperature: 37 °C", "உயரம்: 5 மீ", "Share: 40%"])
        self.assertEqual(rows, [("Resistance", 10.0, "Ω"), ("Acceleration", 9.8, "m/s²"), ("Temperature", 37.0, "°C"),
                                ("உயரம்", 5.0, "மீ"), ("Share", 40.0, "%")])

    def test_plain_rows_still_parse(self):
        self.assertEqual(parse_rows(["A: 10", "B = 2,500"]), [("A", 10.0, ""), ("B", 2500.0, "")])


class FunctionGuardTest(unittest.TestCase):
    def test_ordinary_curves_pass(self):
        for rhs in ("x^2 - 5x + 6", "2*x + 1", "sin(x)", "x^3 - x"):
            self.assertTrue(_to_python(rhs))

    def test_runaway_powers_are_rejected(self):
        for rhs in ("9^9^9^9", "x^99", "x^2^3", "10000000*x", "x" * 200):
            with self.assertRaises(ValueError, msg=rhs):
                _to_python(rhs)

    def test_a_bad_plot_is_refused_quickly(self):
        self.assertIsNone(parse_function(["y = 3"]))
        with self.assertRaises(ValueError):
            parse_function(["y = 9**9**9**9 + x"])


if __name__ == "__main__":
    unittest.main()
