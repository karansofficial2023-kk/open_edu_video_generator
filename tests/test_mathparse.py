import unittest

from app.formulas import looks_chemical, render_line, validate
from app.mathparse import MathParseError, math_to_latex


class MathParseTests(unittest.TestCase):
    def test_plain_math_becomes_typeset_latex(self):
        cases = {
            "I = V / R": r"I = \frac{V}{R}",
            "x = (-b ± sqrt(b^2 - 4ac))/(2a)": r"x = \frac{-b \pm \sqrt{b^{2} - 4 a c}}{2 a}",
            "1/R = 1/R1 + 1/R2": r"\frac{1}{R} = \frac{1}{R_{1}} + \frac{1}{R_{2}}",
            "E = m c^2": r"E = m c^{2}",
            "(x + b/(2a))^2 = (b^2 - 4ac)/(4a^2)": r"\left(x + \frac{b}{2 a}\right)^{2} = \frac{b^{2} - 4 a c}{4 a^{2}}",
        }
        for plain, latex in cases.items():
            self.assertEqual(math_to_latex(plain), latex, plain)

    def test_chemistry_and_equations_take_different_paths(self):
        for chem in ["CuSO4 + Fe -> FeSO4 + Cu", "Cu^{2+}(aq) + 2e^- -> Cu(s)", "H2O"]:
            self.assertTrue(looks_chemical(chem), chem)
        for equation in ["PV = nRT", "V = I R", "x^2 - 5x + 6 = 0"]:
            self.assertFalse(looks_chemical(equation), equation)
            self.assertIsNone(validate(equation))

    def test_unparseable_math_is_reported_not_guessed(self):
        with self.assertRaises(MathParseError):
            math_to_latex("a + ")

    def test_fraction_formula_renders_with_transparent_background(self):
        image = render_line("x = (-b ± sqrt(b^2 - 4ac))/(2a)", "#111111", 80)
        low, high = image.getchannel("A").getextrema()
        self.assertLess(low, 10)
        self.assertGreater(high, 200)
        self.assertGreater(image.height, 80)      # stacked fraction is taller than a single line


if __name__ == "__main__":
    unittest.main()
