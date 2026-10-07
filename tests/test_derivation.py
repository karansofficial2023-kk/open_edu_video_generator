import unittest

from app.derivation import check_chain, equivalent, to_equation

GOOD = ["a x^2 + b x + c = 0", "x^2 + (b/a)x + c/a = 0", "x^2 + (b/a)x = -c/a", r"(x + \frac{b}{2a})^2 = \frac{b^2 - 4ac}{4a^2}"]


class DerivationTests(unittest.TestCase):
    def test_correct_completing_the_square_chain_passes(self):
        self.assertEqual(check_chain([("1", GOOD)]), [])

    def test_wrong_step_is_reported_once(self):
        chain = [("1", GOOD[:3]), ("2", [r"ax^2 + bx + c + \frac{b^2}{4a^2} = \frac{b^2}{4a^2} - c"]), ("3", [GOOD[3]])]
        problems = check_chain(chain)
        self.assertEqual([shot for shot, _ in problems], ["2"])

    def test_worked_numeric_example_is_not_compared_with_the_general_form(self):
        self.assertEqual(check_chain([("1", GOOD), ("2", ["x^2 - 5x + 6 = 0"])]), [])

    def test_superscript_two_is_a_square(self):
        self.assertTrue(equivalent(to_equation("x² + (b/a)x + c/a = 0"), to_equation("a x^2 + b x + c = 0")))

    def test_inequalities_and_reactions_are_skipped(self):
        self.assertIsNone(to_equation("b^2 - 4ac > 0"))
        self.assertIsNone(to_equation("Ag(s) -> Ag^+(aq) + e^-"))
        self.assertIsNone(to_equation("x = ± sqrt(y)"))


if __name__ == "__main__":
    unittest.main()
