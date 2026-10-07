import unittest

from app.formulas import normalize_formula, to_mathtext, validate


class FormulaNormalizeTests(unittest.TestCase):
    def test_unicode_scripts_become_math_scripts(self):
        self.assertEqual(normalize_formula("Λ₀ = λ⁺ + λ⁻"), "Λ_{0} = λ^{+} + λ^{-}")

    def test_caption_prefix_is_dropped(self):
        self.assertEqual(normalize_formula("Ohm's law: V = I R"), "V = I R")

    def test_ions_and_species_are_upright(self):
        out = to_mathtext("Λ° = Λ°(H+) + Λ°(CH3COO-)")
        self.assertIn(r"\mathrm{H}^{+}", out)
        self.assertIn(r"\mathrm{CH_{3}COO}^{-}", out)

    def test_word_subscripts_are_upright(self):
        self.assertIn(r"_{\mathrm{cation}}", normalize_formula("Λ_cation + Λ_anion"))

    def test_variable_pairs_are_not_mistaken_for_chemistry(self):
        self.assertEqual(normalize_formula("PV = nRT"), "PV = nRT")

    def test_all_shapes_validate(self):
        for line in ["Λ°_electrolyte = Λ°_cation + Λ°_anion", "\\Lambda_m^\\circ (CH_3COOH) = \\Lambda_m^\\circ (NaAc)",
                     "Ag(s) -> Ag^+(aq) + e^-", "x = (-b ± sqrt(b^2 - 4ac))/(2a)"]:
            self.assertIsNone(validate(line), line)


if __name__ == "__main__":
    unittest.main()
