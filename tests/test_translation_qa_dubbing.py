"""Regression checks for source-aligned educational dubbing QA."""

from app.translation_qa import check_translation_segment


def test_hindi_indefinite_article_does_not_invent_extra_numbers():
    source = "One ethanol molecule gives a hydrogen ion and an ethoxide ion."
    hindi = "एक इथेनॉल अणु एक हाइड्रोजन आयन और एक एथॉक्साइड आयन देता है।"
    issues = check_translation_segment("28.1", source, hindi, "hi-IN")
    assert not any(item["kind"] == "number_mismatch" for item in issues)


def test_different_explicit_number_still_needs_review():
    issues = check_translation_segment("1.1", "Two atoms combine.", "एक परमाणु जुड़ता है।", "hi-IN")
    assert any(item["kind"] == "number_mismatch" for item in issues)


def test_chemical_formula_change_still_needs_review():
    issues = check_translation_segment("1.1", "NaOH is added.", "NaCl मिलाया जाता है।", "hi-IN")
    assert any(item["kind"] == "formula_mismatch" for item in issues)
