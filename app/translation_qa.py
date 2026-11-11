"""Critical source-to-dub checks for formula and numeric preservation."""
from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from .schema import Storyboard

_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_FORMULA = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][a-z]?\d*|\([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\)\d*)+"
    r"(?:[·.]\d*(?:[A-Z][a-z]?\d*)+)*(?:[⁺⁻+\-]\d*)?(?![A-Za-z0-9])"
)
_ELEMENT = re.compile(r"[A-Z][a-z]?")
# A bare comma is treated as a list separator (e.g. “2,4” => 2 and 4), not a decimal point.
# Thousands separators remain supported; decimal values should use the unambiguous dot form.
_NUMBER = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z0-9])")
_NUMBER_WORD = re.compile(r"(?<![\w\u0900-\u097f])(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                          r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
                          r"fifty|sixty|seventy|eighty|ninety|शून्य|एक|दो|तीन|चार|पाँच|पांच|छह|छः|सात|आठ|नौ|"
                          r"दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह|पन्द्रह|सोलह|सत्रह|अठारह|उन्नीस|बीस|तीस|चालीस|"
                          r"पचास|साठ|सत्तर|अस्सी|नब्बे)(?![\w\u0900-\u097f])", re.IGNORECASE)
_NUMBER_WORD_VALUES = {
    **{word: value for value, word in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())},
    **dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10))),
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11,
    "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "तीस": 30, "चालीस": 40,
    "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}
_SPOKEN_EQUATION = re.compile(
    r"\b(\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by)\s+"
    r"(\d+(?:\.\d+)?)\s+(?:is equal to|equals|is)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE
)
_LETTER_NAMES_HI = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी",
    "H": "एच", "I": "आई", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन",
    "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू",
    "V": "वी", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}
_ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split())
_SCIENCE_TERMS_HI = {
    "red cabbage": ("लाल गोभी", "लाल पत्तागोभी"),
    "bacteria": ("जीवाणु", "बैक्टीरिया"),
    "bacterium": ("जीवाणु", "बैक्टीरियम"),
    "virus": ("विषाणु", "वायरस"),
    "viruses": ("विषाणु", "वायरस"),
    "electric current": ("विद्युत धारा", "बिजली की धारा", "करंट"),
    "vinegar": ("सिरका",),
}
_EQUATION_OPERATORS = {
    "plus": re.compile(r"plus|added to|\+|जोड़|योग|प्लस", re.IGNORECASE),
    "minus": re.compile(r"minus|subtract|(?<!\w)-(?!\w)|घटा|माइनस", re.IGNORECASE),
    "times": re.compile(r"times|multiplied by|multiply|×|\*|गुणा", re.IGNORECASE),
    "divide": re.compile(r"divided by|divide|÷|/|भाग", re.IGNORECASE),
}


def _canonical_formula(value: str) -> str:
    return value.translate(_SUBSCRIPT).replace(" ", "")


def _formula_tokens(text: str) -> list[tuple[str, int, int]]:
    normalized = (text or "").translate(_SUBSCRIPT)
    found = []
    for match in _FORMULA.finditer(normalized):
        token = match.group()
        base = re.sub(r"[⁺⁻+\-]\d*$", "", token)
        elements = _ELEMENT.findall(base)
        remainder = _ELEMENT.sub("", base)
        remainder = re.sub(r"\d+|[()[\]·.]", "", remainder)
        if not elements or remainder or any(element not in _ELEMENTS for element in elements):
            continue
        if len(elements) < 2 and not re.search(r"\d", base):
            continue
        found.append((_canonical_formula(token), match.start(), match.end()))
    return found


def _numbers_outside_formulas(text: str, include_hindi_one: bool = True) -> Counter[str]:
    normalized = (text or "").translate(_SUBSCRIPT).translate(_DIGITS)
    formula_spans = [(start, end) for _token, start, end in _formula_tokens(normalized)]
    # Mask formulas before reading number words/digits so H2O's 2 is never treated as narration content.
    safe = list(normalized)
    for start, end in formula_spans:
        safe[start:end] = " " * (end - start)
    safe_text = "".join(safe)
    values = Counter(value for value, _raw, _start, _end in _number_tokens_outside_formulas(normalized))
    words = [(match.group().casefold(), match.start(), match.end()) for match in _NUMBER_WORD.finditer(safe_text)]
    consumed: set[int] = set()
    index = 0
    while index < len(words):
        word, _start, end = words[index]
        if word == "एक" and not include_hindi_one:
            index += 1
            continue
        value = _NUMBER_WORD_VALUES[word]
        # Combine simple English cardinal phrases such as “twenty five” into 25.
        if value >= 20 and value % 10 == 0 and index + 1 < len(words):
            next_word, next_start, _next_end = words[index + 1]
            unit = _NUMBER_WORD_VALUES[next_word]
            if next_start >= end and next_start - end <= 8 and 1 <= unit <= 9:
                value += unit
                consumed.add(index + 1)
        if index not in consumed:
            values[str(value)] += 1
        index += 1
    return values


def _spoken_number_tokens(text: str) -> list[tuple[str, int, int]]:
    normalized = (text or "").translate(_SUBSCRIPT).translate(_DIGITS)
    formula_spans = [(start, end) for _token, start, end in _formula_tokens(normalized)]
    safe = list(normalized)
    for start, end in formula_spans:
        safe[start:end] = " " * (end - start)
    words = [(match.group().casefold(), match.start(), match.end()) for match in _NUMBER_WORD.finditer("".join(safe))]
    result: list[tuple[str, int, int]] = []
    index = 0
    while index < len(words):
        word, start, end = words[index]
        value = _NUMBER_WORD_VALUES[word]
        if value >= 20 and value % 10 == 0 and index + 1 < len(words):
            next_word, next_start, next_end = words[index + 1]
            unit = _NUMBER_WORD_VALUES[next_word]
            if next_start >= end and next_start - end <= 8 and 1 <= unit <= 9:
                value += unit
                end = next_end
                index += 1
        result.append((str(value), start, end))
        index += 1
    return result


def _number_tokens_outside_formulas(text: str) -> list[tuple[str, str, int, int]]:
    normalized = (text or "").translate(_SUBSCRIPT).translate(_DIGITS)
    formula_spans = [(start, end) for _token, start, end in _formula_tokens(normalized)]
    values = []
    for match in _NUMBER.finditer(normalized):
        if any(start <= match.start() < end for start, end in formula_spans):
            continue
        value = match.group().replace(",", "")
        values.append((value, match.group(), match.start(), match.end()))
    return values


def _equation_as_hindi(source: str) -> str | None:
    """Create a short Hindi equation only for a complete, arithmetically consistent source equation."""
    match = _SPOKEN_EQUATION.fullmatch((source or "").strip().rstrip(".!?"))
    if not match:
        return None
    left, operator, right, result = match.groups()
    a, b, stated = Decimal(left), Decimal(right), Decimal(result)
    op = operator.lower()
    if op == "plus":
        calculated, phrase = a + b, f"{left} में {right} जोड़ने पर {result} मिलता है।"
    elif op == "minus":
        calculated, phrase = a - b, f"{left} में से {right} घटाने पर {result} मिलता है।"
    elif op in {"times", "multiplied by"}:
        calculated, phrase = a * b, f"{left} को {right} से गुणा करने पर {result} मिलता है।"
    else:
        if not b:
            return None
        calculated, phrase = a / b, f"{left} को {right} से भाग देने पर {result} मिलता है।"
    return phrase if calculated == stated else None


def check_translation_segment(shot_id: str, source: str, translation: str, target_language: str = "") -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not (source or "").strip():
        return [{"shot_id": shot_id, "kind": "missing_source", "source": "", "translation": translation or "",
                 "detail": "No original source narration was stored with this translated line."}]
    if not (translation or "").strip():
        return [{"shot_id": shot_id, "kind": "missing_translation", "source": source,
                 "translation": "", "detail": "This timed source line has no translated narration."}]
    for equation in _SPOKEN_EQUATION.finditer(source):
        left, operator, right, result = equation.groups()
        try:
            a, b, expected = Decimal(left), Decimal(right), Decimal(result)
            if operator.lower() == "plus": calculated = a + b
            elif operator.lower() == "minus": calculated = a - b
            elif operator.lower() in {"times", "multiplied by"}: calculated = a * b
            else: calculated = a / b if b else None
        except (InvalidOperation, ZeroDivisionError):
            calculated, expected = None, Decimal(result)
        if calculated is not None and calculated != expected:
            issues.append({"shot_id": shot_id, "kind": "source_math_inconsistent", "source": equation.group(),
                           "translation": translation or "",
                           "detail": f"Source equation evaluates to {calculated}, not {expected}; check source ASR and the visual before dubbing approval."})
    source_formulas = Counter(token for token, _start, _end in _formula_tokens(source))
    translated_formulas = Counter(token for token, _start, _end in _formula_tokens(translation))
    if source_formulas != translated_formulas:
        issues.append({"shot_id": shot_id, "kind": "formula_mismatch",
                       "source": ", ".join(source_formulas.elements()),
                       "translation": ", ".join(translated_formulas.elements()),
                       "detail": "Chemical formula tokens must match the source exactly."})
    source_numbers = _numbers_outside_formulas(source)
    # Hindi "एक" also serves as an indefinite article. Count its presence as one
    # spoken 1 only when the source contains 1 and no less ambiguous 1 was found.
    # Counting every occurrence creates false failures in ordinary prose.
    translated_numbers = _numbers_outside_formulas(translation, include_hindi_one=False)
    if source_numbers["1"] > translated_numbers["1"]:
        hindi_ones = len(re.findall(r"(?<![\w\u0900-\u097f])एक(?![\w\u0900-\u097f])", translation))
        translated_numbers["1"] += min(source_numbers["1"] - translated_numbers["1"], hindi_ones)
    if source_numbers != translated_numbers:
        issues.append({"shot_id": shot_id, "kind": "number_mismatch",
                       "source": ", ".join(source_numbers.elements()),
                       "translation": ", ".join(translated_numbers.elements()),
                       "detail": "Numeric values outside formulas must be preserved exactly (Devanagari digits are accepted)."})
    for equation in _SPOKEN_EQUATION.finditer(source):
        source_operator = equation.group(2).lower()
        expected = "plus" if source_operator == "plus" else "minus" if source_operator == "minus" else \
            "times" if source_operator in {"times", "multiplied by"} else "divide"
        found = {name for name, pattern in _EQUATION_OPERATORS.items() if pattern.search(translation)}
        if found != {expected}:
            issues.append({"shot_id": shot_id, "kind": "equation_operator_mismatch" if found else "equation_operator_unverified",
                           "source": equation.group(), "translation": translation,
                           "detail": f"Source operation is {expected}; translated operation(s): {', '.join(sorted(found)) or 'none recognized'}."})
    if target_language.lower().startswith("hi"):
        for term, accepted in _SCIENCE_TERMS_HI.items():
            if re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", source, re.IGNORECASE):
                if not any(value in translation for value in accepted):
                    issues.append({"shot_id": shot_id, "kind": "science_term_unverified", "source": term,
                                   "translation": translation, "detail": "Expected: " + " / ".join(accepted) + "."})
    return issues


def validate_translation_storyboard(board: Storyboard) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    target_language = (board.language or {}).get("bcp47", "")
    for _scene, segment in board.all_segments():
        shot_id = segment.shot_id or str(segment.segment_number)
        issues.extend(check_translation_segment(shot_id, segment.source_narration, segment.narration, target_language))
    return issues


def repair_protected_content(board: Storyboard) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Repair unambiguous formula/number substitutions before TTS, then report anything unsafe."""
    repairs: list[dict[str, str]] = []
    for _scene, segment in board.all_segments():
        source = segment.source_narration or ""
        text = segment.narration or ""
        original_text = text
        repair_start = len(repairs)
        shot_id = segment.shot_id or str(segment.segment_number)
        if not source.strip() or not text.strip():
            continue
        before = text
        source_formulas = _formula_tokens(source)
        target_formulas = _formula_tokens(text)
        # Restore a source formula/identifier that the Hindi translator spelled phonetically (e.g. B2 -> बी 2).
        if len(target_formulas) < len(source_formulas):
            for expected, _start, _end in source_formulas:
                chunks = []
                for char in expected:
                    if char.isalpha() and char.upper() in _LETTER_NAMES_HI:
                        chunks.append(re.escape(_LETTER_NAMES_HI[char.upper()]))
                    else:
                        chunks.append(re.escape(char))
                alias = r"(?<!\w)" + r"\s*".join(chunks) + r"(?!\w)"
                alias_match = re.search(alias, text, flags=re.IGNORECASE)
                if alias_match:
                    text = text[:alias_match.start()] + expected + text[alias_match.end():]
                    repairs.append({"shot_id": shot_id, "kind": "formula_transliteration", "from": alias_match.group(), "to": expected})
            target_formulas = _formula_tokens(text)
        if len(source_formulas) == len(target_formulas):
            for (expected, _sa, _sb), (actual, start, end) in reversed(list(zip(source_formulas, target_formulas))):
                if expected != actual:
                    text = text[:start] + expected + text[end:]
                    repairs.append({"shot_id": shot_id, "kind": "formula", "from": actual, "to": expected})

        source_numbers = _number_tokens_outside_formulas(source)
        target_numbers = _number_tokens_outside_formulas(text)
        # When token count and order align, fix all changed values right-to-left so narration offsets remain valid.
        if len(source_numbers) == len(target_numbers):
            for (expected, raw_expected, _sa, _sb), (actual, _raw_actual, start, end) in reversed(list(zip(source_numbers, target_numbers))):
                if expected != actual:
                    text = text[:start] + raw_expected + text[end:]
                    repairs.append({"shot_id": shot_id, "kind": "number", "from": actual, "to": expected})
        elif len(source_numbers) > 0:
            # A complete source equation can be safely spoken from a Hindi template if its arithmetic is consistent.
            canonical = _equation_as_hindi(source)
            if canonical and text != canonical:
                text = canonical
                repairs.append({"shot_id": shot_id, "kind": "equation_rebuild", "from": before, "to": canonical})
        elif not source_numbers:
            # If the source spells out values (“six”) and Hindi uses digits, repair a one-to-one numeric mismatch.
            spoken = _spoken_number_tokens(source)
            target_values = _number_tokens_outside_formulas(text)
            if spoken and len(spoken) == len(target_values):
                for (expected, _sa, _sb), (actual, _raw, start, end) in reversed(list(zip(spoken, target_values))):
                    if expected != actual:
                        text = text[:start] + expected + text[end:]
                        repairs.append({"shot_id": shot_id, "kind": "spoken_number", "from": actual, "to": expected})
        if text != original_text and len(repairs) > repair_start:
            repairs.append({"shot_id": shot_id, "kind": "narration_corrected",
                            "original_narration": original_text, "corrected_narration": text})
        segment.narration = text
    return repairs, validate_translation_storyboard(board)
