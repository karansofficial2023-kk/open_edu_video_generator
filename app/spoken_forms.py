"""Dubbing v2 (deep dive): turn approved Hindi narration into TTS-ready language spans.

STRICT RULE: a chemical formula or chemical equation is never read in Hindi.  Every
Latin-script run in the narration (formulas, equations, atom labels such as AB or B1,
acronyms such as AC) is spoken by an English voice, letter by letter for formulas.
Everything else (including bare numbers) is spoken by the Hindi voice, with digits
expanded into Hindi words, because Piper hi_IN cannot read digits or Latin text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_SUPERSCRIPT_CHARGE = str.maketrans({"⁺": "+", "⁻": "-", "−": "-"})
_ARROWS = ("⟶", "⟹", "→", "->", "-->", "=>")

_ELEMENT_SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())

# One formula "term": element groups, bracket groups, hydrates (·), charge and state symbol.
_TERM = (r"(?:\d+\s?(?=[A-Z(\[]))?"
         r"(?:(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*|\[(?:[A-Z][a-z]?\d*)+\]\d*)+"
         r"(?:[·.]\d*(?:[A-Z][a-z]?\d*)+)*)"
         r"(?:\^?\d*[+\-])?(?:\((?:s|l|g|aq)\))?")
_OPERATOR = r"(?:\+|=|⇌|⇄|↔|<->|<=>|⟶|⟹|→|-->|->|=>)"
_EQUATION = re.compile(rf"(?<![A-Za-z0-9]){_TERM}(?:\s*{_OPERATOR}\s*{_TERM})+(?![A-Za-z0-9])")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
_FORMULA_TOKEN = re.compile(rf"(?<![A-Za-z0-9])({_TERM})(?![A-Za-z0-9])")
_STATE = {"s": "solid", "l": "liquid", "g": "gas", "aq": "aqueous"}
_SYMBOL_WORDS = {"+": "plus", "=": "equals", "→": "gives", "⟶": "gives", "⟹": "gives", "->": "gives",
                 "-->": "gives", "=>": "gives", "⇌": "is in equilibrium with", "⇄": "is in equilibrium with",
                 "↔": "is in equilibrium with", "<->": "is in equilibrium with", "<=>": "is in equilibrium with"}
_EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_SMALL = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
             16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
_EN_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

# Common English words that look like element-symbol strings; they are read as words, never spelled.
_ENGLISH_WORDS = {"as", "at", "be", "he", "hi", "in", "is", "it", "no", "so", "us", "we", "an", "am", "go", "if",
                  "by", "me", "my", "of", "on", "or", "to", "up", "a", "i", "his", "has", "can", "con", "pan", "tin",
                  "iron", "gas", "sun", "ion", "ions", "pin", "sin", "son", "ban", "bat", "cat", "con", "hen"}
# Hindi unit spoken forms for "<number> <unit>" (unit is Latin in the narration).
_UNITS_HI = {"cm": "सेंटीमीटर", "mm": "मिलीमीटर", "m": "मीटर", "km": "किलोमीटर", "kg": "किलोग्राम", "g": "ग्राम",
             "mg": "मिलीग्राम", "s": "सेकंड", "ms": "मिलीसेकंड", "ml": "मिलीलीटर", "l": "लीटर", "hz": "हर्ट्ज़",
             "v": "वोल्ट", "a": "एम्पियर", "w": "वाट", "j": "जूल", "n": "न्यूटन", "pa": "पास्कल", "k": "केल्विन",
             "mol": "मोल", "kw": "किलोवाट", "ma": "मिलीएम्पियर", "kv": "किलोवोल्ट", "°c": "डिग्री सेल्सियस",
             "ohm": "ओम", "m/s": "मीटर प्रति सेकंड", "km/h": "किलोमीटर प्रति घंटा", "rpm": "आरपीएम"}


@dataclass(frozen=True)
class Span:
    lang: str      # "hi" or "en"
    text: str      # text exactly as the voice should read it


def normalize_chemistry_text(text: str) -> str:
    """Canonical Latin form: subscripts -> digits, Devanagari digits kept for the Hindi side only."""
    return (text or "").translate(_SUBSCRIPTS).translate(_SUPERSCRIPT_CHARGE)


def english_number(value: int) -> str:
    if value < 0:
        return "minus " + english_number(-value)
    if value < 10:
        return _EN_DIGITS[value]
    if value < 20:
        return _EN_SMALL[value]
    if value < 100:
        tens, rest = divmod(value, 10)
        return _EN_TENS[tens] + (" " + _EN_DIGITS[rest] if rest else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return _EN_DIGITS[hundreds] + " hundred" + (" " + english_number(rest) if rest else "")
    return " ".join(_EN_DIGITS[int(ch)] for ch in str(value))


_HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
    "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
    "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
    "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ "
    "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी अस्सी इक्यासी "
    "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे "
    "छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()


def _hindi_integer(value: int) -> str:
    if value < 100:
        return _HI_0_99[value]
    for size, word in ((10_000_000, "करोड़"), (100_000, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if value >= size:
            head, rest = divmod(value, size)
            return _hindi_integer(head) + " " + word + (" " + _hindi_integer(rest) if rest else "")
    return _HI_0_99[value]


def hindi_number(value: str) -> str:
    """Hindi words for an integer or decimal string (0 .. crores)."""
    whole, _, fraction = value.partition(".")
    spoken = _hindi_integer(int(whole or "0"))
    if fraction:
        spoken += " दशमलव " + " ".join(_HI_0_99[int(ch)] for ch in fraction)
    return spoken


def spell_formula(token: str) -> str:
    """English spoken form of one formula term, e.g. 2H2SO4 -> 'two H two S O four'."""
    token = normalize_chemistry_text(token).strip()
    words: list[str] = []
    state = re.search(r"\((s|l|g|aq)\)$", token)
    if state:
        token = token[:state.start()]
    charge = re.search(r"\^?(\d*)([+\-])$", token)
    charge_words: list[str] = []
    if charge and re.search(r"[A-Za-z)\]\d]", token[:charge.start()] or ""):
        token = token[:charge.start()]
        if charge.group(1):
            charge_words.append(english_number(int(charge.group(1))))
        charge_words.append("plus" if charge.group(2) == "+" else "minus")
    index = 0
    while index < len(token):
        char = token[index]
        if char.isdigit():
            end = index
            while end < len(token) and token[end].isdigit():
                end += 1
            words.append(english_number(int(token[index:end])))
            index = end
        elif char.isalpha():
            words.append(char.upper())
            index += 1
        elif char in "([":
            words.append("open bracket")
            index += 1
        elif char in ")]":
            words.append("close bracket")
            index += 1
        elif char in "·.":
            words.append("dot")
            index += 1
        elif char == "/":
            words.append("by")
            index += 1
        else:
            index += 1
    if charge_words and words:
        words[-1] += ","
    words.extend(charge_words)
    if state:
        words.append(_STATE[state.group(1)])
    return " ".join(words)


def spell_equation(equation: str) -> str:
    """English spoken form of a full chemical equation (operators become words)."""
    text = normalize_chemistry_text(equation)
    parts = re.split(f"({_OPERATOR})", text)
    spoken = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(_OPERATOR, part):
            spoken.append(_SYMBOL_WORDS.get(part, "gives"))
        else:
            spoken.append(spell_formula(part))
    return ", ".join(spoken)


def is_formula_token(token: str) -> bool:
    """True when a Latin token is a chemical formula / label / acronym that must be spelled in English."""
    token = normalize_chemistry_text(token)
    base = re.sub(r"\((?:s|l|g|aq)\)$", "", token)
    base = re.sub(r"\^?\d*[+\-]$", "", base) if re.search(r"[A-Za-z)\]\d]\^?\d*[+\-]$", base) else base
    if base.lower() in _ENGLISH_WORDS or not base:
        return False
    if re.fullmatch(_TERM, base) is None:
        return False
    if re.search(r"\d", base):
        return True
    if base.isupper():                               # AB, CHO, COOH
        return len(base) <= 6
    # Mixed case without digits (NaCl, HCl, Al, Fe): every chunk must be a real element symbol.
    chunks = re.findall(r"[A-Z][a-z]?", base)
    return bool(chunks) and "".join(chunks) == base and all(c in _ELEMENT_SYMBOLS for c in chunks)


def _spoken_latin_run(run: str) -> str:
    out = []
    for word in run.split():
        if re.fullmatch(r"(?:[A-Za-z]-)+[A-Za-z]", word):          # O-N-E -> O N E
            out.append(" ".join(ch.upper() for ch in word if ch.isalpha()))
        elif is_formula_token(word):
            out.append(spell_formula(word))
        elif re.fullmatch(r"[A-Za-z]", word):
            out.append(word.upper())
        else:
            out.append(word)
    return " ".join(out)


def _hindi_text_for_tts(text: str) -> str:
    """Expand digits/percent/degree signs for the Hindi voice. Latin never reaches this function."""
    text = text.translate(_DEVANAGARI_DIGITS)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", lambda m: hindi_number(m.group(1)) + " प्रतिशत", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*°\s*C?", lambda m: hindi_number(m.group(1)) + " डिग्री सेल्सियस", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: hindi_number(m.group()), text)
    text = text.replace("&", " और ")
    return re.sub(r"\s+([,;:.!?])", r"", re.sub(r"\s+", " ", text)).strip()


_MATH_NAMES_HI = {
    "sin": "साइन", "cos": "कॉस", "tan": "टैन", "log": "लॉग", "ln": "लॉग", "sqrt": "वर्गमूल",
    "x": "एक्स", "y": "वाई", "z": "ज़ेड", "a": "ए", "b": "बी", "c": "सी", "d": "डी",
    "m": "एम", "n": "एन", "p": "पी", "q": "क्यू", "r": "आर", "t": "टी",
}


def math_to_hindi(text: str) -> str:
    """Speak mathematical notation with the Hindi voice, including fractions, powers and operators."""
    value = (text or "").translate(_SUBSCRIPTS).translate(_DEVANAGARI_DIGITS)
    value = re.sub(r"([A-Za-z0-9)]+)\s*\^\s*([A-Za-z0-9]+)", r"\1 की घात \2", value)
    value = re.sub(r"([A-Za-z0-9)]+)\s*/\s*([A-Za-z0-9(]+)", r"\1 बटा \2", value)
    for symbol, spoken in {"≥": " से बड़ा या बराबर ", "≤": " से छोटा या बराबर ", "≠": " बराबर नहीं ",
                           "=": " बराबर ", "+": " जमा ", "×": " गुणा ", "*": " गुणा ", "÷": " भाग ",
                           "−": " घटा ", "-": " घटा ", "%": " प्रतिशत ", "∞": " अनंत ",
                           "(": " खुला कोष्ठक ", ")": " बंद कोष्ठक "}.items():
        value = value.replace(symbol, spoken)
    def name(match):
        token = match.group().lower()
        return _MATH_NAMES_HI.get(token, " ".join(_MATH_NAMES_HI.get(char, char) for char in token))
    value = re.sub(r"[A-Za-z]+", name, value)
    return _hindi_text_for_tts(value)


_UNIT_ALT = "|".join(sorted((re.escape(u) for u in _UNITS_HI if u != "°c"), key=len, reverse=True))
_UNIT_PHRASE = re.compile(rf"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s?({_UNIT_ALT})(?![A-Za-z0-9])", re.IGNORECASE)
_CHARGE_TAIL = re.compile(r"(?:\s?\^?\d*[+\-](?![\w+\-]))")
_SCAN = re.compile(rf"(?P<eq>{_EQUATION.pattern})|(?P<term>(?<![A-Za-z0-9]){_TERM}(?![A-Za-z0-9]))|(?P<word>{_LATIN_WORD.pattern})")


# Structural notation (checked against the STEM prompt review): C=O is a double bond, never "C equals O"; C≡N is a triple
# bond; R-OH has a single bond; -COOH is the carboxyl group; -one / -al / -ol are IUPAC suffixes, spoken the way they sound.
_R_GROUPS = {"R", "X", "Y", "Ar"}
_BOND_WORDS = {"=": "double bond", "≡": "triple bond", "-": "dash"}
_BOND = re.compile(rf"(?<![A-Za-z0-9])-?{_TERM}(?:[=≡\-]{_TERM})+(?![A-Za-z0-9])")
_GROUP = re.compile(r"(?<![A-Za-z0-9])-(?:COOH|COOR|COCl|CHO|OH|NH2|NO2|CN|CH3|C2H5|X)(?![A-Za-z0-9])")
_SUFFIX_WORDS = {"one": "suffix own", "al": "suffix al", "ol": "suffix all", "oic": "suffix oh ick", "ane": "suffix ain",
                 "ene": "suffix een", "yne": "suffix ine", "amine": "suffix amine", "oate": "suffix oh ate"}
_SUFFIX = re.compile(r"(?<![A-Za-z0-9])-(" + "|".join(sorted(_SUFFIX_WORDS, key=len, reverse=True)) + r")(?![A-Za-z0-9])")


def _spell_bond(expression: str) -> str | None:
    """'C=O' -> 'C double bond O'. None when the pieces are not atoms/groups (e.g. the physics formula F=BIL)."""
    leading = expression.startswith("-")
    body = expression.lstrip("-")
    parts = re.split(r"[=≡\-]", body)
    operators = re.findall(r"[=≡\-]", body)
    for part in parts:
        chunks = re.findall(r"[A-Z][a-z]?", part)
        if not chunks or "".join(chunks) != re.sub(r"\d", "", part) or any(
                chunk not in _ELEMENT_SYMBOLS and chunk not in _R_GROUPS for chunk in chunks):
            return None
    if all(len(part) == 1 for part in parts) and set(operators) == {"-"}:      # O-N-E, C-O-C spelled as letters
        return ("dash " if leading else "") + " ".join(parts)
    spoken = [spell_formula(parts[0])]
    for operator, part in zip(operators, parts[1:]):
        spoken.extend([_BOND_WORDS[operator], spell_formula(part)])
    return ("dash " if leading else "") + " ".join(spoken)


def split_spans(narration: str, formula_speech_mode: str = "english_stem") -> list[Span]:
    """Split narration into ordered Hindi/English spans. Formulas and equations become English spans."""
    if formula_speech_mode == "hindi_math":
        spoken = math_to_hindi(narration)
        return [Span("hi", spoken)] if spoken else []
    text = normalize_chemistry_text(narration or "")
    for arrow in _ARROWS:
        text = text.replace(arrow, " → ")
    pieces: list[tuple[int, int, str, str]] = []        # (start, end, lang, spoken)
    claimed: list[tuple[int, int]] = []
    for bond in _BOND.finditer(text):
        spoken = _spell_bond(bond.group())
        if spoken:
            pieces.append((bond.start(), bond.end(), "en", spoken))
            claimed.append((bond.start(), bond.end()))
    for group in _GROUP.finditer(text):
        if not any(a <= group.start() < b for a, b in claimed):
            pieces.append((group.start(), group.end(), "en", "dash " + spell_formula(group.group()[1:])))
            claimed.append((group.start(), group.end()))
    for suffix in _SUFFIX.finditer(text):
        if not any(a <= suffix.start() < b for a, b in claimed):
            pieces.append((suffix.start(), suffix.end(), "en", _SUFFIX_WORDS[suffix.group(1)]))
            claimed.append((suffix.start(), suffix.end()))
    for unit in _UNIT_PHRASE.finditer(text):           # "5 cm", "10 m/s" -> one Hindi phrase
        spoken = hindi_number(unit.group(1)) + " " + _UNITS_HI[unit.group(2).lower()]
        pieces.append((unit.start(), unit.end(), "hi", spoken))
        claimed.append((unit.start(), unit.end()))
    english_runs: list[list] = []                       # merged Latin runs: [start, end, spoken_parts]
    for match in _SCAN.finditer(text):
        if any(a <= match.start() < b for a, b in claimed):
            continue
        start, end = match.start(), match.end()
        if match.group("eq"):
            pieces.append((start, end, "en", spell_equation(match.group("eq"))))
            claimed.append((start, end))
            continue
        token = match.group()
        if match.group("term") and is_formula_token(token):
            tail = _CHARGE_TAIL.match(text, end)
            if tail and re.search(r"\d|[A-Za-z)\]]", token) and not token.endswith(("+", "-")):
                token, end = token + "^" + tail.group().strip().lstrip("^"), tail.end()
            spoken = spell_formula(token)
        elif match.group("term") and re.fullmatch(_TERM, token) and not re.fullmatch(r"[A-Za-z]{2,}", token):
            spoken = _spoken_latin_run(token)
        else:
            spoken = _spoken_latin_run(token)
        if english_runs and re.fullmatch(r"[ 	]+", text[english_runs[-1][1]:start]) and                 not (english_runs[-1][3] or match.group("term") and is_formula_token(token)):
            english_runs[-1][1] = end
            english_runs[-1][2].append(spoken)
        elif english_runs and re.fullmatch(r"[ 	]+", text[english_runs[-1][1]:start]):
            english_runs[-1][1] = end
            english_runs[-1][2].append(spoken)
            english_runs[-1][3] = english_runs[-1][3] or bool(match.group("term") and is_formula_token(token))
        else:
            english_runs.append([start, end, [spoken], bool(match.group("term") and is_formula_token(token))])
    for start, end, parts, _is_formula in english_runs:
        pieces.append((start, end, "en", " ".join(parts)))
    spans: list[Span] = []
    cursor = 0
    for start, end, lang, spoken in sorted(pieces):
        if start < cursor:
            continue
        hindi = _hindi_text_for_tts(text[cursor:start])
        if hindi and re.search(r"[\wऀ-ॿ]", hindi):
            spans.append(Span("hi", hindi))
        if spoken.strip():
            spans.append(Span(lang, spoken))
        cursor = end
    tail_text = _hindi_text_for_tts(text[cursor:])
    if tail_text and re.search(r"[\wऀ-ॿ]", tail_text):
        spans.append(Span("hi", tail_text))
    return _merge_adjacent(spans)


def _merge_adjacent(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in spans:
        if merged and merged[-1].lang == span.lang:
            joiner = ", " if span.lang == "en" else " "
            merged[-1] = Span(span.lang, merged[-1].text + joiner + span.text)
        else:
            merged.append(span)
    return merged


def english_span_count(narration: str) -> int:
    return sum(1 for span in split_spans(narration) if span.lang == "en")


def hindi_only_text(narration: str) -> str:
    """The part of the narration a Hindi ASR can be expected to recognise (formulas/labels removed)."""
    return " ".join(span.text for span in split_spans(narration) if span.lang == "hi")
