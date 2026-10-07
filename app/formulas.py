"""Deterministic equation rendering (matplotlib mathtext, open source) with strict preflight validation."""
from __future__ import annotations

import io
import re
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

_ARROWS = {"\u2192": r"\rightarrow ", "\u21cc": r"\rightleftharpoons ", "->": r"\rightarrow ", "<->": r"\leftrightarrow ",
           "\u2264": r"\leq ", "\u2265": r"\geq ", "\u2248": r"\approx ", "\u00b0": r"^{\circ}", "\u00b1": r"\pm ",
           "\u00d7": r"\times ", "\u00b7": r"\cdot ", "\u0394": r"\Delta ", "\u039b": r"\Lambda ", "\u03bb": r"\lambda ",
           "\u03bc": r"\mu ", "\u03c1": r"\rho ", "\u03c3": r"\sigma ", "\u03ba": r"\kappa ", "\u03a9": r"\Omega "}
_SPECIES = re.compile(r"(?<![A-Za-z\\])((?:[A-Z][a-z]?\d*)+)(?![a-z])")


class FormulaError(ValueError):
    pass


def is_prose(line: str) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[A-Za-z]+", "", line))
    has_operator = bool(re.search(r"[=+\-*/^_<>\u2192\u21cc\u2248\u2264\u2265]|->", line))
    return len(words) >= 5 and not has_operator or len(words) >= 9


_ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo "
                "Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb "
                "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm".split())



_SUP = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
        "\u2078": "8", "\u2079": "9", "\u207a": "+", "\u207b": "-", "\u207c": "=", "\u207f": "n", "\u221e": r"\infty"}
_SUB = {"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7",
        "\u2088": "8", "\u2089": "9", "\u208a": "+", "\u208b": "-"}
_SPECIES_TOKEN = re.compile(r"(?<![A-Za-z\\])((?:[A-Z][a-z]?\d*){2,}|[A-Z][a-z]?\d+)(?![a-z])")
_ION = re.compile(r"(?<![A-Za-z\\])([A-Z][A-Za-z0-9]*?)(\d?[+-])(?=[)\s,]|$)")


def _script_runs(text: str, table: dict, marker: str) -> str:
    out, run = [], ""
    for ch in text:
        if ch in table:
            run += table[ch]
            continue
        if run:
            out.append(f"{marker}{{{run}}}")
            run = ""
        out.append(ch)
    if run:
        out.append(f"{marker}{{{run}}}")
    return "".join(out)


def normalize_formula(line: str) -> str:
    """Make storyboard formulas typeset-friendly: Unicode scripts, label prefixes, ions, chemical species, word subscripts."""
    text = line.strip()
    text = _script_runs(_script_runs(text, _SUP, "^"), _SUB, "_")
    # "Kohlrausch's law: x = y" -> keep only the mathematics (a short caption before a colon is not part of the equation)
    head, colon, tail = text.partition(":")
    if colon and "=" in tail and not re.search(r"[=<>+*/^_]", head) and len(head.split()) <= 6:
        text = tail.strip()
    text = text.replace(r"\text{", r"\mathrm{").replace(r"\textrm{", r"\mathrm{")
    # mathtext drops plain spaces: "\mathrm{strong cation}" would read "strongcation"; keep them as explicit spaces
    text = re.sub(r"\\mathrm\{([^{}]*)\}", lambda m: "\\mathrm{" + re.sub(r"(?<!\\) +", lambda _s: "\\ ", m.group(1).strip()) + "}", text)
    if "\\" in text.replace(r"\mathrm", ""):
        return _latex_species(text)      # already LaTeX: the chemistry rewrites below (CH_3 -> CH3, ions, species) would corrupt it, e.g. "\frac{L_0}{A_0}" -> "...{A0"
    if "mathrm" not in text:
        text = re.sub(r"(?<![a-z\\])([A-Z][a-z]?)_\{?(\d+)\}?(?=[A-Z(]|$)", r"\1\2", text)      # CH_3COOH -> CH3COOH

    def ion(match):
        base = re.sub(r"(\D)(\d+)", r"\1_{\2}", match.group(1))
        return r"\mathrm{" + base + "}^{" + match.group(2) + "}"

    text = _ION.sub(ion, text)                       # H+ -> \mathrm{H}^{+}, CH3COO- -> \mathrm{CH_{3}COO}^{-}
    text = re.sub(r"_\{([A-Za-z]{2,})\}", r"_{\\mathrm{\1}}", text)                         # _{cation} -> upright word subscript
    text = re.sub(r"_([A-Za-z]{2,})", r"_{\\mathrm{\1}}", text)                              # _cation
    if "mathrm" not in text:
        def species(match):
            token = match.group(1)
            pieces = re.findall(r"[A-Z][a-z]?", token)
            if all(piece in _ELEMENTS for piece in pieces) and (re.search(r"\d", token) or len(token) >= 3):
                return r"\mathrm{" + re.sub(r"([A-Za-z)])(\d+)", r"\1_{\2}", token) + "}"
            return token
        text = _SPECIES_TOKEN.sub(species, text)
    return text


def _latex_species(text: str) -> str:
    r"""Inside an already-LaTeX line only one safe chemistry rewrite: a bare species in parentheses, "(CH3COOH)" ->
    "(\mathrm{CH_{3}COOH})", so it is upright with real subscripts instead of italic letters and a full-size 3."""
    def species(match):
        token = match.group(1)
        pieces = re.findall(r"[A-Z][a-z]?\d*", token)
        if pieces and "".join(pieces) == token and re.search(r"\d", token) \
                and all(re.match(r"[A-Z][a-z]?", piece).group() in _ELEMENTS for piece in pieces):
            return "(" + r"\mathrm{" + re.sub(r"([A-Za-z])(\d+)", r"\1_{\2}", token) + "})"
        return match.group(0)
    return re.sub(r"\(([A-Za-z0-9]+)\)", species, text)


def looks_chemical(line: str) -> bool:
    """Reactions (arrows), state symbols, ionic charges, or a bare formula such as CuSO4 are chemistry; equations with '=' are math."""
    if re.search(r"\((aq|s|l|g)\)|->|<->|<=>|→|⇌|\^\{?\d*[+-]\}?|\^[+-]", line):
        return True
    if "=" in line:
        return False
    tokens = re.findall(r"[A-Z][a-z]?\d*", line)
    return bool(tokens) and all(re.match(r"[A-Z][a-z]?", t).group() in _ELEMENTS for t in tokens) and bool(re.search(r"[A-Z][a-z]?\d|[A-Z][A-Za-z]*[A-Z]", line))


_PROSE = r"[A-Za-z][A-Za-z' ,]*"
_CAPTION = re.compile(rf"^\s*(?P<caption>{_PROSE}):\s*(?P<rest>.+)$")
_NOTE = re.compile(rf"^(?P<core>.+?)\s*\((?P<note>{_PROSE})\)\s*$")


def _upright(words: str) -> str:
    """Prose inside an equation line: upright text with explicit spaces (mathtext drops plain spaces)."""
    return r"\mathrm{" + r"\ ".join(words.split()) + "}"


def _split_prose(line: str) -> tuple[str, str, str]:
    """('Oxidation', 'Ag -> Ag+ + e-', 'oxidation at anode') from 'Oxidation: Ag -> Ag+ + e- (oxidation at anode)'."""
    caption = note = ""
    rest = line
    match = _CAPTION.match(rest)
    if match and len(match["caption"].split()) <= 4 and "=" not in match["rest"] and re.search(r"[+→↔⇌<>^_/*-]|->", match["rest"]):
        caption, rest = match["caption"].strip(), match["rest"]
    match = _NOTE.match(rest)
    if match and len(match["note"].split()) >= 2 and match["note"].strip().lower() not in {"aq", "s", "l", "g"}:
        rest, note = match["core"], match["note"].strip()
    return caption, rest, note


def to_mathtext(line: str) -> str:
    caption, core, note = _split_prose(line)
    if not (caption or note):
        return _core_mathtext(line)
    body = _core_mathtext(core).strip("$")
    parts = ([_upright(caption + ":"), r"\ \ "] if caption else []) + [body] + ([r"\ \ (", _upright(note), ")"] if note else [])
    return "$" + "".join(parts) + "$"


def _core_mathtext(line: str) -> str:
    line = normalize_formula(line)
    if not looks_chemical(line):
        try:
            from .mathparse import math_to_latex
            return "$" + math_to_latex(line).replace("$", "") + "$"
        except Exception:        # fall back to the legacy symbol mapper below
            pass
    text = line.strip()
    for src, dst in _ARROWS.items():
        text = text.replace(src, dst)
    text = re.sub(r"\((aq|s|l|g)\)", r"\\mathrm{(\1)}", text)
    if "\\" not in text.replace("\\mathrm", "").replace("\\rightarrow", "").replace("\\rightleftharpoons", ""):
        def species(match):
            token = match.group(1)
            if not reaction and not re.search(r"\d", token) and len(token) > 3:
                return token
            spaced = re.sub(r"([A-Z][a-z]?)(\d+)", r"\1_{\2}", token)
            return r"\mathrm{" + spaced + "}"
        reaction = re.search(r"\\mathrm\{\((aq|s|l|g)\)\}|\\rightarrow|\\rightleftharpoons", text)
        if reaction or re.search(r"[A-Z][a-z]?\d", text):
            text = _SPECIES.sub(species, text)
    return "$" + text.replace("$", "") + "$"


def validate(line: str) -> str | None:
    """Return an error message, or None when the line is a renderable formula."""
    if not line.strip():
        return "empty formula line"
    if is_prose(line):
        return "prose in formula_lines; move it to explain_steps and provide the exact equation"
    depth = 0
    for ch in line:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth < 0:
            return "unbalanced brackets"
    if depth != 0:
        return "unbalanced brackets"
    if re.search(r"[\^_]\s*$", line):
        return "dangling ^ or _"
    try:
        mathtext.MathTextParser("path").parse(to_mathtext(line), dpi=72, prop=FontProperties(size=12))
    except Exception as exc:  # matplotlib raises ValueError with the parse position
        return f"mathtext cannot parse: {str(exc).splitlines()[0][:120]}"
    return None


@lru_cache(maxsize=128)
def render_line(line: str, color: str, px: int) -> Image.Image:
    """RGBA image of one equation whose glyph height is about `px` pixels."""
    error = validate(line)
    if error:
        raise FormulaError(f"{line!r}: {error}")
    buffer = io.BytesIO()
    mathtext.math_to_image(to_mathtext(line), buffer, prop=FontProperties(size=px * 0.75), dpi=96, format="png")
    buffer.seek(0)
    raw = Image.open(buffer).convert("L")              # math_to_image renders dark glyphs on an OPAQUE white background
    alpha = ImageOps.invert(raw)                       # ink coverage becomes the alpha channel (antialiasing preserved)
    bbox = alpha.point(lambda v: 255 if v > 8 else 0).getbbox()
    if bbox:
        pad = max(4, px // 5)
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(raw.width, bbox[2] + pad), min(raw.height, bbox[3] + pad))
        alpha = alpha.crop(bbox)
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    tinted = Image.new("RGBA", alpha.size, (r, g, b, 255))
    tinted.putalpha(alpha)
    return tinted
