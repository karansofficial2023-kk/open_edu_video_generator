"""Narration must be speakable: LaTeX that a language model leaves inside a sentence ("$ x = 1 $", "\\frac{1}{x^2}") is read aloud by the
voice ("dollar x dollar") and shown literally in subtitles. This turns it into plain words; exact formulas stay on screen through formula_lines."""
from __future__ import annotations

import re

_GREEK = {"alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta", "epsilon": "epsilon", "theta": "theta", "lambda": "lambda",
          "mu": "mu", "pi": "pi", "sigma": "sigma", "omega": "omega", "phi": "phi", "rho": "rho", "tau": "tau", "Delta": "delta",
          "Sigma": "sigma", "Omega": "omega", "Lambda": "lambda", "Phi": "phi", "Pi": "pi"}
# (command pattern, English words, symbol for non-English text)
_SYMBOLS = [
    (r"\\(?:Rightarrow|implies|Longrightarrow)", " which implies ", " ⇒ "),
    (r"\\(?:Leftrightarrow|iff)", " if and only if ", " ⇔ "),
    (r"\\(?:rightarrow|to|longrightarrow)(?![A-Za-z])", " goes to ", " → "),
    (r"\\pm", " plus or minus ", " ± "),
    (r"\\mp", " minus or plus ", " ∓ "),
    (r"\\(?:times|cdot)", " times ", " × "),
    (r"\\div", " divided by ", " ÷ "),
    (r"\\(?:leq|le)(?![A-Za-z])", " is less than or equal to ", " ≤ "),
    (r"\\(?:geq|ge)(?![A-Za-z])", " is greater than or equal to ", " ≥ "),
    (r"\\(?:neq|ne)(?![A-Za-z])", " is not equal to ", " ≠ "),
    (r"\\approx", " is approximately ", " ≈ "),
    (r"\\infty", " infinity ", " ∞ "),
    (r"\\(?:ldots|dots|cdots)", " and so on ", " … "),
    (r"\\sum", " the sum of ", " Σ "),
    (r"\\int", " the integral of ", " ∫ "),
    (r"\\lim", " the limit of ", " lim "),
    (r"\\ln", " natural log of ", " ln "),
    (r"\\log", " log of ", " log "),
    (r"\\sin", " sine of ", " sin "),
    (r"\\cos", " cosine of ", " cos "),
    (r"\\tan", " tangent of ", " tan "),
]
_OPERATORS = [("=", " equals "), ("<", " is less than "), (">", " is greater than "), ("+", " plus "), ("*", " times "), ("-", " minus ")]


def spoken_math(text: str, english: bool = True) -> str:
    """Replace $...$, $$...$$ and stray LaTeX commands in prose by words (or plain symbols when the lesson is not in English)."""
    if text and english:
        # a ratio "1:1.21" is read by the voice as "one one point twenty-one" and split by the subtitles at its decimal point;
        # a teacher says "1 to 1.21" (clock times such as 10:30 are left alone)
        text = re.sub(r"(?<![\d:])(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)(?![\d:])",
                      lambda m: m.group(0) if re.fullmatch(r"\d{1,2}:\d{2}", m.group(0)) else f"{m.group(1)} to {m.group(2)}", text)
    if not text or ("$" not in text and "\\" not in text):
        return text
    inline = lambda match: " " + latex_to_words(match.group(1), english) + " "
    display = lambda match: " " + latex_to_words(match.group(1), english) + ". "        # a displayed equation is its own spoken sentence
    out = re.sub(r"\$\$(.+?)\$\$", display, text, flags=re.S)
    out = re.sub(r"\$(.+?)\$", inline, out, flags=re.S)
    out = out.replace("$", " ")                                  # an unbalanced delimiter (a formula cut off mid-way)
    if "\\" in out:
        out = latex_to_words(out, english)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([.:;!?])\s*\.", r"\1", out)                   # ".." / ":." left by a display equation before punctuation
    out = re.sub(r"\.\s*([,;:])", r"\1", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    return out.strip()


def latex_to_words(latex: str, english: bool = True) -> str:
    s = latex
    s = re.sub(r"\\(?:left|right|bigl|bigr|big|Big|bigg|Bigg)(?![A-Za-z])", "", s)
    s = re.sub(r"\\[,;:! ]|\\(?:quad|qquad)(?![A-Za-z])", " ", s)
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
    # derivative notation first: f''(x) -> f double dash of x ; f'(x) -> f dash of x
    s = re.sub(r"\b([A-Za-z])\s*''\s*\(([^()]*)\)", r"\1 double dash of \2", s)
    s = re.sub(r"\b([A-Za-z])\s*'\s*\(([^()]*)\)", r"\1 dash of \2", s)
    s = re.sub(r"\b([A-Za-z])\s*''", r"\1 double dash", s)
    s = re.sub(r"\b([A-Za-z])\s*'", r"\1 dash", s)
    s = re.sub(r"\b([fghFGH])\(([^()]{1,12})\)", r"\1 of \2", s)
    # powers and subscripts (before fractions: braces must be gone for the fraction pattern)
    # a closing brace is consumed only when the exponent opened one: "{x^2}" inside a fraction must keep its "}"
    s = re.sub(r"\^\s*(?:\{\s*2\s*\}|2(?!\d))", " squared ", s)
    s = re.sub(r"\^\s*(?:\{\s*3\s*\}|3(?!\d))", " cubed ", s)
    s = re.sub(r"\^\s*\{\s*-\s*1\s*\}", " to the power minus 1 ", s)
    s = re.sub(r"\^\s*\{([^{}]+)\}", r" to the power \1 ", s)
    s = re.sub(r"\^\s*(\d+|[A-Za-z])", r" to the power \1 ", s)
    s = re.sub(r"_\s*\{([^{}]*)\}", r" sub \1 ", s)
    s = re.sub(r"_\s*(\w)", r" sub \1 ", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", (r" the square root of \1 " if english else r" √(\1) "), s)
    for _ in range(8):                                           # nested fractions: innermost first
        replaced = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", (r" \1 over \2 " if english else r" (\1)/(\2) "), s)
        if replaced == s:
            break
        s = replaced
    for pattern, words, symbol in _SYMBOLS:
        s = re.sub(pattern, words if english else symbol, s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: " " + _GREEK.get(m.group(1), m.group(1)) + " ", s)       # \alpha -> alpha, unknown command -> its name
    s = s.replace("{", " ").replace("}", " ").replace("\\", " ")
    if english:
        for symbol, words in _OPERATORS:
            s = s.replace(symbol, words)
        s = re.sub(r"(?<=\w)\s*/\s*(?=\w)", " over ", s)
    return re.sub(r"\s+", " ", s).strip()
