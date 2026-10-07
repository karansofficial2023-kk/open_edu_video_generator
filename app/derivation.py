"""Symbolic derivation check: consecutive equations in a lesson must be algebraically equivalent rewrites of each other.

A wrong intermediate step in a derivation is a critical factual error that no image or voice QA can see. Equations are parsed
with sympy; two equations are treated as equivalent when (lhs - rhs) of one is a constant (possibly parameter-dependent)
multiple of (lhs - rhs) of the other. Lines that cannot be parsed or are not equations (inequalities, +-, labels) are skipped,
never failed: this check only raises *warnings* for SME review.
"""
from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_SUPER = str.maketrans("\u00b2\u00b3\u2074", "234")


def _frac(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", text)
    return text


def to_equation(line: str):
    """Parse 'lhs = rhs' into a sympy expression lhs - rhs, or None when the line is not a single parsable equation."""
    text = line.strip()
    if any(token in text for token in ("\u00b1", r"\pm", "<", ">", "\u2264", "\u2265", "->", "\u2192", ":", "\u21cc")):
        return None
    text = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", text)
    text = text.replace("\u00d7", "*").replace(r"\times", "*").replace(r"\cdot", "*").replace("\u2212", "-")
    text = text.replace("\u00b2", "^2").replace("\u00b3", "^3").replace("\u2074", "^4")
    text = _frac(text)
    text = text.replace("{", "(").replace("}", ")")
    if text.count("=") != 1 or re.search(r"[A-Za-z]{4,}", re.sub(r"sqrt|sin|cos|tan|log|exp", "", text)):
        return None
    left, right = text.split("=")
    try:
        locals_ = {"sqrt": sympy.sqrt, "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "log": sympy.log, "exp": sympy.exp}
        return parse_expr(left, local_dict=locals_, transformations=_TRANSFORMS) - parse_expr(right, local_dict=locals_, transformations=_TRANSFORMS)
    except Exception:
        return None


def equivalent(a, b) -> bool:
    """True when a and b are the same equation up to a nonzero multiplier (rearranging, dividing, moving terms)."""
    if a is None or b is None:
        return True
    try:
        ratio = sympy.simplify(a / b)
    except Exception:
        return True
    if ratio == 0 or ratio.has(sympy.zoo, sympy.nan):
        return False
    variables = (a.free_symbols | b.free_symbols)
    # a constant ratio (it may contain parameters such as a, b, c, but must not depend on the solved-for variable x)
    solved_for = {s for s in variables if str(s) in {"x", "y", "t", "z"}}
    return not (ratio.free_symbols & solved_for)


def check_chain(lines_by_shot: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    """lines_by_shot: [(shot_id, [formula lines])] in lesson order -> [(shot_id, message)] for steps that do not follow.

    A step that is a specialisation of the previous one (fewer symbols, e.g. a worked numeric example) is not compared. After a
    bad step the next line is also compared with the last good equation, so one error is reported once, not for every later line.
    """
    problems = []
    previous = good = None
    previous_line = good_line = ""
    for shot_id, lines in lines_by_shot:
        for line in lines:
            current = to_equation(line)
            if current is None:
                continue
            introduces_symbol = previous is not None and bool(current.free_symbols - previous.free_symbols)   # a definition such as D = b^2 - 4ac
            if previous is not None and not introduces_symbol and not (current.free_symbols < previous.free_symbols):
                if not equivalent(previous, current):
                    if good is not None and previous is not good and equivalent(good, current):
                        pass        # the previous line was the wrong one; this line is consistent with the last good equation
                    else:
                        problems.append((shot_id, f"{line!r} does not follow algebraically from the previous equation {previous_line!r}"))
                else:
                    good, good_line = current, line
            elif previous is None:
                good, good_line = current, line
            previous, previous_line = current, line
    return problems
