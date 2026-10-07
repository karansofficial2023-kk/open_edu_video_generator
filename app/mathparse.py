"""Plain-text mathematics -> LaTeX (matplotlib mathtext subset): fractions, roots, powers, subscripts, Greek, relations.

Storyboards carry formulas as readable plain math (``x = (-b ± sqrt(b^2 - 4ac))/(2a)``); this module turns them into exact
typeset equations deterministically, so no model ever draws symbols.
"""
from __future__ import annotations

import re

GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "nu", "pi", "rho", "sigma", "tau", "phi",
         "omega", "Delta", "Gamma", "Theta", "Lambda", "Sigma", "Phi", "Omega", "eta", "kappa", "psi", "chi", "xi", "zeta"}
FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "min", "max", "det"}
RELATIONS = ["<=>", "<->", "=>", "->", ">=", "<=", "!=", "~=", "==", "=", "<", ">", "≈", "≤", "≥", "≠", "→", "⇌", "∝"]
LATEX_REL = {"<=>": r"\rightleftharpoons", "<->": r"\leftrightarrow", "=>": r"\Rightarrow", "->": r"\rightarrow", ">=": r"\geq",
             "<=": r"\leq", "!=": r"\neq", "~=": r"\approx", "==": "=", "≈": r"\approx", "≤": r"\leq", "≥": r"\geq",
             "≠": r"\neq", "→": r"\rightarrow", "⇌": r"\rightleftharpoons", "∝": r"\propto"}
SYMBOLS = {"±": r"\pm", "×": r"\times", "·": r"\cdot", "°": r"^{\circ}", "∞": r"\infty", "√": r"\sqrt",
           "Δ": r"\Delta", "Λ": r"\Lambda", "λ": r"\lambda", "μ": r"\mu", "ρ": r"\rho", "σ": r"\sigma",
           "π": r"\pi", "α": r"\alpha", "β": r"\beta", "θ": r"\theta", "ω": r"\omega", "Ω": r"\Omega",
           "ε": r"\epsilon", "κ": r"\kappa", "φ": r"\phi", "∑": r"\sum", "∫": r"\int"}
TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z]+)|(?P<sym>[±×·°∞√ΔΛλμρσπαβθωΩεκφ∑∫])|(?P<op>[-+*/^_(),|!]))")


class MathParseError(ValueError):
    pass


def split_relations(line: str) -> list[str]:
    """Split a line into [lhs, rel, rhs, ...] at top-level relation symbols (longest match first)."""
    parts, depth, buf, i = [], 0, "", 0
    while i < len(line):
        ch = line[i]
        depth += ch in "([{"
        depth -= ch in ")]}"
        matched = None
        if depth == 0:
            for rel in sorted(RELATIONS, key=len, reverse=True):
                if line.startswith(rel, i):
                    matched = rel
                    break
        if matched:
            parts += [buf, matched]
            buf, i = "", i + len(matched)
            continue
        buf += ch
        i += 1
    parts.append(buf)
    return parts


class Parser:
    def __init__(self, text: str):
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        text = text.strip()
        while pos < len(text):
            match = TOKEN.match(text, pos)
            if not match:
                raise MathParseError(f"unexpected character {text[pos:pos + 1]!r}")
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
            pos = match.end()
            while pos < len(text) and text[pos].isspace():
                pos += 1
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self):
        token = self.peek()
        self.i += 1
        return token

    def parse(self) -> str:
        if not self.tokens:
            return ""
        result = self.expr()
        if self.i != len(self.tokens):
            raise MathParseError(f"unexpected {self.peek()[1]!r}")
        return result

    def expr(self) -> str:
        out = self.term()
        while self.peek()[1] in {"+", "-"}:
            op = self.take()[1]
            out += f" {op} " + self.term()
        return out

    def term(self) -> str:
        out = self.power()
        while True:
            kind, value = self.peek()
            if value == "*":
                self.take()
                out += r" \cdot " + self.power()
            elif value == "/":
                self.take()
                out = rf"\frac{{{_strip(out)}}}{{{_strip(self.power())}}}"
            elif kind in {"num", "name", "sym"} or value == "(":
                out += " " + self.power()          # implicit multiplication
            else:
                return out

    def power(self) -> str:
        base = self.atom()
        while self.peek()[1] in {"^", "_"}:
            op = self.take()[1]
            exponent = self.script()
            base += f"{op}{{{exponent}}}"
        return base

    def script(self) -> str:
        kind, value = self.peek()
        if value == "(":
            self.take()
            inner = self.expr()
            self.expect(")")
            return inner
        if value == "-":
            self.take()
            return "-" + self.script()
        if value == "{":
            raise MathParseError("braces")
        if kind == "num":
            self.take()
            return value
        if kind == "name":
            self.take()
            return self.name_latex(value, in_script=True)
        raise MathParseError("bad exponent")

    def expect(self, symbol: str):
        if self.take()[1] != symbol:
            raise MathParseError(f"expected {symbol!r}")

    def atom(self) -> str:
        kind, value = self.take()
        if kind == "num":
            return value
        if kind == "sym":
            if value == "√":
                inner = self.atom() if self.peek()[1] != "(" else self.group()
                return rf"\sqrt{{{_strip(inner)}}}"
            return SYMBOLS[value]
        if kind == "name":
            if value == "sqrt":
                if self.peek()[1] != "(":
                    raise MathParseError("sqrt needs parentheses")
                return rf"\sqrt{{{_strip(self.group())}}}"
            if value in FUNCTIONS:
                arg = self.group() if self.peek()[1] == "(" else (self.power() if self.peek()[0] in {"num", "name"} else "")
                return rf"\{value}\left({_strip(arg)}\right)" if arg else rf"\{value}"
            return self.name_latex(value)
        if value == "(":
            self.i -= 1
            return r"\left(" + _strip(self.group()) + r"\right)"
        if value == "-":
            return "-" + self.power()
        if value == "|":
            inner = self.expr()
            self.expect("|")
            return rf"\left|{inner}\right|"
        raise MathParseError(f"unexpected {value!r}")

    def group(self) -> str:
        self.expect("(")
        inner = self.expr()
        while self.peek()[1] == ",":
            self.take()
            inner += ", " + self.expr()
        self.expect(")")
        return "(" + inner + ")"

    def name_latex(self, value: str, in_script: bool = False) -> str:
        if value in GREEK:
            return "\\" + value + (" " if not in_script else "")
        if len(value) >= 4 or (len(value) == 3 and value[0].isupper() and value[1:].islower()):
            return r"\mathrm{" + value + "}"         # a word such as Discriminant is text, not a product of variables
        letters = list(value)
        # single letter followed by digits is a subscript: R1 -> R_{1}
        if len(letters) == 1 and self.peek()[0] == "num" and not in_script and "." not in self.peek()[1] and len(self.peek()[1]) <= 2 \
                and self.tokens[self.i - 1][0] == "name":
            number = self.take()[1]
            return f"{value}_{{{number}}}"
        return " ".join(letters) if not in_script else value


def _strip(text: str) -> str:
    """Remove one redundant pair of outer parentheses (fraction parts)."""
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        depth = 0
        for index, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and index < len(text) - 1:
                return text
        return text[1:-1].strip()
    if text.startswith(r"\left(") and text.endswith(r"\right)"):
        inner = text[len(r"\left("):-len(r"\right)")]
        depth = 0
        for ch in inner:
            depth += ch == "("
            depth -= ch == ")"
            if depth < 0:
                return text
        return inner.strip()
    return text


def math_to_latex(line: str) -> str:
    """Convert one plain-math line (possibly with relations) to LaTeX; raises MathParseError when not parseable."""
    if "\\" in line:
        return line.strip()
    pieces = split_relations(line)
    out = []
    for index, piece in enumerate(pieces):
        if index % 2 == 1:
            out.append(LATEX_REL.get(piece, piece))
        else:
            out.append(Parser(piece).parse())
    return " ".join(part for part in out if part)
