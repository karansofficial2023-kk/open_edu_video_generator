"""Which school subject a lesson belongs to, read from the storyboard itself, and what that means for its visuals.

A teacher treats subjects differently: in mathematics the worked derivation IS the picture (a clean notebook page, no photographs);
in physics and chemistry a photograph earns its place only when it shows the apparatus or phenomenon being explained, otherwise a
clear card is better; in biology the real organism or structure is the best explanation, so photographs (with labels) come first.

The subject comes from the storyboard's own `subject` field when Transcribe filled it, otherwise from the lesson's words and formulas.
Nothing here depends on one particular topic.
"""
from __future__ import annotations

import re

from .schema import Storyboard

MATHS, PHYSICS, CHEMISTRY, BIOLOGY, GENERAL = "maths", "physics", "chemistry", "biology", "general"

_NAMES = {
    MATHS: ("math", "maths", "mathematics", "algebra", "calculus", "geometry", "trigonometry", "statistics", "arithmetic"),
    PHYSICS: ("physics",),
    CHEMISTRY: ("chemistry", "chemical"),
    BIOLOGY: ("biology", "botany", "zoology", "life science", "physiology", "anatomy", "pharmacology", "medicine", "health"),
}

# Vocabulary typical of each subject, used only when the storyboard does not name its subject.
_WORDS = {
    MATHS: r"function|derivative|differentiat\w*|integra\w*|equation|maxim\w*|minim\w*|matrix|matrices|polynomial|theorem|"
           r"probability|logarithm|limit|continuity|continuous|graph of|slope|tangent|vector|quadratic|algebra\w*|proof|real number",
    PHYSICS: r"resist\w*|current|voltage|circuit|force|velocity|acceleration|momentum|energy|power|wave|frequency|magnet\w*|"
             r"electric\w*|charge|emf|cell|lens|mirror|refraction|gravity|pressure|wire|ohm",
    CHEMISTRY: r"electrolyt\w*|molar|mole|ion|ions|acid|base|salt|reaction|compound|element|oxidation|reduction|redox|"
               r"conductivity|solution|bond|atom\w*|molecul\w*|electrod\w*|catalyst|ph\b",
    BIOLOGY: r"cell|cells|plant|flower|pollin\w*|organism|species|tissue|organ|gene|dna|protein|enzyme|disease|blood|"
             r"bacteri\w*|virus|seed|fruit|leaf|root|animal|insect|human body|hormone|cholesterol|drug",
}


def subject_of(board: Storyboard) -> str:
    named = (getattr(board, "subject", "") or "").strip().lower()
    for subject, names in _NAMES.items():
        if any(name in named for name in names):
            return subject
    segments = [seg for _scene, seg in board.all_segments()]
    text = " ".join([board.title or ""] + [seg.narration for seg in segments]).lower()
    scores = {subject: len(re.findall(rf"\b(?:{pattern})\b", text)) for subject, pattern in _WORDS.items()}
    formula_share = sum(1 for seg in segments if seg.formula_lines) / max(1, len(segments))
    scores[MATHS] += round(formula_share * 20) if not scores[CHEMISTRY] and not scores[PHYSICS] else 0
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else GENERAL


def photos_first(subject: str) -> bool:
    """Biology (and anything unknown): a real picture is the default explanation."""
    return subject in {BIOLOGY, GENERAL}


def plain_cards(subject: str) -> bool:
    """Mathematics: explanation cards sit on the plain notebook page, never on a blurred photograph."""
    return subject == MATHS
