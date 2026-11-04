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

MATHS, PHYSICS, CHEMISTRY, BIOLOGY, ENGINEERING, GENERAL = "maths", "physics", "chemistry", "biology", "engineering", "general"

_NAMES = {
    MATHS: ("math", "maths", "mathematics", "algebra", "calculus", "geometry", "trigonometry", "statistics", "arithmetic"),
    PHYSICS: ("physics", "thermodynamics"),
    CHEMISTRY: ("chemistry", "chemical"),
    BIOLOGY: ("biology", "botany", "zoology", "life science", "physiology", "anatomy", "pharmacology", "medicine", "health"),
    ENGINEERING: ("engineering", "thermodynamics", "mechanical", "combustion", "automotive", "aeronautical", "thermal", "propulsion"),
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
    ENGINEERING: r"combustion|equivalence ratio|fuel-?air|air-?fuel|stoichiometric|piston|cylinder|engine|spark plug|flame|"
                r"heat engine|thermal|turbine|nozzle|intake|exhaust|power stroke|manifold",
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


def board_style(subject: str) -> bool:
    """All subjects: explanation screens use the dark teacher's board that builds up (board.py) - the physics/chemistry style.
    Never the old white concept cards."""
    return True


def plain_cards(subject: str) -> bool:
    """All subjects use the dark teacher's board style now."""
    return False


def is_primary_grade(target: Storyboard | str | None) -> bool:
    """Returns True ONLY for primary school (LKG/UKG through 5th standard).
    Higher grades (6th-12th, college, engineering, adult) MUST return False.
    """
    if target is None:
        return False

    if isinstance(target, Storyboard):
        # 1. Any segment explicitly calling for primary karaoke?
        for _scene, seg in target.all_segments():
            if getattr(seg, "subtitle_style", "") == "primary_karaoke":
                return True
        fields = [
            getattr(target, "grade_level", ""),
            getattr(target, "academic_level", ""),
            getattr(target, "subject", ""),
            getattr(target, "title", ""),
        ]
        combined = " ".join(f for f in fields if f).strip().lower()
    else:
        combined = str(target).strip().lower()

    if not combined:
        return False

    # Check negative / higher grade indicators first
    higher_patterns = [
        r"\b(?:class|std|grade|standard)\s*(?:[6-9]|1[0-2]|[6-9]th|1[0-2]th)\b",
        r"\b(?:6th|7th|8th|9th|10th|11th|12th)\s*(?:std|grade|class|standard)\b",
        r"\b(?:middle\s*school|high\s*school|secondary|higher\s*secondary|college|university|undergraduate|postgraduate|phd|b\.?tech|m\.?tech|engineering)\b",
        r"\b(?:thermodynamics|combustion|electroplating|calculus|quantum|relativity)\b",
    ]
    for hp in higher_patterns:
        if re.search(hp, combined):
            return False

    # Check positive primary indicators (LKG through 5th standard)
    primary_patterns = [
        r"\b(?:lkg|ukg|nursery|kindergarten|kg|pre-?k|preschool)\b",
        r"\b(?:class|std|grade|standard)\s*(?:[1-5]|1st|2nd|3rd|4th|5th)\b",
        r"\b(?:1st|2nd|3rd|4th|5th)\s*(?:std|grade|class|standard)\b",
        r"\b(?:primary|elementary|lower\s*primary)\b",
        r"\b(?:evs|environmental\s*studies)\b",
    ]
    for pp in primary_patterns:
        if re.search(pp, combined):
            return True

    return False

