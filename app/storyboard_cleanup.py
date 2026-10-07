from __future__ import annotations

import re

from .schema import Shot, Storyboard
from .speech_text import spoken_math

_CONTRAST = re.compile(r"\b(versus|vs\.?|whereas|compared (?:to|with)|in contrast|on the other hand|unlike)\b", re.I)


def normalize_storyboard(storyboard: Storyboard, config=None) -> None:
    """Make imported storyboards renderable. Topic-specific fixes come from config.text_replacements."""
    replacements = dict(getattr(config, "text_replacements", {}) or {})
    english = ((getattr(storyboard, "language", None) or {}).get("bcp47", "en") or "en").split("-")[0].lower() == "en"
    clean = lambda text: spoken_math(_clean_text(text, replacements), english)       # no LaTeX in anything that is spoken or shown as prose
    storyboard.title = clean(storyboard.title)
    storyboard.source = clean(storyboard.source)
    for scene, segment in storyboard.all_segments():
        scene.title = clean(scene.title)
        scene.narration = clean(scene.narration)
        segment.narration = clean(segment.narration)
        segment.subtitle = clean(segment.subtitle)           # shown on screen instead of the narration whenever it differs from it
        segment.formula_lines = [line for line in segment.formula_lines if re.search(r"\w|\\", line)]    # a lone "=" is a fragment; "Λ₀ = Λ⁺ + Λ⁻" is not
        segment.visual = clean(segment.visual)
        segment.image_prompt = clean(segment.image_prompt)
        segment.keywords = [clean(keyword).lower() for keyword in segment.keywords]
        if not segment.source_references:
            segment.source_references = ["Imported storyboard; verify against curriculum before publication."]
        if segment.shot is None:
            segment.shot = _infer_shot(scene.title, segment.narration)
        else:
            segment.shot.heading = clean(segment.shot.heading)
            segment.shot.learning_objective = clean(segment.shot.learning_objective)
            segment.shot.steps = [clean(step) for step in segment.shot.steps]
            segment.shot.cues = [clean(cue) for cue in segment.shot.cues]
        if segment.shot is not None and segment.shot.template == "formula" and not segment.formula_lines:
            segment.shot = _infer_shot(segment.shot.heading or scene.title, segment.narration)     # never an empty equation card
    merge_filler_shots(storyboard)
    merge_repeated_formulas(storyboard)


def _clean_text(text: str, replacements: dict[str, str]) -> str:
    if not text:
        return text
    cleaned = text.replace("�", "-")
    for old, new in replacements.items():
        cleaned = re.sub(rf"\b{re.escape(old)}\b", _keep_case(new), cleaned, flags=re.I)
    return cleaned


def _keep_case(replacement: str):
    def apply(match: re.Match) -> str:
        return replacement.capitalize() if match.group(0)[:1].isupper() else replacement
    return apply


def _infer_shot(scene_title: str, narration: str) -> Shot:
    """Subject-independent fallback: a two-stage comparison or process built from the narration itself."""
    heading = scene_title.strip() or _heading_from_text(narration)
    template = "comparison" if _CONTRAST.search(narration) else "process"
    return Shot(
        template=template,
        heading=heading,
        learning_objective="Explain the key idea in this segment.",
        steps=_two_steps(narration) if template == "comparison" else _card_steps(narration, heading),     # the two sides / complete statements
    )


def _heading_from_text(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    return " ".join(words[:4]) or "Key Idea"


def _two_steps(text: str) -> list[str]:
    parts = _CONTRAST.split(text)
    if len(parts) >= 3:  # split() keeps the captured connector at odd positions
        clauses = [parts[0].strip(" .,;"), parts[2].strip(" .,;")]
    else:
        clauses = [part.strip(" .") for part in re.split(r",|;|\band\b|\bbut\b|\bthen\b", text) if part.strip()]
    if len(clauses) >= 2 and all(clauses[:2]):
        return [_short_step(clauses[0]), _short_step(clauses[1])]
    return [_short_step(text), "Connect it to the main concept"]


def _short_step(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", text)
    return " ".join(words[:8]) or "Observe the idea"


def _card_steps(narration: str, heading: str = "") -> list[str]:
    from .assets import complete_cards          # complete statements, never a sentence cut in half
    return complete_cards(narration, heading)[:4]


def _content_words(text: str) -> int:
    return len([w for w in re.findall(r"[^\W\d_][\w'-]*", text or "") if len(w) > 2])


def merge_filler_shots(storyboard: Storyboard) -> int:
    """A teacher does not change the board for "Clear?", "Fascinating, isn't it?" or "Keep learning": such lines are spoken over the
    picture that is already showing. Sign-offs, encouragements, rhetorical checks and very short remarks (two content words or
    fewer) are appended to the previous shot's narration instead of getting a screen of their own. Shots that carry their own
    content (formulas, labels, steps from the storyboard) are never merged away. Returns the number of shots merged."""
    from .assets import is_filler_narration
    merged = 0
    previous = None
    for scene in storyboard.scenes:
        kept = []
        for segment in scene.segments:
            own_content = bool(segment.formula_lines or segment.labels or segment.steps or segment.columns)
            filler = is_filler_narration(segment.narration) or _content_words(segment.narration) <= 2
            if previous is not None and filler and not own_content and segment.narration.strip():
                previous.narration = f"{previous.narration.rstrip()} {segment.narration.strip()}".strip()
                if previous.subtitle or segment.subtitle:
                    previous.subtitle = f"{(previous.subtitle or '').rstrip()} {(segment.subtitle or segment.narration).strip()}".strip()
                previous.duration_hint = (previous.duration_hint or 0) + (segment.duration_hint or 0)
                if previous.shot is not None and previous.shot.template in {"process", "comparison"} and not previous.steps:
                    previous.shot.steps = _card_steps(previous.narration, previous.shot.heading)       # cards follow the longer narration
                    previous.shot.stage_fractions, previous.shot.cues = [], []
                merged += 1
                continue
            kept.append(segment)
            previous = segment
        scene.segments = kept
    # a scene made only of remarks (a sign-off scene) is now spoken over the previous scene's last picture
    storyboard.scenes = [scene for scene in storyboard.scenes if scene.segments]
    return merged


def _formula_key(lines: list[str]) -> str:
    """The same equation written two ways ("x + 1/x" and "x + \\frac{1}{x}") is one equation."""
    def norm(line: str) -> str:
        line = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", line)
        line = re.sub(r"\\(left|right|,|;|!|quad)", "", line).replace("\\cdot", "*").replace("\\times", "*")
        line = re.sub(r"[\s{}$]", "", line)
        return re.sub(r"\((\w+)\)", r"\1", line)          # (1)/(x) -> 1/x
    return "|".join(norm(line) for line in lines if line.strip())


def merge_repeated_formulas(storyboard: Storyboard) -> int:
    """A teacher writes an equation on the board once and keeps talking about it; the same equation is not rewritten on a fresh board
    every sentence. A formula shot that repeats the equation(s) of the formula shot before it is merged into that shot,
    together with any short explanation cards in between (cards that carry no labels, photographs or equation of their own). The
    narration is kept in order; the explanation steps are kept too, so the note under the equation changes as the talk goes on.
    A photograph, a labelled picture or a different equation ends the run. Returns the number of shots merged away."""
    from .subject import MATHS, subject_of
    # In mathematics a planned photograph between two writings of the same equation is only a spoken step ("Let the given number be
    # x"): it becomes a card anyway (subject.py), so it does not interrupt the board. In other subjects a photograph ends the run.
    maths = subject_of(storyboard) == MATHS
    merged = 0
    # the board is not wiped at a scene boundary: the run continues into the next scene (an equation that ends one scene and opens
    # the next is written once); everything else that stands between two writings still ends the run
    kept_in: dict[int, list] = {id(scene): [] for scene in storyboard.scenes}
    order: list = []                     # (scene, segment) kept so far, in lesson order
    pending: list = []                   # (scene, segment) plain cards seen after the last formula shot
    blocked = True
    for scene in storyboard.scenes:
        for segment in scene.segments:
            shot = segment.shot
            anchor = next((seg for _sc, seg in reversed(order) if seg.shot and seg.shot.template == "formula"), None)
            is_formula = bool(shot and shot.template == "formula" and segment.formula_lines)
            if is_formula and anchor is not None and _formula_key(segment.formula_lines) == _formula_key(anchor.formula_lines)                     and not blocked:
                for _sc, between in pending + [(scene, segment)]:
                    anchor.narration = f"{anchor.narration.rstrip()} {between.narration.strip()}".strip()
                    if anchor.subtitle or between.subtitle:
                        anchor.subtitle = f"{(anchor.subtitle or '').rstrip()} {(between.subtitle or between.narration).strip()}".strip()
                    anchor.duration_hint = (anchor.duration_hint or 0) + (between.duration_hint or 0)
                    for step in between.explain_steps:
                        if step not in anchor.explain_steps:
                            anchor.explain_steps.append(step)
                for sc, between in pending:
                    kept_in[id(sc)].remove(between)
                    order.remove((sc, between))
                merged += len(pending) + 1
                pending = []
                continue
            card_like = {"process", "comparison"} | ({"photo"} if maths else set())
            plain_card = bool(shot and shot.template in card_like and not segment.labels and not segment.formula_lines
                              and not (shot.asset_path or segment.asset_path))
            if is_formula:
                pending, blocked = [], False
            elif plain_card:
                pending.append((scene, segment))
            else:
                pending, blocked = [], True
            kept_in[id(scene)].append(segment)
            order.append((scene, segment))
    for scene in storyboard.scenes:
        scene.segments = kept_in[id(scene)]
    storyboard.scenes = [scene for scene in storyboard.scenes if scene.segments]
    return merged
