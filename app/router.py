"""Renderer selection from explicit contract fields only (never topic keywords)."""
from __future__ import annotations

import re

from .schema import Segment, Shot

APPROVED_VISUAL_TYPES = {
    "title_card", "realistic_image", "realistic_labeled_image", "realistic_background_with_labels", "diagram_overlay",
    "process_steps", "split_screen", "formula", "graph", "circuit", "short_motion_clip",
}
METADATA_PATTERN = re.compile(
    r"(?i)\b(sharp\s+1920x1080|production asset|no generated text|renderer|placement|prompt|\.png|\.jpg|"
    r"asset[_ ]path|lorem|todo|tbd)\b")


def clean_title(text: str, limit: int = 8) -> str:
    text = METADATA_PATTERN.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip(" .:-|")
    words = text.split()
    return " ".join(words[:limit])


def is_cut_off_sentence_prefix(heading: str, narration: str) -> bool:
    """True when the heading is only the start of the narration sentence ("For weak electrolytes, the graph approaches an")."""
    norm = lambda t: re.sub(r"[^\w ]", "", re.sub(r"\s+", " ", t or "").lower()).strip()
    head, text = norm(heading), norm(narration)
    return bool(head) and len(head) < len(text) and text.startswith(head)


def route_segment(segment: Segment) -> Shot | None:
    """Return the deterministic Shot for a contract segment (None when the segment cannot be routed)."""
    kind = "realistic_labeled_image" if segment.visual_type == "realistic_background_with_labels" else segment.visual_type
    # A sentence prefix is not a title: an empty heading makes the renderer use the scene title.
    # a title card shows the whole lesson title (it wraps); other headings are short topic labels
    heading = ("" if kind != "title_card" and is_cut_off_sentence_prefix(segment.heading, segment.narration)
               else clean_title(segment.heading, 16 if kind == "title_card" else 8))
    if segment.formula_lines:
        return Shot(template="formula", heading=heading)
    if kind == "title_card":
        return Shot(template="title_card", heading=heading, asset_path=segment.asset_path or None)
    if kind == "circuit":
        return Shot(template="circuit", heading=heading)
    if kind == "graph":
        return Shot(template="graph", heading=heading)
    if kind == "split_screen":
        captions = [c.partition("|")[0].strip() for c in segment.columns[:3]]
        return Shot(template="split_screen", heading=heading, steps=captions[:3] if len(captions) >= 2 else [])
    if kind in {"process_steps", "diagram_overlay"}:
        steps = (segment.steps or segment.explain_steps)[:4]
        if len(steps) >= 2 and not _fragments_of(steps, segment.narration):
            return Shot(template="process", heading=heading, steps=steps, stage_fractions=_fractions(len(steps)))
        # A diagram without an explicit step list is a stable still with deterministic overlays.
        return Shot(template="photo", heading=heading, asset_path=segment.asset_path or None)
    if kind == "short_motion_clip":          # the picture is made like any still, then brought to life (motion.py) when enabled
        return Shot(template="video", heading=heading, asset_path=segment.asset_path or None)
    return Shot(template="photo", heading=heading, asset_path=segment.asset_path or None)


def _fragments_of(steps: list[str], narration: str) -> bool:
    """True when the 'steps' are just the narration cut into pieces (not a real ordered mechanism)."""
    norm = lambda t: re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()
    text = norm(narration)
    pieces = [norm(x) for x in steps]
    covered = sum(len(x) for x in pieces if x and x in text)
    return covered >= 0.6 * max(1, len(text))


def _fractions(count: int) -> list[float]:
    return [round(i / count, 3) for i in range(count)]


def route_reason(segment: Segment) -> str:
    if segment.formula_lines:
        return "formula_lines present -> deterministic formula renderer"
    return f"visual_type={segment.visual_type or 'unspecified'} -> {segment.shot.template if segment.shot else 'unrouted'}"
