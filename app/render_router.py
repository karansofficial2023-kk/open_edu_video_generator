from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .schema import Segment


class RenderKind(str, Enum):
    APPROVED_ASSET = "approved_asset"
    GENERATED_IMAGE = "generated_image"
    LABELED_STILL = "labeled_still"
    COMPOSITED_IMAGE = "composited_image"
    DETERMINISTIC_GRAPHIC = "deterministic_graphic"
    GENERATED_MOTION = "generated_motion"


@dataclass(frozen=True)
class RenderDecision:
    kind: RenderKind
    reason: str


MOTION_TEMPLATES = {"video", "video_broll", "short_motion_clip"}
LABELED_TEMPLATES = {
    "labeled_image",
    "realistic_labeled_image",
    "realistic_background_with_labels",
    "diagram_overlay",
}
COMPOSITED_TEMPLATES = {"title_card", "split_screen"}
DETERMINISTIC_TEMPLATES = {
    "process",
    "process_steps",
    "comparison",
    "formula",
    "classification",
    "agents",
    "pros_cons",
    "pollination",
    "protandry",
    "protogyny",
    "wind",
    "water",
    "insect",
    "life_cycle",
}


def route_segment(segment: Segment) -> RenderDecision:
    """Select a renderer from explicit storyboard fields, never from lesson-specific words."""
    shot = segment.shot
    if shot and shot.asset_path:
        return RenderDecision(RenderKind.APPROVED_ASSET, "storyboard supplies an approved asset")
    if shot is None:
        return RenderDecision(RenderKind.GENERATED_IMAGE, "unstructured shot requires a relevant still")

    template = shot.template
    if shot.formula_lines or template == "formula":
        return RenderDecision(RenderKind.DETERMINISTIC_GRAPHIC, "exact formulas must use deterministic rendering")
    if template in MOTION_TEMPLATES:
        if shot.labels or shot.arrows or shot.formula_lines:
            return RenderDecision(
                RenderKind.LABELED_STILL,
                "precise overlays or formulas require a stable frame",
            )
        return RenderDecision(RenderKind.GENERATED_MOTION, "storyboard explicitly requests short motion")
    if template in LABELED_TEMPLATES:
        return RenderDecision(RenderKind.LABELED_STILL, "storyboard requests stable composited labels")
    if template in COMPOSITED_TEMPLATES:
        return RenderDecision(RenderKind.COMPOSITED_IMAGE, "generated background with deterministic typography")
    if template in DETERMINISTIC_TEMPLATES:
        return RenderDecision(RenderKind.DETERMINISTIC_GRAPHIC, "structured educational content")
    return RenderDecision(RenderKind.GENERATED_IMAGE, "storyboard requests a realistic still")
