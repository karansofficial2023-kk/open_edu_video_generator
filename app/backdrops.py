"""A picture behind concept cards.

When no photograph passes review for a shot, the narration is shown as cards. On a plain page that is the weakest screen of the lesson,
so the card screen gets a soft wash of a real, already accepted photograph of the SAME lesson (nearest in the same scene, else nearest
anywhere), blurred and blended into the page colour. It is mood and subject, not evidence: the cards carry the content, and the text
stays dark on a light page.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from .schema import Storyboard

# Shots whose contract asked for a picture. A genuine process / formula / graph screen is not a fallback and keeps its plain page.
PICTURE_TYPES = {"realistic_image", "realistic_labeled_image", "realistic_background_with_labels", "short_motion_clip"}


def choose(board: Storyboard) -> dict[int, str]:
    """{id(segment): photo path} for every card that stands in for a photograph."""
    items = board.all_segments()
    photos = []                                   # (position, scene number, path)
    for position, (scene, seg) in enumerate(items):
        shot = seg.shot
        if shot and shot.template in {"photo", "video"} and shot.asset_path and Path(shot.asset_path).is_file():
            photos.append((position, scene.scene_number, shot.asset_path))
    chosen: dict[int, str] = {}
    if not photos:
        return chosen
    for position, (scene, seg) in enumerate(items):
        shot = seg.shot
        if not (shot and shot.template == "process" and seg.visual_type in PICTURE_TYPES):
            continue
        same_scene = [p for p in photos if p[1] == scene.scene_number]
        pool = same_scene or photos
        # nearest in the lesson's order; on a tie the earlier picture (the one the viewer has just seen)
        best = min(pool, key=lambda p: (abs(p[0] - position), 0 if p[0] < position else 1))
        chosen[id(seg)] = best[2]
    return chosen


def soft(path: str | Path, size: tuple[int, int], page_rgb: tuple[int, int, int], strength: float = 0.70) -> Image.Image:
    """Cover-cropped, heavily blurred photo blended into the page colour (strength = share of page colour)."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = max(size[0] / image.width, size[1] / image.height)
    image = image.resize((max(size[0], round(image.width * scale)), max(size[1], round(image.height * scale))), Image.Resampling.LANCZOS)
    left, top = (image.width - size[0]) // 2, (image.height - size[1]) // 2
    image = image.crop((left, top, left + size[0], top + size[1]))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(8, round(size[1] / 38))))
    return Image.blend(blurred, Image.new("RGB", size, page_rgb), strength)
