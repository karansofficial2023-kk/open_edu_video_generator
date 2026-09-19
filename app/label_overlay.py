from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .visuals import _font, _pixel_wrap


LABEL_TEMPLATES = {
    "labeled_image",
    "diagram_overlay",
    "realistic_labeled_image",
    "realistic_background_with_labels",
    "process_steps",
}
MOTION_TEMPLATES = {"video", "video_broll", "short_motion_clip"}


@dataclass(frozen=True)
class LabelPlan:
    text: str
    target: tuple[int, int] | None
    box: tuple[int, int, int, int] | None
    confidence: float
    reason: str = ""


def is_label_template(template: str) -> bool:
    return template in LABEL_TEMPLATES


def allows_precise_arrows(template: str) -> bool:
    return template not in MOTION_TEMPLATES


def arrow_label_visibility(index: int, label_count: int, t: float, duration: float, mode: str) -> tuple[float, float]:
    if label_count <= 0:
        return 0.0, 0.0
    if mode != "arrow_draw_then_label_fade":
        visible = max(1, min(label_count, int((t / max(duration, 0.1)) * (label_count + 0.9)) + 1))
        return (1.0, 1.0) if index < visible else (0.0, 0.0)
    slot = max(0.45, duration / max(label_count, 1))
    local = t - index * slot
    if local < 0:
        return 0.0, 0.0
    arrow = max(0.0, min(1.0, local / max(0.1, slot * 0.48)))
    label = max(0.0, min(1.0, (local - slot * 0.48) / max(0.1, slot * 0.28)))
    return arrow, label


def build_label_plans(
    shot: Any,
    image_size: tuple[int, int],
    title_safe: tuple[int, int, int, int] | None = None,
    subtitle_safe: tuple[int, int, int, int] | None = None,
    logo_safe: tuple[int, int, int, int] | None = None,
) -> tuple[list[LabelPlan], list[dict[str, Any]]]:
    width, height = image_size
    title_safe = title_safe or (0, 0, width, round(height * 0.16))
    subtitle_safe = subtitle_safe or (0, round(height * 0.84), width, height)
    logo_safe = logo_safe or (round(width * 0.82), 0, width, round(height * 0.15))
    forbidden = [title_safe, subtitle_safe, logo_safe]
    labels = list(getattr(shot, "labels", []) or [])
    plans: list[LabelPlan] = []
    review: list[dict[str, Any]] = []
    occupied: list[tuple[int, int, int, int]] = []

    for index, raw in enumerate(labels):
        text = _label_text(raw, index)
        target, confidence, reason = resolve_label_target(text, raw, width, height)
        if target is None or confidence < 0.7:
            plans.append(LabelPlan(text=text, target=None, box=None, confidence=confidence, reason=reason))
            review.append({"label": text, "reason": reason or "low-confidence target", "confidence": confidence})
            continue
        box = place_label_box(text, target, width, height, forbidden + occupied, raw, index)
        if box is None:
            plans.append(LabelPlan(text=text, target=target, box=None, confidence=confidence, reason="no clear label box position"))
            review.append({"label": text, "reason": "no clear label box position", "confidence": confidence})
            continue
        occupied.append(box)
        plans.append(LabelPlan(text=text, target=target, box=box, confidence=confidence, reason=reason))
    return plans, review


def resolve_label_target(text: str, spec: dict[str, Any], width: int, height: int) -> tuple[tuple[int, int] | None, float, str]:
    explicit = _explicit_xy(spec, width, height)
    if explicit:
        return explicit, 1.0, "manual coordinate"
    placement = " ".join(str(spec.get(key, "")) for key in ("placement", "target", "description", "position", "anchor")).lower()
    combined = f"{text} {placement}".lower()
    if not placement or placement.strip().isdigit():
        return None, 0.0, "missing placement or coordinate override"
    x_hint = _horizontal_hint(combined, width)
    y_hint = _vertical_hint(combined, height)
    semantic = _semantic_target(text.lower(), width, height)
    if semantic is None:
        return None, 0.35, "unknown semantic target"
    sx, sy = semantic
    x = x_hint if x_hint is not None else sx
    y = y_hint if y_hint is not None else sy
    confidence = 0.78 if (x_hint is not None or y_hint is not None) else 0.58
    return (int(x), int(y)), confidence, "semantic target with placement hint"


def place_label_box(
    text: str,
    target: tuple[int, int],
    width: int,
    height: int,
    forbidden: list[tuple[int, int, int, int]],
    spec: dict[str, Any] | None = None,
    index: int = 0,
) -> tuple[int, int, int, int] | None:
    spec = spec or {}
    bw = min(max(210, len(text) * 15 + 48), 360)
    bh = 62
    margin = 34
    tx, ty = target
    side = str(spec.get("box_side") or spec.get("side") or spec.get("placement") or "").lower()
    candidates = []
    if "left" in side:
        candidates.append((margin, ty - bh // 2, margin + bw, ty + bh // 2))
    if "right" in side:
        candidates.append((width - margin - bw, ty - bh // 2, width - margin, ty + bh // 2))
    candidates.extend([
        (margin, 150 + (index % 5) * 86, margin + bw, 150 + (index % 5) * 86 + bh),
        (width - margin - bw, 150 + (index % 5) * 86, width - margin, 150 + (index % 5) * 86 + bh),
        (margin, height - 250 - (index % 3) * 78, margin + bw, height - 188 - (index % 3) * 78),
        (width - margin - bw, height - 250 - (index % 3) * 78, width - margin, height - 188 - (index % 3) * 78),
        (width // 2 - bw // 2, 145 + (index % 2) * 84, width // 2 + bw // 2, 145 + (index % 2) * 84 + bh),
    ])
    for box in candidates:
        box = _clamp_box(box, width, height, margin)
        if not _intersects_any(box, forbidden) and not _contains_point(box, target):
            return box
    return None


def draw_label_overlay(
    image: Image.Image,
    plans: list[LabelPlan],
    t: float,
    duration: float,
    motion_type: str,
    label_style: str = "",
    accent: str = "#26734D",
    gold: str = "#F4C542",
) -> Image.Image:
    frame = image.convert("RGBA")
    arr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
    label_count = len(plans)
    for index, plan in enumerate(plans):
        if plan.target is None or plan.box is None:
            continue
        arrow_progress, label_alpha = arrow_label_visibility(index, label_count, t, duration, motion_type)
        if arrow_progress <= 0 and label_alpha <= 0:
            continue
        color_hex = gold if index % 2 else accent
        color = _bgr(color_hex)
        start = _box_anchor(plan.box, plan.target)
        end = (
            round(start[0] + (plan.target[0] - start[0]) * arrow_progress),
            round(start[1] + (plan.target[1] - start[1]) * arrow_progress),
        )
        if arrow_progress > 0:
            cv2.arrowedLine(arr, start, end, color, 4, tipLength=0.08)
            cv2.circle(arr, plan.target, 7, color, -1)
        if label_alpha > 0:
            arr = _draw_label_box(arr, plan.box, plan.text, color_hex, label_alpha, label_style)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)).convert("RGB")


def write_review_log(path: str | Path, review: list[dict[str, Any]]) -> None:
    if not review:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["label,confidence,reason"]
    for item in review:
        label = str(item.get("label", "")).replace(",", " ")
        reason = str(item.get("reason", "")).replace(",", " ")
        lines.append(f"{label},{item.get('confidence', 0)},{reason}")
    out.write_text("\n".join(lines), encoding="utf-8")


def _label_text(spec: dict[str, Any], index: int) -> str:
    text = str(spec.get("text") or spec.get("name") or spec.get("label") or f"Label {index + 1}").strip()
    return re.sub(r"\s+", " ", text)[:80]


def _explicit_xy(spec: dict[str, Any], width: int, height: int) -> tuple[int, int] | None:
    for key in ("xy", "target_xy", "target", "coordinate", "coords"):
        value = spec.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return _norm_to_px(float(value[0]), float(value[1]), width, height)
        if isinstance(value, str):
            nums = re.findall(r"(?<!\d)(?:0?\.\d+|1(?:\.0+)?|\d{2,4})(?!\d)", value)
            if len(nums) >= 2:
                return _norm_to_px(float(nums[0]), float(nums[1]), width, height)
    return None


def _norm_to_px(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    if 0 <= x <= 1 and 0 <= y <= 1:
        return round(x * width), round(y * height)
    return round(x), round(y)


def _semantic_target(text: str, width: int, height: int) -> tuple[int, int] | None:
    table = [
        (("anther", "pollen source"), (0.37, 0.44)),
        (("stigma", "receptive"), (0.55, 0.38)),
        (("pollen grains", "pollen"), (0.43, 0.45)),
        (("pollen transfer path", "transfer path", "curved arrow"), (0.50, 0.44)),
        (("pollinator", "bee", "insect"), (0.62, 0.42)),
        (("closed flower", "cleistogamy"), (0.50, 0.48)),
        (("wood sorrel",), (0.50, 0.52)),
        (("style", "long style", "short style"), (0.52, 0.48)),
        (("nectar guide", "nectar"), (0.48, 0.50)),
        (("ray floret", "ray florets"), (0.30, 0.46)),
        (("disc floret", "disc florets", "central disk"), (0.50, 0.46)),
        (("flower 1",), (0.32, 0.50)),
        (("flower 2",), (0.68, 0.50)),
    ]
    for keys, point in table:
        if any(key in text for key in keys):
            return round(point[0] * width), round(point[1] * height)
    return None


def _horizontal_hint(text: str, width: int) -> int | None:
    if "left" in text:
        return round(width * 0.33)
    if "right" in text:
        return round(width * 0.67)
    if "center" in text or "middle" in text:
        return round(width * 0.5)
    return None


def _vertical_hint(text: str, height: int) -> int | None:
    if "top" in text or "upper" in text:
        return round(height * 0.35)
    if "bottom" in text or "lower" in text:
        return round(height * 0.66)
    if "center" in text or "middle" in text:
        return round(height * 0.5)
    return None


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(margin, min(width - margin - bw, x1))
    y1 = max(margin, min(height - margin - bh, y1))
    return x1, y1, x1 + bw, y1 + bh


def _intersects_any(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    return any(_intersects(box, other) for other in others)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _contains_point(box: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _box_anchor(box: tuple[int, int, int, int], target: tuple[int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    x = x2 if target[0] > x2 else x1 if target[0] < x1 else (x1 + x2) // 2
    y = y2 if target[1] > y2 else y1 if target[1] < y1 else (y1 + y2) // 2
    return x, y


def _bgr(hex_color: str) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return b, g, r, 255


def _draw_label_box(arr: np.ndarray, box: tuple[int, int, int, int], text: str, color: str, alpha: float, style: str) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = box
    outline = color
    fill_alpha = round(242 * alpha)
    draw.rounded_rectangle(box, radius=10, fill=(255, 255, 255, fill_alpha), outline=outline, width=3)
    font = _font(25, True)
    lines = _pixel_wrap(draw, text, font, max(10, x2 - x1 - 24))
    line_height = sum(font.getmetrics()) + 4
    y = y1 + max(6, ((y2 - y1) - len(lines) * line_height) // 2)
    for line in lines:
        draw.text((x1 + 14, y), line, font=font, fill=(6, 21, 61, round(255 * alpha)))
        y += line_height
    image = Image.alpha_composite(image, overlay)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
