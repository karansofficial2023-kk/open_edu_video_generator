"""Precision labels: verified target anchors, collision-free layout, leader-line-then-label animation.

Rules (production spec section 7): image generation never draws text; Qwen Vision proposes a target box,
an independent second request verifies the crop, low-confidence targets are omitted and reported (never an
approximate arrow); labels sit in clear margins and never cover the title, subtitle or logo zones.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from . import vision
from .config import AppConfig
from .contract import PENDING
from .schema import Segment
from .typography import draw_lines, font, hex_rgb, is_complex_script, parse_style, text_width, wrap_text


@dataclass
class Anchor:
    label: str
    description: str
    box: tuple[float, float, float, float]     # normalized 0..1 (x1, y1, x2, y2)
    confidence: float

    @property
    def point(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


@dataclass
class PlacedLabel:
    text: str
    box: tuple[int, int, int, int]
    anchor_px: tuple[int, int]
    edge_px: tuple[int, int]
    lines: list[str] = field(default_factory=list)


def label_targets(segment: Segment) -> list[tuple[str, str]]:
    """(label, target description) pairs, one-to-one with the labels field."""
    pairs = []
    for index, label in enumerate(segment.labels):
        placement = segment.label_placement[index] if index < len(segment.label_placement) else ""
        description = re.sub(rf"^\s*{re.escape(label)}\s*[-:|]\s*", "", placement, flags=re.I).strip()
        description = re.sub(rf"\s*\|?\s*{PENDING}\s*$", "", description).strip(" |")
        if description.upper() == PENDING or not description:
            description = label
        pairs.append((label, description))
    return pairs


def _fixed_coordinates(placement: str) -> tuple[float, float, float, float] | None:
    match = re.search(r"x\s*=\s*([0-9.]+)\s*[, ]\s*y\s*=\s*([0-9.]+)(?:\s*[, ]\s*w\s*=\s*([0-9.]+)\s*[, ]\s*h\s*=\s*([0-9.]+))?", placement or "", re.I)
    if not match:
        return None
    x, y = float(match.group(1)), float(match.group(2))
    w, h = float(match.group(3) or 0.08), float(match.group(4) or 0.08)
    return (max(0, x - w / 2), max(0, y - h / 2), min(1, x + w / 2), min(1, y + h / 2))


def locate(image_path: Path, segment: Segment, config: AppConfig, cache_dir: Path) -> tuple[list[Anchor], list[dict]]:
    """Return verified anchors and a list of omitted labels with reasons."""
    targets = label_targets(segment)
    if not targets:
        return [], []
    digest = hashlib.sha1(Path(image_path).read_bytes()).hexdigest()[:12]
    cache = cache_dir / f"anchors_{segment.shot_id.replace('.', '_')}_{digest}.json"
    key = json.dumps(targets)
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("key") == key:
            return [Anchor(**{**a, "box": tuple(a["box"])}) for a in stored["anchors"]], stored["omitted"]
    anchors: list[Anchor] = []
    omitted: list[dict] = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    for label, description in targets:
        try:
            anchor = _locate_one(image, label, description, config)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:          # also malformed model output
            omitted.append({"label": label, "reason": f"vision unavailable: {exc}"})
            continue
        if isinstance(anchor, str):
            omitted.append({"label": label, "reason": anchor})
        else:
            anchors.append(anchor)
    vision.unload(config)
    if any(str(o["reason"]).startswith("vision unavailable") for o in omitted):
        return anchors, omitted         # a failure of the review model is not a verdict on the picture: do not cache it, retry next run
    cache.write_text(json.dumps({"key": key, "anchors": [{**a.__dict__, "box": list(a.box)} for a in anchors], "omitted": omitted},
                                indent=2), encoding="utf-8")
    return anchors, omitted


def _locate_one(image: Image.Image, label: str, description: str, config: AppConfig) -> Anchor | str:
    threshold = config.production.label_min_confidence
    proposal = vision.ask(image, (
        "Locate ONE visible physical structure in this image.\n"
        f"Target: {label} - {description}\n"
        "Return JSON: {\"visible\": bool, \"box\": [x1, y1, x2, y2], \"confidence\": 0..1, \"evidence\": \"what you see\"} "
        "where box is a tight bounding box in coordinates normalised to 0-1000 on both axes. "
        "If the structure is not clearly visible, set visible=false. Do not guess."),
        config, purpose=f"locate:{label}")
    if not _flag(proposal.get("visible")):
        return "target not visible in image"
    box = _norm_box(proposal.get("box"))
    confidence = _num(proposal.get("confidence"))
    if box is None:
        return "invalid box returned"
    if confidence < threshold:
        return f"low confidence ({confidence:.2f})"
    # Independent verification: the second request sees only the crop, not the first confidence or box.
    w, h = image.size
    pad_x, pad_y = (box[2] - box[0]) * 0.35 + 0.02, (box[3] - box[1]) * 0.35 + 0.02
    crop = image.crop((round(max(0, box[0] - pad_x) * w), round(max(0, box[1] - pad_y) * h),
                       round(min(1, box[2] + pad_x) * w), round(min(1, box[3] + pad_y) * h)))
    verdict = vision.ask(crop, (
        f"Does this crop clearly show: {label} - {description}?\n"
        "Return JSON: {\"shows_target\": bool, \"confidence\": 0..1, \"what_is_shown\": \"short phrase\"}."),
        config, purpose=f"verify:{label}")
    if not _flag(verdict.get("shows_target")) or _num(verdict.get("confidence")) < threshold:
        return f"verification rejected ({verdict.get('what_is_shown', 'unrelated region')})"
    return Anchor(label=label, description=description, box=box, confidence=min(confidence, _num(verdict.get("confidence"))))


def _flag(value) -> bool:
    """A model may answer a yes/no field with text; bool("false") is True, so read the words."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_box(raw) -> tuple[float, float, float, float] | None:
    try:
        x1, y1, x2, y2 = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    scale = 1000.0 if max(x1, y1, x2, y2) > 1.5 else 1.0
    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
    if x2 <= x1 or y2 <= y1 or x2 - x1 > 0.9 or y2 - y1 > 0.9:
        return None
    return (max(0, x1), max(0, y1), min(1, x2), min(1, y2))


def _overlap(a, b, pad=0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def saliency_map(image: Image.Image):
    """Coarse detail map (0..1, 96x54): label boxes are kept off busy/high-detail subject regions."""
    import cv2
    import numpy as np
    gray = np.asarray(image.convert("L").resize((384, 216)), dtype=np.float32)
    magnitude = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    small = cv2.resize(cv2.GaussianBlur(magnitude, (0, 0), 3), (96, 54), interpolation=cv2.INTER_AREA)
    peak = float(np.percentile(small, 95)) or 1.0
    return np.clip(small / peak, 0, 1)


def _detail_under(saliency, box, size) -> float:
    if saliency is None:
        return 0.0
    h, w = saliency.shape
    x0, x1 = max(0, int(box[0] / size[0] * w)), min(w, max(int(box[2] / size[0] * w), int(box[0] / size[0] * w) + 1))
    y0, y1 = max(0, int(box[1] / size[1] * h)), min(h, max(int(box[3] / size[1] * h), int(box[1] / size[1] * h) + 1))
    region = saliency[y0:y1, x0:x1]
    return float(region.mean()) if region.size else 0.0


def layout(anchors: list[Anchor], size: tuple[int, int], keepout: list[tuple[int, int, int, int]], config: AppConfig,
           style: dict[str, str], saliency=None) -> list[PlacedLabel]:
    """Score candidate positions around each anchor; reject any that collide with the subject anchors, other labels or keep-out zones."""
    width, height = size
    scale = height / 1080
    px = max(round(config.production.label_font_px * scale), round(28 * scale))
    fnt = font(px, True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    pad_x, pad_y = round(16 * scale), round(9 * scale)
    subject_boxes = [tuple(round(v * s) for v, s in zip(a.box, (width, height, width, height))) for a in anchors]
    placed: list[PlacedLabel] = []
    taken: list[tuple[int, int, int, int]] = []
    for anchor, subject in sorted(zip(anchors, subject_boxes), key=lambda p: p[0].point[0]):
        fnt = font(px, True, is_complex_script(anchor.label))
        lines = wrap_text(probe, anchor.label, fnt, round(width * 0.24))
        line_h = sum(fnt.getmetrics()) + 2
        box_w = round(max(text_width(probe, l, fnt) for l in lines)) + 2 * pad_x
        box_h = len(lines) * line_h + 2 * pad_y
        cx, cy = round(anchor.point[0] * width), round(anchor.point[1] * height)
        best, best_score = None, None
        gaps = [round(g * scale) for g in (60, 110, 170, 240, 340, 460)]
        for gap in gaps:
            for dx, dy in ((1, -1), (-1, -1), (1, 1), (-1, 1), (1, 0), (-1, 0), (0, -1), (0, 1)):
                bx = cx + dx * (subject[2] - subject[0]) // 2 + dx * gap - (box_w if dx < 0 else (box_w // 2 if dx == 0 else 0))
                by = cy + dy * (subject[3] - subject[1]) // 2 + dy * gap - (box_h if dy < 0 else (box_h // 2 if dy == 0 else 0))
                candidate = (bx, by, bx + box_w, by + box_h)
                if candidate[0] < round(30 * scale) or candidate[2] > width - round(30 * scale) \
                        or candidate[1] < round(30 * scale) or candidate[3] > height - round(30 * scale):
                    continue
                if any(_overlap(candidate, o, 8) for o in taken) or any(_overlap(candidate, k) for k in keepout):
                    continue
                overlap_subject = sum(_area_overlap(candidate, s) for s in subject_boxes)
                distance = abs((bx + box_w / 2) - cx) + abs((by + box_h / 2) - cy)
                detail = _detail_under(saliency, candidate, size)
                score = overlap_subject * 5 + distance + detail * 2600 * scale
                if best_score is None or score < best_score:
                    best, best_score = candidate, score
            if best is not None and best_score < subject_area(subject) * 0.1 + 300 * scale:
                break
        if best is None:
            continue   # cannot place without collision; caller reports and splits the shot
        edge = _nearest_edge_point(best, (cx, cy))
        placed.append(PlacedLabel(anchor.label, best, (cx, cy), edge, lines))
        taken.append(best)
    return placed


def subject_area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _area_overlap(a, b) -> int:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _nearest_edge_point(box, target):
    x = min(max(target[0], box[0]), box[2])
    y = min(max(target[1], box[1]), box[3])
    if box[0] < target[0] < box[2] and box[1] < target[1] < box[3]:
        return (round((box[0] + box[2]) / 2), box[3])
    return (round(x), round(y))


def draw(overlay: Image.Image, placed: list[PlacedLabel], t: float, start_times: list[float], config: AppConfig,
         style: dict[str, str]) -> None:
    """Animate: leader line grows to the target first, then the matching label fades in; earlier labels persist."""
    scale = overlay.height / 1080
    accent = hex_rgb(style.get("border_color", "#F2B233"), (242, 178, 51))
    draw_ = ImageDraw.Draw(overlay)
    fnt = font(max(round(config.production.label_font_px * scale), round(28 * scale)), True)
    line_h = sum(fnt.getmetrics()) + 2
    pad_x, pad_y = round(16 * scale), round(9 * scale)
    for label, begin in zip(placed, start_times):
        if t < begin:
            continue
        line_progress = min(1.0, (t - begin) / 0.5)
        text_alpha = min(1.0, max(0.0, (t - begin - 0.4) / 0.3))
        ex, ey = label.edge_px
        ax, ay = label.anchor_px
        px = ex + (ax - ex) * line_progress
        py = ey + (ay - ey) * line_progress
        draw_.line([(ex, ey), (px, py)], fill=accent + (255,), width=max(3, round(3 * scale)))
        if line_progress >= 1.0:
            r = round(9 * scale)
            draw_.ellipse((ax - r, ay - r, ax + r, ay + r), fill=accent + (255,), outline=(255, 255, 255, 255), width=max(2, round(2 * scale)))
        if text_alpha > 0:
            a = round(255 * text_alpha)
            draw_.rounded_rectangle(label.box, radius=round(10 * scale), fill=(250, 250, 248, round(240 * text_alpha)),
                                    outline=accent + (a,), width=max(2, round(3 * scale)))
            lfnt = font(fnt.size, True, is_complex_script(label.text))
            draw_lines(draw_, label.lines, lfnt, sum(lfnt.getmetrics()) + 2, label.box[0] + pad_x, label.box[1] + pad_y, (20, 24, 28, a))
