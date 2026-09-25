from __future__ import annotations

import math
import re
import textwrap
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import AppConfig
from .schema import Segment, Storyboard


def _composite_config(config: AppConfig) -> AppConfig:
    if not config.render.burn_captions:
        return config
    result = config.model_copy(deep=True)
    result.render.burn_captions = False
    return result


def _generate_image_with_qa(
    comfy_client,
    segment,
    generated: Path,
    prefix: str,
    output_dir: Path,
    config: AppConfig,
    lesson_context: str = "",
) -> Path:
    concepts = _comparison_concepts(segment)
    if len(concepts) >= 2:
        return _generate_comparison_asset(
            comfy_client, segment, concepts, generated, prefix, output_dir, config, lesson_context
        )
    if not config.vision_qa.enabled:
        return comfy_client.generate_image(segment, generated, prefix)

    from .vision_qa import VisionQAClient, append_qa_audit

    if segment.shot and segment.shot.template == "title_card":
        result = comfy_client.generate_image(segment, generated, prefix)
        comfy_client.free_memory()
        append_qa_audit(output_dir / "vision_qa.json", {
            "segment": segment.segment_number,
            "attempt": 1,
            "image": str(generated),
            "accepted": True,
            "qa_mode": "deterministic_title_background",
            "reasons": ["Title typography is composited by the renderer; semantic image QA is not applicable."],
            "coordinate_policy": "no_labels_on_title_card",
        })
        return result

    qa_client = VisionQAClient(config)
    attempts = max(1, config.vision_qa.max_attempts)
    last_result = None
    for attempt in range(1, attempts + 1):
        candidate = generated.with_stem(f"{generated.stem}_attempt_{attempt}")
        comfy_client.generate_image(segment, candidate, f"{prefix}_attempt_{attempt}")
        comfy_client.free_memory()
        result = qa_client.analyze(segment, candidate)
        append_qa_audit(output_dir / "vision_qa.json", {
            "segment": segment.segment_number,
            "attempt": attempt,
            "image": str(candidate),
            **result.to_dict(),
            "coordinate_policy": "proposals_only_not_verified",
        })
        last_result = result
        if result.accepted:
            candidate.replace(generated)
            return generated
    if config.vision_qa.block_on_failure:
        reasons = "; ".join(last_result.reasons if last_result else []) or "quality thresholds not met"
        raise ValueError(f"Vision QA rejected {prefix} after {attempts} attempts: {reasons}")
    candidate.replace(generated)
    return generated


def _comparison_concepts(segment: Segment) -> list[str]:
    production = " ".join([segment.visual or "", segment.image_prompt or ""])
    match = re.search(
        r"narration-defined mechanism(?:s)?\s*:\s*([^.]+)",
        production,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return list(dict.fromkeys(
        item.strip(" ;,:-") for item in match.group(1).split(";") if item.strip(" ;,:-")
    ))


def _definition_for_concept(narration: str, concept: str, concepts: list[str]) -> str:
    lower = narration.casefold()
    start = lower.find(concept.casefold())
    if start < 0:
        return f"{concept}, as defined by the narration: {narration}"
    ends = [lower.find(other.casefold(), start + len(concept)) for other in concepts if other.casefold() != concept.casefold()]
    ends = [position for position in ends if position > start]
    end = min(ends) if ends else len(narration)
    return narration[start:end].strip(" ,;.")


def _transfer_endpoints(lesson_context: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"(?:transfer|movement)\s+of\s+(.+?)\s+from\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+?)(?:[.;,]|$)",
        lesson_context,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return tuple(part.strip(" ,;:.") for part in match.groups())


def _relationship_visual_cues(definition: str, lesson_context: str = "") -> str:
    lower = definition.casefold()
    if "within the same" in lower or "within one" in lower:
        endpoints = _transfer_endpoints(lesson_context)
        endpoint_cue = ""
        if endpoints:
            material, source, receiver = endpoints
            endpoint_cue = (
                f"Explicitly show the {source} as the source structure, the {receiver} as the receiving structure, "
                f"and a small realistic amount of {material} resting on or contacting the {receiver}. "
            )
        return (
            "Show one complete subject only and make the internal source-to-receiver relationship visibly clear. "
            "Both the source structure and receiving structure named in the definition must be simultaneously "
            "visible, recognizable, and in sharp focus; never use an isolated close-up of only one endpoint. "
            f"{endpoint_cue}"
            "Do not include an insect, animal, hand, tool, or any external transfer agent. If particles or "
            "material are named, show a small realistic amount leaving the source and naturally contacting the "
            "receiving structure within that one subject."
        )
    if "same plant" in lower or "same organism" in lower or "same system" in lower:
        if "flower" in lower:
            return (
                "One complete flowering plant with at least two distinct open flowers visibly connected to the "
                "same continuous stem or branch; both flowers and their shared plant connection fit in the frame."
            )
        return (
            "Show one complete shared plant, organism, or system containing two clearly separate units, with "
            "the relationship occurring between those units while their shared origin remains visible."
        )
    if "different plant" in lower or "different organism" in lower or "different system" in lower:
        return (
            "Show two visibly separate plants, organisms, or systems of the same relevant type, with the "
            "relationship occurring from one separate subject to the other."
        )
    return "Make the narration's physical relationship visually explicit without adding an unmentioned agent."


def _physical_clause_for_concept(definition: str, concept: str) -> str:
    clause = re.sub(
        rf"^\s*{re.escape(concept)}\s+(?:occurs|involves|means|is)\s*",
        "",
        definition,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" ,;:.")
    return clause or definition


def _comparison_panel_scene(definition: str, lesson_context: str = "") -> str:
    lower = definition.casefold()
    endpoints = _transfer_endpoints(lesson_context)
    endpoint_text = ""
    if endpoints:
        _, source, receiver = endpoints
        endpoint_text = f" Its {source} and {receiver} are both visible and botanically recognizable."

    within = re.search(r"within (?:the )?(?:same|one)\s+([a-z][a-z -]+?)(?:[,.;]|$)", definition, re.IGNORECASE)
    if within:
        subject = within.group(1).strip()
        return (
            f"One complete {subject} alone in the frame, with its central structures unobscured and in focus."
            f"{endpoint_text} No insect, animal, hand, tool, or second {subject}."
        )
    if "same plant" in lower or "same organism" in lower or "same system" in lower:
        if "flower" in lower:
            return (
                "One complete flowering plant with at least two distinct open flowers visibly connected to the "
                "same continuous stem or branch; both flowers and their shared plant connection fit in the frame."
            )
        return (
            "One complete shared plant, organism, or system with two distinct relevant units visibly connected "
            "to that same subject; the shared physical origin and both units must fit in the frame."
        )
    if "different plant" in lower or "different organism" in lower or "different system" in lower or "between plants" in lower:
        if "flower" in lower:
            return (
                "Two separate complete flowering plants of the same species, each with at least one open flower; "
                "their separate stems and separate physical origins are clearly visible in the frame."
            )
        return (
            "Two separate complete plants, organisms, or systems of the same relevant type, clearly separated "
            "in the frame so they cannot be mistaken for parts of one subject."
        )
    return definition


def _comparison_search_query(concept: str, definition: str, lesson_context: str = "") -> str:
    terms: list[str] = []
    lower = definition.casefold()
    endpoints = _transfer_endpoints(lesson_context)
    within = re.search(r"within (?:the )?(?:same|one)\s+([a-z][a-z -]+?)(?:[,.;]|$)", definition, re.IGNORECASE)
    if within:
        terms.append(within.group(1).strip())
    if endpoints and "same plant" not in lower and "different plant" not in lower and "between plants" not in lower:
        _, source, receiver = endpoints
        terms.extend([source, receiver])
    if "same plant" in lower:
        terms.extend(["flowering plant", "two flowers", "same branch"])
    elif "different plant" in lower or "between plants" in lower:
        terms.extend(["two flowering plants", "same species"])
    if not terms:
        terms.extend([concept, definition])
    return " ".join(dict.fromkeys(term for term in terms if term)).strip()


def _generate_comparison_asset(
    comfy_client,
    segment: Segment,
    concepts: list[str],
    generated: Path,
    prefix: str,
    output_dir: Path,
    config: AppConfig,
    lesson_context: str = "",
) -> Path:
    panel_dir = generated.parent / f"{generated.stem}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Path] = []
    for index, concept in enumerate(concepts, start=1):
        panel_segment = segment.model_copy(deep=True)
        definition = _definition_for_concept(segment.narration, concept, concepts)
        physical_scene = _comparison_panel_scene(definition, lesson_context)
        panel_segment.visual = physical_scene
        panel_segment.image_prompt = (
            f"Single realistic scientific reference photograph: {physical_scene} "
            "Show only this literal spatial arrangement, with every required subject large and unmistakable, "
            "neutral natural lighting, sharp detail, and a stable camera. "
            "No comparison panel, no collage, no text, no label, no arrow, no caption, no watermark, no logo, "
            "no diagram, no infographic, no invented anatomy."
        )
        if panel_segment.shot:
            panel_segment.shot.template = "photo"
            panel_segment.shot.labels = []
        panel = panel_dir / f"panel_{index:02d}.png"
        from .asset_retrieval import retrieve_approved_asset

        query = _comparison_search_query(concept, definition, lesson_context)
        if not retrieve_approved_asset(panel_segment, query, panel, output_dir, config):
            _generate_image_with_qa(
                comfy_client,
                panel_segment,
                panel,
                f"{prefix}_panel_{index:02d}",
                output_dir,
                config,
                lesson_context,
            )
        panels.append(panel)
    _compose_comparison_panels(panels, concepts, generated, config)
    return generated


def _compose_comparison_panels(
    panels: list[Path], concepts: list[str], output: Path, config: AppConfig
) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    if len(panels) >= 3:
        _compose_comparison_rows(panels, concepts, output, config)
        return
    column_width = width // len(panels)
    canvas = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, concept) in enumerate(zip(panels, concepts)):
        left = index * column_width
        right = width if index == len(panels) - 1 else left + column_width
        with Image.open(panel) as source:
            art = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (right - left, height),
                method=Image.Resampling.LANCZOS,
            )
        canvas.paste(art, (left, 0))
        draw.rectangle((left, 0, right, round(height * 0.13)), fill=(10, 18, 24))
        if index:
            draw.line((left, 0, left, height), fill=(255, 255, 255), width=4)
        _text_box(
            draw,
            concept[:1].upper() + concept[1:],
            (left + 24, 12, right - 24, round(height * 0.13) - 10),
            config,
            round(height * 0.044),
            bold=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _compose_comparison_rows(
    panels: list[Path], concepts: list[str], output: Path, config: AppConfig
) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    safe_bottom = round(height * 0.28) if config.render.burn_captions else 0
    content_height = height - safe_bottom
    row_height = content_height // len(panels)
    title_width = round(width * 0.22)
    canvas = Image.new("RGB", (width, height), (10, 18, 24))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, concept) in enumerate(zip(panels, concepts)):
        top = index * row_height
        bottom = content_height if index == len(panels) - 1 else top + row_height
        with Image.open(panel) as source:
            art = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (width - title_width, bottom - top),
                method=Image.Resampling.LANCZOS,
            )
        canvas.paste(art, (title_width, top))
        draw.rectangle((0, top, title_width, bottom), fill=(10, 18, 24))
        if index:
            draw.line((0, top, width, top), fill=(255, 255, 255), width=4)
        _text_box(
            draw,
            concept[:1].upper() + concept[1:],
            (30, top + 16, title_width - 26, bottom - 16),
            config,
            round(height * 0.038),
            bold=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def render_segment_frames(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> list[Path]:
    from .label_overlay import is_label_template

    output_dir = Path(output_dir)
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    comfy_client = None
    availability_checked = False
    if config.visual_mode.lower() in {"comfyui-image", "hybrid"} and config.comfyui.enabled:
        from .comfyui_client import ComfyUIClient

        comfy_client = ComfyUIClient(config)

    for scene, segment in storyboard.all_segments():
        path = frames_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.png"
        pro_asset_templates = {"split_screen", "title_card"}
        if segment.shot and segment.shot.template in {"video", "video_broll", "short_motion_clip"} and not segment.shot.asset_path:
            if not comfy_client or not config.comfyui.video_enabled:
                raise ValueError(
                    f"Scene {scene.scene_number}.{segment.segment_number} requests generated video, "
                    "but comfyui.video_enabled is false or ComfyUI is disabled."
                )
            if not availability_checked:
                availability_checked = True
                if not comfy_client.is_available():
                    raise RuntimeError(f"ComfyUI is not available at {config.comfyui.api_url}")
            generated_dir = output_dir / "generated_videos"
            generated = generated_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.mp4"
            comfy_client.generate_video(
                segment,
                generated,
                f"open_edu_video_scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}",
            )
            paths.append(generated)
            continue
        if segment.shot and segment.shot.template not in {"photo", "video", "video_broll", "short_motion_clip", *pro_asset_templates} and not is_label_template(segment.shot.template):
            from .animation_renderer import render_frame
            render_frame(segment, scene.title, 0, config).save(path)
            paths.append(path)
            continue
        if segment.shot and segment.shot.asset_path:
            asset = Path(segment.shot.asset_path)
            if not asset.is_file():
                raise ValueError(f"Missing shot asset: {asset}")
            paths.append(asset)
            continue
        reviewed = None
        if config.render.reviewed_assets_dir:
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = Path(config.render.reviewed_assets_dir) / (path.stem + suffix)
                if candidate.is_file():
                    reviewed = candidate
                    break
        if reviewed:
            if segment.shot and is_label_template(segment.shot.template):
                paths.append(reviewed)
                continue
            if segment.shot and segment.shot.template in pro_asset_templates:
                from .animation_renderer import render_frame
                with Image.open(reviewed) as source:
                    render_frame(segment, scene.title, 0, _composite_config(config), source=source).save(path)
                paths.append(path)
                continue
            if config.render.layout == "modern" or segment.shot:
                paths.append(reviewed)
                continue
            render_ai_image_frame(storyboard.title, scene.title, segment, reviewed, path, config)
        elif comfy_client:
            if not availability_checked:
                availability_checked = True
                if not comfy_client.is_available():
                    if not config.comfyui.fallback_to_slideshow:
                        raise RuntimeError(f"ComfyUI is not available at {config.comfyui.api_url}")
                    comfy_client = None
                    render_segment_frame(storyboard.title, scene.title, segment, path, config)
                    paths.append(path)
                    continue
            generated_dir = output_dir / "generated_images"
            generated = generated_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.png"
            _generate_image_with_qa(
                comfy_client,
                segment,
                generated,
                f"open_edu_scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}",
                output_dir,
                config,
                storyboard.source,
            )
            if segment.shot and is_label_template(segment.shot.template):
                paths.append(generated)
                continue
            if segment.shot and segment.shot.template in pro_asset_templates:
                from .animation_renderer import render_frame
                with Image.open(generated) as source:
                    render_frame(segment, scene.title, 0, _composite_config(config), source=source).save(path)
                paths.append(path)
                continue
            if config.render.layout == "modern" or segment.shot:
                paths.append(generated)
                continue
            render_ai_image_frame(storyboard.title, scene.title, segment, generated, path, config)
        else:
            if config.render.layout == "modern":
                raise ValueError(f"Scene {scene.scene_number}.{segment.segment_number} needs a photo asset, "
                                 "ComfyUI, or a structured shot template. Use --plan-only to edit the storyboard.")
            render_segment_frame(storyboard.title, scene.title, segment, path, config)
        paths.append(path)
    return paths


def render_ai_image_frame(
    title: str,
    scene_title: str,
    segment: Segment,
    source_image: str | Path,
    output_path: str | Path,
    config: AppConfig,
) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    # Reserve separate bands so typography never hides or mislabels anatomy.
    margin = round(width * 0.035)
    header = round(height * 0.11)
    footer = round(height * 0.20)
    label_h = round(height * 0.05) if segment.keywords else 0
    image = Image.new("RGB", (width, height), _hex(config.render.subtitle_bg))
    draw = ImageDraw.Draw(image)
    with Image.open(source_image) as source:
        art = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"),
                               (width, height - header - footer - label_h), Image.Resampling.LANCZOS)
    image.paste(art, ((width - art.width) // 2, header + (height - header - footer - label_h - art.height) // 2))
    draw.rectangle((0, header - 4, width, header), fill=_hex(config.render.accent))
    _text_box(draw, scene_title, (margin, 8, width - margin, header - 12),
              config, round(height * 0.040), bold=True)
    if segment.keywords:
        labels = "  |  ".join(dict.fromkeys(word.strip() for word in segment.keywords if word.strip()))
        _text_box(draw, labels, (margin, height - footer - label_h, width - margin, height - footer),
                  config, round(height * 0.025))
    _text_box(draw, segment.narration,
              (margin, height - footer + 10, width - margin, height - 16),
              config, round(height * 0.035))
    image.save(output_path)


def _pixel_wrap(draw, text, font, width):
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=font) <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = ""
        for character in word:
            if line and draw.textlength(line + character, font=font) > width:
                lines.append(line)
                line = ""
            line += character
    if line:
        lines.append(line)
    return lines


def _text_box(draw, text, box, config, size, bold=False):
    left, top, right, bottom = box
    for candidate in range(max(12, size), 11, -1):
        font = (ImageFont.truetype(config.render.font_path, candidate)
                if config.render.font_path else _font(candidate, bold))
        lines = _pixel_wrap(draw, text, font, right - left)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent + max(2, candidate // 6)
        if len(lines) * line_height <= bottom - top:
            y = top + ((bottom - top) - len(lines) * line_height) // 2
            for line in lines:
                x = left + ((right - left) - draw.textlength(line, font=font)) // 2
                draw.text((x, y), line, font=font, fill=_hex(config.render.subtitle_fg))
                y += line_height
            return
    raise ValueError("Text is too long for the frame. Split this narration into shorter storyboard segments.")


def render_segment_frame(
    title: str,
    scene_title: str,
    segment: Segment,
    output_path: str | Path,
    config: AppConfig,
) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    bg = _hex(config.render.background)
    ink = _hex(config.render.ink)
    accent = _hex(config.render.accent)
    second = _hex(config.render.second_accent)

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    title_font = _font(48, bold=True)
    heading_font = _font(34, bold=True)
    body_font = _font(30)
    label_font = _font(24, bold=True)
    small_font = _font(20)

    draw.rectangle([0, 0, width, 82], fill=ink)
    draw.text((44, 22), _fit_text(title, 44), fill=(255, 255, 255), font=label_font)
    draw.text((width - 210, 22), f"{segment.segment_number:02d}", fill=(255, 255, 255), font=label_font)

    draw.rounded_rectangle([44, 112, width - 44, 184], radius=6, outline=accent, width=3)
    draw.text((68, 128), _fit_text(scene_title, 48), fill=ink, font=heading_font)

    _draw_concept_visual(draw, width, height, segment.keywords, accent, second, ink)

    narration_box = [64, height - 172, width - 64, height - 54]
    draw.rounded_rectangle(narration_box, radius=8, fill=(255, 255, 255), outline=(210, 214, 210), width=2)
    lines = _wrap(segment.narration, 68)
    y = narration_box[1] + 20
    for line in lines[:3]:
        draw.text((narration_box[0] + 24, y), line, fill=ink, font=body_font)
        y += 36

    visual_lines = _wrap(segment.visual, 46)
    y = 226
    draw.text((78, y), "Visual Direction", fill=accent, font=label_font)
    y += 36
    for line in visual_lines[:4]:
        draw.text((78, y), line, fill=ink, font=small_font)
        y += 28

    image.save(output_path, quality=95)


def _draw_concept_visual(draw: ImageDraw.ImageDraw, width: int, height: int, keywords: list[str], accent, second, ink) -> None:
    center_x = int(width * 0.68)
    center_y = int(height * 0.47)
    radius = 110
    petal_count = max(6, min(10, len(keywords) + 4))
    for i in range(petal_count):
        angle = 2 * math.pi * i / petal_count
        x = center_x + int(math.cos(angle) * radius)
        y = center_y + int(math.sin(angle) * radius)
        draw.ellipse([x - 58, y - 34, x + 58, y + 34], fill=(245, 221, 151), outline=second, width=3)
    draw.ellipse([center_x - 72, center_y - 72, center_x + 72, center_y + 72], fill=accent, outline=ink, width=4)
    draw.line([center_x, center_y + 72, center_x, height - 225], fill=accent, width=12)
    draw.arc([center_x - 155, center_y + 96, center_x - 15, center_y + 240], 210, 340, fill=accent, width=8)
    draw.arc([center_x + 15, center_y + 116, center_x + 170, center_y + 260], 200, 330, fill=accent, width=8)

    font = _font(21, bold=True)
    labels = keywords[:6] or ["concept", "process", "learning"]
    label_points = [
        (center_x - 275, center_y - 130),
        (center_x + 150, center_y - 135),
        (center_x - 310, center_y + 10),
        (center_x + 170, center_y + 10),
        (center_x - 245, center_y + 145),
        (center_x + 125, center_y + 145),
    ]
    for label, point in zip(labels, label_points):
        x, y = point
        text = label.title()
        box = [x - 12, y - 8, x + 190, y + 36]
        draw.rounded_rectangle(box, radius=7, fill=(255, 255, 255), outline=(204, 204, 204), width=2)
        draw.text((x, y), _fit_text(text, 18), fill=ink, font=font)


@lru_cache(maxsize=256)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False)


def _fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _cover_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _stroke_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke: tuple[int, int, int, int],
) -> None:
    x, y = xy
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        draw.text((x + dx, y + dy), text, fill=stroke, font=font)
    draw.text((x, y), text, fill=fill, font=font)
