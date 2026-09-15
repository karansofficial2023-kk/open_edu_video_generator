from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document

from .schema import Scene, Segment, Shot, Storyboard


def read_input(path: str | Path) -> tuple[str, Storyboard | None]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        storyboard = Storyboard.model_validate_json(input_path.read_text(encoding="utf-8"))
        return storyboard.source or storyboard.title, storyboard
    if suffix == ".docx":
        storyboard = _try_read_storyboard_docx(input_path)
        if storyboard:
            return storyboard.source or _storyboard_text(storyboard), storyboard
        return _read_docx_text(input_path), None
    if suffix in {".txt", ".md"}:
        return input_path.read_text(encoding="utf-8"), None
    raise ValueError(f"Unsupported input type: {input_path.suffix}")


def _read_docx_text(path: Path) -> str:
    doc = Document(path)
    chunks: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            chunks.append(text)
    for table in doc.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if values:
                chunks.append(" | ".join(values))
    return "\n".join(chunks)


def _try_read_storyboard_docx(path: Path) -> Storyboard | None:
    doc = Document(path)
    title = "Educational Video"
    scenes: list[Scene] = []
    current_scene: Scene | None = None
    pending_narration = False

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if text.lower().startswith("storyboard:"):
            title = text.replace("Storyboard:", "", 1).strip() or title
            continue
        scene_match = re.match(r"scene\s+(\d+)\s*:\s*(.+)", text, re.IGNORECASE)
        if scene_match:
            current_scene = Scene(
                scene_number=int(scene_match.group(1)),
                title=scene_match.group(2).strip(),
                narration="",
                segments=[],
            )
            scenes.append(current_scene)
            pending_narration = False
            continue
        if text.lower().startswith("narration/audio/voiceover"):
            pending_narration = True
            continue
        if pending_narration and current_scene is not None:
            current_scene.narration = text.strip().strip('"')
            pending_narration = False

    if not scenes:
        return None

    scene_index = 0
    for table in doc.tables:
        while scene_index < len(scenes) and scenes[scene_index].segments:
            scene_index += 1
        if scene_index >= len(scenes):
            break
        scene = scenes[scene_index]
        headers = [_normalize_header(cell.text) for cell in table.rows[0].cells]
        def col(*names: str) -> int | None:
            wanted = {_normalize_header(name) for name in names}
            for index, header in enumerate(headers):
                if header in wanted:
                    return index
            return None
        number_i = col("S.no", "S no", "No") or 0
        narration_i = col("Splitting the Narration (Sentence wise)", "Narration", "Sentence") or 1
        media_i = col("Media Type")
        motion_i = col("Motion Plan")
        visual_i = col("Visual / Animation", "Visual", "Animation")
        local_i = col("Local Teaching Animation")
        labels_i = col("Labels")
        image_i = col("Image Recommendation")
        prompt_i = col("ComfyUI Prompt")
        wan_i = col("Wan Video Shot")
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) <= max(number_i, narration_i) or not cells[number_i].strip().isdigit():
                continue
            sentence = cells[narration_i].strip().strip('"')
            media_type = _cell(cells, media_i)
            motion = _cell(cells, motion_i)
            visual_main = _cell(cells, visual_i)
            local = _cell(cells, local_i)
            labels = _labels(_cell(cells, labels_i))
            image_recommendation = _cell(cells, image_i)
            comfy_prompt = _cell(cells, prompt_i)
            wan_prompt = _cell(cells, wan_i)
            visual = visual_main or local or image_recommendation or sentence
            planning_text = " | ".join(part for part in [media_type, motion, visual_main, local] if part)
            image_text = wan_prompt or comfy_prompt or image_recommendation or visual_main or local or visual or sentence
            keywords = labels or _keywords(sentence + " " + visual + " " + image_text)
            scene.segments.append(
                Segment(
                    segment_number=int(cells[number_i].strip()),
                    narration=sentence,
                    visual=visual,
                    image_prompt=image_text,
                    keywords=keywords,
                    shot=_shot_from_media_type(media_type, motion, visual, image_text, labels),
                )
            )
        scene_index += 1

    for scene in scenes:
        if not scene.segments and scene.narration:
            scene.segments = _segments_from_text(scene.narration)

    return Storyboard(title=title, source=_storyboard_text_from_scenes(scenes), scenes=scenes)


def _normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _cell(cells: list[str], index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    return cells[index].strip()


def _labels(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n,;]+", text or "") if part.strip()]


def _two_steps(text: str) -> list[str]:
    clauses = [part.strip(" .") for part in re.split(r",|\band\b|\bthen\b|\bwhile\b", text or "") if part.strip()]
    if len(clauses) >= 2:
        return [_short_title(clauses[0], "Observe"), _short_title(clauses[1], "Connect")]
    return [_short_title(text, "Observe"), "Connect to the concept"]


def _shot_from_media_type(media_type: str, motion: str, visual: str, image_prompt: str, labels: list[str] | None = None) -> Shot | None:
    routing_text = " ".join([media_type, motion, visual, image_prompt]).lower()
    heading = _clean_heading(visual or image_prompt)
    media = media_type.lower()
    is_explicit_process = any(term in " ".join([media_type, motion]).lower() for term in ["animation", "diagram", "local animation"])
    if "wan" in routing_text or "video shot" in routing_text:
        return Shot(template="video", heading=heading or "Video")
    if "photo" in media and not "animation" in media:
        return Shot(template="photo", heading=heading or "Photo")
    if not is_explicit_process and "realistic" in image_prompt.lower():
        return Shot(template="photo", heading=heading or "Photo")
    if "graph" in routing_text:
        return Shot(template="process", heading=heading or "Graph Relationship", steps=["Read the axes", "Compare the trend"], stage_fractions=[0, 0.55])
    if "diagram" in routing_text or "animation" in routing_text or "local animation" in routing_text:
        steps = _steps_from_labels_or_visual(labels or [], visual or image_prompt)
        return Shot(template="process", heading=heading or "Key Process", steps=steps, stage_fractions=[0, 0.55])
    return None


def _clean_heading(text: str) -> str:
    text = re.sub(
        r"\b(animation with labels|local animation|static image|photo with labels|wan video|"
        r"step-by-step animation|quick animation|educational diagram|animated diagram|"
        r"simplified diagram|labeled diagram|diagram showing|animation showing)\b",
        " ",
        text or "",
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" .|-:")
    return _short_title(text, "")


def _steps_from_labels_or_visual(labels: list[str], visual: str) -> list[str]:
    clean_labels = [label.strip() for label in labels if label.strip()]
    if len(clean_labels) >= 2:
        return [_short_title(clean_labels[0], "Observe"), _short_title(clean_labels[1], "Connect")]
    return _two_steps(visual)


def storyboard_from_text(title: str, text: str) -> Storyboard:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    scenes: list[Scene] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        scene_title = _short_title(paragraph, f"Scene {index}")
        scenes.append(
            Scene(
                scene_number=index,
                title=scene_title,
                narration=paragraph,
                segments=_segments_from_text(paragraph),
            )
        )
    return Storyboard(title=title, source=text, scenes=scenes)


def _segments_from_text(text: str) -> list[Segment]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    segments: list[Segment] = []
    for index, sentence in enumerate(sentences, start=1):
        keywords = _keywords(sentence)
        visual = "Show a clean educational visual with labeled concepts from the narration."
        if keywords:
            visual = "Show labeled educational visuals for " + ", ".join(keywords[:4]) + "."
        segments.append(
            Segment(
                segment_number=index,
                narration=sentence,
                visual=visual,
                image_prompt=f"Educational illustration about {', '.join(keywords[:5]) or sentence}",
                keywords=keywords,
            )
        )
    return segments


def _keywords(text: str) -> list[str]:
    stop = {
        "about", "above", "after", "again", "also", "because", "before", "being",
        "between", "could", "different", "during", "every", "first", "from",
        "have", "into", "more", "other", "over", "same", "should", "some",
        "such", "than", "that", "their", "them", "then", "there", "these",
        "this", "through", "where", "which", "while", "with", "would", "your",
        "the", "and", "for", "are", "was", "were", "has", "had", "not", "can",
    }
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        if word not in stop and word not in seen:
            seen.add(word)
            result.append(word)
    return result[:8]


def _short_title(text: str, fallback: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    if not words:
        return fallback
    return " ".join(words[:7])


def _storyboard_text(storyboard: Storyboard) -> str:
    return "\n".join(scene.narration for scene in storyboard.scenes if scene.narration)


def _storyboard_text_from_scenes(scenes: list[Scene]) -> str:
    return "\n".join(scene.narration for scene in scenes if scene.narration)


def save_storyboard(storyboard: Storyboard, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(storyboard.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

