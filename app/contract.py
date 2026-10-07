"""Reader for the Transcribe renderer-ready storyboard contract (DOCX vertical records or JSON v2).

The reader only maps fields; it never guesses lesson meaning from topic keywords.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .schema import Scene, Segment, Storyboard

LIST_FIELDS = {"labels", "label_placement", "formula_lines", "explain_steps", "steps", "columns"}
PENDING = "COORDINATES_PENDING_APPROVED_IMAGE"


def split_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = (value or "").strip()
    if not text:
        return []
    parts = re.split(r"\r?\n|;\s*(?=\S)", text)
    cleaned = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", p).strip() for p in parts]
    return [p for p in cleaned if p]


def parse_duration(value) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else 0.0


def segment_from_fields(fields: dict, fallback_number: int = 1) -> Segment:
    get = lambda key: str(fields.get(key) or "").strip()
    shot_id = get("shot_id")
    number = fallback_number
    if "." in shot_id and shot_id.split(".")[-1].isdigit():
        number = int(shot_id.split(".")[-1])
    narration = get("narration")
    review = get("review_notes")
    motion_prompt = get("ltx_video_prompt") or get("wan_video_prompt")
    return Segment(
        segment_number=number,
        narration=narration,
        visual=get("image_requirement") or narration,
        image_prompt=get("image_prompt") or get("image_requirement"),
        keywords=split_list(fields.get("labels")),
        shot_id=shot_id,
        visual_type=get("visual_type"),
        media_type=get("media_type"),
        image_requirement=get("image_requirement"),
        labels=split_list(fields.get("labels")),
        label_placement=split_list(fields.get("label_placement")),
        label_style=get("label_style"),
        motion=get("motion"),
        subtitle=get("subtitle"),
        subtitle_style=get("subtitle_style"),
        formula_lines=split_list(fields.get("formula_lines")),
        explain_steps=split_list(fields.get("explain_steps")),
        duration_hint=parse_duration(fields.get("duration")),
        motion_prompt=motion_prompt,
        animate=get("animate").lower() in {"true", "yes", "1"},
        review_notes=review,
        source_references=[review] if review and review.lower().startswith("source") else [],
        heading=get("heading"),
        asset_path=get("asset_path"),
        steps=split_list(fields.get("steps")),
        columns=split_list(fields.get("columns")),
    )


def looks_like_contract_table(table: Table) -> bool:
    return bool(table.rows) and len(table.rows[0].cells) == 2 and table.rows[0].cells[0].text.strip() == "shot_id"


def read_contract_docx(path: str | Path) -> Storyboard | None:
    doc = Document(path)
    if not any(looks_like_contract_table(t) for t in doc.tables):
        return None
    title = "Educational Video"
    scenes: list[Scene] = []
    scene: Scene | None = None
    pending_narration = False
    pending_heading = ""
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if not text:
                continue
            if text.lower().startswith("storyboard:"):
                title = text.split(":", 1)[1].strip() or title
                continue
            match = re.match(r"scene\s+(\d+)\s*:\s*(.+)", text, re.I)
            if match:
                scene = Scene(scene_number=int(match.group(1)), title=match.group(2).strip(), narration="", segments=[])
                scenes.append(scene)
                pending_narration = False
                continue
            if text.lower().startswith("narration/audio/voiceover"):
                pending_narration = True
                continue
            shot_heading = re.match(r"shot\s+\d+\.\d+\s*:\s*(.+)", text, re.I)
            if shot_heading:
                pending_heading = shot_heading.group(1).strip()
                continue
            if pending_narration and scene is not None:
                scene.narration = text.strip('"')
                pending_narration = False
        elif tag == "tbl":
            table = Table(child, doc)
            if not looks_like_contract_table(table):
                continue
            fields = {r.cells[0].text.strip(): r.cells[1].text.strip() for r in table.rows if len(r.cells) >= 2}
            if scene is None:
                shot_scene = int(str(fields.get("shot_id", "1.1")).split(".")[0] or 1)
                scene = Scene(scene_number=shot_scene, title=f"Scene {shot_scene}", narration="", segments=[])
                scenes.append(scene)
            if pending_heading and not fields.get("heading"):
                fields["heading"] = pending_heading
            pending_heading = ""
            scene.segments.append(segment_from_fields(fields, len(scene.segments) + 1))
    return _finish(title, scenes)


def read_contract_json(data: dict) -> Storyboard | None:
    """Accepts Transcribe schema_version 2.x JSON: scenes[].shots[] (or segments[]) with spec field names."""
    if "scenes" not in data:
        return None
    scenes: list[Scene] = []
    for scene_data in data["scenes"]:
        shots = scene_data.get("shots") or scene_data.get("segments") or []
        if not shots or not any("visual_type" in s or "shot_id" in s for s in shots):
            return None
        number = int(scene_data.get("scene_number") or scene_data.get("sceneNumber") or len(scenes) + 1)
        scene = Scene(scene_number=number, title=str(scene_data.get("scene_title") or scene_data.get("title") or f"Scene {number}"),
                      narration=str(scene_data.get("narration") or ""), segments=[])
        for index, shot in enumerate(shots, start=1):
            scene.segments.append(segment_from_fields(shot, index))
        scenes.append(scene)
    return _finish(str(data.get("title") or "Educational Video"), scenes, data)


def _finish(title: str, scenes: list[Scene], data: dict | None = None) -> Storyboard:
    from .router import route_segment
    for scene in scenes:
        for segment in scene.segments:
            segment.shot = route_segment(segment)
    board = Storyboard(title=title, source="\n".join(s.narration for s in scenes if s.narration), scenes=scenes)
    if data:
        board.required_topics = {k: list(v) for k, v in (data.get("required_topics") or {}).items()}
        board.language = data.get("language") or board.language
        board.subject = str(data.get("subject") or "").strip()
    return board
