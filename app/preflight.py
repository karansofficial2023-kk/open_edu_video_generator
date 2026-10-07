"""Strict preflight: machine-readable issues {shot_id, field, severity, renderer, reason, repair} returned to Transcribe."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import requests

from . import vision
from .assets import is_generic_prompt
from .config import AppConfig
from .contract import PENDING
from .formulas import validate as validate_formula
from .router import APPROVED_VISUAL_TYPES, METADATA_PATTERN
from .schema import Storyboard
from .typography import font


def issue(shot_id, field, severity, renderer, reason, repair):
    return {"shot_id": shot_id, "field": field, "severity": severity, "renderer": renderer, "reason": reason, "repair": repair}


def check_services(config: AppConfig) -> list[dict]:
    issues = []
    if not (Path(config.ffmpeg_path).exists() or shutil.which(config.ffmpeg_path)):
        issues.append(issue("*", "ffmpeg_path", "error", "ffmpeg", "FFmpeg not found", "set ffmpeg_path in the config"))
    try:
        requests.get(f"{config.comfyui.api_url}/system_stats", timeout=5).raise_for_status()
        info = requests.get(f"{config.comfyui.api_url}/object_info/CheckpointLoaderSimple", timeout=10).json()
        models = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if config.production.image_checkpoint not in models:
            issues.append(issue("*", "image_checkpoint", "error", "comfyui", f"{config.production.image_checkpoint} is not installed in ComfyUI",
                                "install the checkpoint or change production.image_checkpoint"))
    except (requests.RequestException, KeyError, ValueError):
        issues.append(issue("*", "comfyui", "error", "comfyui", f"ComfyUI is not reachable at {config.comfyui.api_url}",
                            "start ComfyUI before rendering (fails before voice synthesis by design)"))
    if not Path(config.production.image_workflow_path).exists():
        issues.append(issue("*", "image_workflow_path", "error", "comfyui", "image workflow JSON missing", "restore workflows/flux_schnell_txt2img_api.json"))
    if config.production.semantic_qa and not vision.available(config):
        issues.append(issue("*", "vision_model", "warning", "qwen-vision", "no vision model available in Ollama; semantic QA and label targeting are skipped",
                            f"ollama pull {config.production.vision_model}"))
    if getattr(font(20), "path", None) is None:
        issues.append(issue("*", "font", "warning", "pillow", "no TrueType font found; text will use a low-quality default", "install Segoe UI/Noto or set render.font_path"))
    return issues


def check_language(board: Storyboard, config: AppConfig) -> list[dict]:
    """Language package readiness: voice for the lesson language and shaping support for its script."""
    issues = []
    language = (board.language or {}).get("bcp47", "") or "en"
    prefix = language.split("-")[0].lower()
    if prefix != "en":
        from . import shaping
        from .tts_edge import config_for_language
        from .tts_edge import chosen_provider
        try:
            if chosen_provider(prefix, config) == "edge":              # an installed open-source voice needs no Edge voice
                config_for_language(board, config)
        except ValueError as exc:
            issues.append(issue("*", "language", "error", "tts", str(exc), "configure a voice for this language (see docs/OPEN_VOICES.md)"))
        if (board.language or {}).get("script", "Latn") != "Latn" and not shaping.available():
            issues.append(issue("*", "language", "error", "typography",
                                "complex-script text cannot be shaped correctly without PySide6 (Qt/HarfBuzz)", "pip install PySide6-Essentials"))
    return issues


def check_storyboard(board: Storyboard, config: AppConfig) -> list[dict]:
    issues = check_language(board, config)
    for scene, seg in board.all_segments():
        sid = seg.shot_id or f"{scene.scene_number}.{seg.segment_number}"
        shot = seg.shot
        renderer = shot.template if shot else "unrouted"
        if not seg.narration.strip():
            issues.append(issue(sid, "narration", "error", renderer, "empty narration", "provide the approved narration for this shot"))
        if seg.visual_type and seg.visual_type not in APPROVED_VISUAL_TYPES:
            issues.append(issue(sid, "visual_type", "error", renderer, f"unapproved visual_type {seg.visual_type!r}", f"use one of {sorted(APPROVED_VISUAL_TYPES)}"))
        if shot is None:
            issues.append(issue(sid, "visual_type", "error", "unrouted", "shot cannot be routed to a renderer", "provide visual_type"))
            continue
        if seg.subtitle and METADATA_PATTERN.search(seg.subtitle):
            issues.append(issue(sid, "subtitle", "error", renderer, "subtitle contains production metadata", "keep only viewer-facing text in subtitle"))
        if shot.template == "title_card" and not (shot.heading or scene.title):
            issues.append(issue(sid, "heading", "error", renderer, "title_card has no title", "provide a concise title"))
        for line in seg.formula_lines:
            problem = validate_formula(line)
            if problem:
                issues.append(issue(sid, "formula_lines", "error", "formula", f"{line[:60]!r}: {problem}", "move prose to explain_steps and provide the exact equation"))
        if seg.labels:
            if seg.visual_type == "short_motion_clip":
                issues.append(issue(sid, "labels", "error", renderer, "labels on AI motion cannot be anchored", "split into a motion clip followed by a stable labeled still"))
            if len(seg.label_placement) != len(seg.labels):
                issues.append(issue(sid, "label_placement", "error", renderer, "label_placement count differs from labels",
                                    f"give one target description per label (or {PENDING})"))
            if len(seg.labels) > 6:
                issues.append(issue(sid, "labels", "warning", renderer, "more than 6 labels crowd one frame", "split into focused shots"))
        if shot.template in {"photo", "title_card", "video"} and not shot.asset_path:
            if not seg.image_prompt.strip() and not seg.image_requirement.strip():
                issues.append(issue(sid, "image_prompt", "error", renderer, "no image_prompt or image_requirement", "describe the subject, composition and lighting"))
            elif is_generic_prompt(seg):
                level = "info" if config.production.prompt_director else "warning"
                issues.append(issue(sid, "image_prompt", level, renderer, "generic prompt; will be repaired from approved narration by the prompt director",
                                    "make the prompt name the exact subject and visible evidence"))
        if shot.asset_path and not Path(shot.asset_path).is_file():
            issues.append(issue(sid, "asset_path", "error", renderer, f"asset not found: {shot.asset_path}", "provide the reviewed asset"))
        if shot.template == "split_screen" and not 2 <= len(seg.columns) <= 3:
            issues.append(issue(sid, "columns", "error", renderer, "split_screen needs 2-3 panel entries in columns ('Caption | image prompt')",
                                "provide one panel per column, or split into focused shots"))
        if shot.template == "circuit":
            from .circuit import CircuitError, parse_row
            if not 1 <= len(seg.columns) <= 3:
                issues.append(issue(sid, "columns", "error", renderer, "circuit needs 1-3 rows in columns", "one row per schematic"))
            for row in seg.columns[:3]:
                try:
                    parse_row(row)
                except CircuitError as exc:
                    issues.append(issue(sid, "columns", "error", renderer, str(exc), "format: 'Caption | series: battery 12 V; resistor R1; resistor R2'"))
        if shot.template == "graph":
            from .charts import parse_function, parse_rows
            try:
                is_function = parse_function(seg.columns) is not None
            except Exception as exc:
                issues.append(issue(sid, "columns", "error", renderer, f"function plot cannot be evaluated: {exc}",
                                    "give a plain expression such as 'y = x^2 - 5x + 6' and 'x range: -1 to 6'"))
                is_function = True
            if not is_function:
                if len(parse_rows(seg.columns)) < 2:
                    issues.append(issue(sid, "columns", "error", renderer, "graph needs at least two 'Label: value' rows (or a function 'y = ...') in columns",
                                        "provide the data rows, axis unit in explain_steps and a Source note in review_notes"))
                if "source" not in seg.review_notes.lower():
                    issues.append(issue(sid, "review_notes", "error", renderer, "data graph has no source provenance",
                                        "add 'Source: <publication, date>' to review_notes"))
    return issues


def check_derivations(board: Storyboard) -> list[dict]:
    """Symbolic check that each derivation step follows from the previous equation (warnings for SME review)."""
    try:
        from .derivation import check_chain
    except Exception:           # sympy missing: the check is optional
        return []
    chain = [(seg.shot_id or f"{scene.scene_number}.{seg.segment_number}", seg.formula_lines)
             for scene, seg in board.all_segments() if seg.formula_lines]
    return [issue(shot_id, "formula_lines", "warning", "formula", message, "SME: verify or correct this derivation step")
            for shot_id, message in check_chain(chain)]


def run_preflight(board: Storyboard, config: AppConfig, output_dir: Path, services: bool = True) -> list[dict]:
    issues = (check_services(config) if services else []) + check_storyboard(board, config) + check_derivations(board)
    Path(output_dir, "preflight.json").write_text(json.dumps({"issues": issues}, indent=2, ensure_ascii=False), encoding="utf-8")
    return issues
