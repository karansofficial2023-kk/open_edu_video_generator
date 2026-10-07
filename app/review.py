"""Small, explicit lint checks; this is not an automated fact checker."""
import json
import re
from pathlib import Path

from .animation_renderer import stage_times
from .coverage import coverage_report


def review_storyboard(board, output_dir, config=None, strict_coverage: bool = False):
    findings = []
    replacements = dict(getattr(config, "text_replacements", {}) or {})
    for scene, segment in board.all_segments():
        def add(level, message, code=""):
            findings.append({"scene": scene.scene_number, "segment": segment.segment_number,
                             "level": level, "code": code, "message": message})
        text = segment.narration.lower()
        for term, replacement in replacements.items():
            if re.search(rf"\b{re.escape(term.lower())}\b", text):
                add("error", f"Check terminology: {term!r}; likely intended {replacement!r}.", "terminology")
        if not segment.source_references:
            add("warning", "No source references recorded; verify claims before publication.", "no_sources")
        if segment.shot:
            try:
                stage_times(segment)
            except ValueError as exc:
                add("error", str(exc), "animation_cue")
            if segment.shot.asset_path and not Path(segment.shot.asset_path).is_file():
                add("error", f"Missing asset: {segment.shot.asset_path}", "missing_asset")
        else:
            add("warning", "Legacy segment has no structured shot; it will remain a photograph/slide.", "no_shot")
    topics = dict(getattr(config, "coverage_topics", {}) or {})
    min_ratio = getattr(config, "coverage_min_ratio", 0.75)
    coverage = coverage_report(board, output_dir, topics)
    if coverage["checked"] and coverage["coverage_ratio"] < min_ratio:
        findings.append({"scene": 0, "segment": 0, "level": "error" if strict_coverage else "warning",
                         "code": "coverage_incomplete",
                         "message": "Storyboard misses required topics: " + ", ".join(coverage["missing_topics"])})
    output = Path(output_dir) / "review.json"
    output.write_text(json.dumps({"notice": "Lint findings, not factual certification", "findings": findings},
                                 indent=2), encoding="utf-8")
    return findings
