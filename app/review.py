"""Small, explicit lint checks; this is not an automated scientific fact checker."""
import json
import re
from pathlib import Path

from .animation_renderer import stage_times
from .coverage import coverage_report


def review_storyboard(board, output_dir, expected_topic: str = ""):
    findings = []
    for scene, segment in board.all_segments():
        def add(level, message):
            findings.append({"scene": scene.scene_number, "segment": segment.segment_number,
                             "level": level, "message": message})
        text = segment.narration.lower()
        for term, replacement in {"glystogamy": "cleistogamy", "gystogamy": "cleistogamy",
                                  "hetrostyle": "heterostyly", "protogenic": "protogynous"}.items():
            if re.search(rf"\b{term}\b", text):
                add("error", f"Check terminology: {term!r}; likely intended {replacement!r}.")
        if re.search(r"anther\s+curls", text):
            add("error", "Verify the claim that an anther curls; provide a corrected, species-specific account.")
        if not segment.source_references:
            add("warning", "No source references recorded; verify scientific claims before publication.")
        if segment.shot:
            try:
                stage_times(segment)
            except ValueError as exc:
                add("error", str(exc))
            if segment.shot.asset_path and not Path(segment.shot.asset_path).is_file():
                add("error", f"Missing asset: {segment.shot.asset_path}")
        else:
            add("warning", "Legacy segment has no structured shot; it will remain a photograph/slide.")
    coverage = coverage_report(board, output_dir)
    coverage_scope = f"{expected_topic} {board.title} {board.source}".lower()
    if "types of pollination" in coverage_scope and coverage["coverage_ratio"] < 0.75:
        findings.append({"scene": 0, "segment": 0, "level": "warning",
                         "message": "Storyboard does not cover the full Types of Pollination topic. Check coverage.json."})
    output = Path(output_dir) / "review.json"
    output.write_text(json.dumps({"notice": "Lint findings, not factual certification", "findings": findings},
                                 indent=2), encoding="utf-8")
    return findings
