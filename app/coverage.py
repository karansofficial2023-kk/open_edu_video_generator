from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import Storyboard


def coverage_report(storyboard: Storyboard, output_dir: str | Path, topics: dict[str, list[str]] | None = None) -> dict:
    """Check that each required topic is mentioned; topics come from config/storyboard, never code."""
    required = {**(topics or {}), **storyboard.required_topics}
    text = _norm(" ".join([storyboard.title, storyboard.source] + [
        f"{scene.title} {scene.narration} " + " ".join(segment.narration for segment in scene.segments)
        for scene in storyboard.scenes
    ]))
    missing = [topic for topic, markers in required.items()
               if not any(_norm(marker) in text for marker in markers)]
    report = {
        "notice": "Coverage lint against the supplied topic checklist; verify against the teacher/curriculum source.",
        "checked": bool(required),
        "covered_topics": [topic for topic in required if topic not in missing],
        "missing_topics": missing,
        "coverage_ratio": round((len(required) - len(missing)) / len(required), 3) if required else 1.0,
    }
    Path(output_dir, "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("\u2011", "-").replace("\u2013", "-")).strip()
