from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import Storyboard


POLLINATION_TOPICS = {
    "definition": ["transfer of pollen", "anther", "stigma"],
    "self pollination": ["self-pollination", "autogamy", "geitonogamy"],
    "cross pollination": ["cross-pollination", "different plant", "genetic diversity"],
    "classification": ["self-pollination", "cross-pollination"],
    "wind pollination": ["anemophily", "wind"],
    "water pollination": ["hydrophily", "water", "vallisneria", "hydrilla"],
    "insect pollination": ["entomophily", "insect", "bee"],
    "bird pollination": ["ornithophily", "bird"],
    "bat pollination": ["chiropterophily", "bat"],
    "animal pollination": ["zoophily", "animal"],
    "self advantages": ["pure lines", "external agents", "inbreeding"],
    "cross advantages": ["healthier offspring", "adaptation", "pollination failure"],
}


def coverage_report(storyboard: Storyboard, output_dir: str | Path) -> dict:
    text = _norm(" ".join([storyboard.title, storyboard.source] + [
        f"{scene.title} {scene.narration} " + " ".join(segment.narration for segment in scene.segments)
        for scene in storyboard.scenes
    ]))
    topic_hits = {}
    missing = []
    for topic, markers in POLLINATION_TOPICS.items():
        hits = [marker for marker in markers if _norm(marker) in text]
        topic_hits[topic] = hits
        if not hits:
            missing.append(topic)
    report = {
        "notice": "Coverage lint for Types of Pollination; verify against teacher/curriculum source.",
        "covered_topics": [topic for topic in POLLINATION_TOPICS if topic not in missing],
        "missing_topics": missing,
        "coverage_ratio": round((len(POLLINATION_TOPICS) - len(missing)) / len(POLLINATION_TOPICS), 3),
    }
    Path(output_dir, "coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("\u2011", "-").replace("\u2013", "-")).strip()
