from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .config import AppConfig
from .schema import Segment


@dataclass
class VisionQAResult:
    accepted: bool
    relevance: float
    subject_match: float
    text_present: bool
    major_artifacts: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    target_proposals: dict[str, dict[str, float] | None] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VisionQAClient:
    """Screens generated assets; proposed points are never approved coordinates."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = config.ollama_url.rstrip("/")

    def analyze(self, segment: Segment, image_path: str | Path) -> VisionQAResult:
        path = Path(image_path)
        with Image.open(path) as image:
            width, height = image.size
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        labels = [
            str(item.get("text") or item.get("name") or "").strip()
            for item in (segment.shot.labels if segment.shot else [])
        ]
        labels = [label for label in labels if label]
        prompt = self._review_prompt(segment, labels)
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.vision_qa.model,
                "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                "format": "json",
                "stream": False,
                "keep_alive": 0,
                "options": {"temperature": 0},
            },
            timeout=self.config.vision_qa.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "{}")
        payload = _json_object(content)
        relevance = _score(payload.get("relevance"))
        subject_match = _score(payload.get("subject_match"))
        text_present = bool(payload.get("text_present", False))
        artifacts = _strings(payload.get("major_artifacts"))
        reasons = _strings(payload.get("reasons"))
        proposals = _normalize_proposals(payload.get("target_proposals"), width, height)
        requested = {label.casefold() for label in labels}
        proposals = {name: point for name, point in proposals.items() if name.casefold() in requested}
        accepted = (
            relevance >= self.config.vision_qa.minimum_relevance
            and subject_match >= self.config.vision_qa.minimum_subject_match
            and not text_present
            and not artifacts
        )
        return VisionQAResult(
            accepted=accepted,
            relevance=relevance,
            subject_match=subject_match,
            text_present=text_present,
            major_artifacts=artifacts,
            reasons=reasons,
            target_proposals=proposals,
            raw=payload,
        )

    def _review_prompt(self, segment: Segment, labels: list[str]) -> str:
        expected = segment.image_prompt or segment.visual or segment.narration
        title_note = ""
        if segment.shot and segment.shot.template == "title_card":
            topic = segment.shot.heading or segment.narration
            expected = f"A realistic text-free background visually relevant to this title: {topic}. {expected}"
            title_note = (
                "This is only the background layer of a title card. Do not expect the title, subtitle, "
                "labels, or any other typography inside this image. Do not reject it for being generated "
                "below final delivery resolution; deterministic rendering and upscaling occur later. "
                "A clearly relevant pollinator, flower, or subject placed behind a dark overlay is valid."
            )
        return f"""You are a strict visual quality inspector for an educational video.
Judge only what is visibly present. Do not assume that a structure exists because the
expected description names it. Reject unrelated subjects, visible words, watermarks,
collages, duplicated/deformed anatomy, and visually invented scientific structures.
{title_note}

Expected shot:
{expected}

Requested physical labels (may be empty): {json.dumps(labels)}

Return JSON only with exactly these keys:
{{
  "relevance": 0.0,
  "subject_match": 0.0,
  "text_present": false,
  "major_artifacts": [],
  "reasons": [],
  "target_proposals": {{}}
}}
Scores range from 0 to 1. For each requested label, target_proposals may contain
null or {{"x": number, "y": number, "confidence": number}}. Coordinates should be
normalized from 0 to 1. These are unverified proposals, so use null when uncertain.
Do not approve merely because the image is attractive or broadly on-topic."""


def append_qa_audit(path: str | Path, record: dict[str, Any]) -> None:
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if audit_path.exists():
        try:
            loaded = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except (OSError, json.JSONDecodeError):
            records = []
    records.append(record)
    audit_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_object(value: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Vision QA response must be a JSON object")
    return parsed


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_proposals(value: Any, width: int, height: int) -> dict[str, dict[str, float] | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, float] | None] = {}
    for name, point in value.items():
        if not isinstance(point, dict):
            result[str(name)] = None
            continue
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            result[str(name)] = None
            continue
        if x > 1 or y > 1:
            x /= width
            y /= height
        if not (0 <= x <= 1 and 0 <= y <= 1):
            result[str(name)] = None
            continue
        result[str(name)] = {"x": x, "y": y, "confidence": _score(point.get("confidence"))}
    return result
