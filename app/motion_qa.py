from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from .asset_quality import inspect_image
from .config import AppConfig
from .schema import Segment
from .vision_qa import VisionQAClient


@dataclass(frozen=True)
class MotionQAResult:
    accepted: bool
    consistency: float
    reasons: list[str] = field(default_factory=list)
    regeneration_prompt: str = ""
    first_frame: str = ""
    final_frame: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "consistency": self.consistency,
            "reasons": self.reasons,
            "regeneration_prompt": self.regeneration_prompt,
            "first_frame": self.first_frame,
            "final_frame": self.final_frame,
            "raw": self.raw,
        }


def review_generated_motion(
    segment: Segment,
    video_path: str | Path,
    review_dir: str | Path,
    config: AppConfig,
) -> MotionQAResult:
    first_path, final_path = extract_boundary_frames(video_path, review_dir)
    reasons: list[str] = []
    for label, path in (("first", first_path), ("final", final_path)):
        quality = inspect_image(path)
        reasons.extend(f"{label} frame: {reason}" for reason in quality.reasons)
    if reasons:
        return MotionQAResult(
            accepted=False,
            consistency=0.0,
            reasons=reasons,
            regeneration_prompt="Create a sharp, detailed clip with a stable visible subject throughout.",
            first_frame=str(first_path),
            final_frame=str(final_path),
        )

    qa = VisionQAClient(config)
    first = qa.analyze(segment, first_path)
    final = qa.analyze(segment, final_path)
    with Image.open(first_path) as source:
        first_image = source.convert("RGB")
    with Image.open(final_path) as source:
        final_image = source.convert("RGB")
    pair = qa.compare_motion_frames(segment, first_image, final_image)
    consistency = _score(pair.get("subject_consistency"))
    pair_reasons = _strings(pair.get("reasons"))
    reasons.extend(first.reasons if not first.core_accepted else [])
    reasons.extend(final.reasons if not final.core_accepted else [])
    reasons.extend(pair_reasons)
    accepted = (
        first.core_accepted
        and final.core_accepted
        and consistency >= config.vision_qa.minimum_motion_consistency
        and bool(pair.get("same_subject", False))
        and not bool(pair.get("anatomy_morphing", False))
        and not bool(pair.get("text_present", False))
    )
    regeneration_prompt = str(pair.get("regeneration_prompt") or "").strip()
    if not accepted and not regeneration_prompt:
        regeneration_prompt = "Keep one coherent subject with stable anatomy and a restrained natural physical action."
    return MotionQAResult(
        accepted=accepted,
        consistency=consistency,
        reasons=reasons or ([] if accepted else ["motion consistency requirements were not met"]),
        regeneration_prompt=regeneration_prompt,
        first_frame=str(first_path),
        final_frame=str(final_path),
        raw={"first": first.to_dict(), "final": final.to_dict(), "pair": pair},
    )


def extract_boundary_frames(video_path: str | Path, review_dir: str | Path) -> tuple[Path, Path]:
    source = Path(video_path)
    capture = cv2.VideoCapture(str(source))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2:
            raise ValueError(f"Generated motion clip has fewer than two readable frames: {source}")
        ok, first = capture.read()
        if not ok or first is None:
            raise ValueError(f"Cannot read first frame from generated motion clip: {source}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
        ok, final = capture.read()
        if not ok or final is None:
            raise ValueError(f"Cannot read final frame from generated motion clip: {source}")
    finally:
        capture.release()
    target = Path(review_dir)
    target.mkdir(parents=True, exist_ok=True)
    first_path = target / f"{source.stem}_first.png"
    final_path = target / f"{source.stem}_final.png"
    cv2.imwrite(str(first_path), first)
    cv2.imwrite(str(final_path), final)
    return first_path, final_path


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
