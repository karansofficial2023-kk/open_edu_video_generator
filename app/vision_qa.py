from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import AppConfig
from .schema import Segment


@dataclass
class VisionQAResult:
    accepted: bool
    core_accepted: bool
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
        self.last_target_verification: dict[str, Any] = {}
        self._sam_predictor = None
        self._sam_image_identity: int | None = None

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
                "options": {"temperature": 0, "num_predict": 512},
            },
            timeout=self.config.vision_qa.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "{}")
        try:
            payload = _json_object(content)
        except (json.JSONDecodeError, ValueError) as error:
            return VisionQAResult(
                accepted=False,
                core_accepted=False,
                relevance=0.0,
                subject_match=0.0,
                text_present=False,
                reasons=[f"Vision QA returned malformed JSON: {error}"],
                raw={"response_preview": content[:500]},
            )
        relevance = _score(payload.get("relevance"))
        subject_match = _score(payload.get("subject_match"))
        text_present = bool(payload.get("text_present", False))
        artifacts = _strings(payload.get("major_artifacts"))
        reasons = _strings(payload.get("reasons"))
        proposals = _normalize_proposals(payload.get("target_proposals"), width, height)
        requested = {label.casefold() for label in labels}
        proposals = {name: point for name, point in proposals.items() if name.casefold() in requested}
        proposed_names = {
            name.casefold() for name, point in proposals.items()
            if point and float(point.get("confidence", 0)) >= 0.75
        }
        missing_targets = sorted(requested - proposed_names)
        if missing_targets:
            reasons.append("Requested targets are not all clearly locatable: " + ", ".join(missing_targets))
        core_accepted = (
            relevance >= self.config.vision_qa.minimum_relevance
            and subject_match >= self.config.vision_qa.minimum_subject_match
            and not text_present
            and not artifacts
        )
        accepted = core_accepted and not missing_targets
        return VisionQAResult(
            accepted=accepted,
            core_accepted=core_accepted,
            relevance=relevance,
            subject_match=subject_match,
            text_present=text_present,
            major_artifacts=artifacts,
            reasons=reasons,
            target_proposals=proposals,
            raw=payload,
        )

    def verify_targets(
        self,
        segment: Segment,
        image_path: str | Path,
        proposals: dict[str, dict[str, float] | None],
    ) -> dict[str, dict[str, float]]:
        candidates = {
            name: point for name, point in proposals.items()
            if point and float(point.get("confidence", 0)) >= 0.75
        }
        requested_names = [
            str(item.get("text") or item.get("name") or "").strip()
            for item in (segment.shot.labels if segment.shot else [])
            if str(item.get("text") or item.get("name") or "").strip()
        ]
        for name in requested_names:
            if not any(existing.casefold() == name.casefold() for existing in candidates):
                candidates[name] = {"x": 0.5, "y": 0.5, "confidence": 0.75}
        if not candidates:
            self.last_target_verification = {"status": "no_confident_candidates"}
            return {}
        source_path = Path(image_path)
        annotated_path = source_path.with_name(source_path.stem + "_target_check.png")
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        font_size = max(24, round(min(image.size) * 0.035))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        marker_lines = []
        names = list(candidates)
        for index, name in enumerate(names, start=1):
            point = candidates[name]
            x = round(float(point["x"]) * image.width)
            y = round(float(point["y"]) * image.height)
            radius = max(20, round(min(image.size) * 0.032))
            line_width = max(4, round(min(image.size) * 0.007))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#FF0033", width=line_width)
            draw.line((x - radius, y, x + radius, y), fill="#FFFF00", width=max(3, line_width // 2))
            draw.line((x, y - radius, x, y + radius), fill="#FFFF00", width=max(3, line_width // 2))
            draw.text((x + radius + 6, y - radius), str(index), fill="#FF0033", font=font,
                      stroke_width=3, stroke_fill="white")
            placement = _label_placement(segment, name)
            marker_lines.append(f"{index}. {name}: {placement}")
        annotated_image.save(annotated_path)
        try:
            encoded = base64.b64encode(annotated_path.read_bytes()).decode("ascii")
            response_example = {
                "targets": {
                    name: {"verified": False, "confidence": 0.0, "reason": ""}
                    for name in names
                }
            }
            prompt = f"""You are verifying scientific annotation targets on one fixed image.
Each numbered red ring and yellow crosshair marks a proposed target. Verify only whether
the crosshair center touches the exact visible physical structure named below. Reject a
point on a nearby petal, background, different organ, whole object when a specific part
was requested, or any target hidden/ambiguous in the image.

Targets:
{chr(10).join(marker_lines)}

Return one result for every target using these exact label keys. JSON only:
{json.dumps(response_example)}
Confidence ranges from 0 to 1. Set verified=true only when the crosshair center is on the
named visible structure and confidence is at least 0.85. The marker graphics are temporary
inspection aids, not scientific structures.
"""
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.config.vision_qa.model,
                    "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
                    "format": "json",
                    "stream": False,
                    "keep_alive": 0,
                    "options": {"temperature": 0, "num_predict": 512},
                },
                timeout=self.config.vision_qa.timeout_seconds,
            )
            response.raise_for_status()
            payload = _json_object(response.json().get("message", {}).get("content", "{}"))
            checks = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
            initially_supported: dict[str, Any] = {}
            rejected: dict[str, Any] = {}
            for name, point in candidates.items():
                check = _casefold_lookup(checks, name)
                if not isinstance(check, dict):
                    rejected[name] = {"reason": "missing verifier result"}
                    continue
                confidence = _score(check.get("confidence"))
                if bool(check.get("verified")) and confidence >= 0.85:
                    initially_supported[name] = check
                else:
                    rejected[name] = check
            verified: dict[str, dict[str, float]] = {}
            refined_audit: dict[str, Any] = {}
            for name, point in candidates.items():
                refined, audit = self._refine_target_in_crop(segment, image, name, point)
                refined_audit[name] = audit
                if refined:
                    verified[name] = refined
            collisions = _target_collisions(verified)
            for first, second in collisions:
                verified.pop(first, None)
                verified.pop(second, None)
                refined_audit.setdefault(first, {})["collision_rejection"] = second
                refined_audit.setdefault(second, {})["collision_rejection"] = first
            self.last_target_verification = {
                "status": "completed",
                "candidates": candidates,
                "raw": payload,
                "initially_supported": initially_supported,
                "rejected_initial_checks": rejected,
                "refined_checks": refined_audit,
                "verified": verified,
            }
            return verified
        except Exception as error:
            self.last_target_verification = {
                "status": "error",
                "candidates": candidates,
                "error": f"{type(error).__name__}: {error}",
            }
            raise
        finally:
            annotated_path.unlink(missing_ok=True)

    def _refine_target_in_crop(
        self,
        segment: Segment,
        image: Image.Image,
        name: str,
        proposal: dict[str, float],
    ) -> tuple[dict[str, float] | None, dict[str, Any]]:
        width, height = image.size
        cx = float(proposal["x"]) * width
        cy = float(proposal["y"]) * height
        crop_width = min(width, round(width * 0.56))
        crop_height = min(height, round(height * 0.72))
        left = max(0, min(width - crop_width, round(cx - crop_width / 2)))
        top = max(0, min(height - crop_height, round(cy - crop_height / 2)))
        crop = image.crop((left, top, left + crop_width, top + crop_height))
        scale = min(2.5, 1200 / max(crop.size))
        if scale > 1:
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        placement = _label_placement(segment, name)
        prompt = f"""This is a magnified crop of one scientific photograph.
Locate exactly one visible physical target named: {name}
Target definition and placement: {placement}

Return a point at the center of that exact structure using integer x,y coordinates from
0 to 1000 relative to this crop. Do not return the center of the whole image or whole
specimen when a specific part is requested. If the named structure is not visibly
distinguishable from nearby structures, set visible=false. Determine the point from the
pixels; never default to the crop center and never reuse a point from another label.

Return JSON only with keys "visible" (boolean), "point" (two integer coordinates), and
"reason" (short string). Use null for point when visible is false.
"""
        payload = self._ask_image_json(crop, prompt, 256)
        if not bool(payload.get("visible")):
            return None, {"localization": payload, "status": "not_visible"}
        point = payload.get("point")
        if not isinstance(point, list) or len(point) != 2:
            bbox = payload.get("bbox") or payload.get("bbox_2d")
            if isinstance(bbox, list) and len(bbox) == 4:
                point = [(float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2]
        try:
            crop_x = max(0.0, min(1000.0, float(point[0]))) / 1000
            crop_y = max(0.0, min(1000.0, float(point[1]))) / 1000
        except (TypeError, ValueError, IndexError):
            return None, {"localization": payload, "status": "invalid_point"}
        x = (left + crop_x * crop_width) / width
        y = (top + crop_y * crop_height) / height
        snapped, snap_audit = self._snap_target_with_sam(image, x, y)
        if snapped:
            x, y = snapped
        competing = [
            str(item.get("text") or item.get("name") or "").strip()
            for item in (segment.shot.labels if segment.shot else [])
            if str(item.get("text") or item.get("name") or "").strip().casefold() != name.casefold()
        ]
        exact, verification = self._verify_refined_point(
            image, name, placement, x, y, competing
        )
        audit = {
            "status": "verified" if exact else "rejected",
            "crop_bounds": [left, top, left + crop_width, top + crop_height],
            "localization": payload,
            "sam_snap": snap_audit,
            "normalized_point": [x, y],
            "verification": verification,
        }
        if not exact:
            return None, audit
        return {"x": x, "y": y, "confidence": 0.9}, audit

    def _snap_target_with_sam(
        self,
        image: Image.Image,
        x: float,
        y: float,
    ) -> tuple[tuple[float, float] | None, dict[str, Any]]:
        settings = self.config.vision_qa
        checkpoint = Path(settings.sam_checkpoint) if settings.sam_checkpoint else None
        if not settings.sam_enabled or checkpoint is None or not checkpoint.exists():
            return None, {"status": "disabled_or_missing"}
        try:
            import numpy as np
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            if self._sam_predictor is None:
                model = sam_model_registry[settings.sam_model_type](checkpoint=str(checkpoint))
                model.to(device="cuda" if torch.cuda.is_available() else "cpu")
                self._sam_predictor = SamPredictor(model)
            if self._sam_image_identity != id(image):
                self._sam_predictor.set_image(np.array(image.convert("RGB")))
                self._sam_image_identity = id(image)
            width, height = image.size
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=np.array([[x * width, y * height]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            candidates = []
            for mask, score in zip(masks, scores):
                area = float(mask.mean())
                if not 0.0003 <= area <= 0.15 or float(score) < 0.65:
                    continue
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    continue
                centroid = (float(xs.mean() / width), float(ys.mean() / height))
                distance = math.hypot(centroid[0] - x, centroid[1] - y)
                if distance <= 0.12:
                    candidates.append((distance, -float(score), centroid, area, float(score)))
            if not candidates:
                return None, {"status": "no_suitable_mask"}
            _, _, centroid, area, score = min(candidates)
            return centroid, {
                "status": "snapped",
                "input_point": [x, y],
                "snapped_point": list(centroid),
                "mask_area": area,
                "sam_score": score,
            }
        except Exception as error:
            return None, {"status": "error", "error": f"{type(error).__name__}: {error}"}

    def _verify_refined_point(
        self,
        image: Image.Image,
        name: str,
        placement: str,
        x: float,
        y: float,
        competing_labels: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        width, height = image.size
        crop_width = min(width, round(width * 0.20))
        crop_height = min(height, round(height * 0.30))
        center_x, center_y = x * width, y * height
        pad_x = crop_width // 2
        pad_y = crop_height // 2
        padded = ImageOps.expand(image, border=(pad_x, pad_y, pad_x, pad_y), fill=(235, 237, 240))
        padded_center_x = center_x + pad_x
        padded_center_y = center_y + pad_y
        left = round(padded_center_x - crop_width / 2)
        top = round(padded_center_y - crop_height / 2)
        crop = padded.crop((left, top, left + crop_width, top + crop_height)).convert("RGB")
        scale = min(3.0, 1000 / max(crop.size))
        if scale > 1:
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        competitors = ", ".join(competing_labels or []) or "none"
        instructions = f"""Required physical structure: {name}
Target definition: {placement}
Other labeled structures that must not be confused with this target: {competitors}

Choose decision="exact_target" only when pixels at the exact crop center lie on the required
named structure, not on a neighboring organ, background, stem, petal, or approximate region.
Choose "not_target" when another object occupies the center and "uncertain" when ambiguous.
Do not infer a match merely because the required structure appears elsewhere in the crop.
Return JSON only: {{"decision": "exact_target", "confidence": 0.0, "center_object": ""}}
"""
        contextual_prompt = f"""The first image is the full clean scientific photograph for anatomical context.
The second image is a clean, unmarked magnified crop centered on the candidate target.
Judge the object that physically occupies the exact center of the second image.
{instructions}"""
        crop_prompt = f"""This image is a clean, unmarked magnified crop centered on the candidate target.
Judge the object that physically occupies the exact center of this image.
{instructions}"""
        contextual = self._ask_images_json([image, crop], contextual_prompt, 512)
        primary_crop = self._ask_images_json([crop], crop_prompt, 512)
        secondary_crop: dict[str, Any] = {}
        secondary_model = self.config.vision_qa.secondary_model
        if secondary_model and secondary_model != self.config.vision_qa.model:
            secondary_crop = self._ask_images_json([crop], crop_prompt, 512, model=secondary_model)

        def object_agrees(payload: dict[str, Any]) -> bool:
            center_object = str(payload.get("center_object") or "").strip().casefold()
            requested = name.strip().casefold()
            return bool(center_object) and (
                requested in center_object or center_object in requested
            )

        contextual_exact = (
            str(contextual.get("decision") or "").strip().casefold() == "exact_target"
            and object_agrees(contextual)
        )
        # VLMs can contradict a correct center_object with a not_target flag. Require
        # exact contextual approval, then use crop-level object identity as the tie-breaker.
        exact = contextual_exact and (
            object_agrees(primary_crop) or object_agrees(secondary_crop)
        )
        status = "clean_center_verified" if exact else "clean_center_rejected"
        return exact, {
            "status": status,
            "candidate_point": [x, y],
            "contextual": contextual,
            "primary_crop": primary_crop,
            "secondary_crop": secondary_crop,
        }

    def _ask_image_json(self, image: Image.Image, prompt: str, num_predict: int) -> dict[str, Any]:
        return self._ask_images_json([image], prompt, num_predict)

    def _ask_images_json(
        self,
        images: list[Image.Image],
        prompt: str,
        num_predict: int,
        model: str | None = None,
    ) -> dict[str, Any]:
        encoded_images = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.config.vision_qa.model,
                "messages": [{"role": "user", "content": prompt, "images": encoded_images}],
                "format": "json",
                "stream": False,
                "keep_alive": 0,
                "options": {"temperature": 0, "num_predict": num_predict},
            },
            timeout=self.config.vision_qa.timeout_seconds,
        )
        response.raise_for_status()
        return _json_object(response.json().get("message", {}).get("content", "{}"))

    def _review_prompt(self, segment: Segment, labels: list[str]) -> str:
        expected = segment.image_prompt or segment.visual or segment.narration
        title_note = ""
        label_note = ""
        if labels:
            label_note = (
                "IMPORTANT: This is a STATIC CLEAN BASE IMAGE for later composited scientific labels. "
                "Judge whether every requested physical structure is visibly present, sharp, and unobstructed. "
                "Ignore narration action verbs and do not require motion, particles in transit, arrows, or labels "
                "inside this clean image. Their absence is required and must not lower relevance or subject_match."
            )
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
{label_note}

Narration that the image must support:
{segment.narration}

Production prompt (supporting detail, not visible text to look for):
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
Score relevance and subject_match for the clean base image independently from label
coordinate availability. A missing or uncertain target belongs in target_proposals as
null and must not by itself reduce relevance or subject_match when the named subject is
otherwise correct.
Judge the visible subject and action, not production phrases such as documentary frame,
professional composition, safe margin, or educational style. A sharp still photograph is
valid when it clearly shows the named subject even if an invisible process cannot be frozen
in one frame. For narration describing transfer, timing, maturation, or another process,
give full relevance when the stable image clearly shows every named physical source,
receiver, or structure needed for later arrows and labels. Do not require particles in
motion or completed overlays inside the clean source image. Do not approve merely because
the image is attractive or broadly on-topic."""


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


def _label_placement(segment: Segment, name: str) -> str:
    if not segment.shot:
        return "exact visible physical structure"
    for item in segment.shot.labels:
        label = str(item.get("text") or item.get("name") or "").strip()
        if label.casefold() == name.casefold():
            return str(item.get("placement") or "exact visible physical structure").strip()
    return "exact visible physical structure"


def _casefold_lookup(mapping: dict[str, Any], name: str) -> Any:
    wanted = name.casefold()
    for key, value in mapping.items():
        if str(key).casefold() == wanted:
            return value
    wanted_tokens = set(re.findall(r"[a-z0-9]+", wanted))
    for key, value in mapping.items():
        key_tokens = set(re.findall(r"[a-z0-9]+", str(key).casefold()))
        if wanted_tokens and key_tokens and (wanted_tokens <= key_tokens or key_tokens <= wanted_tokens):
            return value
    return None


def _target_collisions(
    targets: dict[str, dict[str, float]],
    minimum_distance: float = 0.04,
) -> list[tuple[str, str]]:
    collisions: list[tuple[str, str]] = []
    items = list(targets.items())
    for index, (first_name, first) in enumerate(items):
        for second_name, second in items[index + 1:]:
            if first_name.casefold() == second_name.casefold():
                continue
            distance = math.hypot(float(first["x"]) - float(second["x"]),
                                  float(first["y"]) - float(second["y"]))
            if distance < minimum_distance:
                collisions.append((first_name, second_name))
    return collisions


def _point_inside_verified_bbox(
    payload: dict[str, Any],
    point_x: float,
    point_y: float,
    padding: float = 0.015,
) -> tuple[bool, dict[str, Any]]:
    if not bool(payload.get("visible")):
        return False, {"status": "target_not_visible"}
    points = payload.get("points")
    if points is None:
        point = payload.get("point")
        points = [point] if isinstance(point, list) and len(point) == 2 else None
    if isinstance(points, list) and points:
        normalized_points: list[list[float]] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                continue
            try:
                verified_x, verified_y = [max(0.0, min(1000.0, float(value))) / 1000 for value in point]
            except (TypeError, ValueError):
                continue
            normalized_points.append([verified_x, verified_y])
            distance = math.hypot(point_x - verified_x, point_y - verified_y)
            if distance <= 0.11:
                return True, {
                    "status": "point_near_verified_target",
                    "normalized_point": [verified_x, verified_y],
                    "normalized_points": normalized_points,
                    "candidate_point": [point_x, point_y],
                    "distance": distance,
                }
        if normalized_points:
            return False, {
                "status": "point_far_from_verified_target",
                "normalized_points": normalized_points,
                "candidate_point": [point_x, point_y],
            }
    boxes = payload.get("bboxes")
    if boxes is None:
        bbox = payload.get("bbox") or payload.get("bbox_2d")
        boxes = [bbox] if isinstance(bbox, list) and len(bbox) == 4 else None
    if not isinstance(boxes, list) or not boxes:
        return False, {"status": "invalid_target_bbox"}
    normalized_boxes: list[list[float]] = []
    rejected_broad = False
    for bbox in boxes:
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [max(0.0, min(1000.0, float(value))) / 1000 for value in bbox]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) > 0.65:
            rejected_broad = True
            continue
        normalized = [x1, y1, x2, y2]
        normalized_boxes.append(normalized)
        if x1 - padding <= point_x <= x2 + padding and y1 - padding <= point_y <= y2 + padding:
            return True, {
                "status": "point_inside_target_bbox",
                "normalized_bbox": normalized,
                "normalized_bboxes": normalized_boxes,
                "candidate_point": [point_x, point_y],
            }
    if not normalized_boxes:
        status = "target_bbox_too_broad" if rejected_broad else "invalid_target_bbox"
        return False, {"status": status, "candidate_point": [point_x, point_y]}
    return False, {
        "status": "point_outside_target_bbox",
        "normalized_bboxes": normalized_boxes,
        "candidate_point": [point_x, point_y],
    }
