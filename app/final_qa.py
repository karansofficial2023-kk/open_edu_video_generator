from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2

from .asset_quality import fingerprint_distance, inspect_image
from .config import AppConfig
from .label_overlay import build_label_plans, is_label_template
from .schema import Storyboard
from .subtitles import _METADATA_TERMS


def run_final_video_qa(
    storyboard: Storyboard,
    video_path: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    subtitles_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    video = Path(video_path)
    failures: list[str] = []
    warnings: list[str] = []
    probe = _probe(video, config)
    video_stream = next((item for item in probe.get("streams", []) if item.get("codec_type") == "video"), {})
    audio_stream = next((item for item in probe.get("streams", []) if item.get("codec_type") == "audio"), {})
    width, height = int(video_stream.get("width", 0)), int(video_stream.get("height", 0))
    if (width, height) != (config.output_resolution.width, config.output_resolution.height):
        failures.append(f"resolution mismatch: {width}x{height}")
    if not audio_stream:
        failures.append("final video has no audio stream")
    duration = _duration(probe)
    expected_duration = storyboard.all_segments()[-1][1].end if storyboard.all_segments() else 0.0
    if abs(duration - expected_duration) > config.final_qa.duration_tolerance_seconds:
        failures.append(f"duration mismatch: video={duration:.3f}s storyboard={expected_duration:.3f}s")

    sample_dir = output / "final_qa_samples"
    samples = _sample_segment_frames(storyboard, video, sample_dir, config.final_qa.max_samples)
    fingerprints: list[tuple[str, str]] = []
    sample_results = []
    previous_segment = None
    for key, path, segment in samples:
        quality = inspect_image(path)
        if not quality.accepted:
            failures.extend(f"{key}: {reason}" for reason in quality.reasons)
        if (fingerprints and fingerprint_distance(fingerprints[-1][1], quality.fingerprint) <= 2
                and not _same_visual_intent(previous_segment, segment)):
            failures.append(f"{key}: near-duplicate of previous storyboard shot")
        fingerprints.append((key, quality.fingerprint))
        sample_results.append({"shot": key, "path": str(path), **quality.to_dict()})
        previous_segment = segment

    label_results = []
    for scene, segment in storyboard.all_segments():
        if not segment.shot or not is_label_template(segment.shot.template):
            continue
        plans, review = build_label_plans(segment.shot, (width, height))
        key = f"{scene.scene_number}.{segment.segment_number}"
        label_results.append({"shot": key, "labels": len(plans), "review": review})
        failures.extend(f"{key}: label {item['label']} - {item['reason']}" for item in review)

    subtitle_result = _review_subtitles(Path(subtitles_path) if subtitles_path else None, config)
    failures.extend(subtitle_result["failures"])
    warnings.extend(subtitle_result["warnings"])

    visual_audit = _semantic_audit(output)
    failures.extend(visual_audit["failures"])
    warnings.extend(visual_audit["warnings"])
    report = {
        "passed": not failures,
        "video": str(video.resolve()),
        "resolution": [width, height],
        "duration_seconds": duration,
        "expected_duration_seconds": expected_duration,
        "audio_present": bool(audio_stream),
        "sampled_frames": sample_results,
        "labels": label_results,
        "subtitles": subtitle_result,
        "semantic_visual_gate": visual_audit,
        "failures": failures,
        "warnings": warnings,
    }
    report_path = output / "final_qa_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures and config.final_qa.block_on_failure:
        raise ValueError("Final video QA failed: " + "; ".join(failures[:8]))
    return report


def _probe(path: Path, config: AppConfig) -> dict[str, Any]:
    result = subprocess.run(
        [config.ffprobe_path, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _duration(probe: dict[str, Any]) -> float:
    values = [probe.get("format", {}).get("duration")]
    values.extend(item.get("duration") for item in probe.get("streams", []))
    durations = []
    for value in values:
        try:
            durations.append(float(value))
        except (TypeError, ValueError):
            pass
    return max(durations, default=0.0)


def _sample_segment_frames(storyboard: Storyboard, video: Path, output: Path, maximum: int):
    items = storyboard.all_segments()
    if len(items) > maximum:
        stride = len(items) / maximum
        indexes = sorted({min(len(items) - 1, int(i * stride)) for i in range(maximum)})
    else:
        indexes = list(range(len(items)))
    capture = cv2.VideoCapture(str(video))
    samples = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        for index in indexes:
            scene, segment = items[index]
            timestamp = max(segment.start, (segment.start + segment.end) / 2)
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            key = f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}"
            if not ok or frame is None:
                continue
            path = output / f"{key}.png"
            cv2.imwrite(str(path), frame)
            samples.append((key, path, segment))
    finally:
        capture.release()
    return samples


def _same_visual_intent(previous, current) -> bool:
    if previous is None or current is None:
        return False
    previous_asset = previous.shot.asset_path if previous.shot else None
    current_asset = current.shot.asset_path if current.shot else None
    if previous_asset and current_asset and Path(previous_asset).resolve() == Path(current_asset).resolve():
        return True
    left = re.sub(r"\s+", " ", previous.image_prompt or previous.visual or "").strip().casefold()
    right = re.sub(r"\s+", " ", current.image_prompt or current.visual or "").strip().casefold()
    return bool(left and right and left == right)


def _review_subtitles(path: Path | None, config: AppConfig) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    cues = 0
    if path and path.exists():
        blocks = re.split(r"\r?\n\r?\n", path.read_text(encoding="utf-8-sig").strip())
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3:
                continue
            cues += 1
            visible = lines[2:]
            if len(visible) > 2:
                failures.append(f"subtitle cue {cues} exceeds two lines")
            text = " ".join(visible).casefold()
            for term in _METADATA_TERMS:
                if term in text:
                    failures.append(f"subtitle cue {cues} leaks production metadata: {term}")
    elif config.render.burn_captions or config.render.embed_subtitles:
        warnings.append("subtitle file is absent")
    contrast = _contrast_ratio(config.render.subtitle_fg, config.render.subtitle_bg)
    if contrast < 4.5:
        failures.append(f"subtitle foreground/background contrast is too low: {contrast:.2f}:1")
    return {"cue_count": cues, "contrast_ratio": contrast, "failures": failures, "warnings": warnings}


def _semantic_audit(output: Path) -> dict[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    audit_path = output / "vision_qa.json"
    manifest_path = output / "render_manifest.json"
    if audit_path.exists():
        records = json.loads(audit_path.read_text(encoding="utf-8"))
        if any(item.get("qa_mode") == "production_asset_blocked" for item in records):
            failures.append("visual QA contains a blocked production asset")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rejected = [key for key, value in manifest.get("assets", {}).items() if value.get("status") != "approved"]
        failures.extend(f"unapproved visual asset: {key}" for key in rejected)
    else:
        warnings.append("no generated-image manifest; final video may contain deterministic or supplied assets only")
    return {"failures": failures, "warnings": warnings}


def _contrast_ratio(foreground: str, background: str) -> float:
    def luminance(value: str) -> float:
        raw = value.strip().lstrip("#")
        rgb = [int(raw[index:index + 2], 16) / 255 for index in (0, 2, 4)]
        linear = [channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4 for channel in rgb]
        return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]
    first, second = luminance(foreground), luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return round((lighter + .05) / (darker + .05), 3)
