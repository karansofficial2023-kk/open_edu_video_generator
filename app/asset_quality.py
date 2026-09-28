from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat


@dataclass(frozen=True)
class AssetQualityResult:
    accepted: bool
    reasons: list[str]
    width: int
    height: int
    luminance_stddev: float
    edge_mean: float
    fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_image(path: str | Path, prior_fingerprints: set[str] | None = None) -> AssetQualityResult:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    sample = ImageOps.fit(image, (256, 144), method=Image.Resampling.LANCZOS).convert("L")
    luminance_stddev = float(ImageStat.Stat(sample).stddev[0])
    edge_mean = float(ImageStat.Stat(sample.filter(ImageFilter.FIND_EDGES)).mean[0])
    fingerprint = perceptual_fingerprint(sample)
    reasons: list[str] = []
    if width < 512 or height < 288:
        reasons.append(f"image resolution is too low ({width}x{height})")
    if luminance_stddev < 8.0:
        reasons.append("image is blank or has almost no tonal detail")
    if edge_mean < 2.0:
        reasons.append("image has insufficient visible structure or detail")
    if prior_fingerprints and any(fingerprint_distance(fingerprint, item) <= 2 for item in prior_fingerprints):
        reasons.append("image is a near-duplicate of an already approved shot")
    return AssetQualityResult(
        accepted=not reasons,
        reasons=reasons,
        width=width,
        height=height,
        luminance_stddev=round(luminance_stddev, 3),
        edge_mean=round(edge_mean, 3),
        fingerprint=fingerprint,
    )


def perceptual_fingerprint(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + column] > pixels[offset + column + 1] for column in range(8))
    value = sum((1 << index) for index, enabled in enumerate(bits) if enabled)
    return f"{value:016x}"


def fingerprint_distance(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return math.inf


def load_approved_fingerprints(path: str | Path) -> set[str]:
    manifest = Path(path)
    if not manifest.exists():
        return set()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    records = payload.get("assets", {}) if isinstance(payload, dict) else {}
    return {
        str(record.get("fingerprint"))
        for record in records.values()
        if isinstance(record, dict) and record.get("status") == "approved" and record.get("fingerprint")
    }


def update_manifest(path: str | Path, key: str, record: dict) -> None:
    manifest = Path(path)
    payload = {"version": 1, "assets": {}}
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
                payload.setdefault("assets", {})
        except (OSError, json.JSONDecodeError):
            pass
    payload["assets"][key] = record
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest)


def reusable_asset(path: str | Path, key: str, expected_path: str | Path) -> dict | None:
    manifest = Path(path)
    asset = Path(expected_path)
    if not manifest.exists() or not asset.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        record = payload.get("assets", {}).get(key)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(record, dict) or record.get("status") != "approved":
        return None
    if Path(str(record.get("path", ""))).resolve() != asset.resolve():
        return None
    quality = inspect_image(asset)
    if not quality.accepted or quality.fingerprint != record.get("fingerprint"):
        return None
    return record
