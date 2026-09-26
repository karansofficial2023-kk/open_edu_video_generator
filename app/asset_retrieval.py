from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from .config import AppConfig
from .schema import Segment
from .vision_qa import VisionQAClient, append_qa_audit


COMMONS_API = "https://commons.wikimedia.org/w/api.php"


@dataclass(frozen=True)
class CommonsAsset:
    title: str
    thumbnail_url: str
    description_url: str
    license_name: str
    license_url: str
    creator: str


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    return str(value.get("value", "")) if isinstance(value, dict) else ""


_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "it", "of", "on", "or", "that", "the", "their", "this", "to", "with", "where", "which",
    "show", "image", "visual", "exact", "visible", "realistic", "scientific", "educational",
}


def build_asset_search_query(segment: Segment) -> str:
    """Build a topic-neutral Commons query from the current storyboard shot only."""
    source_terms: list[str] = []
    if segment.shot:
        for item in segment.shot.labels:
            label = str(item.get("text") or item.get("name") or "").split(":", 1)[-1]
            source_terms.extend(re.findall(r"[A-Za-z][A-Za-z-]{2,}", label))
    source_terms.extend(str(item) for item in segment.keywords)
    source_terms.extend(re.findall(r"[A-Za-z][A-Za-z-]{3,}", segment.narration or ""))

    result: list[str] = []
    seen: set[str] = set()
    for raw in source_terms:
        for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", raw):
            key = token.casefold()
            if key in _QUERY_STOP_WORDS or key in seen:
                continue
            seen.add(key)
            result.append(token)
            if len(result) >= 10:
                return " ".join(result)
    return " ".join(result)


def build_asset_search_queries(segment: Segment) -> list[str]:
    """Return focused-to-broad Commons queries for only the current shot."""
    text = " ".join(
        value for value in (segment.narration, segment.visual, segment.image_prompt) if value
    )
    subjects: list[str] = []
    patterns = (
        r"\b(?:species\s+like|such\s+as|including|for\s+example)\s+"
        r"([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,2})",
        r"\b(?:plant|flower|organism|material|apparatus)\s+(?:called|named)\s+"
        r"([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,2})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            words = [
                word for word in match.group(1).split()
                if word.casefold() not in _QUERY_STOP_WORDS
            ]
            if words:
                subjects.append(" ".join(words))

    labels = []
    if segment.shot:
        labels = [
            str(item.get("text") or item.get("name") or "").strip()
            for item in segment.shot.labels
            if str(item.get("text") or item.get("name") or "").strip()
        ]

    queries: list[str] = []
    for subject in subjects:
        if labels:
            queries.append(" ".join([subject, *labels[:2]]))
        queries.append(subject)
    queries.append(build_asset_search_query(segment))

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _apply_verified_targets(segment: Segment, verified: dict[str, dict[str, float]]) -> None:
    if not segment.shot:
        return
    for collection in (segment.shot.labels, segment.shot.arrows):
        for item in collection:
            name = str(item.get("text") or item.get("name") or "").strip()
            point = next((value for key, value in verified.items() if key.casefold() == name.casefold()), None)
            if not point:
                continue
            item["target_xy"] = f"{float(point['x']):.6f},{float(point['y']):.6f}"
            item["target_confidence"] = float(point.get("confidence", 0.9))
            item["target_source"] = "vision_verified_retrieved_asset"


class CommonsAssetClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.settings = config.asset_retrieval
        self.headers = {"User-Agent": self.settings.user_agent}

    def search(self, query: str) -> list[CommonsAsset]:
        response = requests.get(
            COMMONS_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": self.settings.max_candidates,
                "prop": "imageinfo",
                "iiprop": "url|mediatype|extmetadata",
                "iiurlwidth": self.settings.thumbnail_width,
                "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit",
                "format": "json",
            },
            headers=self.headers,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {}).values()
        assets: list[CommonsAsset] = []
        allowed = {item.casefold() for item in self.settings.allowed_licenses}
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mediatype") not in {"BITMAP", "DRAWING"} or not info.get("thumburl"):
                continue
            metadata = info.get("extmetadata") or {}
            license_name = _metadata_value(metadata, "LicenseShortName")
            if license_name.casefold() not in allowed:
                continue
            assets.append(CommonsAsset(
                title=page.get("title", ""),
                thumbnail_url=info["thumburl"],
                description_url=info.get("descriptionurl", ""),
                license_name=license_name,
                license_url=_metadata_value(metadata, "LicenseUrl"),
                creator=_metadata_value(metadata, "Artist") or _metadata_value(metadata, "Credit"),
            ))
        return assets

    def download(self, asset: CommonsAsset, destination: Path) -> Path:
        response = requests.get(
            asset.thumbnail_url,
            headers=self.headers,
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination


def retrieve_approved_asset(
    segment: Segment,
    query: str | list[str],
    destination: Path,
    output_dir: Path,
    config: AppConfig,
) -> Path | None:
    if not config.asset_retrieval.enabled or config.asset_retrieval.provider != "wikimedia_commons":
        return None
    client = CommonsAssetClient(config)
    qa = VisionQAClient(config)
    records: list[dict[str, Any]] = []
    queries = [query] if isinstance(query, str) else list(query)
    candidate_index = 0
    for current_query in queries:
        if candidate_index >= client.settings.max_candidates:
            break
        try:
            assets = client.search(current_query)
        except requests.RequestException as exc:
            append_qa_audit(output_dir / "vision_qa.json", {
                "segment": segment.segment_number,
                "accepted": False,
                "qa_mode": "commons_retrieval",
                "query": current_query,
                "reasons": [f"Asset search unavailable: {exc}"],
            })
            continue

        for asset in assets:
            if candidate_index >= client.settings.max_candidates:
                break
            candidate_index += 1
            index = candidate_index
            candidate = destination.with_stem(f"{destination.stem}_commons_{index:02d}")
            try:
                client.download(asset, candidate)
                result = qa.analyze(segment, candidate)
            except (requests.RequestException, OSError, ValueError) as exc:
                records.append({**asdict(asset), "accepted": False, "error": str(exc)})
                continue
            labels = [
                str(item.get("text") or item.get("name") or "").strip()
                for item in (segment.shot.labels if segment.shot else [])
                if str(item.get("text") or item.get("name") or "").strip()
            ]
            verified: dict[str, dict[str, float]] = {}
            verification_audit: dict[str, Any] = {}
            if labels and result.core_accepted:
                verified = qa.verify_targets(segment, candidate, result.target_proposals)
                verification_audit = qa.last_target_verification
            resolved = {name.casefold() for name in verified}
            targets_accepted = not labels or {name.casefold() for name in labels} <= resolved
            accepted = result.core_accepted and targets_accepted
            audit = {
                "segment": segment.segment_number,
                "attempt": index,
                "image": str(candidate),
                "qa_mode": "commons_retrieval",
                "query": current_query,
                "source": asdict(asset),
                **result.to_dict(),
                "accepted": accepted,
                "verified_targets": verified,
                "verification_audit": verification_audit,
                "coordinate_policy": "verified_before_use" if labels else "no_label_targets_requested",
            }
            append_qa_audit(output_dir / "vision_qa.json", audit)
            records.append({**asdict(asset), **result.to_dict(), "accepted": accepted})
            if accepted:
                _apply_verified_targets(segment, verified)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
                attribution = output_dir / "asset_attribution.json"
                existing = json.loads(attribution.read_text(encoding="utf-8")) if attribution.exists() else []
                existing.append({"local_file": str(destination), "query": current_query, **asdict(asset)})
                attribution.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
                return destination
    return None
