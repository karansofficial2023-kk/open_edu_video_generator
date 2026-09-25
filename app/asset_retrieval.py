from __future__ import annotations

import json
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
    query: str,
    destination: Path,
    output_dir: Path,
    config: AppConfig,
) -> Path | None:
    if not config.asset_retrieval.enabled or config.asset_retrieval.provider != "wikimedia_commons":
        return None
    client = CommonsAssetClient(config)
    qa = VisionQAClient(config)
    records: list[dict[str, Any]] = []
    try:
        assets = client.search(query)
    except requests.RequestException as exc:
        append_qa_audit(output_dir / "vision_qa.json", {
            "segment": segment.segment_number,
            "accepted": False,
            "qa_mode": "commons_retrieval",
            "query": query,
            "reasons": [f"Asset search unavailable: {exc}"],
        })
        return None

    for index, asset in enumerate(assets, start=1):
        candidate = destination.with_stem(f"{destination.stem}_commons_{index:02d}")
        try:
            client.download(asset, candidate)
            result = qa.analyze(segment, candidate)
        except (requests.RequestException, OSError, ValueError) as exc:
            records.append({**asdict(asset), "accepted": False, "error": str(exc)})
            continue
        audit = {
            "segment": segment.segment_number,
            "attempt": index,
            "image": str(candidate),
            "qa_mode": "commons_retrieval",
            "query": query,
            "source": asdict(asset),
            **result.to_dict(),
            "coordinate_policy": "proposals_only_not_verified",
        }
        append_qa_audit(output_dir / "vision_qa.json", audit)
        records.append({**asdict(asset), **result.to_dict()})
        if result.accepted:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            attribution = output_dir / "asset_attribution.json"
            existing = json.loads(attribution.read_text(encoding="utf-8")) if attribution.exists() else []
            existing.append({"local_file": str(destination), "query": query, **asdict(asset)})
            attribution.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            return destination
    return None
