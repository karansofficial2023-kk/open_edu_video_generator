"""Wikimedia Commons retrieval (open-license reference imagery) with provenance. Retrieval before generation for real
subjects (spec section 6.1); a candidate is only a candidate - it still has to pass the same technical + semantic QA."""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import requests

from .config import AppConfig

API = "https://commons.wikimedia.org/w/api.php"
ALLOWED = re.compile(r"(?i)^(cc0|public domain|pd\b|cc[- ]by(?:[- ]sa)?(?:[- ]\d(?:\.\d)?)?|cc[- ]by[- ]\d)")
BLOCKED = re.compile(r"(?i)(nc|nd|non-?commercial|no derivatives|fair use|nonfree)")
HEADERS = {"User-Agent": "OpenEduVideoGenerator/1.0 (educational; contact: local)"}


def search_query(text: str, config: AppConfig) -> str:
    """Short noun-phrase search query from approved narration (local Qwen text model; falls back to content words)."""
    try:
        response = requests.post(f"{config.ollama_url}/api/generate", json={
            "model": config.production.director_model, "stream": False, "think": False, "keep_alive": "2m",
            "prompt": "Write a 2-5 word image search query naming the main physical subject of this sentence. "
                      "Return only the query, no quotes.\nSentence: " + text,
            "options": {"temperature": 0.1, "num_predict": 24}}, timeout=120)
        query = re.sub(r"[^\w\s-]", "", response.json().get("response", "")).strip()
        if 1 <= len(query.split()) <= 6:
            return query
    except (requests.RequestException, ValueError):
        pass
    words = [w for w in re.findall(r"[A-Za-z]{5,}", text)][:4]
    return " ".join(words)


def _get(url: str, **kwargs) -> requests.Response:
    """Polite Wikimedia client: spaced requests and bounded backoff on 429/5xx (honours Retry-After)."""
    delay, timeout = 3.0, kwargs.pop("timeout", 60)
    for attempt in range(5):
        time.sleep(1.2)
        response = requests.get(url, headers=HEADERS, timeout=timeout, **kwargs)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        wait = response.headers.get("Retry-After", "")
        time.sleep(float(wait) if wait.isdigit() else delay)
        delay *= 2
    response.raise_for_status()
    return response


def find(query: str, config: AppConfig, limit: int = 8) -> list[dict]:
    params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
              "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime",
              "iiurlwidth": 1920}
    try:
        pages = _get(API, params=params).json().get("query", {}).get("pages", {})
    except (requests.RequestException, ValueError):
        return []
    found = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = (meta.get("LicenseShortName", {}) or {}).get("value", "")
        if info.get("mime") not in {"image/jpeg", "image/png"} or not ALLOWED.search(license_name) or BLOCKED.search(license_name):
            continue
        if info.get("width", 0) < config.production.retrieval_min_width or not info.get("thumburl"):
            continue
        found.append({"title": page.get("title", ""), "url": info["thumburl"], "page": info.get("descriptionurl", ""),
                      "width": info["width"], "height": info["height"], "license": license_name,
                      "author": re.sub(r"<[^>]+>", "", (meta.get("Artist", {}) or {}).get("value", "")).strip(),
                      "retrieved": date.today().isoformat(), "query": query, "index": page.get("index", 99)})
    return sorted(found, key=lambda item: item["index"])


def download(candidate: dict, destination: Path) -> Path:
    destination.write_bytes(_get(candidate["url"], timeout=120).content)
    return destination
